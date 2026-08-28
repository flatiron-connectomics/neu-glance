"""Host arrays in this process and open a real neuroglancer on them.

The other modules here *emit* a state for a viewer somewhere else. This one runs the
viewer's server itself, inside the calling process, so an array that exists only in a
notebook — a probability map straight out of a model, a relabelled crop, a ground-truth
chunk — can be looked at without writing a volume first. It is the replacement for
serving HDF5 chunks through chunkflow to eyeball them.

**It is the only module that imports ``neuroglancer``, and the import is deferred.**
Building a state needs none of it, ``neu-glance --help`` must not pay for a viewer bundle,
and CI installs the wheel with ``--no-deps``. Two tests pin that; see
:mod:`neu_glance.cli`.

Three things about the served path that are not obvious and cost real time to rediscover:

- **A served source is ``python://volume/<viewer-token>.<volume-token>``, and it is not
  portable.** neuroglancer rewrites each ``LocalVolume`` into that form when the state is
  serialized, scoped to the viewer's own token and dead when this process exits. So there
  is no shareable link for a served array — :meth:`Server.state` is for inspection, not
  for sending to anyone.
- **``encoding`` must stay ``npz``.** ``neuroglancer.chunks.encode_raw`` is the one
  function that still calls ``ndarray.tostring``, removed in NumPy 2.0, so
  ``encoding="raw"`` raises ``AttributeError`` on the first chunk the browser asks for —
  after the viewer has opened and looks fine. The default is npz and the declared encoding
  is what the client requests, so leaving it alone is the whole fix. Measured on
  neuroglancer 2.41.2 / numpy 2.5.2: npz encodes, raw raises.
- **``volume_type`` is never inferred.** neuroglancer guesses segmentation only for rank-3
  ``uint16``/``uint32``/``uint64``, so a **uint8 label array is guessed as an image** —
  which averages label ids on downsample and loses the colour hashing and the selection
  UI, silently. ``kind`` decides it here, always explicitly.

Interaction runs the other way too, which is the point of hosting rather than publishing:
:meth:`Server.on_click`, :meth:`Server.boxes` and :meth:`Server.selected_segments` read
the browser's own state back, so a region picked in the viewer is usable in the notebook.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: The one encoding that survives NumPy 2 — see the module docstring. Not a parameter
#: anywhere, deliberately: the failure it prevents happens in the browser, minutes after
#: everything here has reported success.
ENCODING = "npz"

#: What a served array can be — neu-lib's vocabulary, imported rather than copied. `kind`
#: is a fact about the DATA (it decides mean vs mode when coarsening), so it is owned down
#: there; a viewer and a downsampler must not end up with two lists. `probability` is an
#: image as far as neuroglancer is concerned, differing only in the shader it gets and in
#: tolerating a channel axis.
from neu_lib import KINDS  # noqa: E402

#: Name of the annotation layer a viewer gets for drawing boxes in, and the layer
#: :meth:`Server.boxes` reads back.
REGIONS_LAYER = "regions"


class ServeProblem(RuntimeError):
    """The arrays or options given cannot be served as asked."""


def _neuroglancer():
    """The module, or an error naming the extra. Imported here, never at module scope."""
    try:
        import neuroglancer
    except ImportError as e:
        raise ServeProblem(
            "serving needs the `neuroglancer` package, which is an optional extra: "
            "`pip install 'neu-glance[serve]'`. Building a state or a link needs none of "
            "it — only hosting one does."
        ) from e
    return neuroglancer


@dataclass
class ServedLayer:
    """One array to host, and how to show it.

    ``frame`` is a :class:`neu_lib.Frame` — real per-axis voxel sizes in nm and the origin
    in nm. Passing it is what keeps a served crop on top of the volume it came from: an
    origin dropped here puts the crop at nm zero, which is correct for a whole volume and
    wrong for every box out of one, with nothing to show for it.

    ``channel_axis`` marks a leading channel axis (a 3-channel probability map) and is
    **derived from the array's rank** — a served volume is 3 spatial axes, so 4-D means a
    channel axis and 3-D means none, and there is no third possibility to choose between.
    Passing it is allowed and checked; passing it *wrong* is the one thing it can do, which
    is why it is not asked for. The package convention is channel-first, matching
    ``has_channels`` elsewhere.
    """
    array: Any
    kind: str = "image"
    name: str | None = None
    frame: Any = None
    shader: str | None = None
    channel_axis: bool | None = None
    opacity: float | None = None

    # ------------------------------------------------------------ constructors
    #
    # The rule these follow, and it is one line: **infer what the source RECORDS, require
    # what it does not.** A frame, a dataset name and a channel axis are all facts written
    # down somewhere — in an HDF5 file's attributes, a precomputed `info`, or the array's
    # own shape — so reading them is not guessing and dropping them is the silent failure.
    # `kind` is different: an HDF5 file has nowhere agreed-on to say it, and deciding from
    # the dtype is exactly the mistake `volume_type` exists to override, so it is asked for
    # rather than inferred. A volume that records `info["type"]` is the exception, because
    # there the answer is written down too.

    @classmethod
    def from_array(cls, array: Any, kind: str, *, name: str | None = None,
                   frame: Any = None, voxel_size: Sequence[float] | None = None,
                   origin: Sequence[float] | None = None,
                   channel_axis: bool | None = None, **kwargs) -> "ServedLayer":
        """An in-memory array. ``voxel_size``/``origin`` build the frame for you.

            ServedLayer.from_array(prob, "probability", voxel_size=(40, 8, 8))

        ``channel_axis`` comes from the array's rank, which is a fact about its shape
        rather than a guess — see :class:`ServedLayer`.
        """
        from neu_lib import Frame

        if frame is None and voxel_size is not None:
            frame = Frame(voxel_size_nm=tuple(float(v) for v in voxel_size),
                          origin_nm=tuple(float(o) for o in (origin or (0.0, 0.0, 0.0))))
        elif frame is not None and (voxel_size is not None or origin is not None):
            raise ServeProblem(
                "pass either frame= or voxel_size=/origin=, not both — two frames for one "
                "array is a disagreement nothing here can resolve")
        return cls(array=array, kind=kind, name=name, frame=frame,
                   channel_axis=channel_axis, **kwargs)

    @classmethod
    def _read(cls, src: Any, kind: str | None, name: str | None, *, path: str,
              **read_kwargs) -> "ServedLayer":
        """Read through ``neu_vol.read_piece`` and wrap it. Shared by the three readers.

        The reading lives in neu-vol, not here: it opens stores, `write` and `to-hdf5` want
        it too, and this package used to carry a second copy of it.
        """
        from neu_vol import read_piece

        try:
            piece = read_piece(src, kind, **read_kwargs)
        except ValueError as e:
            # neu-vol speaks ValueError; this package's callers — and its CLI — catch
            # ServeProblem. Translated rather than left to leak, so `except ServeProblem`
            # around a constructor means what it says.
            raise ServeProblem(str(e)) from None
        return cls.from_piece(
            piece, name=name or _default_name(path, read_kwargs.get("dataset")))

    @classmethod
    def from_hdf5(cls, path: str, dataset: str | None = None, kind: str | None = None, *,
                  crop: Any = None, name: str | None = None,
                  voxel_size: Sequence[float] | None = None, **kwargs) -> "ServedLayer":
        """One dataset of an HDF5 file, with the frame it records about itself.

            ServedLayer.from_hdf5("piece.h5", kind="segmentation")
            ServedLayer.from_hdf5("gt.h5", "/z07901", "segmentation")

        ``dataset`` is optional when the file holds exactly one 3D+ array; with several the
        error lists them. The frame comes from the file's own ``voxel_size`` / ``axes`` /
        ``voxel_offset`` attributes — which is what ``neu-vol to-hdf5`` writes, so a piece
        this suite produced needs no coordinates retyped. ``voxel_size`` overrides, and is
        required for a file that records none.

        ``kind`` is required: an HDF5 file has nowhere agreed-on to record it, and reading
        it off the dtype is the mistake that shows uint8 labels as an image.

        The format is forced rather than detected, so a file whose extension detection does
        not recognise still opens.

        """
        return cls._read(str(path), kind, name, path=str(path), dataset=dataset,
                         src_format="hdf5", crop=crop, voxel_size=voxel_size, **kwargs)

    @classmethod
    def from_volume(cls, volume: str, kind: str | None = None, *, level: int = 0,
                    crop: Any = None, name: str | None = None,
                    voxel_size: Sequence[float] | None = None,
                    **kwargs) -> "ServedLayer":
        """A box out of a stored volume — zarr, precomputed, an image stack.

            ServedLayer.from_volume("s3://my-bucket/seg_v1", level=1,
                                    crop=((0, 0, 0), (64, 512, 512)))

        ``kind`` is taken from what the volume records (precomputed's ``info["type"]``,
        OME's multiscales ``type``) and is only required where it records nothing. That is
        not inference: it is the same field ``neu-vol copy`` exists to preserve, and
        overriding it silently is what averages label ids into ids that were never in the
        data.

        A whole volume is usually far too large to hold in memory — pass ``crop``.
        """
        return cls._read(volume, kind, name, path=volume, level=level, crop=crop,
                         voxel_size=voxel_size, **kwargs)

    @classmethod
    def from_piece(cls, piece: Any, kind: str | None = None, *, name: str | None = None,
                   **kwargs) -> "ServedLayer":
        """A :class:`neu_lib.Piece` — an array that already carries its frame and kind.

            piece = neu_vol.read_piece("gt.h5:/vol_03700", "segmentation")
            layer = ServedLayer.from_piece(piece)

        ``kind`` defaults to the piece's own, so a piece read from a source that records one
        (a precomputed volume's ``info["type"]``) needs nothing said. Where neither says it
        is **required** rather than guessed: a uint8 label array is indistinguishable from
        an image by dtype, and neuroglancer's own guess reads it as one — averaging label
        ids on downsample and losing the colour hashing and the selection UI.

        The conversion goes this way round, and it has to: ``Piece`` lives in neu-lib, the
        bottom tier, and a ``Piece.as_layer`` would mean the vocabulary package naming a
        viewer type three tiers above it. neu-glance reading a neu-lib type is the allowed
        direction.
        """
        kind = kind or getattr(piece, "kind", None)
        if kind is None:
            raise ServeProblem(
                f"this source records no image/segmentation type, so kind= is required — "
                f"one of {', '.join(KINDS)}. It is not inferred from the dtype: "
                f"neuroglancer's own guess reads a uint8 label array as an image, which "
                f"averages label ids on downsample and loses the colour hashing and the "
                f"selection UI. (A precomputed volume records the type in its `info`, and "
                f"then this is not needed.)")
        return cls(array=piece.array, kind=kind,
                   name=name or getattr(piece, "name", None), frame=piece.frame,
                   **kwargs)

    @classmethod
    def from_source(cls, src: str, kind: str | None = None, *, level: int = 0,
                    crop: Any = None, name: str | None = None,
                    voxel_size: Sequence[float] | None = None,
                    **kwargs) -> "ServedLayer":
        """Anything readable, addressed as ``PATH`` or ``PATH:/DATASET``.

        The form the CLI takes, and the one to reach for when the source could be either —
        the format is detected, and a trailing ``:/name`` selects an array inside an HDF5
        container. Only a **leading slash** makes it a dataset, so a scheme's own colon
        (``s3://…``) is left alone.
        """
        return cls._read(src, kind, name, path=src, level=level, crop=crop,
                         voxel_size=voxel_size, **kwargs)

    # ------------------------------------------------------------ where it is
    @property
    def piece(self) -> Any:
        """This layer's array and frame as a :class:`neu_lib.Piece`.

        The type that answers the geometry questions — ``.bbox``, ``.bounds_nm``,
        ``.crop()`` — so they are not reimplemented here. A layer is a *piece plus how to
        draw it*, and this is the piece.
        """
        from neu_lib import Piece

        if self.frame is None:
            raise ServeProblem(
                f"layer {self.name!r} has no frame, so it has no position to report. "
                f"Build it with a voxel_size= or from a source that records one")
        return Piece(array=self.array, frame=self.frame, kind=self.kind)

    @property
    def bbox(self) -> Any:
        """Where this layer sits, in its frame's voxels. See :attr:`piece`."""
        return self.piece.bbox

    @property
    def bounds_nm(self) -> Any:
        """Where this layer sits, in nanometres. Also what ``crop=`` reads off a layer."""
        return self.piece.bounds_nm

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ServeProblem(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not hasattr(self.array, "shape") or not hasattr(self.array, "dtype"):
            raise ServeProblem(
                f"layer {self.name!r}: expected an array with .shape and .dtype, got "
                f"{type(self.array).__name__}")
        rank = len(self.array.shape)
        if rank not in (3, 4):
            raise ServeProblem(
                f"layer {self.name!r}: a served volume is 3 spatial axes, optionally with "
                f"a leading channel axis, so 3-D or 4-D — got {rank}-D "
                f"{tuple(self.array.shape)}")
        derived = rank == 4
        if self.channel_axis is None:
            self.channel_axis = derived
        elif bool(self.channel_axis) != derived:
            raise ServeProblem(
                f"layer {self.name!r}: channel_axis={self.channel_axis} contradicts a "
                f"{rank}-D array {tuple(self.array.shape)}, which has "
                f"{'a' if derived else 'no'} leading channel axis. Leave it out — the rank "
                f"decides it")


def _default_name(path: str, dataset: str | None) -> str:
    """A layer name from a source: the dataset if there is one, else the file stem."""
    import os

    if dataset:
        return dataset.strip("/").replace("/", "_") or "layer"
    stem = os.path.basename(str(path).rstrip("/"))
    for suffix in (".h5", ".hdf5", ".hdf", ".he5", ".zarr", ".precomputed", ".n5"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "layer"


@dataclass
class Click:
    """Where a click landed, and what was under it.

    ``position`` is in the served frame's voxels, ``zyx``, as floats — the cursor is not on
    a voxel centre. ``values`` maps layer name to the value there, which for a segmentation
    is the label id.
    """
    position: tuple[float, ...]
    values: dict[str, Any] = field(default_factory=dict)
    state: Any = None

    @property
    def voxel(self) -> tuple[int, ...]:
        """``position`` floored to whole voxels, which is the voxel actually clicked."""
        import math

        return tuple(int(math.floor(v)) for v in self.position)


def _coordinate_space(ng, frame: Any = None, *, channel_axis: bool = False,
                     name: str | None = None) -> Any:
    """A ``CoordinateSpace`` from a ``Frame``, or a unit one where there is none.

    Takes the frame rather than a :class:`ServedLayer` so the **spatial** space of a
    channelled layer can be built without one — the viewer's own dimensions are spatial
    only, and routing that through a layer meant constructing a fake 4-D one that failed
    its own validation.

    neuroglancer normalizes units to base SI on construction, so ``8, "nm"`` is stored as
    ``8e-9 m``. That is its business; the numbers handed in stay nm, which is this suite's
    one model space.
    """
    voxel = (1.0, 1.0, 1.0)
    if frame is not None:
        voxel = tuple(float(v) for v in frame.voxel_size_nm)
        if len(voxel) != 3:
            raise ServeProblem(
                f"layer {name!r}: frame has {len(voxel)} axes, expected 3 (z, y, x)")
    names, scales, units = ["z", "y", "x"], list(voxel), ["nm"] * 3
    if channel_axis:
        # A trailing `^` marks a LOCAL dimension — one the layer owns rather than one the
        # viewer navigates. Without it neuroglancer offers a slider for the channel and
        # shows one at a time, instead of handing all three to the shader.
        names, scales, units = ["c^"] + names, [1.0] + scales, [""] + units
    return ng.CoordinateSpace(names=names, scales=scales, units=units)


def _voxel_offset(frame: Any = None, *, channel_axis: bool = False,
                  name: str | None = None) -> list[int] | None:
    """The origin in whole voxels, or ``None``. Non-integral is an error, not a rounding.

    Rounding would shift the volume by up to half a voxel against the source it is meant
    to overlay, which is exactly the kind of drift nothing downstream can detect.
    """
    if frame is None:
        return None
    origin = tuple(float(o) for o in frame.origin_nm)
    if not any(origin):
        return None
    voxel = tuple(float(v) for v in frame.voxel_size_nm)
    offset = []
    for o, v in zip(origin, voxel):
        exact = o / v
        if abs(exact - round(exact)) > 1e-6:
            raise ServeProblem(
                f"layer {name!r}: origin {origin} nm is not a whole number of "
                f"{voxel} nm voxels ({exact:g} on one axis), so it cannot be expressed as "
                f"a voxel offset")
        offset.append(int(round(exact)))
    return ([0] + offset) if channel_axis else offset


def _volume_type(layer: ServedLayer) -> str:
    """``segmentation`` or ``image``, from ``kind`` alone — never from the dtype."""
    return "segmentation" if layer.kind == "segmentation" else "image"


def _shader_for(layer: ServedLayer) -> str | None:
    """The GLSL for this layer, or ``None`` to leave neuroglancer's default.

    A segmentation gets none: the viewer's own label hashing is better than anything
    written here, and a shader would defeat the selection UI it exists for.
    """
    from .shaders import image_shader

    if layer.kind == "segmentation":
        return None
    return image_shader(layer.shader, kind=layer.kind, channels=_channels(layer))


def _channels(layer: ServedLayer) -> int:
    return int(layer.array.shape[0]) if layer.channel_axis else 1


class Server:
    """A running viewer, and the handle the notebook keeps on it.

    Everything that reads the browser's state back lives here rather than being left to
    ``server.viewer``, so a caller does not have to learn neuroglancer's API to get a
    region out of a click. ``.viewer`` is exposed for anything not wrapped.
    """

    def __init__(self, viewer, volumes: dict[str, Any], regions: str | None) -> None:
        self.viewer = viewer
        self.volumes = volumes
        self._regions = regions
        self._actions = 0

    # -------------------------------------------------------------- addressing
    @property
    def url(self) -> str:
        return self.viewer.get_viewer_url()

    def state(self) -> dict:
        """The viewer's current state as JSON.

        **For inspection only.** Its sources are ``python://volume/<token>`` URLs scoped
        to this process, so the state will not resolve anywhere else and is not a link to
        share. Use `neu-glance gen` for that, against a volume that is actually published.
        """
        return self.viewer.state.to_json()

    def _repr_html_(self) -> str:
        return (f'<a href="{self.url}" target="_blank">neuroglancer</a> '
                f'<code>{self.url}</code>')

    def __repr__(self) -> str:
        return f"<Server {self.url} layers={sorted(self.volumes)}>"

    # -------------------------------------------------------------- reading back
    @property
    def position(self) -> tuple[float, ...] | None:
        """Where the viewer is looking, in the served frame's voxels (zyx)."""
        pos = self.viewer.state.position
        return None if pos is None else tuple(float(v) for v in pos)

    def selected_segments(self, layer: str | None = None) -> set[int]:
        """The label ids currently shown in a segmentation layer."""
        name = layer or self._sole_segmentation()
        segments = self.viewer.state.layers[name].segments
        return {int(s) for s in (segments or ())}

    def _sole_segmentation(self) -> str:
        names = [n for n, v in self.volumes.items() if v.volume_type == "segmentation"]
        if len(names) == 1:
            return names[0]
        raise ServeProblem(
            f"say which layer: this viewer has {len(names)} segmentation layers "
            f"({', '.join(sorted(names)) or 'none'})")

    def boxes(self, layer: str | None = None) -> list[tuple[tuple[int, ...], ...]]:
        """Boxes drawn in the viewer, as ``(lo, hi)`` pairs in whole voxels, zyx.

        This is the loop that hosting buys and publishing cannot: draw a box over
        something interesting, and get it back here ready for ``--crop-bbox``,
        :func:`neu_vol.extract_roi` or ``neu-vol write``.

        Corners are sorted per axis, because a box drawn up-and-left has ``point_a``
        greater than ``point_b`` and every consumer of a box in this suite expects
        half-open ``lo < hi``.
        """
        ng = _neuroglancer()
        name = layer or self._regions
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to read boxes from")
        found = []
        for ann in self.viewer.state.layers[name].annotations or ():
            if not isinstance(ann, ng.AxisAlignedBoundingBoxAnnotation):
                continue
            a = [int(round(float(v))) for v in ann.point_a]
            b = [int(round(float(v))) for v in ann.point_b]
            lo = tuple(min(p, q) for p, q in zip(a, b))
            hi = tuple(max(p, q) for p, q in zip(a, b))
            found.append((lo, hi))
        return found

    def points(self, layer: str | None = None) -> list[tuple[int, ...]]:
        """Points drawn in the viewer, in whole voxels, zyx."""
        ng = _neuroglancer()
        name = layer or self._regions
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to read points from")
        return [tuple(int(round(float(v))) for v in ann.point)
                for ann in (self.viewer.state.layers[name].annotations or ())
                if isinstance(ann, ng.PointAnnotation)]

    # -------------------------------------------------------------- callbacks
    def on_click(self, callback: Callable[[Click], None], *, button: int = 0,
                 modifiers: str = "control", scope: str = "data_view",
                 name: str | None = None) -> str:
        """Call ``callback(click)`` when the user clicks in the data.

        Returns the action name, so it can be removed. Default binding is
        ``control+mousedown0``, matching neuroglancer's own tools — a bare click is
        already navigation and rebinding it makes the viewer feel broken.

        Two things this wrapper handles, both of which bite otherwise:

        - the cursor may not be **over data**, in which case neuroglancer reports no
          position; the callback is simply not called rather than being handed ``None``;
        - the callback runs on the **server's event loop thread**, not the notebook's, and
          neuroglancer swallows an exception into a printed traceback. So it is guarded
          here and logged, for the reason ``neu_draw``'s toolbar guards its own handlers:
          an unguarded failure is a click that visibly does nothing.
        """
        _neuroglancer()
        self._actions += 1
        action = name or f"neu-glance-click-{self._actions}"

        def handler(action_state) -> None:
            try:
                position = getattr(action_state, "mouse_voxel_coordinates", None)
                if position is None:
                    return
                values = {}
                for layer_name in self.volumes:
                    try:
                        selected = action_state.selected_values[layer_name]
                        values[layer_name] = getattr(selected, "value", selected)
                    except Exception:                                     # noqa: BLE001
                        pass
                callback(Click(position=tuple(float(v) for v in position),
                               values=values, state=action_state))
            except Exception:                                             # noqa: BLE001
                logger.exception("click handler %s failed", action)

        self.viewer.actions.add(action, handler)
        binding = f"{modifiers}+mousedown{button}" if modifiers else f"mousedown{button}"
        with self.viewer.config_state.txn() as s:
            getattr(s.input_event_bindings, scope)[binding] = action
        logger.info("bound %s to %s (%s)", binding, action, scope)
        return action

    def on_change(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` whenever the browser changes the state.

        The callback takes **no arguments** — that is neuroglancer's contract, and it is
        the right one: the notification says "something changed" and the callback re-reads
        whatever it cares about. Same split as ``neu_draw.Scene.on_change``.
        """
        self.viewer.shared_state.add_changed_callback(callback)

    def screenshot(self, path: str | None = None):
        """A PNG of what the browser is showing. Blocks until it replies.

        Requires a **connected browser** — the image is rendered by the client and posted
        back, so this raises if nobody has the link open.
        """
        reply = self.viewer.screenshot().screenshot
        if path is None:
            return reply.image_pixels
        with open(path, "wb") as f:
            f.write(reply.image)
        return path

    def stop(self) -> None:
        """Stop the shared server, invalidating every viewer URL in this process."""
        ng = _neuroglancer()
        ng.server.stop()


def serve(layers: Sequence[ServedLayer], *, bind: str | None = None, port: int = 0,
          into: dict | None = None, regions: str | None = REGIONS_LAYER,
          position: Sequence[float] | None = None) -> Server:
    """Host ``layers`` and return a :class:`Server` carrying the viewer URL.

    ``bind`` / ``port`` set the address the browser must reach. **They take effect only
    before the first viewer in this process** — neuroglancer stores them in a module
    global consulted at server start — so a second call with a different address is
    ignored, and this warns rather than pretending. ``bind="0.0.0.0"`` makes the URL use
    the host's FQDN, which is what a notebook running on another machine needs.

    ``into`` merges the served layers into an existing state (from
    :func:`neu_glance.load_state` or :func:`neu_glance.parse_url`), keeping its view.

    ``regions`` adds an empty local annotation layer to draw in, read back by
    :meth:`Server.boxes`. Pass ``None`` to leave it out.
    """
    ng = _neuroglancer()
    layers = [ServedLayer(**ln) if isinstance(ln, dict) else ln for ln in layers]
    if not layers:
        raise ServeProblem("nothing to serve: pass at least one layer")

    if bind is not None or port:
        if ng.server.is_server_running():
            logger.warning(
                "a viewer server is already running in this process, so bind=%r port=%r "
                "are ignored — neuroglancer reads them once, at server start", bind, port)
        else:
            ng.set_server_bind_address(bind or "127.0.0.1", port)

    viewer = ng.Viewer()
    volumes: dict[str, Any] = {}
    used: set[str] = set()
    # The viewer's own dimensions, from the first layer's frame and SPATIAL only — a
    # channel axis is local to the layer, not something to navigate. Set explicitly
    # because a `dimensions` block that disagrees with the data loads cleanly and puts
    # every layer in the wrong place, which is this package's oldest silent failure.
    spatial = _coordinate_space(ng, layers[0].frame, name=layers[0].name)

    with viewer.txn() as s:
        if not (into and into.get("dimensions")):
            s.dimensions = spatial
        if into:
            # Keep the incoming view: adding a layer must not move where someone is
            # looking, the same rule `state.merge_into` follows.
            for key, value in into.items():
                if key != "layers":
                    try:
                        setattr(s, key, value)
                    except Exception:                                     # noqa: BLE001
                        logger.debug("state key %r not settable, skipped", key)
            for existing in into.get("layers", ()):
                name = existing.get("name")
                if name:
                    used.add(name)
                    s.layers[name] = existing

        for i, layer in enumerate(layers):
            name = layer.name or f"{layer.kind}{'' if len(layers) == 1 else i}"
            # neuroglancer keys a layer by name, so two sharing one is a collision rather
            # than a duplicate — the same thing `merge_into` renames for.
            base, n = name, 1
            while name in used:
                name, n = f"{base}_{n}", n + 1
            if name != base:
                logger.warning("layer name %r was taken; using %r", base, name)
            used.add(name)

            volume = ng.LocalVolume(
                layer.array,
                _coordinate_space(ng, layer.frame, channel_axis=layer.channel_axis,
                                  name=name),
                volume_type=_volume_type(layer),
                voxel_offset=_voxel_offset(layer.frame,
                                           channel_axis=layer.channel_axis, name=name),
                encoding=ENCODING,
            )
            volumes[name] = volume
            if layer.kind == "segmentation":
                s.layers[name] = ng.SegmentationLayer(source=volume)
            else:
                kwargs: dict[str, Any] = {"source": volume}
                shader = _shader_for(layer)
                if shader:
                    kwargs["shader"] = shader
                if layer.opacity is not None:
                    kwargs["opacity"] = float(layer.opacity)
                s.layers[name] = ng.ImageLayer(**kwargs)

        if regions:
            # Built here rather than on demand so there is something to draw in the moment
            # the link opens; an empty layer costs nothing.
            s.layers[regions] = ng.LocalAnnotationLayer(dimensions=spatial)
        if position is not None:
            s.position = [float(v) for v in position]

    server = Server(viewer, volumes, regions if regions else None)
    logger.info("serving %d layer(s) at %s", len(volumes), server.url)
    return server
