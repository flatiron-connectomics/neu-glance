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
:meth:`Server.annotate` is the other end of that loop — it adds the layer those read from,
to a viewer already open, since the browser follows the state rather than the other way
round.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Self, Sequence

import numpy as np

from neu_lib import Piece

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

#: Name of the annotation layer :meth:`Server.boxes` reads back. **Opt-in**: a viewer that
#: grows a layer nobody asked for is confusing, and the read-back loop is a deliberate
#: workflow rather than something every look at a crop needs. Pass ``annotations=True``
#: (or a name) to `serve`, ``--annotate`` on the command line — spelled differently there
#: because ``gen --annotations`` already means a source to load — or
#: :meth:`Server.annotate` once the viewer is already up.
ANNOTATE_LAYER = "annotations"

#: The annotation tools that can be armed, keyed by the read-back method that understands
#: what they draw. **Only these two, deliberately**: neuroglancer also places lines,
#: ellipsoids and polylines, and :meth:`Server.boxes` / :meth:`Server.points` skip every one
#: of them — so arming one would mean drawing that never comes back, with nothing to say
#: why.
ANNOTATION_TOOLS = {"box": "annotateBoundingBox", "point": "annotatePoint"}


class ServeProblem(RuntimeError):
    """The arrays or options given cannot be served as asked."""


def _as_served_layer(layer: Any) -> ServedLayer:
    """Whatever a caller offered as a layer, as a :class:`ServedLayer`.

    **One place says what a layer input may be**, for the reason
    :meth:`Server._add_layer_in` is one place: :func:`serve` and :meth:`Server.add_layer`
    accepting different things is a wart you meet at the second call site, not the first.
    They had already drifted — one coerced ``dict``, the other any ``Mapping`` — and a
    :class:`neu_lib.Piece` handed to `serve` failed on ``layer.shader`` rather than being
    converted, because a Piece answers ``array``, ``kind``, ``frame`` and ``channel_axis``
    and so gets a long way in before anything notices.
    """
    if isinstance(layer, ServedLayer):
        return layer
    if isinstance(layer, Piece):
        return ServedLayer.from_piece(layer)
    if isinstance(layer, Mapping):
        return ServedLayer(**layer)
    raise ServeProblem(
        f"a layer is a ServedLayer, a neu_lib.Piece, or the dict form of a ServedLayer — "
        f"got {type(layer).__name__}. Build one with ServedLayer.from_array, .from_hdf5, "
        f".from_volume, .from_piece or .from_source.")


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

    ``color`` sets the shader's colour — a name (``"red"``, ``"cyan"``, and the rest of
    :data:`~neu_glance.shaders.BUILTIN_COLORS`), ``"#00ff00"``, ``"00ff00"``, or a
    3-sequence of floats (0..1) or ints (0..255). Any colour matplotlib names works too
    where matplotlib is installed, which it need not be. It applies to a shader that has a
    single colour to set,
    which the ``probability`` default does (red, with **opacity carrying the value**, so the
    map composites over the EM under it). Two probability maps in one viewer in two colours
    is what it is for. ``opacity`` is a separate thing and multiplies with it: the layer's
    overall opacity, where the shader's alpha varies per voxel.

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
    color: Any = None

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
    def _read(cls, src: Any, kind: str | None, name: str | None,
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
        # The name comes off the piece: `read_piece` derives it from the source, so this
        # package no longer keeps a second rule for it.
        return cls.from_piece(piece, name=name)

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
        return cls._read(str(path), kind, name, dataset=dataset, src_format="hdf5",
                         crop=crop, voxel_size=voxel_size, **kwargs)

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
        return cls._read(volume, kind, name, level=level, crop=crop,
                         voxel_size=voxel_size, **kwargs)

    @classmethod
    def from_piece(cls, piece: Any, kind: str | None = None, *, name: str | None = None,
                   **kwargs) -> "ServedLayer":
        """A :class:`neu_lib.Piece` — an array that already carries its frame and kind.

            piece = neu_vol.read_piece("gt.h5:/vol_03700", "segmentation")
            layer = ServedLayer.from_piece(piece)

        ``kind`` and ``name`` both default to the piece's own, so a piece from
        :func:`neu_vol.read_piece` arrives fully described — named after its source and
        knowing what its voxels mean. ``kind`` where neither says is **required** rather
        than guessed: a uint8 label array is indistinguishable from
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
        return cls._read(src, kind, name, level=level, crop=crop,
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


def _tool_type(tool: str | None) -> str | None:
    """neuroglancer's name for an annotation tool, or ``None`` to arm nothing."""
    if tool is None:
        return None
    try:
        return ANNOTATION_TOOLS[tool]
    except KeyError:
        raise ServeProblem(
            f"tool must be one of {', '.join(sorted(ANNOTATION_TOOLS))} or None, got "
            f"{tool!r}. Only these two are offered because they are the two `boxes()` and "
            f"`points()` can read back") from None


def _annotation_layer(ng, dimensions: Any, tool: str | None = "box") -> Any:
    """An empty ``local://annotations`` layer in ``dimensions``, with ``tool`` armed.

    The one place such a layer is built, so ``serve(annotations=...)`` and
    :meth:`Server.annotate` cannot end up making two different things under one name.

    ``dimensions`` is the **viewer's** coordinate space, never a fresh one: an annotation
    layer records its own, and one that disagrees with the viewer's puts every annotation
    somewhere other than where it was drawn — which loads cleanly and reads back as
    coordinates that are simply wrong.
    """
    return ng.LocalAnnotationLayer(dimensions=dimensions, tool=_tool_type(tool))


def _faces_box(lo: Sequence[float], hi: Sequence[float]) -> Any:
    """The whole-voxel box containing a shape given by its **faces**, outward.

    For the annotations whose numbers are boundaries rather than positions — a box's two
    corners, an ellipsoid's extent. Rounds out, and never to nothing: a shape drawn thinner
    than a voxel still occupies the voxel it is in.
    """
    from neu_lib import BBox

    low = [math.floor(v) for v in lo]
    return BBox(tuple(low), tuple(max(math.ceil(h), l + 1) for h, l in zip(hi, low)))


def _annotation_bbox(ann: Any) -> Any:
    """The whole-voxel :class:`neu_lib.BBox` containing one annotation, whatever type.

    Handles **every** neuroglancer annotation type rather than the two
    :meth:`Server.annotate` arms, because the toolbar can place the others and an enclosing
    box that quietly ignored one would be too small with nothing to say so. An unrecognised
    type raises for the same reason.

    **A position and a face are not rounded the same way, and conflating them is a
    one-voxel error in whichever direction you picked.** A point sits *inside* a voxel, so
    the box containing it is ``floor(p) .. floor(p)+1`` — which is what `BBox.from_points`
    means by adding one, and dropping it silently loses the far face. A box's corner is a
    *boundary* between voxels, so it rounds outward to that boundary and no further —
    otherwise enclosing a box would grow it by a voxel on every call, and the round trip
    through :meth:`Server.enclose` would never settle.
    """
    from neu_lib import BBox

    point = getattr(ann, "point", None)
    if point is not None:                                            # a point: a position
        return BBox.from_points([[float(v) for v in point]])
    a, b = getattr(ann, "point_a", None), getattr(ann, "point_b", None)
    if a is not None and b is not None:
        a = [float(v) for v in a]
        b = [float(v) for v in b]
        if isinstance(ann, _neuroglancer().LineAnnotation):           # two positions
            return BBox.from_points([a, b])
        return _faces_box([min(p, q) for p, q in zip(a, b)],          # a box: two faces
                          [max(p, q) for p, q in zip(a, b)])
    centre, radii = getattr(ann, "center", None), getattr(ann, "radii", None)
    if centre is not None and radii is not None:                     # an ellipsoid: faces
        c = [float(v) for v in centre]
        r = [abs(float(v)) for v in radii]
        return _faces_box([x - y for x, y in zip(c, r)], [x + y for x, y in zip(c, r)])
    points = getattr(ann, "points", None)                            # a polyline
    if points is not None:
        return BBox.from_points([[float(v) for v in p] for p in points])
    raise ServeProblem(
        f"cannot tell where a {type(ann).__name__} is, so it cannot be enclosed — this is "
        f"an annotation type this package does not know about")


def _margin(margin: Any, rank: int) -> tuple[int, ...]:
    """A per-axis margin in whole voxels, from a scalar or a zyx sequence."""
    values = [margin] * rank if isinstance(margin, (int, float)) else list(margin)
    if len(values) != rank:
        raise ServeProblem(
            f"margin has {len(values)} axes but the annotations have {rank} — pass one "
            f"number for every axis, or a single number for all of them")
    return tuple(int(round(float(v))) for v in values)


def _as_bbox(box: Any, what: str) -> Any:
    """A :class:`neu_lib.BBox` from one, or from a ``(lo, hi)`` pair."""
    from neu_lib import BBox

    if isinstance(box, BBox):
        return box
    try:
        lo, hi = box
        return BBox(tuple(lo), tuple(hi))
    except (TypeError, ValueError) as e:
        raise ServeProblem(f"{what} must be a BBox or a (lo, hi) pair: {e}") from None


def _spatial_axes(dimensions: Any) -> list[int]:
    """The indices of the navigable axes. A trailing ``^`` marks one local to a layer.

    neuroglancer's own convention, so a channelled layer's ``c^`` is dropped by the same
    rule that put it there rather than by assuming it is axis 0.
    """
    return [i for i, n in enumerate(dimensions.names) if not n.endswith("^")]


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


def _shader_controls(layer: ServedLayer) -> dict[str, Any]:
    """``shaderControls`` for this layer — the state overrides, not new GLSL.

    Only ``color`` so far, and set in the **state** rather than by generating a shader with
    a different default, for the reason :data:`~neu_glance.shaders.SPLIT_CONTROLS` is: every
    layer then carries the same code, and the viewer's control panel shows the value that
    was set rather than a shader nobody can diff against the built-in.
    """
    from .shaders import ShaderProblem, as_hex_color, shader_color_control

    if layer.color is None:
        return {}
    if layer.kind == "segmentation":
        raise ShaderProblem(
            "color= is for an image or probability layer; a segmentation is coloured by "
            "neuroglancer's own label hashing, which is what makes two touching bodies "
            "distinguishable. Pick colours in the viewer's segment list instead.")
    control = shader_color_control(layer.shader, kind=layer.kind,
                                   channels=_channels(layer))
    return {control: as_hex_color(layer.color, "color")}


def _channels(layer: ServedLayer) -> int:
    return int(layer.array.shape[0]) if layer.channel_axis else 1


class Server:
    """A running viewer, and the handle the notebook keeps on it.

    Everything that reads the browser's state back lives here rather than being left to
    ``server.viewer``, so a caller does not have to learn neuroglancer's API to get a
    region out of a click. ``.viewer`` is exposed for anything not wrapped.
    """

    def __init__(self, viewer, volumes: dict[str, Any], annotations: str | None) -> None:
        self.viewer = viewer
        self.volumes = volumes
        self._annotations = annotations
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

    # -------------------------------------------------------------- adding layers
    def _add_layer_in(self, s: Any, layer: ServedLayer,
                      name: str | None = None) -> tuple[str, Any]:
        """Host one array and put its layer in an **open transaction**. Returns the name.

        **The one place a served layer is set up**, called by :func:`serve` inside its
        single transaction and by :meth:`add_layer` inside its own. Two ways of building a
        layer is how the volume type, the shader and the voxel offset drift apart, and
        every one of those fails by rendering something plausible.

        Takes the transaction rather than opening one so `serve` stays a **single** state
        push: a half-populated state reaching the browser is a visible flicker, and the
        dimensions have to land with the layers that are expressed in them.
        """
        ng = _neuroglancer()
        # neuroglancer keys a layer by name, so two sharing one is a collision rather than
        # a duplicate — the same thing `merge_into` renames for. Asked of the transaction's
        # own layers, which is what makes this correct for a viewer that already has some,
        # whether they came from `into` or from an earlier `add_layer`.
        base = name or layer.name or layer.kind
        resolved, n = base, 1
        while resolved in s.layers:
            resolved, n = f"{base}_{n}", n + 1
        if resolved != base:
            logger.warning("layer name %r was taken; using %r", base, resolved)

        volume = ng.LocalVolume(
            layer.array,
            _coordinate_space(ng, layer.frame, channel_axis=layer.channel_axis,
                              name=resolved),
            volume_type=_volume_type(layer),
            voxel_offset=_voxel_offset(layer.frame, channel_axis=layer.channel_axis,
                                       name=resolved),
            encoding=ENCODING,
        )
        self.volumes[resolved] = volume
        if layer.kind == "segmentation":
            # `color=` is refused for a segmentation rather than ignored; asked before the
            # layer is built so the refusal arrives instead of a viewer that came up wrong.
            _shader_controls(layer)
            s.layers[resolved] = ng.SegmentationLayer(source=volume)
        else:
            kwargs: dict[str, Any] = {"source": volume}
            shader = _shader_for(layer)
            if shader:
                kwargs["shader"] = shader
            controls = _shader_controls(layer)
            if controls:
                kwargs["shader_controls"] = controls
            if layer.opacity is not None:
                kwargs["opacity"] = float(layer.opacity)
            s.layers[resolved] = ng.ImageLayer(**kwargs)
        return resolved, volume

    def add_layer(self, layer: Any, *, name: str | None = None,
                  color: Any = None) -> Self:
        """Host one more array in this viewer, live. Returns **the server**, so calls chain.

        The volume counterpart of :meth:`annotate`: the browser follows the state, so an
        array becomes a layer in the tab already open, with nothing re-served and no new
        URL. What :func:`serve` does per layer, one at a time::

            srv = (serve([em])
                   .add_layer(ServedLayer.from_hdf5("gt.h5", "/vol_03700", "segmentation"))
                   .add_layer(ServedLayer.from_array(pred, "probability",
                                                     voxel_size=(40, 8, 8)),
                              name="prediction"))

        Chaining is what the return is for, and the one thing it costs is the **resolved**
        name — which differs from the one asked for only when a collision renamed it (below).
        It is readable off the viewer either way: ``srv.volumes`` is insertion-ordered, so
        ``list(srv.volumes)[-1]`` is the layer just added, and ``srv.state()["layers"]``
        names them all.

        Takes a :class:`ServedLayer`, a :class:`neu_lib.Piece`, or the ``dict`` form — the
        same three :func:`serve` takes, so every constructor and every rule about ``kind``,
        frames and the channel axis applies unchanged. It goes through the same builder
        `serve` does, so there is no second way for a layer to end up configured. A piece
        arrives already named and knowing its kind (``ServedLayer.from_piece``); one whose
        ``kind`` is ``None`` needs ``ServedLayer.from_piece(piece, kind)``, since there is
        nothing here to pass it to.

        **It does not move the view**, the rule `into` follows: adding a layer must not
        change where somebody is looking. A layer whose data sits somewhere else will be
        off screen, and :meth:`bounds` is how to find it.

        A name already in the viewer is **renamed**, not replaced — neuroglancer keys a
        layer by name, so two sharing one is a collision rather than a duplicate.

        ``color`` overrides the layer's own, so one array can go in twice in two colours, or
        a probability map can be recoloured without rebuilding the ServedLayer::

            srv.add_layer(pred, color="#00ff00")     # green, opacity carrying the value
        """
        _neuroglancer()
        layer = _as_served_layer(layer)
        if color is not None:
            # A copy, not a mutation: the caller may be holding that ServedLayer, and two
            # `add_layer` calls with different colours off one layer object is the ordinary
            # way to do a before/after.
            layer = replace(layer, color=color)
        with self.viewer.txn() as s:
            resolved, _ = self._add_layer_in(s, layer, name)
        logger.info("added layer %r to %s", resolved, self.url)
        return self

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

    def annotate(self, name: str | None = None, *, tool: str | None = "box",
                 select: bool = True) -> str:
        """Add a layer to draw in, arm a tool, and return the layer's name.

        The other half of :meth:`boxes` / :meth:`points`, for a viewer already up — the same
        layer ``serve(annotations=True)`` would have made, without having to re-serve the arrays
        to get one. The browser picks the change up live, so the layer appears in a viewer
        that is already open::

            srv = serve([em, gt])
            srv.annotate()              # draw boxes; ctrl+mousedown0 places one
            srv.boxes()                 # -> [((lo), (hi)), ...] in zyx voxels
            srv.annotate(tool="point")  # same layer, now placing points
            srv.points()

        ``tool`` is what a click places — ``"box"``, ``"point"``, or ``None`` to arm
        nothing and leave the viewer's own toolbar to it. Only those two, because they are
        the two this class can read back; see :data:`ANNOTATION_TOOLS`.

        ``select`` opens the layer's side panel and makes it the active layer, which is what
        the armed tool applies to — without it the tool is set but a click does nothing
        until the layer is picked by hand, which reads as the method having failed.

        **Calling it again with the same name keeps the annotations already drawn.** It
        re-arms the tool on the layer that is there rather than replacing it: re-running a
        cell is the ordinary way this gets called twice, and silently emptying the layer
        would discard exactly the work it exists to collect. A name already taken by a
        volume layer is an error instead — that one would replace real data.

        The name defaults to the layer this server already reads back, else
        :data:`ANNOTATE_LAYER`. Whatever it ends up being becomes the default for
        :meth:`boxes` and :meth:`points`.
        """
        ng = _neuroglancer()
        # Resolved before the transaction: a bad tool must not leave a half-added layer.
        tool_type = _tool_type(tool)
        name = str(name) if name else (self._annotations or ANNOTATE_LAYER)

        existing = (self.viewer.state.layers[name]
                    if name in self.viewer.state.layers else None)
        if existing is not None:
            kind = getattr(existing.layer, "type", None)
            if kind != "annotation":
                raise ServeProblem(
                    f"layer {name!r} is already {'an' if kind == 'image' else 'a'} {kind} "
                    f"layer, and replacing it would take its data with it — pass a name "
                    f"for the annotation layer instead, e.g. annotate({name + '_picks'!r})")

        with self.viewer.txn() as s:
            if existing is None:
                s.layers[name] = _annotation_layer(ng, self._dimensions(), tool)
            else:
                s.layers[name].tool = tool_type
            if select:
                s.selected_layer.layer = name
                s.selected_layer.visible = True

        self._annotations = name
        logger.info("annotation layer %r %s; tool %s", name,
                    "reused" if existing is not None else "added", tool or "not armed")
        return name

    # -------------------------------------------------------------- extents
    def _viewer_scales(self) -> list[float]:
        """The viewer's own per-axis scale, in whatever unit neuroglancer normalized to.

        Only ever used as a **ratio** against a layer's own scales, which come from the
        same normalization — so the unit cancels and the nm-to-metres round trip cannot
        introduce the drift that turns 160 nm into voxel 19.999999.
        """
        dims = self._dimensions()
        return [float(dims.scales[i]) for i in _spatial_axes(dims)]

    def _in_viewer_voxels(self, lo: Sequence[float], hi: Sequence[float],
                          scales: Sequence[float]) -> Any:
        """A box in one layer's voxels, as whole voxels of the viewer's own space.

        Rounds **inward**, unlike everything else here, because the only caller is a limit
        to clip against: a viewer voxel the layer covers half of is not a voxel the layer
        can answer for. Exact and a no-op for the usual case, where the layer being clipped
        to is the one whose frame the viewer took.
        """
        from neu_lib import BBox

        ratio = [s / v for s, v in zip(scales, self._viewer_scales())]
        return BBox(tuple(math.ceil(a * r) for a, r in zip(lo, ratio)),
                    tuple(math.floor(b * r) for b, r in zip(hi, ratio)))

    def _served(self, layer: str | None) -> list[str]:
        if layer is None:
            return sorted(self.volumes)
        if layer not in self.volumes:
            raise ServeProblem(
                f"no served layer named {layer!r} — this viewer serves "
                f"{', '.join(sorted(self.volumes)) or 'nothing'}")
        return [layer]

    def bounds(self, layer: str | None = None) -> Any:
        """Where the served arrays are, as a :class:`neu_lib.BBox` in viewer voxels.

        The **volume** bounds: the extent of the array itself, offset included, whether or
        not anything is in it. One layer by name, or the union of all of them. This is what
        ``enclose(clip="volume")`` clips to, and what to intersect a box of your own with.
        """
        from functools import reduce

        from neu_lib import BBox

        boxes = []
        for name in self._served(layer):
            volume = self.volumes[name]
            axes = _spatial_axes(volume.dimensions)
            scales = [float(volume.dimensions.scales[i]) for i in axes]
            offset = [int(volume.voxel_offset[i]) for i in axes]
            shape = [int(volume.shape[i]) for i in axes]
            boxes.append(self._in_viewer_voxels(
                offset, [o + s for o, s in zip(offset, shape)], scales))
        return reduce(BBox.union, boxes, BBox.empty(3))

    def data_bounds(self, layer: str | None = None) -> Any:
        """Where the non-zero voxels are, as a :class:`neu_lib.BBox` in viewer voxels.

        The **data** bounds, which for a segmentation is where the labels are and is usually
        far smaller than :meth:`bounds` — a crop is mostly background. One layer by name, or
        the union of all of them.

        **This reads every voxel of the array.** The arrays are in memory already, so it is
        a pass over RAM rather than a fetch, but it is not free on a large one: it reduces
        along each axis in turn rather than building an index array, which keeps the cost to
        one boolean pass and no allocation per hit.

        A layer holding nothing but zeros raises, rather than returning an empty box that
        would read downstream as a region with no labels in it.
        """
        from functools import reduce

        from neu_lib import BBox

        boxes = []
        for name in self._served(layer):
            volume = self.volumes[name]
            axes = _spatial_axes(volume.dimensions)
            array = np.asarray(volume.data)
            if array.ndim == 4:
                # Channel-first, and a voxel counts if any channel is set there. Reduced
                # before the per-axis scan so the scan is always over 3 axes.
                array = array.any(axis=0)
            mask = array != 0
            lo, hi = [], []
            for a in range(3):
                seen = np.any(mask, axis=tuple(i for i in range(3) if i != a))
                hits = np.flatnonzero(seen)
                if hits.size == 0:
                    raise ServeProblem(
                        f"layer {name!r} is entirely zero, so it has no data bounds — "
                        f"clip to bounds() instead, or say which layer to use")
                lo.append(int(hits[0]))
                hi.append(int(hits[-1]) + 1)
            scales = [float(volume.dimensions.scales[i]) for i in axes]
            offset = [int(volume.voxel_offset[i]) for i in axes]
            boxes.append(self._in_viewer_voxels([o + v for o, v in zip(offset, lo)],
                                                [o + v for o, v in zip(offset, hi)],
                                                scales))
        return reduce(BBox.union, boxes, BBox.empty(3))

    def _clip_box(self, clip: Any) -> Any:
        """Resolve ``enclose``'s ``clip=`` to the box to intersect with."""
        if clip == "volume":
            return self.bounds()
        if clip == "data":
            return self.data_bounds()
        return _as_bbox(clip, "clip")

    def enclose(self, layer: str | None = None, *, margin: Any = 0,
                replace: bool = False, clip: Any = None,
                description: str | None = None) -> Any:
        """Replace what is drawn with the one box containing it. Returns a `neu_lib.BBox`.

        Dragging a box to exact corners in neuroglancer is fiddly; clicking a point at each
        corner is not. So: place points around what you want, call this, and get a real box
        annotation — in the viewer to look at, and as whole voxels zyx to hand to
        ``--crop-bbox``, :func:`neu_vol.extract_roi` or ``neu-vol write``::

            srv.annotate(tool="point")            # click a point at each extreme
            box = srv.enclose(margin=16, clip="volume", replace=True)
            lo, hi = box                          # a BBox unpacks like the pair it is

        **Everything drawn in the layer goes in**, points and boxes alike — and lines,
        ellipsoids and polylines, which the viewer's own toolbar can place. Enclosing a box
        that is nearly right is the other half of this: draw roughly, click a point where it
        should have reached, enclose.

        ``margin`` grows the result on every side, in **voxels** — the units :meth:`boxes`
        and :meth:`points` already speak. A scalar applies to all three axes; a sequence is
        per axis, zyx, which is what an anisotropic volume usually wants. Negative shrinks,
        and a margin that collapses an axis is an error rather than an empty box.

        ``clip`` bounds the result, since a margin reaches happily past the end of the data:

        - ``"volume"`` — the served arrays' own extent, :meth:`bounds`;
        - ``"data"`` — where their non-zero voxels are, :meth:`data_bounds`, which for a
          segmentation is where the labels are and is usually far tighter;
        - a :class:`neu_lib.BBox` or ``(lo, hi)`` pair of your own, e.g.
          ``clip=srv.bounds("seg")`` to stay inside one particular layer;
        - ``None``, the default, which does not clip and **warns** if the box starts below
          zero. Not clipping is the default because this reads a viewer someone is drawing
          in, and silently returning a smaller box than the one now drawn on their screen is
          worse than handing back what they asked for.

        A clip that leaves nothing raises: an empty box reads downstream as a region holding
        no data, which is not what "your margin fell off the edge" means.

        ``replace`` deletes the annotations that went into the box, leaving the layer
        holding just the result — the usual thing to want, since the points were scaffolding.
        It is **off by default**: this reads the viewer, and quietly deleting what someone
        drew is not something to do unasked.

        **With ``replace=False`` the new box is itself an annotation, so calling again
        encloses it too** — and with a margin, the box grows by that margin every time. That
        is not a bug to work around, it is what "enclose what is drawn" means; it is the
        reason ``replace=True`` is the usual call. With no margin it settles, because a box
        annotation's corners are read as the voxel *boundaries* they are — see
        :func:`_annotation_bbox`, where the one-voxel difference between a boundary and a
        position is the whole point.
        """
        ng = _neuroglancer()
        name = layer or self._annotations
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to enclose — call "
                               ".annotate() to add one and draw in it")
        if name not in self.viewer.state.layers:
            raise ServeProblem(f"this viewer has no layer named {name!r}")

        drawn = list(self.viewer.state.layers[name].annotations or ())
        if not drawn:
            raise ServeProblem(
                f"nothing is drawn in layer {name!r}, so there is nothing to enclose — "
                f"place a point or two in the viewer first")

        from functools import reduce

        from neu_lib import BBox

        drawn_boxes = [_annotation_bbox(ann) for ann in drawn]
        ranks = {b.ndim for b in drawn_boxes}
        if len(ranks) > 1:
            raise ServeProblem(
                f"layer {name!r} holds annotations of {sorted(ranks)} dimensions, which "
                f"have no common bounding box")
        found = reduce(BBox.union, drawn_boxes)
        pad = _margin(margin, found.ndim)
        lo_pad = tuple(v - p for v, p in zip(found.lo, pad))
        hi_pad = tuple(v + p for v, p in zip(found.hi, pad))
        # Checked before the BBox is built, not after: a negative margin can put hi *below*
        # lo, which BBox rejects outright — a ValueError about rank, where the caller needs
        # to hear that their margin was too big.
        if any(h <= l for l, h in zip(lo_pad, hi_pad)):
            raise ServeProblem(
                f"margin {pad} leaves an empty box {lo_pad} -> {hi_pad}: it shrinks the "
                f"annotations past nothing on at least one axis")
        grown = BBox(lo_pad, hi_pad)

        if clip is None:
            if any(v < 0 for v in grown.lo):
                # A negative coordinate is out of bounds in any frame, and the box is
                # presumably going straight to a reader that will fail on it. Said rather
                # than fixed, because clipping unasked would hand back a box other than the
                # one now drawn on the caller's screen.
                logger.warning(
                    "enclosing box starts at %s, outside the volume on at least one axis — "
                    "the margin reaches past the data. Pass clip='volume' to trim it",
                    grown.lo)
            box_zyx = grown
        else:
            limit = self._clip_box(clip)
            box_zyx = grown.intersect(limit)
            if box_zyx.is_empty():
                raise ServeProblem(
                    f"clipping {grown.lo} -> {grown.hi} to {limit.lo} -> {limit.hi} leaves "
                    f"nothing: what was drawn lies outside it on at least one axis")
            if box_zyx != grown:
                logger.info("clipped %s -> %s to %s", grown.lo, grown.hi, limit.hi)

        lo, hi = box_zyx
        box = ng.AxisAlignedBoundingBoxAnnotation(
            id=uuid.uuid4().hex, point_a=list(lo), point_b=list(hi),
            description=description)
        with self.viewer.txn() as s:
            keep = list(s.layers[name].annotations or ())
            if replace:
                # Matched by id against what was actually read, rather than just clearing
                # the layer, so an annotation the browser added while this was computing
                # survives. Everything neuroglancer creates carries an id, and so does the
                # box below.
                used = {a.id for a in drawn}
                keep = [a for a in keep if a.id not in used]
            s.layers[name].annotations = keep + [box]

        logger.info("enclosed %d annotation(s) in %s -> %s%s", len(drawn), lo, hi,
                    " (originals deleted)" if replace else "")
        return box_zyx

    def _dimensions(self) -> Any:
        """The viewer's own coordinate space. See :func:`_annotation_layer`."""
        dims = self.viewer.state.dimensions
        if dims is None or not list(dims.names):
            raise ServeProblem(
                "this viewer declares no dimensions, so an annotation layer would have no "
                "coordinate space to record — serve a layer carrying a frame first")
        return dims

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
        name = layer or self._annotations
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to read boxes from — "
                               "call .annotate() to add one and draw in it")
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
        name = layer or self._annotations
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to read points from — "
                               "call .annotate(tool='point') to add one and draw in it")
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
        back. With nobody on the link this does not raise, it **blocks forever**:
        ``viewer_base.screenshot`` waits on the reply with no timeout. See
        :func:`neu_glance.rendering.wait_for_browser` for the bounded form, which is what a
        render loop needs.
        """
        reply = self.viewer.screenshot().screenshot
        if path is None:
            return reply.image_pixels
        with open(path, "wb") as f:
            f.write(reply.image)
        return path

    def stop(self) -> None:
        """Stop the server, invalidating every viewer URL in this process.

        **One server per process, shared by every viewer**, so this stops them all — that
        is neuroglancer's design, not a shortcut here. See :func:`stop_serving` for the
        same thing without a handle.
        """
        stop_serving()


#: How much of the window each cross-section panel gets, per layout. `default_view` fits
#: against a nominal **window**, which is right for a link someone opens full-screen; a
#: 4-panel view gives each slice about half of it in each direction, so fitting to the
#: window leaves the piece overflowing its panel. This is the correction, and it is a
#: fraction rather than a pixel count because the real window size is not knowable here.
_PANEL_FRACTION = {
    "4panel": 0.5, "xy-3d": 0.5, "yz-3d": 0.5, "xz-3d": 0.5,
    "xy": 1.0, "yz": 1.0, "xz": 1.0, "3d": 1.0,
}


def stop_serving() -> bool:
    """Stop the viewer server, whatever started it. ``True`` if one was running.

    The escape hatch for a lost handle — a re-run cell, a renamed variable, a `serve` whose
    result was never assigned. There is **one server per process** and it is a background
    daemon thread, so this reaches it without needing the :class:`Server` that started it,
    and every viewer URL in the process goes dead.

        neu_glance.stop_serving()

    Nothing here ever blocks the notebook, so if a cell is stuck it is the *command* —
    ``neu-glance serve`` runs until interrupted, and in a cell that means forever. Interrupt
    the kernel for that one. Failing everything, the server dies with the kernel, because a
    daemon thread does.
    """
    ng = _neuroglancer()
    if not ng.server.is_server_running():
        return False
    ng.server.stop()
    logger.info("viewer server stopped; every link from this process is now dead")
    return True


def _opening_view(layers: Sequence[ServedLayer], *, layout: str = "4panel",
                  fit: float = 1.0):
    """``(position, cross_section_scale, projection_scale)`` framing every served layer.

    In the **viewer's** units, which are voxels of the first layer's frame — the viewer's
    ``dimensions`` declare a scale per axis and no origin, so a position is simply nm over
    that scale. The box is the union across layers, in nm, because that is the only space
    they share: a ground-truth crop and the image around it have different extents and
    possibly different voxel sizes.

    Scaled up by the layout's panel fraction, so the **whole** piece is inside its panel
    rather than inside a hypothetical full-window view of it. ``fit`` multiplies that again:
    above 1 shows more around the data, below 1 fills more of the panel.

    ``None`` when no layer carries a frame, in which case there is nothing to centre on and
    neuroglancer's own default is as good an answer as any.
    """
    from .state import default_view

    framed = [ln for ln in layers if ln.frame is not None]
    if not framed:
        return None
    los, his = zip(*(ln.bounds_nm for ln in framed))
    lo = tuple(min(v[a] for v in los) for a in range(3))
    hi = tuple(max(v[a] for v in his) for a in range(3))
    voxel = framed[0].frame.voxel_size_nm
    extent = tuple((b - a) / v for a, b, v in zip(lo, hi, voxel))
    offset = tuple(a / v for a, v in zip(lo, voxel))
    centre, cross, projection = default_view(extent, offset)
    scale = fit / _PANEL_FRACTION.get(layout, 0.5)
    return centre, cross * scale, projection * scale


def serve(layers: Sequence[ServedLayer], *, bind: str | None = None, port: int = 0,
          into: dict | None = None, annotations: str | bool | None = None,
          position: Sequence[float] | None = None, layout: str = "4panel",
          fit: float = 1.0) -> Server:
    """Host ``layers`` and return a :class:`Server` carrying the viewer URL.

    Each entry is a :class:`ServedLayer`, a :class:`neu_lib.Piece` (which arrives already
    named and knowing its kind), or the ``dict`` form of a ServedLayer.

    ``bind`` / ``port`` set the address the browser must reach. **They take effect only
    before the first viewer in this process** — neuroglancer stores them in a module
    global consulted at server start — so a second call with a different address is
    ignored, and this warns rather than pretending. ``bind="0.0.0.0"`` makes the URL use
    the host's FQDN, which is what a notebook running on another machine needs.

    ``into`` merges the served layers into an existing state (from
    :func:`neu_glance.load_state` or :func:`neu_glance.parse_url`), keeping its view.

    ``annotations`` adds an empty local annotation layer to draw in, read back by
    :meth:`Server.boxes`. **Off by default** — it is a deliberate workflow, not something
    every look at a crop wants, and a viewer that opens with a layer nobody asked for reads
    as a bug. ``True`` for the default name, or a name of your own. :meth:`Server.annotate`
    adds the same layer to a viewer that is already up, which is the usual way to reach for
    it: the browser picks the new layer up live, so nothing has to be re-served.

    ``position`` overrides where the viewer opens. By default it is **centred on the served
    data and zoomed so the whole piece is in each panel**: neuroglancer with no position
    opens at the origin *corner* and at one voxel per pixel, which for a crop at voxel 3700
    is a view of empty space a long way from anything.

    ``layout`` is passed through and also decides the fit, since a 4-panel view gives each
    cross-section about half the window. ``fit`` above 1 shows more around the data.
    """
    ng = _neuroglancer()
    # A ServedLayer, a neu_lib.Piece, or the dict form — the same three `add_layer` takes.
    layers = [_as_served_layer(ln) for ln in layers]
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
    # The viewer's own dimensions, from the first layer's frame and SPATIAL only — a
    # channel axis is local to the layer, not something to navigate. Set explicitly
    # because a `dimensions` block that disagrees with the data loads cleanly and puts
    # every layer in the wrong place, which is this package's oldest silent failure.
    spatial = _coordinate_space(ng, layers[0].frame, name=layers[0].name)
    annotate_name = ((ANNOTATE_LAYER if annotations is True else str(annotations))
                     if annotations else None)
    # Built before the transaction so the layers can go in through `Server._add_layer_in`,
    # the same builder `Server.add_layer` uses. It only holds references at this point.
    server = Server(viewer, {}, annotate_name)

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
                    s.layers[name] = existing

        for i, layer in enumerate(layers):
            # The index-based default is `serve`'s own: only here is the total known, and
            # `image0`/`image1` reads better than the `image`/`image_1` a bare collision
            # rename would give. Everything after the name is `_add_layer_in`'s.
            server._add_layer_in(
                s, layer, layer.name or f"{layer.kind}{'' if len(layers) == 1 else i}")

        if annotate_name:
            s.layers[annotate_name] = _annotation_layer(ng, spatial)
        # **Centre on the data and zoom to fit it.** `state.default_view`'s own docstring
        # is about this: with no position neuroglancer opens at the origin CORNER, and with
        # no crossSectionScale at one voxel per pixel — so a crop sitting at voxel 3700 of
        # its parent opens as empty space, with nothing to say the data is elsewhere. The
        # union of the layers' boxes, so a small crop over a larger image still frames both.
        if position is not None:
            s.position = [float(v) for v in position]
        else:
            view = _opening_view(layers, layout=layout, fit=fit)
            if view is not None:
                s.position, s.cross_section_scale, s.projection_scale = view
        if layout and not (into and into.get("layout")):
            s.layout = layout

    logger.info("serving %d layer(s) at %s — .stop() or neu_glance.stop_serving() to end it",
                len(server.volumes), server.url)
    return server
