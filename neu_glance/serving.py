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
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

#: The one encoding that survives NumPy 2 — see the module docstring. Not a parameter
#: anywhere, deliberately: the failure it prevents happens in the browser, minutes after
#: everything here has reported success.
ENCODING = "npz"

#: What a served array can be. `probability` is an image as far as neuroglancer is
#: concerned; it differs in the shader it gets and in tolerating a channel axis.
KINDS = ("image", "segmentation", "probability")

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

    ``channel_axis`` marks a leading channel axis (a 3-channel probability map). The
    package convention is channel-first, matching ``has_channels`` elsewhere.
    """
    array: Any
    kind: str = "image"
    name: str | None = None
    frame: Any = None
    shader: str | None = None
    channel_axis: bool = False
    opacity: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ServeProblem(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not hasattr(self.array, "shape") or not hasattr(self.array, "dtype"):
            raise ServeProblem(
                f"layer {self.name!r}: expected an array with .shape and .dtype, got "
                f"{type(self.array).__name__}")
        spatial = len(self.array.shape) - (1 if self.channel_axis else 0)
        if spatial != 3:
            raise ServeProblem(
                f"layer {self.name!r}: {len(self.array.shape)}-D array "
                f"{tuple(self.array.shape)} is {spatial} spatial axes; a served volume is "
                f"3, optionally with a leading channel axis (pass channel_axis=True)")


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
