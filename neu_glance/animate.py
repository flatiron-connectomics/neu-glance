"""Keyframe animation over neuroglancer states — a timeline, and the interpolation under it.

A :class:`Timeline` wraps one **base state** (the link you tuned in the browser) and a list
of :class:`Tween`\\ s, each naming a single property, a window in time and an easing.
Everything no tween mentions holds its base value, so a state can be edited — a new segment,
a different source, a colour change — without re-applying that edit to twenty near-identical
keyframes.

**Pure, like :mod:`neu_glance.state` and :mod:`neu_glance.layers`.** Plain dicts in, plain
dicts out; stdlib only, no numpy, no store, and — deliberately — no ``neuroglancer``. Which
means the whole of this module, where all the subtle bugs are, is covered by a CI that
installs no extras. :mod:`neu_glance.rendering` is the half that needs a viewer.

**Why the interpolation is ours rather than neuroglancer's.** ``ViewerState.interpolate``
exists and looks like exactly the thing to call. Three reasons not to, in order of weight:

1. **It is broken.** ``Layer.interpolate`` (``viewer_state.py:437-443`` in 2.41.2) reads
   ``a.layer_position``; the property is ``local_position``/``localPosition``. It raises
   ``AttributeError`` for a state carrying *any* layer, of any type — so
   ``neuroglancer.tool.video_tool`` is dead as shipped too.
2. **Even fixed, it does not cover this.** ``SegmentationLayer.interpolate`` handles
   ``selectedAlpha``, ``notSelectedAlpha`` and ``objectAlpha`` — not
   ``meshSilhouetteRendering``, not ``segmentColors``. Calling it would mean reimplementing
   most of it anyway.
3. **It does not round-trip a state.** On a state carrying no orientation it *invents*
   ``projectionOrientation`` and ``crossSectionOrientation``, because
   ``quaternion_slerp(None, None, t)`` returns the identity. What comes back is not what went
   in. This module writes only what a tween touched.

The *maths*, though, is right and worth not re-deriving: :func:`interpolate_zoom` and
:func:`interpolate_quaternion` are ported from ``viewer_state.py:75-111``, sign flip and
small-angle fallback included, and a test pins ours against theirs to 1e-6 so the two cannot
drift.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Mapping, Sequence

from .shaders import as_hex_color
from .state import (FIT_MARGIN, StateProblem, default_view, parse_url,
                    split_segment_layer, subset_layers)


class AnimateProblem(RuntimeError):
    """A timeline could not be built, or would not animate what it claims to."""


#: Where a fade starts, rather than exactly zero. Neuroglancer's mesh draw path is
#: ``if (objectAlpha <= 0) return``, so a layer at a true zero is skipped entirely — and it
#: is not established whether its chunks then still count as "visible" for the screenshot
#: reply, which is what makes a capture wait for data to load. A hair above zero is
#: indistinguishable on screen and keeps the layer unambiguously in the scene, so the mesh is
#: there by the time the fade becomes visible.
FADE_FROM = 0.001

#: The top-level keys :meth:`Timeline.view` lifts out of a state. Everything a camera move
#: consists of, and nothing else — a keyframe URL also carries layers and a layout, which are
#: emphatically not what "move the camera to here" should change.
CAMERA_KEYS = ("position", "projectionOrientation", "projectionScale", "projectionDepth",
               "crossSectionOrientation", "crossSectionScale")

#: Neuroglancer's own defaults, read off the ``optional(...)`` declarations in
#: ``viewer_state.py`` (``SegmentationLayer`` at :941-975, ``ImageLayer.opacity`` at :604).
#:
#: **A property absent from a state does not mean zero**, and this is the table that keeps a
#: fade from starting in the wrong place. ``objectAlpha`` absent means fully opaque;
#: ``selectedAlpha`` absent means a half-transparent cross-section, not an invisible one. A
#: tween with no explicit ``start`` on a property the state never mentions reads its opening
#: value from here, and getting it wrong is invisible in the code and obvious only on screen.
LAYER_DEFAULTS: dict[str, Any] = {
    "objectAlpha": 1.0,
    "selectedAlpha": 0.5,
    "notSelectedAlpha": 0.0,
    "saturation": 1.0,
    "meshSilhouetteRendering": 0.0,
    "meshRenderScale": 10.0,
    "crossSectionRenderScale": 1.0,
    "opacity": 0.5,
    "colorSeed": 0,
    "hideSegmentZero": True,
    "visible": True,
    "archived": False,
    "pick": True,
}

#: The same, for the top-level view keys. ``projectionScale`` and ``crossSectionScale`` are
#: deliberately absent: neuroglancer declares them ``optional(float)`` with no default (it
#: fits the data instead), so there is no honest value to start a zoom from and a tween on one
#: must either find it in the state or be given an explicit ``start``.
VIEW_DEFAULTS: dict[str, Any] = {
    "projectionOrientation": [0.0, 0.0, 0.0, 1.0],
    "crossSectionOrientation": [0.0, 0.0, 0.0, 1.0],
    "showSlices": True,
    "showAxisLines": True,
    "showDefaultAnnotations": True,
    "showScaleBar": True,
    "layout": "4panel",
}


# --------------------------------------------------------------------------- #
# easing
# --------------------------------------------------------------------------- #
def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


#: Easing curves, applied to the normalized parameter **before** the interpolator — so an
#: eased exponential zoom is still exponential, and an eased slerp is still a great-circle
#: arc, just traversed at a varying rate.
#:
#: Every entry must satisfy ``e(0) == 0``, ``e(1) == 1`` and be monotone, and a test pins
#: that across the whole table. An overshooting curve (a "back" or "elastic" ease) would drive
#: ``objectAlpha`` past 1 or a colour channel out of range, and neuroglancer clamps without
#: saying anything — so the animation would simply flatten at the ends for no visible reason.
EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "in": lambda t: t * t,
    "out": lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    "in-out": _smoothstep,
    "in-out-cubic": _smootherstep,
    "step": lambda t: 0.0 if t < 1.0 else 1.0,
}

#: The default. "Slow and smooth" is a request for zero velocity at both ends, which is what
#: smoothstep gives and what a linear ramp conspicuously does not — a linear camera move
#: starts and stops with a jerk that reads as a dropped frame.
DEFAULT_EASE = "in-out"


def ease(name: str, t: float) -> float:
    """``EASINGS[name]`` applied to ``t``, clamped to ``[0, 1]``."""
    try:
        curve = EASINGS[name]
    except KeyError:
        raise AnimateProblem(
            f"unknown easing {name!r}; known: {', '.join(sorted(EASINGS))}") from None
    return curve(min(1.0, max(0.0, float(t))))


# --------------------------------------------------------------------------- #
# interpolators
# --------------------------------------------------------------------------- #
def interpolate_linear(a: Any, b: Any, t: float) -> Any:
    """Straight lerp, element-wise for equal-length sequences.

    Sequences of differing length hold ``a``, matching neuroglancer's
    ``interpolate_linear_optional_vectors``: a 3-vector and a 4-vector have no meaningful
    midpoint, and inventing one would put the camera somewhere neither state asked for.
    """
    if a is None or b is None:
        return a if b is None else b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
                and len(a) == len(b)):
            return a
        return [float(x) * (1.0 - t) + float(y) * t for x, y in zip(a, b)]
    return float(a) * (1.0 - t) + float(b) * t


def interpolate_zoom(a: Any, b: Any, t: float) -> Any:
    """Exponential interpolation, for the two zoom scales. Ported from ``viewer_state.py:107``.

    **This is the "slow and smooth" requirement itself, not a refinement of it.** Zoom is
    perceived multiplicatively: linear interpolation from 1000 to 4000 puts the midpoint at
    2500 where the eye expects 2000, so the move visibly accelerates towards the close end and
    reads as a camera that lost control.

    A non-positive endpoint has no logarithm, and a state can carry one. Rather than raising
    mid-render, fall back to a lerp — a slightly wrong-feeling move beats a dead render.
    """
    if a is None or b is None:
        return a if b is None else b
    a, b = float(a), float(b)
    if a <= 0.0 or b <= 0.0:
        return interpolate_linear(a, b, t)
    return a * math.exp(math.log(b / a) * t)


def interpolate_quaternion(a: Any, b: Any, t: float) -> list[float]:
    """Spherical linear interpolation. Ported from ``viewer_state.py:75-104`` (via gl-matrix).

    **The sign flip is the point.** ``q`` and ``-q`` are the same rotation, so without it a
    pair 20° apart can be interpolated the 340° way round and the camera takes a full lazy
    spin to arrive somewhere it was nearly already pointing. That is why this is a port rather
    than "a lerp plus a normalize".

    Inputs are assumed unit, as every orientation neuroglancer writes is; a non-unit input
    gives a non-unit output, exactly as upstream.
    """
    a = list(VIEW_DEFAULTS["projectionOrientation"]) if a is None else [float(v) for v in a]
    b = list(VIEW_DEFAULTS["projectionOrientation"]) if b is None else [float(v) for v in b]
    cosom = sum(x * y for x, y in zip(a, b))
    if cosom < 0.0:
        cosom = -cosom
        b = [-y for y in b]
    if (1.0 - cosom) > 0.000001:
        omega = math.acos(max(-1.0, min(1.0, cosom)))
        sinom = math.sin(omega)
        scale0 = math.sin((1.0 - t) * omega) / sinom
        scale1 = math.sin(t * omega) / sinom
    else:
        scale0, scale1 = 1.0 - t, t
    return [scale0 * x + scale1 * y for x, y in zip(a, b)]


#: Named rotation axes for :meth:`Timeline.orbit`. ``up``/``right``/``forward`` are the
#: **screen's** axes, so they mean the same thing from any starting view — "spin it
#: left-to-right as I am looking at it" — while ``x``/``y``/``z`` are the **volume's** own,
#: so a shot is reproducible across scenes and anatomically meaningful.
AXES: dict[str, tuple[float, float, float]] = {
    "up": (0.0, 1.0, 0.0), "right": (1.0, 0.0, 0.0), "forward": (0.0, 0.0, 1.0),
    "x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0),
}

#: Which axes are read in the camera's frame rather than the volume's.
SCREEN_AXES = frozenset({"up", "right", "forward"})


def _quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Hamilton product, ``xyzw`` order — neuroglancer's, and gl-matrix's."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def axis_quaternion(axis: Any, degrees: float) -> list[float]:
    """A rotation of ``degrees`` about ``axis`` as a unit quaternion, ``xyzw``."""
    vector = AXES.get(axis) if isinstance(axis, str) else axis
    if vector is None:
        raise AnimateProblem(
            f"unknown axis {axis!r}; use a 3-sequence or one of: {', '.join(sorted(AXES))}")
    length = math.sqrt(sum(float(v) ** 2 for v in vector))
    if length == 0.0:
        raise AnimateProblem("an orbit axis cannot be the zero vector")
    half = math.radians(float(degrees)) / 2.0
    scale = math.sin(half) / length
    return [float(vector[0]) * scale, float(vector[1]) * scale, float(vector[2]) * scale,
            math.cos(half)]


def interpolate_orbit(a: Any, b: Any, t: float) -> list[float]:
    """Rotate ``a`` progressively about an axis. ``b`` is ``{"axis", "degrees", "screen"}``.

    **This is why orbiting cannot be expressed as an interpolation between two captured
    views, and the reason it exists as its own kind.** Slerp always takes the shortest arc
    between two orientations, so a full turn is a no-op (start and end are the same rotation)
    and 200 degrees silently becomes 160 the other way. No number of keyframes fixes that; it
    is a property of the interpolation. Here the *angle* is what varies, so any rotation is
    expressible, including several full turns.

    ``screen`` picks the frame the axis is read in — the camera's own (post-multiply, so the
    scene spins the way it looks on screen from any starting view) or the volume's
    (pre-multiply, so the shot is reproducible and anatomically meaningful).
    """
    start = list(VIEW_DEFAULTS["projectionOrientation"]) if a is None else [float(v) for v in a]
    spin = axis_quaternion(b["axis"], float(b["degrees"]) * t)
    return (_quaternion_multiply(start, spin) if b.get("screen", True)
            else _quaternion_multiply(spin, start))


def interpolate_zoom_by(a: Any, b: Any, t: float) -> Any:
    """Scale ``a`` by a factor, geometrically. ``b`` is the factor at the end of the window.

    The relative sibling of :func:`interpolate_zoom`: ``zoom(0.5)`` halves the scale — twice
    as close — from wherever the camera happens to be, without the caller having to know that
    number. Exponential for the same reason, so the rate of approach is even.
    """
    if a is None:
        return a
    return float(a) * (float(b) ** t)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def interpolate_hex_color(a: Any, b: Any, t: float, *, space: str = "linear") -> str:
    """A colour ramp, in **linear light** by default. Returns ``#rrggbb``.

    Both endpoints go through :func:`neu_glance.shaders.as_hex_color`, so a name (``"red"``),
    a hex string and a 3-sequence all work and there is no second colour parser here.

    **Why linear light.** Hex channels are sRGB-encoded, i.e. already gamma-compressed, so
    averaging them is not averaging light. Halfway from black to red is ``#800000`` that way,
    which reads as a dark maroon rather than half-lit — and every ramp in a fade-in animation
    has black or white at one end, so this is the common case, not a corner. Decoding to
    linear, mixing, and re-encoding gives a midpoint that looks like a midpoint.

    ``space="srgb"`` is the naive form, kept reachable for matching a ramp made in another
    tool. The linear midpoint is pinned by a test, because a later "simplification" to a
    straight lerp would silently change the look of every render.
    """
    lo = as_hex_color(a, "colour")
    hi = as_hex_color(b, "colour")
    pairs = [(int(lo[1 + 2 * i:3 + 2 * i], 16) / 255.0,
              int(hi[1 + 2 * i:3 + 2 * i], 16) / 255.0) for i in range(3)]
    if space == "linear":
        mixed = [_linear_to_srgb(_srgb_to_linear(x) * (1.0 - t) + _srgb_to_linear(y) * t)
                 for x, y in pairs]
    elif space == "srgb":
        mixed = [x * (1.0 - t) + y * t for x, y in pairs]
    else:
        raise AnimateProblem(f"unknown colour space {space!r}; use 'linear' or 'srgb'")
    return "#" + "".join(f"{int(round(min(1.0, max(0.0, c)) * 255)):02x}" for c in mixed)


def interpolate_color_map(a: Any, b: Any, t: float, *, default: str | None = None) -> dict:
    """``segmentColors``, interpolated per segment id.

    A key on both sides ramps. A key on only one side has nothing to ramp *from*, so it falls
    back to the layer's ``segmentDefaultColor`` when there is one and otherwise **steps** —
    said out loud here because a segment popping to a new colour partway through a fade, with
    no explanation, gets blamed on the data rather than on this.
    """
    a = {str(k): v for k, v in (a or {}).items()}
    b = {str(k): v for k, v in (b or {}).items()}
    out: dict[str, str] = {}
    for key in list(a) + [k for k in b if k not in a]:
        if key in a and key in b:
            out[key] = interpolate_hex_color(a[key], b[key], t)
        elif key in a:                                  # dropped: ramp out towards the default
            out[key] = (interpolate_hex_color(a[key], default, t) if default is not None
                        else interpolate_step(a[key], None, t))
        else:                                           # added: ramp in from the default
            out[key] = (interpolate_hex_color(default, b[key], t) if default is not None
                        else interpolate_step(None, b[key], t))
        if out[key] is None:
            del out[key]
    return out


def interpolate_step(a: Any, b: Any, t: float) -> Any:
    """Hold ``a``, take ``b`` at the **end** of the transition.

    Not at the midpoint: a boolean or a layout flipping halfway through an otherwise smooth
    move is a visible artefact with nothing on screen to motivate it. "It changes when the
    transition completes" is the reading that matches what an author writing
    ``visible=False`` at ``t=8`` means, and a change wanted elsewhere is a zero-length tween
    placed there.
    """
    return b if t >= 1.0 else a


#: Interpolator by name.
INTERPOLATORS: dict[str, Callable[[Any, Any, float], Any]] = {
    "linear": interpolate_linear,
    "zoom": interpolate_zoom,
    "slerp": interpolate_quaternion,
    "color": interpolate_hex_color,
    "colors": interpolate_color_map,
    "step": interpolate_step,
    # Relative moves. Their `end` is a delta rather than a destination, so they are never
    # reached through PROPERTY_KINDS — a tween has to name them explicitly, which
    # `Timeline.orbit` and `Timeline.zoom` do.
    "orbit": interpolate_orbit,
    "zoom_by": interpolate_zoom_by,
}

#: Which interpolator a property gets. **Anything not named here STEPS**, and that default is
#: load-bearing: a property nobody thought about must not quietly acquire a plausible-looking
#: midpoint. A blended ``source`` URL or a two-thirds-applied shader is not a thing.
PROPERTY_KINDS: dict[str, str] = {
    # per-layer, continuous
    "objectAlpha": "linear",
    "opacity": "linear",
    "selectedAlpha": "linear",
    "notSelectedAlpha": "linear",
    "saturation": "linear",
    "meshSilhouetteRendering": "linear",
    "meshRenderScale": "linear",
    "crossSectionRenderScale": "linear",
    # top-level view
    "position": "linear",
    "crossSectionDepth": "linear",
    "projectionDepth": "linear",
    "projectionScale": "zoom",
    "crossSectionScale": "zoom",
    "projectionOrientation": "slerp",
    "crossSectionOrientation": "slerp",
    # colour
    "segmentDefaultColor": "color",
    "annotationColor": "color",
    "segmentColors": "colors",
    # explicitly stepped, so the table documents the decision rather than relying on the
    # default. `segments` above all: a fractional segment list is meaningless, and it is the
    # property most likely to be reached for instead of splitting the layer.
    "segments": "step",
    "visible": "step",
    "archived": "step",
    "layout": "step",
    "shader": "step",
    "source": "step",
}


def kind_for(prop: str) -> str:
    """The interpolator name for ``prop``. Unknown properties step; see :data:`PROPERTY_KINDS`."""
    return PROPERTY_KINDS.get(prop, "step")


# --------------------------------------------------------------------------- #
# addressing
# --------------------------------------------------------------------------- #
# A path is ("<view property>",) or ("layers", "<layer name>", "<layer property>").
#
# A TUPLE, never a dotted string. Layer names routinely contain dots and spaces — the wasp
# scene has `primary.precomputed` and `head.precomputed` — so any dotted grammar needs an
# escaping scheme, and an escaping scheme needs a parser nobody wants to own or debug.
def _layer_in(state: Mapping[str, Any], name: str) -> dict | None:
    for lyr in state.get("layers") or []:
        if isinstance(lyr, dict) and lyr.get("name") == name:
            return lyr
    return None


def _get_at(state: Mapping[str, Any], path: Sequence[str]) -> Any:
    if len(path) == 1:
        return state.get(path[0])
    layer = _layer_in(state, path[1])
    return None if layer is None else layer.get(path[2])


def _set_at(state: dict, path: Sequence[str], value: Any) -> None:
    if len(path) == 1:
        state[path[0]] = value
        return
    layer = _layer_in(state, path[1])
    if layer is not None:
        layer[path[2]] = value


def _default_for(path: Sequence[str]) -> Any:
    table = VIEW_DEFAULTS if len(path) == 1 else LAYER_DEFAULTS
    return table.get(path[-1])


def _describe(path: Sequence[str]) -> str:
    return path[0] if len(path) == 1 else f"{path[2]} on layer {path[1]!r}"


# --------------------------------------------------------------------------- #
# tweens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tween:
    """One property, moving from one value to another over one window of time."""

    path: tuple[str, ...]
    end: Any
    start: Any = None
    at: float = 0.0
    seconds: float = 0.0
    ease: str = DEFAULT_EASE
    kind: str | None = None

    @property
    def stop(self) -> float:
        return self.at + self.seconds

    def value_at(self, t: float, start: Any) -> Any:
        """The value at ``t``, or :data:`_NOTHING` when this tween has not begun.

        Two boundaries, each of which fails quietly if put the other way round:

        - **After the window closes the tween PINS its end value.** A fade that quietly
          un-fades once its window passes reads as a data problem, not a timeline bug.
        - **Before it opens the tween contributes nothing**, so whatever held the property
          before — the base state, or an earlier tween — still shows.
        """
        interpolate = INTERPOLATORS[self.kind or kind_for(self.path[-1])]
        # Past the window, the pinned value is the interpolator AT t=1, never `end` itself.
        # For an absolute kind those are the same thing; for a RELATIVE one `end` is a delta —
        # an orbit's axis and angle, a zoom's factor — so returning it verbatim would put a
        # dict where the quaternion belongs, or a bare 0.25 where the zoom belongs. It is the
        # last frame of the move that is wrong, which is the one most likely to be checked
        # last.
        if self.seconds <= 0.0:
            return interpolate(start, self.end, 1.0) if t >= self.at else _NOTHING
        if t >= self.stop:
            return interpolate(start, self.end, 1.0)
        if t <= self.at:
            return _NOTHING
        return interpolate(start, self.end, ease(self.ease, (t - self.at) / self.seconds))

    def to_json(self) -> dict:
        obj = {"path": list(self.path), "end": self.end, "at": self.at,
               "seconds": self.seconds, "ease": self.ease}
        if self.start is not None:
            obj["start"] = self.start
        if self.kind is not None:
            obj["kind"] = self.kind
        return obj

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Tween":
        return cls(path=tuple(obj["path"]), end=obj["end"], start=obj.get("start"),
                   at=float(obj.get("at", 0.0)), seconds=float(obj.get("seconds", 0.0)),
                   ease=obj.get("ease", DEFAULT_EASE), kind=obj.get("kind"))


class _Nothing:
    def __repr__(self) -> str:                                          # pragma: no cover
        return "<no contribution>"


#: Distinct from ``None``, which is a legitimate property value.
_NOTHING = _Nothing()

#: How close two window edges have to be to count as touching rather than overlapping.
#: A nanosecond is far below any timing an animation expresses — the shortest sensible
#: interval is a frame, ~33 ms — and far above the float error in the sums that produce them.
_TIME_EPSILON = 1e-9


# --------------------------------------------------------------------------- #
# the timeline
# --------------------------------------------------------------------------- #
@dataclass
class Timeline:
    """A base state plus tweens over it. Every frame it emits is a plain state dict.

    ``fps`` is carried here rather than passed to the renderer so that a timeline written
    beside its frames describes them completely — the frame at index *i* is at ``i / fps``,
    and nothing outside this object is needed to say so.
    """

    base: dict
    fps: float = 30.0
    tweens: list[Tween] = field(default_factory=list)
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cursor: float = 0.0
    notes: list[str] = field(default_factory=list)
    _duration: float | None = None

    def __post_init__(self) -> None:
        self.base = json.loads(json.dumps(dict(self.base)))   # never alias the caller's state

    # -- building ---------------------------------------------------------- #
    def split(self, layer: str, *, prefix: str | None = None,
              names: Mapping[str, str] | None = None,
              segments: Sequence[Any] | None = None) -> dict[str, str]:
        """One layer per segment of ``layer``. Returns ``{segment_id: layer_name}``.

        This is what makes a per-object fade possible at all — see
        :func:`neu_glance.split_segment_layer` for why neuroglancer leaves no alternative.
        The returned mapping is what group definitions are written against, so segment ids
        stay the thing you name and layer names stay an implementation detail.
        """
        template = f"{prefix} {{segment}}" if prefix else "{layer} {segment}"
        self.base, created, notes = split_segment_layer(
            self.base, layer, names=names, segments=segments, name_template=template)
        self.notes.extend(notes)
        return created

    def sets(self, layer: str, subsets: Mapping[str, Sequence[Any]], *,
             colors: Mapping[str, str] | None = None,
             object_alpha: float | None = None) -> dict[str, str]:
        """One layer per named SET of segments. Returns ``{set: layer_name}``.

        The populations sibling of :meth:`split` — see
        :func:`neu_glance.subset_layers`. Use this when whole groups fade together and
        :meth:`split` when individual objects do; four hundred Kenyon cells want one layer,
        not four hundred.
        """
        self.base, created, notes = subset_layers(
            self.base, layer, subsets, colors=colors, object_alpha=object_alpha)
        self.notes.extend(notes)
        return created

    def set(self, *, layer: str | None = None, **props: Any) -> "Timeline":
        """An opening value at ``t=0``, held until a later tween takes the property over."""
        for prop, value in props.items():
            self._add(Tween(path=self._path(layer, prop), end=value, at=0.0, seconds=0.0))
        return self

    def tween(self, *, layer: str | None = None, at: float | None = None,
              seconds: float, ease: str = DEFAULT_EASE, start: Any = None,
              kind: str | None = None, **props: Any) -> "Timeline":
        """Move each named property to its given value over ``seconds``, starting at ``at``.

        ``at=None`` uses the cursor, which only :meth:`hold` moves — nothing here advances it
        implicitly, so two tweens written one after the other run *together* unless you say
        otherwise. That is the behaviour a fade-and-move-at-once needs, and the alternative
        (an implicitly advancing cursor) makes simultaneity the awkward case.

        ``start=None`` reads the value the property last held: an earlier tween's end if there
        is one, else the base state, else :data:`LAYER_DEFAULTS`/:data:`VIEW_DEFAULTS`. **Not
        zero** — a property a state does not mention is very often 1.0.
        """
        when = self.cursor if at is None else float(at)
        for prop, value in props.items():
            self._add(Tween(path=self._path(layer, prop), end=value, start=start,
                            at=when, seconds=float(seconds), ease=ease, kind=kind))
        return self

    def group(self, name: str, layers: Sequence[str]) -> "Timeline":
        """Name a set of layers that appear together — a neuron and the neuropils it innervates."""
        missing = [n for n in layers if _layer_in(self.base, n) is None]
        if missing:
            raise AnimateProblem(
                f"group {name!r} names layer(s) not in this state: {', '.join(missing)}. "
                f"Split a segmentation layer first, and build groups from what split() returns")
        self.groups[name] = tuple(layers)
        return self

    def sequence(self, names: Sequence[str], *, start: float | None = None,
                 seconds: float = 1.2, stagger: float | None = None,
                 ease: str = DEFAULT_EASE, fade_from: float = FADE_FROM) -> "Timeline":
        """Fade in each group in turn, ``stagger`` seconds apart. Returns ``self``.

        ``stagger=None`` means back to back (``stagger = seconds``). A stagger shorter than
        ``seconds`` overlaps the fades, which usually reads better than strict succession.

        Each layer also gets an opening ``objectAlpha`` of ``fade_from`` at ``t=0``, so the
        scene starts empty without the caller having to remember to say so — the failure
        otherwise being an animation where everything is already visible and the fades do
        nothing at all.
        """
        begin = self.cursor if start is None else float(start)
        step = float(seconds) if stagger is None else float(stagger)
        for i, name in enumerate(names):
            try:
                layers = self.groups[name]
            except KeyError:
                raise AnimateProblem(
                    f"no group named {name!r}; defined: {', '.join(sorted(self.groups)) or '(none)'}"
                ) from None
            for layer in layers:
                self.set(layer=layer, objectAlpha=fade_from)
                self.tween(layer=layer, at=begin + i * step, seconds=seconds, ease=ease,
                           objectAlpha=1.0)
        return self

    def view(self, target: Mapping[str, Any] | str, *, at: float | None = None,
             seconds: float, ease: str = DEFAULT_EASE) -> "Timeline":
        """Move the camera to the view held by another state. Returns ``self``.

        ``target`` is a state dict or a neuroglancer URL — **the ergonomic centre of this
        module**. Framing a 3D view well is a thing done by dragging, not by writing
        quaternions, so the intended workflow is to fly the camera in the browser, copy the
        link, and paste it here. Only :data:`CAMERA_KEYS` are taken; the target's layers,
        layout and segment lists are ignored, because "move the camera to here" must not
        quietly swap the scene.
        """
        if isinstance(target, str):
            try:
                target = parse_url(target)
            except (ValueError, StateProblem) as e:
                raise AnimateProblem(
                    f"view() takes a state dict or a neuroglancer URL: {e}. For a state on "
                    f"disk, read it with neu_glance.load_state() first") from None
        present = {k: target[k] for k in CAMERA_KEYS if k in target}
        if not present:
            raise AnimateProblem(
                f"that state carries none of the camera keys ({', '.join(CAMERA_KEYS)}), so "
                f"there is no view in it to move to")
        return self.tween(at=at, seconds=seconds, ease=ease, **present)

    def orbit(self, degrees: float, *, axis: Any = "up", at: float | None = None,
              seconds: float, ease: str = DEFAULT_EASE) -> "Timeline":
        """Rotate the camera ``degrees`` about ``axis``. Relative — no keyframe needed.

        ``axis`` defaults to ``"up"``, the **screen's** vertical, so the scene spins
        left-to-right as you are looking at it whatever the starting view: a turntable. Name
        a volume axis (``"z"``, or a 3-sequence) when the shot should be anatomically
        meaningful and reproducible across scenes instead.

        A full turn is ``orbit(360)``. That is the case :func:`interpolate_orbit` exists for —
        it cannot be written as an interpolation between two views, because slerp takes the
        shortest arc and the two ends of a full turn are the same orientation.
        """
        axis_quaternion(axis, 0.0)      # validate NOW, not at frame 400 of the render
        return self.tween(at=at, seconds=seconds, ease=ease, kind="orbit",
                          projectionOrientation={"axis": axis, "degrees": float(degrees),
                                                 "screen": (isinstance(axis, str)
                                                            and axis in SCREEN_AXES)})

    def zoom(self, factor: float, *, at: float | None = None, seconds: float,
             ease: str = DEFAULT_EASE) -> "Timeline":
        """Scale the 3D zoom by ``factor``. Relative: ``0.5`` is twice as close.

        Multiplicative rather than absolute so it composes with everything else and needs no
        knowledge of the current ``projectionScale`` — which after a :meth:`frame_on` is a
        number nobody wrote down.
        """
        if factor <= 0:
            raise AnimateProblem(f"a zoom factor must be positive, got {factor!r}")
        return self.tween(at=at, seconds=seconds, ease=ease, kind="zoom_by",
                          projectionScale=float(factor))

    def voxel_size_nm(self) -> tuple[float, float, float]:
        """The state's own voxel size in nm, zyx, from its ``dimensions``."""
        dims = self.base.get("dimensions") or {}
        try:
            return tuple(float(dims[axis][0]) * 1e9 for axis in ("z", "y", "x"))
        except (KeyError, TypeError, IndexError):
            raise AnimateProblem(
                "this state has no usable `dimensions`, so a box in nanometres cannot be "
                "converted to the coordinates its `position` is written in. Pass "
                "units='voxels' if the box is already in those.") from None

    def frame_on(self, box: Any, *, at: float | None = None, seconds: float,
                 ease: str = DEFAULT_EASE, margin: float = 1.15, units: str = "nm",
                 zoom: bool = True, min_scale: float | None = None,
                 rotation_safe: bool = False) -> "Timeline":
        """Pan (and by default zoom) so ``box`` fills the view.

        ``box`` is ``(lo_zyx, hi_zyx)``, or anything with ``.lo``/``.hi`` so a
        :class:`neu_lib.BBox` works. :func:`neu_glance.sources.segment_boxes` produces them
        from a volume's meshes, which is the intended way to say "point the camera at that
        neuron" without reading coordinates off the screen.

        ``units`` defaults to ``"nm"``, because nanometres are the suite's model space
        (NM-SPACE) and what `segment_boxes` returns — but neuroglancer's ``position`` is in
        the state's own **voxels**, so the two are converted here using the state's
        ``dimensions``. Getting that wrong does not raise: it points the camera somewhere
        plausible and wrong by the voxel size, which on an 8 nm volume is a factor of eight.
        Pass ``units="voxels"`` for a box already in the viewer's coordinates.

        The framing itself is :func:`neu_glance.state.default_view` — what ``neu-glance gen``
        already uses to open a link on a whole volume, so a framed shot and a generated link
        agree about what "fits" means. ``margin`` is its ``FIT_MARGIN``, room around the
        object; ``zoom=False`` pans without changing the zoom, for following something at a
        fixed scale rather than fitting it.

        **``rotation_safe=True`` if the shot turns.** ``projectionScale`` sets how much world
        the frame's HEIGHT spans (measured: the vertical fill of a fixed object is constant
        across aspect ratios while the horizontal is not), so a box framed at one angle can
        exceed the frame at another as its long axis swings towards the vertical. This fits
        the box's diagonal — its bounding sphere — which no rotation can grow.

        Centring matters as much as scale here, and this does both: an orbit pivots about the
        state's ``position``, so an object framed at a pivot 10 um off its centre swings
        bodily through the frame and leaves it, whatever the zoom.
        """
        lo, hi = (box.lo, box.hi) if hasattr(box, "lo") else box
        lo, hi = [float(v) for v in lo], [float(v) for v in hi]
        if units == "nm":
            voxel = self.voxel_size_nm()
            lo = [v / s for v, s in zip(lo, voxel)]
            hi = [v / s for v, s in zip(hi, voxel)]
        elif units != "voxels":
            raise AnimateProblem(f"units must be 'nm' or 'voxels', got {units!r}")
        extent = [max(1e-9, b - a) for a, b in zip(lo, hi)]
        centre, _cross, projection = default_view(extent, lo)
        moves: dict[str, Any] = {"position": [float(v) for v in centre[::-1]]}
        if zoom:
            scale = float(projection) * float(margin) / FIT_MARGIN
            if rotation_safe:
                # The box's DIAGONAL, i.e. its bounding sphere — the only extent that cannot
                # grow under rotation. Framing on the box itself fits the object at the angle
                # you framed it and clips at others, and the clipping is worst where the long
                # axis swings towards the vertical. Costs empty space at every angle in
                # exchange for never losing the subject at any.
                scale *= math.sqrt(sum(e * e for e in extent)) / max(extent)
            # `min_scale` is a floor on how CLOSE the camera will go, and it is about motion
            # rather than framing. Fitting each object exactly means the zoom travels as far
            # as the objects differ in size, so a sequence visiting a large region and then a
            # small one lurches in and out — the small ones read as rushed however long the
            # move is given, because the distance covered is what makes it feel fast. A floor
            # keeps the small ones a little loose and the sequence even.
            moves["projectionScale"] = max(scale, float(min_scale)) if min_scale else scale
        return self.tween(at=at, seconds=seconds, ease=ease, **moves)

    def hold(self, seconds: float) -> "Timeline":
        """Advance the cursor. The only thing that moves it."""
        self.cursor += float(seconds)
        return self

    def _path(self, layer: str | None, prop: str) -> tuple[str, ...]:
        if layer is None:
            return (prop,)
        if _layer_in(self.base, layer) is None:
            known = ", ".join(repr(lyr.get("name")) for lyr in self.base.get("layers") or []
                              if isinstance(lyr, dict))
            raise AnimateProblem(
                f"no layer named {layer!r} in this state. It has: {known}")
        return ("layers", layer, prop)

    def _add(self, tween: Tween) -> None:
        for other in self.tweens:
            if other.path != tween.path:
                continue
            # Touching endpoints are fine — one tween handing off to the next is the normal
            # way to write a there-and-back. Genuine overlap is not: silently dropping one of
            # two fades on one property would leave an animation that is subtly wrong and has
            # nothing anywhere to point at.
            #
            # The tolerance is what makes "abutting" survive arithmetic. A caller building a
            # sequence writes `start + i * step` for one window and `start + (i+1) * step` for
            # the next, and those reach the same instant by different routes:
            # `4.0 + 7*0.8 + 0.8` is 10.400000000000002 while `4.0 + 8*0.8` is 10.4. Without
            # a tolerance a perfectly-formed follow-the-action sequence is refused for an
            # overlap of two femtoseconds, and the message is baffling because the printed
            # numbers are identical.
            if tween.at < other.stop - _TIME_EPSILON and other.at < tween.stop - _TIME_EPSILON:
                raise AnimateProblem(
                    f"two tweens on {_describe(tween.path)} overlap: "
                    f"[{other.at:g}, {other.stop:g}] and [{tween.at:g}, {tween.stop:g}]. "
                    f"Split them into non-overlapping windows, or set the value once")
            if tween.seconds == 0.0 and other.seconds == 0.0 and tween.at == other.at:
                raise AnimateProblem(
                    f"{_describe(tween.path)} is set twice at t={tween.at:g}")
        self.tweens.append(tween)

    # -- reading out -------------------------------------------------------- #
    @property
    def duration(self) -> float:
        """Seconds. The furthest tween end, or the cursor, whichever is later."""
        if self._duration is not None:
            return self._duration
        return max([self.cursor] + [t.stop for t in self.tweens])

    @duration.setter
    def duration(self, seconds: float | None) -> None:
        self._duration = None if seconds is None else float(seconds)

    @property
    def frame_count(self) -> int:
        """Frames, **including** the last one — a 2 s timeline at 30 fps is 61 frames."""
        return int(round(self.duration * self.fps)) + 1

    def resolved(self) -> list[Tween]:
        """The tweens in application order, each with its ``start`` filled in.

        Chaining happens here: on any one property the first tween starts from the base state
        and each later one starts where the previous ended. Resolving once, up front, is what
        makes ``tween(...); tween(...)`` on the same property mean what it looks like — a
        second tween reading the *base* value instead would snap back to the opening state
        every time.
        """
        out: list[Tween] = []
        previous: dict[tuple[str, ...], Any] = {}
        for tween in sorted(self.tweens, key=lambda t: (t.path, t.at, t.seconds)):
            start = tween.start
            if start is None:
                if tween.path in previous:
                    start = previous[tween.path]
                else:
                    start = _get_at(self.base, tween.path)
                    if start is None:
                        start = _default_for(tween.path)
            if start is None and (tween.kind or kind_for(tween.path[-1])) != "step":
                raise AnimateProblem(
                    f"nothing to start from for {_describe(tween.path)}: the state does not "
                    f"carry it and there is no known default. Pass start= explicitly")
            out.append(replace(tween, start=start))
            # What the NEXT tween on this path starts from is where this one ends up, which
            # is not always its `end`: a relative move's `end` is a delta (an orbit's axis and
            # angle, a zoom's factor), so the value has to be evaluated. Running the
            # interpolator at t=1 is exactly that, and for every absolute kind it returns
            # `end` unchanged — so one rule covers both.
            previous[tween.path] = INTERPOLATORS[tween.kind or kind_for(tween.path[-1])](
                start, tween.end, 1.0)
        return sorted(out, key=lambda t: (t.at, t.seconds, t.path))

    def at(self, seconds: float) -> dict:
        """The state at ``seconds``. A plain dict; the caller's base is never mutated."""
        state = copy.deepcopy(self.base)
        for tween in self.resolved():
            value = tween.value_at(float(seconds), tween.start)
            if value is not _NOTHING:
                _set_at(state, tween.path, value)
        return state

    def frames(self, *, start_frame: int = 0,
               end_frame: int | None = None) -> Iterator[tuple[int, float, dict]]:
        """``(index, seconds, state)`` per frame.

        Times are ``index / fps`` exactly, never accumulated — a running sum drifts, and a
        drifting clock puts the last frames of a long render at times no keyframe ever named.
        """
        stop = self.frame_count if end_frame is None else min(end_frame, self.frame_count)
        for i in range(max(0, start_frame), stop):
            t = i / self.fps
            yield i, t, self.at(t)

    def check(self) -> list[str]:
        """Raise on anything unworkable; return notes worth printing. Called by the renderer."""
        notes = list(self.notes)
        self.resolved()
        if self.duration <= 0:
            notes.append("this timeline has zero duration, so it renders a single frame")
        for tween in self.tweens:
            if len(tween.path) == 3 and _layer_in(self.base, tween.path[1]) is None:
                raise AnimateProblem(f"tween names layer {tween.path[1]!r}, which is not in "
                                     f"the state any more")
            if (tween.kind or kind_for(tween.path[-1])) == "step" and tween.seconds > 0:
                notes.append(f"{_describe(tween.path)} does not interpolate, so its "
                             f"{tween.seconds:g}s window is a step at the end of it")
        return notes

    # -- serialization ------------------------------------------------------ #
    def to_json(self) -> dict:
        """The whole timeline, base state included, for writing beside the frames."""
        return {"fps": self.fps, "duration": self.duration, "base": self.base,
                "groups": {k: list(v) for k, v in self.groups.items()},
                "tweens": [t.to_json() for t in self.tweens]}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Timeline":
        """The inverse of :meth:`to_json`, so a rendered timeline can be edited and re-run."""
        tl = cls(base=obj["base"], fps=float(obj.get("fps", 30.0)))
        tl.tweens = [Tween.from_json(t) for t in obj.get("tweens", ())]
        tl.groups = {k: tuple(v) for k, v in (obj.get("groups") or {}).items()}
        if obj.get("duration") is not None:
            tl.duration = float(obj["duration"])
        return tl
