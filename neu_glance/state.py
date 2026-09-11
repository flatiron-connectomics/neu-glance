"""Assemble a viewer state, encode it into a URL, and read one back.

Neuroglancer keeps its entire state in the URL fragment, so a link *is* the state: which
volumes are loaded, where the view is, which segments are selected. Building one by hand means
getting the coordinate space and the source schemes right, which is exactly the sort of thing
that fails silently — a wrong ``dimensions`` block puts every layer in the wrong place and
still loads.

**The fragment is never sent to a server.** Everything after ``#!`` stays in the browser, so a
link carries no data anywhere — but it does mean the whole state travels in the URL, and a
large inline annotation layer makes for a long one (see :data:`LONG_URL`).

**Pure, like :mod:`neu_glance.layers`.** :func:`build_state` takes plain layer dicts and never asks
where a source lives. Keeping it that way is what will let a locally served volume
(``http://localhost:PORT/…``, no store to inspect) become a layer with no changes here.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Mapping, Sequence

from .layers import output_dimensions
from .shaders import as_hex_color

DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com/"

# Where URLs start being awkward to paste into things — mail clients wrap, some chat
# tools truncate, and a few proxies cap the request line even though a fragment is never
# sent. Not an error, just worth saying out loud.
LONG_URL = 8000

# Neuroglancer's own layout names. Passing anything through would be friendlier to new ones,
# but a typo produces a viewer that silently falls back, which is worse than being told.
LAYOUTS = ("4panel", "xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "3d")

# A nominal viewport edge in pixels, used only to turn "fit the volume" into a zoom
# number. The real panel size is not knowable here — it depends on the window and the
# layout — so this errs large, which errs zoomed OUT. Being a factor of two off is
# harmless; starting at the origin corner at one voxel per pixel is not.
NOMINAL_VIEWPORT_PX = 1000

# Leave a little space around the volume rather than cropping it to the panel edge.
FIT_MARGIN = 1.15


class StateProblem(RuntimeError):
    """A state or layer could not be read, or would not be well formed."""


def annotation_extent(layers: Sequence[dict]) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` covering the inline annotations in ``layers``.

    The fallback when every layer came from a file: a bounding-box layer knows where the
    data is even though no volume was named, and framing those boxes is a better opening
    view than the origin.
    """
    lo: list[float | None] = [None, None, None]
    hi: list[float | None] = [None, None, None]
    for layer in layers:
        for ann in layer.get("annotations", []) or []:
            pts = [ann[k] for k in ("pointA", "pointB", "point", "center") if k in ann]
            for p in pts:
                for a in range(3):
                    v = float(p[2 - a])                     # stored xyz, wanted zyx
                    lo[a] = v if lo[a] is None else min(lo[a], v)
                    hi[a] = v if hi[a] is None else max(hi[a], v)
    if any(v is None for v in lo):
        return None
    extent = tuple(max(1.0, hi[a] - lo[a]) for a in range(3))
    return extent, tuple(lo)


def default_view(extent_zyx: Sequence[float],
                 offset_zyx: Sequence[float] = (0, 0, 0)) -> tuple[list, float, float]:
    """``(centre_zyx, cross_section_scale, projection_scale)`` framing a whole volume.

    Neuroglancer with no ``position`` opens at the origin **corner** and with no
    ``crossSectionScale`` opens at one voxel per pixel, which on a 13750-voxel volume is
    a view of its empty edge. Both scales are in canonical voxels — per viewport pixel
    for the cross sections, across the viewport for the projection — so fitting is a
    division by a nominal panel size.
    """
    centre = [float(o) + float(e) / 2 for o, e in zip(offset_zyx, extent_zyx)]
    span = max(float(e) for e in extent_zyx) * FIT_MARGIN
    return centre, span / NOMINAL_VIEWPORT_PX, span


def build_state(layers: Sequence[dict], *, voxel_size_zyx=None, units: str | None = None,
                position_zyx: Sequence[float] | None = None,
                layout: str = "4panel",
                cross_section_scale: float | None = None,
                projection_scale: float | None = None,
                selected: str | None = None,
                show_slices: bool | None = None,
                frame: tuple | None = None) -> tuple[dict, str | None]:
    """A viewer state. ``position_zyx`` is zyx and is reversed here, like every coordinate.

    ``dimensions`` has to agree with the layers or nothing lines up, so it is derived
    from a volume's recorded voxel size rather than assumed.

    ``frame`` is an ``(extent_zyx, offset_zyx)`` pair used to fill in whichever of the
    position and the two zooms the caller did not specify — see :func:`default_view`.
    Without it the state carries none of the three and neuroglancer opens at the origin
    corner, fully zoomed in.

    ``show_slices=False`` sets neuroglancer's ``showSlices``, which hides the cross-section
    planes *inside the 3D panel* — worth it when the point of the link is meshes or
    skeletons, which the slices otherwise sit across. It does not touch the 2D panels; use
    ``layout="3d"`` for that. Left as ``None`` the key is omitted and the viewer's own
    default (shown) applies.
    """
    dims, warning = output_dimensions(voxel_size_zyx, units)
    state: dict[str, Any] = {"dimensions": dims, "layers": list(layers),
                             "layout": layout}

    fitted = default_view(*frame) if frame else (None, None, None)
    position = position_zyx if position_zyx is not None else fitted[0]
    cross = cross_section_scale if cross_section_scale is not None else fitted[1]
    projection = projection_scale if projection_scale is not None else fitted[2]

    if position is not None:
        state["position"] = [float(v) for v in tuple(position)[::-1]]
    if cross is not None:
        state["crossSectionScale"] = float(cross)
    if projection is not None:
        state["projectionScale"] = float(projection)
    if selected:
        state["selectedLayer"] = {"visible": True, "layer": selected}
    if show_slices is not None:
        # Written only when asked for, like the position and the two zooms above: a state
        # that carries no `showSlices` gets neuroglancer's own default (true), which is
        # what someone opening the link would otherwise expect.
        state["showSlices"] = bool(show_slices)
    return state, warning


def state_url(state: Mapping[str, Any], viewer: str = DEFAULT_VIEWER) -> str:
    """``<viewer>#!<url-encoded state>``.

    Percent-encoded rather than raw: neuroglancer accepts both, but a raw fragment
    survives only until something in the chain — a chat client, a wiki, a shell — decides
    what to do with the quotes and braces in it.
    """
    encoded = urllib.parse.quote(json.dumps(dict(state), separators=(",", ":")), safe="")
    return f"{viewer.rstrip('/')}/#!{encoded}"


def parse_url(url: str) -> dict:
    """The state back out of a neuroglancer URL, for inspecting or editing one."""
    if "#!" not in url:
        raise ValueError("not a neuroglancer state URL: no '#!' fragment")
    fragment = url.split("#!", 1)[1]
    return json.loads(urllib.parse.unquote(fragment))


def load_layer(path: str, read_bytes=None) -> list[dict]:
    """A layer (or a whole state's worth of layers) from a JSON file.

    Accepts what ``neu-glance bboxes`` and ``neu-glance annotate`` write either way round — a bare
    layer object, or a full state whose ``layers`` are taken — so it does not matter which of
    the two the caller happened to generate.
    """
    obj = _read_json(path, read_bytes)
    if isinstance(obj, dict) and "layers" in obj:
        return list(obj["layers"])
    if isinstance(obj, list):
        return list(obj)
    if not isinstance(obj, dict) or "type" not in obj:
        raise StateProblem(
            f"{path} is not a neuroglancer layer or state: expected an object with a "
            f"'type' (a layer) or a 'layers' array (a state)")
    return [obj]


def load_state(source: str, read_bytes=None) -> dict:
    """An existing state, from a neuroglancer URL **or** a JSON file.

    Accepting both is what makes ``--into`` usable directly on a link copied out of the
    browser, with no intermediate ``neu-glance parse`` step.
    """
    if "#!" in source:
        return parse_url(source)
    obj = _read_json(source, read_bytes)
    if not isinstance(obj, dict) or "layers" not in obj:
        raise StateProblem(
            f"{source} is not a neuroglancer state: expected an object with a 'layers' "
            f"array. A bare layer file goes to --layer, not --into.")
    return obj


def split_segment_layer(state: Mapping[str, Any], layer: str, *,
                        segments: Sequence[Any] | None = None,
                        names: Mapping[str, str] | None = None,
                        name_template: str = "{layer} {segment}",
                        object_alpha: float | None = None,
                        ) -> tuple[dict, dict[str, str], list[str]]:
    """One layer per segment, so each can be shown, coloured or FADED on its own.

    Returns ``(state, {segment_id: layer_name}, notes)``.

    **Neuroglancer has no per-segment alpha.** ``objectAlpha`` is a property of the layer —
    the mesh draw path is literally ``if (objectAlpha <= 0) return`` — so a scene where one
    neuron fades in while its neighbour holds still has to be a scene with one layer per
    neuron. Every copy names the same ``source``, and neuroglancer memoizes a datasource per
    URL, so the ``info`` is read once and each mesh is fetched once however many layers point
    at it.

    Each copy is the original layer **verbatim**, then overridden. That is what carries
    ``selectedAlpha``, ``meshSilhouetteRendering``, ``saturation``, ``colorSeed`` and whatever
    else the state happens to hold, with no whitelist here to go stale against a viewer
    release.

    **The colours come out identical, including hashed ones.** An explicit entry in
    ``segmentColors`` is carried across (restricted to the one segment, so the copy holds no
    dead entries). A segment with no entry is hash-coloured by neuroglancer from
    ``(colorSeed, segment_id)`` and nothing else, both of which the verbatim copy preserves —
    so it keeps exactly the colour it had. Getting this wrong would give a perfectly
    attractive scene in entirely the wrong colours, with nothing anywhere to say so.

    Three details, each of which fails quietly:

    - **A ``!``-prefixed id is starred but NOT visible.** Those do not become layers; they
      stay on the original, which keeps the stars a user set.
    - **The original layer is kept**, emptied and hidden, rather than removed. It is named by
      ``selectedLayer``, by a synapse layer's ``linkedSegmentationLayer``, and possibly by a
      ``toolPalettes`` entry; neuroglancer resolves a dangling layer name to nothing without
      complaining. (The flip side, worth knowing: a synapse layer filtering on that layer now
      filters on an empty segment set, so re-point it at one of the copies if you turn it on.)
    - **Each copy replaces the original IN PLACE** in the ``layers`` array rather than being
      appended. Neuroglancer draws in layer order, so moving twenty-five layers to the end
      changes what is in front of what.

    Segment ids stay **strings** throughout, the same reason :func:`neu_glance.volume_layer`
    keeps them: a 19-digit uint64 through a JSON number comes back rounded.
    """
    out = json.loads(json.dumps(dict(state)))            # deep copy; never mutate the caller's
    existing = list(out.get("layers") or [])
    index = next((i for i, lyr in enumerate(existing)
                  if isinstance(lyr, dict) and lyr.get("name") == layer), None)
    if index is None:
        known = ", ".join(repr(lyr.get("name")) for lyr in existing if isinstance(lyr, dict))
        raise StateProblem(f"no layer named {layer!r} in this state. It has: {known}")

    source = existing[index]
    if source.get("type") != "segmentation":
        raise StateProblem(
            f"layer {layer!r} is a {source.get('type')!r} layer; only a segmentation layer "
            f"has segments to split")

    raw = [str(s) for s in (source.get("segments") or [])]
    starred = [s for s in raw if s.startswith("!")]
    visible = [s for s in raw if not s.startswith("!")]
    if segments is not None:
        wanted = [str(s) for s in segments]
        missing = [s for s in wanted if s not in visible]
        if missing:
            raise StateProblem(
                f"layer {layer!r} has no visible segment(s) {', '.join(missing)}. Its visible "
                f"segments are: {', '.join(visible) or '(none)'}"
                + (f". {', '.join(starred)} are starred but hidden ('!'), so they cannot be "
                   f"split" if starred else ""))
        visible = wanted

    notes: list[str] = []
    if not visible:
        notes.append(f"layer {layer!r} has no visible segments, so nothing was split")

    colors = {str(k): v for k, v in (source.get("segmentColors") or {}).items()}
    taken = {lyr.get("name") for lyr in existing if isinstance(lyr, dict)}
    created: dict[str, str] = {}
    copies: list[dict] = []

    for i, sid in enumerate(visible):
        copy = json.loads(json.dumps(source))
        copy["segments"] = [sid]
        if sid in colors:
            copy["segmentColors"] = {sid: colors[sid]}
        else:
            copy.pop("segmentColors", None)
        if object_alpha is not None:
            copy["objectAlpha"] = float(object_alpha)
        copy.pop("visible", None)

        name = (names or {}).get(sid) or name_template.format(
            layer=layer, segment=sid, index=i)
        if name in taken:
            suffix = 2
            while f"{name}-{suffix}" in taken:
                suffix += 1
            notes.append(f"renamed split layer {name!r} to {name}-{suffix}: the state already "
                         f"has a layer by that name")
            name = f"{name}-{suffix}"
        taken.add(name)
        copy["name"] = name
        created[sid] = name
        copies.append(copy)

    emptied = json.loads(json.dumps(source))
    emptied["segments"] = starred
    emptied.pop("segmentColors", None)
    emptied["visible"] = False
    out["layers"] = existing[:index] + copies + [emptied] + existing[index + 1:]
    return out, created, notes


def subset_layers(state: Mapping[str, Any], layer: str,
                  subsets: Mapping[str, Sequence[Any]], *,
                  colors: Mapping[str, str] | None = None,
                  object_alpha: float | None = None,
                  name_template: str = "{subset}",
                  ) -> tuple[dict, dict[str, str], list[str]]:
    """One layer per NAMED SET of segments. Returns ``(state, {set: layer_name}, notes)``.

    The many-bodies sibling of :func:`split_segment_layer`. That one gives every segment its
    own layer, which is what an animation needs to fade objects **individually**; this gives
    every *set* one, which is what it needs to fade whole populations — and a set of four
    hundred Kenyon cells has no business being four hundred layers.

    **Colour is per layer, via ``segmentDefaultColor``, not a `segmentColors` map.** A map
    would need one identical entry per body — four hundred copies of the same string in the
    state — where the default is a single field meaning exactly "everything in this layer".

    That makes dropping the inherited ``segmentColors`` load-bearing rather than tidiness: an
    entry for any body in the set **overrides** the default, so a scene where a handful of
    neurons had been coloured by hand would render that handful in their old colours and the
    rest in the new one. A set that is *almost* uniformly coloured looks like a data problem,
    not a leftover.

    The rules that :func:`split_segment_layer` documents apply here too: the original layer is
    kept, emptied and hidden, because a dangling layer name resolves to nothing in silence;
    the copies replace it **in place**, since neuroglancer draws in layer order; and ids stay
    strings, because a 19-digit uint64 through a JSON number comes back rounded.
    """
    out = json.loads(json.dumps(dict(state)))            # deep copy; never mutate the caller's
    existing = list(out.get("layers") or [])
    index = next((i for i, lyr in enumerate(existing)
                  if isinstance(lyr, dict) and lyr.get("name") == layer), None)
    if index is None:
        known = ", ".join(repr(lyr.get("name")) for lyr in existing if isinstance(lyr, dict))
        raise StateProblem(f"no layer named {layer!r} in this state. It has: {known}")

    source = existing[index]
    if source.get("type") != "segmentation":
        raise StateProblem(
            f"layer {layer!r} is a {source.get('type')!r} layer; only a segmentation layer "
            f"has segments to group")

    notes: list[str] = []
    taken = {lyr.get("name") for lyr in existing if isinstance(lyr, dict)}
    created: dict[str, str] = {}
    copies: list[dict] = []

    for i, (subset, ids) in enumerate(subsets.items()):
        members = [str(s).lstrip("!") for s in ids]
        if not members:
            notes.append(f"set {subset!r} is empty, so no layer was made for it")
            continue
        copy = json.loads(json.dumps(source))
        copy["segments"] = members
        copy.pop("segmentColors", None)
        copy.pop("visible", None)
        if colors and subset in colors:
            # Through `as_hex_color`, not `str`. `segmentDefaultColor` must be `#rrggbb`, and
            # anything else is IGNORED rather than refused — a matplotlib-style `(0.12, 0.47,
            # 0.71)` stringifies to something neuroglancer cannot parse, so every body falls
            # back to its hash colour and the scene comes out in entirely the wrong palette
            # with nothing anywhere to say so. This also makes names and 3-sequences work,
            # which is what anyone reaching for a palette already has.
            copy["segmentDefaultColor"] = as_hex_color(colors[subset], f"colour for {subset!r}")
        if object_alpha is not None:
            copy["objectAlpha"] = float(object_alpha)

        name = name_template.format(subset=subset, layer=layer, index=i)
        if name in taken:
            suffix = 2
            while f"{name}-{suffix}" in taken:
                suffix += 1
            notes.append(f"renamed set layer {name!r} to {name}-{suffix}: the state already "
                         f"has a layer by that name")
            name = f"{name}-{suffix}"
        taken.add(name)
        copy["name"] = name
        created[subset] = name
        copies.append(copy)

    emptied = json.loads(json.dumps(source))
    emptied["segments"] = []
    emptied.pop("segmentColors", None)
    emptied["visible"] = False
    out["layers"] = existing[:index] + copies + [emptied] + existing[index + 1:]
    return out, created, notes


def _read_json(path: str, read_bytes=None):
    if read_bytes is None:
        from neu_vol.location import read_bytes as read_bytes_impl
        read_bytes = read_bytes_impl
    raw = read_bytes(path)
    if raw is None:
        raise StateProblem(f"no such file: {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise StateProblem(f"{path} is not valid JSON: {e}") from None


def merge_into(base: Mapping[str, Any], layers: Sequence[dict], *,
               layout: str | None = None,
               position_zyx: Sequence[float] | None = None,
               cross_section_scale: float | None = None,
               projection_scale: float | None = None,
               selected: str | None = None,
               show_slices: bool | None = None) -> tuple[dict, list[str]]:
    """``base`` with ``layers`` appended. Returns the state and any notes worth printing.

    Two rules, both of which fail quietly if broken:

    - **The incoming frame and view are preserved.** ``dimensions``, ``position`` and the two
      zooms are left exactly as they were unless explicitly overridden here. "Add a layer to
      my view" must not move the view, and it must not re-derive ``dimensions`` — a state
      whose dimensions disagree with its layers loads fine and puts everything in the wrong
      place.
    - **Layer names must stay unique.** Neuroglancer keys a layer by name, so two layers
      sharing one is not a duplicate but a collision. A clashing incoming layer is renamed
      with a ``-2`` suffix and the rename is reported, rather than silently shadowing the
      layer already there.
    """
    state = json.loads(json.dumps(dict(base)))      # deep copy; never mutate the caller's
    existing = list(state.get("layers") or [])
    taken = {lyr.get("name") for lyr in existing if isinstance(lyr, dict)}
    notes: list[str] = []

    added = []
    for layer in layers:
        layer = dict(layer)
        name = layer.get("name")
        if name in taken:
            suffix = 2
            while f"{name}-{suffix}" in taken:
                suffix += 1
            layer["name"] = f"{name}-{suffix}"
            notes.append(f"renamed incoming layer {name!r} to {layer['name']!r}: the state "
                         f"already has a layer by that name")
        taken.add(layer["name"])
        added.append(layer)

    state["layers"] = existing + added
    if not state.get("dimensions"):
        notes.append("the state being added to declares no `dimensions`, so the layers' own "
                     "coordinates may not line up")

    if layout is not None:
        state["layout"] = layout
    if position_zyx is not None:
        state["position"] = [float(v) for v in tuple(position_zyx)[::-1]]
    if cross_section_scale is not None:
        state["crossSectionScale"] = float(cross_section_scale)
    if projection_scale is not None:
        state["projectionScale"] = float(projection_scale)
    if selected:
        state["selectedLayer"] = {"visible": True, "layer": selected}
    if show_slices is not None:
        state["showSlices"] = bool(show_slices)
    return state, notes
