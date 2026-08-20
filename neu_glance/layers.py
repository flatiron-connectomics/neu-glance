"""Local annotation layers: authored from coordinates, or from a volume's occupancy boxes.

**Pure.** Everything here takes plain data and returns plain data — no store access, no
network, no volume opening. That is what lets the whole layer format be tested without a
fixture, and it is why the occupancy *analysis* stayed in neu-vol
(``ops.annotate.labeled_regions``) while only the JSON emission moved here: the analysis
needs a store, this does not.

**The annotations are local, not a precomputed annotation source, and that is the point.**
Neuroglancer builds its annotation list by iterating the layer's source, and
``MultiscaleAnnotationSource`` — the class behind every *precomputed* annotation source —
defines ``[Symbol.iterator]`` as an empty generator. A precomputed annotation layer therefore
renders in the viewport but contributes no rows to the Annotations tab: no list to click
through, and ``[`` / ``]`` do not step. Local annotations, carried inline in the state, list
and navigate. For a set too large to carry inline — synapses over a whole volume — the
precomputed format is the answer and :mod:`neu_glance.sources` addresses it; the two are
complementary rather than alternatives.

Coordinates are **zyx in memory** throughout, converted to xyz in exactly one place —
:func:`build_annotation` — because that is the conversion no test notices: a mirrored
annotation is a perfectly valid annotation somewhere else in the volume.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

# Neuroglancer's annotation type strings, used to keep each annotation on one line
# when rendering. From `AnnotationType` in src/annotation/index.ts.
ANNOTATION_TYPES = ("point", "line", "axis_aligned_bounding_box", "ellipsoid",
                    "polyline")

# OME-NGFF spells units out; precomputed always means nm. Anything not here leaves the
# layer unitless, which the caller is warned about — a unitless annotation layer does
# not align with a layer that has physical units.
_UNIT_METRES = {
    "nm": 1e-9, "nanometer": 1e-9, "nanometre": 1e-9,
    "um": 1e-6, "µm": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6,
    "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
    "m": 1.0, "meter": 1.0, "metre": 1.0,
}

#: The annotation kinds authored here: neuroglancer's own ``type`` string, and the
#: geometry fields each one carries. Both come from ``annotationTypeHandlers`` /
#: ``annotationToJson`` in neuroglancer's ``src/annotation/index.ts``, which is the
#: authority — a bbox is stored as two *corners*, not an origin and a size.
#:
#: ``polyline`` is deliberately absent. It takes an arbitrary number of points per
#: annotation, which one flat CSV row cannot carry, and inventing a grouping column for
#: it would be a format of our own rather than a way of writing theirs.
KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "point": ("point", ("point",)),
    "box": ("axis_aligned_bounding_box", ("pointA", "pointB")),
    "line": ("line", ("pointA", "pointB")),
    "ellipsoid": ("ellipsoid", ("center", "radii")),
}

#: The zyx column names each kind needs, one tuple per geometry field above.
CSV_COLUMNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "point": (("z", "y", "x"),),
    "box": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "line": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "ellipsoid": (("z", "y", "x"), ("rz", "ry", "rx")),
}

#: Columns any kind may carry. ``segments`` is what links an annotation to bodies, so
#: selecting one in the viewer selects them; neuroglancer writes those ids as *strings*.
OPTIONAL_COLUMNS = ("id", "description", "segments")

#: How many of a kind's leading coordinate groups are *positions*. An ellipsoid's second
#: group is radii — an extent, not a place — so a bounds check must not treat it as one.
POSITION_GROUPS = {"point": 1, "box": 2, "line": 2, "ellipsoid": 1}


def output_dimensions(voxel_size_zyx, units: str | None) -> tuple[dict, str | None]:
    """``{"x": [scale, "m"], ...}`` for the layer, and a warning if units are unknown.

    Declared on the layer itself so its coordinates are interpreted in its own frame
    rather than whatever the viewer happens to be displaying in — which is what lets
    the layer be pasted into any state of the same volume.
    """
    if voxel_size_zyx is None:
        return ({d: [1, ""] for d in "xyz"},
                "no voxel size recorded: the layer is unitless, so it will only line "
                "up if the viewer's own dimensions are voxels. Pass --voxel-size.")
    metres = _UNIT_METRES.get(str(units or "nm").lower())
    if metres is None:
        return ({d: [1, ""] for d in "xyz"},
                f"unrecognised unit {units!r}: the layer is unitless and may not line "
                f"up. Pass --voxel-size to state the size in nm.")
    xyz = tuple(voxel_size_zyx)[::-1]
    return {d: [float(v) * metres, "m"] for d, v in zip("xyz", xyz)}, None


def positions(record: dict):
    """The position tuples of a record, skipping any that are extents (zyx)."""
    return record["coords"][:POSITION_GROUPS[record["kind"]]]


def read_annotation_csv(text: str, kind: str, *, source: str = "<csv>") -> list[dict]:
    """Rows of a CSV as annotation records: ``{"kind", "coords", id?, ...}``.

    ``coords`` is one zyx tuple per geometry field of ``kind`` — so a point has one and
    a box has two — in whatever units the file is written in; converting them is
    :func:`rescale`'s job, not this one's.

    Columns are addressed **by name**, never by position: a synapse table has its own
    column order and often extra columns, and silently reading the wrong three numbers
    is the failure this avoids. Unknown columns are ignored.
    """
    import csv
    import io

    if kind not in CSV_COLUMNS:
        raise ValueError(f"unknown annotation kind {kind!r}; known: {sorted(KINDS)}")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{source}: no data rows (a header naming the columns is "
                         f"required)")
    needed = [c for group in CSV_COLUMNS[kind] for c in group]
    present = {(k or "").strip() for k in rows[0]}
    missing = [c for c in needed if c not in present]
    if missing:
        raise ValueError(
            f"{source}: {kind} needs column(s) {', '.join(missing)} — expected "
            f"{', '.join(needed)}{' plus any of ' + ', '.join(OPTIONAL_COLUMNS)}. "
            f"Found: {', '.join(sorted(present)) or '(no header)'}")

    records = []
    for n, row in enumerate(rows, start=2):        # row 1 is the header
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        try:
            coords = tuple(tuple(float(clean[c]) for c in group)
                           for group in CSV_COLUMNS[kind])
        except ValueError as e:
            raise ValueError(f"{source} line {n}: {e}") from None
        rec: dict[str, Any] = {"kind": kind, "coords": coords}
        if clean.get("id"):
            rec["id"] = clean["id"]
        if clean.get("description"):
            rec["description"] = clean["description"]
        if clean.get("segments"):
            rec["segments"] = _parse_segments(clean["segments"], f"{source} line {n}")
        records.append(rec)
    return records


def _parse_segments(value: str, where: str) -> list[str]:
    """Segment ids, however they were separated. Kept as strings, as the state wants."""
    ids = [p for p in value.replace("|", " ").replace(",", " ").split() if p]
    for i in ids:
        # A float here means a spreadsheet turned a 19-digit body id into 1.23e+18, and
        # the annotation would then link to a body that does not exist.
        if not i.isdigit():
            raise ValueError(f"{where}: segment id {i!r} is not a whole number")
    return ids


def rescale(records: list[dict], factor_zyx: Sequence[float]) -> list[dict]:
    """Multiply every coordinate of every record by ``factor_zyx`` (per axis).

    One operation covers both conversions the CLI offers, because both are per-axis
    scalings: voxels at scale N to level-0 voxels (the real per-level ratio), and
    physical nm to level-0 voxels (the reciprocal of the voxel size). Radii scale
    exactly as positions do.
    """
    out = []
    for r in records:
        out.append({**r, "coords": tuple(tuple(c * f for c, f in zip(group, factor_zyx))
                                         for group in r["coords"])})
    return out


def build_annotation(record: dict, ident: str) -> dict:
    """One annotation object, zyx in and **xyz out** — the only place that flips.

    A ``box`` gets its corners sorted per axis: neuroglancer stores two corners with no
    requirement about order, and a reversed pair renders as nothing at all. A ``line``
    is left alone, because for a line the order *is* the direction.
    """
    kind = record["kind"]
    type_name, fields = KINDS[kind]
    coords = list(record["coords"])
    if kind == "box":
        lo, hi = coords
        coords = [tuple(min(a, b) for a, b in zip(lo, hi)),
                  tuple(max(a, b) for a, b in zip(lo, hi))]
    ann: dict[str, Any] = {"type": type_name, "id": ident}
    if record.get("description"):
        ann["description"] = record["description"]
    for field, group in zip(fields, coords):
        ann[field] = [float(v) for v in tuple(group)[::-1]]          # zyx -> xyz
    if record.get("segments"):
        # Related segments: an array per relationship, and a local layer has exactly
        # one. Ids are strings — a uint64 body id does not survive a JSON number.
        ann["segments"] = [[str(s) for s in record["segments"]]]
    return ann


def local_layer(annotations: list[dict], dims: dict, *, name: str = "annotations",
                color: str = "#ffee00") -> dict:
    """The layer envelope for inline (``local://annotations``) annotations."""
    return {
        "type": "annotation",
        "name": name,
        # opens the layer panel straight onto the clickable list
        "tab": "annotations",
        "source": {"url": "local://annotations",
                   "transform": {"outputDimensions": dims}},
        "annotationColor": color,
        "annotations": annotations,
    }


def boxes_layer(regions: list[dict], dims: dict, *, name: str = "regions",
                color: str = "#ffee00", kind: str = "box",
                label: str = "r") -> dict:
    """The occupancy layer: one annotation per region of ``labeled_regions``.

    ``regions`` are zyx ``{"lo", "hi"}``; the flip to xyz happens in
    :func:`build_annotation`, as it does for every annotation this module writes.

    Named ``boxes_layer`` rather than ``annotation_layer`` because
    :func:`neu_glance.sources.annotation_layer` builds a layer for a *precomputed annotation
    source* — a different thing that took the same name in the module this came from.
    """
    annotations = []
    for i, r in enumerate(regions):
        lo, hi = tuple(r["lo"]), tuple(r["hi"])
        ident = f"{label}{i:02d}"
        extent = "x".join(str(hi[a] - lo[a]) for a in range(3))
        note = f"{ident}  {extent} vox"
        if r.get("n_labels") is not None:
            note += f"  {r['n_labels']} labels"
        record = ({"kind": "point",
                   "coords": (tuple((lo[a] + hi[a]) / 2 for a in range(3)),)}
                  if kind == "point" else {"kind": "box", "coords": (lo, hi)})
        annotations.append(build_annotation({**record, "description": note}, ident))
    return local_layer(annotations, dims, name=name, color=color)


def render(obj: Any) -> str:
    """``json.dumps`` with each annotation kept on a single line.

    This output is meant to be *pasted* into neuroglancer's JSON editor, so it is read by a
    person: one line per annotation is the difference between twelve rows and two hundred.
    """
    holes: dict[str, str] = {}

    def fold(o):
        if isinstance(o, dict):
            if o.get("type") in ANNOTATION_TYPES:
                # Plain ASCII on purpose. json.dumps escapes control characters, so a
                # NUL sentinel is written out as a six-character escape sequence, and
                # the substitution below then misses every one of them silently.
                key = f"__neu_glance_annotation_{len(holes)}__"
                holes[key] = json.dumps(o, separators=(", ", ": "))
                return key
            return {k: fold(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fold(v) for v in o]
        return o

    text = json.dumps(fold(obj), indent=1)
    for key, line in holes.items():
        text = text.replace(f'"{key}"', line)
    return text
