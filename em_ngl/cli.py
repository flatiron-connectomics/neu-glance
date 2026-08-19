"""``em-ngl`` — the command line over em_ngl.

Argparse, matching its siblings (`em-vol`, `em-morpho`, `em-annot`) rather than introducing a
second CLI framework into the family. Heavy imports stay inside the subcommand that needs them
so ``--help`` is fast.

Five subcommands, and the division between them is *what they produce*:

- ``gen`` composes a **state** from volumes, annotation sources and layer files
- ``annotate`` makes a **layer** from coordinates or a CSV
- ``bboxes`` makes a **layer** from a volume's occupancy
- ``parse`` reads a URL back into its state JSON
- ``shaders`` lists or prints the built-in annotation shaders

The three producers share one output stage: ``--format {layer,state,url}`` decides the
serialization and ``--into`` merges into an existing state instead of building a new one. That
is why they are separate subcommands rather than one — the *inputs* differ completely, while
the output is uniform.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__

_ANN_FLAGS = (("points", "point"), ("boxes", "box"), ("lines", "line"),
              ("ellipsoids", "ellipsoid"))

#: The same column spec as ``layers.CSV_COLUMNS``, repeated here because the parser needs it at
#: build time and importing the module at parser-build time would pull em-volume-tools into
#: every ``em-ngl --help``. A test asserts the two agree, which is what keeps the duplication
#: honest.
_ANN_CSV_COLUMNS = {
    "point": (("z", "y", "x"),),
    "box": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "line": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "ellipsoid": (("z", "y", "x"), ("rz", "ry", "rx")),
}

_LAYOUTS = ("4panel", "xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "3d")
_DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com/"
_SHADER_NAMES = ("synapse",)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _ftriple(value, name):
    """'z,y,x' -> a 3-tuple of floats, or None. Voxel sizes are not always integral."""
    if value is None:
        return None
    parts = tuple(float(v) for v in value.replace("x", ",").split(","))
    if len(parts) != 3:
        raise SystemExit(f"--{name} needs 3 comma-separated values, got {value!r}")
    return parts


def _write_text(path: str, text: str) -> None:
    """Write to a local path or an object store, uniformly.

    Through ``location`` rather than ``open()`` so ``--out s3://…`` works: the file driver
    creates parent directories and the s3 driver bootstraps credentials.
    """
    from em_volume_tools.location import write_bytes

    write_bytes(path, text.encode())


def _read_ann_file(path: str) -> str:
    """CSV text from a path, an object-store URL, or ``-`` for stdin."""
    if path == "-":
        return sys.stdin.read()
    from em_volume_tools.location import read_bytes

    data = read_bytes(path)
    if data is None:
        raise SystemExit(f"could not read {path}")
    return data.decode()


def _inline_records(args) -> list[dict]:
    """Records from the repeatable inline flags, in the order they were given."""
    records = []
    for _plural, kind in _ANN_FLAGS:
        for value in getattr(args, kind) or []:
            n = 3 * len(_ANN_CSV_COLUMNS[kind])
            parts = [p for p in value.replace(" ", "").split(",") if p]
            if len(parts) != n:
                raise SystemExit(
                    f"--{kind} needs {n} comma-separated numbers "
                    f"({', '.join(c for g in _ANN_CSV_COLUMNS[kind] for c in g)}), "
                    f"got {value!r}")
            try:
                flat = [float(p) for p in parts]
            except ValueError:
                raise SystemExit(f"--{kind} {value!r}: not all numbers") from None
            records.append({"kind": kind,
                            "coords": tuple(tuple(flat[i:i + 3])
                                            for i in range(0, n, 3))})
    return records


def _volume_frame(volume: str | None, voxel_size: str | None):
    """``(voxel_size_zyx, units, level-0 spatial shape or None)`` for a layer's frame.

    The frame is what makes the coordinates mean anything: an annotation layer declares its own
    ``outputDimensions``, and one that disagrees with the volume puts every annotation in the
    wrong place while still loading.

    Uses ``read_source_metadata`` plus ``volume_extent``, never ``describe`` — the latter opens
    every level to probe for a foreign marker, which a frame does not need.
    """
    voxel = _ftriple(voxel_size, "voxel-size")
    if not volume:
        return voxel, ("nm" if voxel else None), None

    from em_volume_tools.source_metadata import detect_backend, read_source_metadata

    from .sources import volume_extent

    fmt = detect_backend(volume.rstrip("/"))
    if fmt is None:
        raise SystemExit(f"no volume found at {volume}")
    meta = read_source_metadata({"backend": fmt, "path": volume.rstrip("/")}) or {}
    extent = volume_extent(volume.rstrip("/"), fmt)
    return (voxel or (meta.get("voxel_size") and tuple(meta["voxel_size"])),
            "nm" if voxel else meta.get("units"),
            tuple(extent[0]) if extent else None)


def _level0_factor(volume: str, scale: int) -> tuple[int, ...]:
    """How many level-0 voxels one scale-``scale`` voxel spans, per axis.

    Read from each level's OWN recorded voxel size, never ``2**scale``: real pyramids are
    anisotropic and ``(1, 2, 2)`` — halve x/y, leave z — is common.
    """
    from em_volume_tools.source_metadata import (detect_backend, location_spec,
                                                 read_level_voxel_sizes)

    fmt = detect_backend(volume.rstrip("/"))
    if fmt is None:
        raise SystemExit(f"--scale {scale} needs the volume's per-level voxel sizes and "
                         f"nothing at {volume} looks like a volume")
    per_level = read_level_voxel_sizes(location_spec(volume.rstrip("/"), fmt))
    if not per_level:
        raise SystemExit(f"--scale {scale} needs the volume's per-level voxel sizes, and "
                         f"{volume} records none.")
    if scale >= len(per_level):
        raise SystemExit(f"--scale {scale}: the volume records only {len(per_level)} "
                         f"level(s) (0-{len(per_level) - 1})")
    factor = tuple(s / b for s, b in zip(per_level[scale], per_level[0]))
    if any(abs(f - round(f)) > 1e-6 for f in factor):
        raise SystemExit(f"--scale {scale}: its voxel size {per_level[scale]} is not an "
                         f"integer multiple of level 0's {per_level[0]}, so a coordinate "
                         f"there does not land on level-0 voxels.")
    return tuple(int(round(f)) for f in factor)


# --------------------------------------------------------------------------- #
# the shared output stage
# --------------------------------------------------------------------------- #
def _add_output_flags(q: argparse.ArgumentParser, *, formats: tuple[str, ...],
                      default: str) -> None:
    """``--format``, ``--out`` and the state-shaping flags every producer shares."""
    q.add_argument("--format", dest="fmt", choices=formats, default=None,
                   help=f"what to emit (default: {default}). 'layer' is a bare layer "
                        f"object for pasting into a state's `layers` array; 'state' is a "
                        f"whole state as JSON, for neuroglancer's {{}} editor; 'url' is a "
                        f"link carrying that state.")
    q.add_argument("--out", default=None, metavar="PATH_OR_URL",
                   help="write it here instead of stdout (local or s3://…)")
    q.add_argument("--into", default=None, metavar="URL_OR_JSON",
                   help="add the layers to an EXISTING state rather than building a new "
                        "one — a neuroglancer URL or a state JSON file. The state's own "
                        "dimensions, position and zoom are kept, so your view does not "
                        "move; a layer whose name is already taken is renamed and reported. "
                        "Implies --format url unless --format says otherwise.")
    if "state" in formats or "url" in formats:
        q.add_argument("--layout", default=None, choices=_LAYOUTS,
                       help="neuroglancer panel layout (default: 4panel, or whatever "
                            "--into already had)")
        q.add_argument("--position", default=None, metavar="Z,Y,X",
                       help="where to put the crosshair, in level-0 voxels")
        q.add_argument("--position-order", choices=("zyx", "xyz"), default="zyx",
                       help="axis order of --position. Everything in this package is zyx, "
                            "but neuroglancer DISPLAYS xyz — so pass xyz to use numbers "
                            "copied straight out of the viewer (default: zyx)")
        q.add_argument("--cross-section-scale", type=float, default=None, metavar="S",
                       help="zoom of the 2D panels: nm per screen pixel, smaller is closer")
        q.add_argument("--projection-scale", type=float, default=None, metavar="S",
                       help="zoom of the 3D panel")
        q.add_argument("--hide-slices", action="store_true",
                       help="set showSlices false, hiding the cross-section planes inside "
                            "the 3D panel — the usual thing to want when the link is about "
                            "meshes or skeletons. Omitted from the state unless passed")
        q.add_argument("--select", default=None, metavar="LAYER_NAME",
                       help="open the side panel on this layer")
        q.add_argument("--select-last", action="store_true",
                       help="the same, on whichever layer was added last")
        q.add_argument("--viewer", default=_DEFAULT_VIEWER,
                       help=f"viewer base URL for --format url (default: "
                            f"{_DEFAULT_VIEWER})")
    q.set_defaults(default_format=default)


def _resolve_format(args) -> str:
    """The format actually being emitted.

    ``--format`` defaults to ``None`` so an explicit choice is distinguishable from the
    command's own default, which is what lets ``--into`` quietly promote ``layer`` to ``url``
    while still refusing an explicit ``--format layer``. Resolved in ONE place because
    :func:`_with_volume` needs the same answer as :func:`_emit` — reading the raw ``args.fmt``
    in both is how the volume layer got prepended to a bare-layer output.
    """
    into = getattr(args, "into", None)
    if args.fmt is None:
        return "url" if into else args.default_format
    if into and args.fmt == "layer":
        raise SystemExit(
            "--into merges layers into a state, so --format layer has nothing to emit. "
            "Use --format url (what --into gives by default) or state.")
    return args.fmt


def _emit(args, layers: list[dict], *, voxel=None, units=None, frame=None,
          selected: str | None = None) -> int:
    """Serialize ``layers`` per ``--format``/``--into``/``--out``. Returns an exit code.

    One implementation for all three producers, which is what keeps ``em-ngl bboxes --format
    url`` and ``em-ngl gen`` from drifting into two different notions of a state.
    """
    from .layers import render
    from .state import LONG_URL, StateProblem, load_state, merge_into, state_url

    err = sys.stderr
    into = getattr(args, "into", None)
    fmt = _resolve_format(args)

    if fmt == "layer":
        if len(layers) != 1:
            raise SystemExit(
                f"--format layer emits ONE layer object and there are {len(layers)}. Use "
                f"--format state to get all of them in a state.")
        text = render(layers[0])
    else:
        selected_name = getattr(args, "select", None) or (
            layers[-1]["name"] if getattr(args, "select_last", False) and layers
            else selected)
        position = _ftriple(getattr(args, "position", None), "position")
        if position is not None and getattr(args, "position_order", "zyx") == "xyz":
            position = tuple(reversed(position))
        shaping = dict(
            layout=getattr(args, "layout", None),
            position_zyx=position,
            cross_section_scale=getattr(args, "cross_section_scale", None),
            projection_scale=getattr(args, "projection_scale", None),
            selected=selected_name,
            show_slices=False if getattr(args, "hide_slices", False) else None,
        )
        if into:
            try:
                base = load_state(into)
            except (StateProblem, ValueError) as e:
                raise SystemExit(str(e)) from None
            state, notes = merge_into(base, layers, **shaping)
            print(f"added {len(layers)} layer(s) to {len(base.get('layers') or [])} "
                  f"already in {into if '#!' not in into else 'the given URL'}", file=err)
            print(f"  kept the existing dimensions, position and zoom", file=err)
            for note in notes:
                print(f"  {note}", file=err)
        else:
            # Where the view will open. Worth saying because all three of these are silent
            # in the viewer: with no frame at all neuroglancer opens at the origin CORNER
            # fully zoomed in, which looks like an empty or broken layer.
            if position is not None:
                print(f"  position {tuple(position)} zyx "
                      f"(given as {getattr(args, 'position_order', 'zyx')})", file=err)
            elif frame:
                print(f"  view centred on the {tuple(int(v) for v in frame[0])} zyx frame, "
                      f"zoomed to fit it", file=err)
            else:
                print(f"  no volume or annotation established a frame — neuroglancer will "
                      f"open at the origin, zoomed in. Pass --position to place the view.",
                      file=err)
            from .state import build_state

            state, warning = build_state(
                layers, voxel_size_zyx=voxel, units=units, frame=frame,
                **{k: v for k, v in shaping.items() if k != "layout"},
                layout=shaping["layout"] or "4panel")
            if warning:
                print(f"  WARNING: {warning}", file=err)
        text = (render(state) if fmt == "state"
                else state_url(state, getattr(args, "viewer", _DEFAULT_VIEWER)))
        if fmt == "url" and len(text) > LONG_URL:
            print(f"  note: the URL is {len(text)} characters — long enough that some "
                  f"clients will wrap or truncate it. --format state avoids that.",
                  file=err)

    if args.out:
        _write_text(args.out, text + "\n")
        print(f"wrote {args.out}", file=err)
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="em-ngl",
        description="Neuroglancer states, layers and links.\n\n"
                    "Everything a viewer consumes and nothing that produces data: the "
                    "volumes come from em-volume-tools and the annotation sources from "
                    "em-annotation, and neither of those knows a viewer exists.\n\n"
                    "Neuroglancer keeps its whole state in the URL fragment, so a link IS "
                    "the state. Everything after '#!' stays in the browser and is never "
                    "sent to a server — but the whole state travels in the URL, so a large "
                    "inline annotation layer makes for a long one.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"em-ngl {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # --- gen ----------------------------------------------------------------
    q = sub.add_parser(
        "gen", help="a viewer state from volumes, annotation sources and layer files",
        description="Compose a neuroglancer state and emit it as a link or as JSON.\n\n"
                    "Reads the volumes to get the source scheme and the coordinate space "
                    "right, which is the part that fails silently by hand: a `dimensions` "
                    "block that disagrees with the data loads fine and puts every layer in "
                    "the wrong place.\n\n"
                    "  em-ngl bboxes s3://.../gt_v2 --label gt --out gt.json\n"
                    "  em-ngl gen --image s3://.../em --seg s3://.../gt_v2 \\\n"
                    "      --layer gt.json --segments 1,2,3 --layout xy-3d\n\n"
                    "URL or JSON to stdout, summary to stderr. Reads only.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--image", action="append", metavar="VOLUME",
                   help="an image volume (repeatable). Drawn beneath the segmentations")
    q.add_argument("--seg", action="append", metavar="VOLUME",
                   help="a segmentation volume (repeatable)")
    q.add_argument("--segments", action="append", metavar="IDS",
                   help="comma-separated segment ids to select, applied to the --seg "
                        "volumes in order. Repeat for a second segmentation")
    q.add_argument("--annotations", action="append", metavar="SOURCE",
                   help="a precomputed ANNOTATION source — what `em-annot "
                        "annotation-source` writes (repeatable). Added as its own layer, "
                        "because unlike mesh and skeletons an annotation source is never "
                        "named in a volume's info. Its relationships are bound to the first "
                        "--seg layer, which is what makes 'only this body's synapses' work")
    q.add_argument("--annotation-split", action="store_true",
                   help="add each annotation source as TWO layers on the one source: "
                        "'-pre' filtered on the presynaptic relationship (the selected "
                        "body's OUTPUTS) and '-post' on the postsynaptic one (its INPUTS), "
                        "each showing only its own endpoint marker. One layer filtered on "
                        "both conflates the two directions and the markers overlap")
    q.add_argument("--annotation-shader", default=None, metavar="NAME_OR_PATH",
                   help="GLSL for the annotation layers: a built-in name ("
                        + ", ".join(_SHADER_NAMES) + "), a file, or 'none'. Default picks a "
                        "built-in whose properties the source declares, since a shader "
                        "naming an absent property fails to compile and the layer then "
                        "draws nothing")
    q.add_argument("--no-filter-by-segmentation", dest="filter_by_segmentation",
                   action="store_false",
                   help="show every annotation instead of only those on the selected "
                        "segments. By default the filter is ON, so the annotations follow "
                        "whatever you select — which means a link with no --segments opens "
                        "showing none until you pick a body. Toggle it per relationship in "
                        "the layer's ANNOTATIONS tab. The relationships stay bound either "
                        "way")
    q.add_argument("--layer", action="append", metavar="PATH_OR_URL",
                   help="a JSON file holding a layer, or a state whose layers are taken — "
                        "e.g. the output of `em-ngl bboxes` (repeatable)")
    q.add_argument("--image-opacity", type=float, default=None, metavar="F",
                   help="opacity for the --image layers")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding what the volumes record. "
                        "Required when every layer comes from --layer, since a layer file "
                        "carries its own frame but does not establish the viewer's")
    _add_output_flags(q, formats=("url", "state"), default="url")
    q.set_defaults(func=cmd_gen)

    # --- annotate -----------------------------------------------------------
    q = sub.add_parser(
        "annotate", help="a layer of annotations from coordinates you give",
        description="Build a neuroglancer annotation layer from points, boxes, lines or "
                    "ellipsoids you already have — a synapse table, an ROI list, a few "
                    "coordinates typed by hand.\n\n"
                    "The annotations are LOCAL, carried inline in the state, and that is "
                    "the point: only local annotations appear in the Annotations tab, which "
                    "is what makes them clickable and steppable. For a whole volume's worth "
                    "of synapses, write a precomputed source with `em-annot "
                    "annotation-source` and add it with `em-ngl gen --annotations`.\n\n"
                    "Coordinates are zyx and in level-0 voxels unless --scale or --nm says "
                    "otherwise.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--volume", default=None, metavar="VOLUME",
                   help="a volume to take the frame from, and to bounds-check against. "
                        "Optional, but without it you must pass --voxel-size")
    for plural, kind in _ANN_FLAGS:
        cols = ", ".join(c for g in _ANN_CSV_COLUMNS[kind] for c in g)
        q.add_argument(f"--{plural}", action="append", metavar="CSV",
                       help=f"a CSV of {plural} with columns {cols} (repeatable; a path, a "
                            f"URL, or - for stdin). Optional columns: id, description, "
                            f"segments")
        q.add_argument(f"--{kind}", action="append", metavar="COORDS",
                       help=f"one {kind} inline as {cols} (repeatable)")
    q.add_argument("--scale", type=int, default=None, metavar="N",
                   help="the coordinates are in scale-N voxels; convert using the volume's "
                        "own per-level voxel sizes. Needs --volume")
    q.add_argument("--nm", action="store_true",
                   help="the coordinates are in physical nm; convert using the level-0 "
                        "voxel size")
    q.add_argument("--name", default="annotations", help="layer name")
    q.add_argument("--color", default="#ffee00", help="annotation colour")
    q.add_argument("--label", default="a", metavar="PREFIX",
                   help="prefix for generated annotation ids (default: %(default)s)")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding the volume's")
    _add_output_flags(q, formats=("layer", "state", "url"), default="layer")
    q.set_defaults(func=cmd_annotate)

    # --- bboxes -------------------------------------------------------------
    q = sub.add_parser(
        "bboxes", help="a layer of boxes marking where a sparse volume's data is",
        description="Ask a volume where its data is and get one box per written region.\n\n"
                    "For a volume holding a handful of labeled regions inside a large empty "
                    "frame, finding them is the hard part. The boxes come from listing which "
                    "chunk objects exist — TensorStore never persists an all-fill chunk, so "
                    "the set of present keys IS the occupied footprint — then tightening "
                    "each box to its nonzero voxels at a coarse level.\n\n"
                    "The analysis lives in em-volume-tools (`ops.annotate.labeled_regions`); "
                    "this turns its answer into a layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the volume to inspect (path or s3://…)")
    q.add_argument("--level", type=int, default=0,
                   help="which level's chunk grid to list (default: %(default)s)")
    q.add_argument("--tighten-level", type=int, default=None, metavar="N",
                   help="tighten each box to its nonzero voxels at level N. Defaults to "
                        "the greater of 2 and --level; 0 is exact and slowest")
    q.add_argument("--no-tighten", action="store_true",
                   help="skip tightening: the boxes stay chunk-aligned")
    q.add_argument("--kind", choices=("box", "point"), default="box",
                   help="draw each region as a box or as a single centre point")
    q.add_argument("--name", default=None, help="layer name")
    q.add_argument("--label", default=None, metavar="PREFIX",
                   help="prefix for the region ids (default: r)")
    q.add_argument("--color", default="#ffee00", help="annotation colour")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding the volume's")
    _add_output_flags(q, formats=("layer", "state", "url"), default="layer")
    q.set_defaults(func=cmd_bboxes)

    # --- parse --------------------------------------------------------------
    q = sub.add_parser(
        "parse", help="a neuroglancer URL back into its state JSON",
        description="Decode the state out of a link, for reading or editing it.\n\n"
                    "Note that `gen --into` accepts a URL directly, so this is not a "
                    "required step in that workflow — it is the inspection tool.")
    q.add_argument("url", help="the neuroglancer URL (or - to read one from stdin)")
    q.add_argument("--layers", action="store_true",
                   help="print just the layer names and types, one per line")
    q.add_argument("--out", default=None, metavar="PATH_OR_URL",
                   help="write the state JSON here instead of stdout")
    q.set_defaults(func=cmd_parse)

    # --- shaders ------------------------------------------------------------
    q = sub.add_parser(
        "shaders", help="the built-in annotation shaders",
        description="List the built-in shaders, or print one to edit.\n\n"
                    "A shader lives in the viewer state rather than in the source, so a "
                    "link is the only place one can be shipped. Print one, edit it, and "
                    "pass the file back with `--annotation-shader path.glsl`.")
    q.add_argument("name", nargs="?", default=None,
                   help="print this shader's GLSL. Omit to list them")
    q.set_defaults(func=cmd_shaders)

    return p


def _parse_args(argv=None):
    return build_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_gen(args) -> int:
    """A viewer state from volumes, annotation sources and layer files."""
    from .shaders import ShaderProblem
    from .sources import (SourceProblem, annotation_layer, annotation_layer_pair,
                          annotation_source_extent, annotation_source_voxel_size,
                          volume_extent, volume_layer)
    from .state import StateProblem, annotation_extent, load_layer

    err = sys.stderr
    layers: list[dict] = []
    frame_info: dict | None = None
    extents: list[Any] = []
    shader_notes: list[str] = []

    try:
        # --image before --seg so the segmentation draws over the image, which is the
        # order anyone wants and the opposite of alphabetical.
        for volume in args.image or []:
            layer, info = volume_layer(volume, kind="image", opacity=args.image_opacity)
            layers.append(layer)
            frame_info = frame_info or info
            extents.append(volume_extent(volume, info["format"]))
        for i, volume in enumerate(args.seg or []):
            picked = args.segments[i] if i < len(args.segments or []) else None
            layer, info = volume_layer(
                volume, kind="segmentation",
                segments=[int(s) for s in picked.replace(",", " ").split()] if picked
                else None)
            layers.append(layer)
            frame_info = frame_info or info
            extents.append(volume_extent(volume, info["format"]))

        # Annotations after the volumes: the relationships bind to a segmentation layer by
        # NAME, so that layer must exist first, and the first --seg is the one they refer to.
        linked_layer = next((lyr for lyr in layers if lyr["type"] == "segmentation"), None)
        linked = linked_layer["name"] if linked_layer else None
        filtering = bool(args.filter_by_segmentation)
        for source in args.annotations or []:
            if args.annotation_split:
                added, info = annotation_layer_pair(
                    source, shader=args.annotation_shader, linked_segmentation=linked,
                    filter_by_segmentation=filtering)
            else:
                layer, info = annotation_layer(
                    source, shader=args.annotation_shader, linked_segmentation=linked,
                    filter_by_segmentation=filtering)
                added = [layer]
            layers.extend(added)
            shader_notes.append(f"{added[0]['name']}: {info['shader'] or 'no shader'}")
            extents.append(annotation_source_extent(info["info"]))
            if frame_info is None:
                voxel = annotation_source_voxel_size(info["info"])
                if voxel:
                    frame_info = {"voxel_size": voxel, "units": "nm",
                                  "format": "annotations"}
        for path in args.layer or []:
            layers.extend(load_layer(path))
    except (SourceProblem, ShaderProblem, StateProblem, ValueError,
            json.JSONDecodeError) as e:
        raise SystemExit(str(e)) from None

    if not layers:
        raise SystemExit("nothing to show: pass at least one of --image, --seg, "
                         "--annotations or --layer")

    voxel = _ftriple(args.voxel_size, "voxel-size")
    units = "nm" if voxel else (frame_info or {}).get("units")
    if voxel is None:
        voxel = (frame_info or {}).get("voxel_size")
    # With --into the existing state supplies `dimensions`, so no voxel size is needed.
    if voxel is None and not args.into:
        raise SystemExit(
            "no voxel size available: --layer files carry their own frame but do not "
            "establish the viewer's, and no --image/--seg/--annotations source recorded "
            "one. Pass --voxel-size Z,Y,X (nm).")

    known = [e for e in extents if e]
    fit = max(known, key=lambda e: max(e[0])) if known else annotation_extent(layers)

    print(f"{len(layers)} layer(s): "
          + ", ".join(f"{lyr['name']} ({lyr['type']})" for lyr in layers), file=err)
    if voxel:
        print(f"  voxel size {tuple(voxel)} zyx, units {units or '?'}", file=err)
    for note in shader_notes:
        print(f"  shader {note}", file=err)
    _report_bindings(layers, file=err)

    return _emit(args, layers, voxel=voxel, units=units, frame=fit)


def _report_bindings(layers, *, file) -> None:
    """Say which relationships are bound and whether the filter will hide everything."""
    selected_any = any(lyr.get("segments") for lyr in layers)
    for lyr in layers:
        if lyr.get("linkedSegmentationLayer"):
            target = next(iter(lyr["linkedSegmentationLayer"].values()))
            bound = ", ".join(sorted(lyr["linkedSegmentationLayer"]))
            filtered = lyr.get("filterBySegmentation")
            how = (f"; filtered on {', '.join(filtered)}" if filtered
                   else "; showing every annotation")
            print(f"  {lyr['name']}: {bound} bound to {target}{how}", file=file)
            if filtered and not selected_any:
                # Not a warning: it is what the filter means. Said out loud because an
                # empty viewport otherwise looks like a broken layer.
                print(f"    no segments selected, so this opens EMPTY until you pick a "
                      f"body (or pass --no-filter-by-segmentation)", file=file)
        elif lyr["type"] == "annotation" and isinstance(lyr.get("source"), str):
            print(f"  {lyr['name']}: NOT bound to a segmentation — pass --seg for the "
                  f"relationship index to be usable", file=file)


def cmd_annotate(args) -> int:
    """An annotation layer built from coordinates you supply."""
    from .layers import (build_annotation, local_layer, output_dimensions, positions,
                         read_annotation_csv, rescale)

    err = sys.stderr
    records = []
    for plural, kind in _ANN_FLAGS:
        for path in getattr(args, plural) or []:
            try:
                records += read_annotation_csv(_read_ann_file(path), kind, source=path)
            except ValueError as e:
                raise SystemExit(str(e)) from None
    records += _inline_records(args)
    if not records:
        raise SystemExit(
            "nothing to annotate: give at least one of --points/--boxes/--lines/"
            "--ellipsoids (a CSV path, a URL, or - for stdin) or an inline "
            "--point/--box/--line/--ellipsoid")

    voxel, units, shape = _volume_frame(args.volume, args.voxel_size)
    print(f"{args.volume or '(no volume: frame from --voxel-size)'}", file=err)
    if voxel:
        print(f"  frame       {'x'.join(f'{v:g}' for v in voxel)} {units or '?'} "
              f"per level-0 voxel", file=err)

    # Coordinates land in the layer as level-0 voxels, because that is the frame
    # `outputDimensions` states. Both other input units are one per-axis scaling.
    if args.scale:
        if not args.volume:
            raise SystemExit("--scale needs --volume: the conversion uses the volume's own "
                             "per-level voxel sizes, which cannot be guessed")
        factor = _level0_factor(args.volume, args.scale)
        records = rescale(records, factor)
        print(f"  scale {args.scale}     x{factor} (zyx) -> level-0 voxels", file=err)
    elif args.nm:
        if not voxel:
            raise SystemExit("--nm needs the level-0 voxel size: pass --voxel-size or a "
                             "--volume that records one")
        records = rescale(records, tuple(1.0 / v for v in voxel))
        print(f"  nm          /{tuple(voxel)} (zyx) -> level-0 voxels", file=err)

    ids, annotations = set(), []
    for i, r in enumerate(records):
        ident = str(r.get("id") or f"{args.label}{i:03d}")
        if ident in ids:
            raise SystemExit(f"duplicate annotation id {ident!r}: neuroglancer keys its "
                             f"annotations by id, so duplicates collide. Fix the `id` "
                             f"column, or drop it and let them be numbered.")
        ids.add(ident)
        annotations.append(build_annotation(r, ident))

    dims, warning = output_dimensions(voxel, units)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    print(f"  annotations {len(annotations)}: "
          + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())), file=err)
    n_seg = sum(1 for r in records if r.get("segments"))
    if n_seg:
        print(f"  segments    {n_seg} annotation(s) linked to body ids", file=err)

    # A wrong unit is the failure mode here, and it does not look like one: coordinates
    # 8x off are still valid annotations, just somewhere else. The volume's extent is the
    # only check available, so make it loudly.
    if shape:
        outside = [i for i, r in enumerate(records)
                   if any(not (0 <= c <= shape[a])
                          for g in positions(r) for a, c in enumerate(g))]
        where = f"the level-0 extent {shape} (zyx)"
        if outside:
            print(f"\n  WARNING: {len(outside)} annotation(s) fall outside {where} — first "
                  f"at index {outside[0]}. Check whether the coordinates are in another "
                  f"scale's voxels (--scale N) or in nm (--nm).", file=err)
        else:
            print(f"  bounds      all inside {where}", file=err)
    if warning:
        print(f"\n  WARNING: {warning}", file=err)

    layer = local_layer(annotations, dims, name=args.name, color=args.color)
    layers = _with_volume(args, [layer])
    return _emit(args, layers, voxel=voxel, units=units,
                 frame=(shape, (0, 0, 0)) if shape else None, selected=layer["name"])


def cmd_bboxes(args) -> int:
    """A layer of boxes over a sparse volume's occupied regions."""
    from em_volume_tools.ops.annotate import NoOccupancy, labeled_regions

    from .layers import boxes_layer, output_dimensions

    err = sys.stderr
    volume = args.volume.rstrip("/")
    tighten = None if args.no_tighten else (
        max(2, args.level) if args.tighten_level is None else args.tighten_level)
    try:
        regions, ctx = labeled_regions(volume, level=args.level, tighten_level=tighten)
    except (FileNotFoundError, NoOccupancy, ValueError) as e:
        raise SystemExit(str(e)) from None

    voxel, units, shape = _volume_frame(volume, args.voxel_size)
    dims, warning = output_dimensions(voxel, units)

    print(f"{volume}", file=err)
    print(f"  level {args.level}: {len(regions)} region(s)", file=err)
    if not regions:
        print("  no stored chunks — the volume is empty at this level", file=err)
        return 1
    print(f"\n{'#':>3}  {'z':>15} {'y':>15} {'x':>15}  {'extent zyx':>18} "
          f"{'labels':>7}", file=err)
    for i, r in enumerate(regions):
        span = " ".join(f"{r['lo'][a]:6d}-{r['hi'][a]:6d}" for a in range(3))
        ext = "x".join(str(r["hi"][a] - r["lo"][a]) for a in range(3))
        n = "-" if r["n_labels"] is None else f"{r['n_labels']:,}"
        print(f"{i:>3}  {span}  {ext:>18} {n:>7}", file=err)
    if warning:
        print(f"\n  WARNING: {warning}", file=err)

    layer = boxes_layer(regions, dims,
                        name=args.name or f"{volume.rsplit('/', 1)[-1]}-regions",
                        color=args.color, kind=args.kind, label=args.label or "r")
    layers = _with_volume(args, [layer], volume=volume)
    frame = ((shape, (0, 0, 0)) if shape else
             (tuple(regions[0]["hi"][a] - regions[0]["lo"][a] for a in range(3)),
              tuple(regions[0]["lo"])))
    return _emit(args, layers, voxel=voxel, units=units, frame=frame,
                 selected=layer["name"])


def _with_volume(args, layers: list[dict], volume: str | None = None) -> list[dict]:
    """Prepend the volume's own layer when a whole state is being emitted.

    A bare layer is the default output and needs no volume. A *state* without one shows
    annotations floating in nothing, so the volume goes in — which is what the old
    ``--state`` flag did, except now through the one state builder rather than a second.

    Not with ``--into``, though: that state already has whatever volumes it was built with, and
    adding this one again would either duplicate it or collide with its name.
    """
    if _resolve_format(args) == "layer" or getattr(args, "into", None):
        return layers
    volume = volume or getattr(args, "volume", None)
    if not volume:
        return layers
    from .sources import SourceProblem, volume_layer

    try:
        layer, _info = volume_layer(volume)
    except SourceProblem as e:
        raise SystemExit(str(e)) from None
    return [layer, *layers]


def cmd_parse(args) -> int:
    """A neuroglancer URL back into its state JSON."""
    from .layers import render
    from .state import parse_url

    url = sys.stdin.read().strip() if args.url == "-" else args.url
    try:
        state = parse_url(url)
    except (ValueError, json.JSONDecodeError) as e:
        raise SystemExit(str(e)) from None

    if args.layers:
        for lyr in state.get("layers") or []:
            if isinstance(lyr, dict):
                print(f"{lyr.get('name', '(unnamed)')}\t{lyr.get('type', '?')}")
            else:
                # A state's `layers` array may carry a bare string — the reference male-CNS
                # state has a comment sitting in one — so this must not assume a dict.
                print(f"{lyr}\t(not a layer object)")
        return 0

    text = render(state)
    if args.out:
        _write_text(args.out, text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_shaders(args) -> int:
    """List the built-in annotation shaders, or print one."""
    from .shaders import SHADERS

    if args.name is None:
        for name, entry in SHADERS.items():
            print(f"{name}\t{entry['doc']}")
            print(f"  reads: {', '.join(entry['properties'])}")
        print("\nA shader naming a property the source does not declare fails to compile "
              "and the layer draws NOTHING, so `gen` checks before applying one.")
        return 0
    if args.name not in SHADERS:
        raise SystemExit(f"no built-in shader {args.name!r}; known: "
                         + ", ".join(SHADERS))
    print(SHADERS[args.name]["source"], end="")
    return 0


def main(argv=None) -> int:
    import logging

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
