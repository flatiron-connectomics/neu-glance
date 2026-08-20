"""Layers built from something on a store: a volume, or a precomputed annotation source.

This is the only module that reads a store, and it reads as little as it can. The whole
neu-vol dependency is concentrated here so :mod:`neu_glance.state` and
:mod:`neu_glance.layers` stay pure — which matters beyond tidiness: a layer whose source is a
locally served volume (`http://localhost:PORT/...`) has no store to inspect, so state
assembly must never require one.

Two things here fail silently if you get them wrong:

- **The source scheme.** `precomputed://` vs `zarr://` is decided by
  :func:`~neu_vol.source_metadata.detect_backend`, not by the path.
- **Volume or annotation source.** Both are addressed `precomputed://` and both have an
  `info` at the root, so nothing about the URL tells them apart — and an annotation layer
  pointed at a volume loads happily and draws nothing at all. The ``@type`` is the only
  honest check, so :func:`read_annotation_info` makes it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .shaders import SPLIT_CONTROLS, ShaderProblem, pick_shader

#: How neuroglancer addresses each backend. The classic `scheme://` form rather than the
#: newer `kvstore|adapter:` pipeline syntax: both work in current builds, only this one
#: works in older ones, and a link is the thing most likely to be opened by someone
#: running a different viewer.
def _scheme_map() -> dict[str, str]:
    from neu_vol.source_metadata import PRECOMPUTED_GZ

    return {"neuroglancer_precomputed": "precomputed",
            PRECOMPUTED_GZ: "precomputed",
            "zarr3": "zarr"}


#: The `@type` a precomputed annotation source declares.
ANNOTATION_TYPE = "neuroglancer_annotations_v1"

#: The relationship whose *own* end of the line is the given side, and the shader controls that
#: show only that end. Filtering on `body_pre` gives the selected body's OUTPUTS; on `body_post`
#: its INPUTS. Two layers on one source, which is how the reference dataset presents them.
SPLIT_SIDES = (("pre", "body_pre"), ("post", "body_post"))


class SourceProblem(RuntimeError):
    """A named volume or annotation source could not be used as a layer source."""


def volume_extent(volume: str, fmt: str) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` in level-0 voxels, or None if not determinable.

    Read as cheaply as the format allows: precomputed carries every scale's ``size`` and
    ``voxel_offset`` in the one ``info`` this has already fetched, so it costs nothing,
    while zarr needs the level-0 array's own metadata — one open, not the every-level
    probe that ``describe`` does.
    """
    from neu_vol.location import read_json
    from neu_vol.source_metadata import read_source_metadata

    if str(fmt).startswith("neuroglancer_precomputed"):
        info = read_json(volume.rstrip("/") + "/info") or {}
        scales = info.get("scales")
        if not scales:
            return None
        finest = min(scales, key=lambda s: tuple(s["resolution"]))
        size = [int(v) for v in finest["size"]][::-1]                    # xyz -> zyx
        off = [int(v) for v in finest.get("voxel_offset", [0, 0, 0])][::-1]
        return tuple(size), tuple(off)

    meta = read_source_metadata({"backend": fmt, "path": volume})
    if not meta:
        return None
    try:
        from neu_vol.backends.base import open_backend

        shape = tuple(int(s) for s in open_backend(meta["data_spec"]).shape)
    except Exception:
        return None
    spatial = shape[-3:] if len(shape) > 3 else shape       # drop a channel axis
    return spatial, (0, 0, 0)


def volume_layer(volume: str, *, kind: str | None = None, name: str | None = None,
                 segments: Sequence[int] | None = None,
                 opacity: float | None = None) -> tuple[dict, dict]:
    """One layer for ``volume``, plus what its metadata says about the frame.

    ``kind`` overrides the volume's own record of whether it is an image or a
    segmentation; without it the recorded value decides, because a segmentation shown as
    an image is a grey mush and the mistake is easy to miss on a small ROI.
    """
    from neu_vol.source_metadata import detect_backend, read_source_metadata

    volume = volume.rstrip("/")
    fmt = detect_backend(volume)
    if fmt is None:
        raise SourceProblem(f"no volume found at {volume}")
    scheme = _scheme_map().get(fmt)
    if scheme is None:
        raise SourceProblem(f"{volume} is {fmt}, which neuroglancer cannot read")

    # `read_source_metadata`, NOT `describe`: this needs the voxel size, the units and
    # the recorded kind, all of which are in `info`. `describe` additionally OPENS EVERY
    # LEVEL and probes for a foreign marker — the expensive tier documented in
    # source_metadata — and a link needs none of it. Two volumes here meant ~20 store
    # opens for numbers already read.
    meta = read_source_metadata({"backend": fmt, "path": volume}) or {}
    resolved = kind or meta.get("kind") or "image"
    layer: dict[str, Any] = {
        "type": "segmentation" if resolved == "segmentation" else "image",
        "name": name or volume.rsplit("/", 1)[-1],
        "source": f"{scheme}://{volume}",
    }
    if segments:
        # Strings, not ints: neuroglancer segment ids are uint64 and JSON numbers are
        # doubles, so a real id above 2**53 would arrive rounded.
        layer["segments"] = [str(int(s)) for s in segments]
    if opacity is not None:
        layer["opacity"] = float(opacity)
    return layer, {"voxel_size": meta.get("voxel_size"), "units": meta.get("units"),
                   "format": fmt}


def read_annotation_info(source: str) -> dict:
    """The ``info`` of a precomputed annotation source, refusing anything else."""
    from neu_vol.location import read_json

    source = source.rstrip("/")
    info = read_json(source, "info")
    if info is None:
        raise SourceProblem(f"no info at {source}")
    if info.get("@type") != ANNOTATION_TYPE:
        raise SourceProblem(
            f"{source} is {info.get('@type') or 'not an annotation source'}, not "
            f"{ANNOTATION_TYPE}. Volumes go to --image or --seg; --annotations wants a "
            f"precomputed ANNOTATION source (what `neu-mark annotation-source` writes).")
    return info


def annotation_layer(source: str, *, name: str | None = None,
                     shader: str | None = None,
                     linked_segmentation: str | None = None,
                     filter_by_segmentation: bool = True,
                     filter_relationships: Sequence[str] | None = None,
                     controls: Mapping[str, Any] | None = None,
                     info: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    """One annotation layer for a precomputed annotation source, plus its frame.

    ``linked_segmentation`` is the name of a segmentation layer in the same state, and it is
    **what makes the relationship index do anything in the viewer**. The source's
    ``relationships`` are keyed on segment id, but neuroglancer only consults them once each
    relationship is bound to a layer whose selected segments it can read; without the binding
    the layer draws every annotation and "this body's synapses" is not available at all.

    ``filter_relationships`` narrows which of them the filter uses. Every relationship stays
    *bound* regardless — binding is what makes a relationship usable, filtering is what decides
    whether it restricts the view — and filtering on a subset is what turns one source into
    "this body's outputs" and "this body's inputs" as separate layers.
    """
    source = source.rstrip("/")
    info = dict(info) if info is not None else read_annotation_info(source)
    layer: dict[str, Any] = {
        "type": "annotation",
        "name": name or source.rsplit("/", 1)[-1],
        "source": f"precomputed://{source}",
    }
    try:
        shader_source, why = pick_shader(info, shader)
    except ShaderProblem as e:
        raise SourceProblem(str(e)) from None
    if shader_source:
        layer["shader"] = shader_source
    if controls:
        layer["shaderControls"] = dict(controls)

    relationships = [r["id"] for r in info.get("relationships", []) or []]
    if linked_segmentation and relationships:
        layer["linkedSegmentationLayer"] = {r: linked_segmentation for r in relationships}
        if filter_by_segmentation:
            picked = [r for r in (filter_relationships or relationships)
                      if r in relationships]
            if filter_relationships and not picked:
                raise SourceProblem(
                    f"none of {', '.join(filter_relationships)} is a relationship of "
                    f"{source}; it declares " + (", ".join(relationships) or "none"))
            layer["filterBySegmentation"] = picked

    return layer, {"info": info, "shader": why, "relationships": relationships}


def annotation_layer_pair(source: str, *, name: str | None = None,
                          shader: str | None = None,
                          linked_segmentation: str | None = None,
                          filter_by_segmentation: bool = True) -> tuple[list[dict], dict]:
    """Two layers on one source: the selected body's outputs, then its inputs.

    A single layer filtered on both relationships answers "every synapse touching this body",
    which conflates the two directions — and drawn together the endpoint markers overlap at any
    zoom that shows more than a few. Splitting costs nothing on the store (one source, two
    layers) and each half then has its own visibility, colour and marker size.

    The ``info`` is read ONCE and reused for both halves, so a split costs no extra store
    access.
    """
    base = (name or source.rstrip("/").rsplit("/", 1)[-1])
    info = read_annotation_info(source)
    layers, detail = [], None
    for side, relationship in SPLIT_SIDES:
        layer, detail = annotation_layer(
            source, name=f"{base}-{side}", shader=shader,
            linked_segmentation=linked_segmentation,
            filter_by_segmentation=filter_by_segmentation,
            filter_relationships=[relationship], controls=SPLIT_CONTROLS[side],
            info=info)
        layers.append(layer)
    return layers, detail


def annotation_source_extent(info: Mapping[str, Any]) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` from an annotation source's declared bounds.

    The format requires ``lower_bound``/``upper_bound``, so an annotations-only link can
    still open framed on its data rather than at the origin corner — the same job
    :func:`volume_extent` does for a volume.
    """
    lower, upper = info.get("lower_bound"), info.get("upper_bound")
    if not lower or not upper or len(lower) != 3 or len(upper) != 3:
        return None
    lo = [float(v) for v in reversed(lower)]            # stored xyz, wanted zyx
    hi = [float(v) for v in reversed(upper)]
    return tuple(max(1.0, h - l) for l, h in zip(lo, hi)), tuple(lo)


def annotation_source_voxel_size(info: Mapping[str, Any]) -> tuple | None:
    """Level-0 voxel size in nm, zyx, from the source's ``dimensions``.

    ``dimensions`` is ``{axis: [scale, unit]}`` in SI, so metres become nanometres here.
    Lets an annotations-only link establish the viewer's frame without --voxel-size.
    """
    dims = info.get("dimensions")
    if not isinstance(dims, Mapping):
        return None
    try:
        scales = [dims[axis] for axis in ("x", "y", "z")]
    except KeyError:
        return None
    out = []
    for scale, unit in scales:
        if unit != "m":
            return None
        out.append(float(scale) * 1e9)
    return tuple(reversed(out))                        # xyz read, zyx returned
