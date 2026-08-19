"""em-ngl — neuroglancer states, layers and links.

Everything that produces something a *viewer* consumes, and nothing that produces data. The
split from em-volume-tools is deliberate and one-way: em-volume-tools writes volumes and
em-annotation writes annotation sources, and neither knows a viewer exists. This package sits
above both, so a shader that reads a synapse property lives here rather than being pushed down
into the library that happens to own the store access.

Three things it makes, all from the same pieces:

- a **state**, serialized as a URL or as JSON (:mod:`em_ngl.state`)
- a **layer** for something on a store — a volume or a precomputed annotation source
  (:mod:`em_ngl.sources`)
- a **layer** of local annotations, from coordinates or from occupancy boxes
  (:mod:`em_ngl.layers`)

:mod:`em_ngl.layers` and :mod:`em_ngl.state` are pure — plain data in, plain data out, no
store access — and :mod:`em_ngl.sources` is the only module that reads anything. Keeping that
line is what will let a locally served volume become a layer without touching state assembly.

Top-level names resolve lazily (PEP 562) so ``em-ngl --help`` does not pay for
em-volume-tools' import graph; ``cli`` reads ``__version__`` from here.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: name -> module it lives in. Kept explicit rather than star-importing, so the lazy
#: resolution below cannot silently start pulling a heavy module for a light name.
_EXPORTS = {
    "build_state": "state",
    "state_url": "state",
    "parse_url": "state",
    "load_layer": "state",
    "load_state": "state",
    "merge_into": "state",
    "default_view": "state",
    "annotation_extent": "state",
    "volume_layer": "sources",
    "volume_extent": "sources",
    "annotation_layer": "sources",
    "annotation_layer_pair": "sources",
    "read_annotation_info": "sources",
    "annotation_source_extent": "sources",
    "annotation_source_voxel_size": "sources",
    "local_layer": "layers",
    "boxes_layer": "layers",
    "build_annotation": "layers",
    "read_annotation_csv": "layers",
    "output_dimensions": "layers",
    "rescale": "layers",
    "render": "layers",
    "SHADERS": "shaders",
    "pick_shader": "shaders",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    """Resolve a top-level export on first use.

    Eager imports here would make every ``em-ngl --help`` pay for em-volume-tools, since
    :mod:`em_ngl.cli` reads ``__version__`` from this module. A test pins that.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return list(__all__)
