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
        return cls._from_read(
            _read_source(str(path), dataset, "hdf5", level=0, crop=crop,
                         voxel_size=voxel_size),
            path=str(path), kind=kind, name=name, **kwargs)

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
        return cls._from_read(
            _read_source(volume, None, None, level=level, crop=crop,
                         voxel_size=voxel_size),
            path=volume, kind=kind, name=name, **kwargs)

    @classmethod
    def from_piece(cls, piece: Any, kind: str, *, name: str | None = None,
                   **kwargs) -> "ServedLayer":
        """A :class:`neu_lib.Piece` — an array that already carries its frame.

            piece = neu_vol.read_piece("gt.h5:/z07901")
            layer = ServedLayer.from_piece(piece, "segmentation")

        The conversion goes this way round, and it has to: ``Piece`` lives in neu-lib, the
        bottom tier, and a ``Piece.as_layer`` would mean the vocabulary package naming a
        viewer type three tiers above it. neu-glance reading a neu-lib type is the allowed
        direction.
        """
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
        path, _, dataset = src.partition(":")
        if not dataset.startswith("/"):
            path, dataset = src, ""
        return cls._from_read(
            _read_source(path, dataset or None, None, level=level, crop=crop,
                         voxel_size=voxel_size),
            path=path, kind=kind, name=name, **kwargs)

    @classmethod
    def _from_read(cls, read: dict, *, path: str, kind: str | None,
                   name: str | None, **kwargs) -> "ServedLayer":
        """Assemble a layer from :func:`_read_source`'s result. Shared by the three above."""
        kind = kind or read["kind"]
        if kind is None:
            raise ServeProblem(
                f"{path} records no image/segmentation type, so kind= is required — one of "
                f"{', '.join(KINDS)}. It is not inferred from the dtype: neuroglancer's own "
                f"guess reads a uint8 label array as an image, which averages label ids on "
                f"downsample and loses the colour hashing and the selection UI. (A "
                f"precomputed volume records the type in its `info`, and then this is not "
                f"needed.)")
        return cls(array=read["array"], kind=kind,
                   name=name or _default_name(path, read["dataset"]),
                   frame=read["frame"], channel_axis=read["channel_axis"], **kwargs)

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
        return Piece(array=self.array, frame=self.frame)

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


def _crop_request(crop: Any, what: str = "crop"):
    """``(voxel_box, nm_bounds)`` — exactly one of them, or both ``None``.

    Three ways to say which box you want, because there are three things a caller has in
    hand:

    * ``((z0,y0,x0), (z1,y1,x1))`` or the flat ``(z0,…,x1)``, in **whole voxels of the
      level being read** — the direct form, and what the CLI takes;
    * anything carrying ``bounds_nm`` — a :class:`neu_lib.Piece`, or another
      :class:`ServedLayer` — meaning *the same physical box as that*. This is the one that
      answers "show me the image under this ground-truth crop", and it is why
      nanometres exist as the shared model space: the two frames have different voxel
      sizes and different origins, so no voxel box is transferable between them.
    * a ``{"nm": (lo, hi)}`` mapping, for a physical box with nothing to carry it.

    Voxels are resolved here; nanometres cannot be, because converting them needs the
    target level's own voxel size and origin, which are not known until it is opened.
    """
    if crop is None:
        return None, None
    bounds = getattr(crop, "bounds_nm", None)
    if bounds is not None:
        return None, tuple(bounds)
    if isinstance(crop, Mapping):
        if "nm" not in crop:
            raise ServeProblem(f"{what} mapping must carry 'nm': (lo, hi); got {crop!r}")
        return None, tuple(crop["nm"])
    flat = tuple(crop)
    if len(flat) == 6 and all(np.isscalar(v) or isinstance(v, (int, float))
                             for v in flat):
        lo, hi = tuple(int(v) for v in flat[:3]), tuple(int(v) for v in flat[3:])
    elif len(flat) == 2:
        lo, hi = tuple(int(v) for v in flat[0]), tuple(int(v) for v in flat[1])
    else:
        raise ServeProblem(
            f"{what} must be ((z0,y0,x0), (z1,y1,x1)) or (z0,y0,x0,z1,y1,x1) in whole "
            f"VOXELS of the level being read, a Piece/ServedLayer to take the same "
            f"physical box as, or {{'nm': (lo, hi)}} — got {crop!r}")
    if len(lo) != 3 or len(hi) != 3:
        raise ServeProblem(f"{what} corners are zyx, so 3 values each; got {lo} / {hi}")
    return (lo, hi), None


def _read_source(path: str, dataset: str | None, fmt: str | None, *, level: int,
                 crop: Any, voxel_size: Any, backend: Any = None) -> dict:
    """Read one source and everything known about it. The shared half of the constructors.

    Returns ``{array, frame, channel_axis, kind, dataset, format}`` — ``kind`` being
    whatever the source *records*, or ``None`` when it records nothing and the caller has
    to say. A precomputed volume records it in ``info["type"]`` and OME in the multiscales
    ``type``, which is authoritative and must never be second-guessed: getting it wrong
    averages label ids into ids that were never in the data. An HDF5 file has nowhere
    agreed-on to say, so it records nothing.

    ``backend`` short-circuits the opening for a caller that already has one; its
    ``to_spec()`` still supplies the metadata, so the frame is read the same way either
    way and the axis-order rule stays in one place.

    The frame travels with the array, and a crop shifts its origin — which is what keeps a
    served box on top of the volume it came from rather than at nm zero (invariant 1).

    Every store read here is wrapped in ``neu_vol.logs.quiet_reads``. These constructors
    are called straight from a notebook, so from the caller's point of view they *are* the
    entry point and there is no ``main()`` to wrap — and an S3 open logs two
    ``AuthCredentialsProvider`` lines at ``E`` severity per prefix that are **not**
    failures, only the two providers that missed before the environment one succeeded.
    """
    from neu_lib import Frame
    from neu_vol import describe, open_backend
    from neu_vol.logs import quiet_reads
    from neu_vol.source_metadata import (level_spec, location_spec,
                                         read_level_voxel_sizes, read_source_metadata,
                                         require_one_array)

    crop, crop_nm = _crop_request(crop)
    with quiet_reads():
        if backend is not None:
            spec = dict(backend.to_spec())
            fmt = fmt or spec.get("backend")
            dataset = spec.get("dataset") or dataset
            meta = read_source_metadata(spec) or {}
        elif fmt is None:
            described = describe(path, dataset=dataset or None)
            if described["shape"] is None:
                require_one_array(described, path, "neu-glance serve")
            fmt = described["format"]
            spec = described["spec"]
            meta = described["meta"] or {}
            dataset = described.get("dataset") or dataset
        else:
            # A format named outright, for a file whose name detection would not recognise.
            spec = location_spec(path, fmt, dataset=dataset or None)
            meta = read_source_metadata(spec) or {}
            dataset = spec.get("dataset") or dataset

        single = fmt in ("hdf5", "image_stack")
        if level and single:
            raise ServeProblem(
                f"level {level} needs a multiscale volume; {path} is {fmt}, a single array")

        # **A zarr OME group is not an array**, so level 0 must go through the metadata's
        # own `data_spec` (which names the level's subdirectory) rather than the group
        # path. Addressing the path directly failed to open at all — the same trap
        # `ops/pack.py` documents, and it hid because precomputed selects a scale with
        # `scale_index` on one path and so worked fine.
        if backend is not None:
            read_spec = spec
        elif single:
            read_spec = spec
        elif level:
            read_spec = level_spec(path, fmt, level, dataset=dataset or None)
        elif meta.get("data_spec"):
            read_spec = meta["data_spec"]
        else:
            read_spec = spec          # a bare array: it has no levels to descend into
        backend = backend or open_backend(read_spec)
        shape = tuple(int(s) for s in backend.shape)
        per_level = read_level_voxel_sizes(spec) or []
        voxel = (tuple(voxel_size) if voxel_size
                 else (tuple(per_level[level]) if level < len(per_level) else None)
                 or (tuple(meta["voxel_size"]) if meta.get("voxel_size") else None))
        if voxel is None:
            raise ServeProblem(
                f"{path} records no voxel size, so nothing here knows its physical "
                f"scale; pass voxel_size=(z, y, x)")

        channel_axis = len(shape) == 4
        spatial = shape[1:] if channel_axis else shape
        recorded = (tuple(float(o) for o in meta["offset"])
                    if meta.get("offset") else (0.0, 0.0, 0.0))
        if crop_nm is not None:
            # A physical box, into THIS level's voxels — which needs the level's own voxel
            # size and origin, and so cannot be done before it is opened. Grown outward, so
            # the read contains the box asked for rather than dropping a face when the
            # levels do not divide evenly.
            box = Frame(voxel_size_nm=voxel, origin_nm=recorded).voxel_box(crop_nm)
            crop = (tuple(max(0, v) for v in box.lo),
                    tuple(min(e, v) for v, e in zip(box.hi, spatial)))
            logger.info("crop %s nm -> level-%d voxels %s:%s", crop_nm, level, *crop)
            if any(b <= a for a, b in zip(*crop)):
                raise ServeProblem(
                    f"the physical box {crop_nm} nm does not overlap {path}'s level-{level} "
                    f"extent {spatial} at {voxel} nm/voxel starting {recorded} nm. If the "
                    f"box came from another dataset's crop, the two are not the same "
                    f"volume")
            # **A clamp that removes most of the box means the wrong volume**, not an edge
            # case. Taking a physical box off one dataset's crop and reading it out of
            # another's image is the easy mistake — the numbers are plausible, the read
            # succeeds, and what comes back is a thin slab nobody asked for. So say it, with
            # the fraction, rather than logging the conversion and moving on.
            asked = math.prod(b - a for a, b in zip(box.lo, box.hi))
            got = math.prod(b - a for a, b in zip(*crop))
            if got < asked:
                clipped = [f"{'zyx'[a]} {box.lo[a]}:{box.hi[a]} -> {crop[0][a]}:{crop[1][a]}"
                           for a in range(3) if (box.lo[a], box.hi[a]) != (crop[0][a],
                                                                          crop[1][a])]
                warn = logger.warning if got * 2 < asked else logger.info
                warn("the physical box does not fit %s's level-%d extent %s and was "
                     "clipped to %.0f%% of it (%s)%s", path, level, spatial,
                     100.0 * got / asked, "; ".join(clipped),
                     ". Losing most of a box usually means it came from a different "
                     "dataset than this volume" if got * 2 < asked else "")
        lo = (0, 0, 0)
        if crop:
            lo, hi = crop
            for axis, (start, stop, extent) in enumerate(zip(lo, hi, spatial)):
                if not 0 <= start < stop <= extent:
                    nm = " — a crop in NANOMETRES looks like this" if any(
                        v > e * 4 for v, e in zip(hi, spatial)) else ""
                    raise ServeProblem(
                        f"crop {lo}:{hi} does not fit {path}'s level-{level} extent "
                        f"{spatial} on axis {axis}. The box is in whole VOXELS of that "
                        f"level, zyx, half-open{nm}")
            region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        else:
            region = tuple(slice(0, int(s)) for s in spatial)
        if channel_axis:
            region = (slice(0, shape[0]),) + region

        # A crop out of something that already knows where it belongs lands at the SUM,
        # the same rule `neu-vol to-hdf5 --crop-bbox` follows.
        origin = tuple(o + i * v for o, i, v in zip(recorded, lo, voxel))

        return {"array": backend.read_region(region),
                "frame": Frame(voxel_size_nm=voxel, origin_nm=origin),
                "channel_axis": channel_axis,
                "kind": meta.get("kind"),
                "dataset": dataset,
                "format": fmt}


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
