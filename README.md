# neu-glance

Neuroglancer viewer states, annotation layers and shareable links.

Everything that produces something a **viewer** consumes, and nothing that produces data.
neu-vol writes volumes, neu-mark writes precomputed annotation sources, and
neither of them knows a viewer exists — so a shader that reads a synapse property lives here
rather than being pushed down into whichever library happens to own the store access.

```bash
neu-glance gen --image s3://bucket/em --seg s3://bucket/seg_v1 \
    --annotations s3://bucket/seg_v1/synapses_v1 --annotation-split --segments 12345
```

## The six subcommands

| | produces |
| --- | --- |
| `neu-glance gen` | a **state** from volumes, annotation sources and layer files |
| `neu-glance annotate` | a **layer** from coordinates or a CSV |
| `neu-glance bboxes` | a **layer** from a volume's occupancy |
| `neu-glance serve` | a **running viewer** over arrays held in this process |
| `neu-glance parse` | a URL back into its state JSON |
| `neu-glance shaders` | lists or prints the built-in shaders |

The three producers share one output stage. `--format {layer,state,url}` chooses the
serialization — a bare layer to paste into a state's `layers` array, a whole state for
neuroglancer's `{}` editor, or a link carrying that state — and `--out` writes it somewhere
instead of stdout.

`--into` merges the new layers into an **existing** state, given as a URL or a JSON file. The
state's own `dimensions`, position and zoom are kept, so adding a layer does not move your
view, and a layer whose name is already taken is renamed and reported rather than silently
shadowing the one already there.

## Looking at something that is not published

`serve` hosts arrays from this process and prints a viewer link — a ground-truth crop, a
box out of a volume, a probability map straight out of a model. It replaces serving HDF5
chunks through chunkflow to eyeball them.

```bash
neu-glance serve --seg piece.h5                          # a crop, labels and all
neu-glance serve --image vol --seg gt --crop-bbox 0,0,0,64,512,512
neu-glance serve --image piece.h5:/raw --prob piece.h5:/affinity
```

Needs the optional extra, `pip install 'neu-glance[serve]'`, because building a state needs
none of it and CI has no business downloading a viewer bundle.

From a notebook it is the same thing with a handle on it, and the handle is the point — the
server runs in **your kernel**, so the browser's state is readable here:

```python
from neu_glance import serve, ServedLayer

layers = [
    ServedLayer.from_hdf5("gt.h5", "/z07901", "segmentation"),   # frame from the file
    ServedLayer.from_volume("s3://my-bucket/em", level=1,        # kind from its info
                            crop=((0, 0, 0), (64, 512, 512))),
    ServedLayer.from_array(prob, "probability", voxel_size=(40, 8, 8)),
]
srv = serve(layers)
srv                                  # renders the link
srv.boxes()                          # boxes you drew, as (lo, hi) in zyx voxels
srv.selected_segments()              # label ids you clicked
srv.on_click(lambda c: print(c.voxel, c.values))
```

Taking the same physical box as something else — the "show me the image under this
ground-truth crop" case — is what `crop=` does with a layer or a `neu_lib.Piece`:

```python
gt = ServedLayer.from_hdf5("gt.h5", "/vol_03700", "segmentation")
em = ServedLayer.from_volume("s3://my-bucket/em", crop=gt)      # the same box, in nm
em2 = ServedLayer.from_volume("s3://my-bucket/em", level=2, crop=gt)   # and coarser
serve([em, gt])
```

Nanometres are the only space that transfers: the two frames have different voxel sizes
*and* different origins, so a voxel box from one means nothing in the other. `layer.bbox`,
`layer.bounds_nm` and `layer.piece` report where a layer is, and a box clipped to a
fraction of what was asked for warns — losing most of it usually means it came from a
different dataset than the volume.

The constructors follow one rule: **infer what the source records, require what it
does not.** A frame, a dataset name and the channel axis are all written down — in an HDF5
file's attributes, a precomputed `info`, or the array's own rank — so reading them is not
guessing, and dropping them is the silent failure. `kind` is asked for, because an HDF5 file
has nowhere agreed-on to record it and reading it off the dtype is the mistake neuroglancer
itself makes. A volume that records `info["type"]` is the exception, and then `from_volume`
needs nothing.

`srv.boxes()` closes the loop: pick a region in the viewer, hand it straight to
`--crop-bbox`, `extract_roi` or `neu-vol write`. It needs an annotation layer to draw in,
which is **opt-in** — `serve(..., regions=True)` or `--regions` — because a viewer that
opens with a layer nobody asked for reads as a bug.

The viewer opens **centred on the served data and zoomed to fit it**. Neuroglancer's own
default is the origin *corner* at one voxel per pixel, which for a crop sitting at voxel
3700 of its parent is a view of empty space a long way from anything.

Three things about it that are not obvious:

- **A served link is not shareable.** Each array is addressed
  `python://volume/<viewer-token>`, scoped to the process and dead when it exits. So
  `serve` has no `--format url`; use `gen` against a volume that is actually published.
- **`--seg` is served as labels whatever the dtype.** neuroglancer guesses segmentation
  only for uint16/32/64, so a uint8 label array would be read as an image — averaging
  label ids on downsample and losing the colour hashing and the selection UI, silently.
- **The frame travels with the array.** A crop keeps its origin, so it lands on top of the
  volume it came from rather than at nm zero. A source that records one (`neu-vol to-hdf5`
  writes it) needs no `--voxel-size`.

## Things that fail silently, and where they are handled

Neuroglancer is forgiving in the worst way: a wrong state loads cleanly and shows you
something plausible. The comments in each module say which mistake they exist to prevent, but
the four worth knowing up front:

- **A `dimensions` block that disagrees with the data** puts every layer in the wrong place
  and still loads. `gen` derives it from a volume's recorded voxel size rather than assuming.
- **A volume and an annotation source are both `precomputed://` with an `info` at the root**,
  so nothing about a URL tells them apart — and an annotation layer pointed at a volume draws
  nothing at all. `sources.read_annotation_info` checks the `@type`.
- **A shader naming a `prop_` the source does not declare fails to compile**, and the layer
  then draws nothing, with the error visible only in the layer's shader tab. It does not fall
  back. `shaders.pick_shader` refuses the pairing instead.
- **`linkedSegmentationLayer` is what makes a relationship index do anything.** The source
  keys its relationships on segment id, but the viewer only consults them once each is bound
  to a layer whose selection it can read. Without the binding there is no "this body's
  synapses" at all.

## Layout

```
neu_glance/
├── layers.py    local annotation layers — from coordinates, or from occupancy boxes
├── sources.py   layers for something on a store: a volume, an annotation source
├── serving.py   host arrays HERE and run a viewer on them; the only neuroglancer import
├── shaders.py   GLSL, and the rule for choosing one — annotation and image families
├── state.py     assemble a state, encode a URL, read one back, merge into one
└── cli.py       neu-glance
```

`layers.py` and `state.py` are **pure** — plain data in, plain data out, no store access —
and `sources.py` is the only module that reads anything. That line is deliberate: a layer
whose source is a locally served volume has no store to inspect, so state assembly must never
require one.

## Install

Part of the `neu-env` conda environment, installed editable alongside its siblings:

```bash
pip install --no-deps -e ./neu-glance
```

`--no-deps` is load-bearing across this family — see the neu-suite notes.
