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
| `neu-glance help` | what `--help` prints, as a subcommand |

The three producers share one output stage. `--format {layer,state,url}` chooses the
serialization — a bare layer to paste into a state's `layers` array, a whole state for
neuroglancer's `{}` editor, or a link carrying that state — and `--out` writes it somewhere
instead of stdout.

`--into` merges the new layers into an **existing** state, given as a URL or a JSON file. The
state's own `dimensions`, position and zoom are kept, so adding a layer does not move your
view, and a layer whose name is already taken is renamed and reported rather than silently
shadowing the one already there.

**Put a neuroglancer URL in single quotes.** Inside double quotes an interactive bash expands
`!`, and neuroglancer uses it twice: after the `#` that opens the state, and before a hidden
segment's id (`"!12345"`). Losing the first gives a link starting `#7B` and an error; the
second can silently splice an old command from your history into the segment list.

```bash
neu-glance gen --into 'https://neuroglancer-demo.appspot.com/#!%7B...' --layer boxes.json
```

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

**A probability map is drawn in one colour with the opacity carrying the value** — red by
default, so the EM under it stays visible where the model is unsure and is progressively
covered where it is confident. `color=` picks another, which is how two maps go into one
viewer distinguishably; the viewer's own control changes it afterwards, and
`shader="colormap"` restores the older two-colour gradient.

```python
srv.add_layer(ServedLayer.from_array(pred, "probability", voxel_size=(40, 8, 8)),
              color="green")     # or "#00ff00", (0, 1, 0), (0, 255, 0)
```

Colours take a name, `#rrggbb`, or a 3-sequence (floats 0..1, ints 0..255). A short list of
names — red, green, blue, cyan, magenta, yellow, orange, purple, lime, white, black, gray —
needs nothing installed; anything else matplotlib names (`"forestgreen"`, `"tab:blue"`)
works where matplotlib happens to be there, and where it is not you are asked for hex or a
tuple. **matplotlib is not a dependency and this does not make it one** — every built-in
name carries matplotlib's own value for it, so installing it later cannot change what a
colour means.

The colour is set as a `shaderControls` override rather than by generating a shader with a
different default, so every layer carries the same code and the viewer's panel shows what
was set. It applies to a shader that has a single colour: `grayscale` has none and `rgb` has
three, and asking for one there **raises** rather than setting a control neuroglancer would
ignore in silence.

A `neu_lib.Piece` is itself a layer input, to `serve` and `add_layer` alike — it already
carries the frame, the kind and a name, so a crop read (or cleaned) with neu-vol goes
straight to a viewer:

```python
serve([neu_vol.read_piece("gt.h5:/vol_03700", "segmentation")])
```

A piece whose `kind` is `None` needs `ServedLayer.from_piece(piece, kind)`, since neither
call has a `kind=` to pass.

The constructors follow one rule: **infer what the source records, require what it
does not.** A frame, a dataset name and the channel axis are all written down — in an HDF5
file's attributes, a precomputed `info`, or the array's own rank — so reading them is not
guessing, and dropping them is the silent failure. `kind` is asked for, because an HDF5 file
has nowhere agreed-on to record it and reading it off the dtype is the mistake neuroglancer
itself makes. A volume that records `info["type"]` is the exception, and then `from_volume`
needs nothing.

**The browser follows the state, so a viewer already open can grow layers.** Nothing is
re-served and the URL does not change — `srv.add_layer()` hosts one more array, taking the
same three inputs (and going through the same builder) that `serve` does:

```python
srv = (serve([em])
       .add_layer(ServedLayer.from_hdf5("gt.h5", "/vol_03700", "segmentation"))
       .add_layer(ServedLayer.from_array(pred, "probability", voxel_size=(40, 8, 8)),
                  name="prediction"))
```

`add_layer` returns the server, so the calls chain. It does not move the view — adding a
layer must not change where you are looking — so a layer whose data is elsewhere lands off
screen, and `srv.bounds("prediction")` is how to find it. A name already taken is renamed,
not replaced; the resolved name is read off the viewer, `list(srv.volumes)[-1]` being the
one just added.

`srv.boxes()` closes the loop: pick a region in the viewer, hand it straight to
`--crop-bbox`, `extract_roi` or `neu-vol write`. It needs an annotation layer to draw in,
which is **opt-in** — because a viewer that opens with a layer nobody asked for reads as a
bug. `srv.annotate()` adds one to a viewer that is already open, which is usually where you
realise you want it; `serve(..., annotations=True)` and `--annotate` open with one already
there. (The CLI flag is `--annotate`, not `--annotations`: `gen --annotations SOURCE` already
means a precomputed annotation source to load.)

```python
srv.annotate()               # a layer to draw in, box tool armed and selected
                             # ... ctrl+mousedown0 in the viewer to place boxes
srv.boxes()                  # -> [((lo), (hi)), ...]
srv.annotate(tool="point")   # same layer, now placing points — nothing drawn is lost
srv.points()
```

The browser follows the state, so the layer appears in the tab you already have open —
nothing is re-served. Calling it again keeps what is already drawn, and `tool` is `"box"`
or `"point"` only: those are the two `boxes()` and `points()` can read back, and arming a
line or an ellipsoid would be drawing that never comes back.

Dragging a box to exact corners in neuroglancer is fiddly; clicking a point at each corner
is not. `srv.enclose()` is the conversion — **the containing box of everything drawn**,
added to the layer and returned as a `neu_lib.BBox` in whole voxels:

```python
srv.annotate(tool="point")             # click a point at each extreme
box = srv.enclose(margin=(1, 8, 8),    # grown per axis, zyx, in voxels
                  clip="volume",       # ... but not off the end of the data
                  replace=True)        # the points were scaffolding; drop them
lo, hi = box                           # a BBox unpacks like the pair it is
```

Everything in the layer goes in — points, boxes, and the lines, ellipsoids and polylines
the viewer's own toolbar can place. So the other half of it is fixing a box that is nearly
right: draw roughly, click a point where it should have reached, `enclose(replace=True)`.
Corners round **outward** where `boxes()` rounds to nearest, since a box that exists to
contain things must not round in past the point that put it there. With `replace=False` the
new box is itself an annotation, so calling again encloses *it* too and a margin grows the
box each time — which is what "enclose what is drawn" means, and the reason `replace=True`
is the usual call.

A margin reaches past the end of the array happily, so `clip=` bounds the result against
the two extents the server can see for itself:

```python
srv.bounds()                  # the served arrays' extent — clip="volume"
srv.data_bounds()             # where their non-zero voxels are — clip="data"
srv.bounds("seg")             # one layer, for clip=srv.bounds("seg")
```

Both are `BBox`es in viewer voxels, unioned over the served layers or taken one by name, and
both read each layer's own recorded voxel size — so a second layer served at a coarser scale
converts rather than being assumed to share the first one's grid. `clip` also takes a `BBox`
or a plain `(lo, hi)` pair. It is **off by default**: `enclose` reads a viewer somebody is
drawing in, and handing back a smaller box than the one now on their screen is worse than
handing back what they asked for — so unclipped it warns when the box starts below zero, and
a clip that leaves nothing raises rather than returning an empty box.

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

## Animating a state

A `Timeline` wraps one base state — the link you tuned in the browser — and a list of tweens
over it. Each tween names one property, a window in time and an easing; everything no tween
mentions holds its base value.

```python
from neu_glance import load_state
from neu_glance.animate import Timeline
from neu_glance.rendering import record

tl = Timeline(load_state("scene.json"), fps=30)

# Neuroglancer's objectAlpha is per LAYER and there is no per-segment alpha, so an object
# that fades on its own has to be a layer on its own. split() does that, and hands back
# {segment_id: layer_name} so groups are written against segment ids.
neurons = tl.split("my-segmentation", prefix="neuron")
rois    = tl.split("my-rois",         prefix="roi")

tl.tween(layer="my-head-mask", at=1, seconds=3, ease="out",
         meshSilhouetteRendering=10)                    # opaque shell -> transparent one

tl.group("chain-1", [neurons["3832372"], rois["20"], rois["30"]])
tl.group("chain-2", [neurons["10063300"], rois["26"]])
tl.sequence(["chain-1", "chain-2"], start=4, seconds=1.2, stagger=0.8)

tl.view("https://neuroglancer-demo.appspot.com/#!...", at=6, seconds=9)

record(tl, "frames/", size=(1920, 1080))                # prints a URL; open it and wait
```

### Moving the camera

Three ways, which compose:

```python
tl.view(url, at=6, seconds=9)          # ABSOLUTE: fly it in the browser, paste the link
tl.orbit(360, at=6, seconds=10)        # RELATIVE: a turntable about the screen's vertical
tl.orbit(180, axis="z", at=6, seconds=8)   # ...or about a volume axis, reproducibly
tl.zoom(0.5, at=6, seconds=10)         # RELATIVE: half the scale = twice as close
tl.frame_on(box, at=2, seconds=3)      # point at an object and fit it
```

**`orbit` is not sugar for two `view()` calls.** Slerp takes the shortest arc between two
orientations, so a 360° turn is a no-op — the two ends are the same rotation — and 200°
silently becomes 160° the other way. Orbit varies the *angle*, so any rotation works. Verified
against a real render: 0° and 360° come back pixel-identical.

`frame_on` takes `(lo_zyx, hi_zyx)` in nanometres by default (the suite's model space), and
converts to the viewer's own voxels using the state's `dimensions`. Boxes can come from
anywhere; `sources.segment_boxes(volume, ids)` reads them from the segments' meshes, which
needs `neu-glance[framing]`.

```python
boxes = neu_glance.sources.segment_boxes(volume, ["3832372"])
tl.frame_on(boxes["3832372"], at=2, seconds=3)
tl.orbit(360, at=5, seconds=12)        # ...then turn around it
```

`record` prints the `ffmpeg` line to turn the frames into an mp4 and does not run it — a
render takes hours and an encode takes seconds, so a failed encode should cost nothing, and
`ffmpeg` is often a module rather than something on `PATH`.

Four things worth knowing:

- **`view()` takes a URL.** Framing a 3D view is done by dragging, not by writing
  quaternions, so the intended loop is to fly the camera in the browser, copy the link, and
  paste it. Only the camera keys are taken — the target's layers and layout are ignored.
- **The cursor moves only on `hold()`.** Two tweens written one after the other run
  *together*, because fading while moving is the common case.
- **Interrupting is safe and re-running resumes.** Frames are written atomically and a
  resumed render checks the PNG signature, not just that a file exists.
- **`neu_glance.animate` is pure** — stdlib only, no neuroglancer — so a timeline can be
  built, sampled and tested with nothing installed. `tl.at(4.5)` is just a state dict, and
  `state_url` turns any moment into a link you can open.

Everything in `animate` is deliberately ours rather than `neuroglancer.ViewerState.interpolate`,
which in 2.41.2 raises `AttributeError` for a state carrying any layer (`Layer.interpolate`
reads `layer_position`; the property is `localPosition`), covers neither
`meshSilhouetteRendering` nor `segmentColors`, and materialises orientation keys the input
never had.

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
├── animate.py    a timeline of tweens over one state, and the interpolation under it
├── layers.py     local annotation layers — from coordinates, or from occupancy boxes
├── rendering.py  drive a viewer frame by frame and write PNGs
├── sources.py    layers for something on a store: a volume, an annotation source
├── serving.py    host arrays HERE and run a viewer on them
├── shaders.py    GLSL, and the rule for choosing one — annotation and image families
├── state.py      assemble a state, encode a URL, read one back, merge into one, split a layer
└── cli.py        neu-glance
```

`layers.py`, `state.py` and `animate.py` are **pure** — plain data in, plain data out, no
store access — and `sources.py` is the only module that reads anything. That line is
deliberate: a layer whose source is a locally served volume has no store to inspect, so state
assembly must never require one, and a timeline's interpolation stays covered by a CI that
installs no extras.

`serving.py` and `rendering.py` are the two that need a viewer. **Neither imports
`neuroglancer` at module scope** — every such import is inside the function that needs it,
which is what keeps `import neu_glance` and `neu-glance --help` off that import graph. A test
checks the indentation rather than the filename, because "only `serving.py` imports it" was
only ever a proxy for the rule that matters.

## Install

Part of the `neu-env` conda environment, installed editable alongside its siblings:

```bash
pip install --no-deps -e ./neu-glance
```

`--no-deps` is load-bearing across this family — see the neu-suite notes.
