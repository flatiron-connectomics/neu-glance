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

## The five subcommands

| | produces |
| --- | --- |
| `neu-glance gen` | a **state** from volumes, annotation sources and layer files |
| `neu-glance annotate` | a **layer** from coordinates or a CSV |
| `neu-glance bboxes` | a **layer** from a volume's occupancy |
| `neu-glance parse` | a URL back into its state JSON |
| `neu-glance shaders` | lists or prints the built-in annotation shaders |

The three producers share one output stage. `--format {layer,state,url}` chooses the
serialization — a bare layer to paste into a state's `layers` array, a whole state for
neuroglancer's `{}` editor, or a link carrying that state — and `--out` writes it somewhere
instead of stdout.

`--into` merges the new layers into an **existing** state, given as a URL or a JSON file. The
state's own `dimensions`, position and zoom are kept, so adding a layer does not move your
view, and a layer whose name is already taken is renamed and reported rather than silently
shadowing the one already there.

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
├── shaders.py   GLSL, and the rule for choosing one
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
