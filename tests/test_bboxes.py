"""`em-ngl bboxes`: a layer from where a sparse volume's data actually is.

The occupancy analysis itself lives in em-volume-tools and is tested there; what these check
is the turn from boxes into a layer. The weight is on the zyx/xyz flip, because getting it
wrong mirrors every annotation through the z=x diagonal and still produces a layer that loads.
"""

import json
import os

import numpy as np
import pytest

from em_volume_tools import convert
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import zarr3_create_spec

from em_ngl import cli
from em_ngl.layers import boxes_layer, output_dimensions, render


def _sparse(tmp_path, name, *, profile, chunk=(8, 8, 8)):
    """A real two-level volume, 32^3, holding two separated 8^3 labeled blocks.

    Built through `convert` so the levels, `info`/`zarr.json` and the elision of all-fill
    chunks are the production ones — occupancy here means what it means in a real run.
    """
    seg = np.zeros((32, 32, 32), np.uint64)
    seg[0:8, 0:8, 0:8] = 3                       # cell (0,0,0)
    seg[16:24, 24:32, 8:16] = 4                  # cell (2,3,1)
    seg[16:24, 24:32, 8:16][0, 0, 0] = 5         # a second label, to count
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, seg.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in seg.shape), seg)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", profile=profile,
            chunk=chunk, factors=[(2, 2, 2)], min_dim=8, delete_existing=True)
    return dst


# --------------------------------------------------------------------------- #
# the JSON
# --------------------------------------------------------------------------- #
def test_layer_coordinates_are_flipped_to_xyz():
    """zyx in memory, xyz on the wire. Reversed, every box is mirrored."""
    regions = [{"lo": (1, 2, 3), "hi": (4, 5, 6), "n_labels": 7}]
    layer = boxes_layer(regions, {d: [8e-9, "m"] for d in "xyz"})
    ann = layer["annotations"][0]
    assert ann["pointA"] == [3, 2, 1]
    assert ann["pointB"] == [6, 5, 4]


def test_points_sit_at_the_centre():
    regions = [{"lo": (0, 0, 0), "hi": (10, 20, 30), "n_labels": None}]
    ann = boxes_layer(regions, {}, kind="point")["annotations"][0]
    assert ann["type"] == "point"
    assert ann["point"] == [15.0, 10.0, 5.0]         # xyz


def test_layer_declares_its_own_dimensions():
    """So the layer can be pasted into any state of the volume, whatever the viewer is
    displaying in — the coordinates are read in the layer's frame, not the global one."""
    layer = boxes_layer([], {"x": [8e-9, "m"]})
    assert layer["source"]["url"] == "local://annotations"
    assert layer["source"]["transform"]["outputDimensions"] == {"x": [8e-9, "m"]}
    assert layer["tab"] == "annotations"


def test_voxel_size_becomes_metres():
    dims, warning = output_dimensions((8.0, 4.0, 4.0), "nm")     # zyx
    assert warning is None
    assert dims == {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [8e-9, "m"]}


@pytest.mark.parametrize("units", [None, "furlong"])
def test_unusable_units_warn_instead_of_inventing_a_scale(units):
    """A wrong scale would place the boxes somewhere plausible and wrong; unitless at
    least fails visibly, and the warning names the flag that fixes it."""
    voxel = None if units is None else (8.0, 8.0, 8.0)
    dims, warning = output_dimensions(voxel, units)
    assert dims == {d: [1, ""] for d in "xyz"}
    assert "--voxel-size" in warning


def test_render_stays_valid_json_with_one_line_per_annotation():
    """Pins a bug that produced a loadable-looking layer with no annotations in it.

    The renderer swaps each annotation for a placeholder, dumps, then substitutes the
    one-line form back. A placeholder built from control characters is re-escaped by
    json.dumps, so every substitution missed and the annotations shipped as the
    placeholder *strings* — valid JSON, twelve entries, none of them an annotation.
    """
    regions = [{"lo": (0, 0, 0), "hi": (8, 8, 8), "n_labels": 2},
               {"lo": (8, 8, 8), "hi": (16, 16, 16), "n_labels": 3}]
    layer = boxes_layer(regions, {d: [8e-9, "m"] for d in "xyz"})
    text = render(layer)

    assert "\\u0000" not in text and "em_ngl_annotation" not in text
    back = json.loads(text)
    assert [a["type"] for a in back["annotations"]] == \
        ["axis_aligned_bounding_box"] * 2
    assert back["annotations"][0]["pointB"] == [8, 8, 8]
    for ann in back["annotations"]:
        assert sum(1 for line in text.splitlines()
                   if line.strip().startswith(f'{{"type": "axis_aligned'
                                              f'_bounding_box", "id": "{ann["id"]}"')) == 1


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_stdout_is_only_json_so_it_can_be_redirected(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    assert cli.main(["bboxes", dst, "--tighten-level", "1"]) == 0
    captured = capsys.readouterr()
    layer = json.loads(captured.out)          # the whole of stdout, or this raises
    assert layer["type"] == "annotation"
    assert len(layer["annotations"]) == 2
    assert "region(s)" in captured.err, "the summary belongs on stderr"


def test_out_writes_the_file_and_leaves_stdout_empty(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    out = str(tmp_path / "layer.json")
    cli.main(["bboxes", dst, "--tighten-level", "1", "--out", out])
    assert capsys.readouterr().out == ""
    with open(out) as f:
        assert len(json.load(f)["annotations"]) == 2


def test_a_state_is_loadable_and_carries_the_volume_layer(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    cli.main(["bboxes", dst, "--tighten-level", "1", "--format", "state"])
    state = json.loads(capsys.readouterr().out)
    assert [lyr["type"] for lyr in state["layers"]] == ["segmentation", "annotation"]
    assert state["layers"][0]["source"] == f"precomputed://{dst}"
    assert state["selectedLayer"]["layer"] == state["layers"][1]["name"]
    # the view opens on the data rather than at the origin of an empty frame
    assert state["position"] != [0, 0, 0]


def test_a_state_names_the_zarr_driver_for_a_zarr_volume(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local")
    cli.main(["bboxes", dst, "--tighten-level", "1", "--format", "state"])
    state = json.loads(capsys.readouterr().out)
    assert state["layers"][0]["source"] == f"zarr://{dst}"


def test_label_and_name_flow_through(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    cli.main(["bboxes", dst, "--tighten-level", "1", "--label", "gt",
              "--name", "gt-chunks", "--color", "#ff0000"])
    layer = json.loads(capsys.readouterr().out)
    assert layer["name"] == "gt-chunks"
    assert layer["annotationColor"] == "#ff0000"
    assert [a["id"] for a in layer["annotations"]] == ["gt00", "gt01"]


def test_an_empty_volume_reports_nothing_to_annotate(tmp_path, capsys):
    """Distinguishable from a failure: exit 1, and it says why."""
    from em_volume_tools.ops.create import create_volume

    dst = str(tmp_path / "empty")
    create_volume(dst, format="precomputed", shape=(32, 32, 32), dtype="uint64",
                  voxel_size=(8, 8, 8), chunk=(8, 8, 8), kind="segmentation")
    assert cli.main(["bboxes", dst]) == 1
    assert "the volume is empty" in capsys.readouterr().err


def test_a_missing_volume_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="no volume found"):
        cli.main(["bboxes", str(tmp_path / "nope")])


def test_out_goes_through_the_kvstore_not_open(tmp_path, capsys):
    """So `--out s3://...` works at all.

    Pinned with a local path whose parent does not exist: the file driver creates it,
    while `open()` would raise. That is the same code path a remote --out takes, and it
    is the only part of it a test can exercise without a bucket.
    """
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    out = str(tmp_path / "does" / "not" / "exist" / "layer.json")
    assert cli.main(["bboxes", dst, "--tighten-level", "1", "--out", out]) == 0
    with open(out) as f:
        assert len(json.load(f)["annotations"]) == 2
    assert capsys.readouterr().out == "", "--out must keep stdout clean"


def test_bboxes_is_wired_to_the_subcommand():
    assert cli._parse_args(["bboxes", "v"]).func is cli.cmd_bboxes
    assert not os.path.exists("v"), "parsing must not touch the volume"
