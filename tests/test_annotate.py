"""`neu-glance annotate`: a local annotation layer from coordinates the caller supplies.

The assertions that matter are the ones about *where* an annotation ends up, because a
misplaced annotation is a perfectly valid annotation: the zyx -> xyz flip, and the unit
conversions. Neither fails loudly, and both put things somewhere real but wrong.
"""

import json

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec

from neu_glance import cli, layers as _layers


@pytest.fixture
def volume(tmp_path):
    """A small anisotropic pyramid: level 1 is (1,2,2) of level 0, not (2,2,2).

    Anisotropy is the point — it is what distinguishes reading the real per-level voxel
    sizes from assuming 2**N, and (1,2,2) is the common real pyramid.
    """
    data = np.zeros((16, 32, 32), dtype=np.uint8)
    data[2:6, 4:8, 4:8] = 3
    src = str(tmp_path / "src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / "vol.zarr")
    convert(src, dst, voxel_size=(40, 8, 8), kind="segmentation", profile="local",
            chunk=(8, 8, 8), factors=[(1, 2, 2)], max_levels=2, min_dim=8,
            delete_existing=True)
    return dst


def _run(capsys, *argv):
    """Run the subcommand and return ``(layer, stderr)``."""
    assert cli.main(["annotate",*argv]) == 0
    out, err = capsys.readouterr()
    return json.loads(out), err


def _csv(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------- #
# geometry and the flip
# --------------------------------------------------------------------------- #
def test_points_from_csv_are_written_xyz(tmp_path, volume, capsys):
    """Columns are read as zyx and written as xyz. Getting this wrong mirrors the set."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n4,5,6\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path)

    assert layer["source"]["url"] == "local://annotations"
    assert [a["point"] for a in layer["annotations"]] == [[3, 2, 1], [6, 5, 4]]
    assert [a["type"] for a in layer["annotations"]] == ["point", "point"]


def test_columns_are_found_by_name_not_position(tmp_path, volume, capsys):
    """A real synapse table has its own column order and extra columns."""
    path = _csv(tmp_path, "p.csv",
                "body,x,confidence,z,note,y\n99,3,0.8,1,hi,2\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path)
    assert layer["annotations"][0]["point"] == [3, 2, 1]


def test_a_missing_column_says_which_one(tmp_path, volume):
    path = _csv(tmp_path, "p.csv", "z,y\n1,2\n")
    with pytest.raises(SystemExit, match=r"needs column\(s\) x"):
        cli.main(["annotate","--volume", volume, "--points", path])


def test_box_corners_are_sorted_but_line_direction_is_kept(tmp_path, volume, capsys):
    """A reversed bbox renders as nothing; a reversed line is a line the other way."""
    boxes = _csv(tmp_path, "b.csv", "z0,y0,x0,z1,y1,x1\n8,8,8,2,2,2\n")
    lines = _csv(tmp_path, "l.csv", "z0,y0,x0,z1,y1,x1\n8,8,8,2,2,2\n")
    layer, _ = _run(capsys, "--volume", volume, "--boxes", boxes, "--lines", lines)

    box, line = layer["annotations"]
    assert box["type"] == "axis_aligned_bounding_box"
    assert (box["pointA"], box["pointB"]) == ([2, 2, 2], [8, 8, 8])
    assert (line["pointA"], line["pointB"]) == ([8, 8, 8], [2, 2, 2])


def test_ellipsoids_carry_centre_and_radii(tmp_path, volume, capsys):
    path = _csv(tmp_path, "e.csv", "z,y,x,rz,ry,rx\n4,5,6,1,2,3\n")
    layer, err = _run(capsys, "--volume", volume, "--ellipsoids", path)
    ann = layer["annotations"][0]
    assert ann["type"] == "ellipsoid"
    assert (ann["center"], ann["radii"]) == ([6, 5, 4], [3, 2, 1])
    # radii are an extent, not a place: they must not be bounds-checked as coordinates
    assert "all inside" in err


def test_one_layer_may_mix_kinds(tmp_path, volume, capsys):
    """Local annotations allow it; a precomputed source is one type per source."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path,
                    "--box", "0,0,0,4,4,4", "--line", "1,1,1,2,2,2")
    assert [a["type"] for a in layer["annotations"]] == [
        "point", "axis_aligned_bounding_box", "line"]


# --------------------------------------------------------------------------- #
# ids, descriptions, segments
# --------------------------------------------------------------------------- #
def test_ids_and_descriptions_come_from_the_csv(tmp_path, volume, capsys):
    path = _csv(tmp_path, "p.csv",
                "id,z,y,x,description\nsyn1,1,2,3,pre T-bar\n,4,5,6,\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path, "--label", "x")
    first, second = layer["annotations"]
    assert (first["id"], first["description"]) == ("syn1", "pre T-bar")
    assert second["id"] == "x001", "rows without an id are numbered by position"
    assert "description" not in second


def test_segments_are_strings_in_a_per_relationship_array(tmp_path, volume, capsys):
    """A uint64 body id does not survive a JSON number, and the field is nested."""
    path = _csv(tmp_path, "p.csv",
                "z,y,x,segments\n1,2,3,22032970 22032971\n4,5,6,10000000000000000001\n")
    layer, err = _run(capsys, "--volume", volume, "--points", path)
    assert layer["annotations"][0]["segments"] == [["22032970", "22032971"]]
    assert layer["annotations"][1]["segments"] == [["10000000000000000001"]]
    assert "2 annotation(s) linked to body ids" in err


def test_a_segment_id_that_went_through_a_spreadsheet_is_rejected(tmp_path, volume):
    """`1.23e+18` is what Excel does to a body id, and it links to nothing."""
    path = _csv(tmp_path, "p.csv", "z,y,x,segments\n1,2,3,1.23e+18\n")
    with pytest.raises(SystemExit, match="not a whole number"):
        cli.main(["annotate","--volume", volume, "--points", path])


def test_duplicate_ids_are_refused(tmp_path, volume):
    path = _csv(tmp_path, "p.csv", "id,z,y,x\nsame,1,2,3\nsame,4,5,6\n")
    with pytest.raises(SystemExit, match="duplicate annotation id"):
        cli.main(["annotate","--volume", volume, "--points", path])


# --------------------------------------------------------------------------- #
# units: the failure mode of the whole command
# --------------------------------------------------------------------------- #
def test_scale_uses_the_real_per_level_voxel_sizes(tmp_path, volume, capsys):
    """With factors (1,2,2), a scale-1 voxel is 1 level-0 voxel in z and 2 in y/x.

    2**N would move z as well, putting every annotation in the wrong plane.
    """
    path = _csv(tmp_path, "p.csv", "z,y,x\n3,4,5\n")
    layer, err = _run(capsys, "--volume", volume, "--points", path, "--scale", "1")
    assert layer["annotations"][0]["point"] == [10, 8, 3]      # xyz: x2, x2, x1
    assert "x(1, 2, 2)" in err


def test_nm_divides_by_the_level0_voxel_size(tmp_path, volume, capsys):
    """The pyramid is 40x8x8 nm, so z does not convert like y and x."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n400,80,160\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path, "--nm")
    assert layer["annotations"][0]["point"] == [20, 10, 10]    # xyz


def test_scale_without_a_volume_is_refused(tmp_path):
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    with pytest.raises(SystemExit, match="--scale needs --volume"):
        cli.main(["annotate","--points", path, "--scale", "2",
                  "--voxel-size", "8,8,8"])


def test_coordinates_outside_the_volume_are_reported(tmp_path, volume, capsys):
    """The only check available against a unit mistake, so it has to be loud."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n99999,2,3\n")
    _, err = _run(capsys, "--volume", volume, "--points", path)
    assert "WARNING: 1 annotation(s) fall outside" in err
    assert "--scale" in err and "--nm" in err


def test_without_a_frame_the_layer_is_unitless_and_says_so(tmp_path, capsys):
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    layer, err = _run(capsys, "--points", path)
    assert layer["source"]["transform"]["outputDimensions"]["x"] == [1, ""]
    assert "unitless" in err


def test_the_frame_comes_from_the_volume(tmp_path, volume, capsys):
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path)
    dims = layer["source"]["transform"]["outputDimensions"]
    assert dims["z"] == [40e-9, "m"] and dims["x"] == [8e-9, "m"]


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def test_inline_flags_need_the_right_count(volume):
    with pytest.raises(SystemExit, match="--box needs 6"):
        cli.main(["annotate","--volume", volume, "--box", "1,2,3"])


def test_nothing_to_annotate_is_an_error(volume):
    with pytest.raises(SystemExit, match="nothing to annotate"):
        cli.main(["annotate","--volume", volume])


def test_out_writes_the_layer(tmp_path, volume, capsys):
    out = tmp_path / "layer.json"
    assert cli.main(["annotate","--volume", volume, "--point", "1,2,3",
                     "--out", str(out), "--name", "syn", "--color", "#ff0044"]) == 0
    layer = json.loads(out.read_text())
    assert layer["name"] == "syn" and layer["annotationColor"] == "#ff0044"
    assert layer["tab"] == "annotations", "so the panel opens on the clickable list"


def test_the_layer_composes_with_gen(tmp_path, volume, capsys):
    """The whole point of emitting a layer: `gen --layer` inlines it."""
    from neu_glance.state import parse_url

    out = tmp_path / "layer.json"
    cli.main(["annotate", "--volume", volume, "--point", "8,8,8",
              "--out", str(out)])
    capsys.readouterr()
    assert cli.main(["gen", "--seg", volume, "--layer", str(out)]) == 0
    state = parse_url(capsys.readouterr().out.strip().splitlines()[-1])
    kinds = [ly["type"] for ly in state["layers"]]
    assert kinds == ["segmentation", "annotation"]
    assert state["layers"][1]["annotations"][0]["point"] == [8, 8, 8]


def test_the_cli_column_spec_matches_the_layers_one():
    """The parser needs the columns at build time, so cli.py repeats them.

    Importing `layers` at parser-build time would pull neu-vol into every
    `neu-glance --help`, so the duplication stays — and this is what keeps it from drifting.
    """
    assert cli._ANN_CSV_COLUMNS == _layers.CSV_COLUMNS
    assert {k for _p, k in cli._ANN_FLAGS} == set(_layers.KINDS)


def test_occupancy_boxes_still_go_through_the_shared_builder(tmp_path, volume, capsys):
    """`bboxes` and `annotate` share local_layer/build_annotation."""
    assert cli.main(["bboxes", volume, "--no-tighten"]) == 0
    layer = json.loads(capsys.readouterr().out)
    assert layer["source"]["url"] == "local://annotations"
    ann = layer["annotations"][0]
    assert ann["type"] == "axis_aligned_bounding_box" and ann["id"] == "r00"
    assert ann["pointA"][0] <= ann["pointB"][0]
