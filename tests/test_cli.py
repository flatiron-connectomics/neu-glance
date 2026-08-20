"""The output stage the three producers share: ``--format``, ``--into``, ``--out``.

This is where the split from `neu-vol` actually changed behaviour rather than moving it, so the
assertions here are about the new contract: one state builder behind all three formats, and a
merge that leaves the incoming view alone.
"""

import json

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec

from neu_glance import cli
from neu_glance.state import parse_url


@pytest.fixture
def volume(tmp_path):
    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[2:6, 4:8, 4:8] = 3
    src = str(tmp_path / "src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / "vol.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", profile="local",
            chunk=(8, 8, 8), min_dim=8, delete_existing=True)
    return dst


def _out(capsys):
    return capsys.readouterr().out.strip()


# --------------------------------------------------------------------------- #
# --format
# --------------------------------------------------------------------------- #
def test_gen_defaults_to_a_url_and_bboxes_to_a_layer(volume, capsys):
    """Each command keeps the default its `neu-vol` predecessor had, so the break is in the
    names only and not in what a command does when you run it the same way."""
    cli.main(["gen", "--image", volume])
    assert _out(capsys).startswith("https://")

    cli.main(["bboxes", volume, "--no-tighten"])
    assert json.loads(_out(capsys))["type"] == "annotation"


def test_every_producer_can_emit_all_three_formats(volume, capsys):
    cli.main(["bboxes", volume, "--no-tighten", "--format", "layer"])
    assert "annotations" in json.loads(_out(capsys))

    cli.main(["bboxes", volume, "--no-tighten", "--format", "state"])
    state = json.loads(_out(capsys))
    assert [ly["type"] for ly in state["layers"]] == ["segmentation", "annotation"]

    cli.main(["bboxes", volume, "--no-tighten", "--format", "url"])
    assert parse_url(_out(capsys)) == state


def test_a_state_from_a_producer_includes_the_volume(volume, capsys):
    """A layer is the default and needs no volume. A whole state without one shows
    annotations floating in nothing."""
    cli.main(["annotate", "--volume", volume, "--point", "1,2,3", "--format", "state"])
    state = json.loads(_out(capsys))
    assert state["layers"][0]["type"] == "segmentation"
    assert state["layers"][1]["annotations"][0]["point"] == [3, 2, 1]   # zyx -> xyz


def test_gen_does_not_offer_format_layer(volume, capsys):
    """`gen` composes a whole state, so a bare layer is not one of its outputs. Argparse
    rejects it rather than the command failing later with a count mismatch."""
    with pytest.raises(SystemExit):
        cli.main(["gen", "--image", volume, "--format", "layer"])
    assert "invalid choice" in capsys.readouterr().err


def test_into_does_not_add_the_volume_again(tmp_path, volume, capsys):
    """The state being merged into already has whatever volumes it was built with; adding
    this one again would duplicate it or collide with its name."""
    _base, path = _base_state(tmp_path)
    cli.main(["bboxes", volume, "--no-tighten", "--into", path, "--format", "state"])
    state = json.loads(_out(capsys))
    assert [ly["type"] for ly in state["layers"]] == ["image", "annotation"]


def test_out_keeps_stdout_clean_for_every_format(tmp_path, volume, capsys):
    path = str(tmp_path / "deep" / "x.json")
    cli.main(["bboxes", volume, "--no-tighten", "--out", path])
    assert capsys.readouterr().out == ""
    with open(path) as f:
        assert json.load(f)["type"] == "annotation"


# --------------------------------------------------------------------------- #
# --into
# --------------------------------------------------------------------------- #
def _base_state(tmp_path, name="base.json", extra=None):
    state = {
        "dimensions": {d: [4e-9, "m"] for d in "xyz"},
        "position": [1, 2, 3],
        "crossSectionScale": 7,
        "projectionScale": 99,
        "layout": "xy",
        "layers": [{"type": "image", "name": "em", "source": "precomputed://s3://b/em"}],
    }
    state.update(extra or {})
    path = tmp_path / name
    path.write_text(json.dumps(state))
    return state, str(path)


def test_into_appends_and_keeps_the_existing_view(tmp_path, volume, capsys):
    """"Add a layer to my view" must not move the view — and must not re-derive
    `dimensions`, since a state whose dimensions disagree with its layers loads fine and
    puts everything in the wrong place."""
    base, path = _base_state(tmp_path)
    cli.main(["annotate", "--volume", volume, "--point", "1,2,3", "--into", path])
    state = parse_url(_out(capsys))

    assert [ly["name"] for ly in state["layers"]] == ["em", "annotations"]
    for key in ("dimensions", "position", "crossSectionScale", "projectionScale", "layout"):
        assert state[key] == base[key], key


def test_into_accepts_a_url_as_well_as_a_file(tmp_path, volume, capsys):
    """So it works on a link copied straight out of the browser, with no `parse` step."""
    from neu_glance.state import state_url

    base, _path = _base_state(tmp_path)
    cli.main(["bboxes", volume, "--no-tighten", "--into", state_url(base)])
    state = parse_url(_out(capsys))
    assert len(state["layers"]) == 2
    assert state["position"] == base["position"]


def test_into_renames_a_clashing_layer_and_says_so(tmp_path, volume, capsys):
    """Neuroglancer keys a layer by name, so two layers sharing one is a collision rather
    than a duplicate — and the second silently shadows the first."""
    _base, path = _base_state(
        tmp_path, extra={"layers": [{"type": "annotation", "name": "annotations"}]})
    cli.main(["annotate", "--volume", volume, "--point", "1,2,3", "--into", path])
    captured = capsys.readouterr()
    state = parse_url(captured.out.strip())
    assert [ly["name"] for ly in state["layers"]] == ["annotations", "annotations-2"]
    assert "renamed incoming layer" in captured.err


def test_into_implies_a_url_but_state_still_works(tmp_path, volume, capsys):
    _base, path = _base_state(tmp_path)
    cli.main(["bboxes", volume, "--no-tighten", "--into", path])
    assert _out(capsys).startswith("https://")

    cli.main(["bboxes", volume, "--no-tighten", "--into", path, "--format", "state"])
    assert json.loads(_out(capsys))["layout"] == "xy"


def test_into_with_format_layer_is_refused(tmp_path, volume):
    _base, path = _base_state(tmp_path)
    with pytest.raises(SystemExit, match="--into merges"):
        cli.main(["bboxes", volume, "--no-tighten", "--into", path, "--format", "layer"])


def test_into_needs_a_state_not_a_bare_layer(tmp_path, volume):
    layer = tmp_path / "layer.json"
    layer.write_text(json.dumps({"type": "annotation", "name": "a"}))
    with pytest.raises(SystemExit, match="not a neuroglancer state"):
        cli.main(["bboxes", volume, "--no-tighten", "--into", str(layer)])


def test_into_needs_no_voxel_size_because_the_state_has_dimensions(tmp_path, capsys):
    """Without --into this same invocation is refused for having no frame."""
    _base, path = _base_state(tmp_path)
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "boxes", "annotations": []}))

    with pytest.raises(SystemExit, match="no voxel size available"):
        cli.main(["gen", "--layer", str(box)])
    assert cli.main(["gen", "--layer", str(box), "--into", path]) == 0


def test_into_can_override_the_view_when_asked(tmp_path, volume, capsys):
    _base, path = _base_state(tmp_path)
    cli.main(["bboxes", volume, "--no-tighten", "--into", path,
              "--layout", "3d", "--position", "9,9,9"])
    state = parse_url(_out(capsys))
    assert state["layout"] == "3d"
    assert state["position"] == [9, 9, 9]


def test_into_does_not_mutate_the_file_it_read(tmp_path, volume, capsys):
    base, path = _base_state(tmp_path)
    cli.main(["bboxes", volume, "--no-tighten", "--into", path])
    capsys.readouterr()
    with open(path) as f:
        assert json.load(f) == base


# --------------------------------------------------------------------------- #
# parse and shaders
# --------------------------------------------------------------------------- #
def test_parse_round_trips_a_url(tmp_path, volume, capsys):
    cli.main(["gen", "--image", volume])
    url = _out(capsys)
    cli.main(["parse", url])
    assert json.loads(_out(capsys)) == parse_url(url)


def test_parse_can_list_just_the_layers(volume, capsys):
    cli.main(["gen", "--seg", volume])
    url = _out(capsys)
    cli.main(["parse", url, "--layers"])
    rows = [ln.split("\t") for ln in _out(capsys).splitlines()]
    assert rows == [["vol.zarr", "segmentation"]]


def test_parse_survives_a_layers_entry_that_is_not_an_object(tmp_path, capsys):
    """A published reference state carries a bare string in `layers` as a comment, so this
    must not assume every entry is a dict."""
    from neu_glance.state import state_url

    url = state_url({"layers": ["# a note", {"type": "image", "name": "em"}]})
    cli.main(["parse", url, "--layers"])
    assert "not a layer object" in _out(capsys)


def test_parse_rejects_something_that_is_not_a_state_url():
    with pytest.raises(SystemExit, match="no '#!' fragment"):
        cli.main(["parse", "https://example.org/"])


def test_shaders_lists_and_prints(capsys):
    cli.main(["shaders"])
    listing = _out(capsys)
    assert "synapse" in listing and "conf_pre" in listing

    cli.main(["shaders", "synapse"])
    assert "void main()" in _out(capsys)


def test_an_unknown_shader_name_is_refused():
    with pytest.raises(SystemExit, match="no built-in shader"):
        cli.main(["shaders", "nope"])


# --------------------------------------------------------------------------- #
# the import contract
# --------------------------------------------------------------------------- #
def test_importing_the_package_does_not_pull_in_neu_vol():
    """`cli` reads __version__ from __init__, so an eager import there would make every
    `neu-glance --help` pay for tensorstore. The lazy exports are what keep it cheap."""
    import subprocess
    import sys

    code = ("import sys, neu_glance; "
            "assert 'neu_vol' not in sys.modules; "
            "assert 'numpy' not in sys.modules; "
            "assert neu_glance.build_state")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_building_the_parser_does_not_pull_in_neu_vol():
    import subprocess
    import sys

    code = ("import sys; from neu_glance import cli; cli.build_parser(); "
            "assert 'neu_vol' not in sys.modules, "
            "'building the parser imported neu_vol'")
    subprocess.run([sys.executable, "-c", code], check=True)
