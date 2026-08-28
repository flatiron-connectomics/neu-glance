"""Hosting arrays locally: `neu-glance serve` and :mod:`neu_glance.serving`.

Split deliberately into two halves. The first needs **no** `neuroglancer` — it is the CI
condition, where the wheel is installed `--no-deps` and the `serve` extra never is, so the
parser must still build and `--help` must still work. The second skips without it.

Three failures these exist to prevent, none of which raise anything on the way out:

- a **uint8 label array** served as an image, because neuroglancer guesses segmentation
  only for uint16/32/64 — losing the colour hashing and the selection UI;
- a **dropped origin**, which puts a crop at nm zero instead of on top of the volume it
  came from;
- **`encoding="raw"`**, whose `ndarray.tostring` was removed in NumPy 2.0 and which fails
  in the browser, minutes after everything here has reported success.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# no neuroglancer needed — the CI condition
# --------------------------------------------------------------------------- #
def test_the_parser_builds_without_neuroglancer():
    """`serve` must not make `neu-glance --help` need a viewer bundle.

    Enforced elsewhere too (`test_cli.py` asserts the import graph), but this is the one
    that names the subcommand at fault if it regresses.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "sys.modules['neuroglancer'] = None;"
        "from neu_glance.cli import build_parser, _parse_args;"
        "p = build_parser();"
        "a = _parse_args(['serve', '--seg', 'x.h5']);"
        "assert a.func.__name__ == 'cmd_serve', a.func;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_a_missing_extra_names_the_extra():
    """The failure lands at *run* time, so its message is the only guidance there is."""
    import neu_glance.serving as serving

    real = __import__

    def blocked(name, *a, **k):
        if name == "neuroglancer":
            raise ImportError("blocked")
        return real(name, *a, **k)

    import builtins

    saved = builtins.__import__
    builtins.__import__ = blocked
    try:
        with pytest.raises(serving.ServeProblem, match=r"neu-glance\[serve\]"):
            serving._neuroglancer()
    finally:
        builtins.__import__ = saved


def test_encoding_is_npz_and_never_raw():
    """`encode_raw` is the one function still calling `ndarray.tostring`, removed in
    NumPy 2.0. It is reached only if the encoding says so, and the declared encoding is
    what the client requests — so pinning this constant is the whole fix."""
    from neu_glance.serving import ENCODING

    assert ENCODING == "npz"


def test_the_cli_shader_names_agree_with_the_registry():
    """The tuple is duplicated into `cli.py` so the parser needs no import at build time —
    the same trade `_ANN_CSV_COLUMNS` makes, and it needs the same check."""
    from neu_glance.cli import _IMAGE_SHADER_NAMES
    from neu_glance.shaders import IMAGE_SHADERS

    assert set(_IMAGE_SHADER_NAMES) == set(IMAGE_SHADERS) | {"none"}


# --------------------------------------------------------------------------- #
# image shaders
# --------------------------------------------------------------------------- #
def test_the_default_shader_follows_kind_and_channel_count():
    from neu_glance.shaders import image_shader

    assert "invlerp" in image_shader(None, kind="image")
    assert "threshold" in image_shader(None, kind="probability")
    # three channels beats the kind default: showing one of three and calling it the
    # picture is the more surprising choice
    assert "getDataValue(2)" in image_shader(None, kind="probability", channels=3)
    assert image_shader("none") is None


def test_a_channel_count_mismatch_raises():
    """`rgb` on one channel reads past the end and fails to compile; a one-channel shader
    on three shows a third of the data with nothing to say so."""
    from neu_glance.shaders import ShaderProblem, image_shader

    with pytest.raises(ShaderProblem, match="reads 3 channel"):
        image_shader("rgb", channels=1)
    with pytest.raises(ShaderProblem, match="reads 1 channel"):
        image_shader("grayscale", channels=3)
    with pytest.raises(ShaderProblem, match="unknown image shader"):
        image_shader("nope")


def test_the_probability_shader_discards_below_rather_than_keeping_above():
    """NaN fails every comparison, so `discard if v < t` leaves an unscored voxel visible
    at every threshold while the inverted form would hide it at all of them. A model that
    declined to predict somewhere writes NaN there."""
    from neu_glance.shaders import COLORMAP_SHADER

    assert "if (v < threshold) discard;" in COLORMAP_SHADER
    assert ">= threshold" not in COLORMAP_SHADER


def test_annotation_and_image_shaders_stay_separate_registries():
    """An annotation shader names `prop_`s that a source must declare; an image shader
    names a channel count. `pick_shader`'s refusal logic applies only to the first."""
    from neu_glance.shaders import IMAGE_SHADERS, SHADERS

    assert not set(SHADERS) & set(IMAGE_SHADERS)
    assert all("properties" in e for e in SHADERS.values())
    assert all("channels" in e for e in IMAGE_SHADERS.values())


# --------------------------------------------------------------------------- #
# the rest needs neuroglancer
# --------------------------------------------------------------------------- #
ng = pytest.importorskip("neuroglancer", reason="the serve extra is not installed")

from neu_glance.serving import (ServedLayer, ServeProblem,  # noqa: E402
                                _coordinate_space, _volume_type, _voxel_offset, serve)


@pytest.fixture
def frame():
    from neu_lib import Frame

    return Frame(voxel_size_nm=(40.0, 8.0, 8.0), origin_nm=(400.0, 160.0, 240.0))


@pytest.fixture
def stop_server():
    yield
    ng.server.stop()


def _labels(shape=(8, 16, 16), dtype="uint8"):
    return (np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) % 7)


# --- the layer spec --------------------------------------------------------- #
def test_a_bad_kind_or_shape_is_refused_before_anything_is_served():
    with pytest.raises(ServeProblem, match="kind must be"):
        ServedLayer(_labels(), kind="labels")
    with pytest.raises(ServeProblem, match="3-D or 4-D"):
        ServedLayer(np.zeros((4, 4)), kind="image")
    with pytest.raises(ServeProblem, match="expected an array"):
        ServedLayer([1, 2, 3], kind="image")


# --- the frame, which is the silent one ------------------------------------- #
def test_the_coordinate_space_carries_the_frames_nm_voxel_size(frame):
    space = _coordinate_space(ng, frame)
    assert list(space.names) == ["z", "y", "x"]
    # neuroglancer normalizes to base SI on construction, so 40 nm is stored as 4e-8 m
    assert space.to_json() == {"z": [4e-08, "m"], "y": [8e-09, "m"], "x": [8e-09, "m"]}


def test_a_channel_axis_becomes_a_LOCAL_dimension(frame):
    """The trailing `^` is what hands all three channels to the shader. Without it
    neuroglancer offers a slider and shows one at a time."""
    space = _coordinate_space(ng, frame, channel_axis=True)
    assert list(space.names) == ["c^", "z", "y", "x"]
    assert space.to_json()["c^"] == [1.0, ""], "a channel is unitless, not 1 nm"


def test_the_origin_becomes_a_voxel_offset(frame):
    assert _voxel_offset(frame) == [10, 20, 30]
    # no origin, nothing to offset
    from neu_lib import Frame

    assert _voxel_offset(Frame(voxel_size_nm=(8, 8, 8), origin_nm=(0, 0, 0))) is None
    assert _voxel_offset(None) is None


def test_a_channel_axis_gets_a_zero_offset(frame):
    assert _voxel_offset(frame, channel_axis=True) == [0, 10, 20, 30]


def test_a_non_integral_origin_raises_rather_than_rounding():
    """Rounding would shift the volume by up to half a voxel against the thing it is meant
    to overlay, which nothing downstream can detect."""
    from neu_lib import Frame

    with pytest.raises(ServeProblem, match="not a whole number"):
        _voxel_offset(Frame(voxel_size_nm=(8, 8, 8), origin_nm=(4, 0, 0)))


# --- the uint8 trap -------------------------------------------------------- #
def test_kind_decides_volume_type_never_the_dtype():
    """neuroglancer guesses segmentation only for rank-3 uint16/32/64, so a **uint8**
    label array is guessed as an IMAGE — which averages label ids on downsample and loses
    both the colour hashing and the selection UI, silently."""
    assert _volume_type(ServedLayer(_labels(dtype="uint8"), kind="segmentation")) \
        == "segmentation"
    assert _volume_type(ServedLayer(_labels(dtype="uint64"), kind="image")) == "image"
    assert _volume_type(ServedLayer(_labels(), kind="probability")) == "image"

    # ...and it survives all the way onto the served volume
    guessed = ng.LocalVolume(_labels(dtype="uint8"),
                             ng.CoordinateSpace(names=["z", "y", "x"], units="nm",
                                                scales=[8, 8, 8]))
    assert guessed.volume_type == "image", "the guess this overrides"


# --- serving --------------------------------------------------------------- #
def test_serving_sets_the_viewers_own_dimensions(frame, stop_server):
    """A `dimensions` block that disagrees with the data loads cleanly and puts every
    layer in the wrong place — this package's oldest silent failure. Spatial only: a
    channel axis is local to the layer, not something to navigate."""
    server = serve([ServedLayer(np.zeros((3, 4, 4, 4)), kind="probability",
                                channel_axis=True, frame=frame, name="p")])
    state = server.state()
    assert list(state["dimensions"]) == ["z", "y", "x"]


def test_each_kind_gets_the_right_layer_type_and_shader(frame, stop_server):
    server = serve([
        ServedLayer(_labels(), kind="image", name="em", frame=frame),
        ServedLayer(_labels(), kind="segmentation", name="seg", frame=frame),
        ServedLayer(_labels().astype("float32"), kind="probability", name="p",
                    frame=frame),
    ])
    by_name = {lyr["name"]: lyr for lyr in server.state()["layers"]}
    assert by_name["em"]["type"] == "image" and by_name["em"].get("shader")
    assert by_name["p"]["type"] == "image" and "threshold" in by_name["p"]["shader"]
    # A segmentation gets NO shader: the viewer's label hashing beats anything written
    # here, and a shader would defeat the selection UI it exists for.
    assert by_name["seg"]["type"] == "segmentation"
    assert not by_name["seg"].get("shader")
    assert {n: v.volume_type for n, v in server.volumes.items()} == {
        "em": "image", "seg": "segmentation", "p": "image"}


def test_a_served_source_is_a_python_url(frame, stop_server):
    """And therefore not shareable — which is why `serve` has no --format url."""
    server = serve([ServedLayer(_labels(), name="a", frame=frame)])
    source = server.state()["layers"][0]["source"]
    source = source[0] if isinstance(source, list) else source
    url = source["url"] if isinstance(source, dict) else source
    assert url.startswith("python://volume/")


def test_a_name_collision_is_renamed_not_shadowed(frame, stop_server):
    """neuroglancer keys a layer by name, so two sharing one is a collision rather than a
    duplicate — the same thing `state.merge_into` renames for."""
    server = serve([ServedLayer(_labels(), name="dup", frame=frame),
                    ServedLayer(_labels(), name="dup", frame=frame)])
    names = [lyr["name"] for lyr in server.state()["layers"]]
    assert "dup" in names and "dup_1" in names


def test_an_annotation_layer_is_there_to_draw_in(frame, stop_server):
    server = serve([ServedLayer(_labels(), name="a", frame=frame)])
    assert any(lyr["name"] == "regions" for lyr in server.state()["layers"])
    assert server.boxes() == [], "empty, but present and readable"

    plain = serve([ServedLayer(_labels(), name="b", frame=frame)], regions=None)
    assert not any(lyr["name"] == "regions" for lyr in plain.state()["layers"])


def test_nothing_to_serve_is_an_error(stop_server):
    with pytest.raises(ServeProblem, match="at least one layer"):
        serve([])


# --- reading the browser's state back -------------------------------------- #
def test_boxes_come_back_sorted_per_axis(frame, stop_server):
    """A box drawn up-and-left has point_a greater than point_b, and every consumer of a
    box in this suite expects half-open lo < hi."""
    server = serve([ServedLayer(_labels(), name="a", frame=frame)])
    with server.viewer.txn() as s:
        s.layers["regions"].annotations = [
            ng.AxisAlignedBoundingBoxAnnotation(id="1", point_a=[9, 8, 7],
                                               point_b=[1, 2, 3]),
            ng.PointAnnotation(id="2", point=[4, 5, 6]),
        ]
    assert server.boxes() == [((1, 2, 3), (9, 8, 7))]
    assert server.points() == [(4, 5, 6)], "points are read apart from boxes"


def test_selected_segments_reads_the_segmentation_layer(frame, stop_server):
    server = serve([ServedLayer(_labels(), kind="segmentation", name="seg",
                                frame=frame)])
    assert server.selected_segments() == set()
    with server.viewer.txn() as s:
        s.layers["seg"].segments = {3, 5}
    assert server.selected_segments() == {3, 5}


def test_asking_for_segments_with_no_sole_segmentation_says_so(frame, stop_server):
    server = serve([ServedLayer(_labels(), kind="image", name="em", frame=frame)])
    with pytest.raises(ServeProblem, match="say which layer"):
        server.selected_segments()


def test_a_click_handler_is_bound_and_survives_a_bad_callback(frame, stop_server):
    """Handlers run on the server's loop thread and neuroglancer swallows an exception
    into a printed traceback, so an unguarded failure is a click that visibly does
    nothing."""
    server = serve([ServedLayer(_labels(), name="a", frame=frame)])
    action = server.on_click(lambda click: None)
    bindings = server.viewer.config_state.state.input_event_bindings.data_view.to_json()
    assert bindings["control+mousedown0"] == action

    boom = server.on_click(lambda click: 1 / 0, button=2)
    handler = next(iter(server.viewer.actions._action_handlers[boom]))

    class _State:
        mouse_voxel_coordinates = np.array([1.0, 2.0, 3.0])
        selected_values = {}

    handler(_State())          # must not raise

    # no position means the cursor is not over data; the callback is simply not called
    seen = []
    named = server.on_click(seen.append, button=1)
    nohit = next(iter(server.viewer.actions._action_handlers[named]))

    class _Nowhere:
        mouse_voxel_coordinates = None
        selected_values = {}

    nohit(_Nowhere())
    assert seen == []


def test_click_position_floors_to_the_voxel_actually_clicked():
    from neu_glance.serving import Click

    assert Click(position=(1.7, 2.2, 3.9)).voxel == (1, 2, 3)


# --- the constructors ------------------------------------------------------ #
#
# They follow one rule: infer what the source RECORDS, require what it does not. So the
# tests come in two shapes — a recorded fact must not be dropped, and an unrecorded one
# must not be guessed.


def _piece(tmp_path, name="piece.h5", **attrs):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / name)
    with h5py.File(path, "w") as f:
        d = f.create_dataset("data", data=_labels((6, 8, 8), "uint64"))
        for key, value in attrs.items():
            d.attrs[key] = value
    return path


def test_from_array_builds_the_frame_for_you():
    layer = ServedLayer.from_array(_labels(), "segmentation", voxel_size=(40, 8, 8),
                                   origin=(80, 16, 24))
    assert layer.kind == "segmentation"
    assert tuple(layer.frame.voxel_size_nm) == (40.0, 8.0, 8.0)
    assert tuple(layer.frame.origin_nm) == (80.0, 16.0, 24.0)
    assert layer.channel_axis is False


def test_the_channel_axis_comes_from_the_RANK_and_cannot_disagree_with_it():
    """A served volume is 3 spatial axes, so 4-D means a channel axis and 3-D means none.
    There is no third possibility, which makes the flag something a caller can only get
    wrong — so it is derived, and a contradiction is an error rather than a silent
    reinterpretation of the axes."""
    assert ServedLayer.from_array(np.zeros((3, 4, 4, 4)), "probability").channel_axis
    assert not ServedLayer.from_array(np.zeros((4, 4, 4)), "image").channel_axis
    assert ServedLayer(np.zeros((3, 4, 4, 4)), kind="probability").channel_axis, \
        "the plain constructor derives it too"
    with pytest.raises(ServeProblem, match="contradicts a 4-D array"):
        ServedLayer.from_array(np.zeros((3, 4, 4, 4)), "probability", channel_axis=False)
    with pytest.raises(ServeProblem, match="3-D or 4-D"):
        ServedLayer.from_array(np.zeros((4, 4)), "image")


def test_two_frames_for_one_array_is_refused():
    from neu_lib import Frame

    with pytest.raises(ServeProblem, match="either frame="):
        ServedLayer.from_array(_labels(), "image",
                               frame=Frame(voxel_size_nm=(8, 8, 8)),
                               voxel_size=(4, 4, 4))


def test_from_hdf5_takes_the_frame_the_file_records(tmp_path):
    """`neu-vol to-hdf5` writes these, so a piece this suite produced needs no
    coordinates retyped — and dropping them is the silent failure."""
    path = _piece(tmp_path, voxel_size=np.asarray([40.0, 8.0, 8.0]),
                  voxel_offset=np.asarray([2, 3, 4], "int64"), axes="zyx")
    layer = ServedLayer.from_hdf5(path, kind="segmentation")
    assert layer.array.shape == (6, 8, 8)
    assert tuple(layer.frame.voxel_size_nm) == (40.0, 8.0, 8.0)
    assert tuple(layer.frame.origin_nm) == (80.0, 24.0, 32.0)
    assert layer.name == "data", "the dataset names the layer"


def test_from_hdf5_needs_a_kind_and_says_why(tmp_path):
    """An HDF5 file has nowhere agreed-on to record it, and reading it off the dtype is
    the mistake `volume_type` exists to override."""
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    with pytest.raises(ServeProblem, match="kind= is required"):
        ServedLayer.from_hdf5(path)
    assert ServedLayer.from_hdf5(path, kind="image").kind == "image"


def test_from_hdf5_needs_a_voxel_size_when_the_file_records_none(tmp_path):
    path = _piece(tmp_path)
    with pytest.raises(ServeProblem, match="records no voxel size"):
        ServedLayer.from_hdf5(path, kind="image")
    layer = ServedLayer.from_hdf5(path, kind="image", voxel_size=(30, 6, 6))
    assert tuple(layer.frame.voxel_size_nm) == (30.0, 6.0, 6.0)


def test_a_crop_lands_at_the_sum_of_the_pieces_origin_and_the_box(tmp_path):
    """The same rule `neu-vol to-hdf5 --crop-bbox` follows: a box out of a piece that
    already knows where it belongs goes there, not at the box's own offset."""
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]),
                  voxel_offset=np.asarray([100, 100, 100], "int64"), axes="zyx")
    layer = ServedLayer.from_hdf5(path, kind="segmentation",
                                  crop=((1, 2, 3), (5, 6, 7)))
    assert layer.array.shape == (4, 4, 4)
    assert tuple(layer.frame.origin_nm) == (808.0, 816.0, 824.0)   # 100*8 + crop*8


def test_a_box_outside_the_extent_is_refused(tmp_path):
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    with pytest.raises(ServeProblem, match="does not fit"):
        ServedLayer.from_hdf5(path, kind="image", crop=((0, 0, 0), (99, 8, 8)))


def test_a_multi_dataset_container_must_be_told_which_array(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / "bag.h5")
    with h5py.File(path, "w") as f:
        for key in ("a", "b"):
            d = f.create_dataset(key, data=_labels((4, 4, 4), "uint64"))
            d.attrs["voxel_size"] = np.asarray([8.0, 8.0, 8.0])
            d.attrs["axes"] = "zyx"

    with pytest.raises(KeyError, match="2 volumetric datasets"):
        ServedLayer.from_hdf5(path, kind="segmentation")
    assert ServedLayer.from_hdf5(path, "/b", "segmentation").name == "b"


def test_a_level_on_a_single_array_is_an_error(tmp_path):
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    with pytest.raises(ServeProblem, match="needs a multiscale volume"):
        ServedLayer.from_source(path, "image", level=2)


def test_from_source_splits_a_dataset_only_on_a_LEADING_slash(tmp_path):
    """`s3://…` carries a colon of its own, so only `:/name` selects an array."""
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    assert ServedLayer.from_source(f"{path}:/data", "image").name == "data"
    # no leading slash: the whole string is the path, and there is no such file
    with pytest.raises((ServeProblem, FileNotFoundError)):
        ServedLayer.from_source(f"{path}:data", "image")


def test_from_volume_takes_kind_from_what_the_volume_RECORDS(tmp_path):
    """precomputed writes it as `info["type"]`. That is the same field `neu-vol copy`
    exists to preserve, and overriding it silently averages label ids into ids that were
    never in the data."""
    from neu_vol import convert
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    src = str(tmp_path / "src.zarr")
    data = _labels((16, 32, 32), "uint32")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, "uint32",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    vol = str(tmp_path / "vol")
    convert(src, vol, voxel_size=(40, 8, 8), kind="segmentation",
            profile="local-neuroglancer", chunk=(8, 8, 8), factors=[(1, 2, 2)],
            min_dim=8, delete_existing=True)

    layer = ServedLayer.from_volume(vol)          # no kind passed
    assert layer.kind == "segmentation"
    assert tuple(layer.frame.voxel_size_nm) == (40.0, 8.0, 8.0)
    # an explicit kind still wins
    assert ServedLayer.from_volume(vol, "image").kind == "image"


def test_a_coarser_level_reports_its_OWN_voxel_size(tmp_path):
    """Never 2**level: real pyramids are anisotropic, and this one halves x/y only."""
    from neu_vol import convert
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    src = str(tmp_path / "src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, (16, 32, 32), "uint32",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in (16, 32, 32)),
                    _labels((16, 32, 32), "uint32"))
    vol = str(tmp_path / "vol")
    convert(src, vol, voxel_size=(40, 8, 8), kind="segmentation",
            profile="local-neuroglancer", chunk=(8, 8, 8), factors=[(1, 2, 2)],
            min_dim=8, delete_existing=True)

    assert tuple(ServedLayer.from_volume(vol, level=1).frame.voxel_size_nm) \
        == (40.0, 16.0, 16.0)
