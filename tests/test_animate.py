"""Timelines, interpolation and the layer split — all of it without a browser.

Nothing here needs neuroglancer, and that is the point: the interpolation core is where the
subtle bugs live, and none of them crash. A fade that starts from the wrong value, a zoom
interpolated linearly, a quaternion taking the long way round, a property that quietly
acquires a midpoint it should never have had — every one of those renders a complete,
plausible-looking animation that is simply not the one that was asked for. So these check
*values*.

The tail of the file needs neuroglancer, and only to pin two things against it: that our
ported maths still agrees with the viewer's own, and that the states we synthesise are ones
the viewer will actually accept.

Source URLs here are `s3://my-bucket/...` throughout. Real data locations do not go in this
repo.
"""

import json
from types import SimpleNamespace

import pytest

from neu_glance.animate import (CAMERA_KEYS, EASINGS, FADE_FROM, LAYER_DEFAULTS,
                                PROPERTY_KINDS, AnimateProblem, Timeline, Tween, ease,
                                interpolate_color_map, interpolate_hex_color,
                                interpolate_quaternion, interpolate_zoom, kind_for)
from neu_glance.state import StateProblem, split_segment_layer, state_url


def _seg(name, segments, **extra):
    layer = {"type": "segmentation", "name": name,
             "source": f"precomputed://s3://my-bucket/{name}",
             "segments": list(segments)}
    layer.update(extra)
    return layer


def _state(*layers, **extra):
    state = {"dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]},
             "layers": list(layers), "layout": "3d",
             "position": [100.0, 200.0, 300.0],
             "projectionScale": 12000.0,
             "projectionOrientation": [0.0, 0.0, 0.0, 1.0]}
    state.update(extra)
    return state


def _layer(state, name):
    return next(lyr for lyr in state["layers"] if lyr["name"] == name)


# --------------------------------------------------------------------------- #
# easing
# --------------------------------------------------------------------------- #
def test_every_easing_is_pinned_at_both_ends_and_never_overshoots():
    """An overshooting curve drives objectAlpha past 1, and neuroglancer clamps SILENTLY.

    The animation then flattens at the ends of every transition with nothing on screen or in
    the state to say why, which reads as a rendering problem rather than a curve problem.
    """
    for name in EASINGS:
        assert ease(name, 0.0) == pytest.approx(0.0), name
        assert ease(name, 1.0) == pytest.approx(1.0), name
        samples = [ease(name, i / 100) for i in range(101)]
        assert all(0.0 <= v <= 1.0 for v in samples), name
        assert all(b >= a - 1e-12 for a, b in zip(samples, samples[1:])), name


def test_easing_clamps_outside_the_unit_interval():
    """A tween evaluated a hair past its window must not extrapolate past its end value."""
    assert ease("in-out", 1.5) == pytest.approx(1.0)
    assert ease("in-out", -0.5) == pytest.approx(0.0)


def test_an_unknown_easing_names_the_ones_that_exist():
    with pytest.raises(AnimateProblem, match="in-out"):
        ease("bounce", 0.5)


# --------------------------------------------------------------------------- #
# interpolators
# --------------------------------------------------------------------------- #
def test_zoom_is_interpolated_geometrically_not_linearly():
    """Zoom is perceived multiplicatively; a linear ramp visibly accelerates near the close end.

    12000 -> 3000 halfway is 6000, not 7500. This IS the "slow and smooth" requirement rather
    than a refinement of it — the linear version reads as a camera that lost control.
    """
    assert interpolate_zoom(12000.0, 3000.0, 0.5) == pytest.approx(6000.0)
    assert interpolate_zoom(1000.0, 4000.0, 0.5) == pytest.approx(2000.0)
    assert interpolate_zoom(12000.0, 3000.0, 0.0) == pytest.approx(12000.0)
    assert interpolate_zoom(12000.0, 3000.0, 1.0) == pytest.approx(3000.0)


def test_a_zoom_through_a_non_positive_scale_falls_back_rather_than_raising():
    """math.log(0) raises, and a state can carry a zero scale. Dying mid-render is worse."""
    assert interpolate_zoom(0.0, 100.0, 0.5) == pytest.approx(50.0)


def test_a_quaternion_takes_the_SHORT_way_round():
    """q and -q are the same rotation. Without the sign flip the camera takes a lazy full spin.

    Half of the way from the identity to (almost) its negation must stay near one of them, not
    swing out to the far side of the sphere.
    """
    a = [0.0, 0.0, 0.0, 1.0]
    b = [0.0, 0.0, -0.0871557, -0.9961947]        # -(a 10-degree rotation): 170 deg apart raw
    mid = interpolate_quaternion(a, b, 0.5)
    assert sum(x * y for x, y in zip(mid, a)) > 0.99      # stayed near a, did not spin away


def test_slerp_of_two_nearly_equal_quaternions_does_not_divide_by_zero():
    """sin(omega) goes to zero as the angle does; the small-angle branch is what keeps it finite."""
    a = [0.0, 0.0, 0.0, 1.0]
    b = [1e-9, 0.0, 0.0, 1.0]
    assert interpolate_quaternion(a, b, 0.5) == pytest.approx(a, abs=1e-6)


def test_slerp_is_exact_at_both_ends_and_keeps_unit_length():
    a = [0.10992055, 0.44167164, 0.85067576, 0.26304829]
    b = [0.0, 0.0, 0.0, 1.0]
    assert interpolate_quaternion(a, b, 0.0) == pytest.approx(a)
    assert interpolate_quaternion(a, b, 1.0) == pytest.approx(b)
    for i in range(11):
        q = interpolate_quaternion(a, b, i / 10)
        assert sum(v * v for v in q) == pytest.approx(1.0, abs=1e-6)


def test_a_colour_ramp_to_black_does_not_go_muddy_in_the_middle():
    """Hex channels are gamma-encoded, so averaging them is not averaging light.

    Every fade-in ramp has black or white at one end, so this is the common case rather than a
    corner. The naive midpoint of black to red is #800000, which reads as a dark maroon; in
    linear light it is #bc0000, which reads as half-lit. Pinned so a later "simplification" to
    a straight lerp cannot silently change the look of every render.
    """
    assert interpolate_hex_color("#000000", "#ff0000", 0.5) == "#bc0000"
    assert interpolate_hex_color("#000000", "#ff0000", 0.5, space="srgb") == "#800000"


def test_colour_endpoints_are_exact_and_names_are_accepted():
    """Parsing goes through shaders.as_hex_color, so there is no second colour parser here."""
    assert interpolate_hex_color("black", "red", 0.0) == "#000000"
    assert interpolate_hex_color("black", "red", 1.0) == "#ff0000"
    assert interpolate_hex_color((0, 0, 0), "#ffffff", 1.0) == "#ffffff"


def test_segment_colours_interpolate_per_key_and_step_where_one_side_has_none():
    """A segment popping to a new colour partway through a fade gets blamed on the data."""
    a = {"1": "#000000", "2": "#ff0000"}
    b = {"1": "#ff0000", "3": "#00ff00"}
    mid = interpolate_color_map(a, b, 0.5)
    assert mid["1"] == "#bc0000"                 # in both: ramps
    assert mid["2"] == "#ff0000"                 # only in a: holds until the end
    assert "3" not in mid                        # only in b: appears at the end
    assert interpolate_color_map(a, b, 1.0)["3"] == "#00ff00"


def test_a_property_nobody_thought_about_STEPS_rather_than_being_averaged():
    """A blended `source` URL or a two-thirds-applied shader is not a thing.

    Stepping by default means an unrecognised property is held and then swapped, which is at
    worst abrupt. Lerping by default would produce nonsense that still renders.
    """
    assert kind_for("someFutureProperty") == "step"
    assert kind_for("source") == "step"
    assert kind_for("segments") == "step"
    assert kind_for("meshSilhouetteRendering") == "linear"
    assert kind_for("projectionScale") == "zoom"


def test_the_segment_list_is_never_blended():
    """A fractional segment list is meaningless, and `segments` is the property most likely to
    be reached for instead of splitting the layer."""
    assert PROPERTY_KINDS["segments"] == "step"


# --------------------------------------------------------------------------- #
# tween semantics
# --------------------------------------------------------------------------- #
def test_a_tween_pins_its_end_value_after_its_window_closes():
    """A fade that quietly un-fades once its window passes reads as a data problem."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.tween(layer="s", at=1.0, seconds=2.0, objectAlpha=0.25)
    assert _layer(tl.at(3.0), "s")["objectAlpha"] == pytest.approx(0.25)
    assert _layer(tl.at(99.0), "s")["objectAlpha"] == pytest.approx(0.25)


def test_a_tween_contributes_nothing_before_it_starts():
    tl = Timeline(_state(_seg("s", ["1"], objectAlpha=0.7)), fps=10.0)
    tl.tween(layer="s", at=5.0, seconds=1.0, objectAlpha=0.0)
    assert _layer(tl.at(0.0), "s")["objectAlpha"] == pytest.approx(0.7)
    assert _layer(tl.at(5.0), "s")["objectAlpha"] == pytest.approx(0.7)


def test_an_absent_property_starts_from_NEUROGLANCERS_DEFAULT_not_from_zero():
    """objectAlpha absent means fully opaque, not invisible.

    Starting a fade-OUT from zero would make it a no-op, and starting a fade-in from 1.0 the
    same — either way the tween is written, the render succeeds, and nothing moves.
    """
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)        # no objectAlpha in the state
    tl.tween(layer="s", at=0.0, seconds=2.0, objectAlpha=0.0)
    assert LAYER_DEFAULTS["objectAlpha"] == 1.0
    assert _layer(tl.at(1.0), "s")["objectAlpha"] == pytest.approx(0.5)


def test_a_later_tween_starts_where_the_previous_one_ENDED_not_at_the_base_value():
    """Otherwise a there-and-back snaps to the opening value at the start of the second leg."""
    tl = Timeline(_state(_seg("s", ["1"], objectAlpha=1.0)), fps=10.0)
    tl.tween(layer="s", at=0.0, seconds=2.0, ease="linear", objectAlpha=0.0)
    tl.tween(layer="s", at=4.0, seconds=2.0, ease="linear", objectAlpha=1.0)
    assert _layer(tl.at(3.0), "s")["objectAlpha"] == pytest.approx(0.0)   # held between legs
    assert _layer(tl.at(5.0), "s")["objectAlpha"] == pytest.approx(0.5)   # ramps 0 -> 1


def test_two_tweens_overlapping_on_one_property_are_refused():
    """Silently dropping one of two fades leaves an animation subtly wrong with nothing to
    point at. Touching endpoints are fine — that is an ordinary hand-off."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.tween(layer="s", at=0.0, seconds=2.0, objectAlpha=0.0)
    with pytest.raises(AnimateProblem, match="overlap"):
        tl.tween(layer="s", at=1.0, seconds=2.0, objectAlpha=1.0)
    tl.tween(layer="s", at=2.0, seconds=1.0, objectAlpha=1.0)            # abutting: allowed


def test_a_tween_naming_a_layer_that_is_not_there_RAISES():
    """Neuroglancer's own Layers.interpolate silently skips an unmatched layer.

    That makes a typo look like a rendering bug: everything else animates and one object sits
    still. Better to refuse and list what the state actually has.
    """
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    with pytest.raises(AnimateProblem, match="'s'"):
        tl.tween(layer="typo", at=0.0, seconds=1.0, objectAlpha=0.0)


def test_a_layer_name_containing_a_dot_and_a_space_addresses_correctly():
    """Why a path is a tuple and not a dotted string: the real scene has `head.precomputed`."""
    tl = Timeline(_state(_seg("head.precomputed v2", ["1"])), fps=10.0)
    tl.tween(layer="head.precomputed v2", at=0.0, seconds=1.0, meshSilhouetteRendering=10.0)
    assert _layer(tl.at(1.0), "head.precomputed v2")["meshSilhouetteRendering"] == 10.0


def test_interpolating_never_mutates_the_state_it_was_given():
    """A mutated base compounds: every later frame is built on the previous frame's edits."""
    original = _state(_seg("s", ["1"], objectAlpha=1.0))
    snapshot = json.dumps(original)
    tl = Timeline(original, fps=10.0)
    tl.tween(layer="s", at=0.0, seconds=1.0, objectAlpha=0.0)
    tl.at(0.5)
    tl.at(1.0)
    assert json.dumps(original) == snapshot
    assert _layer(tl.base, "s")["objectAlpha"] == 1.0


def test_setting_the_same_property_twice_at_time_zero_is_refused():
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.set(layer="s", objectAlpha=0.0)
    with pytest.raises(AnimateProblem, match="set twice"):
        tl.set(layer="s", objectAlpha=1.0)


def test_a_step_property_given_a_window_says_so_rather_than_pretending_to_ramp():
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.tween(layer="s", at=0.0, seconds=3.0, visible=False)
    assert any("does not interpolate" in n for n in tl.check())
    assert _layer(tl.at(1.5), "s").get("visible", True) is True          # still on midway
    assert _layer(tl.at(3.0), "s")["visible"] is False


# --------------------------------------------------------------------------- #
# frame enumeration
# --------------------------------------------------------------------------- #
def test_the_frame_count_INCLUDES_the_last_frame():
    """A 2 s timeline at 30 fps is 61 frames. Off by one here truncates every render."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=30.0)
    tl.tween(layer="s", at=0.0, seconds=2.0, objectAlpha=0.0)
    assert tl.duration == pytest.approx(2.0)
    assert tl.frame_count == 61
    assert len(list(tl.frames())) == 61


def test_frame_times_are_exact_multiples_of_the_frame_interval():
    """A running sum drifts, and a drifting clock puts the last frames of a long render at
    times no keyframe ever named."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=30.0)
    tl.hold(10.0)
    for i, t, _ in tl.frames():
        assert t == pytest.approx(i / 30.0, abs=1e-12)


def test_a_zero_length_timeline_still_renders_one_frame():
    tl = Timeline(_state(_seg("s", ["1"])), fps=30.0)
    assert tl.frame_count == 1
    assert any("zero duration" in n for n in tl.check())


def test_start_and_end_frame_select_a_range_with_the_SAME_numbering():
    """Resume and split renders write into one directory; a renumbered range overwrites."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.hold(2.0)
    chunk = list(tl.frames(start_frame=5, end_frame=9))
    assert [i for i, _, _ in chunk] == [5, 6, 7, 8]
    assert chunk[0][1] == pytest.approx(0.5)


def test_the_first_frame_is_the_base_state_where_no_tween_has_opened():
    tl = Timeline(_state(_seg("s", ["1"], objectAlpha=0.4)), fps=10.0)
    tl.tween(layer="s", at=1.0, seconds=1.0, objectAlpha=1.0)
    assert _layer(next(iter(tl.frames()))[2], "s")["objectAlpha"] == pytest.approx(0.4)


def test_the_cursor_moves_only_when_hold_is_called():
    """Two tweens written one after another must run TOGETHER unless told otherwise —
    a fade-while-moving is the common case, not the awkward one."""
    tl = Timeline(_state(_seg("a", ["1"]), _seg("b", ["2"])), fps=10.0)
    tl.tween(layer="a", seconds=1.0, objectAlpha=0.0)
    tl.tween(layer="b", seconds=1.0, objectAlpha=0.0)
    assert [t.at for t in tl.tweens] == [0.0, 0.0]
    tl.hold(5.0)
    tl.tween(layer="a", seconds=1.0, objectAlpha=1.0)
    assert tl.tweens[-1].at == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# the layer split
# --------------------------------------------------------------------------- #
def test_splitting_gives_one_layer_per_segment_carrying_the_source():
    """Neuroglancer has no per-segment alpha, so this is the only route to a per-object fade."""
    state = _state(_seg("seg", ["1", "2", "3"]))
    out, created, _ = split_segment_layer(state, "seg", name_template="n{segment}")
    assert created == {"1": "n1", "2": "n2", "3": "n3"}
    for name in created.values():
        layer = _layer(out, name)
        assert layer["source"] == "precomputed://s3://my-bucket/seg"
        assert layer["type"] == "segmentation"
    assert [_layer(out, n)["segments"] for n in created.values()] == [["1"], ["2"], ["3"]]


def test_each_split_layer_keeps_its_segments_colour_INCLUDING_a_hashed_one():
    """Losing a colour gives a perfectly attractive scene in entirely the wrong colours.

    An explicit entry is carried across. A segment with no entry is hash-coloured by
    neuroglancer from (colorSeed, segment_id) and nothing else, both of which a verbatim copy
    preserves — so it must come out identical without any colour being written.
    """
    state = _state(_seg("seg", ["1", "2"], segmentColors={"1": "#19ffb6"}, colorSeed=17))
    out, created, _ = split_segment_layer(state, "seg", name_template="n{segment}")
    assert _layer(out, "n1")["segmentColors"] == {"1": "#19ffb6"}
    assert "segmentColors" not in _layer(out, "n2")            # hashed, and left that way
    assert _layer(out, "n1")["colorSeed"] == 17
    assert _layer(out, "n2")["colorSeed"] == 17


def test_the_render_properties_carry_across_without_a_whitelist():
    """A whitelist here would go stale against the next viewer release, silently dropping
    whatever it had not heard of."""
    state = _state(_seg("seg", ["1"], selectedAlpha=0, meshSilhouetteRendering=3.3,
                        saturation=0.5, notSelectedAlpha=0.1, tab="segments"))
    out, created, _ = split_segment_layer(state, "seg", name_template="n{segment}")
    layer = _layer(out, "n1")
    assert layer["meshSilhouetteRendering"] == 3.3
    assert layer["selectedAlpha"] == 0
    assert layer["saturation"] == 0.5
    assert layer["notSelectedAlpha"] == 0.1
    assert layer["tab"] == "segments"


def test_a_starred_but_hidden_segment_does_not_become_a_VISIBLE_layer():
    """A `!` prefix means starred and NOT shown. Splitting those in would put objects on
    screen that the author had deliberately switched off."""
    state = _state(_seg("seg", ["1", "!2", "3"]))
    out, created, _ = split_segment_layer(state, "seg", name_template="n{segment}")
    assert set(created) == {"1", "3"}
    assert _layer(out, "seg")["segments"] == ["!2"]            # kept, so the star survives


def test_the_original_layer_is_kept_so_references_to_it_do_not_dangle():
    """`selectedLayer`, a synapse layer's `linkedSegmentationLayer` and a toolPalettes entry
    all name a layer by name, and neuroglancer resolves a missing one to nothing in silence."""
    state = _state(_seg("seg", ["1"]), selectedLayer={"layer": "seg", "visible": True})
    out, _, _ = split_segment_layer(state, "seg", name_template="n{segment}")
    assert _layer(out, "seg")["visible"] is False
    assert _layer(out, "seg")["segments"] == []
    assert out["selectedLayer"]["layer"] == "seg"              # still resolves


def test_the_split_layers_replace_the_original_IN_PLACE_so_draw_order_is_kept():
    """Neuroglancer draws in layer order; moving twenty-five layers to the end changes what
    is in front of what."""
    state = _state({"type": "image", "name": "em", "source": "zarr://s3://my-bucket/em"},
                   _seg("seg", ["1", "2"]),
                   _seg("rois", ["9"]))
    out, _, _ = split_segment_layer(state, "seg", name_template="n{segment}")
    assert [lyr["name"] for lyr in out["layers"]] == ["em", "n1", "n2", "seg", "rois"]


def test_segment_ids_stay_STRINGS_through_the_split():
    """A 19-digit uint64 through a JSON number comes back rounded."""
    big = "18446744073709551615"
    state = _state(_seg("seg", [big]))
    out, created, _ = split_segment_layer(state, "seg", name_template="n{segment}")
    assert created == {big: f"n{big}"}
    assert _layer(out, f"n{big}")["segments"] == [big]


def test_splitting_never_mutates_the_state_it_was_given():
    state = _state(_seg("seg", ["1", "2"]))
    snapshot = json.dumps(state)
    split_segment_layer(state, "seg", name_template="n{segment}")
    assert json.dumps(state) == snapshot


def test_a_split_layer_name_that_collides_is_renamed_and_REPORTED():
    """Neuroglancer keys a layer by name, so a collision is a collision, not a duplicate."""
    state = _state(_seg("seg", ["1"]), _seg("n1", ["9"]))
    out, created, notes = split_segment_layer(state, "seg", name_template="n{segment}")
    assert created == {"1": "n1-2"}
    assert any("renamed" in n for n in notes)


def test_explicit_names_beat_the_template():
    """A 19-digit body id in the layer bar is unreadable; names are what make it usable."""
    state = _state(_seg("seg", ["1", "2"]))
    out, created, _ = split_segment_layer(
        state, "seg", names={"1": "KC-alpha"}, name_template="n{segment}")
    assert created == {"1": "KC-alpha", "2": "n2"}


def test_splitting_a_layer_that_is_not_there_or_is_not_a_segmentation_says_which():
    state = _state({"type": "image", "name": "em", "source": "zarr://s3://my-bucket/em"})
    with pytest.raises(StateProblem, match="'em'"):
        split_segment_layer(state, "nope")
    with pytest.raises(StateProblem, match="image"):
        split_segment_layer(state, "em")


def test_asking_to_split_a_hidden_segment_explains_the_exclamation_mark():
    state = _state(_seg("seg", ["1", "!2"]))
    with pytest.raises(StateProblem, match="starred but hidden"):
        split_segment_layer(state, "seg", segments=["2"])


# --------------------------------------------------------------------------- #
# groups, sequences and the camera
# --------------------------------------------------------------------------- #
def test_a_sequence_starts_every_layer_HIDDEN_without_being_asked():
    """Otherwise the scene opens with everything already visible and the fades do nothing —
    a render that succeeds, takes an hour, and shows the wrong thing."""
    tl = Timeline(_state(_seg("seg", ["1", "2"])), fps=10.0)
    made = tl.split("seg", prefix="n")
    tl.group("g1", [made["1"]])
    tl.group("g2", [made["2"]])
    tl.sequence(["g1", "g2"], start=1.0, seconds=1.0, stagger=0.5)
    opening = tl.at(0.0)
    assert _layer(opening, made["1"])["objectAlpha"] == pytest.approx(FADE_FROM)
    assert _layer(opening, made["2"])["objectAlpha"] == pytest.approx(FADE_FROM)


def test_a_sequence_staggers_the_groups_and_finishes_them_all():
    tl = Timeline(_state(_seg("seg", ["1", "2"])), fps=10.0)
    made = tl.split("seg", prefix="n")
    tl.group("g1", [made["1"]])
    tl.group("g2", [made["2"]])
    tl.sequence(["g1", "g2"], start=1.0, seconds=1.0, stagger=0.5, ease="linear")
    at_15 = tl.at(1.5)
    assert _layer(at_15, made["1"])["objectAlpha"] == pytest.approx(0.5005)   # halfway
    assert _layer(at_15, made["2"])["objectAlpha"] == pytest.approx(FADE_FROM)
    done = tl.at(2.5)
    assert _layer(done, made["1"])["objectAlpha"] == pytest.approx(1.0)
    assert _layer(done, made["2"])["objectAlpha"] == pytest.approx(1.0)


def test_a_group_naming_a_layer_that_is_not_there_is_refused_at_GROUP_time():
    """Catching it here names the mistake; catching it at render time names a frame index."""
    tl = Timeline(_state(_seg("seg", ["1"])), fps=10.0)
    with pytest.raises(AnimateProblem, match="not in this state"):
        tl.group("g", ["neuron 404"])


def test_an_undefined_group_in_a_sequence_lists_the_defined_ones():
    tl = Timeline(_state(_seg("seg", ["1"])), fps=10.0)
    tl.group("g1", ["seg"])
    with pytest.raises(AnimateProblem, match="g1"):
        tl.sequence(["typo"], seconds=1.0)


def test_view_takes_ONLY_the_camera_keys_from_the_state_it_is_given():
    """"Move the camera to here" must not quietly swap the scene — a keyframe URL also
    carries layers, a layout and segment lists."""
    tl = Timeline(_state(_seg("seg", ["1"])), fps=10.0)
    target = _state(_seg("other", ["9"]), projectionScale=3000.0, layout="4panel")
    tl.view(target, at=0.0, seconds=2.0)
    assert {t.path[0] for t in tl.tweens} <= set(CAMERA_KEYS)
    end = tl.at(2.0)
    assert end["projectionScale"] == pytest.approx(3000.0)
    assert end["layout"] == "3d"                                     # untouched
    assert [lyr["name"] for lyr in end["layers"]] == ["seg"]         # untouched


def test_view_accepts_a_neuroglancer_url_because_that_is_how_a_view_gets_tuned():
    """Framing a 3D view is done by dragging, not by writing quaternions."""
    tl = Timeline(_state(_seg("seg", ["1"])), fps=10.0)
    target = _state(_seg("seg", ["1"]), projectionScale=3000.0)
    tl.view(state_url(target), at=0.0, seconds=2.0)
    assert tl.at(2.0)["projectionScale"] == pytest.approx(3000.0)


def test_view_on_a_state_with_no_camera_in_it_says_so():
    tl = Timeline(_state(_seg("seg", ["1"])), fps=10.0)
    with pytest.raises(AnimateProblem, match="camera"):
        tl.view({"layers": []}, at=0.0, seconds=1.0)


def test_a_zoom_tween_with_no_scale_anywhere_asks_for_an_explicit_start():
    """neuroglancer declares projectionScale optional with NO default — it fits the data
    instead — so there is no honest value to start a zoom from."""
    state = _state(_seg("seg", ["1"]))
    del state["projectionScale"]
    tl = Timeline(state, fps=10.0)
    tl.tween(seconds=2.0, projectionScale=3000.0)
    with pytest.raises(AnimateProblem, match="start="):
        tl.check()


# --------------------------------------------------------------------------- #
# relative camera moves
# --------------------------------------------------------------------------- #
def test_a_FULL_TURN_is_expressible_which_is_the_whole_point_of_orbit():
    """Slerp between two views cannot do this, and no number of keyframes fixes it.

    Slerp takes the shortest arc, so the two ends of a 360-degree turn are the same
    orientation and interpolating between them is a no-op; 200 degrees silently becomes 160
    the other way. Orbit varies the ANGLE instead, so any rotation is expressible.
    """
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.orbit(360, at=0.0, seconds=4.0, ease="linear")
    quarters = [tl.at(t)["projectionOrientation"] for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert quarters[2] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-9)      # halfway = 180
    # q and -q are the SAME rotation, so a full turn comes back to the start's negation.
    assert quarters[4] == pytest.approx([-v for v in quarters[0]], abs=1e-9)
    assert all(sum(v * v for v in q) == pytest.approx(1.0, abs=1e-9) for q in quarters)


def test_a_relative_move_PINS_ITS_EVALUATED_END_not_its_delta():
    """Past its window a tween pins its end — and a relative move's `end` is a DELTA.

    Returning it verbatim puts the orbit's `{axis, degrees}` dict where the quaternion
    belongs, and the zoom's bare factor where the scale belongs. It is the last frame of the
    move that breaks, which is the one checked last.
    """
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.orbit(90, at=0.0, seconds=2.0)
    tl.zoom(0.25, at=0.0, seconds=2.0)
    for t in (2.0, 5.0):
        end = tl.at(t)
        assert isinstance(end["projectionOrientation"], list)
        assert len(end["projectionOrientation"]) == 4
        assert end["projectionScale"] == pytest.approx(3000.0)


def test_zoom_is_multiplicative_and_geometric():
    """Relative so it composes and needs no knowledge of the current scale — which after a
    frame_on is a number nobody wrote down."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.zoom(0.25, at=0.0, seconds=2.0, ease="linear")
    assert tl.at(1.0)["projectionScale"] == pytest.approx(6000.0)     # geometric midpoint
    assert tl.at(2.0)["projectionScale"] == pytest.approx(3000.0)
    with pytest.raises(AnimateProblem, match="positive"):
        tl.zoom(0.0, seconds=1.0)


def test_a_screen_axis_and_a_volume_axis_compose_on_OPPOSITE_sides():
    """Which side you multiply on is what makes a spin follow the camera or the specimen.

    Post-multiplying reads the axis in the camera's frame, so `up` means "left-to-right as I
    am looking at it" from any starting view; pre-multiplying reads it in the volume's, so a
    shot about `z` is anatomically meaningful and reproducible across scenes.
    """
    tilted = _state(_seg("s", ["1"]), projectionOrientation=[0.5, 0.5, 0.5, 0.5])
    screen = Timeline(tilted, fps=10.0)
    screen.orbit(90, axis="up", at=0.0, seconds=1.0)
    volume = Timeline(tilted, fps=10.0)
    volume.orbit(90, axis="y", at=0.0, seconds=1.0)
    assert screen.at(1.0)["projectionOrientation"] != volume.at(1.0)["projectionOrientation"]
    # From the identity there is nothing to tell them apart, which is the sanity check.
    flat_screen = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    flat_screen.orbit(90, axis="up", at=0.0, seconds=1.0)
    flat_volume = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    flat_volume.orbit(90, axis="y", at=0.0, seconds=1.0)
    assert flat_screen.at(1.0)["projectionOrientation"] == pytest.approx(
        flat_volume.at(1.0)["projectionOrientation"])


def test_an_unknown_orbit_axis_lists_the_known_ones():
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    with pytest.raises(AnimateProblem, match="forward"):
        tl.orbit(90, axis="sideways", seconds=1.0)
    with pytest.raises(AnimateProblem, match="zero vector"):
        tl.orbit(90, axis=(0, 0, 0), seconds=1.0)


def test_relative_moves_CHAIN_from_where_the_previous_one_ended():
    """Two orbits in a row must add up, not restart from the base state each time."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.orbit(90, at=0.0, seconds=1.0)
    tl.orbit(90, at=2.0, seconds=1.0)
    once = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    once.orbit(180, at=0.0, seconds=1.0)
    assert tl.at(3.0)["projectionOrientation"] == pytest.approx(
        once.at(1.0)["projectionOrientation"], abs=1e-9)


# --------------------------------------------------------------------------- #
# framing on an object
# --------------------------------------------------------------------------- #
def test_frame_on_centres_the_box_and_fits_it():
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.frame_on(((0.0, 100.0, 200.0), (50.0, 300.0, 600.0)), at=0.0, seconds=2.0,
                units="voxels")
    end = tl.at(2.0)
    assert end["position"] == pytest.approx([400.0, 200.0, 25.0])       # centre, zyx -> xyz
    assert end["projectionScale"] == pytest.approx(400.0 * 1.15)        # widest span + margin


def test_frame_on_converts_NANOMETRES_to_the_viewers_own_voxels():
    """`segment_boxes` returns nm — the suite's model space — but neuroglancer's `position`
    is in the state's voxels. Confusing them does not raise: it points the camera somewhere
    plausible and wrong by the voxel size, a factor of eight on an 8 nm volume.
    """
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)                   # 8 nm isotropic
    tl.frame_on(((0.0, 800.0, 1600.0), (400.0, 2400.0, 4800.0)), at=0.0, seconds=2.0)
    assert tl.at(2.0)["position"] == pytest.approx([400.0, 200.0, 25.0])


def test_frame_on_accepts_anything_with_lo_and_hi_so_a_BBox_works():
    box = SimpleNamespace(lo=(0.0, 100.0, 200.0), hi=(50.0, 300.0, 600.0))
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.frame_on(box, at=0.0, seconds=1.0, units="voxels")
    assert tl.at(1.0)["position"] == pytest.approx([400.0, 200.0, 25.0])


def test_frame_on_without_zoom_pans_only():
    """For following something at a fixed scale rather than fitting it."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.frame_on(((0.0, 100.0, 200.0), (50.0, 300.0, 600.0)), at=0.0, seconds=1.0,
                units="voxels", zoom=False)
    assert tl.at(1.0)["projectionScale"] == pytest.approx(12000.0)


def test_a_state_with_no_dimensions_says_so_rather_than_framing_the_wrong_place():
    state = _state(_seg("s", ["1"]))
    del state["dimensions"]
    tl = Timeline(state, fps=10.0)
    with pytest.raises(AnimateProblem, match="units='voxels'"):
        tl.frame_on(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), at=0.0, seconds=1.0)


def test_framing_and_orbiting_compose():
    """Frame an object, then turn around it — the shot this whole API exists for."""
    tl = Timeline(_state(_seg("s", ["1"])), fps=10.0)
    tl.frame_on(((0.0, 100.0, 200.0), (50.0, 300.0, 600.0)), at=0.0, seconds=2.0,
                units="voxels")
    tl.orbit(180, at=2.0, seconds=4.0)
    tl.zoom(0.5, at=2.0, seconds=4.0)
    end = tl.at(6.0)
    assert end["position"] == pytest.approx([400.0, 200.0, 25.0])       # framing held
    assert end["projectionScale"] == pytest.approx(400.0 * 1.15 * 0.5)
    assert end["projectionOrientation"] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-9)


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def test_a_timeline_round_trips_through_json():
    """The timeline written beside the frames is what makes a render reproducible and its
    timings editable without re-deriving them."""
    tl = Timeline(_state(_seg("seg", ["1", "2"])), fps=24.0)
    made = tl.split("seg", prefix="n")
    tl.group("g", [made["1"]])
    tl.sequence(["g"], start=1.0, seconds=1.0)
    tl.tween(seconds=3.0, projectionScale=3000.0)
    back = Timeline.from_json(json.loads(json.dumps(tl.to_json())))
    assert back.fps == 24.0
    assert back.duration == pytest.approx(tl.duration)
    assert back.groups == tl.groups
    assert [t.to_json() for t in back.tweens] == [t.to_json() for t in tl.tweens]
    assert back.at(1.5) == tl.at(1.5)


def test_an_explicit_duration_overrides_the_derived_one():
    """A trailing hold of stillness at the end of an animation is a normal thing to want."""
    tl = Timeline(_state(_seg("seg", ["1"])), fps=10.0)
    tl.tween(layer="seg", seconds=1.0, objectAlpha=0.0)
    assert tl.duration == pytest.approx(1.0)
    tl.duration = 4.0
    assert tl.frame_count == 41


# --------------------------------------------------------------------------- #
# pinned against neuroglancer itself — still no browser
# --------------------------------------------------------------------------- #
ng = pytest.importorskip("neuroglancer", reason="the serve extra is not installed")

from neuroglancer import viewer_state as _vs            # noqa: E402


def test_our_zoom_and_slerp_agree_with_neuroglancers_own():
    """Ported from viewer_state.py:75-111 rather than re-derived. This is what stops the two
    drifting after a later cleanup here or an upgrade there."""
    import numpy as np
    for a, b in [(12000.0, 3000.0), (1.0, 1000.0), (0.25, 0.5)]:
        for t in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert interpolate_zoom(a, b, t) == pytest.approx(_vs.interpolate_zoom(a, b, t),
                                                              rel=1e-6)
    pairs = [([0.0, 0.0, 0.0, 1.0], [0.10992055, 0.44167164, 0.85067576, 0.26304829]),
             ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.0871557, -0.9961947]),
             ([0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 1.0])]
    for a, b in pairs:
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            theirs = _vs.quaternion_slerp(np.array(a, np.float32), np.array(b, np.float32), t)
            assert interpolate_quaternion(a, b, t) == pytest.approx(list(theirs), abs=1e-6)


def test_our_layer_defaults_agree_with_the_viewers_declared_defaults():
    """A viewer release changing one of these would move where every unspecified fade starts,
    and nothing else would notice."""
    declared = {}
    for name in ("objectAlpha", "selectedAlpha", "notSelectedAlpha", "saturation",
                 "meshSilhouetteRendering", "meshRenderScale", "crossSectionRenderScale"):
        layer = _vs.SegmentationLayer()
        declared[name] = getattr(layer, name)
    declared["opacity"] = _vs.ImageLayer().opacity
    for name, value in declared.items():
        assert LAYER_DEFAULTS[name] == pytest.approx(value), name


def test_every_frame_state_is_accepted_by_neuroglancers_own_state_model():
    """The guard that pure dict interpolation still yields SCHEMA-LEGAL states.

    A colour that is not #rrggbb, an orientation of length three, a float where a list
    belongs — all of those would otherwise surface as a blank viewport at render time, an hour
    in. `ViewerState` construction works fine even though `ViewerState.interpolate` does not.
    """
    state = _state(_seg("seg", ["1", "2"], segmentColors={"1": "#19ffb6"},
                        meshSilhouetteRendering=3.3),
                   {"type": "image", "name": "em", "source": "zarr://s3://my-bucket/em"})
    tl = Timeline(state, fps=12.0)
    made = tl.split("seg", prefix="n")
    tl.group("g", [made["1"]])
    tl.group("h", [made["2"]])
    tl.sequence(["g", "h"], start=0.5, seconds=1.0, stagger=0.5)
    tl.view(_state(_seg("seg", ["1"]), projectionScale=3000.0,
                   projectionOrientation=[0.10992055, 0.44167164, 0.85067576, 0.26304829]),
            at=1.0, seconds=2.0)
    frames = list(tl.frames())
    assert len(frames) > 1
    for _, _, frame in frames:
        ng.ViewerState(json.loads(json.dumps(frame))).to_json()


def test_neuroglancers_own_interpolate_is_deliberately_not_used():
    """`ViewerState.interpolate` raises for a state with ANY layer, in 2.41.2.

    `Layer.interpolate` (viewer_state.py:437-443) reads `a.layer_position`; the property is
    `local_position`/`localPosition`. `neuroglancer.tool.video_tool` is dead as shipped for
    the same reason. If this test ever starts failing, the bug has been fixed upstream — which
    still would not make it usable here, because it covers neither meshSilhouetteRendering nor
    segmentColors and it materialises defaults the input never had.
    """
    a = ng.ViewerState(_state(_seg("seg", ["1"])))
    b = ng.ViewerState(_state(_seg("seg", ["1"], objectAlpha=0.0)))
    with pytest.raises(AttributeError, match="layer_position"):
        ng.ViewerState.interpolate(a, b, 0.5)


def test_importing_animate_does_not_pull_in_neuroglancer():
    """The interpolation core stays testable in a CI that installs no extras."""
    import subprocess
    import sys
    code = ("import neu_glance.animate, sys; "
            "assert 'neuroglancer' not in sys.modules; "
            "assert 'numpy' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
