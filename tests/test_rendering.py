"""The render driver, without a browser.

Everything here except the actual round trip: state shaping, frame paths, resume, the retry
on a wrong-sized reply, the refresh cap, and the ffmpeg command. The one thing that genuinely
needs a browser — a frame coming back — is covered by the fake reply below and by rendering
the real animation by hand.

Source URLs are `s3://my-bucket/...` throughout. Real data locations do not go in this repo.
"""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from neu_glance.animate import Timeline

pytest.importorskip("neuroglancer", reason="the serve extra is not installed")

from neu_glance.rendering import (  # noqa: E402
    CHROME_OFF, PNG_MAGIC, RenderProblem, _RequestQueue, _Tail, _write_atomic,
    ffmpeg_command, frame_path, is_written, pending_frames, print_encode_hint, record,
    render_state, stale_frames, wait_for_browser,
    BROWSERS, GPU_FLAGS, find_browser, headless_browser)


def _state(**extra):
    state = {"dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]},
             "layers": [{"type": "segmentation", "name": "seg",
                         "source": "precomputed://s3://my-bucket/seg",
                         "segments": ["1", "2"]}],
             "layout": "3d", "projectionScale": 12000.0}
    state.update(extra)
    return state


def _timeline(fps=10.0):
    tl = Timeline(_state(), fps=fps)
    tl.tween(layer="seg", at=0.0, seconds=1.0, objectAlpha=0.0)
    return tl


class _Reply:
    """What the browser posts back: PNG bytes plus the size it actually rendered."""

    def __init__(self, width, height, image=PNG_MAGIC + b"fake"):
        self.width, self.height, self.image = width, height, image


class _Config:
    """The mutable half of a viewer's config_state, as `capture_screenshots` uses it."""

    def __init__(self):
        self.prefetch = []
        self.show_ui_controls = None
        self.show_panel_borders = None
        self.viewer_size = None


class _FakeViewer:
    """A viewer that answers instantly, so the capture loop can run with no browser.

    `reply_size=None` echoes whatever size was asked for; a fixed pair forces a mismatch.
    """

    def __init__(self, *, url="http://host:1/v/tok/", reply_size=None, answers=True,
                 statistics=()):
        self.url, self.reply_size, self.answers = url, reply_size, answers
        self.statistics = list(statistics)
        self.config = _Config()
        self.states = []
        self.probes = 0
        self.config_touched = 0

    def get_viewer_url(self):
        return self.url

    def set_state(self, state):
        self.states.append(state)

    @property
    def config_state(self):
        viewer = self

        class _Handle:
            @staticmethod
            @contextlib.contextmanager
            def txn():
                viewer.config_touched += 1
                yield viewer.config
        return _Handle

    def async_screenshot(self, callback, include_depth=False, statistics_callback=None):
        self.probes += 1
        for loaded, total, downloading in self.statistics:
            if statistics_callback is not None:
                statistics_callback(SimpleNamespace(total=SimpleNamespace(
                    visible_chunks_gpu_memory=loaded, visible_chunks_total=total,
                    visible_chunks_downloading=downloading)))
        if not self.answers:
            return
        size = self.reply_size or self.config.viewer_size or (64, 64)
        callback(SimpleNamespace(screenshot=_Reply(*size)))


# --------------------------------------------------------------------------- #
# shaping the state
# --------------------------------------------------------------------------- #
def test_the_render_state_switches_off_every_panel_and_the_navigation_furniture():
    """Axis lines and each volume's bounding box are ON by default in neuroglancer.

    They help you navigate and they ruin a mesh render, so the two defaults are inverted here.
    A frame with the layer list panel across it is not obviously wrong until you look.
    """
    shaped = render_state(_state())
    for key in CHROME_OFF:
        assert shaped[key]["visible"] is False
    assert shaped["showAxisLines"] is False
    assert shaped["showDefaultAnnotations"] is False


def test_the_render_state_keeps_the_scene_alone():
    """It shapes the viewer's furniture, not what is being rendered."""
    shaped = render_state(_state())
    assert shaped["layout"] == "3d"
    assert shaped["projectionScale"] == 12000.0
    assert [lyr["name"] for lyr in shaped["layers"]] == ["seg"]


def test_the_render_state_merges_into_an_existing_panel_block_rather_than_replacing_it():
    """`selectedLayer` carries a layer name as well as a visibility."""
    shaped = render_state(_state(selectedLayer={"layer": "seg", "visible": True, "size": 364}))
    assert shaped["selectedLayer"] == {"layer": "seg", "visible": False, "size": 364}


def test_a_tool_palette_is_dropped_because_it_names_layers_a_split_has_emptied():
    shaped = render_state(_state(toolPalettes={"All controls": {"row": 1, "query": "+"}}))
    assert "toolPalettes" not in shaped


def test_the_render_state_never_mutates_the_state_it_was_given():
    state = _state()
    snapshot = json.dumps(state)
    render_state(state)
    assert json.dumps(state) == snapshot


def test_the_memory_knobs_are_left_alone_unless_asked_for():
    """Neuroglancer's screenshot tool forces 3 GiB / 3 GiB / 32; the viewer's own defaults are
    a better starting point, and these matter only once a scene is twenty-five mesh layers."""
    assert "gpuMemoryLimit" not in render_state(_state())
    assert render_state(_state(), gpu_memory_limit=8 << 30)["gpuMemoryLimit"] == 8 << 30


# --------------------------------------------------------------------------- #
# frames on disk
# --------------------------------------------------------------------------- #
def test_a_frame_pattern_with_no_integer_field_is_refused():
    """ffmpeg reads a sequence through the printf field; without one it reads a single file."""
    with pytest.raises(RenderProblem, match="integer field"):
        frame_path("/tmp", 3, "frame.png")


def test_a_truncated_frame_is_NOT_treated_as_done(tmp_path):
    """A frame killed mid-write exists and is zero length. `os.path.exists` calls that done,
    which leaves one torn frame in the middle of a sequence that nothing notices until the
    encode — or worse, that the encode accepts."""
    good, empty, junk = (tmp_path / n for n in ("g.png", "e.png", "j.png"))
    good.write_bytes(PNG_MAGIC + b"...")
    empty.write_bytes(b"")
    junk.write_bytes(b"\x89PNG")                      # a real mid-write truncation
    assert is_written(str(good))
    assert not is_written(str(empty))
    assert not is_written(str(junk))
    assert not is_written(str(tmp_path / "absent.png"))


def test_a_frame_is_written_atomically_so_a_kill_leaves_no_half_png(tmp_path):
    path = str(tmp_path / "f.png")
    _write_atomic(path, PNG_MAGIC + b"body")
    assert is_written(path)
    assert not os.path.exists(path + ".tmp")


def test_resume_skips_frames_that_are_already_written_and_keeps_their_numbering(tmp_path):
    """Resume and a split render share a directory; renumbering would overwrite."""
    tl = _timeline()
    for i in (0, 1, 2):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    todo = pending_frames(tl, str(tmp_path))
    assert [i for i, _, _ in todo] == list(range(3, tl.frame_count))
    assert todo[0][1] == pytest.approx(0.3)
    assert not pending_frames(tl, str(tmp_path), resume=False)[0][0]     # index 0 back again


def test_frames_left_over_past_the_END_of_the_timeline_are_REMOVED(tmp_path, capsys):
    """These silently corrupt the video, which is why they are hunted rather than warned about.

    `ffmpeg -i frame_%05d.png` consumes the whole numbered sequence and has no idea which run
    wrote what. Shorten an animation, re-render, and the encode appends the tail of the
    previous one — a video ending in a scene the timeline no longer contains, from a directory
    where every individual frame is perfectly valid.
    """
    tl = _timeline()
    for i in range(tl.frame_count + 5):                  # a longer previous run
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    survivors = sorted(p.name for p in tmp_path.glob("frame_*.png"))
    assert len(survivors) == tl.frame_count
    assert "removing 5 frame(s) past the end" in capsys.readouterr().err


def test_stale_detection_reads_the_index_out_of_the_PATTERN(tmp_path):
    """A caller may rename the frames, and the sequence has to stay identifiable."""
    for i in (0, 1, 9, 10):
        (tmp_path / f"shot-{i:03d}.png").write_bytes(PNG_MAGIC)
    (tmp_path / "notes.txt").write_bytes(b"not a frame")
    stale = stale_frames(str(tmp_path), 9, "shot-%03d.png")
    assert [os.path.basename(p) for p in stale] == ["shot-009.png", "shot-010.png"]


def test_resuming_onto_a_CHANGED_timeline_re_renders_instead_of_mixing_two(tmp_path, capsys):
    """The worst version of stale frames, because nothing marks them.

    Frames left past the end of a shortened animation at least sit in a block at the end.
    Frames rendered from an earlier version of the timeline sit in the MIDDLE, are valid PNGs
    of the right size, and show the animation someone stopped wanting. The timeline written
    beside them is what makes that detectable at all.
    """
    first = _timeline()
    record(first, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    for path in tmp_path.glob("frame_*.png"):
        path.write_bytes(PNG_MAGIC + b"old")             # mark them as the previous render

    changed = Timeline(_state(), fps=first.fps)
    changed.tween(layer="seg", at=0.0, seconds=1.0, objectAlpha=0.5)     # a different fade
    written = record(changed, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=True)
    assert "differs from the one these frames were rendered with" in capsys.readouterr().err
    assert len(written) == changed.frame_count
    assert b"old" not in (tmp_path / "frame_00000.png").read_bytes()


def test_resuming_an_UNCHANGED_timeline_still_skips_the_work(tmp_path):
    """The check must not make resume useless — an interrupted long render is the point."""
    tl = _timeline()
    record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    assert record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=True) == []


def test_rendering_fresh_rewrites_frames_that_are_already_there(tmp_path):
    """Iterating on timings must not need the output directory deleted by hand."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"old")
    assert record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=True) == []
    written = record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=False)
    assert len(written) == tl.frame_count
    assert b"old" not in (tmp_path / "frame_00000.png").read_bytes()


def test_pending_frames_does_not_build_the_states_up_front():
    """A long timeline is thousands of full states; holding them all costs far more than
    recomputing one per frame."""
    todo = pending_frames(_timeline(), "/nonexistent")
    assert all(len(item) == 3 and isinstance(item[2], str) for item in todo)


# --------------------------------------------------------------------------- #
# the capture queue
# --------------------------------------------------------------------------- #
def _queue(tmp_path, *, size=(64, 32), attempts=2, frames=2):
    from neuroglancer.tool.screenshot import CaptureScreenshotRequest
    tl = _timeline()
    items = [(i, i / tl.fps, frame_path(str(tmp_path), i)) for i in range(frames)]
    return _RequestQueue(tl, items, size=size, attempts=attempts,
                         request_type=CaptureScreenshotRequest)


def test_a_reply_of_the_right_size_is_written(tmp_path):
    q = _queue(tmp_path)
    request = next(iter(q))
    request.response_callback(_Reply(64, 32))
    assert is_written(q.written[0])


def test_a_wrong_sized_reply_is_NOT_written_and_the_frame_is_requeued(tmp_path):
    """`async_screenshot`, which the capture loop uses, has no size retry — where
    `Viewer.screenshot` retries five times. Without this, a browser window smaller than the
    requested size silently yields frames at two sizes and ffmpeg finds out hours later."""
    q = _queue(tmp_path)
    request = next(iter(q))
    request.response_callback(_Reply(60, 30))
    assert q.written == []
    assert not os.path.exists(frame_path(str(tmp_path), 0))
    assert q.queue[0][0] == 0                                # back at the front


def test_a_frame_that_keeps_coming_back_wrong_sized_gives_up_and_says_why(tmp_path):
    q = _queue(tmp_path, attempts=2)
    for _ in range(2):
        next(iter(q)).response_callback(_Reply(60, 30))
    with pytest.raises(RenderProblem, match="window is probably smaller"):
        next(iter(q)).response_callback(_Reply(60, 30))


def test_the_config_callback_asks_for_the_requested_viewer_size(tmp_path):
    class _Config:
        viewer_size = None
    config = _Config()
    next(iter(_queue(tmp_path, size=(800, 600)))).config_callback(config)
    assert config.viewer_size == (800, 600)


def test_each_request_carries_ITS_OWN_frames_state(tmp_path):
    """Prefetch buffers several requests at once; a shared or late-bound state would render
    the same frame repeatedly and nothing would look wrong until playback."""
    q = _queue(tmp_path, frames=3)
    it = iter(q)
    states = [next(it).state for _ in range(3)]
    # Frame 0 sits exactly on the tween's start, where it contributes nothing and the layer
    # carries no objectAlpha at all — so the default stands in, as it does at render time.
    alphas = [next(lyr for lyr in s["layers"] if lyr["name"] == "seg").get("objectAlpha", 1.0)
              for s in states]
    assert len(set(alphas)) == 3


def test_the_queue_stops_when_it_runs_out(tmp_path):
    q = _queue(tmp_path, frames=1)
    it = iter(q)
    next(it).response_callback(_Reply(64, 32))
    with pytest.raises(StopIteration):
        next(it)


# --------------------------------------------------------------------------- #
# record: the parts reachable without a browser
# --------------------------------------------------------------------------- #
def test_an_odd_width_or_height_is_refused_with_the_reason(tmp_path):
    """libx264 under yuv420p needs even dimensions and the error it gives is unhelpful."""
    with pytest.raises(RenderProblem, match="even"):
        record(_timeline(), str(tmp_path), size=(1921, 1080))
    with pytest.raises(RenderProblem, match="yuv420p"):
        record(_timeline(), str(tmp_path), size=(1920, 1081))


def test_a_fully_rendered_directory_needs_no_browser_at_all(tmp_path, capsys):
    """Re-running a finished render must not open a viewer and sit there waiting."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    assert record(tl, str(tmp_path), size=(64, 32)) == []
    assert "ffmpeg" in capsys.readouterr().err


def test_record_writes_the_timeline_beside_the_frames(tmp_path):
    """The sequence describes itself: fps, duration and the base state it was rendered from,
    so the timings can be edited and re-rendered without re-deriving them."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    record(tl, str(tmp_path), size=(64, 32))
    written = json.loads((tmp_path / "timeline.json").read_text())
    assert written["fps"] == tl.fps
    assert Timeline.from_json(written).frame_count == tl.frame_count


def test_the_state_is_shaped_ONCE_before_interpolation_not_per_frame(tmp_path):
    """Two places shaping a frame is how the two drift, and the result still renders."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    record(tl, str(tmp_path), size=(64, 32))
    written = json.loads((tmp_path / "timeline.json").read_text())
    assert written["base"]["showAxisLines"] is False          # shaped
    assert tl.base.get("showAxisLines") is None               # the caller's is untouched


# --------------------------------------------------------------------------- #
# waiting for a browser
# --------------------------------------------------------------------------- #
def test_the_probe_PUSHES_A_STATE_because_an_empty_viewer_is_never_ready(tmp_path):
    """The trap this exists to prevent, measured against a real headless browser.

    A screenshot reply proves a browser is attached, so probing an empty viewer looks like the
    cheapest possible check — no layers, nothing to load, instant answer. It is the opposite:
    the client's `maybeSendScreenshot` bails with `if (!viewer.isReady() && !force) { ... }`,
    and a viewer holding no layers and no dimensions never becomes ready. Measured: an empty
    viewer gave 44 statistics messages and NO reply in 45s; the same viewer answered in 0.1s
    once any state was pushed. Requiring `state` is what stops that coming back.
    """
    viewer = _FakeViewer()
    state = _state()
    wait_for_browser(viewer, state)
    assert viewer.states == [state], "the state must be pushed before the probe"
    assert viewer.probes == 1


def test_waiting_for_a_browser_changes_NOTHING_about_the_viewers_config():
    """`viewerSize` is not a cheap way to make a probe screenshot small.

    The client sets the container's CSS width and height to it and then scales the whole
    thing to fill the window — `transform: scale(min(clientWidth/w, clientHeight/h))`. A
    64x64 viewer in a 1900px window is neuroglancer's own UI magnified thirty times: panel
    borders as thick as a finger and letters a foot high. Someone opening that link sees what
    looks like catastrophically broken data and has no way to tell it is a viewport setting.
    So the probe touches no config at all; the capture loop sets size and chrome per frame,
    which is where they belong.
    """
    viewer = _FakeViewer()
    wait_for_browser(viewer, _state())
    assert viewer.config_touched == 0
    assert viewer.config.viewer_size is None
    assert viewer.config.show_ui_controls is None


def test_a_browser_that_never_attaches_times_out_and_names_the_bind_hint():
    """A viewer on loopback is unreachable from another machine, and the URL looks fine."""
    viewer = _FakeViewer(url="http://127.0.0.1:8080/v/tok/", answers=False)
    with pytest.raises(RenderProblem, match="loopback"):
        wait_for_browser(viewer, _state(), timeout=0.3, notify_every=0.1)


def test_the_bind_hint_is_omitted_when_the_url_already_names_a_host():
    viewer = _FakeViewer(url="http://workstation.example.org:8080/v/tok/", answers=False)
    with pytest.raises(RenderProblem, match="no browser attached") as excinfo:
        wait_for_browser(viewer, _state(), timeout=0.3, notify_every=0.1)
    assert "loopback" not in str(excinfo.value)


def test_STATISTICS_alone_prove_a_browser_is_attached_so_slow_is_not_disconnected():
    """Statistics travel the same POST /action channel a reply does.

    So one arriving is proof the client can reach the server, and a scene that is merely slow
    to load must not be reported as a browser that never showed up — which is what sent the
    first diagnosis of this chasing a network problem that did not exist.
    """
    viewer = _FakeViewer(answers=False, statistics=[(300, 1900, 12)])
    with pytest.raises(RenderProblem, match="has sent nothing for") as excinfo:
        wait_for_browser(viewer, _state(), timeout=0.2, notify_every=0.1, stall_timeout=0.25)
    assert "300/1900" in str(excinfo.value)
    assert "no browser attached" not in str(excinfo.value)


def test_loading_progress_is_reported_rather_than_a_bare_wait(capsys):
    viewer = _FakeViewer(answers=False, statistics=[(300, 1900, 12)])
    with pytest.raises(RenderProblem):
        wait_for_browser(viewer, _state(), timeout=5.0, notify_every=0.1, stall_timeout=0.25)
    err = capsys.readouterr().err
    assert "browser attached" in err and "300/1900 chunks" in err


# --------------------------------------------------------------------------- #
# the capture loop, end to end against a fake viewer
# --------------------------------------------------------------------------- #
def test_record_renders_every_frame_and_hands_the_tab_back_as_a_NORMAL_viewer(tmp_path):
    """A `viewerSize` left set outlives the render: the tab stays pinned at a fixed pixel
    size and CSS-scaled to fit, which reads as a rendering fault rather than a leftover."""
    tl = _timeline()
    written = record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    assert len(written) == tl.frame_count
    assert all(is_written(p) for p in written)


def test_record_restores_the_viewer_even_when_the_render_FAILS(tmp_path):
    """The restore is in a finally block, so a killed render still leaves a usable tab."""
    viewer = _FakeViewer(reply_size=(11, 11))          # never the requested size
    with pytest.raises(RenderProblem, match="window is probably smaller"):
        record(_timeline(), str(tmp_path), size=(64, 32), viewer=viewer, size_attempts=1)
    assert viewer.config.viewer_size is None
    assert viewer.config.show_ui_controls is True


def test_record_pushes_the_opening_frame_before_capturing(tmp_path):
    """A blank tab gives no sign anything is happening, and every mesh in the scene would
    otherwise start loading inside the first capture — which is the slow one regardless."""
    viewer = _FakeViewer()
    tl = _timeline()
    record(tl, str(tmp_path), size=(64, 32), viewer=viewer)
    # The SHAPED opening frame — chrome already off, so what appears in the tab is what the
    # first captured frame will be rather than a version with the panels still up.
    shaped = Timeline(base=render_state(tl.base), fps=tl.fps, tweens=list(tl.tweens))
    assert viewer.states[0] == shaped.at(0.0)
    assert viewer.states[0]["showAxisLines"] is False


# --------------------------------------------------------------------------- #
# progress reporting
# --------------------------------------------------------------------------- #
def test_a_progress_bar_SWALLOWS_the_capture_loops_per_frame_lines(tmp_path, capsys):
    """tqdm writes `\\r` with no newline, so anything else printing lands on the bar's line.

    NOTES-neu-mark.md records exactly this against neuclease's bars: the output becomes one
    unreadable concatenated line. So quiet mode discards the capture loop's chatter rather
    than merely redirecting it — a bar owns the terminal or there is no bar.
    """
    record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer(), verbose=False)
    err = capsys.readouterr().err
    assert "Requesting screenshot" not in err
    assert "frame" in err                       # the bar itself still says something


def test_verbose_keeps_every_line_and_shows_no_bar(tmp_path, capsys):
    record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer(), verbose=True)
    err = capsys.readouterr().err
    assert err.count("Requesting screenshot") == _timeline().frame_count
    assert "frame/s]" not in err                # no tqdm bar


def test_a_FAILED_quiet_render_still_prints_the_last_lines_it_swallowed(tmp_path, capsys):
    """Quiet must not mean losing the evidence — otherwise a failure reports a bare exception
    with the diagnostic output discarded."""
    viewer = _FakeViewer(reply_size=(11, 11))
    with pytest.raises(RenderProblem):
        record(_timeline(), str(tmp_path), size=(64, 32), viewer=viewer, size_attempts=1,
               verbose=False)
    err = capsys.readouterr().err
    assert "last lines from the capture loop" in err
    assert "Requesting screenshot" in err


def test_without_tqdm_the_fallback_is_periodic_lines_with_an_ETA(tmp_path, capsys, monkeypatch):
    """tqdm is opportunistic, never declared — the same treatment shaders.py gives matplotlib.

    And the fallback is not a degraded mode: in a log file or a batch job, periodic lines with
    an ETA are what you want, where a `\\r` bar renders as thousands of concatenated lines.
    """
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer(), verbose=False,
           progress_interval=0.0)
    err = capsys.readouterr().err
    assert "s/frame" in err and "min left" in err
    assert "frame/s]" not in err


def test_the_tail_buffer_is_BOUNDED_so_a_stalled_render_cannot_grow_it():
    """A stalled render emits a statistics line a second, indefinitely."""
    tail = _Tail(keep=3)
    for i in range(500):
        tail.write(f"line {i}\n")
    assert list(tail.lines) == ["line 497", "line 498", "line 499"]


# --------------------------------------------------------------------------- #
# the ffmpeg command
# --------------------------------------------------------------------------- #
def test_the_ffmpeg_command_names_the_pattern_the_frame_rate_and_the_PIXEL_FORMAT():
    """Without `-pix_fmt yuv420p` the file encodes perfectly and will not play in QuickTime,
    PowerPoint or Slack — a discovery to make now rather than in front of an audience."""
    cmd = ffmpeg_command("/out", 24.0)
    assert "-pix_fmt yuv420p" in cmd
    assert "-framerate 24" in cmd
    assert os.path.join("/out", "frame_%05d.png") in cmd
    assert cmd.endswith(os.path.join("/out", "animation.mp4"))


def test_the_encode_hint_says_OUT_LOUD_that_nothing_was_encoded(capsys):
    """A bare command line printed among progress lines reads as a report, not an instruction.

    The reader finds no .mp4 afterwards and concludes the encode failed. It never ran — and
    deliberately so, because a render takes minutes and an encode takes seconds.
    """
    print_encode_hint("/out", 30.0)
    err = capsys.readouterr().err
    assert "NOTHING HAS BEEN ENCODED" in err
    assert "run this yourself" in err
    assert ffmpeg_command("/out", 30.0) in err


def test_the_ffmpeg_command_names_no_site_specific_module():
    """Publishing discipline, mechanically: a site's module names are as local as its data
    locations, and this string is printed to users and quoted into docs."""
    cmd = ffmpeg_command("/out", 30.0)
    assert "module" not in cmd
    assert "nix" not in cmd


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
def test_the_default_path_needs_no_selenium():
    """The webdriver route wants selenium and a matching chromedriver, neither of which is a
    given. A browser you open yourself needs nothing beyond the serve extra."""
    code = ("import neu_glance.rendering, sys; assert 'selenium' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_neuroglancer_import_in_the_package_is_DEFERRED():
    """No module imports neuroglancer at module scope. That is the rule that carries weight.

    The older wording — "`serving.py` is the only module that imports neuroglancer" — was a
    proxy for it, and `rendering` breaks the proxy while keeping the property: it needs
    `neuroglancer.tool.screenshot`, which belongs nowhere near `serving`. What actually
    matters is that `import neu_glance`, `neu-glance --help` and every pure code path stay off
    neuroglancer's import graph, and an indented import is exactly that. So the check is
    indentation, not location.
    """
    import neu_glance
    root = os.path.dirname(os.path.abspath(neu_glance.__file__))
    offenders, deferred = [], []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        for lineno, line in enumerate(open(os.path.join(root, name)), 1):
            if not line.strip().startswith(("import neuroglancer", "from neuroglancer")):
                continue
            (deferred if line[0] in " \t" else offenders).append(f"{name}:{lineno}")
    assert offenders == [], f"neuroglancer imported at module scope in {offenders}"
    assert deferred, "nothing imports neuroglancer any more — has the gate moved?"


# --------------------------------------------------------------------------- #
# launching a browser
# --------------------------------------------------------------------------- #
def test_finding_a_browser_prefers_an_explicit_binary_over_the_search(tmp_path):
    fake = tmp_path / "my-chrome"
    fake.write_text("#!/bin/sh\n")
    assert find_browser(str(fake)) == str(fake)
    assert find_browser("definitely-not-a-browser-anywhere") is None


def test_the_headless_launcher_needs_NO_selenium():
    """The realisation this whole path rests on. `neuroglancer.webdriver` uses selenium, so
    headless looked like it required selenium plus a matching chromedriver — but selenium is
    for *controlling* a page, and nothing here controls one. State changes and screenshot
    replies travel the viewer's own channel; the browser only has to stay loaded."""
    code = ("import neu_glance.rendering as r, sys; "
            "r.find_browser('nope'); "
            "assert 'selenium' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_launcher_cleans_up_its_throwaway_profile(tmp_path):
    """Without --user-data-dir a second render collides with the first's profile lock and
    attaches to the running instance, which then never loads the second viewer. The directory
    is per-render, so it has to go afterwards or they accumulate."""
    before = set(os.listdir(tempfile.gettempdir()))
    with headless_browser("http://localhost:1/v/x/", size=(64, 64), binary="/bin/sleep",
                          gpu=False) as process:
        assert process.pid > 0
    new = [n for n in set(os.listdir(tempfile.gettempdir())) - before
           if n.startswith("neu-glance-render-")]
    assert new == []


def test_the_launcher_sizes_the_window_to_the_RENDER(monkeypatch):
    """`viewerSize` scales its container to fit the window, so a window smaller than the frame
    means every screenshot comes back the wrong size and every frame is requeued."""
    seen = {}

    class _Fake:
        def __init__(self, command, **kw):
            seen["command"] = command
            self.pid = 1

        def terminate(self): pass

        def wait(self, timeout=None): return 0

    monkeypatch.setattr(subprocess, "Popen", _Fake)
    with headless_browser("http://x/", size=(1280, 720), binary="/bin/true"):
        pass
    assert "--window-size=1280,720" in seen["command"]
    assert seen["command"][-1] == "http://x/"


def test_the_gpu_flags_are_ON_by_default_because_software_rendering_is_10x_slower(monkeypatch):
    """Measured on one 1920x1080 mesh scene: 2.17 s/frame with SwiftShader, 0.21 with these.
    A render that silently falls back is a render that takes all afternoon and looks fine."""
    seen = {}

    class _Fake:
        def __init__(self, command, **kw):
            seen["command"] = command
            self.pid = 1

        def terminate(self): pass

        def wait(self, timeout=None): return 0

    monkeypatch.setattr(subprocess, "Popen", _Fake)
    with headless_browser("http://x/", size=(64, 64), binary="/bin/true"):
        pass
    assert all(flag in seen["command"] for flag in GPU_FLAGS)


def test_an_unknown_browser_mode_is_refused(tmp_path):
    with pytest.raises(RenderProblem, match="'auto', 'headless' or 'none'"):
        record(_timeline(), str(tmp_path), size=(64, 32), browser="chrome-ish")


def test_insisting_on_headless_with_none_installed_names_the_alternative(tmp_path):
    with pytest.raises(RenderProblem, match="browser='none'"):
        record(_timeline(), str(tmp_path), size=(64, 32), browser="headless",
               browser_binary="definitely-not-a-browser-anywhere")


def test_a_caller_supplying_its_own_viewer_launches_nothing(tmp_path, monkeypatch):
    """The spike scripts drive one viewer across several renders; launching a browser per
    call would leave one running per render."""
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("launched a browser"))
    assert record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer())
