"""Render a :class:`neu_glance.animate.Timeline` to a sequence of PNGs, through a real viewer.

Neuroglancer renders in the browser, so a frame is a round trip: Python pushes a state, the
client loads whatever chunks that state makes visible, draws, and posts the image back.
:mod:`neu_glance.animate` decides *what* each frame is; this module is the part that needs a
browser, and it is deliberately the smaller half.

**A headless browser is the default, and it needs no selenium.** That was not obvious and
cost a while to notice: ``neuroglancer.webdriver`` uses selenium, so the headless path looked
like it required selenium plus a matching chromedriver. It does not. Selenium exists to
*control* a page, and nothing here controls one — every state change and every screenshot
reply travels the viewer's own channel, so the browser's entire job is to load a URL and stay
loaded. That is ``subprocess.Popen``. See :func:`headless_browser`.

It is also **faster than a browser on a desk**, which is the opposite of what one expects
from "software rendering, no window". Measured on one 1920x1080 mesh scene: 0.21 s/frame
headless on the local GPU, against 0.57 s/frame through a browser on another machine, which
pays a network round trip per frame. Headless without :data:`GPU_FLAGS` is 2.17 s/frame, so
those flags are most of it. ``record(..., browser="none")`` still prints a URL and waits, for
a machine with no browser installed or when you want to watch it work.

**Every neuroglancer import here is deferred**, inside the function that needs it, and the
missing-extra error comes from :func:`neu_glance.serving._neuroglancer` so there is one
message rather than two. This is the second module in the package to import neuroglancer at
all — the rule that matters is not "only ``serving.py`` does" but "no module does it at
*module scope*", so that ``import neu_glance``, ``neu-glance --help`` and every pure path stay
off that import graph. ``test_rendering.py`` checks the indentation to enforce it.

Three things this does that ``neuroglancer.tool.screenshot`` does not, each learned from the
way it fails:

* **A wrong-sized reply is never written.** ``async_screenshot`` has no size retry, where
  ``Viewer.screenshot`` retries five times — so a short window can silently yield frames at
  two different sizes, and ffmpeg finds out hours later.
* **Frames are written atomically and resume checks the PNG signature.** ``os.path.exists``
  alone treats a frame killed mid-write as done, and one corrupt frame in seven hundred is
  not noticed until the encode.
* **There is a cap on browser refreshes.** Upstream retries forever, so an unattended render
  against a dead tab hangs until someone looks.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any, Mapping, Sequence

from .animate import Timeline
from .serving import _neuroglancer

#: The first eight bytes of any PNG. A resumed render checks for these rather than trusting
#: that a file exists — see the module docstring.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Panels and chrome that must be off in a rendered frame. Set on the state, not per frame:
#: two places shaping a frame is how the two drift, and every such failure renders something
#: plausible rather than raising.
CHROME_OFF = {
    "selectedLayer": {"visible": False},
    "statistics": {"visible": False},
    "layerListPanel": {"visible": False},
    "helpPanel": {"visible": False},
}


class RenderProblem(RuntimeError):
    """A render could not start, or could not finish the frame it was on."""


class _Tail(io.TextIOBase):
    """A bounded sink for the capture loop's chatter: keeps the last few lines, drops the rest.

    Bounded rather than a plain buffer because a stalled render emits a statistics line per
    second indefinitely, and the point of quiet mode is not to trade a wall of text for a
    memory leak.
    """

    def __init__(self, keep: int = 40) -> None:
        self.lines: deque[str] = deque(maxlen=keep)
        self._partial = ""

    def write(self, text: str) -> int:
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.strip():
                self.lines.append(line.strip())
        return len(text)


class _Progress:
    """A tqdm bar, or periodic lines, or nothing — one interface over all three.

    **tqdm is used only if it happens to be installed**, the treatment `shaders.py` gives
    matplotlib: it is not declared anywhere in this package and must not become a dependency
    for a convenience. The fallback is not a degraded mode — periodic lines with an ETA are
    what you want in a log file or a batch job anyway, where a `\\r` bar renders as thousands
    of concatenated lines.

    The bar owns the terminal while it is up. `NOTES-neu-mark.md` records what happens
    otherwise: tqdm writes `\\r` without newlines, so anything else printing meanwhile lands
    on the same line and the result is unreadable. That is why `record` swallows the capture
    loop's per-frame prints whenever a bar is showing.
    """

    def __init__(self, total: int, *, verbose: bool = False, description: str = "rendering",
                 stream=None, min_interval: float = 10.0) -> None:
        self.total, self.verbose, self.min_interval = total, verbose, min_interval
        self.stream = stream or sys.stderr
        self.done = 0
        self.started = self.last = time.monotonic()
        self.bar = None
        if verbose:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print(f"{description}: {total} frames", file=self.stream)
        else:
            self.bar = tqdm(total=total, unit="frame", desc=description, file=self.stream,
                            dynamic_ncols=True)

    def advance(self, _path: str | None = None) -> None:
        self.done += 1
        if self.bar is not None:
            self.bar.update(1)
            return
        if self.verbose:
            return
        now = time.monotonic()
        if now - self.last < self.min_interval and self.done < self.total:
            return
        self.last = now
        per = (now - self.started) / max(1, self.done)
        left = per * (self.total - self.done)
        print(f"  {self.done}/{self.total} frames ({self.done / max(1, self.total):.0%}), "
              f"{per:.2f}s/frame, ~{left / 60:.1f} min left", file=self.stream)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


# --------------------------------------------------------------------------- #
# shaping the state
# --------------------------------------------------------------------------- #
def render_state(state: Mapping[str, Any], *, show_axis_lines: bool = False,
                 show_default_annotations: bool = False,
                 show_scale_bar: bool | None = None,
                 gpu_memory_limit: int | None = None,
                 system_memory_limit: int | None = None,
                 concurrent_downloads: int | None = None) -> dict:
    """A state with the viewer's furniture switched off, ready to be a frame.

    Applied **once to the base state before interpolation**, never per frame afterwards.

    The two annotation-ish defaults are inverted from neuroglancer's own on purpose: a viewer
    shows axis lines and each volume's bounding box because they help you navigate, and a
    rendered animation of meshes almost never wants either. They are still switchable, since a
    scale bar in particular is sometimes exactly the point.

    The three memory knobs matter once a scene is twenty-five mesh layers. Neuroglancer's own
    screenshot tool defaults them to 3 GiB / 3 GiB / 32; left alone here, so the viewer's
    defaults apply unless a caller has a reason.
    """
    out = json.loads(json.dumps(dict(state)))
    for key, value in CHROME_OFF.items():
        existing = out.get(key)
        out[key] = {**existing, **value} if isinstance(existing, dict) else dict(value)
    out["showAxisLines"] = bool(show_axis_lines)
    out["showDefaultAnnotations"] = bool(show_default_annotations)
    if show_scale_bar is not None:
        out["showScaleBar"] = bool(show_scale_bar)
    for key, value in (("gpuMemoryLimit", gpu_memory_limit),
                       ("systemMemoryLimit", system_memory_limit),
                       ("concurrentDownloads", concurrent_downloads)):
        if value is not None:
            out[key] = int(value)
    # A tool palette is UI, and it names layers — which a split will have emptied. Nothing
    # renders from it, and leaving it in a frame state is one more thing to go stale.
    out.pop("toolPalettes", None)
    return out


# --------------------------------------------------------------------------- #
# frames on disk
# --------------------------------------------------------------------------- #
def frame_path(out_dir: str, index: int, pattern: str = "frame_%05d.png") -> str:
    """``<out_dir>/frame_00042.png``. ``pattern`` must be usable verbatim by ``ffmpeg -i``."""
    try:
        name = pattern % index
    except TypeError:
        raise RenderProblem(
            f"frame pattern {pattern!r} has no integer field; ffmpeg reads a sequence through "
            f"one, so it needs something like 'frame_%05d.png'") from None
    return os.path.join(out_dir, name)


def is_written(path: str) -> bool:
    """True only for a file that exists **and begins with the PNG signature**.

    A frame killed mid-write exists and is zero or half length. Treating it as done leaves one
    corrupt frame in the middle of a sequence, which nothing notices until the encode — or
    worse, does not fail the encode and simply shows a torn frame.
    """
    try:
        with open(path, "rb") as f:
            return f.read(len(PNG_MAGIC)) == PNG_MAGIC
    except OSError:
        return False


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stale_frames(out_dir: str, frame_count: int,
                 pattern: str = "frame_%05d.png") -> list[str]:
    """Frame files numbered at or past ``frame_count`` — leftovers from a longer earlier run.

    **These silently corrupt the video, which is why they are worth hunting.** ``ffmpeg -i
    frame_%05d.png`` consumes the whole numbered sequence it finds; it has no idea which run
    wrote what. So shortening an animation and re-rendering leaves the tail of the previous
    one still on disk, and the encode appends it — a video that ends with a scene the timeline
    no longer contains, from an output directory where every individual frame is valid.
    """
    matcher = re.compile("^" + re.sub(r"%0?\d*d", r"(\\d+)", re.escape(pattern)
                                      .replace("\\%", "%")) + "$")
    out = []
    for name in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        found = matcher.match(name)
        if found and int(found.group(1)) >= frame_count:
            out.append(os.path.join(out_dir, name))
    return out


def pending_frames(timeline: Timeline, out_dir: str, *, resume: bool = True,
                   start_frame: int = 0, end_frame: int | None = None,
                   pattern: str = "frame_%05d.png") -> list[tuple[int, float, str]]:
    """``(index, seconds, path)`` for the frames still to render. States are built later.

    The state dicts are deliberately not built here: a long timeline is thousands of full
    states, and holding them all costs far more than recomputing one per frame.
    """
    stop = timeline.frame_count if end_frame is None else min(end_frame, timeline.frame_count)
    out = []
    for i in range(max(0, start_frame), stop):
        path = frame_path(out_dir, i, pattern)
        if resume and is_written(path):
            continue
        out.append((i, i / timeline.fps, path))
    return out


# --------------------------------------------------------------------------- #
# the browser
# --------------------------------------------------------------------------- #
def open_viewer(*, bind: str | None = None, port: int = 0):
    """A viewer for a browser to attach to. Returns it; its ``.get_viewer_url()`` is the link.

    ``bind`` and ``port`` **take effect only before the first viewer in the process**, the
    same once-only constraint :func:`neu_glance.serve` documents — neuroglancer keeps them in
    a module global read at server start. ``bind="0.0.0.0"`` makes the URL use the host's
    FQDN, which is what a browser on another machine needs.
    """
    ng = _neuroglancer()
    if bind is not None or port:
        if ng.server.is_server_running():
            print(f"a viewer server is already running in this process, so bind={bind!r} "
                  f"port={port!r} are ignored — neuroglancer reads them once, at server start",
                  file=sys.stderr)
        else:
            ng.set_server_bind_address(bind or "127.0.0.1", port)
    return ng.Viewer()


#: Browser binaries to look for, in preference order.
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium")

#: Flags that put WebGL on a real GPU instead of the software rasteriser.
#:
#: **This is the difference between a render and an afternoon.** Headless Chrome defaults to
#: SwiftShader, which draws correctly and slowly; measured on one 1920x1080 mesh scene:
#: 2.17 s/frame by default, **0.21 s/frame** with these. That is also 2.7x faster than the
#: same scene through a browser on someone's desk, which pays a round trip per frame.
GPU_FLAGS = ("--use-angle=vulkan", "--enable-features=Vulkan", "--enable-gpu",
             "--ignore-gpu-blocklist")

#: Flags that make Chrome survive being run headless as an ordinary user on a shared machine.
HEADLESS_FLAGS = ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-extensions", "--mute-audio")


def find_browser(binary: str | None = None) -> str | None:
    """A chrome/chromium executable, or ``None``. ``binary`` overrides the search."""
    if binary:
        return binary if os.path.exists(binary) else shutil.which(binary)
    for name in BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    return None


@contextlib.contextmanager
def headless_browser(url: str, *, size: tuple[int, int] = (1920, 1080),
                     binary: str | None = None, gpu: bool = True,
                     extra_args: Sequence[str] = ()):
    """Run a browser on ``url`` for the duration of the block, with no window and no tab.

    **No selenium and no chromedriver.** Those exist to *control* a page; nothing here
    controls it. The viewer's own channel carries every state change and every screenshot
    reply, so all the browser has to do is load the URL and stay loaded — which is
    ``subprocess.Popen``. Believing otherwise is what kept this an attended process for
    longer than it needed to be.

    The window is sized to the render, since ``viewerSize`` scales its container to fit and a
    window smaller than the frame means every screenshot comes back the wrong size.

    The profile is a throwaway directory, removed on the way out: without ``--user-data-dir``
    a second render collides with the first's profile lock and simply attaches to the running
    instance, which then never loads the second viewer.
    """
    executable = find_browser(binary)
    if executable is None:
        raise RenderProblem(
            f"no browser found (looked for {', '.join(BROWSERS)}). Pass binary=..., or render "
            f"attended by opening the viewer URL yourself — `record(..., browser='none')`.")
    profile = tempfile.mkdtemp(prefix="neu-glance-render-")
    runtime = os.path.join(profile, "runtime")
    os.makedirs(runtime, mode=0o700, exist_ok=True)
    command = [executable, *HEADLESS_FLAGS, f"--user-data-dir={profile}",
               f"--window-size={size[0]},{size[1]}", *(GPU_FLAGS if gpu else ()),
               *extra_args, url]

    # **A private XDG_RUNTIME_DIR, because a stale one is invisible and fatal.** Chrome puts
    # sockets there, and a long-lived tmux or screen server hands its children the environment
    # of the login that STARTED it — including an `XDG_RUNTIME_DIR` that systemd deletes once
    # that login's last session ends. The variable still points somewhere; the directory is
    # gone. Renders then work from a fresh shell and fail from tmux, which reads as anything
    # but an environment problem. A directory of our own removes the question.
    environment = {**os.environ, "XDG_RUNTIME_DIR": runtime}
    environment.pop("DBUS_SESSION_BUS_ADDRESS", None)

    log_path = os.path.join(profile, "browser.log")
    log = open(log_path, "wb")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log,
                               env=environment)
    print(f"launched {os.path.basename(executable)} headless"
          f"{' on the GPU' if gpu else ' (software rendering)'}", file=sys.stderr)

    def check() -> str | None:
        """``None`` while it is running; why it stopped once it has."""
        code = process.poll()
        if code is None:
            return None
        log.flush()
        try:
            with open(log_path, "rb") as f:
                tail = f.read()[-2000:].decode("utf-8", "replace").strip()
        except OSError:                                         # pragma: no cover
            tail = ""
        detail = ("\n    " + "\n    ".join(tail.splitlines()[-8:])) if tail else ""
        hint = ""
        if gpu:
            hint = ("\n    The GPU flags may not work here — retry with gpu=False "
                    "(--software-gl), which is slower but needs nothing from the driver.")
        return (f"{os.path.basename(executable)} exited with code {code} before the render "
                f"began.{hint}{detail}")

    try:
        yield check
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:                       # pragma: no cover
            process.kill()
        log.close()
        shutil.rmtree(profile, ignore_errors=True)


def _await_reply(viewer, *, timeout, notify_every, message, on_timeout, watch=None):
    """Block on one screenshot reply, reporting periodically. Raises what `on_timeout` builds.

    `watch` is checked each tick and short-circuits the wait. A browser that dies at startup
    otherwise costs the FULL connect timeout — ten minutes of a silent, motionless run before
    anything is said — and the message when it finally comes blames the wrong thing.
    """
    ready = threading.Event()
    viewer.async_screenshot(lambda _reply: ready.set())
    waited = 0.0
    while not ready.wait(max(0.1, notify_every)):
        waited += notify_every
        if watch is not None:
            stopped = watch()
            if stopped:
                raise RenderProblem(stopped)
        if waited >= timeout:
            raise on_timeout()
        print(f"{message} ({waited:g}s)", file=sys.stderr)


def wait_for_browser(viewer, state: Mapping[str, Any], *, timeout: float = 600.0,
                     notify_every: float = 15.0, stall_timeout: float = 180.0,
                     launched: bool = False, watch=None) -> None:
    """Push ``state``, then block until a browser has it open and loaded.

    ``launched=True`` when one was started for us: the wait is the same, but telling someone
    to open a link that is already open — and that nothing will ever be typed into — sends
    them looking for a step that does not exist, and makes a working headless render look
    stuck. The failure it reports differs too: a headless browser that never attaches means
    the process died or cannot reach the port, not that a tab was left unopened.

    Without this the first real frame simply sits there: ``capture_screenshots`` pushes a
    state and waits, and an unopened viewer is indistinguishable from a slow one until the
    refresh timeout fires with a message about the browser being unresponsive.

    **``state`` is required, and that is the whole lesson.** A screenshot reply proves a
    browser is attached, so an empty viewer looks like the cheapest possible probe — nothing
    to load, instant answer. It is the opposite: the client's ``maybeSendScreenshot`` bails
    out with ``if (!viewer.isReady() && !force) { sendStatistics(); return; }``, and a viewer
    holding **no layers and no dimensions never becomes ready**. Measured against a real
    headless browser: an empty viewer produced 44 statistics messages and no reply in 45 s,
    while the same viewer answered in 0.1 s once any state was pushed. Requiring the argument
    is what stops that being rediscovered.

    Two signals, and they mean different things:

    - **the first statistics message means a browser is attached** — statistics travel the
      same ``POST /action/<token>`` channel a reply does, so one arriving is proof the client
      can reach the server. Only the absence of *any* signal is a connection problem;
    - **the reply means the scene is loaded**, which is worth waiting for here because it is
      the cost of the first frame either way.

    So ``timeout`` bounds the wait for a browser, and after that ``stall_timeout`` bounds
    silence rather than slowness — the same distinction ``capture_screenshots`` draws, and for
    the same reason: to a wall clock a slow frame and a dead browser look identical.

    **This touches no config, deliberately.** The probe used to ask for 64x64 to be cheap.
    ``viewerSize`` sets the container's CSS width and height and then scales it to fill the
    window (``transform: scale(min(clientWidth / w, clientHeight / h))``), so a 64x64 viewer in
    a 1900px window is neuroglancer's own UI magnified thirty times: blocky panel borders and
    letters a foot high. Whoever opened that link reasonably concluded the data was broken.
    """
    url = viewer.get_viewer_url()
    if launched:
        print(f"waiting for the headless browser\n    {url}", file=sys.stderr)
    else:
        print(f"open this in a browser and leave it open:\n    {url}", file=sys.stderr)

    # **Attachment is proved against an EMPTY-BUT-VALID state, not the real scene.** A
    # screenshot reply needs the viewer to be ready, and a scene of several hundred meshes is
    # not ready for a long time — so waiting for the real one conflates "no browser" with "a
    # browser working hard", and prints `still waiting for a browser` for a minute while it
    # loads. A state with dimensions and no layers is ready almost at once (measured: 0.1 s),
    # which separates the two questions.
    empty = {"dimensions": state.get("dimensions") or {}, "layers": [],
             "layout": state.get("layout", "3d")}
    viewer.set_state(empty)
    _await_reply(viewer, timeout=timeout, notify_every=notify_every,
                 message=(f"still waiting for {'the headless browser' if launched else 'a browser'}"
                          + ("" if launched else f":\n    {url}")),
                 on_timeout=lambda: RenderProblem(
                     f"no browser attached after {timeout:g}s.\n    {url}" + (
                         "\n    The browser was started here, so it has died or cannot reach "
                         "the viewer port. Try browser='none' to open one yourself, or "
                         "gpu=False if the GPU flags are not supported." if launched else
                         "\n    The viewer is bound to loopback, so only a browser on this "
                         "machine can reach it — pass bind='0.0.0.0' or forward the port."
                         if "localhost" in url or "127.0.0.1" in url else "")),
                 watch=watch)
    print("browser attached; loading the scene", file=sys.stderr)

    viewer.set_state(dict(state))
    loaded = threading.Event()
    progress: dict[str, Any] = {"at": time.monotonic(), "loaded": 0, "total": 0, "best": 0,
                                "downloading": 0}

    def on_statistics(statistics):
        total = statistics.total
        got = total.visible_chunks_gpu_memory
        if got > progress["best"]:                 # PROGRESS, not merely a message
            progress["best"] = got
            progress["at"] = time.monotonic()
        progress.update(loaded=got, total=total.visible_chunks_total,
                        downloading=total.visible_chunks_downloading)

    viewer.async_screenshot(lambda _reply: loaded.set(), statistics_callback=on_statistics)

    bar = _Progress(0, verbose=True)               # placeholder until a total is known
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    else:
        bar = tqdm(total=0, unit="chunk", desc="loading the scene", file=sys.stderr,
                   dynamic_ncols=True)
    try:
        while not loaded.wait(1.0 if tqdm else max(0.1, notify_every)):
            if watch is not None:
                stopped = watch()
                if stopped:
                    raise RenderProblem(stopped)
            # **Stalled means NO PROGRESS, not silence.** Statistics keep arriving once a
            # second whatever happens, so a wait that watches for silence never fires — and
            # the thing it needs to catch sits at `N-1 / N chunks, 0 downloading` forever.
            silent = time.monotonic() - progress["at"]
            if silent >= stall_timeout and progress["total"]:
                missing = progress["total"] - progress["loaded"]
                raise RenderProblem(
                    f"the scene stopped loading {silent:.0f}s ago at "
                    f"{progress['loaded']}/{progress['total']} chunks, "
                    f"{progress['downloading']} downloading, and the viewer will never report "
                    f"itself ready — so no frame can be captured.\n"
                    f"    {missing} chunk(s) short. The usual cause is a SEGMENT WITH NO MESH: "
                    f"neuroglancer counts it among the visible chunks and waits for it "
                    f"forever. `neu_glance.sources.segments_with_meshes(volume, ids)` filters "
                    f"those out — measured at 3s for 2200 bodies.")
            if tqdm:
                if progress["total"] and bar.total != progress["total"]:
                    bar.total = progress["total"]
                bar.n = progress["loaded"]
                bar.set_postfix_str(f"{progress['downloading']} downloading", refresh=False)
                bar.refresh()
            elif progress["total"]:
                print(f"  loading: {progress['loaded']}/{progress['total']} chunks "
                      f"({progress['downloading']} downloading)", file=sys.stderr)
    finally:
        if tqdm:
            bar.close()
    print("scene loaded; rendering", file=sys.stderr)


# --------------------------------------------------------------------------- #
# the render
# --------------------------------------------------------------------------- #
class _RequestQueue:
    """Feeds :func:`capture_screenshots`, and can put a frame back.

    A frame comes back when the reply is the wrong size. That is transient rather than fatal —
    ``Viewer.screenshot`` retries five times for the same reason — but ``async_screenshot``,
    which the capture loop uses, does not retry at all. So the retry lives here.
    """

    def __init__(self, timeline, items, *, size, attempts, request_type, on_write=None):
        self.timeline = timeline
        self.queue = deque(items)
        self.size = size
        self.attempts = attempts
        self.request_type = request_type
        self.on_write = on_write
        self.tries: dict[int, int] = {}
        self.written: list[str] = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue:
            raise StopIteration
        index, seconds, path = self.queue.popleft()
        width, height = self.size

        def config_callback(s):
            s.viewer_size = (width, height)

        def response_callback(reply):
            if (reply.width, reply.height) != (width, height):
                self.tries[index] = self.tries.get(index, 0) + 1
                if self.tries[index] > self.attempts:
                    raise RenderProblem(
                        f"frame {index} came back {reply.width}x{reply.height} instead of "
                        f"{width}x{height}, {self.attempts} times running. The browser window "
                        f"is probably smaller than the requested size — enlarge it, or render "
                        f"smaller. Frames already written are kept; re-run to resume")
                print(f"frame {index}: got {reply.width}x{reply.height}, wanted "
                      f"{width}x{height} — requeued", file=sys.stderr)
                self.queue.appendleft((index, seconds, path))
                return
            _write_atomic(path, reply.image)
            self.written.append(path)
            if self.on_write is not None:
                self.on_write(path)

        return self.request_type(
            state=self.timeline.at(seconds),
            description=f"frame {index} @ {seconds:.3f}s",
            config_callback=config_callback,
            response_callback=response_callback)


def record(timeline: Timeline, out_dir: str, *, size: tuple[int, int] = (1920, 1080),
           viewer=None, bind: str | None = None, port: int = 0,
           prefetch: int = 1, refresh_timeout: float = 120.0, give_up_after: int = 5,
           connect_timeout: float = 600.0, size_attempts: int = 5,
           resume: bool = True, start_frame: int = 0, end_frame: int | None = None,
           pattern: str = "frame_%05d.png", shape: bool = True,
           verbose: bool = False, progress_interval: float = 10.0,
           browser: str = "auto", browser_binary: str | None = None,
           gpu: bool = True) -> list[str]:
    """Render ``timeline`` into ``out_dir``. Returns the frame paths written this run.

    ``verbose=True`` prints the capture loop's own line per frame and per statistics message;
    the default shows a progress bar (or periodic lines with an ETA where tqdm is not
    installed) and keeps only the last few of those, printing them if the render fails.

    ``browser`` decides who draws. ``"auto"`` (the default) runs a **headless** chrome or
    chromium if one is on the PATH and falls back to printing a URL for you to open if not;
    ``"headless"`` insists and raises if none is found; ``"none"`` always waits for you.
    Headless is the default because it is both unattended and *faster* — measured on one
    1920x1080 mesh scene, 0.21 s/frame against 0.57 s/frame through a browser on a desk
    elsewhere, which pays a network round trip per frame. See :func:`headless_browser`.

    Writes ``timeline.json`` beside the frames, so the sequence describes itself and the
    timings can be edited and re-rendered without re-deriving them.

    Interrupting at any point is safe — every written frame is complete — and re-running
    resumes. ``start_frame``/``end_frame`` render a slice with the *same* numbering, so a fade
    section at one frame rate and a camera move at another can share a directory.
    """
    _neuroglancer()          # fail on the missing extra before anything else, as `serve` does

    width, height = (int(v) for v in size)
    for label, value in (("width", width), ("height", height)):
        if value % 2:
            raise RenderProblem(
                f"{label} must be even ({value} is not): libx264 with the yuv420p pixel "
                f"format needs even dimensions, and the error it gives otherwise is unhelpful")

    os.makedirs(out_dir, exist_ok=True)
    for note in timeline.check():
        print(note, file=sys.stderr)

    shaped = timeline
    if shape:
        shaped = Timeline(base=render_state(timeline.base), fps=timeline.fps,
                          tweens=list(timeline.tweens), groups=dict(timeline.groups),
                          cursor=timeline.cursor)
        shaped.duration = timeline.duration

    # **Resuming is only correct if the timeline has not changed**, and nothing else checks
    # that. Nudge a timing, re-run, and the frames already on disk were rendered from the OLD
    # animation — kept, mixed invisibly with new ones, and every single one a valid PNG. That
    # is worse than the stale tail below, because it is in the middle of the sequence where
    # nothing marks it. The timeline written beside the frames is the fingerprint that makes
    # it detectable, so compare and fall back to a full render rather than trusting the flag.
    recorded = os.path.join(out_dir, "timeline.json")
    if resume and os.path.exists(recorded):
        try:
            with open(recorded) as f:
                previous = json.load(f)
        except (OSError, ValueError):
            previous = None
        if previous != shaped.to_json():
            print("the timeline differs from the one these frames were rendered with — "
                  "re-rendering all of them rather than mixing two animations", file=sys.stderr)
            resume = False

    # Leftovers from a longer earlier run go, whether or not this one is resuming: ffmpeg
    # reads the whole numbered sequence, so leaving them appends the end of a previous
    # animation to this one. Removed rather than refused, and named out loud — a warning about
    # a correctness problem the user cannot see in any single frame is a warning that gets
    # scrolled past.
    leftovers = stale_frames(out_dir, shaped.frame_count, pattern)
    if leftovers:
        print(f"removing {len(leftovers)} frame(s) past the end of this timeline "
              f"({os.path.basename(leftovers[0])}..{os.path.basename(leftovers[-1])}) — they "
              f"are from a longer earlier run and ffmpeg would append them", file=sys.stderr)
        for path in leftovers:
            os.remove(path)

    todo = pending_frames(shaped, out_dir, resume=resume, start_frame=start_frame,
                          end_frame=end_frame, pattern=pattern)
    total = shaped.frame_count if end_frame is None else end_frame - start_frame
    _write_atomic(os.path.join(out_dir, "timeline.json"),
                  json.dumps(shaped.to_json(), indent=2).encode())
    print(f"{shaped.duration:g}s at {shaped.fps:g} fps = {shaped.frame_count} frames at "
          f"{width}x{height}; {len(todo)} to render"
          + (f" ({total - len(todo)} already present)" if resume and total > len(todo) else ""),
          file=sys.stderr)
    if not todo:
        print_encode_hint(out_dir, shaped.fps, pattern=pattern)
        return []

    # The opening frame, shown the moment someone is watching rather than a blank viewer that
    # gives no sign the run is alive — and it is what makes the viewer READY, without which no
    # screenshot request is ever answered. It also starts every mesh loading now instead of
    # inside the first capture, which is the slow one either way.
    opening = shaped.at(todo[0][1])
    capture = dict(shaped=shaped, todo=todo, out_dir=out_dir, size=(width, height),
                   pattern=pattern, prefetch=prefetch, refresh_timeout=refresh_timeout,
                   give_up_after=give_up_after, size_attempts=size_attempts,
                   verbose=verbose, progress_interval=progress_interval)

    if viewer is not None:                       # a caller driving its own viewer
        viewer.set_state(opening)
        return _capture(viewer, **capture)

    if browser not in ("auto", "headless", "none"):
        raise RenderProblem(f"browser must be 'auto', 'headless' or 'none', got {browser!r}")
    executable = None if browser == "none" else find_browser(browser_binary)
    if browser == "headless" and executable is None:
        raise RenderProblem(
            f"browser='headless' but none was found (looked for {', '.join(BROWSERS)}). Pass "
            f"browser_binary=..., or browser='none' to open one yourself.")

    viewer = open_viewer(bind=bind, port=port)
    if executable is None:
        wait_for_browser(viewer, opening, timeout=connect_timeout)
        return _capture(viewer, **capture)
    with headless_browser(viewer.get_viewer_url(), size=(width, height),
                          binary=browser_binary, gpu=gpu) as alive:
        wait_for_browser(viewer, opening, timeout=connect_timeout, launched=True, watch=alive)
        return _capture(viewer, **capture)


def _capture(viewer, *, shaped, todo, out_dir, size, pattern, prefetch, refresh_timeout,
             give_up_after, size_attempts, verbose, progress_interval) -> list[str]:
    """The loop itself, once something is drawing. Shared by every browser arrangement."""
    from neuroglancer.tool.screenshot import CaptureScreenshotRequest, capture_screenshots

    width, height = size
    refreshes = {"n": 0}

    def refresh_browser_callback():
        refreshes["n"] += 1
        if refreshes["n"] > give_up_after:
            raise RenderProblem(
                f"the browser sent no progress for {refresh_timeout:g}s, {give_up_after} times "
                f"running. Frames already written are kept — reload {viewer.get_viewer_url()} "
                f"and re-run to resume")
        print(f"no progress for {refresh_timeout:g}s ({refreshes['n']}/{give_up_after}) — "
              f"reload the tab if it has stopped responding", file=sys.stderr)

    progress = _Progress(len(todo), verbose=verbose, min_interval=progress_interval,
                         description=f"rendering {width}x{height}")
    queue = _RequestQueue(shaped, todo, size=(width, height), attempts=size_attempts,
                          request_type=CaptureScreenshotRequest, on_write=progress.advance)

    # `capture_screenshots` reports on stdout, where this package puts only payload — and its
    # statistics callback runs on the tornado thread, so the redirect has to be held for the
    # whole render rather than around each call.
    #
    # Where it goes depends on `verbose`, and quiet mode SWALLOWS rather than forwards. A
    # progress bar owns the terminal: tqdm writes `\r` with no newline, so anything else
    # printing meanwhile lands on the same line (`NOTES-neu-mark.md` records exactly this
    # against neuclease's bars). The last few lines are kept and printed if the render fails,
    # so quiet never means losing the evidence.
    tail = _Tail()
    try:
        with contextlib.redirect_stdout(sys.stderr if verbose else tail):
            capture_screenshots(viewer, iter(queue), refresh_browser_callback,
                                refresh_timeout, num_to_prefetch=prefetch)
    except BaseException:
        progress.close()
        if tail.lines:
            print("\n--- last lines from the capture loop ---", file=sys.stderr)
            for line in tail.lines:
                print(f"    {line}", file=sys.stderr)
        raise
    finally:
        progress.close()
        # Hand the tab back as an ordinary viewer. A `viewerSize` left set keeps the container
        # at a fixed pixel size and CSS-scaled to fit the window, which at any size but the
        # window's own looks like a rendering fault rather than a leftover setting. `None`
        # restores the responsive layout.
        with contextlib.suppress(Exception):
            with viewer.config_state.txn() as s:
                s.viewer_size = None
                s.show_ui_controls = True

    print(f"wrote {len(queue.written)} frame(s) to {out_dir}", file=sys.stderr)
    print_encode_hint(out_dir, shaped.fps, pattern=pattern)
    return queue.written


def print_encode_hint(out_dir: str, fps: float, *, pattern: str = "frame_%05d.png",
                      file=None) -> None:
    """Say, unmistakably, that the frames are frames and the encode is the reader's to run.

    The command on its own is ambiguous in the worst way: printed among progress lines with no
    verb in front of it, it reads as a report of something that already happened, and the
    obvious conclusion on finding no ``.mp4`` is that the encode failed. It did not run.
    """
    print(f"\nRendering is done — the output is a directory of PNG frames.\n"
          f"NOTHING HAS BEEN ENCODED: run this yourself to get a video.\n"
          f"(ffmpeg is often not on PATH — on a cluster it is usually a module.)\n\n"
          f"    {ffmpeg_command(out_dir, fps, pattern=pattern)}\n",
          file=file or sys.stderr)


def ffmpeg_command(out_dir: str, fps: float, *, output: str | None = None, crf: int = 18,
                   pattern: str = "frame_%05d.png") -> str:
    """The command that turns the frames into an mp4. Printed, never run.

    **Not run** because ``ffmpeg`` is very often not on ``PATH`` — on an HPC site it is a
    module, and ``module`` is a shell function rather than an executable, so there is no
    portable way to reach it from here. Separating the two is also just better: a render takes
    hours and an encode takes seconds, so a failed encode should cost nothing, and re-encoding
    at a different quality should not mean re-rendering.

    ``-pix_fmt yuv420p`` is not optional. Without it the file encodes perfectly and then will
    not play in QuickTime, PowerPoint or Slack, which is a discovery to make now rather than
    in front of an audience.
    """
    dest = output or os.path.join(out_dir, "animation.mp4")
    return (f"ffmpeg -y -framerate {fps:g} -i {os.path.join(out_dir, pattern)} "
            f"-c:v libx264 -pix_fmt yuv420p -crf {crf} {dest}")
