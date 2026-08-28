"""GLSL shaders for annotation layers, and the rule for choosing one.

**A shader lives in the viewer state, not in the source**, so a link is the only place one
can be shipped — which is why these live here rather than in whatever wrote the annotations.
That separation is also the reason this package exists: the writer (`neu-mark
annotation-source`) has no business carrying a shader, and neu-vol has no business
knowing what a synapse is.

The one hard rule: **a shader naming a `prop_` the source does not declare fails to compile,
and neuroglancer then draws NOTHING**, with the error visible only in the layer's shader tab.
It does not fall back to a default. So :func:`pick_shader` checks a shader's properties
against the source's `info` and refuses rather than shipping one that cannot run.
"""

from __future__ import annotations

from typing import Any, Mapping

#: The endpoint markers, not the line, are what stays legible. A synapse is a few hundred
#: nanometres long, so at any zoom that shows more than one the line is sub-pixel — and a line
#: drawn in a blend of the two endpoint colours then swamps the markers and reads as a single
#: flat colour. So `show_pre`/`show_post` gate the two markers independently and the line is
#: drawn only when BOTH are on, which is also what makes one source usable as two layers.
#:
#: The confidence test is written as **discard-if-below**, never keep-if-at-or-above, and that
#: is load-bearing rather than stylistic. An unknown confidence is NaN, and per the dataset's
#: proofreaders an unknown conf usually means a synapse a HUMAN added — the most trustworthy
#: annotations in the set, whose confidence was simply never quantified. NaN fails every
#: comparison, so `discard if conf < min` leaves them visible at every threshold while
#: `keep if conf >= min` would hide them at all of them. The two forms are equivalent for real
#: numbers and opposite here.
SYNAPSE_SHADER = """\
#uicontrol bool show_pre checkbox(default=true)
#uicontrol bool show_post checkbox(default=true)

#uicontrol float pre_size slider(min=0.0, max=20.0, default=6.0)
#uicontrol float post_size slider(min=0.0, max=20.0, default=4.0)

#uicontrol vec3 pre_color color(default="#ff2000")
#uicontrol vec3 post_color color(default="#00c0ff")
#uicontrol vec3 line_color color(default="#ffffff")

#uicontrol float min_conf slider(min=0.0, max=1.0, default=0.0, step=0.01)

void main() {
  // An UNKNOWN confidence is NaN, which fails this comparison and so is never hidden.
  // That is deliberate: a synapse with no confidence value was usually added by hand by a
  // proofreader, making it MORE trustworthy rather than less. Keep this as discard-if-below.
  if (prop_conf_pre() < min_conf || prop_conf_post() < min_conf) discard;

  setEndpointMarkerColor(vec4(pre_color, 1.0), vec4(post_color, 1.0));
  setLineColor(line_color);

  // The line is only meaningful when both ends are shown; otherwise it points at something
  // deliberately hidden, and at these scales its colour would drown the markers.
  if (show_pre && show_post) {
    setLineWidth(1.0);
    setEndpointMarkerSize(pre_size, post_size);
  } else if (show_pre) {
    setLineWidth(0.0);
    setEndpointMarkerSize(pre_size, 0.0);
  } else if (show_post) {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, post_size);
  } else {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, 0.0);
  }
}
"""

#: Built-in shaders by name, each declaring the properties it reads so
#: :func:`pick_shader` can refuse an unrunnable pairing.
SHADERS: dict[str, dict[str, Any]] = {
    "synapse": {
        "properties": ("conf_pre", "conf_post"),
        "doc": "pre/post endpoint markers, independently toggleable, with a confidence "
               "threshold",
        "source": SYNAPSE_SHADER,
    },
}

#: `shaderControls` overrides for the two halves of a split pair. Set in the STATE rather than
#: by generating two shaders, so both layers carry the same code and a user editing one can see
#: exactly which control the other flipped.
SPLIT_CONTROLS = {
    "pre": {"show_pre": True, "show_post": False},
    "post": {"show_pre": False, "show_post": True},
}


class ShaderProblem(RuntimeError):
    """A shader was named that this source cannot feed, or cannot be found."""


def pick_shader(info: Mapping[str, Any], name: str | None,
                read_bytes=None) -> tuple[str | None, str | None]:
    """``(shader source, why)`` for an annotation layer.

    ``name`` may be a built-in name, a path to a file of GLSL, ``"none"``, or ``None`` to
    choose automatically — meaning the first built-in whose properties the source actually
    declares. Automatic rather than a fixed default, because a shader naming an absent
    ``prop_`` does not degrade: it fails to compile and the layer draws nothing.

    ``read_bytes`` is injected so this module needs no store access of its own; the CLI passes
    :func:`neu_vol.location.read_bytes`.
    """
    declared = {p["id"] for p in info.get("properties", []) or []}
    if name == "none":
        return None, None
    if name is None:
        for key, entry in SHADERS.items():
            if declared.issuperset(entry["properties"]):
                return entry["source"], f"auto-selected {key!r} ({entry['doc']})"
        return None, ("no built-in shader matches this source's properties "
                      + (f"({', '.join(sorted(declared))})" if declared else "(none)"))
    if name in SHADERS:
        entry = SHADERS[name]
        missing = sorted(set(entry["properties"]) - declared)
        if missing:
            raise ShaderProblem(
                f"shader {name!r} reads {', '.join(missing)}, which this source does not "
                f"declare. A shader naming a property that is absent fails to compile and "
                f"the layer draws nothing. Declared: "
                + (", ".join(sorted(declared)) or "no properties"))
        return entry["source"], f"built-in {name!r}"

    reader = read_bytes
    if reader is None:
        from neu_vol.location import read_bytes as reader

    raw = reader(name)
    if raw is None:
        raise ShaderProblem(
            f"shader {name!r} is neither a built-in name ("
            + ", ".join(SHADERS) + ", none) nor a readable file")
    return raw.decode(), f"from {name}"


# --------------------------------------------------------------------------- #
# Image shaders — a separate registry, and separate for a reason
#
# An annotation shader reads `prop_<name>()`, so `pick_shader` has to check those names
# against what the source declares or the layer silently draws nothing. An IMAGE shader
# reads `getDataValue(channel)`, which every image source has by construction: there is
# nothing to validate, and no way to name a channel that does not exist except by asking
# for one past the end — which is a shape error, catchable here.
#
# So these are chosen by DATA KIND and channel count, not by property matching, and
# `pick_shader`'s refusal logic deliberately does not apply to them.
# --------------------------------------------------------------------------- #

#: Window/level over one channel. The default for `--kind image`: EM data arrives with its
#: useful range in some arbitrary part of the dtype, and neuroglancer's own default maps
#: the whole range, which shows grey mud.
GRAYSCALE_SHADER = """\
#uicontrol invlerp normalized
#uicontrol float gamma slider(min=0.05, max=3.0, default=1.0, step=0.05)

void main() {
  float v = pow(normalized(), gamma);
  emitGrayscale(v);
}
"""

#: One channel through a two-colour gradient, with a floor. For a probability map, where
#: the interesting question is "where is this above t", not "what is the whole range".
#:
#: The threshold is written **discard-if-below**, never keep-if-at-or-above, for the reason
#: `SYNAPSE_SHADER` is: NaN fails every comparison, so `discard if v < t` leaves an
#: unscored voxel visible at every threshold while the inverted form would hide it at all
#: of them. A probability map from a model can carry NaN wherever it declined to predict.
COLORMAP_SHADER = """\
#uicontrol invlerp normalized
#uicontrol vec3 low_color color(default="#000060")
#uicontrol vec3 high_color color(default="#ffe000")
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.0, step=0.01)
#uicontrol float opacity slider(min=0.0, max=1.0, default=1.0, step=0.01)

void main() {
  float v = normalized();
  // discard-if-below, so a NaN voxel stays visible at every threshold
  if (v < threshold) discard;
  emitRGBA(vec4(mix(low_color, high_color, v), opacity));
}
"""

#: Three channels to three colours with independent gains. For a 3-channel probability map
#: — affinities, or a boundary/interior/background triple — where the useful view is all
#: three at once rather than one at a time.
RGB_SHADER = """\
#uicontrol vec3 color0 color(default="#ff2000")
#uicontrol vec3 color1 color(default="#00c000")
#uicontrol vec3 color2 color(default="#2060ff")
#uicontrol float gain0 slider(min=0.0, max=4.0, default=1.0, step=0.05)
#uicontrol float gain1 slider(min=0.0, max=4.0, default=1.0, step=0.05)
#uicontrol float gain2 slider(min=0.0, max=4.0, default=1.0, step=0.05)
#uicontrol float opacity slider(min=0.0, max=1.0, default=1.0, step=0.01)

void main() {
  vec3 rgb = color0 * (toNormalized(getDataValue(0)) * gain0)
           + color1 * (toNormalized(getDataValue(1)) * gain1)
           + color2 * (toNormalized(getDataValue(2)) * gain2);
  emitRGBA(vec4(clamp(rgb, 0.0, 1.0), opacity));
}
"""

#: Built-in image shaders by name. Unlike :data:`SHADERS` these declare no properties —
#: see the section comment — but they do declare how many channels they read, which is the
#: one thing that can be wrong about them.
IMAGE_SHADERS: dict[str, dict[str, Any]] = {
    "grayscale": {"channels": 1, "doc": "window/level with gamma, single channel",
                  "source": GRAYSCALE_SHADER},
    "colormap": {"channels": 1, "doc": "one channel through a two-colour gradient, with "
                                       "a discard-if-below threshold",
                 "source": COLORMAP_SHADER},
    "rgb": {"channels": 3, "doc": "three channels to three colours with independent gains",
            "source": RGB_SHADER},
}

#: Which shader a served layer gets when the caller names none.
_DEFAULT_IMAGE_SHADER = {"image": "grayscale", "probability": "colormap"}


def image_shader(name: str | None, *, kind: str = "image", channels: int = 1) -> str | None:
    """GLSL for an image-like layer, or ``None`` for neuroglancer's own default.

    ``name`` may be a built-in name, ``"none"``, or ``None`` to take the default for
    ``kind`` — grayscale for an image, a gradient for a probability map. A 3-channel array
    defaults to ``rgb`` whatever the kind, since showing one channel of three and calling
    it the picture is the more surprising choice.

    A shader/channel-count mismatch **raises**: `rgb` on a single-channel volume reads
    `getDataValue(1)` past the end, and a one-channel shader on a 3-channel volume shows a
    third of the data with nothing to say so.
    """
    if name == "none":
        return None
    if name is None:
        name = "rgb" if channels == 3 else _DEFAULT_IMAGE_SHADER.get(kind, "grayscale")
    entry = IMAGE_SHADERS.get(name)
    if entry is None:
        raise ShaderProblem(
            f"unknown image shader {name!r}; built-ins are "
            + ", ".join(IMAGE_SHADERS) + ", none")
    if entry["channels"] != channels:
        raise ShaderProblem(
            f"shader {name!r} reads {entry['channels']} channel(s) and this volume has "
            f"{channels}. Asking for a channel that is not there fails to compile, and "
            f"reading one of three shows a third of the data with nothing to say so.")
    return entry["source"]
