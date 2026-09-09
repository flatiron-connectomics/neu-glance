"""Naming segments, and finding out where they are.

Both exist so an animation can be written about `"LAL(L)"` rather than `22`, and can point a
camera at it. Both fail the same quiet way if they are wrong: a name that does not resolve
drops a region out of a shot, and a box read from the wrong place frames empty space — and a
render succeeds either way, at which point the mistake is a video nobody can explain.

Fixtures are built by hand rather than by a writer, because this package does not WRITE
segment properties or legacy meshes. It reads what other people published.
"""

import json
import struct

import numpy as np
import pytest

from neu_glance.sources import (SourceProblem, segment_boxes, segment_ids, segment_labels)


def _volume(tmp_path, *, labels=None, meshes=None, mesh_type="neuroglancer_legacy_mesh",
            properties_dir="segment_properties", transform=None):
    root = tmp_path / "vol"
    root.mkdir(exist_ok=True)
    info = {"@type": "neuroglancer_multiscale_volume", "scales": []}

    if labels is not None:
        info["segment_properties"] = properties_dir
        (root / properties_dir).mkdir(exist_ok=True)
        (root / properties_dir / "info").write_text(json.dumps({
            "@type": "neuroglancer_segment_properties",
            "inline": {"ids": list(labels),
                       "properties": [{"id": "label", "type": "label",
                                       "values": [labels[k] for k in labels]}]}}))
    if meshes is not None:
        info["mesh"] = "mesh"
        (root / "mesh").mkdir(exist_ok=True)
        mesh_info = {"@type": mesh_type}
        if transform is not None:
            mesh_info["transform"] = list(transform)
        (root / "mesh" / "info").write_text(json.dumps(mesh_info))
        for body, vertices in meshes.items():
            v = np.asarray(vertices, np.float32)
            faces = np.zeros((1, 3), np.uint32)
            (root / "mesh" / f"{body}.ngmesh").write_bytes(
                struct.pack("<I", len(v)) + v.tobytes() + faces.tobytes())
            (root / "mesh" / f"{body}:0").write_text(
                json.dumps({"fragments": [f"{body}.ngmesh"]}))
    (root / "info").write_text(json.dumps(info))
    return str(root)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def test_labels_come_back_paired_with_their_ids(tmp_path):
    vol = _volume(tmp_path, labels={"20": "LA(L)", "22": "LAL(L)"})
    assert segment_labels(vol) == {"20": "LA(L)", "22": "LAL(L)"}


def test_names_resolve_to_ids_in_the_order_asked(tmp_path):
    """Order matters: a group lists its regions in the order they should appear."""
    vol = _volume(tmp_path, labels={"20": "LA(L)", "22": "LAL(L)", "30": "ME(L)"})
    assert segment_ids(vol, ["ME(L)", "LA(L)"]) == ["30", "20"]


def test_a_name_the_source_does_not_have_RAISES_and_suggests(tmp_path):
    """A mistyped region silently dropped is a shot that renders perfectly and is missing a
    piece, which nobody notices until the video is watched."""
    vol = _volume(tmp_path, labels={"22": "LAL(L)", "23": "LAL(R)"})
    with pytest.raises(SourceProblem, match="did you mean"):
        segment_ids(vol, ["LAL(X)"])


def test_a_label_shared_by_several_segments_refuses_to_guess(tmp_path):
    """Labels are not unique in this format, so picking one would be arbitrary."""
    vol = _volume(tmp_path, labels={"1": "PB", "2": "PB"})
    with pytest.raises(SourceProblem, match="more than one segment"):
        segment_ids(vol, ["PB"])


def test_ids_and_labels_of_different_lengths_are_refused(tmp_path):
    """The format pairs them BY POSITION, so a mismatch mislabels every segment after it —
    zipping short would hand back confident, wrong names."""
    vol = _volume(tmp_path, labels={"1": "a", "2": "b"})
    properties = tmp_path / "vol" / "segment_properties" / "info"
    document = json.loads(properties.read_text())
    document["inline"]["properties"][0]["values"] = ["a"]
    properties.write_text(json.dumps(document))
    with pytest.raises(SourceProblem, match="pairs them by position"):
        segment_labels(vol)


def test_a_volume_with_no_properties_subresource_says_so(tmp_path):
    """Never guessed: a default subdirectory would load nothing and report no labels."""
    vol = _volume(tmp_path, meshes={5: [[0, 0, 0]]})
    with pytest.raises(SourceProblem, match="declares no 'segment_properties'"):
        segment_labels(vol)


def test_a_properties_source_of_the_wrong_type_is_refused(tmp_path):
    vol = _volume(tmp_path, labels={"1": "a"})
    (tmp_path / "vol" / "segment_properties" / "info").write_text(
        json.dumps({"@type": "neuroglancer_annotations_v1"}))
    with pytest.raises(SourceProblem, match="not 'neuroglancer_segment_properties'"):
        segment_labels(vol)


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #
def test_a_box_is_the_meshs_extent_in_ZYX(tmp_path):
    """readback hands back xyz — the storage order — and everything here is zyx."""
    vol = _volume(tmp_path, meshes={5: [[1, 2, 3], [7, 20, 300]]})
    (lo, hi) = segment_boxes(vol, ["5"])["5"]
    assert lo == pytest.approx((3.0, 2.0, 1.0))
    assert hi == pytest.approx((300.0, 20.0, 7.0))


def test_the_mesh_subresource_is_resolved_from_INFO_not_defaulted(tmp_path):
    """`read_body_mesh` declares `mesh_dir: str = "mesh"`, so passing None through overrides
    its default with None and it looks for a directory literally named "None" — absent, so
    every body reads as having no mesh and the batch returns an empty dict. Structurally
    broken, and indistinguishable from a volume that genuinely holds none."""
    vol = _volume(tmp_path, meshes={5: [[0, 0, 0], [1, 1, 1]]})
    assert segment_boxes(vol, ["5"], mesh_dir=None) != {}


def test_a_volume_with_no_mesh_subresource_says_so(tmp_path):
    vol = _volume(tmp_path, labels={"1": "a"})
    with pytest.raises(SourceProblem, match="declares no 'mesh'"):
        segment_boxes(vol, ["1"])


def test_a_mesh_format_that_cannot_be_decoded_RAISES_rather_than_returning_nothing(tmp_path):
    """Invariant SKIP-MISSING: skip_missing is about BODIES, and must not absorb a broken
    source. Every structural failure looks like "no meshes here" — an empty dict, no error."""
    from neu_morpho.readback import UnsupportedSubresource

    vol = _volume(tmp_path, meshes={5: [[0, 0, 0]]},
                  mesh_type="neuroglancer_mesh_from_the_future")
    with pytest.raises(UnsupportedSubresource):
        segment_boxes(vol, ["5"], skip_missing=True)


def test_an_absent_body_is_skipped_or_reported_as_asked(tmp_path):
    vol = _volume(tmp_path, meshes={5: [[0, 0, 0]]})
    assert segment_boxes(vol, ["5", "404"], skip_missing=True) .keys() == {"5"}
    with pytest.raises(SourceProblem, match="no mesh for segment 404"):
        segment_boxes(vol, ["404"], skip_missing=False)


def test_a_starred_but_hidden_id_is_still_readable(tmp_path):
    """Segment lists carry `!` for starred-but-hidden, and a box is wanted regardless."""
    vol = _volume(tmp_path, meshes={5: [[0, 0, 0], [1, 1, 1]]})
    assert "5" in segment_boxes(vol, ["!5"])


def test_boxes_come_back_in_NANOMETRES_through_the_sources_transform(tmp_path):
    """Invariant NM-SPACE: one model space, whatever the publisher stored. A source declaring
    a scale would otherwise hand back a box in its own units, framing the camera on a region
    the right shape and the wrong size."""
    vol = _volume(tmp_path, meshes={5: [[0, 0, 0], [1, 1, 1]]},
                  transform=[4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0])
    (lo, hi) = segment_boxes(vol, ["5"])["5"]
    assert hi == pytest.approx((4.0, 4.0, 4.0))
