import json

import numpy as np
import pytest

from rgi.geometry import Shell, decode, footprint_iou, select_order

_RECTANGLE = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float)
_L_SHAPE = np.array([[0, 0], [5, 0], [5, 2], [2, 2], [2, 4], [0, 4]], dtype=float)


@pytest.mark.parametrize("footprint", [_RECTANGLE, _L_SHAPE], ids=["rectangle", "concave-L"])
def test_planes_and_predicted_order_round_trip(footprint):
    shell = Shell(footprint, -0.4, 2.3)
    count = len(footprint)
    slots = np.array([3, 1, 7, 5, 0, 4])[:count]
    candidates = np.zeros((8, 4))
    candidates[slots] = shell.planes * np.linspace(0.5, 2.0, count)[:, None]
    presence = np.full(8, 0.2)
    presence[slots] = 0.5
    logits = np.full((8, 8), -5.0)
    logits[slots, np.roll(slots, -1)] = 8.0
    logits[:, presence < 0.5] = 100.0  # Absent successors do not participate.
    np.fill_diagonal(logits, 100.0)
    order = select_order(presence, logits, 0.5)
    first = int(np.argmin(slots))
    assert order == np.roll(slots, -first).tolist()

    reconstructed = decode(candidates, order, shell.floor, shell.ceiling)
    np.testing.assert_allclose(reconstructed.footprint, np.roll(footprint, -first, axis=0), atol=1e-12)
    np.testing.assert_allclose(reconstructed.planes, np.roll(shell.planes, -first, axis=0), atol=1e-12)
    assert footprint_iou(shell, reconstructed) == pytest.approx(1.0)
    assert reconstructed.floor == -0.4
    assert reconstructed.ceiling == 2.3


def test_finite_faces_and_pillar_annotation_round_trip():
    pillar = {"footprint": [[0.4, 0.4], [0.9, 0.4], [0.9, 0.8], [0.4, 0.8]], "ceiling": 2.4}
    shell = Shell(_L_SHAPE, -0.1, 2.7, [pillar])
    assert len(shell.planes) == len(_L_SHAPE)
    np.testing.assert_allclose(shell.faces[:, 0, :2], _L_SHAPE)
    np.testing.assert_allclose(shell.faces[:, 1, :2], np.roll(_L_SHAPE, -1, axis=0))
    np.testing.assert_allclose(shell.faces[:, :2, 2], shell.floor)
    np.testing.assert_allclose(shell.faces[:, 2:, 2], shell.ceiling)
    np.testing.assert_allclose(
        np.einsum("wvc,wc->wv", shell.faces, shell.planes[:, :3]) + shell.planes[:, 3, None], 0.0
    )
    winding_normals = np.cross(shell.faces[:, 1] - shell.faces[:, 0], shell.faces[:, 2] - shell.faces[:, 1])
    assert np.all(np.einsum("ij,ij->i", shell.planes[:, :3], winding_normals) > 0)

    document = json.loads(json.dumps(shell.to_dict()))
    restored = Shell.from_dict(document)
    assert restored.to_dict() == document
    assert restored.pillars[0].floor is None
    assert restored.pillars[0].ceiling == 2.4
    assert document["walls"][0] == {
        "id": 0, "predecessor": 5, "successor": 1, "plane": [0, -1, 0, 0],
        "vertices": [[0, 0, -0.1], [5, 0, -0.1], [5, 0, 2.7], [0, 0, 2.7]],
    }
    assert footprint_iou(shell, Shell(_L_SHAPE, -0.1, 2.7)) == 1.0


def test_footprint_iou_for_partial_overlap():
    left = Shell(_RECTANGLE, 0, 2.5)
    right = Shell(_RECTANGLE + [2, 0], 1, 3)
    assert footprint_iou(left, right) == pytest.approx(1 / 3)


def test_order_breaks_premature_cycles_deterministically():
    logits = np.full((4, 4), -2.0)
    logits[[0, 1, 2, 3], [1, 0, 3, 2]] = 5.0
    np.fill_diagonal(logits, 20.0)
    assert select_order(np.ones(4), logits, 0.5) == [0, 1, 2, 3]
    assert select_order(np.ones(4), logits + np.arange(4)[:, None] * 5, 0.5) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="fewer than three"):
        select_order([1, 0, 0, 1], logits, 0.5)


@pytest.mark.parametrize(
    ("footprint", "error"),
    [
        (_RECTANGLE[::-1], "counterclockwise"),
        (np.vstack((_RECTANGLE, _RECTANGLE[0])), "degenerate edge"),
        ([[0, 0], [2, 0], [4, 0], [4, 3], [0, 3]], "adjacent walls"),
        ([[0, 0], [4, 3], [0, 3], [4, 0]], "invalid footprint"),
    ],
)
def test_rejects_invalid_footprints(footprint, error):
    with pytest.raises(ValueError, match=error):
        Shell(footprint, 0, 2.5)


def test_rejects_invalid_pillar_annotations():
    with pytest.raises(ValueError, match="strictly inside"):
        Shell(_RECTANGLE, 0, 3, [{"footprint": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}])
    with pytest.raises(ValueError, match="rectangular"):
        Shell(_RECTANGLE, 0, 3, [{"footprint": [[1, 0.5], [2, 0.5], [2, 1], [1.5, 1]]}])
    with pytest.raises(ValueError, match="inside the shell height"):
        Shell(_RECTANGLE, 0, 3, [{"footprint": [[1, 0.5], [2, 0.5], [2, 1], [1, 1]], "ceiling": 4}])


def test_decode_rejects_parallel_reversed_and_bad_plane_cycles():
    planes = Shell(_RECTANGLE, 0, 2.5).planes
    with pytest.raises(ValueError, match="parallel"):
        decode(planes, [0, 2, 1, 3], 0, 2.5)
    with pytest.raises(ValueError, match="counterclockwise"):
        decode(planes, [0, 3, 2, 1], 0, 2.5)
    with pytest.raises(ValueError, match="duplicate"):
        decode(planes, [0, 1, 1, 3], 0, 2.5)
    with pytest.raises(ValueError, match="ceiling must be above"):
        decode(planes, range(4), 3, 2.5)
    inward = planes.copy()
    inward[1] *= -1
    with pytest.raises(ValueError, match="slot 1 does not face outward"):
        decode(inward, range(4), 0, 2.5)
    unstable = planes.copy()
    unstable[0] *= 1e-12
    with pytest.raises(ValueError, match="near-zero"):
        decode(unstable, range(4), 0, 2.5)
    tipped = planes.copy()
    tipped[0, 2] = 0.1
    with pytest.raises(ValueError, match="must be vertical"):
        decode(tipped, range(4), 0, 2.5)


def test_decode_rejects_self_intersection():
    xy = np.array([[0, 0], [4, 3], [0, 3], [4, 0]], dtype=float)
    edges = np.roll(xy, -1, axis=0) - xy
    normals = edges[:, [1, 0]] * [1, -1]
    offsets = -(xy * normals).sum(axis=1)
    planes = np.column_stack((normals, np.zeros(4), offsets))
    with pytest.raises(ValueError, match="invalid footprint"):
        decode(planes, range(4), 0, 2.5)
