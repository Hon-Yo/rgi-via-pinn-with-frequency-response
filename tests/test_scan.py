import numpy as np
import pytest

from rgi.geometry import Shell
from rgi.scan import extract_shell


L_SHAPE = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 4), (0, 4)]


def _mesh(path, walls, transform=lambda p: p):
    with path.open("w", encoding="utf-8") as stream:
        for a, b, low, high in walls:
            for point in ((*a, low), (*b, low), (*b, high), (*a, high)):
                stream.write("v " + " ".join(map(str, transform(point))) + "\n")
            stream.write("f -4/-1 -3/-1 -2/-1 -1/-1\n")
    return path


def _walls(footprint, low=0, high=3):
    return [(a, b, low, high) for a, b in zip(footprint, [*footprint[1:], footprint[0]], strict=True)]


def _extract(path, **changes):
    return extract_shell(path, **{"meters_per_unit": 1, "up_axis": "z", "inside": (1, 1), "floor": 0,
                                 "ceiling": 3, "slice_fractions": (0.74, 0.8, 0.86, 0.92, 0.97),
                                 "simplify_m": 0.03, "agreement_m": 0.12, "max_gap_m": 0,
                                 "max_walls": 12, **changes})


def test_extract_registered_concave_shell_with_clutter_and_pillar(tmp_path):
    pillar = [(2, 2), (2.3, 2), (2.3, 2.3), (2, 2.3)]
    clutter = [(0.3, 0.3), (0.6, 0.3), (0.6, 0.6), (0.3, 0.6)]
    path = _mesh(tmp_path / "scan.obj", _walls(L_SHAPE) + _walls(pillar) + _walls(clutter, high=2.65),
                 lambda p: ((p[0] + 10) * 100, p[2] * 100, -p[1] * 100))
    registration = np.eye(4)
    registration[0, 3] = -10
    shell, metadata, _ = _extract(path, meters_per_unit=0.01, up_axis="y", registration=registration,
                                  pillars=[{"footprint": pillar, "floor": 0, "ceiling": 3}])
    expected = Shell(L_SHAPE, 0, 3)
    assert {tuple(p) for p in shell.planes} == {tuple(p) for p in expected.planes}
    assert len(shell.faces) == 6 and len(shell.pillars) == 1
    np.testing.assert_allclose(shell.pillars[0].footprint, pillar)
    assert all(row["agrees"] for row in metadata["slices"])


def test_small_mesh_seam_requires_opt_in_and_is_reported(tmp_path):
    walls = _walls(L_SHAPE)
    walls[0] = ((0.02, 0), (5, 0), 0, 3)
    path = _mesh(tmp_path / "seam.obj", walls)
    with pytest.raises(ValueError, match="closed boundary"):
        _extract(path)
    shell, metadata, _ = _extract(path, max_gap_m=0.025)
    assert {tuple(p) for p in shell.planes} == {tuple(p) for p in Shell(L_SHAPE, 0, 3).planes}
    assert all(row["gap_bridges"] == 1 for row in metadata["slices"])


def test_disagreeing_cross_sections_cannot_create_a_training_label(tmp_path):
    widths = (4, 4.5, 5)
    walls = [wall for width, (low, high) in zip(widths, ((0, 1), (1, 2), (2, 3)), strict=True)
             for wall in _walls([(0, 0), (width, 0), (width, 3), (0, 3)], low, high)]
    with pytest.raises(ValueError, match="slice footprints disagree"):
        _extract(_mesh(tmp_path / "stepped.obj", walls), slice_fractions=(0.2, 0.5, 0.8))
