"""Extract a reviewable, vertically extruded shell from registered OBJ cross-sections."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely import get_parts, hausdorff_distance, linestrings, polygonize, union_all
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

from .geometry import Shell


# Right-handed OBJ up-axis conversions, applied before the optional registration.
_UP = {"x": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
       "y": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "z": np.eye(3)}


def _mesh(path):
    vertices, faces, quads = [], [], []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            kind, *values = fields
            if kind not in ("v", "f"):
                continue
            try:
                if kind == "v":
                    if len(values) < 3:
                        raise ValueError("expected XYZ vertex")
                    vertices.append(tuple(map(float, values[:3])))
                    continue
                if len(values) not in (3, 4):
                    raise ValueError("triangulate faces with more than four vertices before export")
                indices = [int(value.split("/")[0]) for value in values]
                if any(idx == 0 or idx < -len(vertices) for idx in indices):
                    raise ValueError("invalid OBJ vertex index")
                ids = [idx - 1 if idx > 0 else idx + len(vertices) for idx in indices]
                if len(ids) == 3:
                    faces.append(ids)
                else:
                    quads.append(ids)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    xyz = np.asarray(vertices, dtype=float)
    if not vertices or not (faces or quads) or not np.isfinite(xyz).all():
        raise ValueError("OBJ needs finite vertices and at least one triangle or quad")
    if any(max(face) >= len(xyz) for face in (*faces, *quads)):
        raise ValueError("OBJ face references a nonexistent vertex")
    if quads:
        points = xyz[np.asarray(quads)]
        edges = np.roll(points, -1, axis=1) - points
        normals = np.cross(edges, np.roll(edges, -1, axis=1))
        lengths = np.linalg.norm(normals, axis=2)
        unit = normals / np.where(lengths > 0, lengths, 1)[:, :, None]
        if np.any(lengths == 0) or not np.allclose(unit, unit[:, :1], atol=1e-5, rtol=0):
            raise ValueError("OBJ contains a nonplanar, concave or degenerate quad; triangulate before export")
        for a, b, c, d in quads:
            faces.extend(((a, b, c), (a, c, d)))
    return xyz, np.asarray(faces, dtype=np.intp)


def _segments(vertices, faces, levels, height):
    # Half-open triangles: shared vertices are intersected consistently at an exact level.
    ids = (levels[:, 0] <= height) & (levels[:, 2] > height)
    triangles = vertices[faces[ids]]
    lower, middle, upper = np.moveaxis(triangles, 1, 0)
    use_upper = middle[:, 2] <= height
    start = np.where(use_upper[:, None], middle, lower)
    end = np.where(use_upper[:, None], upper, middle)

    def at_z(a, b):
        t = ((height - a[:, 2]) / (b[:, 2] - a[:, 2]))[:, None]
        return a[:, :2] + t * (b[:, :2] - a[:, :2])

    return np.stack((at_z(lower, upper), at_z(start, end)), axis=1)


def _section(segments, snap, gap, interior):
    if not len(segments):
        return None, 0
    keep = np.any(np.rint(segments[:, 0] / snap) != np.rint(segments[:, 1] / snap), axis=1)
    if not keep.any():
        return None, 0
    linework = union_all(linestrings(segments[keep]), grid_size=snap)
    lines = [part for part in get_parts(linework) if part.geom_type == "LineString" and not part.is_empty]
    bridges = []
    if gap > 0 and lines:
        endpoints, counts = np.unique([p for line in lines for p in (line.coords[0], line.coords[-1])],
                                      return_counts=True, axis=0)
        loose = endpoints[counts == 1]
        if len(loose) > 1:
            distance, nearest = cKDTree(loose).query(loose, k=2)
            for i, j in enumerate(nearest[:, 1]):
                if i < j and nearest[j, 1] == i and distance[i, 1] <= gap:
                    bridges.append((loose[i], loose[j]))
        if bridges:
            combined = union_all([linework, *linestrings(bridges)], grid_size=snap)
            lines = [part for part in get_parts(combined) if part.geom_type == "LineString"]
    if not lines:
        return None, len(bridges)
    # Discard internal holes (which could be pillars, furniture, or scan artefacts).
    # Accept only a bounded polygonized cell actually containing the registered seed.
    cells = [poly for poly in get_parts(polygonize(lines)) if poly.contains(interior)]
    return (max((Polygon(cell.exterior) for cell in cells), key=lambda poly: poly.area, default=None), len(bridges))


def extract_shell(path, *, meters_per_unit, up_axis, inside, floor, ceiling, slice_fractions,
                  simplify_m, agreement_m, max_gap_m, max_walls, registration=None, pillars=()):
    """Return a training Shell, extraction provenance, and accepted or rejected closed slice traces."""
    fractions = np.asarray(slice_fractions, dtype=float)
    if (fractions.ndim != 1 or len(fractions) < 3 or not np.all((0 < fractions) & (fractions < 1))
            or len(np.unique(fractions)) != len(fractions)):
        raise ValueError("supply at least three distinct slice fractions strictly between 0 and 1")
    if not (np.isfinite([floor, ceiling, meters_per_unit, simplify_m, agreement_m, max_gap_m, *inside]).all()
            and floor < ceiling and meters_per_unit > 0 and 0 < simplify_m <= agreement_m and max_gap_m >= 0):
        raise ValueError("invalid metric scale, elevations, interior point, or tolerances")
    matrix = np.eye(4) if registration is None else np.asarray(registration, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
        raise ValueError("registration must be a rigid 4×4 transform (rotation and translation in meters)")
    vertices, faces = _mesh(path)
    vertices = vertices @ (matrix[:3, :3] @ _UP[up_axis] * meters_per_unit).T + matrix[:3, 3]
    faces = np.take_along_axis(faces, np.argsort(vertices[faces, 2], axis=1), axis=1)
    levels = vertices[faces, 2]
    snap = min(0.001, simplify_m / 10)
    interior = Point(inside)
    sections, rows = [], []
    for fraction in sorted(fractions):
        height = float(floor + fraction * (ceiling - floor))
        poly, bridges = _section(_segments(vertices, faces, levels, height), snap, max_gap_m, interior)
        rows.append({"z_m": height, "gap_bridges": bridges, "area_m2": None if poly is None else poly.area})
        if poly is not None:
            sections.append((len(rows) - 1, poly))

    required = len(fractions) // 2 + 1
    if len(sections) < required:
        raise ValueError(f"only {len(sections)}/{len(fractions)} slices have a closed boundary around --inside; "
                         "check registration, slice heights and scan gaps, or annotate the footprint manually")
    choices = []
    for row, polygon in sections:
        candidate = orient(polygon.simplify(simplify_m, preserve_topology=True))
        if not candidate.contains(interior):
            continue
        distances = {}
        for i, other in sections:
            distance = float(hausdorff_distance(candidate.boundary, other.boundary, densify=0.25))
            if distance <= agreement_m and candidate.intersection(other).area / candidate.union(other).area >= 0.95:
                distances[i] = distance
        if len(distances) >= required:
            rank = (-len(distances), sum(distances.values()) / len(distances), rows[row]["gap_bridges"], row)
            choices.append((rank, candidate, distances))
    if not choices:
        raise ValueError("slice footprints disagree; scan may be incomplete or nonvertical; review heights or annotate manually")
    rank, polygon, distances = min(choices, key=lambda item: item[0])
    footprint = np.asarray(polygon.exterior.coords)[:-1]
    if len(footprint) > max_walls:
        raise ValueError(f"extracted {len(footprint)} walls but the configured model allows {max_walls}; "
                         "review the scan and simplification (do not discard meaningful recesses)")
    shell = Shell(footprint, floor, ceiling, pillars)
    for i, row in enumerate(rows):
        row.update(agrees=i in distances, boundary_difference_m=distances.get(i))
    metadata = {"source_obj": Path(path).name, "meters_per_obj_unit": meters_per_unit, "obj_up_axis": up_axis,
                "registration_after_scaling_and_up_conversion": matrix.tolist(), "inside_xy_m": list(inside),
                "simplify_m": simplify_m, "snap_m": snap, "agreement_m": agreement_m, "max_gap_m": max_gap_m,
                "required_agreeing_slices": required, "chosen_slice_z_m": rows[rank[-1]]["z_m"],
                "slices": rows}
    traces = [(rows[i]["z_m"], np.asarray(poly.exterior.coords), i in distances) for i, poly in sections]
    return shell, metadata, traces
