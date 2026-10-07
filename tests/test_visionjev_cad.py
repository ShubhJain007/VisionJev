"""CAD generators (``laya.visionjev_cad``): every label is recomputed here from the construction parameters by code
that does not use the generator's label logic, and wherever possible by asking the OpenCascade kernel about the solid
itself: the genus of the solid is its number of through holes, point-in-solid probes measure hole depths, bounding
boxes give overall sizes, and shape-to-shape distance decides which parts of an assembly touch. Also: determinism for a
fixed seed, image files that exist and open, and the structural guarantees (rank-uniform numeric options, uniform gold
panel, balanced yes / no)."""
import hashlib
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("cadquery")

from laya import visionjev_cad as C  # noqa: E402

DATA_VALIDATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_validator.py")


# ---- independent kernel measurements ----------------------------------------------------------------------------------


def _n(shape, kind):
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape.wrapped, kind, m)
    return m.Extent()


def genus(shape):
    """Number of through holes of a solid, from the Euler-Poincare formula V - E + F - (L - F) = 2 (S - G)."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_VERTEX, TopAbs_WIRE

    V, E, F, L, S = [_n(shape, k) for k in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE, TopAbs_SHELL)]
    g = S - (V - E + 2 * F - L) / 2
    assert g == int(g)
    return int(g)


def inside(shape, p):
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN

    return BRepClass3d_SolidClassifier(shape.wrapped, gp_Pnt(*map(float, p)), 1e-6).State() == TopAbs_IN


def gap(a, b):
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    return BRepExtrema_DistShapeShape(a.wrapped, b.wrapped).Value()


def probe_depth(shape, u, v, T, step=0.25):
    """Depth of the void under (u, v) measured from the top face z = T; T if it is open to the underside."""
    z = T - step / 2
    while z > 0:
        if inside(shape, (u, v, z)):
            return round(T - z - step / 2, 3)
        z -= step
    return T


_ROWS = {}


def rows(name, n, split="test"):
    have = _ROWS.setdefault((name, split), [])
    have.extend(C.make_row(name, split, i, render=False) for i in range(len(have), n))
    return have[:n]


def gold_text(r):
    return r["q"]["criteria"][r["label"]]


# ---- 1. feature counts ------------------------------------------------------------------------------------------------


def expected_count(ins, meta):
    if "asm" in meta:
        parts = dict(C.asm_parts(meta["asm"]))
        bolts, holes = len(parts["HEX BOLT"].Solids()), genus(parts["COVER PLATE"])
        assert bolts == len(meta["asm"]["bolts"]) and holes == len(meta["asm"]["holes"])
        return holes - bolts if "empty" in ins else bolts
    spec = meta["spec"]
    shape, feats = C.build(spec), spec.get("feats", [])
    kinds = Counter(f["kind"] for f in feats)
    if "blind" in ins:
        return kinds["hole"] - genus(shape)
    if "all the way through" in ins:
        assert genus(shape) == sum(1 for f in feats if f.get("depth") is None)
        return genus(shape)
    if "round holes" in ins:
        assert genus(shape) == len(feats)  # every hole and slot of these plates is a through feature
        return genus(shape) - kinds["slot"]
    if "upright" in ins:
        return sum(1 for f in feats if f["face"].startswith("leg"))
    if "bracket have in total" in ins:
        return genus(shape)
    if "counterbored" in ins:
        return sum(1 for f in feats if f.get("cb"))
    if "slots" in ins:
        return genus(shape) - kinds["hole"]
    if "bolt" in ins and spec["fam"] == "flange":
        assert spec["n"] == len(C.bolt_xy(spec))
        return genus(shape) - 1  # minus the centre bore
    if "corners" in ins:
        return sum(1 for r in spec["outline"]["r"] if r > 0)
    if "lightening" in ins:
        return genus(shape) - 1
    if "teeth" in ins:
        p = 2 * math.pi / spec["n"]
        tip = spec["tip_d"] / 2 * math.cos(0.15 * p) - 1e-3  # the flat tip faces of the prism, counted on the solid
        n = sum(1 for f in shape.Faces() if abs(f.Center().z - spec["T"] / 2) < 1e-6 and math.hypot(f.Center().x, f.Center().y) > tip)
        assert n == spec["n"]
        return n
    if "cylindrical sections" in ins:
        return len(spec["segs"])
    if "grooves" in ins:
        return len(spec["grooves"])
    if "top face of the block" in ins and "drilled" in ins:
        return sum(1 for f in feats if not inside(shape, (f["u"], f["v"], spec["T"] - 0.5)))
    if "underside" in ins:
        return sum(1 for f in feats if not inside(shape, (f["u"], f["v"], 0.5)))
    if "bosses" in ins:
        return sum(1 for f in feats if inside(shape, (f["u"] + f["d"] / 2 - 1, f["v"], spec["T"] + 1)))
    raise AssertionError("no rule for: " + ins)


def test_feature_counts_match_the_solid():
    kinds = Counter()
    for r in rows("cad_feature_count", 220):
        want = expected_count(r["meta"]["ins"], r["meta"])
        assert gold_text(r) == str(want), (r["meta"]["ins"], want, r["q"])
        assert r["meta"]["gold"] == want and r["why"] and len(set(r["q"]["criteria"])) == 4
        kinds[r["kind"]] += 1
    assert len(kinds) >= 10, kinds  # every family of count question appears


# ---- 2. view matching -------------------------------------------------------------------------------------------------
AXES = {"top": (0, 1), "front": (0, 2), "right": (1, 2)}


def wire_points(spec, ask):
    """All edges of the solid projected along the asked view (no hidden-line removal): (n, 2) points."""
    a, b = AXES[ask]
    pts = []
    for e in C.build(spec).Edges():
        for t in np.linspace(0, 1, 25):
            p = e.positionAt(float(t))
            pts.append(((p.x, p.y, p.z)[a], (p.x, p.y, p.z)[b]))
    return np.array(pts)


def bitmap(pts, scale, res=180):
    c = (pts.min(0) + pts.max(0)) / 2
    ij = np.clip(np.round((pts - c) * scale + res / 2).astype(int), 0, res - 1)
    m = np.zeros((res, res), bool)
    m[ij[:, 1], ij[:, 0]] = True
    return m


def depth_map(spec, ask, box, n=44):
    """First-hit distance of rays cast along the view direction on a grid: what a viewer sees, without HLR."""
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector

    d = {"top": (0, 0, -1), "front": (0, 1, 0), "right": (-1, 0, 0)}[ask]
    a, b = AXES[ask]
    it = IntCurvesFace_ShapeIntersector()
    it.Load(C.build(spec).wrapped, 1e-6)
    out = np.full((n, n), 1e9)
    for i, u in enumerate(np.linspace(box[0], box[1], n)):
        for j, v in enumerate(np.linspace(box[2], box[3], n)):
            p = [0.0, 0.0, 0.0]
            p[a], p[b] = u, v
            p[3 - a - b] = -500.0 * d[3 - a - b]
            it.Perform(gp_Lin(gp_Pnt(*p), gp_Dir(*d)), 0.0, 2000.0)
            out[i, j] = min((it.WParameter(k) for k in range(1, it.NbPnt() + 1)), default=1e9)
    return out


def cut_bitmap(spec, y, box, step=1.5):
    shape = C.build(spec)
    xs, zs = np.arange(box[0] + step / 2, box[1], step), np.arange(box[2] + step / 2, box[3], step)
    return np.array([[inside(shape, (x, y, z)) for x in xs] for z in zs])


def test_view_match_gold_is_the_part_and_distractors_differ():
    pos = Counter()
    for r in rows("cad_view_match", 36):
        m = r["meta"]
        gold = r["label"]
        pos[gold] += 1
        assert json.dumps(m["panels"][gold], sort_keys=True) == json.dumps(m["spec"], sort_keys=True)
        assert m["mutations"][gold] == "gold" and sorted(x != "gold" for x in m["mutations"]) == [False, True, True, True]
        assert r["q"]["criteria"] in (["panel A", "panel B", "panel C", "panel D"], ["panel 1", "panel 2", "panel 3", "panel 4"])
        if m["ask"] == "section":
            bbs = [C.build(s).BoundingBox() for s in m["panels"]]
            box = (min(b.xmin for b in bbs), max(b.xmax for b in bbs), min(b.zmin for b in bbs), max(b.zmax for b in bbs))
            sigs = [cut_bitmap(s, m["cut"], box) for s in m["panels"]]
            plans = [wire_points(s, "top") for s in m["panels"]]  # the plan with the cutting line is shown too
            scale = 170 / max(np.ptp(p, axis=0).max() for p in plans)
            for k, s in enumerate(sigs):
                assert k == gold or (s != sigs[gold]).sum() >= 3 or \
                    (bitmap(plans[k], scale) != bitmap(plans[gold], scale)).sum() >= 3, (k, m["mutations"])
        else:
            pts = [wire_points(s, m["ask"]) for s in m["panels"]]
            scale = 170 / max(np.ptp(p, axis=0).max() for p in pts)
            sigs = [bitmap(p, scale) for p in pts]
            lo, hi = np.min([p.min(0) for p in pts], 0), np.max([p.max(0) for p in pts], 0)
            box = (lo[0] - 1, hi[0] + 1, lo[1] - 1, hi[1] + 1)
            dm = None
            for k, s in enumerate(sigs):
                if k == gold or (s != sigs[gold]).sum() >= 3:
                    continue
                dm = depth_map(m["spec"], m["ask"], box) if dm is None else dm  # same edges: what faces the viewer must differ
                assert (np.abs(depth_map(m["panels"][k], m["ask"], box) - dm) > 0.5).sum() >= 3, (k, m["mutations"])
    assert set(pos) == {0, 1, 2, 3}


# ---- 3. dimension reading ---------------------------------------------------------------------------------------------


def _bb(s):
    return C.build(s).BoundingBox()


def _us(s):
    return sorted(f["u"] for f in s["feats"] if f["face"] == "top")


QUANTITY = {
    "the diameter of the largest section of the shaft": lambda s: max(d for d, _ in s["segs"]),
    "the diameter of the smallest section of the shaft": lambda s: min(d for d, _ in s["segs"]),
    "the length of the longest section of the shaft": lambda s: max(l for _, l in s["segs"]),
    "the length of the section that has the largest diameter": lambda s: max(s["segs"])[1],
    "the diameter of the leftmost section of the shaft": lambda s: s["segs"][0][0],
    "the diameter of the rightmost section of the shaft": lambda s: s["segs"][-1][0],
    "the overall length of the shaft": lambda s: _bb(s).xlen,
    "the combined length of the two leftmost sections": lambda s: s["segs"][0][1] + s["segs"][1][1],
    "the distance from the left end of the shaft to the start of the rightmost section": lambda s: _bb(s).xlen - s["segs"][-1][1],
    "the overall length of the plate": lambda s: _bb(s).xlen,
    "the overall width of the plate": lambda s: _bb(s).ylen,
    "the thickness of the plate": lambda s: _bb(s).zlen,
    "the diameter of the holes": lambda s: s["feats"][0]["d"],
    "the distance from the left edge of the plate to the centre of the first (leftmost) hole": lambda s: _us(s)[0],
    "the distance from the left edge of the plate to the centre of the last (rightmost) hole": lambda s: _us(s)[-1],
    "the distance from the centre of the last (rightmost) hole to the right edge of the plate": lambda s: _bb(s).xlen - _us(s)[-1],
    "the distance between the centres of the first and the last hole": lambda s: _us(s)[-1] - _us(s)[0],
    "the outside diameter of the flange": lambda s: _bb(s).xlen,
    "the diameter of the centre bore": lambda s: s["bore_d"],
    "the diameter of the hub": lambda s: s["hub_d"],
    "the bolt circle diameter": lambda s: 2 * math.hypot(*C.bolt_xy(s)[0]),
    "the thickness of the flange plate": lambda s: s["T"],
    "the overall height of the part": lambda s: _bb(s).zlen,
    "the diameter of the bolt holes": lambda s: s["bolt_d"],
    "the radial wall thickness of the hub (between the bore and the outside of the hub)": lambda s: (s["hub_d"] - s["bore_d"]) / 2,
    "the height of the hub above the flange face": lambda s: _bb(s).zlen - s["T"],
    "the radial distance from the bolt circle to the outside edge of the flange": lambda s: (_bb(s).xlen - s["pcd"]) / 2,
    "the overall length of the bracket base": lambda s: _bb(s).xlen,
    "the overall height of the bracket": lambda s: _bb(s).zlen,
    "the width of the bracket (front to back)": lambda s: _bb(s).ylen,
    "the material thickness of the bracket": lambda s: s["T"],
    "the diameter of the holes in the base": lambda s: [f["d"] for f in s["feats"] if f["face"] == "base"][0],
    "the height of the upright leg above the top face of the base": lambda s: _bb(s).zlen - s["T"],
    "the length of the base measured from the inside face of the upright leg": lambda s: _bb(s).xlen - s["T"],
}
COMPARE = {
    "Is the leftmost section of the shaft longer than the rightmost section?": lambda s: (s["segs"][0][1], s["segs"][-1][1]),
    "Is the largest diameter of the shaft more than twice its smallest diameter?": lambda s: (
        max(d for d, _ in s["segs"]), 2 * min(d for d, _ in s["segs"])),
    "Is the leftmost section larger in diameter than the rightmost section?": lambda s: (s["segs"][0][0], s["segs"][-1][0]),
    "Is the longest section of the shaft also longer than the largest diameter?": lambda s: (
        max(l for _, l in s["segs"]), max(d for d, _ in s["segs"])),
    "Is the hole diameter larger than the plate thickness?": lambda s: (s["feats"][0]["d"], s["T"]),
    "Is the last hole closer to the right edge than the first hole is to the left edge?": lambda s: (_us(s)[0], s["L"] - _us(s)[-1]),
    "Is the plate more than twice as long as it is wide?": lambda s: (s["L"], 2 * s["W"]),
    "Is the bore diameter larger than the height of the hub above the flange face?": lambda s: (s["bore_d"], s["hub_h"]),
    "Is the bolt circle diameter more than twice the bore diameter?": lambda s: (s["pcd"], 2 * s["bore_d"]),
    "Is the hub wall (radial, bore to outside of hub) thicker than the flange plate?": lambda s: ((s["hub_d"] - s["bore_d"]) / 2, s["T"]),
    "Is the hub diameter more than half the outside diameter of the flange?": lambda s: (s["hub_d"], s["OD"] / 2),
    "Is the bracket taller than its base is long?": lambda s: (s["H"], s["L"]),
    "Is the bracket wider (front to back) than it is tall?": lambda s: (s["W"], s["H"]),
    "Is the hole diameter larger than the material thickness?": lambda s: (s["feats"][0]["d"], s["T"]),
}


def in_asked_unit(printed, units, ask):
    if units == ask:
        return printed
    return printed * 25.4 if ask == "mm" else printed / 25.4


def show(v, unit):
    return ("%g" % round(v, 2 if unit == "mm" else 3)) + " " + unit


def test_dimension_answers_recomputed_from_parameters():
    modes, kinds = Counter(), Counter()
    for r in rows("cad_dimension_read", 320):
        m, ins = r["meta"], r["q"]["instructions"]
        modes[m["mode"]] += 1
        kinds[r["q"]["type"]] += 1
        assert (m["units"] != m["ask_units"]) == (m["mode"] == "trap")
        if ins in COMPARE:
            a, b = COMPARE[ins](m["spec"])
            assert a != b and r["label"] == int(a > b)
            continue
        thr = re.match(r"Is (.+) greater than ([0-9.]+) (mm|in)\?$", ins)
        what = re.match(r"What is (.+?)(, in (millimetres|inches))?\?$", ins)
        phrase = (thr or what).group(1)
        printed = round(QUANTITY[phrase](m["spec"]) * m["scale"], 4)
        true = in_asked_unit(printed, m["units"], m["ask_units"])
        if thr:
            assert thr.group(3) == m["ask_units"] and r["label"] == int(true > float(thr.group(2)))
            assert r["label"] == int(m["units"] == "in")  # only the title block decides these
        else:
            assert gold_text(r) == show(true, m["ask_units"]), (ins, m["units"], m["ask_units"], true, r["q"]["criteria"])
            if m["mode"] != "plain":  # the unit structure: both readings of two printed numbers are offered
                other = show(printed if m["mode"] == "trap" else in_asked_unit(printed, "in" if m["ask_units"] == "mm" else "mm", m["ask_units"]),
                             m["ask_units"])
                assert other in r["q"]["criteria"] and other != gold_text(r)
                assert ("25.4" in r["why"]) == (m["mode"] == "trap")
    assert 0.10 < modes["trap"] / 320 < 0.21 and 0.09 < modes["twin"] / 320 < 0.21, modes
    assert kinds["noul"] > 40 and kinds["choice"] > 150


# ---- 4. drawing pairs -------------------------------------------------------------------------------------------------


def _strip(s, *keys):
    return {k: v for k, v in s.items() if k not in keys}


def test_drawing_pair_differences_are_as_stated():
    seen = Counter()
    for r in rows("cad_drawing_pair", 70):
        m = r["meta"]
        a, b, cls = m["a"], m["b"], m["cls"]
        sa, sb = C.build(a), C.build(b)
        seen[cls] += 1
        assert C.chiral(_strip(a, "mirror")) or C.chiral(_strip(b, "mirror"))
        if r["q"]["type"] == "noul":
            assert r["label"] == int(cls == "same")
        else:
            assert cls != "same" and gold_text(r) == C.PAIR_OPTS[cls]
        if cls == "same":
            assert a == b
        elif cls == "count":
            assert abs(genus(sa) - genus(sb)) == 1 and abs(len(a["feats"]) - len(b["feats"])) == 1
            assert _strip(a, "feats") == _strip(b, "feats")
        elif cls == "dim":
            assert genus(sa) == genus(sb) and len(a["feats"]) == len(b["feats"]) and abs(sa.Volume() - sb.Volume()) > 1
            changed = [k for k in ("L", "W", "H", "T") if a.get(k) != b.get(k)]
            ds = [(x["d"], y["d"]) for x, y in zip(a["feats"], b["feats"]) if x.get("d") != y.get("d")]
            assert (len(changed) == 1) != bool(ds)
            if changed:
                big, small = max(a[changed[0]], b[changed[0]]), min(a[changed[0]], b[changed[0]])
                assert big / small >= 1.2
        elif cls == "fillet":
            assert genus(sa) == genus(sb) and a["feats"] == b["feats"] and abs(sa.Volume() - sb.Volume()) > 0.5
            if a["fam"] == "bracket":
                assert (a["rin"] == 0) != (b["rin"] == 0)
            else:
                assert sum(x != y for x, y in zip(a["cr"], b["cr"])) == 1 and sum(a["cr"]) != sum(b["cr"])
        else:
            assert _strip(a, "mirror") == _strip(b, "mirror") and bool(a.get("mirror")) != bool(b.get("mirror"))
            ca, cb = sa.Center(), sb.Center()
            assert abs(sa.Volume() - sb.Volume()) < 1e-3 and abs(ca.x + cb.x - a["L"]) < 1e-3 and abs(ca.y - cb.y) < 1e-6
    assert set(seen) == {"same", "count", "dim", "fillet", "mirror"}


# ---- 5. section depth -------------------------------------------------------------------------------------------------


def hole_facts(spec, i):
    """(depth, is through, has counterbore) of row hole i, probed on the solid."""
    f = spec["feats"][i]
    shape, T = C.build(spec), spec["T"]
    depth = probe_depth(shape, f["u"], f["v"], T)
    cb = not inside(shape, (f["u"] + f["d"] / 2 + 1.0, f["v"], T - 0.5))
    return depth, depth == T, cb


def test_section_answers_match_probes_of_the_solid():
    kinds = Counter()
    for r in rows("cad_section_depth", 160):
        m, ins = r["meta"], r["q"]["instructions"]
        spec = m["spec"]
        kinds[m["kind"]] += 1
        if spec["fam"] == "flange":
            H = spec["T"] + spec["hub_h"]
            if m["kind"] == "through":
                assert r["label"] == int(not inside(C.build(spec), (0, 0, 0.5)))
            else:
                x = spec["bore_d"] / 2
                while not inside(C.build(spec), (x, 0, H - 0.5)):
                    x += 0.25
                wall = 0.0
                while inside(C.build(spec), (x + wall, 0, H - 0.5)):
                    wall += 0.25
                assert abs(float(gold_text(r).split()[0]) - wall) <= 0.5
            continue
        facts = {m["names"][i]: hole_facts(spec, i) for i in range(m["k"])}
        for i in range(m["k"]):  # the probed depth agrees with the parameter (0.25 mm probe step)
            want = spec["T"] if spec["feats"][i].get("depth") is None else spec["feats"][i]["depth"]
            assert abs(facts[m["names"][i]][0] - want) <= 0.26
        asked = re.findall(r"H\d", ins)
        if m["kind"] == "through":
            assert r["label"] == int(facts[asked[0]][1])
        elif m["kind"] == "deeper":
            assert facts[asked[0]][0] != facts[asked[1]][0] and r["label"] == int(facts[asked[0]][0] > facts[asked[1]][0])
        elif m["kind"] == "deepest":
            pick = (min if "shallowest" in ins else max)(facts, key=lambda k: facts[k][0])
            assert gold_text(r) == pick and r["q"]["criteria"] == sorted(facts)
        elif m["kind"] == "count_through":
            assert gold_text(r) == str(sum(f[1] for f in facts.values()))
        elif m["kind"] == "type":
            d, thru, cb = facts[asked[0]]
            assert gold_text(r) == "a %s %s hole" % ("counterbored" if cb else "plain", "through" if thru else "blind")
        else:
            assert abs(float(gold_text(r).split()[0]) - (spec["T"] - facts[asked[0]][0])) <= 0.26
    assert len(kinds) == 6, kinds


# ---- 6. assemblies ----------------------------------------------------------------------------------------------------


def test_assembly_answers_match_the_solids():
    kinds = Counter()
    for r in rows("cad_assembly", 150):
        m, ins = r["meta"], r["q"]["instructions"]
        kinds[r["kind"]] += 1
        if r["kind"] == "fasteners":
            assert gold_text(r) == str(expected_count(ins, m))
            continue
        item = {m["numbering"][k]: k for k in range(len(m["names"]))}
        asked = [item[int(x)] for x in re.findall(r"item (\d+)", ins)]
        opts = [item[int(re.search(r"\d+", o).group())] for o in r["q"]["criteria"]]
        gold = opts[r["label"]]
        assert [int(re.search(r"\d+", o).group()) for o in r["q"]["criteria"]] == sorted(m["numbering"][k] for k in opts)
        if r["kind"] == "order":
            solids = [s for _, s in C.asm_parts(m["asm"], explode=20.0)]
            up = sorted(range(len(solids)), key=lambda k: solids[k].BoundingBox().zmin)  # the base with its stud is lowest
            rank = {k: i for i, k in enumerate(up)}
            if "goes on first" in ins:
                assert rank[asked[0]] == 0 and rank[gold] == 1
            elif "fitted last" in ins:
                assert rank[gold] == len(solids) - 1
            elif "immediately after" in ins:
                assert rank[gold] == rank[asked[1]] + 1
            else:
                assert sorted((rank[asked[0]], rank[asked[1]])) == [rank[gold] - 1, rank[gold] + 1]
            continue
        solids = [s for _, s in C.asm_parts(m["asm"])]
        touch = lambda i, j: gap(solids[i], solids[j]) < 1e-6  # noqa: E731
        pairs = {frozenset((i, j)) for i in range(len(solids)) for j in range(i) if touch(i, j)}
        if m["asm"]["type"] == "bushing":
            assert pairs == C.touching(m["asm"])  # the declared contacts are exactly the kernel's
        box = [x.BoundingBox() for x in solids]
        if "threaded" in ins:
            assert box[gold].zmax > box[asked[0]].zmin + 1 and touch(gold, asked[0])  # the shank ends inside the gold part ...
            assert all(box[o].zmin >= box[gold].zmax - 1e-6 for o in opts if o != gold)  # ... and in no other option
        elif "between" in ins:
            assert all(touch(gold, a) for a in asked)
            assert all(not all(touch(o, a) for a in asked) for o in opts if o != gold)
        elif "bore of" in ins:
            assert touch(gold, asked[0]) and box[asked[0]].zmin < solids[gold].Center().z < box[asked[0]].zmax
            for o in opts:
                assert o == gold or not touch(o, asked[0]) or not box[asked[0]].zmin < solids[o].Center().z < box[asked[0]].zmax
        elif "directly on top of" in ins or "rest" in ins:  # the layer whose underside is the asked part's top face
            on = lambda o: touch(o, asked[0]) and abs(box[o].zmin - box[asked[0]].zmax) < 1e-6  # noqa: E731
            if "rest" in ins:
                on = lambda o: touch(o, asked[0]) and abs(box[o].zmax - box[asked[0]].zmin) < 1e-6  # noqa: E731
            assert [o for o in opts if on(o)] == [gold]
        elif "directly under" in ins and "heads" not in ins:
            under = lambda o: touch(o, asked[0]) and abs(box[o].zmax - box[asked[0]].zmin) < 1e-6  # noqa: E731
            assert [o for o in opts if under(o)] == [gold]
        elif "under the heads" in ins:
            head = box[asked[0]].zmax - 0.7 * m["asm"]["d"]  # underside of the hex heads
            assert [o for o in opts if touch(o, asked[0]) and abs(box[o].zmax - head) < 1e-6] == [gold]
        else:
            assert "direct contact" in ins and touch(gold, asked[0])
            assert all(not touch(o, asked[0]) for o in opts if o != gold)
    assert set(kinds) == {"fasteners", "mates", "order"}


# ---- determinism, files, structure ------------------------------------------------------------------------------------


def test_rows_are_deterministic_for_a_fixed_seed():
    for name in C.MAKERS:
        for i in (0, 7):
            a, b = C.make_row(name, "train", i, render=False), C.make_row(name, "train", i, render=False)
            assert (a["q"], a["label"], a["why"], a["kind"]) == (b["q"], b["label"], b["why"], b["kind"])
            assert json.dumps(a["meta"], sort_keys=True, default=str) == json.dumps(b["meta"], sort_keys=True, default=str)
        x, y = C.make_row(name, "train", 3)["image"](), C.make_row(name, "train", 3)["image"]()
        assert x.size == y.size and np.array_equal(np.asarray(x), np.asarray(y))
        assert C.make_row(name, "train", 3, render=False)["meta"] != C.make_row(name, "val", 3, render=False)["meta"]


def test_built_datasets_have_images_manifest_and_distinct_splits(tmp_path):
    from PIL import Image

    names = list(C.MAKERS)
    for name in names:
        man = C.build_dataset(name, str(tmp_path), n_train=5, n_val=3, workers=2, log=lambda m: None)
        d = tmp_path / name
        assert man["counts"] == {"train": 5, "val": 3} and (d / "_READY").exists()
        lic = json.load(open(d / "manifest.json"))["license"]
        assert lic["commercial"] is True and lic["annotations"] == "generated in-house, exact labels" and "CadQuery" in lic["images"]
        hashes = {}
        for split in ("train", "val"):
            recs = [json.loads(l) for l in open(d / (split + ".jsonl"))]
            assert len(recs) == man["counts"][split]
            for rec in recs:
                assert set(rec) == {"id", "image", "state_text", "question", "label", "rationale", "kind", "image_license"}
                assert rec["rationale"] and rec["kind"] and rec["image"].endswith(".png")
                q = rec["question"]
                assert 0 <= rec["label"] < (2 if q["type"] == "noul" else len(q["criteria"]))
                with Image.open(d / rec["image"]) as im:
                    im.load()
                    assert im.mode == "RGB" and max(im.size) <= 896 and min(im.size) >= 560
                    assert np.asarray(im).std() > 3  # not a blank sheet
                hashes.setdefault(split, set()).add(hashlib.sha256(open(d / rec["image"], "rb").read()).hexdigest())
        assert not hashes["train"] & hashes["val"] and len(hashes["train"]) == 5 and len(hashes["val"]) == 3
    if os.path.exists(DATA_VALIDATOR):
        import importlib.util

        spec = importlib.util.spec_from_file_location("data_validator", DATA_VALIDATOR)
        dv = importlib.util.module_from_spec(spec)
        import sys

        sys.modules["data_validator"] = dv
        spec.loader.exec_module(dv)
        rep, _ = dv.run([str(tmp_path / n) for n in names], quiet=True)
        assert rep.n_errors == 0, dict(rep.errors)


def _rank(r):
    vals = [float(re.sub(r"[^0-9.\-]", "", o)) for o in r["q"]["criteria"]]
    return sorted(vals).index(vals[r["label"]])


def test_numeric_options_are_rank_uniform():
    """The gold's rank among the numeric options carries no information: a rank prior learned per question template
    on one half of the rows does no better than chance on the other half, and every rank occurs."""
    for name, n in (("cad_feature_count", 700), ("cad_dimension_read", 900)):
        rs = [r for r in rows(name, n, split="rank")
              if r["q"]["type"] == "choice" and (name != "cad_dimension_read" or r["meta"]["mode"] == "plain")]
        tmpl = lambda r: re.sub(r"\d+(\.\d+)?", "#", r["q"]["instructions"])  # noqa: E731
        half = len(rs) // 2
        prior = {}
        for r in rs[:half]:
            prior.setdefault(tmpl(r), Counter())[_rank(r)] += 1
        hit = sum(prior.get(tmpl(r), Counter({0: 1})).most_common(1)[0][0] == _rank(r) for r in rs[half:])
        ranks = Counter(_rank(r) for r in rs)
        assert len(rs) > 300 and all(ranks[k] / len(rs) > 0.12 for k in range(4)), (name, ranks)
        assert hit / (len(rs) - half) < 0.25 + 0.09, (name, hit / (len(rs) - half), ranks)
    ranks = Counter()  # the unit rows offer {a, b, conv(a), conv(b)}: the gold is equally the raw or the converted one
    for r in rows("cad_dimension_read", 900, split="rank"):
        if r["q"]["type"] == "choice" and r["meta"]["mode"] != "plain":
            ranks[_rank(r) // 2 if r["meta"]["ask_units"] == "mm" else 1 - _rank(r) // 2] += 1
    total = sum(ranks.values())
    assert total > 120 and 0.38 < ranks[0] / total < 0.62, ranks


def test_gold_position_and_yes_no_balance():
    pos = Counter(r["label"] for r in rows("cad_view_match", 100, split="test"))
    assert all(11 <= pos[k] <= 41 for k in range(4)), pos
    pair = rows("cad_drawing_pair", 100, split="test")
    yes = [r["label"] for r in pair if r["q"]["type"] == "noul"]
    assert 0.36 < sum(yes) / len(yes) < 0.64, (sum(yes), len(yes))
    cls = Counter(r["meta"]["cls"] for r in pair if r["q"]["type"] == "choice")
    assert all(cls[k] / sum(cls.values()) > 0.14 for k in ("count", "dim", "fillet", "mirror")), cls
    for name in ("cad_section_depth", "cad_dimension_read"):
        yes = [r["label"] for r in rows(name, 400, split="pos") if r["q"]["type"] == "noul"]
        assert 0.38 < sum(yes) / len(yes) < 0.62, (name, sum(yes), len(yes))
