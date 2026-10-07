"""Procedural CAD / technical-drawing image sets with exact labels (Hypothesis/V6.3.md section 5.2).

Parts are parametric (every feature is a number in a ``spec`` dict); solids are built with CadQuery on OpenCascade,
orthographic views come from OCCT hidden-line removal (``HLRBRep_Algo``), sections from a boolean cut plus the cut
faces, shaded views from a z-buffer over the kernel's tessellation, and everything is drawn with matplotlib (Agg,
bundled DejaVu / STIX fonts, our own title block). Labels are computed from the parameters, never from pixels.

Part families: plates and blocks (through / blind / counterbored holes, slots, pockets, bosses, rounded or chamfered
outline corners), L- and U-brackets (holes in the base and the legs, bend fillet), stepped shafts (grooves, keyway),
flanges (hub, bore, bolt circle), toothed wheels, and small assemblies (bolted cover, shaft in a flanged bushing, an
exploded stack).

Datasets (``<root>/<name>/{train,val}.jsonl``, PNG images under ``images/``):

* ``cad_feature_count``   (choice) how many holes / through holes / counterbored holes / slots / rounded corners /
                          teeth / grooves / shaft sections / bosses / bolts ... ; rank-uniform numeric options whose
                          first distractors are the named miscounts (counted the blind ones, counted the hidden ones,
                          counted the empty holes).
* ``cad_view_match``      (choice, 4) an isometric view (with a FRONT arrow) and four candidate panels: which one is
                          the TOP / FRONT / RIGHT / SECTION A-A view. The three distractors are the same view of
                          near-miss parts (a feature moved, removed, resized, or the part mirrored); every distractor
                          differs from the gold both in the asked view and in the isometric view.
* ``cad_dimension_read``  (choice / noul) a dimensioned drawing: read one dimension, add a chain, derive a wall, or
                          compare two. 30% of the choice rows carry the unit structure {a, b, conv(a), conv(b)}: in
                          half of them the title block is in the other unit (the trap) and the converted value is
                          the answer, in the other half the unit matches and the printed value is. The printed
                          numbers have the same distribution in both, so only the title block decides.
* ``cad_drawing_pair``    (noul / choice) two panels: same part? and, when not, what changed (hole count, a
                          dimension, a missing fillet, mirrored). Parts are chiral by construction, so a mirrored
                          part is never the same part turned over.
* ``cad_section_depth``   (choice / noul) a plan view with a cutting line and the section: through or blind, which
                          labelled hole is deepest, how many are through, hole type, remaining wall.
* ``cad_assembly``        (choice) assemblies with numbered balloons: how many bolts, which item touches which (from
                          a section), order of an exploded stack.

    python -m laya.visionjev_cad all            # 4,000 train / 400 val each (assembly 2,000 / 200), 8 workers
    python -m laya.visionjev_cad cad_view_match --train 200 --val 20 --root /tmp/x
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict

import numpy as np

from .visionjev_data import DATA_ROOT

MM_PER_INCH = 25.4
FONTS = ["DejaVu Sans", "DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif", "STIXGeneral"]  # bundled with matplotlib
#: view name -> (direction towards the viewer, direction drawn to the right). Up on the page is their cross product.
VIEWS = {"front": ((0, -1, 0), (1, 0, 0)), "top": ((0, 0, 1), (1, 0, 0)), "right": ((1, 0, 0), (0, 1, 0)),
         "iso": ((1, -1, 1), (1, 1, 0)), "iso_l": ((-1, -1, 1), (1, -1, 0)), "iso_hi": ((1, -1.0, 1.8), (1, 1, 0))}
VIEW_TITLE = {"front": "FRONT VIEW", "top": "TOP VIEW", "right": "RIGHT VIEW"}


def _cq():
    import cadquery as cq

    return cq


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def basis(view):
    """(right, up, towards-viewer) unit vectors of a named view."""
    n, x = VIEWS[view]
    n, x = _unit(n), _unit(x)
    return x, np.cross(n, x), n


def project(pts, view):
    x, y, n = basis(view)
    p = np.asarray(pts, float)
    return np.stack([p @ x, p @ y], -1)


# ---- kernel: hidden-line projection, sections, tessellation ---------------------------------------------------------


def _edge_points(edge):
    """Polyline of one TopoDS_Edge (3D points, in edge parameter order)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line

    a = BRepAdaptor_Curve(edge)
    t0, t1 = a.FirstParameter(), a.LastParameter()
    kind = a.GetType()
    if kind == GeomAbs_Line:
        n = 2
    elif kind in (GeomAbs_Circle, GeomAbs_Ellipse):
        n = max(5, int(abs(t1 - t0) / (2 * math.pi) * 72) + 2)
    else:
        n = 28
    out = np.empty((n, 3))
    for i, t in enumerate(np.linspace(t0, t1, n)):
        p = a.Value(float(t))
        out[i] = (p.X(), p.Y(), p.Z())
    return out


def hlr(shape, view):
    """OCCT hidden-line removal. Returns {"vis": [...], "hid": [...], "smooth": [...]} polylines (n, 2) in the view's
    page coordinates (right, up). ``smooth`` are the visible tangent edges (fillet boundaries)."""
    from OCP.BRepLib import BRepLib
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.HLRAlgo import HLRAlgo_Projector
    from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    x, _, n = basis(view)
    algo = HLRBRep_Algo()
    algo.Add(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*map(float, n)), gp_Dir(*map(float, x)))))
    algo.Update()
    algo.Hide()
    h = HLRBRep_HLRToShape(algo)
    out = {}
    for key, fns in (("vis", (h.VCompound, h.OutLineVCompound)), ("hid", (h.HCompound, h.OutLineHCompound)),
                     ("smooth", (h.Rg1LineVCompound,))):
        polys = []
        for fn in fns:
            c = fn()
            if c.IsNull():
                continue
            BRepLib.BuildCurves3d_s(c)
            ex = TopExp_Explorer(c, TopAbs_EDGE)
            while ex.More():
                p = _edge_points(TopoDS.Edge_s(ex.Current()))[:, :2]
                if len(p) >= 2 and np.abs(p[-1] - p[0]).sum() + np.abs(np.diff(p, axis=0)).sum() > 1e-6:
                    polys.append(p)
                ex.Next()
        out[key] = polys
    return out


def _wire_points(wire):
    """Ordered closed polyline (3D) of a TopoDS_Wire."""
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_REVERSED

    pts = []
    ex = BRepTools_WireExplorer(wire)
    while ex.More():
        e = ex.Current()
        p = _edge_points(e)
        if e.Orientation() == TopAbs_REVERSED:
            p = p[::-1]
        pts.append(p[:-1])
        ex.Next()
    if not pts:  # a wire the explorer cannot order (e.g. one closed edge): take its edges as they come
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        te = TopExp_Explorer(wire, TopAbs_EDGE)
        while te.More():
            pts.append(_edge_points(TopoDS.Edge_s(te.Current()))[:-1])
            te.Next()
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _halfspace_box(axis, pos, sign, R=2000.0):
    """Axis-aligned box covering coordinate[axis] > pos (sign +1) or < pos (sign -1)."""
    cq = _cq()
    lo = [-R, -R, -R]
    size = [2 * R, 2 * R, 2 * R]
    lo[axis] = pos if sign > 0 else pos - R
    size[axis] = R
    return cq.Solid.makeBox(size[0], size[1], size[2], cq.Vector(*lo))


def section(shape, view, pos):
    """Cut ``shape`` with the plane through ``pos`` perpendicular to the viewing direction of an axis-aligned view
    (front: y = pos, right: x = pos, top: z = pos). Returns (cut shape behind the plane, [(outer, [inner, ...])] cut
    faces as page-coordinate polygons)."""
    cq = _cq()
    n = np.array(VIEWS[view][0], float)
    axis = int(np.argmax(np.abs(n)))
    sign = 1 if n[axis] > 0 else -1
    rest = shape.cut(_halfspace_box(axis, pos, sign))
    origin = [0.0, 0.0, 0.0]
    origin[axis] = pos
    plane = cq.Face.makePlane(4000, 4000, cq.Vector(*origin), cq.Vector(*map(float, n)))
    faces = []
    for f in shape.intersect(plane).Faces():
        outer = project(_wire_points(f.outerWire().wrapped), view)
        inner = [r for r in (project(_wire_points(w.wrapped), view) for w in f.innerWires()) if len(r) >= 3]
        if len(outer) >= 3:
            faces.append((outer, inner))
    return rest, faces


def tessellate(shape, tol=0.08, ang=0.18):
    verts, tris = shape.tessellate(tol, ang)
    return np.array([(v.x, v.y, v.z) for v in verts], float), np.array(tris, np.int64).reshape(-1, 3)


def _raster_loop(tri, col, tid, W, H, img, idb):
    zb = np.full((H, W), -1e30)
    for t in range(tri.shape[0]):
        x0, y0, z0 = tri[t, 0, 0], tri[t, 0, 1], tri[t, 0, 2]
        x1, y1, z1 = tri[t, 1, 0], tri[t, 1, 1], tri[t, 1, 2]
        x2, y2, z2 = tri[t, 2, 0], tri[t, 2, 1], tri[t, 2, 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        minx = max(int(math.floor(min(x0, x1, x2))), 0)
        maxx = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(math.floor(min(y0, y1, y2))), 0)
        maxy = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        for y in range(miny, maxy + 1):
            py = y + 0.5
            for x in range(minx, maxx + 1):
                px = x + 0.5
                a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                c = 1.0 - a - b
                if a >= -1e-9 and b >= -1e-9 and c >= -1e-9:
                    z = a * z0 + b * z1 + c * z2
                    if z > zb[y, x]:
                        zb[y, x] = z
                        img[y, x, 0] = col[t, 0]
                        img[y, x, 1] = col[t, 1]
                        img[y, x, 2] = col[t, 2]
                        idb[y, x] = tid[t]


_RASTER = None


def _raster():
    """The z-buffer loop, compiled with numba when it is installed (it ships with CadQuery's dependencies)."""
    global _RASTER
    if _RASTER is None:
        try:
            import numba

            _RASTER = numba.njit(cache=False)(_raster_loop)
        except Exception:  # noqa: BLE001 - plain Python is only slower
            _RASTER = _raster_loop
    return _RASTER


def shade(parts, view, scale, box, bg, ss=2):
    """Flat-shaded orthographic render. ``parts``: [(shape, rgb)], ``box``: page-coordinate bbox (u0, v0, u1, v1) in
    mm, ``scale``: px per mm. Returns (uint8 image [H, W, 3], part-index buffer [H, W], -1 = background); row 0 is
    the top of the picture."""
    x, y, n = basis(view)
    light = _unit(-0.45 * x + 0.55 * y + 0.75 * n)
    u0, v0, u1, v1 = box
    W, H = int(math.ceil((u1 - u0) * scale)) * ss, int(math.ceil((v1 - v0) * scale)) * ss
    tris, cols, ids = [], [], []
    for k, (shape, rgb) in enumerate(parts):
        V, T = tessellate(shape)
        if not len(T):
            continue
        P = V[T]  # (t, 3, 3)
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        ln = np.linalg.norm(nrm, axis=1)
        keep = ln > 1e-12
        P, nrm = P[keep], nrm[keep] / ln[keep, None]
        nrm = np.where((nrm @ n)[:, None] < 0, -nrm, nrm)  # only faces turned to the viewer are ever seen
        lam = 0.5 + 0.5 * np.clip(nrm @ light, 0, 1)
        tri = np.empty((len(P), 3, 3))
        tri[:, :, 0] = (P @ x - u0) * scale * ss
        tri[:, :, 1] = (v1 - P @ y) * scale * ss
        tri[:, :, 2] = P @ n
        tris.append(tri)
        cols.append((lam[:, None] * np.asarray(rgb, float)[None, :]).astype(np.float32))
        ids.append(np.full(len(P), k, np.int32))
    img = np.empty((H, W, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    idb = np.full((H, W), -1, np.int32)
    if tris:
        _raster()(np.concatenate(tris), np.concatenate(cols), np.concatenate(ids), W, H, img, idb)
    img = img.reshape(H // ss, ss, W // ss, ss, 3).mean((1, 3))
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), idb[::ss, ::ss]


class View:
    """A 2D drawing in page millimetres: visible / hidden / thin polylines, hatched cut faces, bbox."""

    def __init__(self, vis, hid=(), smooth=(), faces=(), centre=(), extra=None):
        self.vis, self.hid, self.smooth, self.faces, self.centre = list(vis), list(hid), list(smooth), list(faces), list(centre)
        pts = np.concatenate(self.vis + self.hid + [f[0] for f in self.faces]) if (self.vis or self.hid or self.faces) \
            else np.zeros((1, 2))
        self.gb = self.b = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
        if extra is not None:
            self.grow(extra)

    def grow(self, pts):
        """Extend the layout bbox (not the geometry bbox ``gb``) to include annotation points."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        self.b = (min(self.b[0], float(pts[:, 0].min())), min(self.b[1], float(pts[:, 1].min())),
                  max(self.b[2], float(pts[:, 0].max())), max(self.b[3], float(pts[:, 1].max())))

    @property
    def size(self):
        return self.b[2] - self.b[0], self.b[3] - self.b[1]


def ortho(shape, view, hidden=True):
    h = hlr(shape, view)
    return View(h["vis"], h["hid"] if hidden else (), h["smooth"])


def section_view(shapes, view, pos):
    """Section of one shape or of several parts (each part gets its own hatch index)."""
    cq = _cq()
    shapes = shapes if isinstance(shapes, (list, tuple)) else [shapes]
    rests, faces = [], []
    for k, s in enumerate(shapes):
        rest, fs = section(s, view, pos)
        rests.append(rest)
        faces += [(o, i, k) for o, i in fs]
    comp = rests[0] if len(rests) == 1 else cq.Compound.makeCompound(rests)
    return View(hlr(comp, view)["vis"], faces=faces)


# ---- drawing sheet (matplotlib) -------------------------------------------------------------------------------------


def sample_style(rng):
    font = rng.choice(FONTS)
    ink = rng.choice([(0.04, 0.04, 0.04), (0.04, 0.04, 0.04), (0.08, 0.1, 0.28), (0.18, 0.18, 0.18)])
    return {
        "bg": rng.choice([(1, 1, 1), (1, 1, 1), (1, 1, 1), (0.94, 0.94, 0.94), (0.89, 0.89, 0.89), (0.97, 0.97, 0.95)]),
        "ink": ink, "dimcol": rng.choice([ink, ink, (0.1, 0.2, 0.55), (0.45, 0.1, 0.1), (0.25, 0.25, 0.25)]),
        "lw": rng.uniform(1.5, 2.5), "hid_lw": rng.uniform(0.9, 1.2), "hid_dash": rng.choice([(4, 2.5), (3, 2), (6, 3)]),
        "thin": rng.uniform(0.7, 1.0), "font": font, "fs": rng.choice([12, 13, 13, 14]),
        "dia": "⌀" if font != "DejaVu Serif" and rng.random() < 0.7 else "Ø",
        "term": rng.choice(["arrow", "arrow", "arrow", "tick", "dot", "open"]), "dimtext": rng.choice(["above", "above", "break"]),
        "hatch": {"ang": rng.choice([45, 45, 45, 135, 60, 30]), "gap": rng.uniform(5.5, 9.0), "cross": rng.random() < 0.12,
                  "fill": rng.choice([None, None, (0.86, 0.86, 0.86), (0.8, 0.86, 0.93)])},
        "centre": rng.random() < 0.5, "border": rng.choice(["single", "single", "double", "none"]),
        "rot": rng.uniform(-1, 1) if rng.random() < 0.3 else 0.0, "noise": rng.uniform(2, 4.5) if rng.random() < 0.10 else 0.0,
        "blur": rng.random() < 0.08, "seed": rng.randrange(1 << 30),
        "shade_rgb": rng.choice([(0.72, 0.76, 0.82), (0.8, 0.8, 0.8), (0.85, 0.78, 0.62), (0.62, 0.74, 0.86),
                                 (0.7, 0.8, 0.7), (0.86, 0.7, 0.66)]),
    }


class Sheet:
    """A pixel-coordinate canvas (origin bottom-left) with the drafting primitives the generators need."""

    def __init__(self, w, h, st):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.w, self.h, self.st = w, h, st
        self.fig = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor=st["bg"])
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.rend = self.canvas.get_renderer()
        self.boxes = []  # reserved text boxes (x0, y0, x1, y1)

    # -- primitives
    def poly(self, polys, lw=None, col=None, dash=None, z=3):
        from matplotlib.collections import LineCollection

        polys = [np.asarray(p, float) for p in polys]
        if not polys:
            return
        self.ax.add_collection(LineCollection(polys, linewidths=(lw or self.st["lw"]) * 0.72, colors=[col or self.st["ink"]],
                                              linestyles=[(0, dash)] if dash else "solid", zorder=z, capstyle="round",
                                              joinstyle="round"))

    def fill(self, pts, col, z=1, edge=None, lw=1.0):
        from matplotlib.patches import Polygon

        self.ax.add_patch(Polygon(np.asarray(pts, float), closed=True, facecolor=col, zorder=z,
                                  edgecolor=edge if edge is not None else "none", linewidth=lw * 0.72))

    def circle(self, x, y, r, fc="none", ec=None, lw=None, z=5):
        from matplotlib.patches import Circle

        self.ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec or self.st["ink"], linewidth=(lw or self.st["thin"]) * 0.72,
                                 zorder=z))

    def measure(self, s, size=None, weight="normal"):
        t = self.ax.text(0, 0, s, fontsize=(size or self.st["fs"]) * 0.72, family=self.st["font"], weight=weight)
        bb = t.get_window_extent(self.rend)
        t.remove()
        return bb.width, bb.height

    def text(self, x, y, s, size=None, ha="center", va="center", rot=0, col=None, weight="normal", bg=None, z=7,
             reserve=True):
        kw = {}
        if bg is not None:
            kw["bbox"] = dict(facecolor=bg, edgecolor="none", pad=1.2)
        t = self.ax.text(x, y, s, fontsize=(size or self.st["fs"]) * 0.72, family=self.st["font"], ha=ha, va=va, rotation=rot,
                         color=col or self.st["ink"], weight=weight, zorder=z, **kw)
        bb = t.get_window_extent(self.rend)
        box = (bb.x0, bb.y0, bb.x1, bb.y1)
        if reserve:
            self.boxes.append(box)
        return box

    def free(self, box, pad=2):
        x0, y0, x1, y1 = box
        if x0 < 4 or y0 < 4 or x1 > self.w - 4 or y1 > self.h - 4:
            return False
        return not any(x0 - pad < b[2] and b[0] < x1 + pad and y0 - pad < b[3] and b[1] < y1 + pad for b in self.boxes)

    def _term(self, x, y, dx, dy, col):
        """Dimension-line terminator at (x, y); (dx, dy) points from the dimension line out through the tip."""
        kind, a = self.st["term"], 9.0
        nx, ny = -dy, dx
        if kind == "arrow":
            self.fill([(x, y), (x - a * dx + 3 * nx, y - a * dy + 3 * ny), (x - a * dx - 3 * nx, y - a * dy - 3 * ny)], col, z=6)
        elif kind == "open":
            self.poly([[(x - a * dx + 3.5 * nx, y - a * dy + 3.5 * ny), (x, y), (x - a * dx - 3.5 * nx, y - a * dy - 3.5 * ny)]],
                      self.st["thin"], col, z=6)
        elif kind == "tick":
            self.poly([[(x - 4.5 * (dx + nx), y - 4.5 * (dy + ny)), (x + 4.5 * (dx + nx), y + 4.5 * (dy + ny))]],
                      self.st["thin"] * 1.6, col, z=6)
        else:
            self.circle(x, y, 2.6, fc=col, ec=col, z=6)

    def dim_h(self, xa, xb, ya, yb, yl, s, tight=False):
        """Horizontal dimension between feature points (xa, ya), (xb, yb) with the dimension line at y = yl."""
        st, c = self.st, self.st["dimcol"]
        for x, y in ((xa, ya), (xb, yb)):
            sg = 1 if yl >= y else -1
            self.poly([[(x, y + sg * 4), (x, yl + sg * 5)]], st["thin"], c, z=4)
        x0, x1 = sorted((xa, xb))
        tw, th = self.measure(s)
        if x1 - x0 >= tw + 26 or (tight and x1 - x0 >= max(tw + 6, 26)):
            self.poly([[(x0, yl), (x1, yl)]], st["thin"], c, z=4)
            self._term(x0, yl, -1, 0, c)
            self._term(x1, yl, 1, 0, c)
            if st["dimtext"] == "above" or tight:
                return self.text((x0 + x1) / 2, yl + th / 2 + 2.5, s, col=c)
            return self.text((x0 + x1) / 2, yl, s, col=c, bg=st["bg"])
        right = x1 + 16 + tw + 6 < self.w - 8  # too narrow: arrows outside, text beside
        if right:
            self.poly([[(x0 - 14, yl), (x1 + 16 + tw + 2, yl)]], st["thin"], c, z=4)
            box = self.text(x1 + 16 + tw / 2, yl + th / 2 + 2.5, s, col=c)
        else:
            self.poly([[(x0 - 16 - tw - 2, yl), (x1 + 14, yl)]], st["thin"], c, z=4)
            box = self.text(x0 - 16 - tw / 2, yl + th / 2 + 2.5, s, col=c)
        self._term(x0, yl, 1, 0, c)
        self._term(x1, yl, -1, 0, c)
        return box

    def dim_v(self, ya, yb, xa, xb, xl, s):
        """Vertical dimension between feature points (xa, ya), (xb, yb) with the dimension line at x = xl."""
        st, c = self.st, self.st["dimcol"]
        for x, y in ((xa, ya), (xb, yb)):
            sg = 1 if xl >= x else -1
            self.poly([[(x + sg * 4, y), (xl + sg * 5, y)]], st["thin"], c, z=4)
        y0, y1 = sorted((ya, yb))
        tw, th = self.measure(s)
        if y1 - y0 >= tw + 26:
            self.poly([[(xl, y0), (xl, y1)]], st["thin"], c, z=4)
            self._term(xl, y0, 0, -1, c)
            self._term(xl, y1, 0, 1, c)
            if st["dimtext"] == "above":
                return self.text(xl - th / 2 - 2.5, (y0 + y1) / 2, s, col=c, rot=90)
            return self.text(xl, (y0 + y1) / 2, s, col=c, rot=90, bg=st["bg"])
        up = y1 + 16 + tw + 6 < self.h - 8
        if up:
            self.poly([[(xl, y0 - 14), (xl, y1 + 16 + tw + 2)]], st["thin"], c, z=4)
            box = self.text(xl - th / 2 - 2.5, y1 + 16 + tw / 2, s, col=c, rot=90)
        else:
            self.poly([[(xl, y0 - 16 - tw - 2), (xl, y1 + 14)]], st["thin"], c, z=4)
            box = self.text(xl - th / 2 - 2.5, y0 - 16 - tw / 2, s, col=c, rot=90)
        self._term(xl, y0, 0, 1, c)
        self._term(xl, y1, 0, -1, c)
        return box

    def leader(self, tip, s, angles=(45, 135, 315, 225, 70, 110, 290, 250, 20, 160, 340, 200), lengths=(30, 46, 64, 84),
               arrow=True, col=None, weight="normal", size=None):
        """A leader from ``tip`` to a text label; the first (length, angle) whose label box is free is used."""
        c = col or self.st["dimcol"]
        tw, th = self.measure(s, size, weight)
        pick = None
        for L in lengths:
            for a in angles:
                dx, dy = math.cos(math.radians(a)), math.sin(math.radians(a))
                ex, ey = tip[0] + L * dx, tip[1] + L * dy
                sg = 1 if dx >= 0 else -1
                box = (min(ex + sg * 10, ex + sg * (12 + tw)) - 1, ey - th / 2 - 1, max(ex + sg * 10, ex + sg * (12 + tw)) + 1,
                       ey + th / 2 + 1)
                if self.free(box):
                    pick = (ex, ey, sg, dx, dy)
                    break
            if pick:
                break
        if pick is None:
            a = angles[0]
            dx, dy = math.cos(math.radians(a)), math.sin(math.radians(a))
            pick = (tip[0] + lengths[0] * dx, tip[1] + lengths[0] * dy, 1 if dx >= 0 else -1, dx, dy)
        ex, ey, sg, dx, dy = pick
        self.poly([[tip, (ex, ey), (ex + sg * 9, ey)]], self.st["thin"], c, z=5)
        if arrow:
            nx, ny = -dy, dx
            self.fill([tip, (tip[0] + 9 * dx + 3 * nx, tip[1] + 9 * dy + 3 * ny), (tip[0] + 9 * dx - 3 * nx, tip[1] + 9 * dy - 3 * ny)],
                      c, z=6)
        else:
            self.circle(tip[0], tip[1], 2.2, fc=c, ec=c, z=6)
        return self.text(ex + sg * 11, ey, s, size=size, ha="left" if sg > 0 else "right", col=c, weight=weight)

    def balloon(self, tip, s, angles=(60, 120, 30, 150, 90, 300, 240, 330, 210, 0, 180), lengths=(46, 62, 80, 100), r=12):
        """An item balloon (circled number) with a leader ending in a dot on the part."""
        c = self.st["ink"]
        pick = None
        for L in lengths:
            for a in angles:
                dx, dy = math.cos(math.radians(a)), math.sin(math.radians(a))
                cx, cy = tip[0] + L * dx, tip[1] + L * dy
                box = (cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2)
                if self.free(box, pad=3):
                    pick = (cx, cy, dx, dy)
                    break
            if pick:
                break
        if pick is None:
            dx, dy = math.cos(math.radians(angles[0])), math.sin(math.radians(angles[0]))
            pick = (tip[0] + lengths[0] * dx, tip[1] + lengths[0] * dy, dx, dy)
        cx, cy, dx, dy = pick
        self.poly([[tip, (cx - r * dx, cy - r * dy)]], self.st["thin"], c, z=6)
        self.circle(tip[0], tip[1], 2.4, fc=c, ec=c, z=7)
        self.circle(cx, cy, r, fc=self.st["bg"], ec=c, lw=self.st["thin"] * 1.3, z=7)
        self.boxes.append((cx - r, cy - r, cx + r, cy + r))
        self.text(cx, cy, s, size=self.st["fs"] + 1, weight="bold", z=8, reserve=False)
        return cx, cy

    def hatch(self, outer, inners, k=0):
        """Section lining of one cut face (polygon with holes); ``k`` picks the direction / spacing per part."""
        from matplotlib.collections import LineCollection
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        h = self.st["hatch"]
        verts, codes = [], []
        for ring in [outer] + list(inners):
            ring = np.asarray(ring, float)
            verts += list(ring) + [ring[0]]
            codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 1) + [Path.CLOSEPOLY]
        fills = [h["fill"], (0.93, 0.87, 0.78), (0.82, 0.9, 0.82), (0.9, 0.84, 0.9), (0.8, 0.86, 0.93), (0.92, 0.92, 0.8)]
        fc = fills[k % len(fills)] if (h["fill"] is not None) else "none"
        patch = PathPatch(Path(np.array(verts), codes), facecolor=fc if fc is not None else "none", edgecolor="none", zorder=1)
        self.ax.add_patch(patch)
        ang = math.radians(h["ang"] + (90 if k % 2 else 0) + (25 if (k // 2) % 2 else 0))
        gap = h["gap"] * (1.0, 0.72, 1.35)[(k // 2) % 3]
        o = np.asarray(outer, float)
        c = o.mean(0)
        R = float(np.abs(o - c).max()) * 1.5 + 4
        d, nrm = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
        segs = [[c + nrm * t - d * R, c + nrm * t + d * R] for t in np.arange(-R, R, gap)]
        if h["cross"]:
            segs += [[c + d * t - nrm * R, c + d * t + nrm * R] for t in np.arange(-R, R, gap)]
        lc = LineCollection(segs, linewidths=self.st["thin"] * 0.72 * 0.9, colors=[self.st["ink"]], zorder=1.5)
        self.ax.add_collection(lc)
        lc.set_clip_path(patch)

    # -- composite
    def border(self):
        b, m = self.st["border"], 9
        if b == "none":
            return
        self.poly([[(m, m), (self.w - m, m), (self.w - m, self.h - m), (m, self.h - m), (m, m)]], self.st["lw"] * 0.9, z=2)
        if b == "double":
            m2 = m + 5
            self.poly([[(m2, m2), (self.w - m2, m2), (self.w - m2, self.h - m2), (m2, self.h - m2), (m2, m2)]], self.st["thin"], z=2)

    def title_block(self, cells, x1=None, y0=18, cols=3, cw=112, ch=25):
        """Our own title block: a grid of (caption, value) cells anchored at the bottom-right corner (x1, y0)."""
        x1 = (self.w - 18) if x1 is None else x1
        rows = int(math.ceil(len(cells) / cols))
        x0, y1 = x1 - cols * cw, y0 + rows * ch
        self.fill([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], self.st["bg"], z=5.5, edge=self.st["ink"], lw=self.st["lw"] * 0.8)
        for k, (cap, val) in enumerate(cells):
            r, c = divmod(k, cols)
            cx, cy = x0 + c * cw, y1 - (r + 1) * ch
            self.poly([[(cx, cy), (cx + cw, cy), (cx + cw, cy + ch), (cx, cy + ch), (cx, cy)]], self.st["thin"], z=5.6)
            self.text(cx + 3, cy + ch - 2, cap, size=7.5, ha="left", va="top", z=5.8, reserve=False)
            size = 11.0
            while size > 6.5 and self.measure(val, size, "bold")[0] > cw - 8:
                size -= 0.5
            self.text(cx + cw / 2, cy + ch * 0.36, val, size=size, z=5.8, reserve=False, weight="bold")
        self.boxes.append((x0 - 2, y0 - 2, x1 + 2, y1 + 2))
        return x0, y0, x1, y1

    def image(self):
        """Render, then the light scan effects (rotation up to 1 degree, noise, blur). Returns an RGB PIL image."""
        from PIL import Image, ImageFilter

        self.canvas.draw()
        im = Image.fromarray(np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy())
        st = self.st
        if st["rot"]:
            im = im.rotate(st["rot"], resample=Image.BICUBIC, fillcolor=tuple(int(255 * c) for c in st["bg"]))
        if st["blur"]:
            im = im.filter(ImageFilter.GaussianBlur(0.6))
        if st["noise"]:
            g = np.random.default_rng(st["seed"])
            a = np.asarray(im, np.float32) + g.normal(0, st["noise"], (im.size[1], im.size[0], 1))
            im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return im


class Placed:
    """A view placed on a sheet: page mm -> sheet px."""

    def __init__(self, view, cx, cy, s):
        self.view, self.s = view, s
        self.ox = cx - s * (view.b[0] + view.b[2]) / 2
        self.oy = cy - s * (view.b[1] + view.b[3]) / 2
        self.x0, self.y0 = self.P(view.b[0], view.b[1])
        self.x1, self.y1 = self.P(view.b[2], view.b[3])

    def P(self, u, v):
        return self.ox + self.s * u, self.oy + self.s * v

    def T(self, poly):
        return np.asarray(poly, float) * self.s + (self.ox, self.oy)


def fit(views, w, h, smax=4.0):
    """Largest common scale at which every view fits a (w, h) px cell."""
    return min([smax] + [min(w / max(v.size[0], 1e-6), h / max(v.size[1], 1e-6)) for v in views])


def draw_view(sh, view, cx, cy, s, hidden=True, smooth=False, lw=None):
    p = Placed(view, cx, cy, s)
    st = sh.st
    for outer, inner, k in view.faces:
        sh.hatch(p.T(outer), [p.T(i) for i in inner], k)
    if hidden and view.hid:
        sh.poly([p.T(q) for q in view.hid], st["hid_lw"], dash=st["hid_dash"], z=2.5)
    if view.centre:
        sh.poly([p.T(q) for q in view.centre], st["thin"] * 0.9, col=st["dimcol"], dash=(12, 3, 2, 3), z=2.2)
    if smooth and view.smooth:
        sh.poly([p.T(q) for q in view.smooth], st["thin"], z=2.8)
    sh.poly([p.T(q) for q in view.vis], lw or st["lw"], z=3)
    return p


def draw_shaded(sh, parts, view, cx, cy, s, edges=True, pad=1.0, v=None):
    """Shaded render of [(shape, rgb)] centred at (cx, cy); returns (Placed, part-index buffer, its px origin).
    ``v``: the precomputed visible-line View (its layout bbox may include annotations)."""
    cq = _cq()
    if v is None:
        comp = parts[0][0] if len(parts) == 1 else cq.Compound.makeCompound([p[0] for p in parts])
        h = hlr(comp, view)
        v = View(h["vis"], (), h["smooth"])
    p = Placed(v, cx, cy, s)
    box = (v.gb[0] - pad, v.gb[1] - pad, v.gb[2] + pad, v.gb[3] + pad)
    img, idb = shade(parts, view, s, box, sh.st["bg"])
    x0, y0 = p.P(box[0], box[1])
    sh.ax.imshow(img, extent=(x0, x0 + img.shape[1], y0, y0 + img.shape[0]), zorder=1.2, interpolation="bilinear", aspect="auto")
    sh.ax.set_xlim(0, sh.w)
    sh.ax.set_ylim(0, sh.h)
    if edges:
        sh.poly([p.T(q) for q in v.vis], max(1.0, sh.st["lw"] * 0.7), col=(0.08, 0.08, 0.08), z=3)
    return p, idb, (x0, y0)


def visible_anchor(idb, origin, k, min_depth=2.5):
    """The most interior visible pixel (sheet px) of part ``k`` in a part-index buffer, or None if the part shows
    less than a ``min_depth`` px deep region."""
    from scipy import ndimage

    m = np.pad(idb == k, 1)
    if not m.any():
        return None
    dist = ndimage.distance_transform_edt(m)
    if dist.max() < min_depth:
        return None
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return origin[0] + (x - 1) + 0.5, origin[1] + (idb.shape[0] - (y - 1)) - 0.5


# ---- parametric parts -------------------------------------------------------------------------------------------------
X3, Y3, Z3 = np.eye(3)
PRISM = ("plate", "block", "bracket")


def _poly_face(pts, radii, chamf, to3d):
    """Planar face of a closed polygon with a fillet radius / chamfer per vertex (0 = sharp)."""
    cq = _cq()
    n = len(pts)
    P = [np.asarray(p, float) for p in pts]
    corner = []
    for i in range(n):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % n]
        d1, d2 = _unit(p0 - p1), _unit(p2 - p1)
        r = radii[i] if radii else 0
        c = chamf[i] if chamf else 0
        if r > 0:
            half = math.acos(float(np.clip(d1 @ d2, -1, 1))) / 2
            t = r / math.tan(half)
            ctr = p1 + _unit(d1 + d2) * (r / math.sin(half))
            corner.append((p1 + d1 * t, p1 + d2 * t, ctr + _unit(p1 - ctr) * r))
        elif c > 0:
            corner.append((p1 + d1 * c, p1 + d2 * c, None))
        else:
            corner.append((p1, p1, None))
    V = lambda p: cq.Vector(*map(float, to3d(p)))  # noqa: E731
    edges = []
    for i in range(n):
        a, b, mid = corner[i]
        if mid is not None:
            edges.append(cq.Edge.makeThreePointArc(V(a), V(mid), V(b)))
        elif np.abs(a - b).sum() > 1e-7:
            edges.append(cq.Edge.makeLine(V(a), V(b)))
        nxt = corner[(i + 1) % n][0]
        if np.abs(b - nxt).sum() > 1e-7:
            edges.append(cq.Edge.makeLine(V(b), V(nxt)))
    return cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))


def _prism(pts, radii, chamf, origin, eu, ev, vec):
    cq = _cq()
    o = np.asarray(origin, float)
    return cq.Solid.extrudeLinear(_poly_face(pts, radii, chamf, lambda p: o + eu * p[0] + ev * p[1]), cq.Vector(*map(float, vec)))


def _cyl(r, h, p, d):
    cq = _cq()
    return cq.Solid.makeCylinder(float(r), float(h), cq.Vector(*map(float, p)), cq.Vector(*map(float, d)))


def _obround(l, w, ang):
    """Polygon + radii of a slot of overall length l, width w, at ``ang`` degrees in its face."""
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    pts = [(-l / 2, -w / 2), (l / 2, -w / 2), (l / 2, w / 2), (-l / 2, w / 2)]
    return [(x * c - y * s, x * s + y * c) for x, y in pts], [w / 2 * 0.9995] * 4


def frames(spec):
    """Face name -> (origin, eu, ev, inward, thickness, (u0, u1, v0, v1)) for the prismatic families."""
    if spec["fam"] in ("plate", "block"):
        L, W, T = spec["L"], spec["W"], spec["T"]
        return {"top": (Z3 * T, X3, Y3, -Z3, T, (0, L, 0, W)), "bottom": (Z3 * 0, X3, Y3, Z3, T, (0, L, 0, W))}
    L, W, H, T, r = spec["L"], spec["W"], spec["H"], spec["T"], spec.get("rin", 0)
    U = spec["shape"] == "U"
    fr = {"base": (Z3 * T, X3, Y3, -Z3, T, (T + r, L - (T + r if U else 0), 0, W)),
          "leg0": (X3 * T, Y3, Z3, -X3, T, (0, W, T + r, H))}
    if U:
        fr["leg1"] = (X3 * (L - T), Y3, Z3, X3, T, (0, W, T + r, H))
    return fr


def _half(ft):
    k = ft["kind"]
    if k == "hole":
        r = max(ft["d"], ft["cb"][0] if ft.get("cb") else 0) / 2
        return r, r
    if k == "boss":
        return ft["d"] / 2, ft["d"] / 2
    if k == "slot":
        return (ft["l"] / 2, ft["w"] / 2) if ft.get("ang", 0) == 0 else (ft["w"] / 2, ft["l"] / 2)
    return ft["l"] / 2, ft["w"] / 2  # pocket


def outline(spec):
    o = spec.get("outline")
    if o:
        return o["pts"], o.get("r"), o.get("c")
    L, W = spec["L"], spec["W"]
    return [(0, 0), (L, 0), (L, W), (0, W)], spec.get("cr"), spec.get("cc")


def _edge_dist(p, pts):
    best = 1e9
    for i in range(len(pts)):
        a, b = np.asarray(pts[i - 1], float), np.asarray(pts[i], float)
        t = float(np.clip(((p - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def _inside(p, pts):
    x, y, c = p[0], p[1], False
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i - 1], pts[i]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def feats_ok(spec, margin=4.0, gap=4.0):
    """Every feature inside its face with a margin, no two overlapping (top and bottom share one footprint)."""
    fr = frames(spec)
    fs = spec["feats"]
    for ft in fs:
        _, _, _, _, th, (u0, u1, v0, v1) = fr[ft["face"]]
        hu, hv = _half(ft)
        if ft["u"] - hu < u0 + margin or ft["u"] + hu > u1 - margin or ft["v"] - hv < v0 + margin or ft["v"] + hv > v1 - margin:
            return False
        if spec.get("outline") and ft["face"] in ("top", "bottom"):
            p = np.array([ft["u"], ft["v"]], float)
            if not _inside(p, spec["outline"]["pts"]) or _edge_dist(p, spec["outline"]["pts"]) < max(hu, hv) + margin:
                return False
        for key in ("depth", "bd"):
            if ft.get(key) is not None and not 2 <= ft[key] <= th + (ft.get("h", 0) if key == "bd" else 0) - 2.5:
                return False
        if ft.get("cb") and not (ft["cb"][0] >= ft["d"] + 4 and 2 <= ft["cb"][1] <= (ft.get("depth") or th) - 2):
            return False
        if ft["kind"] == "boss" and ft.get("bore") and ft["bore"] > ft["d"] - 5:
            return False
    grp = lambda f: "z" if f["face"] in ("top", "bottom") else f["face"]  # noqa: E731
    for i in range(len(fs)):
        for j in range(i):
            a, b = fs[i], fs[j]
            if grp(a) != grp(b):
                continue
            (ha, va), (hb, vb) = _half(a), _half(b)
            du, dv = abs(a["u"] - b["u"]), abs(a["v"] - b["v"])
            if a["kind"] in ("hole", "boss") and b["kind"] in ("hole", "boss"):
                if math.hypot(du, dv) < ha + hb + gap:
                    return False
            elif du < ha + hb + gap and dv < va + vb + gap:
                return False
    return True


def valid(spec):
    f = spec["fam"]
    if f in PRISM:
        if f == "bracket" and not (spec["L"] >= 3 * spec["T"] + 10 and spec["H"] >= 2 * spec["T"] + 8):
            return False
        return min(spec["L"], spec["W"], spec["T"]) >= 3 and feats_ok(spec)
    if f == "shaft":
        segs = spec["segs"]
        if any(abs(segs[i][0] - segs[i + 1][0]) < 5 for i in range(len(segs) - 1)) or any(d < 10 or l < 10 for d, l in segs):
            return False
        used = defaultdict(list)
        for g in spec.get("grooves", []):
            d, l = segs[g["seg"]]
            if not (g["off"] >= 4 and g["off"] + g["w"] <= l - 4 and 6 <= g["dg"] <= d - 4):
                return False
            used[g["seg"]].append((g["off"], g["off"] + g["w"]))
        k = spec.get("keyway")
        if k:
            d, l = segs[k["seg"]]
            if not (k["off"] >= 3 and k["off"] + k["l"] <= l - 3 and k["w"] <= d * 0.45 and k["l"] >= k["w"] + 3):
                return False
            used[k["seg"]].append((k["off"], k["off"] + k["l"]))
        for iv in used.values():
            iv.sort()
            if any(iv[i + 1][0] < iv[i][1] + 3 for i in range(len(iv) - 1)):
                return False
        return True
    if f == "flange":
        cd = spec["cb"][0] if spec.get("cb") else spec["bolt_d"]
        ok = spec["bore_d"] <= spec["hub_d"] - 8 and spec["pcd"] - cd >= spec["hub_d"] + 6 and spec["pcd"] + cd <= spec["OD"] - 6
        ok = ok and spec["pcd"] * math.sin(math.pi / spec["n"]) >= cd + 4 and spec["bore_d"] >= 8 and spec["n"] >= 2
        if spec.get("bore_depth") is not None:
            ok = ok and 4 <= spec["bore_depth"] <= spec["T"] + spec["hub_h"] - 3
        if spec.get("cb"):
            ok = ok and spec["cb"][0] >= spec["bolt_d"] + 4 and 2 <= spec["cb"][1] <= spec["T"] - 2
        return bool(ok)
    if f == "gear":
        ok = spec["bore_d"] + 10 <= spec.get("hub_d", spec["root_d"] - 8) and spec["n"] >= 6
        if spec.get("holes"):
            ok = ok and spec["hole_pcd"] - spec["hole_d"] >= max(spec["bore_d"], spec.get("hub_d", 0)) + 8
            ok = ok and spec["hole_pcd"] + spec["hole_d"] <= spec["root_d"] - 8
            ok = ok and spec["hole_pcd"] * math.sin(math.pi / max(spec["holes"], 2)) >= spec["hole_d"] + 4
        return bool(ok)
    return True


def extent(spec):
    """Bounding box (Lx, Ly, Lz) of the part; the origin is its lower-left-front corner, except round parts and
    shafts, which are centred on their axis."""
    f = spec["fam"]
    if f in ("plate", "block"):
        hmax = max([ft["h"] for ft in spec["feats"] if ft["kind"] == "boss"] + [0])
        return spec["L"], spec["W"], spec["T"] + hmax
    if f == "bracket":
        return spec["L"], spec["W"], spec["H"]
    if f == "shaft":
        return sum(l for _, l in spec["segs"]), max(d for d, _ in spec["segs"]), max(d for d, _ in spec["segs"])
    if f == "flange":
        return spec["OD"], spec["OD"], spec["T"] + spec["hub_h"]
    return spec["tip_d"], spec["tip_d"], spec["T"] + spec.get("hub_h", 0)


def gear_outline(spec):
    n, ra, rr = spec["n"], spec["tip_d"] / 2, spec["root_d"] / 2
    p = 2 * math.pi / n
    pts = []
    for k in range(n):
        th = k * p + spec.get("a0", 0.0)
        for r, a in ((rr, th - 0.30 * p), (ra, th - 0.15 * p), (ra, th + 0.15 * p), (rr, th + 0.30 * p)):
            pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def shaft_profile(spec):
    """(x, r) outline of the shaft above its axis, with grooves and end chamfers."""
    segs, c = spec["segs"], spec.get("chamfer", 0)
    pts, x0 = [(0.0, 0.0)], 0.0
    for i, (d, l) in enumerate(segs):
        r = d / 2
        pts.append((x0 + (c if i == 0 and c else 0), r))
        if i == 0 and c:
            pts.insert(-1, (x0, r - c))
        for g in sorted([g for g in spec.get("grooves", []) if g["seg"] == i], key=lambda g: g["off"]):
            a, b, rg = x0 + g["off"], x0 + g["off"] + g["w"], g["dg"] / 2
            pts += [(a, r), (a, rg), (b, rg), (b, r)]
        if i == len(segs) - 1 and c:
            pts += [(x0 + l - c, r), (x0 + l, r - c)]
        else:
            pts.append((x0 + l, r))
        x0 += l
    pts.append((x0, 0.0))
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def seg_start(spec, i):
    return sum(l for _, l in spec["segs"][:i])


_BUILD_CACHE = {}


def build(spec):
    """The CadQuery solid of a spec (cached per process)."""
    key = json.dumps(spec, sort_keys=True)
    if key in _BUILD_CACHE:
        return _BUILD_CACHE[key]
    cq = _cq()
    f = spec["fam"]
    cuts, adds = [], []
    if f in PRISM:
        if f == "bracket":
            L, W, H, T, r = spec["L"], spec["W"], spec["H"], spec["T"], spec.get("rin", 0)
            ro = (r + T) if (r and spec.get("rout")) else 0
            if spec["shape"] == "L":
                pts, rad = [(0, 0), (L, 0), (L, T), (T, T), (T, H), (0, H)], [ro, 0, 0, r, 0, 0]
            else:
                pts = [(0, 0), (L, 0), (L, H), (L - T, H), (L - T, T), (T, T), (T, H), (0, H)]
                rad = [ro, ro, 0, 0, r, r, 0, 0]
            body = _prism(pts, rad, None, (0, 0, 0), X3, Z3, Y3 * W)
        else:
            pts, rad, ch = outline(spec)
            body = _prism(pts, rad, ch, (0, 0, 0), X3, Y3, Z3 * spec["T"])
        fr = frames(spec)
        for ft in spec["feats"]:
            o, eu, ev, inw, th, _ = fr[ft["face"]]
            c = o + eu * ft["u"] + ev * ft["v"]
            k = ft["kind"]
            if k == "hole":
                dep = ft.get("depth")
                cuts.append(_cyl(ft["d"] / 2, (dep if dep is not None else th + 1) + 1, c - inw, inw))
                if ft.get("cb"):
                    cuts.append(_cyl(ft["cb"][0] / 2, ft["cb"][1] + 1, c - inw, inw))
            elif k == "slot":
                p2, r2 = _obround(ft["l"], ft["w"], ft.get("ang", 0))
                dep = ft.get("depth")
                cuts.append(_prism(p2, r2, None, c - inw, eu, ev, inw * ((dep if dep is not None else th + 1) + 1)))
            elif k == "pocket":
                l, w = ft["l"] / 2, ft["w"] / 2
                cuts.append(_prism([(-l, -w), (l, -w), (l, w), (-l, w)], [ft.get("r", 0)] * 4, None, c - inw, eu, ev,
                                   inw * (ft["depth"] + 1)))
            elif k == "boss":
                adds.append(_cyl(ft["d"] / 2, ft["h"], c, -inw))
                if ft.get("bore"):
                    bd = ft.get("bd")
                    cuts.append(_cyl(ft["bore"] / 2, (bd if bd is not None else th + ft["h"] + 1) + 1, c - inw * (ft["h"] + 1), inw))
    elif f == "shaft":
        face = _poly_face(shaft_profile(spec), None, None, lambda p: (p[0], 0.0, p[1]))
        body = cq.Solid.revolve(face, 360.0, cq.Vector(0, 0, 0), cq.Vector(1, 0, 0))
        k = spec.get("keyway")
        if k:
            d = spec["segs"][k["seg"]][0]
            xc = seg_start(spec, k["seg"]) + k["off"] + k["l"] / 2
            p2, r2 = _obround(k["l"], k["w"], 0)
            cuts.append(_prism(p2, r2, None, (xc, 0, d / 2 - k["t"]), X3, Y3, Z3 * (k["t"] + 2)))
        b = spec.get("bore")
        if b:
            Lt = extent(spec)[0]
            cuts.append(_cyl(b["d"] / 2, b["depth"] + 1, (-1, 0, 0) if b["end"] == 0 else (Lt + 1, 0, 0),
                             X3 if b["end"] == 0 else -X3))
    elif f == "flange":
        T, hh = spec["T"], spec["hub_h"]
        body = _cyl(spec["OD"] / 2, T, (0, 0, 0), Z3)
        if hh:
            adds.append(_cyl(spec["hub_d"] / 2, hh, (0, 0, T), Z3))
        bd = spec.get("bore_depth")
        cuts.append(_cyl(spec["bore_d"] / 2, (bd if bd is not None else T + hh + 1) + 1, (0, 0, T + hh + 1), -Z3))
        for x, y in bolt_xy(spec):
            cuts.append(_cyl(spec["bolt_d"] / 2, T + 2, (x, y, -1), Z3))
            if spec.get("cb"):
                cuts.append(_cyl(spec["cb"][0] / 2, spec["cb"][1] + 1, (x, y, T + 1), -Z3))
    elif f == "gear":
        T = spec["T"]
        body = _prism(gear_outline(spec), None, None, (0, 0, 0), X3, Y3, Z3 * T)
        hh = spec.get("hub_h", 0)
        if hh:
            adds.append(_cyl(spec["hub_d"] / 2, hh, (0, 0, T), Z3))
        cuts.append(_cyl(spec["bore_d"] / 2, T + hh + 2, (0, 0, -1), Z3))
        if spec.get("key"):
            kw, kh = spec["key"]
            cuts.append(cq.Solid.makeBox(kw, spec["bore_d"] / 2 + kh, T + hh + 2, cq.Vector(-kw / 2, 0, -1)))
        for k in range(spec.get("holes", 0)):
            a = 2 * math.pi * k / spec["holes"] + spec.get("h0", 0.0)
            cuts.append(_cyl(spec["hole_d"] / 2, T + 2, (spec["hole_pcd"] / 2 * math.cos(a), spec["hole_pcd"] / 2 * math.sin(a), -1), Z3))
    else:
        raise ValueError(f)
    if adds:
        body = body.fuse(*adds).clean()
    if cuts:
        body = body.cut(*cuts)
    if spec.get("mirror"):
        body = body.mirror("YZ", cq.Vector(mirror_x(spec), 0, 0))
    if len(_BUILD_CACHE) > 64:
        _BUILD_CACHE.clear()
    _BUILD_CACHE[key] = body
    return body


def mirror_x(spec):
    """x of the mirror plane: mid-length for parts with a corner origin, the axis for round parts."""
    return extent(spec)[0] / 2 if spec["fam"] in PRISM + ("shaft",) else 0.0


def bolt_xy(spec):
    return [(spec["pcd"] / 2 * math.cos(spec.get("a0", 0.0) + 2 * math.pi * k / spec["n"]),
             spec["pcd"] / 2 * math.sin(spec.get("a0", 0.0) + 2 * math.pi * k / spec["n"])) for k in range(spec["n"])]


def feat_xyz(spec, ft):
    """3D point of a prismatic feature on its face (mirror flag applied)."""
    o, eu, ev, _, _, _ = frames(spec)[ft["face"]]
    p = o + eu * ft["u"] + ev * ft["v"]
    if spec.get("mirror"):
        p = p * (-1, 1, 1) + (2 * mirror_x(spec), 0, 0)
    return p


def centre_marks(spec, view):
    """Thin centre lines (page mm) for the round features that show as circles in this view, and shaft axes."""
    f, out = spec["fam"], []

    def cross(u, v, r):
        e = r + 2.5
        out.extend([np.array([(u - e, v), (u + e, v)]), np.array([(u, v - e), (u, v + e)])])

    if f in PRISM:
        face_of = {"top": ("top", "bottom", "base"), "right": ("leg0", "leg1"), "front": ()}[view]
        for ft in spec["feats"]:
            if ft["face"] in face_of and ft["kind"] in ("hole", "boss"):
                u, v = project(feat_xyz(spec, ft), view)
                cross(u, v, _half(ft)[0])
    elif f == "shaft" and view in ("front", "top"):
        out.append(np.array([(-5.0, 0.0), (extent(spec)[0] + 5.0, 0.0)]))
    elif f == "shaft":
        cross(0, 0, extent(spec)[1] / 2)
    elif f in ("flange", "gear") and view == "top":
        cross(0, 0, extent(spec)[0] / 2)
        if f == "flange":
            a = np.linspace(0, 2 * math.pi, 90)
            out.append(np.stack([spec["pcd"] / 2 * np.cos(a), spec["pcd"] / 2 * np.sin(a)], 1))
    elif f in ("flange", "gear"):
        out.append(np.array([(0.0, -4.0), (0.0, extent(spec)[2] + 4.0)]))
    return out


def openings(spec):
    """[(feature id, [points around each mouth of the feature])] for the round / slotted features one counts: a
    through feature has two mouths, a blind one only the mouth on its face."""
    out, f = [], spec["fam"]
    ring = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    if f in PRISM:
        fr = frames(spec)
        for i, ft in enumerate(spec["feats"]):
            o, eu, ev, inw, th, _ = fr[ft["face"]]
            hu, hv = _half(ft)
            top = o - inw * (ft["h"] if ft["kind"] == "boss" else 0)
            mouths = [[top + eu * (ft["u"] + (hu + 0.6) * c) + ev * (ft["v"] + (hv + 0.6) * s_) for c, s_ in ring]]
            if is_through(ft):
                mouths.append([q + inw * th for q in mouths[0]])
            out.append((i, mouths))
    elif f in ("flange", "gear"):
        xy = bolt_xy(spec) if f == "flange" else [
            (spec["hole_pcd"] / 2 * math.cos(2 * math.pi * k / spec["holes"] + spec.get("h0", 0.0)),
             spec["hole_pcd"] / 2 * math.sin(2 * math.pi * k / spec["holes"] + spec.get("h0", 0.0))) for k in range(spec.get("holes", 0))]
        r = (spec["bolt_d"] if f == "flange" else spec.get("hole_d", 0)) / 2 + 0.6
        for i, (x, y) in enumerate(xy):
            out.append((i, [[np.array([x + r * c, y + r * s_, z]) for c, s_ in ring] for z in (spec["T"], 0.0)]))
    if spec.get("mirror"):
        mx = mirror_x(spec)
        out = [(i, [[q * (-1, 1, 1) + (2 * mx, 0, 0) for q in m] for m in ms]) for i, ms in out]
    return out


def all_visible(spec, view, frac=0.6):
    """Is every countable feature seen from ``view`` (at least ``frac`` of one mouth's rim not hidden by the part)?"""
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector

    n = basis(view)[2]
    it = IntCurvesFace_ShapeIntersector()
    it.Load(build(spec).wrapped, 1e-6)
    for _, mouths in openings(spec):
        best = 0.0
        for m in mouths:
            seen = 0
            for q in m:
                q = q + n * 0.05
                it.Perform(gp_Lin(gp_Pnt(*map(float, q)), gp_Dir(*map(float, n))), 0.3, 1e4)
                seen += it.NbPnt() == 0
            best = max(best, seen / len(m))
        if best < frac:
            return False
    return True


def mode_view(mode):
    return "iso_hi" if "hi" in mode else ("iso_l" if mode.startswith("iso_l") else "iso")


def part_view(spec, view, hidden=True, centre=False, cut=None):
    """Orthographic (or, with ``cut`` = plane position, section) view of a spec."""
    shape = build(spec)
    v = section_view(shape, view, cut) if cut is not None else ortho(shape, view, hidden)
    if centre:
        v.centre = centre_marks(spec, view)
    return v


# ---- samplers ---------------------------------------------------------------------------------------------------------


def put(rng, spec, face, n, make, distinct_u=False, pattern=None, tries=120):
    """Place ``n`` features (``make(rng)`` -> feature dict without position) on a face. ``pattern`` "row" / "grid" /
    "corners" places copies of one feature regularly; default is random positions on a 1 mm grid."""
    u0, u1, v0, v1 = frames(spec)[face][5]
    if pattern:
        ft = make(rng)
        hu, hv = _half(ft)
        for _ in range(tries):
            mu, mv = rng.randint(6, 16) + hu, rng.randint(6, 14) + hv
            if pattern == "corners" and n == 4:
                pos = [(u0 + mu, v0 + mv), (u1 - mu, v0 + mv), (u1 - mu, v1 - mv), (u0 + mu, v1 - mv)]
            elif pattern == "row":
                v = rng.randint(int(v0 + mv), max(int(v0 + mv), int(v1 - mv)))
                pos = [(u0 + mu + (u1 - u0 - 2 * mu) * k / max(n - 1, 1), v) for k in range(n)]
            else:
                ny = 2 if n % 2 == 0 and n >= 4 else 1
                nx = n // ny
                pos = [(u0 + mu + (u1 - u0 - 2 * mu) * i / max(nx - 1, 1), v0 + mv + (v1 - v0 - 2 * mv) * j / max(ny - 1, 1))
                       for j in range(ny) for i in range(nx)]
                if ny == 1:
                    pos = [(u, (v0 + v1) / 2) for u, _ in pos]
            new = [dict(ft, face=face, u=round(u), v=round(v)) for u, v in pos]
            spec["feats"] += new
            if len(new) == n and feats_ok(spec):
                return True
            del spec["feats"][-len(new):]
        return False
    for _ in range(n):
        ft = make(rng)
        hu, hv = _half(ft)
        for _ in range(tries):
            cand = dict(ft, face=face, u=rng.randint(int(u0 + hu + 4), int(u1 - hu - 4)), v=rng.randint(int(v0 + hv + 4), int(v1 - hv - 4)))
            if distinct_u and any(abs(o["u"] - cand["u"]) < _half(o)[0] + hu + 3 for o in spec["feats"]
                                  if o["face"] in ("top", "bottom")):
                continue
            spec["feats"].append(cand)
            if feats_ok(spec):
                break
            spec["feats"].pop()
        else:
            return False
    return True


def hole(d=None, depth=None, cb=False):
    def make(rng):
        dd = d or rng.choice([5, 6, 8, 10, 12])
        ft = {"kind": "hole", "d": dd}
        if depth is not None:
            ft["depth"] = depth(rng) if callable(depth) else depth
        if cb:
            ft["cb"] = [dd + rng.choice([5, 6, 8]), rng.choice([3, 4, 5])]
        return ft
    return make


def slot(rng):
    w = rng.choice([6, 8, 10])
    return {"kind": "slot", "l": w + rng.choice([10, 14, 18, 24]), "w": w, "ang": rng.choice([0, 0, 90])}


def new_plate(rng, block=False, big=False):
    if block:
        L, W = rng.randrange(80, 135, 5), rng.randrange(55, 95, 5)
        return {"fam": "block", "L": L, "W": min(W, L), "T": rng.randrange(25, 50, 5), "feats": []}
    L = rng.randrange(110 if big else 80, 165, 5)
    return {"fam": "plate", "L": L, "W": rng.randrange(60 if big else 45, min(L, 105) + 1, 5), "T": rng.choice([6, 8, 10, 12, 15, 20]),
            "feats": []}


def new_bracket(rng, shape=None):
    shape = shape or rng.choice(["L", "L", "U"])
    T = rng.choice([5, 6, 8, 10])
    s = {"fam": "bracket", "shape": shape, "L": rng.randrange(70, 125, 5), "W": rng.randrange(40, 85, 5),
         "H": rng.randrange(45, 90, 5), "T": T, "rin": rng.choice([0, 0, 4, 6, 8]), "feats": []}
    s["rout"] = bool(s["rin"] and rng.random() < 0.5)
    return s


def new_shaft(rng, nseg=None, grooves=None, keyway=None):
    for _ in range(200):
        n = nseg or rng.randint(2, 5)
        segs = [[rng.randrange(14, 62, 2), rng.randrange(14, 56, 2)] for _ in range(n)]
        s = {"fam": "shaft", "segs": segs, "grooves": [], "keyway": None, "chamfer": rng.choice([0, 0, 1, 2])}
        if sum(l for _, l in segs) > 190 or not valid(s):
            continue
        ng = rng.choice([0, 0, 1, 2]) if grooves is None else grooves
        ok = True
        for _ in range(ng):
            for _ in range(40):
                i = rng.randrange(n)
                d, l = segs[i]
                g = {"seg": i, "off": rng.randint(4, max(4, l - 8)), "w": rng.choice([2, 3, 4]), "dg": d - rng.choice([4, 6, 8])}
                s["grooves"].append(g)
                if valid(s):
                    break
                s["grooves"].pop()
            else:
                ok = False
        kw = (rng.random() < 0.5) if keyway is None else keyway
        if kw and ok:
            for _ in range(40):
                i = rng.randrange(n)
                d, l = segs[i]
                w = rng.choice([4, 5, 6, 8])
                s["keyway"] = {"seg": i, "off": rng.randint(3, max(3, l - 12)), "l": rng.randrange(10, 36, 2), "w": w, "t": max(2, w // 2)}
                if valid(s):
                    break
                s["keyway"] = None
            else:
                ok = False
        if ok:
            return s
    raise RuntimeError("no shaft")


def new_flange(rng, n=None, hub=None):
    for _ in range(400):
        OD = rng.randrange(80, 165, 5)
        s = {"fam": "flange", "OD": OD, "T": rng.choice([8, 10, 12, 15, 18]), "hub_d": rng.randrange(28, 80, 2),
             "hub_h": (rng.choice([0, 12, 16, 20, 25, 30, 35]) if hub is None else hub), "bore_d": rng.randrange(10, 50, 2),
             "bore_depth": None, "pcd": rng.randrange(50, 150, 2), "n": n or rng.randint(3, 10), "bolt_d": rng.choice([5, 6, 8, 10, 12]),
             "cb": None, "a0": rng.choice([0.0, math.pi / 2, math.pi / 4, math.pi / 6])}
        if valid(s):
            return s
    raise RuntimeError("no flange")


def new_gear(rng, n=None):
    for _ in range(400):
        n_ = n or rng.randint(8, 24)
        tip = rng.randrange(80, 150, 2)
        m = tip / (n_ + 2)
        s = {"fam": "gear", "n": n_, "tip_d": tip, "root_d": round(tip - 4.4 * m, 1), "T": rng.choice([8, 10, 12, 16]),
             "bore_d": rng.randrange(10, 32, 2), "a0": rng.uniform(0, 2 * math.pi / n_)}
        if rng.random() < 0.4:
            s["hub_d"], s["hub_h"] = s["bore_d"] + rng.choice([12, 16, 20]), rng.choice([8, 12, 16])
        if rng.random() < 0.4:
            s["key"] = [rng.choice([4, 5, 6]), rng.choice([2, 3])]
        if rng.random() < 0.45:
            s["holes"], s["hole_d"], s["h0"] = rng.randint(3, 6), rng.choice([8, 10, 12, 14]), rng.uniform(0, 1.0)
            s["hole_pcd"] = round((max(s["bore_d"], s.get("hub_d", 0)) + s["root_d"]) / 2)
        if valid(s) and s["root_d"] - max(s["bore_d"], s.get("hub_d", 0)) >= 16:
            return s
    raise RuntimeError("no gear")


def copy_spec(spec):
    return json.loads(json.dumps(spec))


def mutate(spec, rng, kind):
    """A near-miss part: ``kind`` in mirror / remove / add / move / resize / fillet / dim / count. None if it cannot
    be applied to this part."""
    s = copy_spec(spec)
    f = s["fam"]
    if kind == "mirror":
        s["mirror"] = not s.get("mirror", False)
        return s
    if f in PRISM:
        fs = s["feats"]
        holes = [i for i, ft in enumerate(fs) if ft["kind"] == "hole"]
        if kind in ("remove", "count") and (kind == "remove" or rng.random() < 0.5 or len(holes) > 8):
            pool = holes if kind == "count" else list(range(len(fs)))
            if len(pool) == 0 or len(fs) < 2:
                return None
            fs.pop(rng.choice(pool))
        elif kind in ("add", "count"):
            if not holes:
                return None
            src = fs[rng.choice(holes)]
            u0, u1, v0, v1 = frames(s)[src["face"]][5]
            for _ in range(80):
                fs.append(dict(src, u=rng.randint(int(u0) + 6, int(u1) - 6), v=rng.randint(int(v0) + 6, int(v1) - 6)))
                if feats_ok(s):
                    break
                fs.pop()
            else:
                return None
        elif kind == "move":
            if not fs:
                return None
            ft = fs[rng.randrange(len(fs))]
            u0, u1, v0, v1 = frames(s)[ft["face"]][5]
            ax = rng.choice(["u", "v"])
            span = (u1 - u0) if ax == "u" else (v1 - v0)
            ft[ax] = round(ft[ax] + rng.choice([-1, 1]) * rng.uniform(max(9, 0.18 * span), max(12, 0.5 * span)))
        elif kind == "resize":
            if fs and rng.random() < 0.7:
                ft = fs[rng.randrange(len(fs))]
                fac = rng.choice([0.5, 0.58, 1.6, 1.8])
                key = rng.choice({"hole": ["d"], "boss": ["d", "h"], "slot": ["l"], "pocket": ["l", "w"]}[ft["kind"]])
                new = max(3, round(ft[key] * fac))
                if ft.get("cb") and key == "d":
                    ft["cb"][0] = ft["cb"][0] + new - ft[key]
                if ft["kind"] == "slot" and new < ft["w"] + 6:
                    return None
                ft[key] = new
            else:
                key = rng.choice(["L", "W", "H"] if f == "bracket" else ["L", "W", "T"])
                s[key] = round(s[key] * rng.choice([0.62, 0.68, 1.45, 1.6]))
        elif kind == "dim":
            ds = sorted({ft["d"] for ft in fs if ft["kind"] == "hole"})
            if ds and rng.random() < 0.4:
                d = rng.choice(ds)
                nd = max(4, round(d * rng.choice([0.6, 1.5, 1.7])))
                for ft in fs:
                    if ft["kind"] == "hole" and ft["d"] == d:
                        if ft.get("cb"):
                            ft["cb"][0] += nd - d
                        ft["d"] = nd
            else:
                key = rng.choice(["L", "H", "W"] if f == "bracket" else ["L", "W"])
                s[key] = round(s[key] * rng.choice([0.68, 0.74, 1.32, 1.42]))
        elif kind == "fillet":
            if f == "bracket":
                if not s.get("rin"):
                    return None
                s["rin"], s["rout"] = 0, False
            else:
                key = "outline" if s.get("outline") else None
                rad = s["outline"]["r"] if key else s.get("cr")
                idx = [i for i, r in enumerate(rad or []) if r > 0]
                if not idx:
                    return None
                rad[rng.choice(idx)] = 0
        else:
            return None
        return s if valid(s) else None
    if f == "shaft":
        segs = s["segs"]
        if kind == "remove":
            opts = ["g"] * len(s["grooves"]) + (["k"] if s["keyway"] else [])
            if not opts:
                return None
            if rng.choice(opts) == "g":
                s["grooves"].pop(rng.randrange(len(s["grooves"])))
            else:
                s["keyway"] = None
        elif kind == "add":
            i = rng.randrange(len(segs))
            s["grooves"].append({"seg": i, "off": rng.randint(4, max(4, segs[i][1] - 8)), "w": 3, "dg": segs[i][0] - 6})
        elif kind == "move":
            opts = ["g"] * len(s["grooves"]) + (["k"] if s["keyway"] else [])
            if not opts:
                return None
            ft = rng.choice(s["grooves"]) if rng.choice(opts) == "g" else s["keyway"]
            if rng.random() < 0.5 and len(segs) > 1:
                ft["seg"] = rng.choice([i for i in range(len(segs)) if i != ft["seg"]])
                ft["off"] = rng.randint(4, max(4, segs[ft["seg"]][1] - 8))
                if "dg" in ft:
                    ft["dg"] = segs[ft["seg"]][0] - 6
            else:
                ft["off"] = ft["off"] + rng.choice([-1, 1]) * rng.randint(8, 20)
        elif kind in ("resize", "dim"):
            i = rng.randrange(len(segs))
            j = rng.choice([0, 1])
            old = segs[i][j]
            segs[i][j] = 2 * round(old * rng.choice([0.6, 0.7, 1.35, 1.5]) / 2)
            for g in s["grooves"]:
                if g["seg"] == i and j == 0:
                    g["dg"] += segs[i][0] - old
        else:
            return None
        return s if valid(s) and extent(s)[0] <= 230 else None
    if f == "flange":
        if kind in ("remove", "add", "count"):
            s["n"] += -1 if kind == "remove" or (kind == "count" and rng.random() < 0.5) else 1
        elif kind == "move":
            s["a0"] = s["a0"] + math.pi / s["n"]
        elif kind in ("resize", "dim"):
            key = rng.choice(["bore_d", "hub_d", "pcd", "bolt_d", "OD"] + (["hub_h"] if s["hub_h"] else []))
            s[key] = 2 * round(s[key] * rng.choice([0.6, 0.7, 0.8, 1.25, 1.4, 1.6]) / 2)
        else:
            return None
        return s if valid(s) else None
    return None


# ---- sheets, layout, dimensioning ------------------------------------------------------------------------------------
NAMES = {"plate": ["MOUNTING PLATE", "COVER PLATE", "BASE PLATE", "ADAPTER PLATE", "CLAMP PLATE"],
         "block": ["HOUSING BLOCK", "MANIFOLD BLOCK", "FIXTURE BLOCK", "BEARING BLOCK"],
         "bracket": ["ANGLE BRACKET", "SUPPORT BRACKET", "MOUNTING BRACKET"], "shaft": ["STEPPED SHAFT", "SPINDLE", "DRIVE SHAFT"],
         "flange": ["FLANGE", "HUB FLANGE", "END FLANGE"], "gear": ["TOOTHED WHEEL", "SPROCKET", "INDEX WHEEL"],
         "asm": ["COVER ASSEMBLY", "BEARING ASSEMBLY", "CLAMP ASSEMBLY", "STACK ASSEMBLY"]}
UNIT_TEXT = {"mm": ["mm", "MILLIMETRES", "mm", "MM"], "in": ["INCH", "INCHES", "in", "INCH"]}


class Num:
    """How a drawing prints its numbers: spec millimetres x ``scale``, fixed decimals or shortest form."""

    def __init__(self, scale=1.0, dec=None):
        self.scale, self.dec = scale, dec

    def val(self, v):
        return round(v * self.scale, 4)

    def s(self, v):
        x = self.val(v)
        return ("%.*f" % (self.dec, x)) if self.dec is not None else ("%g" % x)


def title_cells(rng, fam, units="mm", angle="third", extra=None):
    cells = [("TITLE", rng.choice(NAMES[fam])), ("DWG NO", "%s-%04d" % (rng.choice("ABCDKMR"), rng.randrange(1000, 9999))),
             ("UNITS", rng.choice(UNIT_TEXT[units])), ("SCALE", rng.choice(["1:1", "1:2", "2:1", "1:5", "NTS"])),
             ("PROJECTION", "%s ANGLE" % angle.upper()), ("SHEET", "1 / 1")]
    if extra:
        cells[3] = extra
    order = list(range(len(cells)))
    rng.shuffle(order)
    return [cells[i] for i in order]


def new_sheet(rng, w=896, h=672, **over):
    st = sample_style(rng)
    st.update(over)
    sh = Sheet(w, h, st)
    sh.border()
    return sh


def layout(sh, views, arr, area, padx=46, pady=44, smax=4.2, draw=True, **kw):
    """Place named views on a grid (``arr``: rows of names or None, top row first) at one common scale inside
    ``area`` = (x0, y0, x1, y1) px, leaving (padx, pady) around every view for dimensions. Returns name -> Placed."""
    ncol = max(len(r) for r in arr)
    colw = [max([views[r[c]].size[0] for r in arr if c < len(r) and r[c]] + [0]) for c in range(ncol)]
    rowh = [max([views[n].size[1] for n in r if n] + [0]) for r in arr]
    aw, ah = area[2] - area[0], area[3] - area[1]
    s = min(smax, (aw - ncol * 2 * padx) / max(sum(colw), 1e-6), (ah - len(arr) * 2 * pady) / max(sum(rowh), 1e-6))
    cw = [c * s + 2 * padx for c in colw]
    rh = [r * s + 2 * pady for r in rowh]
    x = area[0] + (aw - sum(cw)) / 2
    out = {}
    ytop = area[3] - (ah - sum(rh)) / 2
    for ri, r in enumerate(arr):
        xc = x
        for ci in range(ncol):
            n = r[ci] if ci < len(r) else None
            if n:
                cx, cy = xc + cw[ci] / 2, ytop - rh[ri] / 2
                out[n] = draw_view(sh, views[n], cx, cy, s, **kw) if draw else Placed(views[n], cx, cy, s)
            xc += cw[ci]
        ytop -= rh[ri]
    return out


def arrange(names, angle):
    """Grid of view names for a first- or third-angle layout (plain panels, no projection symbol)."""
    names = set(names)
    f = "front" if "front" in names else None
    t = "top" if "top" in names else None
    r = "right" if "right" in names else None
    if angle == "third":
        rows = [[t, None], [f, r]]
    else:
        rows = [[r, f], [None, t]]
    rows = [row for row in rows if any(row)]
    keep = [c for c in range(2) if any(row[c] for row in rows)]
    return [[row[c] for c in keep] for row in rows]


def dia(st, s, count=None):
    return ("%d× " % count if count else "") + st["dia"] + s


def _reserve_feats(sh, spec, p, faces=("top", "bottom", "base")):
    """Reserve the footprint of the features drawn in a plan view and the view's outline, so leader texts do not
    land on them."""
    for x0, y0, x1, y1 in ((p.x0, p.y0, p.x1, p.y0), (p.x0, p.y1, p.x1, p.y1), (p.x0, p.y0, p.x0, p.y1), (p.x1, p.y0, p.x1, p.y1)):
        sh.boxes.append((x0 - 3, y0 - 3, x1 + 3, y1 + 3))
    for ft in spec["feats"]:
        if ft["face"] in faces:
            cx, cy = p.P(ft["u"], ft["v"])
            hu, hv = _half(ft)
            sh.boxes.append((cx - hu * p.s - 2, cy - hv * p.s - 2, cx + hu * p.s + 2, cy + hv * p.s + 2))


def dim_plate(sh, spec, P, num, rng, chain=None, counts=False):
    """Overall dimensions, a hole-row chain (``chain`` = ordered feature indices) and hole diameters. Returns
    {name: printed value}."""
    out, st = {}, sh.st
    L, W, T = spec["L"], spec["W"], spec["T"]
    top, front = P.get("top"), P.get("front")
    base = front or top
    sh.dim_h(base.P(0, 0)[0], base.P(L, 0)[0], base.y0, base.y0, base.y0 - 28, num.s(L))
    out["L"] = num.val(L)
    if top:
        sh.dim_v(top.P(0, 0)[1], top.P(0, W)[1], top.x1, top.x1, top.x1 + 30, num.s(W))
        out["W"] = num.val(W)
        if chain:
            xs = [0] + [spec["feats"][i]["u"] for i in chain]
            for a, b, i in zip(xs[:-1], xs[1:], [None] + list(chain[:-1])):
                ya = top.y1 if i is None else top.P(0, spec["feats"][i]["v"])[1] + 6
                sh.dim_h(top.P(a, 0)[0], top.P(b, 0)[0], ya, top.P(0, spec["feats"][chain[0]]["v"])[1] + 6, top.y1 + 26,
                         num.s(b - a), tight=True)
            out["offset"] = num.val(xs[1])
            out["pitches"] = [num.val(b - a) for a, b in zip(xs[1:-1], xs[2:])]
        _reserve_feats(sh, spec, top)
        seen = set()
        for ft in spec["feats"]:
            if ft["kind"] == "hole" and ft["d"] not in seen and ft["face"] == "top":
                seen.add(ft["d"])
                n = sum(1 for o in spec["feats"] if o["kind"] == "hole" and o["d"] == ft["d"])
                cx, cy = top.P(ft["u"], ft["v"])
                a = rng.choice([225, 315, 200, 340])
                r = ft["d"] / 2 * top.s
                sh.leader((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))),
                          dia(st, num.s(ft["d"]), n if counts and n > 1 else None), angles=(a, 225, 315, 200, 340, 250, 290, 45, 135))
                out.setdefault("hole_d", []).append(num.val(ft["d"]))
    if front:
        sh.dim_v(front.P(0, 0)[1], front.P(0, T)[1], front.x0, front.x0, front.x0 - 26, num.s(T))
        out["T"] = num.val(T)
    return out


def dim_shaft(sh, spec, P, num, overall=True):
    out, st, f = {}, sh.st, P["front"]
    x = 0
    for d, l in spec["segs"]:
        sh.dim_h(f.P(x, 0)[0], f.P(x + l, 0)[0], f.P(0, -d / 2)[1], f.P(0, -d / 2)[1], f.y0 - 26, num.s(l), tight=True)
        x += l
    out["len"] = [num.val(l) for _, l in spec["segs"]]
    if overall:
        sh.dim_h(f.P(0, 0)[0], f.P(x, 0)[0], f.y0 - 26, f.y0 - 26, f.y0 - 54, num.s(x))
        out["L"] = num.val(x)
    x = 0
    for i, (d, l) in enumerate(spec["segs"]):
        tip = f.P(x + l * (0.3 if i % 2 else 0.6), d / 2)
        ty = f.y1 + 22 + 20 * (i % 2)
        s = dia(st, num.s(d))
        tw, th = sh.measure(s)
        sh.poly([[tip, (tip[0], ty - th / 2 - 2)]], st["thin"], st["dimcol"], z=5)
        sh.fill([tip, (tip[0] - 3, tip[1] + 9), (tip[0] + 3, tip[1] + 9)], st["dimcol"], z=6)
        sh.text(tip[0], ty, s, col=st["dimcol"])
        x += l
    out["dia"] = [num.val(d) for d, _ in spec["segs"]]
    return out


def dim_flange(sh, spec, P, num, rng, counts=True):
    out, st = {}, sh.st
    f, top = P["front"], P.get("top")
    T, hh, H = spec["T"], spec["hub_h"], spec["T"] + spec["hub_h"]
    r = spec["OD"] / 2
    rb = spec["bore_d"] / 2
    sh.dim_h(f.P(-rb, 0)[0], f.P(rb, 0)[0], f.y0, f.y0, f.y0 - 26, dia(st, num.s(spec["bore_d"])))
    out["bore_d"] = num.val(spec["bore_d"])
    sh.dim_h(f.P(-r, 0)[0], f.P(r, 0)[0], f.y0, f.y0, f.y0 - 56, dia(st, num.s(spec["OD"])))
    out["OD"] = num.val(spec["OD"])
    if hh:
        rh = spec["hub_d"] / 2
        sh.dim_h(f.P(-rh, 0)[0], f.P(rh, 0)[0], f.y1, f.y1, f.y1 + 26, dia(st, num.s(spec["hub_d"])))
        out["hub_d"] = num.val(spec["hub_d"])
        sh.dim_v(f.P(0, 0)[1], f.P(0, H)[1], f.x1, f.P(rh, 0)[0], f.x1 + 30, num.s(H))
        out["H"] = num.val(H)
    sh.dim_v(f.P(0, 0)[1], f.P(0, T)[1], f.x0, f.x0, f.x0 - 26, num.s(T))
    out["T"] = num.val(T)
    if top:
        a = spec.get("a0", 0.0) + math.pi / spec["n"]
        rp = spec["pcd"] / 2
        ang = math.degrees(a) % 360
        sh.leader(top.P(rp * math.cos(a), rp * math.sin(a)), "PCD " + num.s(spec["pcd"]),
                  angles=(ang, ang + 25, ang - 25, 45, 135, 315, 225), lengths=(max(30, (r - rp) * top.s + 22), (r - rp) * top.s + 44, 90))
        out["pcd"] = num.val(spec["pcd"])
        x, y = bolt_xy(spec)[0]
        cx, cy = top.P(x, y)
        rr = spec["bolt_d"] / 2 * top.s
        a2 = math.degrees(math.atan2(y, x)) % 360
        sh.leader((cx + rr * math.cos(math.radians(a2)), cy + rr * math.sin(math.radians(a2))),
                  dia(st, num.s(spec["bolt_d"]), spec["n"] if counts else None),
                  angles=(a2, a2 + 30, a2 - 30, a2 + 60, a2 - 60), lengths=((r - rp) * top.s + 20, (r - rp) * top.s + 40, 90))
        out["bolt_d"] = num.val(spec["bolt_d"])
    return out


def dim_bracket(sh, spec, P, num, rng):
    out, st = {}, sh.st
    f, top = P["front"], P.get("top")
    L, W, H, T = spec["L"], spec["W"], spec["H"], spec["T"]
    sh.dim_h(f.P(0, 0)[0], f.P(L, 0)[0], f.y0, f.y0, f.y0 - 28, num.s(L))
    sh.dim_v(f.P(0, 0)[1], f.P(0, H)[1], f.x0, f.x0, f.x0 - 28, num.s(H))
    sh.dim_v(f.P(0, 0)[1], f.P(0, T)[1], f.x1, f.x1, f.x1 + 26, num.s(T))
    out.update(L=num.val(L), H=num.val(H), T=num.val(T))
    if top:
        sh.dim_v(top.P(0, 0)[1], top.P(0, W)[1], top.x1, top.x1, top.x1 + 30, num.s(W))
        out["W"] = num.val(W)
        _reserve_feats(sh, spec, top)
        seen = set()
        for ft in spec["feats"]:
            if ft["face"] == "base" and ft["d"] not in seen:
                seen.add(ft["d"])
                cx, cy = top.P(ft["u"], ft["v"])
                rr = ft["d"] / 2 * top.s
                sh.leader((cx + rr * 0.707, cy + rr * 0.707), dia(st, num.s(ft["d"])), angles=(45, 60, 30, 315, 135))
                out.setdefault("hole_d", []).append(num.val(ft["d"]))
    return out


def part_sheet(rng, spec, mode, dims=False, title=None, hidden=True, size=(896, 672), st_over=None, note=None):
    """One part on a sheet. ``mode``: "+"-joined orthographic views ("top", "top+front", "front+top+right", ...) or
    "iso_line" / "iso_shaded" / "iso_hi_shaded" ... Returns (sheet, {view: Placed}, printed dims)."""
    sh = new_sheet(rng, *size, **(st_over or {}))
    st = sh.st
    title = (rng.random() < 0.7) if title is None else title
    angle = rng.choice(["third", "third", "first"])
    area = (20, 58 if title else 20, size[0] - 20, size[1] - 20)
    P, printed = {}, {}
    if mode.startswith("iso"):
        vname = "iso_hi" if "hi" in mode else ("iso_l" if mode.startswith("iso_l_") else "iso")
        v = ortho(build(spec), vname, hidden=False)
        s = fit([v], area[2] - area[0] - 120, area[3] - area[1] - 100, smax=5.0)
        cx, cy = (area[0] + area[2]) / 2, (area[1] + area[3]) / 2
        if mode.endswith("shaded"):
            P["iso"] = draw_shaded(sh, [(build(spec), st["shade_rgb"])], vname, cx, cy, s)[0]
        else:
            P["iso"] = draw_view(sh, v, cx, cy, s, smooth=True)
    else:
        names = mode.split("+")
        views = {n: part_view(spec, n, hidden=hidden, centre=st["centre"] or (dims and spec["fam"] == "flange")) for n in names}
        P = layout(sh, views, arrange(names, angle), area, padx=70 if dims else 40, pady=62 if dims else 36, hidden=hidden)
        if dims:
            num = Num()
            f = spec["fam"]
            if f in ("plate", "block"):
                printed = dim_plate(sh, spec, P, num, rng)
            elif f == "shaft" and "front" in P:
                printed = dim_shaft(sh, spec, P, num)
            elif f == "flange" and "front" in P:
                printed = dim_flange(sh, spec, P, num, rng, counts=False)
            elif f == "bracket" and "front" in P:
                printed = dim_bracket(sh, spec, P, num, rng)
        if len(names) > 1 and rng.random() < 0.5:
            for n in names:
                sh.text((P[n].x0 + P[n].x1) / 2, P[n].y1 + (12 if not dims else -1e4), VIEW_TITLE[n], size=10, reserve=False)
    if note:
        sh.text(28, size[1] - 26, note, size=11, ha="left", reserve=False)
    if title:
        sh.title_block(title_cells(rng, spec["fam"], angle=angle))
    return sh, P, printed


# ---- questions -------------------------------------------------------------------------------------------------------


def _num_choice(ins, gold, mistakes, fmt, rng, near, k=4, lo=None, hi=None, r=None):
    """Numeric options with the gold's RANK uniform (the convention of ``visionjev_gen._num_choice``: named mistakes
    first on each side, then near-misses ``gold + d``). Stricter than that helper: the rank is drawn from all ``k``
    ranks first and the item is dropped (None) when it cannot be filled. Choosing among the feasible ranks only makes
    a small gold (a count of 0 or 1) sit at the bottom of its option set almost always, which a text-only model
    learns. Callers that can resample the part pass a rank ``r`` drawn before sampling it, so the rank is uniform
    even marginally (a part too small for the rank is simply drawn again)."""
    g = fmt(gold)
    below, above, seen = [], [], {g}
    pool = [(m, 0) for m in mistakes] + [(gold + d, 1) for d in near]
    rng.shuffle(pool)
    pool.sort(key=lambda t: t[1])
    for v, _ in pool:
        if (lo is not None and v < lo) or (hi is not None and v > hi) or v == gold:
            continue
        f = fmt(v)
        if f in seen:
            continue
        seen.add(f)
        (below if v < gold else above).append(f)
    r = rng.randrange(k) if r is None else r
    if r > len(below) or k - 1 - r > len(above):
        return None
    return _mc(ins, [g] + below[:r] + above[:k - 1 - r], 0, rng)


def _count_q(ins, gold, mistakes, rng, hi=None, r=None):
    return _num_choice(ins, gold, [m for m in mistakes if m >= 0], lambda v: "%d" % int(v), rng, [-3, -2, -1, 1, 2, 3], lo=0, hi=hi, r=r)


def _mc(ins, opts, gold, rng, shuffle=True):
    order = list(range(len(opts)))
    if shuffle:
        rng.shuffle(order)
    return {"type": "choice", "instructions": ins, "criteria": [opts[i] for i in order]}, order.index(gold)


def _noul(ins):
    return {"type": "noul", "instructions": ins, "criteria": None}


def holes_of(spec, **match):
    return [ft for ft in spec["feats"] if ft["kind"] == "hole" and all(ft.get(k) == v for k, v in match.items())]


def is_through(ft):
    return ft["kind"] in ("hole", "slot") and ft.get("depth") is None


# ---- 1. cad_feature_count --------------------------------------------------------------------------------------------


def _fc_holes(rng):
    if rng.random() < 0.65:
        n = rng.randint(1, 10)
        spec = new_plate(rng, big=n > 5)
        d = rng.choice([6, 8, 10])
        pat = rng.choice([None, None, "row", "grid"] + (["corners"] if n == 4 else []))
        if not put(rng, spec, "top", n, hole(d if pat else None, cb=False), pattern=pat if n > 1 else None):
            return None
        ns = rng.choice([0, 0, 1, 2])
        if ns and not put(rng, spec, "top", ns, slot):
            return None
        mode = rng.choice(["top", "top", "top+front", "iso_shaded", "iso_line", "iso_hi_shaded"])
        ins = rng.choice(["How many round holes does this plate have?", "Count the round holes in the plate (slots are not holes)."])
        why = "The plate has %d round hole%s%s." % (n, "" if n == 1 else "s", (" and %d slot%s, which %s not counted" % (
            ns, "" if ns == 1 else "s", "is" if ns == 1 else "are")) if ns else "")
        return spec, mode, ins, n, [n + ns] if ns else [], why
    spec = new_bracket(rng, "L")
    nb, nl = rng.randint(1, 4), rng.randint(0, 3)
    if not (put(rng, spec, "base", nb, hole()) and put(rng, spec, "leg0", nl, hole())):
        return None
    mode = rng.choice(["front+top+right", "iso_shaded", "iso_line"])
    if rng.random() < 0.5:
        return (spec, mode, "How many holes does this bracket have in total?", nb + nl, [nb, nl],
                "There are %d hole(s) in the base and %d in the upright leg: %d in total." % (nb, nl, nb + nl))
    return (spec, mode, "How many holes are in the upright (vertical) leg of the bracket?", nl, [nb + nl, nb],
            "The upright leg has %d hole(s); the other %d are in the base." % (nl, nb))


def _fc_through(rng):
    t = rng.randint(0, 7)
    k = rng.randint(max(t, 2), 7)
    spec = new_plate(rng, block=rng.random() < 0.5, big=True)
    T = spec["T"]
    kinds = [None] * t + ["blind"] * (k - t)
    rng.shuffle(kinds)
    for kd in kinds:
        dep = None if kd is None else rng.randint(max(3, int(T * 0.3)), max(3, int(T * 0.65)))
        if not put(rng, spec, "top", 1, hole(depth=dep) if dep else hole(), distinct_u=True):
            return None
    if rng.random() < 0.5:
        ins, gold, mist = "How many of the holes go all the way through the part?", t, [k, k - t]
    else:
        ins, gold, mist = "How many of the holes are blind (do not go all the way through)?", k - t, [k, t]
    why = "The front view's hidden lines show %d of the %d holes running the full thickness; %d stop short (blind)." % (t, k, k - t)
    return spec, "top+front", ins, gold, mist, why


def _fc_cbore(rng):
    c = rng.randint(0, 8)
    k = rng.randint(max(c, 2), 8)
    spec = new_plate(rng, big=True)
    d = rng.choice([5, 6, 8])
    if not (put(rng, spec, "top", c, hole(d, cb=True)) and put(rng, spec, "top", k - c, hole(d))):
        return None
    rng.shuffle(spec["feats"])
    mode = rng.choice(["top", "top", "iso_shaded", "iso_hi_shaded", "top+front"])
    return (spec, mode, "How many of the holes are counterbored?", c, [k, k - c],
            "%d of the %d holes show the second, larger concentric circle of a counterbore; %d are plain." % (c, k, k - c))


def _fc_slots(rng):
    ns, k = rng.randint(0, 5), rng.randint(0, 5)
    if ns + k == 0:
        return None
    spec = new_plate(rng, big=True)
    if not (put(rng, spec, "top", ns, slot) and put(rng, spec, "top", k, hole())):
        return None
    rng.shuffle(spec["feats"])
    mode = rng.choice(["top", "top", "iso_shaded", "iso_line"])
    return (spec, mode, rng.choice(["How many slots (elongated openings) does the plate have?", "Count the slots in this plate."]),
            ns, [k, ns + k], "There are %d slot(s) and %d round hole(s); only the elongated openings are slots." % (ns, k))


def _fc_bolt(rng):
    n = rng.randint(3, 12)
    try:
        spec = new_flange(rng, n=n)
    except RuntimeError:
        return None
    mode = rng.choice(["top", "top", "top+front", "iso_hi_shaded", "iso_shaded"])
    return (spec, mode, rng.choice(["How many bolt holes are on the bolt circle of this flange?",
                                    "How many bolt holes does the flange have (not counting the centre bore)?"]), n, [n + 1],
            "%d equally spaced bolt holes lie on the bolt circle; the centre bore is not one of them." % n)


OUTLINES = {"rect": lambda L, W, a, b: [(0, 0), (L, 0), (L, W), (0, W)],
            "L": lambda L, W, a, b: [(0, 0), (L, 0), (L, b), (a, b), (a, W), (0, W)],
            "T": lambda L, W, a, b: [(a, 0), (L - a, 0), (L - a, b), (L, b), (L, W), (0, W), (0, b), (a, b)],
            "U": lambda L, W, a, b: [(0, 0), (L, 0), (L, W), (L - a, W), (L - a, b), (a, b), (a, W), (0, W)]}


def _fc_fillet(rng):
    spec = new_plate(rng, big=True)
    L, W = spec["L"], spec["W"]
    shape = rng.choice(list(OUTLINES))
    a = round(L * (0.3 if shape in ("T", "U") else 0.45))
    b = round(W * 0.5)
    pts = OUTLINES[shape](L, W, a, b)
    nr = rng.randint(0, len(pts))
    state = ["r"] * nr + [rng.choice(["s", "s", "c"]) for _ in range(len(pts) - nr)]
    rng.shuffle(state)
    r = rng.choice([7, 8, 10])
    spec["outline"] = {"pts": pts, "r": [r if s == "r" else 0 for s in state], "c": [r if s == "c" else 0 for s in state]}
    nr, nc = state.count("r"), state.count("c")
    if rng.random() < 0.5:
        put(rng, spec, "top", rng.randint(1, 3), hole())
    mode = rng.choice(["top", "top", "iso_hi_shaded", "iso_hi_line"])
    why = "Of the %d outline corners, %d are rounded, %d chamfered (cut straight) and %d sharp." % (len(pts), nr, nc, len(pts) - nr - nc)
    return (spec, mode, "How many corners of the plate outline are rounded (filleted)?", nr, [nr + nc, len(pts) - nr, nc], why)


def _fc_teeth(rng):
    n = rng.randint(8, 24)
    try:
        spec = new_gear(rng, n=n)
    except RuntimeError:
        return None
    mode = rng.choice(["top", "top", "iso_hi_shaded", "iso_hi_line"])
    if spec.get("holes") and rng.random() < 0.4:
        k = spec["holes"]
        return (spec, mode, "How many lightening holes does the wheel have (not counting the centre bore)?", k, [k + 1],
                "%d holes are spaced around the web; the centre bore is separate." % k)
    return spec, mode, rng.choice(["How many teeth does this wheel have?", "Count the teeth on the wheel."]), n, [], "The wheel has %d teeth." % n


def _fc_shaft(rng):
    ng, n = rng.randint(0, 4), rng.randint(2, 5)
    try:
        spec = new_shaft(rng, nseg=n, grooves=ng, keyway=rng.random() < 0.3)
    except RuntimeError:
        return None
    mode = rng.choice(["front", "front", "front+right", "iso_shaded"])
    if rng.random() < 0.5:
        return (spec, mode, "How many grooves are cut into the shaft?", ng, [n, n - 1],
                "The shaft has %d narrow groove(s) and %d cylindrical sections." % (ng, n))
    return (spec, mode, "How many cylindrical sections of different diameter make up the shaft (ignore grooves)?", n, [n + ng, n - 1],
            "The shaft steps through %d sections; the %d groove(s) are not sections." % (n, ng))


def _fc_hidden(rng):
    a, b = rng.randint(0, 5), rng.randint(0, 5)
    if a + b == 0:
        return None
    spec = new_plate(rng, block=True)
    T = spec["T"]
    dep = lambda r: r.randint(5, int(T * 0.45))  # noqa: E731
    if not (put(rng, spec, "top", a, hole(depth=dep)) and put(rng, spec, "bottom", b, hole(depth=dep))):
        return None
    rng.shuffle(spec["feats"])
    if rng.random() < 0.55:
        ins, gold, mist = "How many holes are drilled into the top face of the block?", a, [a + b, b]
    else:
        ins, gold, mist = "How many holes are drilled from the underside (hidden in the top view)?", b, [a + b, a]
    why = "The top view shows %d hole(s) with visible outlines (top face) and %d with dashed hidden outlines (underside)." % (a, b)
    return spec, rng.choice(["top", "top+front"]), ins, gold, mist, why


def _fc_boss(rng):
    nb, k = rng.randint(0, 4), rng.randint(0, 4)
    if nb + k == 0:
        return None
    spec = new_plate(rng, block=True)
    mk = lambda r: {"kind": "boss", "d": r.choice([14, 16, 20]), "h": r.choice([8, 10, 14]), "bore": r.choice([0, 0, 6])}  # noqa: E731
    if not (put(rng, spec, "top", nb, mk) and put(rng, spec, "top", k, hole(depth=10))):
        return None
    rng.shuffle(spec["feats"])
    return (spec, rng.choice(["iso_shaded", "iso_hi_shaded", "top+front", "iso_line"]),
            "How many raised bosses stand on the top face of the block?", nb, [nb + k, k],
            "%d cylindrical boss(es) rise above the top face; the other %d round feature(s) are holes." % (nb, k))


FC_KINDS = [("holes", _fc_holes, 3), ("through_holes", _fc_through, 2), ("counterbores", _fc_cbore, 1.5), ("slots", _fc_slots, 1.5),
            ("bolt_holes", _fc_bolt, 1.5), ("rounded_corners", _fc_fillet, 1.5), ("teeth", _fc_teeth, 1.5), ("shaft", _fc_shaft, 2),
            ("hidden_holes", _fc_hidden, 1.5), ("bosses", _fc_boss, 1.5), ("fasteners", None, 2)]


def _pick(rng, kinds):
    return rng.choices(kinds, [k[2] for k in kinds])[0]


def make_feature_count(rng, render=True):
    r = rng.randrange(4)  # the gold's rank among the options, fixed before the part is drawn (see _num_choice)
    for _ in range(400):
        kind, fn, _ = _pick(rng, FC_KINDS)
        if fn is None:
            out = _asm_fasteners(rng, render, r=r)
            if out is None:
                continue
            return dict(out, kind="fasteners")
        item = fn(rng)
        if item is None:
            continue
        spec, mode, ins, gold, mistakes, why = item
        q = _count_q(ins, gold, mistakes, rng, r=r)
        if q is None:
            continue
        if mode.startswith("iso") and not all_visible(spec, mode_view(mode)):
            continue  # a boss, hub or leg hides one of the features to count
        dims = (not mode.startswith("iso")) and spec["fam"] in ("plate", "shaft", "flange") and mode != "top" and rng.random() < 0.35
        if dims and spec["fam"] in ("shaft", "flange") and "front" not in mode:
            dims = False
        srng = random.Random(rng.random())
        image = (lambda: part_sheet(srng, spec, mode, dims=dims)[0].image()) if render else None
        return {"q": q[0], "label": q[1], "why": why, "kind": kind, "image": image,
                "meta": {"spec": spec, "gold": gold, "mode": mode, "ins": ins}}
    raise RuntimeError("feature_count")


# ---- assemblies ------------------------------------------------------------------------------------------------------
PART_RGB = [(0.62, 0.74, 0.86), (0.9, 0.78, 0.55), (0.72, 0.84, 0.7), (0.86, 0.68, 0.66), (0.78, 0.72, 0.88), (0.8, 0.8, 0.8),
            (0.6, 0.82, 0.82)]


def _hex(af, h, p):
    R = af / math.sqrt(3)
    return _prism([(R * math.cos(math.radians(60 * k)), R * math.sin(math.radians(60 * k))) for k in range(6)], None, None, p, X3, Y3,
                  Z3 * h)


def _fuse(a, *more):
    return a.fuse(*more).clean() if more else a


def new_bolted(rng, n_holes=None, n_bolts=None):
    """Base + optional gasket + cover plate, bolts (with optional washers) in some of the holes."""
    d = rng.choice([6, 8, 10])
    rnd = rng.random() < 0.4
    if rnd:
        OD = rng.randrange(100, 150, 10)
        nh = n_holes or rng.randint(3, 8)
        r = OD / 2 - d - 5
        a0 = rng.choice([0.0, math.pi / nh, math.pi / 2])
        holes = [[round(r * math.cos(a0 + 2 * math.pi * k / nh), 2), round(r * math.sin(a0 + 2 * math.pi * k / nh), 2)] for k in range(nh)]
        if 2 * r * math.sin(math.pi / nh) < 2.8 * d + 6:
            return None
        s = {"type": "bolted", "round": True, "OD": OD}
    else:
        L, W = rng.randrange(100, 160, 10), rng.randrange(70, 110, 10)
        m = d + 6
        nh = n_holes or rng.choice([2, 3, 4, 4, 5, 6, 6, 8])
        xs3 = [m, L / 2, L - m]
        pos = {2: [(m, W / 2), (L - m, W / 2)], 3: [(m, m), (L - m, m), (L / 2, W - m)], 4: [(m, m), (L - m, m), (L - m, W - m), (m, W - m)],
               5: [(m, m), (L - m, m), (L - m, W - m), (m, W - m), (L / 2, W / 2)], 6: [(x, y) for y in (m, W - m) for x in xs3],
               7: [(x, y) for y in (m, W - m) for x in xs3] + [(L / 2, W / 2)],
               8: [(x, y) for y in (m, W - m) for x in xs3] + [(m, W / 2), (L - m, W / 2)]}
        if nh not in pos:
            return None
        holes = [[x - L / 2, y - W / 2] for x, y in pos[nh]]
        s = {"type": "bolted", "round": False, "L": L, "W": W}
    nb = nh if n_bolts is None else n_bolts
    if not 1 <= nb <= nh:
        return None
    s.update(d=d, holes=holes, bolts=sorted(rng.sample(range(nh), nb)), Hb=rng.choice([20, 25, 30]), Tc=rng.choice([6, 8, 10]),
             tg=rng.choice([0, 0, 2, 3]), washers=rng.random() < 0.5, cavity=rng.random() < 0.6)
    return s


def bolted_parts(s, explode=0.0):
    """[(name, solid)] in stack order; ``explode`` lifts each level by that many mm more than the one below."""
    cq = _cq()
    d, Hb, Tc, tg = s["d"], s["Hb"], s["Tc"], s["tg"]

    def slab(z, t, hole_d, depth=None, cavity=False):
        if s["round"]:
            b = _cyl(s["OD"] / 2, t, (0, 0, z), Z3)
        else:
            b = cq.Solid.makeBox(s["L"], s["W"], t, cq.Vector(-s["L"] / 2, -s["W"] / 2, z))
        cuts = [_cyl(hole_d / 2, (depth or t + 2), (x, y, z + t + (0 if depth else 1)), -Z3) for x, y in s["holes"]]
        if cavity:
            cr = (s["OD"] / 2 - 2.6 * d - 8) if s["round"] else (min(s["L"], s["W"]) / 2 - 2.6 * d - 6)
            if cr >= 8 and not any(math.hypot(x, y) < cr + d for x, y in s["holes"]):
                cuts.append(_cyl(cr, (t - 6) if depth else t + 2, (0, 0, z + t + (0 if depth else 1)), -Z3))
        return b.cut(*cuts)

    e = explode
    parts = [("BASE", slab(0, Hb, d, depth=1.6 * d + 2, cavity=s["cavity"]))]
    z = Hb
    if tg:
        parts.append(("GASKET", slab(z + e * len(parts), tg, d + 1.5, cavity=s["cavity"])))
        z += tg
    parts.append(("COVER PLATE", slab(z + e * len(parts), Tc, d + 1.5)))
    z += Tc
    tw = 3 if s["washers"] else 0
    if tw:
        lvl = e * len(parts)
        ws = [_cyl(1.25 * d, tw, (x, y, z + lvl), Z3).cut(_cyl(d / 2 + 0.4, tw + 2, (x, y, z + lvl - 1), Z3))
              for k, (x, y) in enumerate(s["holes"]) if k in s["bolts"]]
        parts.append(("WASHER", cq.Compound.makeCompound(ws)))
    lvl = e * len(parts)
    bs = []
    for k, (x, y) in enumerate(s["holes"]):
        if k in s["bolts"]:
            bot = Hb - 1.3 * d
            bs.append(_fuse(_cyl(d / 2, z + tw - bot, (x, y, bot + lvl), Z3), _hex(1.7 * d, 0.7 * d, (x, y, z + tw + lvl))))
    parts.append(("HEX BOLT", cq.Compound.makeCompound(bs)))
    return parts


def new_bushing(rng):
    ds = rng.choice([16, 20, 24])
    Db = ds + 2 * rng.choice([4, 5, 6])
    return {"type": "bushing", "ds": ds, "Db": Db, "Lh": rng.randrange(36, 64, 4), "Wh": Db + rng.choice([24, 30, 36]),
            "Hh": Db + rng.choice([24, 30, 36]), "tb": rng.choice([8, 10, 12]), "f": rng.choice([4, 5, 6]), "Df": Db + rng.choice([12, 16]),
            "sl": rng.choice([22, 28, 34]), "sr": rng.choice([44, 52, 60]), "gap": rng.choice([6, 8, 10]), "tc": rng.choice([8, 10]),
            "Dc": ds + rng.choice([14, 18])}


def bushing_parts(s, explode=0.0):
    cq = _cq()
    ds, Db, Lh, Wh, Hh = s["ds"], s["Db"], s["Lh"], s["Wh"], s["Hh"]
    zc = Hh / 2
    ax = lambda x: (x, 0, zc)  # noqa: E731
    base = cq.Solid.makeBox(Lh + 30, Wh + 30, s["tb"], cq.Vector(-15, -Wh / 2 - 15, -s["tb"] - 1.2 * explode))
    housing = cq.Solid.makeBox(Lh, Wh, Hh, cq.Vector(0, -Wh / 2, 0)).cut(_cyl(Db / 2, Lh + 2, ax(-1), X3))
    e = explode
    bush = _fuse(_cyl(Db / 2, Lh, ax(0 - 2.2 * e), X3), _cyl(s["Df"] / 2, s["f"], ax(-s["f"] - 2.2 * e), X3))
    bush = bush.cut(_cyl(ds / 2, Lh + s["f"] + 2, ax(-s["f"] - 1 - 2.2 * e), X3))
    shaft = _cyl(ds / 2, s["f"] + s["sl"] + Lh + s["sr"], ax(-s["f"] - s["sl"] - 5.5 * e), X3)
    xc = Lh + s["gap"] + 1.6 * e
    collar = _cyl(s["Dc"] / 2, s["tc"], ax(xc), X3).cut(_cyl(ds / 2, s["tc"] + 2, ax(xc - 1), X3))
    return [("BASE PLATE", base), ("HOUSING", housing), ("FLANGED BUSHING", bush), ("SHAFT", shaft), ("COLLAR", collar)]


STACK_SHAPES = ["SPACER", "GASKET", "CLAMP PLATE", "WASHER", "SLEEVE", "DISC", "HEX NUT", "CAP RING"]


def new_stack(rng, n=None):
    n = n or rng.randint(4, 6)
    d = rng.choice([10, 12, 16])
    names = rng.sample(STACK_SHAPES[:6], n - 2) + ["HEX NUT"]
    parts = []
    for nm in names:
        t = {"SPACER": rng.choice([10, 14, 18]), "GASKET": 3, "CLAMP PLATE": rng.choice([6, 8]), "WASHER": 3, "SLEEVE": rng.choice([16, 22]),
             "DISC": rng.choice([5, 8]), "HEX NUT": round(0.9 * d), "CAP RING": 8}[nm]
        size = {"SPACER": rng.choice([26, 32]), "GASKET": rng.choice([50, 60]), "CLAMP PLATE": rng.choice([56, 66]), "WASHER": rng.choice([30, 36]),
                "SLEEVE": d + rng.choice([8, 10]), "DISC": rng.choice([44, 54]), "HEX NUT": round(1.8 * d), "CAP RING": rng.choice([40, 48])}[nm]
        parts.append({"name": nm, "t": t, "size": size})
    return {"type": "stack", "d": d, "base": rng.choice([70, 80, 90]), "tb": rng.choice([8, 10]), "round_base": rng.random() < 0.5,
            "parts": parts}


def stack_parts(s, explode=0.0):
    cq = _cq()
    d, B, tb = s["d"], s["base"], s["tb"]
    total = sum(p["t"] for p in s["parts"])
    plate = _cyl(B / 2, tb, (0, 0, 0), Z3) if s["round_base"] else cq.Solid.makeBox(B, B, tb, cq.Vector(-B / 2, -B / 2, 0))
    out = [("BASE WITH STUD", _fuse(plate, _cyl(d / 2, total + 6 + tb, (0, 0, 0), Z3)))]
    z = tb
    for k, p in enumerate(s["parts"]):
        zz = z + explode * (k + 1) + (total + 10 if explode else 0)
        nm, t, sz = p["name"], p["t"], p["size"]
        if nm == "CLAMP PLATE":
            b = cq.Solid.makeBox(sz, sz * 0.7, t, cq.Vector(-sz / 2, -sz * 0.35, zz))
        elif nm == "HEX NUT":
            b = _hex(sz, t, (0, 0, zz))
        elif nm == "CAP RING":
            b = _fuse(_cyl(sz / 2, t * 0.5, (0, 0, zz), Z3), _cyl(sz / 2 - 8, t * 0.5, (0, 0, zz + t * 0.5), Z3))
        else:
            b = _cyl(sz / 2, t, (0, 0, zz), Z3)
        out.append((nm, b.cut(_cyl(d / 2 + (0.0 if nm == "HEX NUT" else 0.5), t + 2, (0, 0, zz - 1), Z3))))
        z += t
    return out


def asm_parts(s, explode=0.0):
    return {"bolted": bolted_parts, "bushing": bushing_parts, "stack": stack_parts}[s["type"]](s, explode)


def face_idb(view, p, pad=2):
    """Part-index buffer of a section view's cut faces, aligned like ``shade``'s (for balloon anchors)."""
    from PIL import Image, ImageDraw

    w, h = int(p.x1 - p.x0) + 2 * pad, int(p.y1 - p.y0) + 2 * pad
    im = Image.new("I", (w, h), 0)
    dr = ImageDraw.Draw(im)

    def area(poly):
        x, y = poly[:, 0], poly[:, 1]
        return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2

    for outer, inner, k in sorted(view.faces, key=lambda f: -area(f[0])):
        to = lambda q: [(float(x - p.x0 + pad), float(h - (y - p.y0 + pad))) for x, y in p.T(q)]  # noqa: E731
        dr.polygon(to(outer), fill=k + 1)
        for i in inner:
            dr.polygon(to(i), fill=0)
    return np.asarray(im, np.int32) - 1, (p.x0 - pad, p.y0 - pad)


def parts_table(sh, rows, x0, y1, cw=(44, 150), rh=19):
    """ITEM / DESCRIPTION list with its top-left corner at (x0, y1)."""
    st = sh.st
    H = rh * (len(rows) + 1)
    sh.fill([(x0, y1 - H), (x0 + sum(cw), y1 - H), (x0 + sum(cw), y1), (x0, y1)], st["bg"], z=5.5, edge=st["ink"], lw=st["lw"] * 0.8)
    for r, (a, b) in enumerate([("ITEM", "DESCRIPTION")] + rows):
        y = y1 - rh * (r + 1)
        sh.poly([[(x0, y), (x0 + sum(cw), y)]], st["thin"], z=5.6)
        sh.text(x0 + cw[0] / 2, y + rh / 2, str(a), size=10 if r else 8.5, z=5.8, reserve=False, weight="bold" if r else "normal")
        sh.text(x0 + cw[0] + 6, y + rh / 2, b, size=10 if r else 8.5, ha="left", z=5.8, reserve=False)
    sh.poly([[(x0 + cw[0], y1), (x0 + cw[0], y1 - H)]], st["thin"], z=5.6)
    sh.boxes.append((x0 - 2, y1 - H - 2, x0 + sum(cw) + 2, y1 + 2))


def place_balloons(sh, anchors, p, centre=None):
    """Balloons for {item: anchor px}: each leaves the drawing's bbox in the direction away from its centre."""
    cx, cy = centre or ((p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2)
    sh.boxes.append((p.x0 + 6, p.y0 + 6, p.x1 - 6, p.y1 - 6))  # keep balloons off the drawing itself
    for item, tip in sorted(anchors.items(), key=lambda kv: -abs(kv[1][0] - cx)):
        base = math.degrees(math.atan2(tip[1] - cy, tip[0] - cx)) if abs(tip[0] - cx) + abs(tip[1] - cy) > 2 else 90.0
        angs = [base + d for d in (0, 20, -20, 40, -40, 65, -65, 90, -90, 130, -130, 180)]
        best = None
        for a in angs:
            dx, dy = math.cos(math.radians(a)), math.sin(math.radians(a))
            ts = []
            if dx > 1e-6:
                ts.append((p.x1 - tip[0]) / dx)
            if dx < -1e-6:
                ts.append((p.x0 - tip[0]) / dx)
            if dy > 1e-6:
                ts.append((p.y1 - tip[1]) / dy)
            if dy < -1e-6:
                ts.append((p.y0 - tip[1]) / dy)
            L = max(min(ts), 0) + 22
            c = (tip[0] + L * dx, tip[1] + L * dy)
            if sh.free((c[0] - 14, c[1] - 14, c[0] + 14, c[1] + 14), pad=3) and (best is None or L < best[0] - 25):
                best = (L, a)
        L, a = best or (60, base)
        sh.balloon(tip, str(item), angles=(a,), lengths=(L,))


def asm_sheet(rng, s, numbering, mode, explode=0.0, table=None, balloons=True, size=(896, 672), cut=None, view=None, title=None,
              centreline=False, skip=(), need_all=False):
    """Draw an assembly. ``mode``: "shaded" | "line" | "section"; ``numbering``: part index -> item number."""
    parts = asm_parts(s, explode)
    cq = _cq()
    sh = new_sheet(rng, *size)
    st = sh.st
    table = (rng.random() < 0.6) if table is None else table
    title = (rng.random() < 0.6) if title is None else title
    tw = 200 if (table and balloons) else 0
    area = (24 + tw, 60 if title else 24, size[0] - 24, size[1] - 24)
    cx, cy = (area[0] + area[2]) / 2, (area[1] + area[3]) / 2
    aw, ah = area[2] - area[0] - 150, area[3] - area[1] - 130
    comp = cq.Compound.makeCompound([p[1] for p in parts])
    view = view or "iso"
    cols = list(PART_RGB)
    rng.shuffle(cols)
    solids = [(p[1], cols[k % len(cols)]) for k, p in enumerate(parts)]
    if mode == "section":
        v = section_view([p[1] for p in parts], view, cut)
        sc = fit([v], aw, ah, smax=5.0)
        p = draw_view(sh, v, cx, cy, sc)
        idb, org = face_idb(v, p)
        sh.text(cx, p.y0 - 22 - (0 if not balloons else 30), "SECTION %s–%s" % ("A", "A"), size=12, weight="bold")
    else:
        v = ortho(comp, view, hidden=False)
        sc = fit([v], aw, ah, smax=5.0)
        if mode == "shaded":
            p, idb, org = draw_shaded(sh, solids, view, cx, cy, sc)
        else:
            p = draw_view(sh, v, cx, cy, sc, hidden=False)
            box = (v.b[0] - 1, v.b[1] - 1, v.b[2] + 1, v.b[3] + 1)
            idb = shade(solids, view, sc, box, st["bg"], ss=1)[1]
            org = p.P(box[0], box[1])
    if centreline:
        a, b = project([(0, 0, -6), (0, 0, max(float(q[1].BoundingBox().zmax) for q in parts) + 8)], view)
        sh.poly([[p.P(*a), p.P(*b)]], st["thin"], col=st["dimcol"], dash=(12, 3, 2, 3), z=3.5)
    if balloons:
        anchors = {}
        for k in range(len(parts)):
            a = None if parts[k][0] in skip else visible_anchor(idb, org, k, 5.0 if need_all else 2.5)
            if a is not None:
                anchors[numbering[k]] = a
        if need_all and len(anchors) < len(parts):
            return None
        if table:
            parts_table(sh, sorted((numbering[k], parts[k][0]) for k in range(len(parts))), 26, size[1] - 26)
        place_balloons(sh, anchors, p)
    if title:
        sh.title_block(title_cells(rng, "asm"))
    return sh


def _numbering(rng, n):
    items = list(range(1, n + 1))
    rng.shuffle(items)
    return items


def _asm_fasteners(rng, render=True, r=None):
    nb = rng.randint(1, 8)
    nh = rng.randint(nb, min(8, nb + 3))
    s = new_bolted(rng, n_holes=nh, n_bolts=nb)
    if s is None:
        return None
    nh, nb = len(s["holes"]), len(s["bolts"])
    if rng.random() < 0.7 or nh == nb:
        ins = rng.choice(["How many bolts are fitted in this assembly?", "Count the hex-head bolts installed in the cover plate."])
        gold, mist = nb, [nh, nh - nb]
    else:
        ins, gold, mist = "How many holes in the cover plate are left empty (no bolt fitted)?", nh - nb, [nb, nh]
    q = _count_q(ins, gold, mist, rng, r=r)
    if q is None:
        return None
    mode, view, ex = rng.choice([("line", "top", 0.0), ("line", "top", 0.0), ("shaded", "iso_hi", 0.0), ("shaded", "iso", 0.0),
                                 ("shaded", "iso_hi", rng.choice([14.0, 18.0]))])
    n_parts = len(asm_parts(s))
    numbering = _numbering(rng, n_parts)
    why = "%d of the %d cover holes carry a hex-head bolt; %d %s empty." % (nb, nh, nh - nb, "is" if nh - nb == 1 else "are")
    srng = random.Random(rng.random())
    bal = srng.random() < 0.5
    skip = ("BASE", "GASKET") if view == "top" else ()
    image = (lambda: asm_sheet(srng, s, numbering, mode, explode=ex, view=view, balloons=bal, skip=skip).image()) if render else None
    return {"q": q[0], "label": q[1], "why": why, "image": image, "meta": {"asm": s, "gold": gold, "ins": ins, "mode": mode, "view": view}}


def touching(s):
    """Pairs of part indices in direct contact, from the construction (assembled state)."""
    names = [n for n, _ in asm_parts(s)]
    ix = {n: i for i, n in enumerate(names)}
    if s["type"] == "bushing":
        pairs = [("BASE PLATE", "HOUSING"), ("HOUSING", "FLANGED BUSHING"), ("FLANGED BUSHING", "SHAFT"), ("SHAFT", "COLLAR")]
    elif s["type"] == "bolted":
        chain = [n for n in ("BASE", "GASKET", "COVER PLATE", "WASHER", "HEX BOLT") if n in ix]
        pairs = list(zip(chain[:-1], chain[1:])) + [("BASE", "HEX BOLT")]
    else:
        pairs = list(zip(names[:-1], names[1:])) + [("BASE WITH STUD", "HEX NUT")]
    return {frozenset((ix[a], ix[b])) for a, b in pairs}


def _asm_mates(rng, render=True):
    if rng.random() < 0.55:
        s = new_bushing(rng)
        names = [n for n, _ in asm_parts(s)]
        num = _numbering(rng, len(names))
        it = lambda n: "item %d" % num[names.index(n)]  # noqa: E731
        k = rng.randrange(5)
        if k == 0:
            ask, gold, opts = "Which item is in direct contact with %s?" % it("BASE PLATE"), "HOUSING", ["HOUSING", "FLANGED BUSHING", "SHAFT", "COLLAR"]
            why = "%s (the base plate) only touches %s (the housing) standing on it." % (it("BASE PLATE"), it("HOUSING"))
        elif k == 1:
            ask, gold, opts = "Which item is in direct contact with %s?" % it("COLLAR"), "SHAFT", ["BASE PLATE", "HOUSING", "FLANGED BUSHING", "SHAFT"]
            why = "%s (the collar) sits on %s (the shaft) with a gap to the housing and bushing." % (it("COLLAR"), it("SHAFT"))
        elif k == 2:
            ask = "Which item fits directly inside the bore of %s?" % it("HOUSING")
            gold, opts = "FLANGED BUSHING", ["BASE PLATE", "FLANGED BUSHING", "SHAFT", "COLLAR"]
            why = "%s (the flanged bushing) is pressed into the bore of %s; the shaft runs inside the bushing." % (it("FLANGED BUSHING"), it("HOUSING"))
        elif k == 3:
            ask = "Which item lies between %s and %s, touching both?" % tuple(rng.sample([it("SHAFT"), it("HOUSING")], 2))
            gold, opts = "FLANGED BUSHING", ["BASE PLATE", "FLANGED BUSHING", "COLLAR"]
            why = "%s (the bushing) separates the shaft from the housing bore." % it("FLANGED BUSHING")
        else:
            ask = "Which item lies between %s and %s, touching both?" % tuple(rng.sample([it("FLANGED BUSHING"), it("COLLAR")], 2))
            gold, opts = "SHAFT", ["BASE PLATE", "HOUSING", "SHAFT"]
            why = "%s (the shaft) runs in the bushing and carries the collar." % it("SHAFT")
        cut, view = 0.0, "front"
    else:
        for _ in range(50):
            s = new_bolted(rng)
            if s is not None and not s["round"]:
                ys = sorted({h[1] for h in s["holes"]})
                row = [k for k, h in enumerate(s["holes"]) if h[1] == ys[0]]
                if all(k in s["bolts"] for k in row):
                    break
        else:
            return None
        names = [n for n, _ in asm_parts(s)]
        num = _numbering(rng, len(names))
        it = lambda n: "item %d" % num[names.index(n)]  # noqa: E731
        others = [n for n in names if n != "HEX BOLT"]
        above_base = "GASKET" if "GASKET" in names else "COVER PLATE"
        under_head = "WASHER" if "WASHER" in names else "COVER PLATE"
        k = rng.choice(["thread", "on_base", "under_cover", "under_head"] + (["rest"] if "WASHER" in names else []))
        if k == "thread":
            ask, gold, opts = "Into which item are the bolts (%s) threaded?" % it("HEX BOLT"), "BASE", others
            why = "The bolt shanks pass through clearance holes and end in %s (the base), where there is no clearance." % it("BASE")
        elif k == "on_base":
            ask, gold, opts = "Which item lies directly on top of %s?" % it("BASE"), above_base, [n for n in names if n != "BASE"]
            why = "The layer resting on %s (the base) is %s (the %s)." % (it("BASE"), it(above_base), above_base.lower())
        elif k == "under_cover":
            gold = "GASKET" if "GASKET" in names else "BASE"
            ask, opts = "Which item lies directly under %s?" % it("COVER PLATE"), [n for n in names if n != "COVER PLATE"]
            why = "%s (the cover plate) sits on %s (the %s)." % (it("COVER PLATE"), it(gold), gold.lower())
        elif k == "under_head":
            ask, gold, opts = "Which item sits directly under the heads of the bolts (%s)?" % it("HEX BOLT"), under_head, others
            why = "The bolt heads bear on %s (the %s)." % (it(under_head), under_head.lower() + ("s" if under_head == "WASHER" else ""))
        else:
            ask = "On which item do the washers (%s) rest?" % it("WASHER")
            gold, opts = "COVER PLATE", [n for n in names if n not in ("WASHER", "HEX BOLT")]
            why = "%s (the washers) rest on %s (the cover plate)." % (it("WASHER"), it("COVER PLATE"))
        cut, view = ys[0], "front"
    if len(opts) < 3:
        return None
    opts = sorted(opts, key=lambda n: num[names.index(n)])
    q = {"type": "choice", "instructions": ask, "criteria": ["item %d" % num[names.index(n)] for n in opts]}
    srng = random.Random(rng.random())
    image = (lambda: asm_sheet(srng, s, num, "section", cut=cut, view=view, need_all=True).image()) if render else None
    return {"q": q, "label": opts.index(gold), "why": why, "kind": "mates", "image": image,
            "meta": {"asm": s, "numbering": num, "names": names, "gold": gold, "options": opts, "ins": ask}}


def _asm_order(rng, render=True):
    s = new_stack(rng)
    names = [n for n, _ in asm_parts(s)]
    n = len(names)
    num = _numbering(rng, n)
    it = lambda k: "item %d" % num[k]  # noqa: E731
    pre = "The parts slide onto the stud of the base (%s) from above, in the order of the exploded view. " % it(0)
    k = rng.randrange(4)
    if k == 0:
        ask, gold = pre + "Which item goes on first?", 1
        why = "The part nearest the base in the exploded stack is %s, so it goes on first." % it(1)
    elif k == 1:
        ask, gold = pre + "Which item is fitted last?", n - 1
        why = "The top part of the exploded stack, %s, is fitted last." % it(n - 1)
    elif k == 2:
        j = rng.randrange(1, n - 1)
        ask, gold = pre + "Which item is fitted immediately after %s?" % it(j), j + 1
        why = "%s sits directly above %s in the stack, so it follows it." % (it(j + 1), it(j))
    else:
        j = rng.randrange(1, n - 1)
        a, b = rng.sample([it(j - 1), it(j + 1)], 2)
        ask, gold = "In the exploded stack, which item sits directly between %s and %s?" % (a, b), j
        why = "Reading the stack from the base up, %s lies between %s and %s." % (it(j), it(j - 1), it(j + 1))
    excl = {0} | ({j} if k == 2 else set()) | ({j - 1, j + 1} if k == 3 else set())
    opts = sorted([i for i in range(n) if i not in excl or i == gold], key=lambda i: num[i])
    if len(opts) > 4:
        keep = [gold] + rng.sample([i for i in opts if i != gold], 3)
        opts = sorted(keep, key=lambda i: num[i])
    if len(opts) < 3:
        return None
    q = {"type": "choice", "instructions": ask, "criteria": [it(i) for i in opts]}
    srng = random.Random(rng.random())
    mode, view = srng.choice([("shaded", "iso"), ("line", "iso"), ("line", "front"), ("shaded", "iso")])
    ex = srng.choice([10.0, 13.0, 16.0]) if view == "front" else srng.choice([24.0, 28.0, 32.0])

    def image():
        for k in range(4):  # widen the gaps until every part shows enough of itself for its balloon
            seed = srng.random()
            sh = asm_sheet(random.Random(seed), s, num, mode, explode=ex * (1 + 0.35 * k), view=view, centreline=True, size=(768, 896),
                           need_all=True)
            if sh is not None:
                return sh.image()
        raise RuntimeError("stack not drawable")

    image = image if render else None
    return {"q": q, "label": opts.index(gold), "why": why, "kind": "order", "image": image,
            "meta": {"asm": s, "numbering": num, "names": names, "gold": gold, "options": opts, "ins": ask}}


def make_assembly(rng, render=True):
    rk = rng.randrange(4)
    for _ in range(400):
        r = rng.random()
        out = _asm_fasteners(rng, render, r=rk) if r < 0.36 else _asm_mates(rng, render) if r < 0.7 else _asm_order(rng, render)
        if out is not None:
            out.setdefault("kind", "fasteners")
            return out
    raise RuntimeError("assembly")


# ---- 2. cad_view_match -----------------------------------------------------------------------------------------------


def view_raster(v, scale, res=200, hidden=True):
    """Visible (and hidden) lines of a view on a res x res bitmap, bbox centred, for comparing two views."""
    from PIL import Image, ImageDraw

    out = []
    cx, cy = (v.gb[0] + v.gb[2]) / 2, (v.gb[1] + v.gb[3]) / 2
    groups = [v.vis + [np.vstack([f[0], f[0][:1]]) for f in v.faces] + [np.vstack([i, i[:1]]) for f in v.faces for i in f[1]]]
    if hidden:
        groups.append(v.hid)
    for polys in groups:
        im = Image.new("L", (res, res), 0)
        dr = ImageDraw.Draw(im)
        for q in polys:
            pts = [(float((x - cx) * scale + res / 2), float(res / 2 - (y - cy) * scale)) for x, y in q]
            dr.line(pts, fill=255, width=1)
        out.append(im)
    return out


def view_diff(a, b, scale, res=200, hidden=True):
    """Pixels of one view's lines that are more than 2 px away from any line of the other (both directions)."""
    from PIL import ImageFilter

    n = 0
    for x, y in zip(view_raster(a, scale, res, hidden), view_raster(b, scale, res, hidden)):
        xa, ya = np.asarray(x) > 0, np.asarray(y) > 0
        xd, yd = np.asarray(x.filter(ImageFilter.MaxFilter(5))) > 0, np.asarray(y.filter(ImageFilter.MaxFilter(5))) > 0
        n += int((xa & ~yd).sum() + (ya & ~xd).sum())
    return n


def rich_part(rng, fam=None):
    """A part with enough features to make near-misses: used by view matching and pairs."""
    fam = fam or rng.choice(["plate", "plate", "block", "bracket", "bracket", "shaft", "flange"])
    for _ in range(100):
        if fam == "plate":
            spec = new_plate(rng, big=True)
            spec["cr"] = [rng.choice([0, 0, 8, 12]) for _ in range(4)]
            ok = put(rng, spec, "top", rng.randint(2, 4), hole()) and put(rng, spec, "top", rng.randint(0, 1), hole(cb=True)) \
                and put(rng, spec, "top", rng.randint(0, 2), slot)
        elif fam == "block":
            spec = new_plate(rng, block=True)
            boss = lambda r: {"kind": "boss", "d": r.choice([16, 20, 24]), "h": r.choice([8, 12, 16]), "bore": r.choice([0, 8])}  # noqa: E731
            pocket = lambda r: {"kind": "pocket", "l": r.choice([24, 30, 36]), "w": r.choice([14, 18, 22]), "depth": r.choice([8, 12]),  # noqa: E731
                                "r": r.choice([0, 3, 5])}
            ok = put(rng, spec, "top", rng.randint(0, 1), boss) and put(rng, spec, "top", rng.randint(0, 1), pocket) \
                and put(rng, spec, "top", rng.randint(1, 3), hole(depth=rng.choice([None, 10, 14])))
            ok = ok and len(spec["feats"]) >= 3
        elif fam == "bracket":
            spec = new_bracket(rng)
            ok = put(rng, spec, "base", rng.randint(1, 3), hole()) and put(rng, spec, "leg0", rng.randint(1, 2), hole())
            if spec["shape"] == "U":
                ok = ok and put(rng, spec, "leg1", rng.randint(0, 2), hole())
        elif fam == "shaft":
            try:
                return new_shaft(rng, nseg=rng.randint(3, 5), grooves=rng.randint(0, 2), keyway=rng.random() < 0.7)
            except RuntimeError:
                continue
        else:
            spec = new_flange(rng, hub=rng.choice([12, 16, 20, 25, 30]))
            if spec["n"] % 2 == 0:
                spec["a0"] = rng.choice([0.0, math.pi / spec["n"]])
            ok = True
        if ok and valid(spec):
            return spec
    raise RuntimeError("no part")


def front_marker(spec, view="iso"):
    """Page-mm arrow (tail, tip) on the ground in front of the part pointing at its front face, and the label point."""
    L, W, _ = extent(spec)
    round_ = spec["fam"] in ("flange", "gear")
    xm = 0.0 if round_ else L / 2
    y0 = -W / 2 if (round_ or spec["fam"] == "shaft") else 0.0
    z0 = -W / 2 if spec["fam"] == "shaft" else 0.0
    size = max(L, W)
    pts = project([(xm, y0 - 0.42 * size, z0), (xm, y0 - 0.12 * size, z0), (xm, y0 - 0.5 * size, z0)], view)
    return pts[0], pts[1], pts[2]


def draw_iso(sh, spec, cx, cy, w, h, shaded, view="iso", marker=True, smax=4.0):
    v = ortho(build(spec), view, hidden=False)
    if marker:
        v.grow(front_marker(spec, view))
    s = fit([v], w, h, smax=smax)
    if shaded:
        p = draw_shaded(sh, [(build(spec), sh.st["shade_rgb"])], view, cx, cy, s, v=v)[0]
    else:
        p = draw_view(sh, v, cx, cy, s, smooth=True)
    if marker:
        a, b, c = [p.P(*q) for q in front_marker(spec, view)]
        d = _unit(np.array(b) - np.array(a))
        n = np.array([-d[1], d[0]])
        col = sh.st["dimcol"]
        sh.poly([[a, tuple(np.array(b) - 8 * d)]], sh.st["lw"] * 1.2, col, z=5)
        sh.fill([b, tuple(np.array(b) - 13 * d + 6 * n), tuple(np.array(b) - 13 * d - 6 * n)], col, z=6)
        sh.text(c[0], c[1] - 6, "FRONT", size=11, weight="bold", col=col)
    return p


def cut_line(sh, p, y, x0, x1, letter="A"):
    """Cutting-plane line across a plan view at page-mm y, arrows pointing away from the viewer of a front section."""
    st = sh.st
    a, b = p.P(x0, y), p.P(x1, y)
    a, b = (a[0] - 16, a[1]), (b[0] + 16, b[1])
    sh.poly([[a, b]], st["lw"] * 0.9, col=st["dimcol"], dash=(9, 2.5, 1.5, 2.5), z=4.5)
    for q in (a, b):
        sh.poly([[q, (q[0], q[1] + 20)]], st["lw"] * 1.2, col=st["dimcol"], z=4.5)
        sh.fill([(q[0], q[1] + 30), (q[0] - 5, q[1] + 18), (q[0] + 5, q[1] + 18)], st["dimcol"], z=5)
    sh.text(a[0] - 11, a[1] + 8, letter, size=13, weight="bold", col=st["dimcol"])
    sh.text(b[0] + 11, b[1] + 8, letter, size=13, weight="bold", col=st["dimcol"])


VM_TH = 16
VM_ASK = {"top": ["Which panel shows the TOP view of the part in the isometric view?",
                  "The arrow marks the front of the part. Which panel is its top view (seen from above)?"],
          "front": ["Which panel shows the FRONT view of the part (seen along the FRONT arrow)?",
                    "Which panel is the front view of the part in the isometric view?"],
          "right": ["Which panel shows the RIGHT-side view of the part (seen from the right of the FRONT direction)?",
                    "Which panel is the right-hand side view of the part in the isometric view?"],
          "section": ["Which panel shows SECTION A–A, the cut marked in the plan view, seen in the direction of the arrows?",
                      "Which panel is section A–A of this part?"]}


def _cut_pos(spec):
    """y of a section plane through the part's features (front section): the row most features share, or the axis."""
    if spec["fam"] in ("flange", "shaft", "gear"):
        return 0.0
    vs = Counter(ft["v"] for ft in spec["feats"] if ft["face"] in ("top", "bottom", "base"))
    return float(vs.most_common(1)[0][0]) if vs else spec["W"] / 2


def make_view_match(rng, render=True):
    for _ in range(60):
        spec = rich_part(rng)
        fam = spec["fam"]
        thin = fam == "plate" and spec["T"] < 12  # a thin plate's side views are strips too narrow to read
        ask = rng.choice({"plate": ["top", "top", "top"] if thin else ["top", "top", "front", "right", "section"],
                          "block": ["top", "front", "right", "section", "section"],
                          "bracket": ["top", "front", "right"], "shaft": ["front", "top", "right"],
                          "flange": ["top", "front", "section", "section"]}[fam])
        hidden = ask != "section" and rng.random() < 0.65
        cut = _cut_pos(spec) if ask == "section" else None
        vname = "front" if ask == "section" else ask
        gold_v = part_view(spec, vname, hidden=hidden, cut=cut)
        iso_g = ortho(build(spec), "iso", hidden=False)
        plan_g = part_view(spec, "top", hidden=False)
        ex = max(extent(spec))
        sc, isc = 190 / ex, 190 / (1.5 * ex)
        cands, kinds = [], Counter()
        for _ in range(70):
            if len(cands) == 3:
                break
            kind = rng.choice(["move", "move", "remove", "mirror", "resize", "resize", "add"])
            if kinds[kind] >= (1 if kind == "mirror" else 2):
                continue
            s2 = mutate(spec, rng, kind)
            if s2 is None:
                continue
            try:
                v2 = part_view(s2, vname, hidden=hidden, cut=cut)
                if view_diff(v2, gold_v, sc, hidden=hidden) < VM_TH or any(view_diff(v2, c[1], sc, hidden=hidden) < VM_TH for c in cands):
                    continue
                if view_diff(ortho(build(s2), "iso", hidden=False), iso_g, isc, hidden=False) < VM_TH:
                    continue
                if ask == "section" and view_diff(part_view(s2, "top", hidden=False), plan_g, sc, hidden=False) < VM_TH:
                    continue  # the plan view with the cutting line must tell them apart as well
            except Exception:  # noqa: BLE001 - a degenerate boolean: skip this near-miss
                continue
            cands.append((s2, v2, kind))
            kinds[kind] += 1
        if len(cands) < 3:
            continue
        gold = rng.randrange(4)
        order = cands[:]
        rng.shuffle(order)
        panels = order[:gold] + [(spec, gold_v, "gold")] + order[gold:]
        labels = ["1", "2", "3", "4"] if (ask == "section" or rng.random() < 0.3) else ["A", "B", "C", "D"]
        q = {"type": "choice", "instructions": rng.choice(VM_ASK[ask]), "criteria": ["panel %s" % x for x in labels]}
        names = {"move": "a feature moved", "remove": "a feature missing", "mirror": "the part mirrored", "resize": "a size changed",
                 "add": "an extra feature"}
        why = "Panel %s matches the %s of the part; the others show near-miss parts (%s)." % (
            labels[gold], "section A–A" if ask == "section" else ask + " view", ", ".join(
                "%s: %s" % (labels[i], names[pn[2]]) for i, pn in enumerate(panels) if i != gold))
        srng = random.Random(rng.random())

        def image():
            sh = new_sheet(srng, 896, 640)
            st = sh.st
            shaded = srng.random() < 0.6
            if ask == "section":
                draw_iso(sh, spec, 172, 470, 250, 230, shaded)
                pv = part_view(spec, "top", hidden=False, centre=st["centre"])
                pv.grow([(pv.gb[0] - 8, pv.gb[1]), (pv.gb[2] + 8, pv.gb[3])])
                pp = draw_view(sh, pv, 172, 170, fit([pv], 210, 190, smax=3.0))
                cut_line(sh, pp, cut, pv.gb[0], pv.gb[2])
                sh.text(172, 330, "ISOMETRIC VIEW", size=10, reserve=False)
                sh.text(172, 40, "PLAN (TOP VIEW)", size=10, reserve=False)
            else:
                draw_iso(sh, spec, 172, 330, 270, 420, shaded)
                sh.text(172, 606, "ISOMETRIC VIEW", size=10, reserve=False)
            sh.poly([[(332, 22), (332, 618)]], st["thin"], z=2)
            vs = [c[1] for c in panels]
            s = fit(vs, 225, 215, smax=3.6)
            for k, (_, v, _) in enumerate(panels):
                r, c = divmod(k, 2)
                x0, y0 = 344 + c * 272, 326 - r * 300
                sh.poly([[(x0, y0), (x0 + 262, y0), (x0 + 262, y0 + 290), (x0, y0 + 290), (x0, y0)]], st["thin"], z=2)
                draw_view(sh, v, x0 + 131, y0 + 135, s, hidden=hidden)
                sh.fill([(x0, y0 + 262), (x0 + 30, y0 + 262), (x0 + 30, y0 + 290), (x0, y0 + 290)], st["bg"], z=5.5, edge=st["ink"], lw=st["thin"])
                sh.text(x0 + 15, y0 + 276, labels[k], size=16, weight="bold", reserve=False)
            return sh.image()

        return {"q": q, "label": gold, "why": why, "kind": ask, "image": image if render else None,
                "meta": {"spec": spec, "panels": [c[0] for c in panels], "ask": ask, "cut": cut, "hidden": hidden,
                         "mutations": [c[2] for c in panels]}}
    raise RuntimeError("view_match")


# ---- 4. cad_drawing_pair ---------------------------------------------------------------------------------------------


def _sym(spec, flip):
    """Are the features (nearly, within 10 mm) unchanged by ``flip`` = "x" (left-right) or "y" (front-back)?"""
    fr = frames(spec)
    L, W = spec["L"], spec["W"]

    def img(ft):
        f, u, v = ft["face"], ft["u"], ft["v"]
        if flip == "x":
            if f in ("top", "bottom", "base"):
                u = L - u
            else:
                f = {"leg0": "leg1", "leg1": "leg0"}[f]
        else:
            if f in ("top", "bottom", "base"):
                v = W - v
            else:
                u = W - u
        return f, u, v

    sig = lambda ft: (ft["kind"], ft.get("d"), ft.get("l"), ft.get("w"), bool(ft.get("cb")), ft.get("h"), ft.get("depth"))  # noqa: E731
    left = list(spec["feats"])
    for ft in spec["feats"]:
        f, u, v = img(ft)
        if f not in fr:
            return False
        hit = next((o for o in left if o["face"] == f and sig(o) == sig(ft) and abs(o["u"] - u) <= 10 and abs(o["v"] - v) <= 10), None)
        if hit is None:
            return False
        left.remove(hit)
    if spec["fam"] != "bracket" and flip == "x":
        return True
    return spec["fam"] != "bracket" or (flip == "x" and spec["shape"] == "U") or flip == "y"


def chiral(spec):
    """True when no mirror image of the part can be turned back into the part: the features break every mirror
    plane the blank has, and plates carry a feature that shows which face is up."""
    f = spec["fam"]
    if f in ("plate", "block"):
        up = any(ft.get("cb") or ft["kind"] in ("boss",) for ft in spec["feats"])
        return up and abs(spec["L"] - spec["W"]) >= 10 and not _sym(spec, "x") and not _sym(spec, "y")
    if f == "bracket":
        if spec["shape"] == "L":
            return abs(spec["L"] - spec["H"]) >= 10 and not _sym(spec, "y")
        return not _sym(spec, "x") and not _sym(spec, "y")
    return False


PAIR_OPTS = {"count": "the number of holes", "dim": "one dimension (a length or a diameter)",
             "fillet": "a fillet (rounded corner or bend radius) is missing in one of them", "mirror": "one is the mirror image of the other"}
PAIR_WHY = {"count": "One drawing has %d holes, the other %d.", "dim": "%s differs: %s in one drawing, %s in the other.",
            "fillet": "One %s is rounded in one drawing and sharp in the other.", "mirror": "The second part is the left-right mirror image of the first."}


def pair_part(rng):
    for _ in range(300):
        fam = rng.choice(["plate", "plate", "block", "bracket", "bracket"])
        if fam == "bracket":  # L only: in a U the far leg can hide the holes that make the part chiral
            spec = new_bracket(rng, "L")
            spec["rin"], spec["rout"] = rng.choice([5, 6, 8]), rng.random() < 0.5
            ok = put(rng, spec, "base", rng.randint(1, 3), hole()) and put(rng, spec, "leg0", rng.randint(1, 2), hole())
        else:
            spec = new_plate(rng, block=fam == "block", big=True)
            spec["cr"] = [rng.choice([0, 10, 12]) for _ in range(4)]
            if not any(spec["cr"]):
                spec["cr"][rng.randrange(4)] = 10
            ok = put(rng, spec, "top", rng.randint(1, 2), hole(cb=True)) and put(rng, spec, "top", rng.randint(1, 4), hole())
        if ok and valid(spec) and chiral(spec):
            return spec
    raise RuntimeError("no chiral part")


def _pair_view(spec, mode):
    if mode == "top":  # a bracket's plan shows its leg holes as hidden lines, so every hole is on the drawing
        return part_view(spec, "top", hidden=spec["fam"] == "bracket")
    return ortho(build(spec), "iso_l" if mode.startswith("iso_l") else "iso", hidden=False)


def _rot180(v):
    c = np.array([(v.gb[0] + v.gb[2]) / 2, (v.gb[1] + v.gb[3]) / 2])
    return View([2 * c - q for q in v.vis], [2 * c - q for q in v.hid])


def make_drawing_pair(rng, render=True):
    qtype = "noul" if rng.random() < 0.5 else "choice"  # drawn once: a retry must not favour the classes that are easy to build
    cls = "same" if (qtype == "noul" and rng.random() < 0.5) else rng.choice(["count", "dim", "fillet", "mirror"])
    for _ in range(400):
        a = pair_part(rng)
        pres = rng.choice([("top", "top"), ("top", "top"), ("iso_shaded", "top"), ("iso_shaded", "iso_l_shaded"), ("iso_line", "iso_l_line"),
                           ("iso_shaded", "iso_line")])
        if a["fam"] == "bracket" and pres == ("top", "top"):
            pres = ("iso_shaded", "iso_l_shaded")
        if rng.random() < 0.5:
            pres = pres[::-1]
        if not all(m == "top" or all_visible(a, mode_view(m)) for m in pres):
            continue
        b, detail = a, ""
        if cls != "same":
            ex = max(extent(a))
            for _ in range(40):
                b = mutate(a, rng, cls)
                if b is None:
                    continue
                sc = 190 / (1.5 * ex)
                if min(view_diff(_pair_view(a, m), _pair_view(b, m), sc) for m in pres) < 2 * VM_TH:
                    b = None  # the change must show in both panels
                elif cls == "mirror" and "top" in pres and view_diff(_rot180(_pair_view(a, "top")), _pair_view(b, "top"), sc) < 2 * VM_TH:
                    b = None  # a plan that is merely turned half round is not a mirror image
                elif not all(m == "top" or all_visible(b, mode_view(m)) for m in pres):
                    b = None
                else:
                    break
            if b is None:
                continue
            if cls == "count":
                detail = PAIR_WHY[cls] % (len(holes_of(a)), len(holes_of(b)))
            elif cls == "dim":
                key = next((k for k in ("L", "W", "H", "T") if a.get(k) != b.get(k)), None)
                if key:
                    detail = PAIR_WHY[cls] % ({"L": "The overall length", "W": "The overall width", "H": "The height", "T": "The thickness"}[key],
                                              "%g mm" % a[key], "%g mm" % b[key])
                else:
                    da, db = sorted({f["d"] for f in holes_of(a)} - {f["d"] for f in holes_of(b)}), sorted(
                        {f["d"] for f in holes_of(b)} - {f["d"] for f in holes_of(a)})
                    detail = PAIR_WHY[cls] % ("A hole diameter", "%g mm" % da[0], "%g mm" % db[0])
            elif cls == "fillet":
                detail = PAIR_WHY[cls] % ("bend (inside radius)" if a["fam"] == "bracket" else "outline corner")
            else:
                detail = PAIR_WHY[cls]
        first, second = (a, b) if rng.random() < 0.5 else (b, a)
        tags = rng.choice([("DRAWING 1", "DRAWING 2"), ("DRAWING A", "DRAWING B"), ("LEFT", "RIGHT")])
        if qtype == "noul":
            q = _noul(rng.choice(["Do the two drawings show the same part?", "Is the part in the second drawing identical to the part in the first?",
                                  "Are these two views of one and the same part?"]))
            label = int(cls == "same")
            why = "Every feature, size and position agrees, so it is the same part." if cls == "same" else "Not the same part. " + detail
        else:
            keys = list(PAIR_OPTS)
            q, label = _mc("The two drawings show parts that differ in exactly one respect. What is it?", [PAIR_OPTS[k] for k in keys],
                           keys.index(cls), rng)
            why = detail
        srng = random.Random(rng.random())
        dims = pres == ("top", "top") and srng.random() < 0.4

        def image():
            sh = new_sheet(srng, 896, 600)
            st = sh.st
            sh.poly([[(448, 22), (448, 578)]], st["thin"], z=2)
            for k, (spec, mode) in enumerate(((first, pres[0]), (second, pres[1]))):
                cx = 232 + k * 432
                sh.text(cx, 556, tags[k], size=13, weight="bold", reserve=False)
                v = _pair_view(spec, mode)
                shrink = srng.uniform(0.78, 1.0)
                if mode == "top":
                    v.centre = centre_marks(spec, "top") if st["centre"] else []
                    s = fit([v], 300 * shrink, 360 * shrink, smax=3.6)
                    p = draw_view(sh, v, cx, 285, s)
                    if dims:
                        dim_plate(sh, spec, {"top": p}, Num(), srng)
                else:
                    s = fit([v], 350 * shrink, 400 * shrink, smax=3.6)
                    vn = "iso_l" if mode.startswith("iso_l") else "iso"
                    if mode.endswith("shaded"):
                        draw_shaded(sh, [(build(spec), st["shade_rgb"])], vn, cx, 285, s, v=v)
                    else:
                        draw_view(sh, v, cx, 285, s, smooth=True)
            return sh.image()

        return {"q": q, "label": label, "why": why, "kind": "%s_%s" % (qtype, cls), "image": image if render else None,
                "meta": {"a": first, "b": second, "cls": cls, "pres": pres}}
    raise RuntimeError("drawing_pair")


# ---- 5. cad_section_depth --------------------------------------------------------------------------------------------
HOLE_TYPES = {"through": "a plain through hole", "blind": "a plain blind hole", "cb_through": "a counterbored through hole",
              "cb_blind": "a counterbored blind hole"}


def hole_type(ft):
    return ("cb_" if ft.get("cb") else "") + ("through" if ft.get("depth") is None else "blind")


def section_row(rng, n=None, types=None):
    """A block with ``n`` holes on one line (the section plane), left to right, depths clearly different."""
    for _ in range(200):
        spec = new_plate(rng, block=True)
        spec["L"], spec["T"] = rng.randrange(110, 165, 5), rng.randrange(20, 45, 5)
        T = spec["T"]
        n_ = len(types) if types else (n or rng.randint(3, 5))
        ts = types or [rng.choice(list(HOLE_TYPES)) for _ in range(n_)]
        yc = rng.randint(14, spec["W"] - 14)
        depths, feats, u = [], [], rng.randint(12, 20)
        for t in ts:
            d = rng.choice([8, 10, 12])
            ft = {"kind": "hole", "d": d, "face": "top", "v": yc}
            if "blind" in t:
                for _ in range(40):
                    dep = rng.randint(max(6, int(0.25 * T)), int(0.78 * T))
                    if all(abs(dep - o) >= max(3, 0.13 * T) for o in depths) and T - dep >= 4:
                        break
                else:
                    dep = None
                if dep is None:
                    break
                depths.append(dep)
                ft["depth"] = dep
            if t.startswith("cb"):
                ft["cb"] = [d + rng.choice([6, 8]), rng.choice([3, 4, 5])]
                if ft.get("depth") is not None and ft["cb"][1] > ft["depth"] - 3:
                    break
            r = _half(ft)[0]
            ft["u"] = u + round(r)
            u = ft["u"] + round(r) + rng.randint(7, 16)
            feats.append(ft)
        if len(feats) != n_ or u > spec["L"] - 4 or [hole_type(f) for f in feats] != list(ts):
            continue
        spec["feats"] = feats
        k = len(feats)
        if rng.random() < 0.6:  # holes that the section does not cut
            put(rng, spec, "top", rng.randint(1, 2), hole(depth=rng.choice([None, 8])))
            if any(abs(ft["v"] - yc) < _half(ft)[0] + 5 + 2 for ft in spec["feats"][k:]):
                continue
        if valid(spec):
            return spec, yc, k
    raise RuntimeError("no section row")


def depth_of(spec, ft):
    return spec["T"] if ft.get("depth") is None else ft["depth"]


def make_section_depth(rng, render=True):
    kind = rng.choices(["through", "deeper", "deepest", "count_through", "type", "wall"], [2.5, 1.5, 2, 1.5, 2, 1.8])[0]
    want = rng.random() < 0.5  # kind, wanted answer and option rank are drawn once, so retries cannot skew the labels
    rk = rng.randrange(4)
    for _ in range(400):
        names_shuffled = rng.random() < 0.5
        dim_hole = None
        if kind in ("through", "wall") and rng.random() < 0.25:  # a flange cut through its axis
            spec = new_flange(rng, hub=rng.choice([14, 18, 22, 28]))
            H = spec["T"] + spec["hub_h"]
            if kind == "through":
                if not want:
                    spec["bore_depth"] = rng.randint(max(5, int(0.35 * H)), int(0.75 * H))
                if not valid(spec):
                    continue
                q = _noul(rng.choice(["Does the centre bore pass completely through the part?", "Is the centre bore a through hole?"]))
                label = int(want)
                why = "In the section the bore %s." % ("runs from the hub face to the underside" if want else
                                                        "stops %g mm short of the underside" % (H - spec["bore_depth"]))
            else:
                gold = (spec["hub_d"] - spec["bore_d"]) / 2
                qq = _num_choice("What is the radial wall thickness of the hub (between the bore and the outside of the hub)?", gold,
                                 [spec["hub_d"] - spec["bore_d"], spec["bore_d"] / 2, spec["hub_d"] / 2, spec["T"]],
                                 lambda v: "%g mm" % v, rng, [-3, -2, -1, 1, 2, 3], lo=1, r=rk)
                if qq is None:
                    continue
                q, label = qq
                why = "Hub %s%g minus bore %s%g leaves %g on the diameter, so the wall is %g mm on each side." % (
                    "⌀", spec["hub_d"], "⌀", spec["bore_d"], spec["hub_d"] - spec["bore_d"], gold)
            meta = {"spec": spec, "cut": 0.0, "kind": kind}
            names, k = [], 0
        else:
            types = None
            if kind == "type":
                t = rng.choice(list(HOLE_TYPES))
                n = rng.randint(3, 5)
                types = [t] + [rng.choice(list(HOLE_TYPES)) for _ in range(n - 1)]
                rng.shuffle(types)
            elif kind == "deepest":
                n = rng.randint(3, 5)
                one = rng.random() < 0.35
                types = [rng.choice(["blind", "cb_blind", "blind"]) for _ in range(n)]
                if one:
                    types[rng.randrange(n)] = rng.choice(["through", "cb_through"])
            elif kind == "through":
                n = rng.randint(2, 5)
                types = [rng.choice(["through", "cb_through"]), rng.choice(["blind", "cb_blind"])] + [rng.choice(list(HOLE_TYPES)) for _ in range(n - 2)]
                rng.shuffle(types)
            elif kind == "wall":
                types = [rng.choice(["blind", "cb_blind", "blind"])] + [rng.choice(list(HOLE_TYPES)) for _ in range(rng.randint(1, 3))]
                if rng.random() < 0.5:
                    types = types[::-1]
            try:
                spec, yc, k = section_row(rng, types=types)
            except RuntimeError:
                continue
            hs = spec["feats"][:k]
            T = spec["T"]
            ids = list(range(1, k + 1))
            if names_shuffled:
                rng.shuffle(ids)
            names = ["H%d" % i for i in ids]
            by_name = sorted(range(k), key=lambda i: ids[i])
            if kind == "through":
                pool = [i for i in range(k) if is_through(hs[i]) == want]
                i = rng.choice(pool)
                q = _noul(rng.choice(["Is hole %s a through hole (open at both faces)?", "Does hole %s go all the way through the part?"]) % names[i])
                label = int(want)
                why = "In section A–A hole %s %s." % (names[i], "is open from the top face to the underside" if want else
                                                          "ends %g mm deep in the %g mm thick part" % (hs[i]["depth"], T))
            elif kind == "deeper":
                i, j = rng.sample(range(k), 2)
                if depth_of(spec, hs[i]) == depth_of(spec, hs[j]):
                    continue
                label = int(depth_of(spec, hs[i]) > depth_of(spec, hs[j]))
                q = _noul("Is hole %s deeper than hole %s?" % (names[i], names[j]))
                why = "%s is %g mm deep and %s is %g mm deep (a through hole counts as the full %g mm)." % (
                    names[i], depth_of(spec, hs[i]), names[j], depth_of(spec, hs[j]), T)
            elif kind == "deepest":
                ds = [depth_of(spec, h) for h in hs]
                sh_ = rng.random() < 0.35
                best = min(ds) if sh_ else max(ds)
                if ds.count(best) != 1:
                    continue
                i = ds.index(best)
                q = {"type": "choice", "instructions": "Which of the labelled holes is the %s?" % ("shallowest" if sh_ else "deepest"),
                     "criteria": [names[j] for j in by_name]}
                label = by_name.index(i)
                why = "Depths in the section: %s; %s is the %s." % (", ".join("%s %g mm%s" % (names[j], ds[j], " (through)" if is_through(hs[j]) else "")
                                                                             for j in by_name), names[i], "shallowest" if sh_ else "deepest")
            elif kind == "count_through":
                t = sum(is_through(h) for h in hs)
                extra = len(spec["feats"]) - k
                qq = _count_q("How many of the holes cut by section A–A go all the way through the part?", t, [k, k - t, t + extra], rng, r=rk)
                if qq is None:
                    continue
                q, label = qq
                why = "Section A–A cuts %d holes; %d of them are open to the underside and %d are blind." % (k, t, k - t)
            elif kind == "type":
                i = rng.choice([j for j in range(k) if hole_type(hs[j]) == t])
                keys = list(HOLE_TYPES)
                q, label = _mc("What kind of hole is %s?" % names[i], [HOLE_TYPES[x] for x in keys], keys.index(t), rng)
                why = "%s %s and %s." % (names[i], "has a wider counterbore at the top" if hs[i].get("cb") else "has one diameter all the way",
                                         "runs through the part" if is_through(hs[i]) else "stops inside the part")
            else:
                ends = [j for j in (0, k - 1) if not is_through(hs[j])]  # an end hole: its depth dimension stays clear of the others
                if not ends:
                    continue
                i = rng.choice(ends)
                gold = T - hs[i]["depth"]
                others = [T - h["depth"] for h in hs if not is_through(h) and h is not hs[i]]
                qq = _num_choice("How much material is left under hole %s (from its bottom to the underside of the part)?" % names[i], gold,
                                 [hs[i]["depth"], T] + others, lambda v: "%g mm" % v, rng, [-3, -2, -1, 1, 2, 3], lo=1, r=rk)
                if qq is None:
                    continue
                q, label = qq
                why = "The part is %g mm thick and hole %s is %g mm deep: %g − %g = %g mm." % (T, names[i], hs[i]["depth"], T, hs[i]["depth"], gold)
                dim_hole = i
            meta = {"spec": spec, "cut": yc, "kind": kind, "names": names, "k": k}
        srng = random.Random(rng.random())

        def image():
            sh = new_sheet(srng, 896, 672)
            st = sh.st
            flange = spec["fam"] == "flange"
            cut = meta["cut"]
            sec = part_view(spec, "front", cut=cut)
            plan_on = flange or srng.random() < 0.75
            views = {"front": sec}
            if plan_on:
                pv = part_view(spec, "top", hidden=False, centre=st["centre"] or flange)
                pv.grow([(pv.gb[0] - 9, pv.gb[1]), (pv.gb[2] + 9, pv.gb[3])])
                views["top"] = pv
            angle = srng.choice(["third", "third", "first"])
            title = srng.random() < 0.6
            P = layout(sh, views, arrange(list(views), angle), (20, 58 if title else 20, 876, 652), padx=74, pady=62, smax=4.4)
            f = P["front"]
            sh.text((f.x0 + f.x1) / 2, f.y0 - (84 if (kind == "wall" and flange) else 22), "SECTION A–A", size=12, weight="bold")
            if plan_on:
                cut_line(sh, P["top"], cut, views["top"].gb[0], views["top"].gb[2])
            if flange and kind == "wall":
                dim_flange(sh, spec, {"front": f}, Num(), srng)
            for i in range(k):
                ft = spec["feats"][i]
                sh.text(f.P(ft["u"], 0)[0], f.y1 + 13, names[i], size=12, weight="bold")
                if plan_on:
                    t = P["top"]
                    cx, cy = t.P(ft["u"], ft["v"])
                    r = _half(ft)[0] * t.s
                    sh.text(cx + r + 3, cy + r + 8, names[i], size=11, ha="left")
            if dim_hole is not None:
                ft = spec["feats"][dim_hole]
                T = spec["T"]
                left = dim_hole == 0
                xe = f.x0 if left else f.x1
                xh = f.P(ft["u"] + (-1 if left else 1) * ft["d"] / 2, 0)[0]
                sg = -1 if left else 1
                sh.dim_v(f.P(0, T)[1], f.P(0, T - ft["depth"])[1], xe, xh, xe + sg * 28, Num().s(ft["depth"]))
                xo = f.x1 if left else f.x0
                sh.dim_v(f.P(0, 0)[1], f.P(0, T)[1], xo, xo, xo - sg * 28, Num().s(T))
            if title:
                sh.title_block(title_cells(srng, spec["fam"], angle=angle))
            return sh.image()

        return {"q": q, "label": label, "why": why, "kind": kind, "image": image if render else None, "meta": meta}
    raise RuntimeError("section_depth")


# ---- 3. cad_dimension_read -------------------------------------------------------------------------------------------
# Each family returns (spec, facts): facts = {"read": [(phrase, value, why)], "chain": [...], "noul": [(question, a, b)]}
# with values in spec millimetres; "is a greater than b" is the noul truth (None when a == b).


def _dr_shaft(rng):
    for _ in range(200):
        n = rng.randint(3, 5)
        segs = [[rng.randrange(16, 62, 2), rng.randrange(20, 50, 2)] for _ in range(n)]
        spec = {"fam": "shaft", "segs": segs, "grooves": [], "keyway": None, "chamfer": 0}
        ds, ls = [d for d, _ in segs], [l for _, l in segs]
        if valid(spec) and sum(ls) <= 180 and len(set(ds)) == n and len(set(ls)) == n:
            break
    else:
        raise RuntimeError
    big, small, long_ = ds.index(max(ds)), ds.index(min(ds)), ls.index(max(ls))
    tot = sum(ls)
    read = [("the diameter of the largest section of the shaft", ds[big], "The largest diameter printed is %s."),
            ("the diameter of the smallest section of the shaft", ds[small], "The smallest diameter printed is %s."),
            ("the length of the longest section of the shaft", ls[long_], "The longest single length in the chain is %s."),
            ("the length of the section that has the largest diameter", ls[big], "The section with the largest diameter is %s long."),
            ("the diameter of the leftmost section of the shaft", ds[0], "The leftmost section is dimensioned %s in diameter."),
            ("the diameter of the rightmost section of the shaft", ds[-1], "The rightmost section is dimensioned %s in diameter.")]
    chain = [("the overall length of the shaft", tot, "No overall length is printed; the chain adds up: " + " + ".join(["%s"] * n) + " = %s.", ls),
             ("the combined length of the two leftmost sections", ls[0] + ls[1], "%s + %s = %s.", ls[:2]),
             ("the distance from the left end of the shaft to the start of the rightmost section", tot - ls[-1],
              "Add every section except the last: " + " + ".join(["%s"] * (n - 1)) + " = %s.", ls[:-1])]
    noul = [("Is the leftmost section of the shaft longer than the rightmost section?", ls[0], ls[-1],
             "The leftmost section is %s long and the rightmost %s."),
            ("Is the largest diameter of the shaft more than twice its smallest diameter?", ds[big], 2 * ds[small],
             "The largest diameter is %s; twice the smallest is %s."),
            ("Is the leftmost section larger in diameter than the rightmost section?", ds[0], ds[-1],
             "The leftmost diameter is %s and the rightmost %s."),
            ("Is the longest section of the shaft also longer than the largest diameter?", ls[long_], ds[big],
             "The longest section is %s long; the largest diameter is %s.")]
    return spec, {"read": read, "chain": chain, "noul": noul, "all": ds + ls, "views": "front", "overall": False}


def _dr_plate(rng):
    for _ in range(300):
        spec = new_plate(rng, big=True)
        L, W, T = spec["L"], spec["W"], spec["T"]
        n = rng.randint(2, 4)
        d = rng.choice([6, 8, 10, 12])
        a = rng.randrange(12, 34, 2)
        ps = [rng.randrange(18, 46, 2) for _ in range(n - 1)]
        if a + sum(ps) > L - 12 or len(set(ps + [a])) < n or L - a - sum(ps) == a or d == T:
            continue
        v = rng.randint(d, W - d)
        u = a
        for k in range(n):
            spec["feats"].append({"kind": "hole", "d": d, "face": "top", "u": u, "v": v})
            u += ps[k] if k < n - 1 else 0
        if valid(spec):
            break
    else:
        raise RuntimeError
    last = a + sum(ps)
    read = [("the overall length of the plate", L, "The overall length is dimensioned %s."), ("the overall width of the plate", W, "The width is dimensioned %s."),
            ("the thickness of the plate", T, "The front view gives the thickness as %s."), ("the diameter of the holes", d, "The holes are called out as %s."),
            ("the distance from the left edge of the plate to the centre of the first (leftmost) hole", a, "The first dimension of the chain is %s.")]
    chain = [("the distance from the left edge of the plate to the centre of the last (rightmost) hole", last,
              "Add the chain: " + " + ".join(["%s"] * n) + " = %s.", [a] + ps),
             ("the distance from the centre of the last (rightmost) hole to the right edge of the plate", L - last,
              "Overall length minus the chain: %s − (" + " + ".join(["%s"] * n) + ") = %s.", [L, a] + ps)]
    if n >= 3:
        chain.append(("the distance between the centres of the first and the last hole", sum(ps), " + ".join(["%s"] * (n - 1)) + " = %s.", ps))
    noul = [("Is the hole diameter larger than the plate thickness?", d, T, "The holes are %s in diameter; the plate is %s thick."),
            ("Is the last hole closer to the right edge than the first hole is to the left edge?", a, L - last,
             "The first hole is %s from the left edge; the last is %s from the right edge (overall length minus the chain)."),
            ("Is the plate more than twice as long as it is wide?", L, 2 * W, "The length is %s; twice the width is %s.")]
    return spec, {"read": read, "chain": chain, "noul": noul, "all": [L, W, T, d, a] + ps, "views": "top+front", "chain_ix": list(range(n))}


def _dr_flange(rng):
    for _ in range(300):
        spec = new_flange(rng, hub=rng.choice([14, 18, 22, 26, 32]))
        OD, T, hd, hh, bd, pcd, bo = spec["OD"], spec["T"], spec["hub_d"], spec["hub_h"], spec["bore_d"], spec["pcd"], spec["bolt_d"]
        vals = [OD, T, hd, T + hh, bd, pcd, bo]
        if len(set(vals)) == len(vals) and (hd - bd) / 2 != T and bd != hh:
            break
    else:
        raise RuntimeError
    H = T + hh
    read = [("the outside diameter of the flange", OD, "The outside diameter is dimensioned %s."), ("the diameter of the centre bore", bd, "The bore is dimensioned %s."),
            ("the diameter of the hub", hd, "The hub is dimensioned %s."), ("the bolt circle diameter", pcd, "The bolt circle (PCD) is given as %s."),
            ("the thickness of the flange plate", T, "The flange plate is %s thick."), ("the overall height of the part", H, "The overall height is dimensioned %s."),
            ("the diameter of the bolt holes", bo, "The bolt holes are called out as %s.")]
    chain = [("the radial wall thickness of the hub (between the bore and the outside of the hub)", (hd - bd) / 2, "(%s − %s) / 2 = %s.", [hd, bd]),
             ("the height of the hub above the flange face", hh, "Overall height minus flange thickness: %s − %s = %s.", [H, T]),
             ("the radial distance from the bolt circle to the outside edge of the flange", (OD - pcd) / 2, "(%s − %s) / 2 = %s.", [OD, pcd])]
    noul = [("Is the bore diameter larger than the height of the hub above the flange face?", bd, hh,
             "The bore is %s in diameter; the hub stands %s above the flange face (overall height minus flange thickness)."),
            ("Is the bolt circle diameter more than twice the bore diameter?", pcd, 2 * bd, "The bolt circle is %s; twice the bore is %s."),
            ("Is the hub wall (radial, bore to outside of hub) thicker than the flange plate?", (hd - bd) / 2, T,
             "The hub wall is %s (half of hub minus bore); the flange plate is %s thick."),
            ("Is the hub diameter more than half the outside diameter of the flange?", hd, OD / 2,
             "The hub is %s in diameter; half the outside diameter is %s.")]
    return spec, {"read": read, "chain": chain, "noul": noul, "all": vals, "views": "front+top"}


def _dr_bracket(rng):
    for _ in range(300):
        spec = new_bracket(rng, "L")
        spec["rin"], spec["rout"] = rng.choice([0, 0, 5]), False
        d = rng.choice([6, 8, 10, 12])
        L, W, H, T = spec["L"], spec["W"], spec["H"], spec["T"]
        if len({L, W, H, T, d}) == 5 and put(rng, spec, "base", rng.randint(1, 2), hole(d)) and put(rng, spec, "leg0", rng.randint(0, 1), hole(d)):
            break
    else:
        raise RuntimeError
    read = [("the overall length of the bracket base", L, "The base is dimensioned %s long."), ("the overall height of the bracket", H, "The height is dimensioned %s."),
            ("the width of the bracket (front to back)", W, "The plan view gives the width as %s."), ("the material thickness of the bracket", T, "The thickness is dimensioned %s."),
            ("the diameter of the holes in the base", d, "The base holes are called out as %s.")]
    chain = [("the height of the upright leg above the top face of the base", H - T, "Overall height minus thickness: %s − %s = %s.", [H, T]),
             ("the length of the base measured from the inside face of the upright leg", L - T, "Overall length minus thickness: %s − %s = %s.", [L, T])]
    noul = [("Is the bracket taller than its base is long?", H, L, "The height is %s; the base is %s long."),
            ("Is the bracket wider (front to back) than it is tall?", W, H, "The width is %s; the height is %s."),
            ("Is the hole diameter larger than the material thickness?", d, T, "The holes are %s in diameter; the material is %s thick.")]
    return spec, {"read": read, "chain": chain, "noul": noul, "all": [L, W, H, T, d], "views": "front+top"}


DR_FAMS = [_dr_shaft, _dr_plate, _dr_flange, _dr_bracket]
UNIT_NAME = {"mm": "millimetres", "in": "inches"}


def _ufmt(v, unit):
    return ("%g" % round(v, 2 if unit == "mm" else 3)) + " " + unit


def make_dimension_read(rng, render=True):
    rk = rng.randrange(4)
    for _ in range(400):
        fam = rng.choice(DR_FAMS)
        r = rng.random()
        mode = "plain" if r < 0.70 else "trap" if r < 0.85 else "twin"  # trap: the title block is in the other unit
        noul = rng.random() < 0.3
        if mode == "plain":
            ud = "mm" if rng.random() < 0.8 else "in"
            uq = ud
        else:
            uq = "mm" if rng.random() < 0.7 else "in"
            ud = uq if mode == "twin" else ("in" if uq == "mm" else "mm")
        small = mode != "plain" or ud == "in" or rng.random() < 0.15
        num = Num(0.1 if small else 1.0, rng.choice([None, None, 1, 2]) if small else rng.choice([None, None, None, 1]))
        conv = MM_PER_INCH if uq == "mm" else 1 / MM_PER_INCH  # printed number -> asked unit, when the units differ
        want = rng.random() < 0.5
        spec = facts = None
        if noul and mode == "plain":
            k = rng.randrange(4)
            for _ in range(80):  # resample the part until this question has the wanted answer: balanced per question
                try:
                    spec, facts = fam(rng)
                except RuntimeError:
                    break
                qn = facts["noul"][k % len(facts["noul"])]
                if qn[1] != qn[2] and (qn[1] > qn[2]) == want:
                    break
            else:
                continue
            if facts is None:
                continue
            q, label = _noul(qn[0]), int(want)
            why = qn[3] % (num.s(qn[1]), num.s(qn[2])) + (" So yes." if want else " So no.")
            kind, gold = "compare", None
        else:
            try:
                spec, facts = fam(rng)
            except RuntimeError:
                continue
            chain = rng.random() < 0.45
            item = rng.choice(facts["chain"] if chain else facts["read"])
            phrase, val = item[0], item[1]
            a = num.val(val)
            terms = tuple(num.s(t) for t in item[3]) if chain else ()
            base_why = item[2] % (terms + (num.s(val),))
            kind = ("chain" if chain else "read") + ("" if mode == "plain" else "_unit_" + mode)
            if noul:  # threshold between the two readings of the printed number: only the title block decides
                lo, hi = sorted((a, a * conv))
                thr = float("%.2g" % math.sqrt(lo * hi))
                if not lo * 1.5 < thr < hi / 1.5:
                    continue
                q = _noul("Is %s greater than %s?" % (phrase, _ufmt(thr, uq)))
                true = a * conv if ud != uq else a
                label = int(true > thr)
                why = "%s The title block gives the units as %s, so that is %s: %s than %s." % (
                    base_why, UNIT_NAME[ud], _ufmt(true, uq), "more" if label else "not more", _ufmt(thr, uq))
                kind, gold = "threshold_unit_" + mode, true
            elif mode == "plain":
                others = [num.val(x) for x in facts["all"] if num.val(x) != a]
                qq = _num_choice("What is %s?" % phrase, a, others, lambda v: _ufmt(v, ud), rng,
                                 [round(a * (1 + f), 1 if small else 0) - a for f in (-0.4, -0.25, -0.12, 0.12, 0.25, 0.45)], lo=num.val(2), r=rk)
                if qq is None:
                    continue
                q, label = qq
                why, gold = base_why, a
            else:
                pool = [num.val(x) for x in facts["all"] if num.val(x) != a] + [round(a * f, 1) for f in (0.6, 0.8, 1.25, 1.6)]
                pool = [x for x in pool if x != a and a / 8 <= x <= a * 8]  # so the two raw and the two converted values stay apart
                if not pool:
                    continue
                b = rng.choice(pool)
                gold = a * conv if mode == "trap" else a
                opts = [_ufmt(x, uq) for x in (a, b, a * conv, b * conv)]
                if len(set(opts)) < 4:
                    continue
                q, label = _mc("What is %s, in %s?" % (phrase, UNIT_NAME[uq]), opts, 2 if mode == "trap" else 0, rng)
                if mode == "trap":
                    why = "%s The title block gives the units as %s: %s %s %s = %s." % (
                        base_why, UNIT_NAME[ud], num.s(val), "×" if uq == "mm" else "÷", "25.4", _ufmt(gold, uq))
                else:
                    why = "%s The title block gives the units as %s, so no conversion is needed: %s." % (base_why, UNIT_NAME[ud], _ufmt(gold, uq))
        srng = random.Random(rng.random())
        views = facts["views"]

        def image():
            sh = new_sheet(srng, 896, 672)
            st = sh.st
            names = views.split("+")
            angle = srng.choice(["third", "third", "first"])
            flange = spec["fam"] == "flange"
            vs = {}
            for nme in names:
                if flange and nme == "front" and srng.random() < 0.65:
                    vs[nme] = part_view(spec, "front", cut=0.0)
                else:
                    vs[nme] = part_view(spec, nme, centre=st["centre"] or (flange and nme == "top"))
            P = layout(sh, vs, arrange(names, angle), (20, 74, 876, 652), padx=80, pady=70 if spec["fam"] != "shaft" else 90)
            if spec["fam"] == "plate":
                dim_plate(sh, spec, P, num, srng, chain=facts["chain_ix"], counts=True)
            elif spec["fam"] == "shaft":
                dim_shaft(sh, spec, P, num, overall=False)
            elif flange:
                dim_flange(sh, spec, P, num, srng)
            else:
                dim_bracket(sh, spec, P, num, srng)
            sh.title_block(title_cells(srng, spec["fam"], units=ud, angle=angle))
            return sh.image()

        return {"q": q, "label": label, "why": why, "kind": kind, "image": image if render else None,
                "meta": {"spec": spec, "units": ud, "ask_units": uq, "mode": mode, "scale": num.scale, "gold": gold,
                         "phrase": q["instructions"]}}
    raise RuntimeError("dimension_read")


# ---- build -----------------------------------------------------------------------------------------------------------
MAKERS = {"cad_feature_count": make_feature_count, "cad_view_match": make_view_match, "cad_dimension_read": make_dimension_read,
          "cad_drawing_pair": make_drawing_pair, "cad_section_depth": make_section_depth, "cad_assembly": make_assembly}
SIZES = {n: (4000, 400) for n in MAKERS}
SIZES["cad_assembly"] = (2000, 200)
USE = {"cad_feature_count": "count features on a drawing or render", "cad_view_match": "match an isometric view to its orthographic / section view",
       "cad_dimension_read": "read, add and compare dimensions; unit traps", "cad_drawing_pair": "same part or not, and what changed",
       "cad_section_depth": "read hole depth and type from a section", "cad_assembly": "fasteners, mating parts, assembly order"}


def make_row(name, split, i, render=True, salt=0):
    """Row ``i`` of a split: deterministic in (name, split, i, salt). Returns the maker's dict (``image`` is a
    callable that draws the PIL image, or None with ``render=False``)."""
    return MAKERS[name](random.Random("cad-%s-%s-%d-%d" % (name, split, i, salt)), render)


def _work(job):
    name, split, i, out_dir, salt = job
    for k in range(6):  # a drawing the renderer rejects is redrawn from the next salt (still deterministic)
        try:
            r = make_row(name, split, i, True, salt + k)
            im = r["image"]()
            break
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    else:
        raise RuntimeError("row %s %s %d: %s" % (name, split, i, err))
    rel = "images/%s-%06d.png" % (split, i)
    im.save(os.path.join(out_dir, rel))
    with open(os.path.join(out_dir, rel), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    rec = {"id": "%s-%s-%d" % (name, split, i), "image": rel, "state_text": None, "question": r["q"], "label": r["label"],
           "rationale": r["why"], "kind": r["kind"], "image_license": "generated (VisionJev)"}
    return split, i, rec, sha, salt + k


def text_baseline(train, val):
    """Question-only probes (no image): per question template (digits masked), the best train-set guess of the gold
    by (a) its rank among the options sorted by value, (b) its option text, (c) for noul, the majority answer."""
    import re

    def tmpl(r):
        return re.sub(r"\d+(\.\d+)?", "#", r["question"]["instructions"])

    def ranks(r):
        o = [str(x) for x in r["question"]["criteria"]]
        try:
            v = [float(re.sub(r"[^0-9.\-]", "", x)) for x in o]
            order = sorted(range(len(o)), key=lambda i: v[i])
            return [order.index(i) for i in range(len(o))]
        except ValueError:
            return list(range(len(o)))

    by_rank, by_text, by_noul = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for r in train:
        t = tmpl(r)
        if r["question"]["type"] == "noul":
            by_noul[t][r["label"]] += 1
        else:
            by_rank[(t, len(r["question"]["criteria"]))][ranks(r)[r["label"]]] += 1
            by_text[t][re.sub(r"\d+(\.\d+)?", "#", str(r["question"]["criteria"][r["label"]]))] += 1
    out = {}
    ch = [r for r in val if r["question"]["type"] == "choice"]
    if ch:
        hr = ht = 0
        for r in ch:
            t, rk = tmpl(r), ranks(r)
            c = by_rank[(t, len(rk))]
            hr += max(range(len(rk)), key=lambda i: (c[rk[i]], -i)) == r["label"]
            c2 = by_text[t]
            o = [re.sub(r"\d+(\.\d+)?", "#", str(x)) for x in r["question"]["criteria"]]
            ht += max(range(len(o)), key=lambda i: (c2[o[i]], -i)) == r["label"]
        out["choice"] = {"n": len(ch), "chance": round(sum(1 / len(r["question"]["criteria"]) for r in ch) / len(ch), 3),
                         "rank_prior": round(hr / len(ch), 3), "option_text_prior": round(ht / len(ch), 3)}
    nl = [r for r in val if r["question"]["type"] == "noul"]
    if nl:
        hit = sum((by_noul[tmpl(r)].most_common(1)[0][0] if by_noul[tmpl(r)] else 1) == r["label"] for r in nl)
        out["noul"] = {"n": len(nl), "majority": round(max(Counter(r["label"] for r in nl).values()) / len(nl), 3),
                       "template_prior": round(hit / len(nl), 3)}
    return out


def build_dataset(name, root=DATA_ROOT, n_train=None, n_val=None, workers=8, log=print):
    from multiprocessing import get_context

    n_train = SIZES[name][0] if n_train is None else n_train
    n_val = SIZES[name][1] if n_val is None else n_val
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    jobs = [(name, "train", i, tmp, 0) for i in range(n_train)] + [(name, "val", i, tmp, 0) for i in range(n_val)]
    rows, shas, t0 = {"train": {}, "val": {}}, {}, time.time()
    with get_context("spawn").Pool(min(workers, 8)) as pool:  # spawn: a fork after the kernel has run can deadlock
        pending = jobs
        while pending:
            redo = []
            for k, (split, i, rec, sha, salt) in enumerate(pool.imap_unordered(_work, pending, chunksize=8)):
                if sha in shas and shas[sha] != (split, i):  # identical bytes already used: draw this row again
                    redo.append((name, split, i, tmp, salt + 1))
                    continue
                shas[sha] = (split, i)
                rows[split][i] = rec
                if (k + 1) % 500 == 0:
                    log("%s: %d / %d rows (%.0f s)" % (name, k + 1, len(pending), time.time() - t0))
            pending = redo
    out = {s: [rows[s][i] for i in sorted(rows[s])] for s in rows}
    for split, rs in out.items():
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import cadquery
    import matplotlib

    manifest = {"dataset": name, "source": "generated (laya.visionjev_cad): parametric parts, labels computed from the parameters",
                "revision": "v1", "use": USE[name],
                "license": {"annotations": "generated in-house, exact labels",
                            "images": "generated (CadQuery / OpenCascade hidden-line and section geometry, matplotlib rendering)",
                            "commercial": True, "verified": "generated in-house"},
                "tools": {"cadquery": cadquery.__version__, "matplotlib": matplotlib.__version__, "fonts": sorted(set(FONTS))},
                "seed": "random.Random('cad-<name>-<split>-<i>-<salt>')", "counts": {s: len(rs) for s, rs in out.items()},
                "kinds": {s: dict(Counter(r["kind"] for r in rs)) for s, rs in out.items()},
                "label_counts": {s: {str(k): v for k, v in Counter(r["label"] for r in rs).items()} for s, rs in out.items()},
                "question_only": text_baseline(out["train"], out["val"]), "unique_image_hashes": len(shas),
                "validator": {"skip_content_leakage": True}}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: %s in %.0f s; question-only %s" % (name, manifest["counts"], time.time() - t0, manifest["question_only"]))
    return manifest


# ---- second priority: circuit_component_count from CGHD (human hand-drawn circuits, CC BY 4.0) ---------------------
CGHD_ZIP = os.path.join(os.path.dirname(DATA_ROOT), "raw", "cghd", "cghd-zenodo-16.zip")
CGHD_CITE = {"source": "CGHD: A Public Ground-Truth Dataset for Handwritten Circuit Diagram Images (Zenodo record 17469897, v16)",
             "doi": "10.5281/zenodo.17469897", "author": "Johannes Bayer et al. (DFKI)", "license": "CC BY 4.0",
             "modified": "photo downscaled to 896 px; counts derived from the human bounding-box annotations"}
CGHD_GROUPS = {
    "resistors (fixed, adjustable or photo-resistors)": ("resistor", "resistor.adjustable", "resistor.photo"),
    "capacitors (polarized, unpolarized or adjustable)": ("capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable"),
    "inductors or coils": ("inductor", "inductor.ferrite"), "diodes (including LEDs and Zener diodes)": ("diode", "diode.light_emitting", "diode.zener", "diode.thyrector"),
    "transistors (BJT, FET or photo)": ("transistor.bjt", "transistor.fet", "transistor.photo"), "switches": ("switch",),
    "voltage sources or batteries": ("voltage.dc", "voltage.ac", "voltage.battery"),
    "integrated circuits (chips)": ("integrated_circuit", "integrated_circuit.voltage_regulator", "integrated_circuit.ne555"),
    "operational amplifiers": ("operational_amplifier", "operational_amplifier.schmitt_trigger"), "transformers": ("transformer",),
    "lamps": ("lamp",), "loudspeakers": ("speaker",), "fuses": ("fuse",), "motors": ("motor",), "relays": ("relay",), "crystals": ("crystal",),
    "thyristors, triacs or diacs": ("thyristor", "triac", "diac"), "logic gates (AND, OR, NOT, NAND, NOR, XOR)": ("and", "or", "not", "nand", "nor", "xor"),
    "ground symbols": ("gnd",), "microphones": ("microphone",),
}
CGHD_VAL_DRAFTERS = {"28", "29", "30", "31"}


def cghd_index(zip_path=CGHD_ZIP):
    """[(image path in the zip, drafter, {class: count})] from every annotation XML in the archive."""
    import xml.etree.ElementTree as ET
    import zipfile

    out = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for n in sorted(names):
            if "/annotations/" not in n or not n.endswith(".xml"):
                continue
            root = ET.fromstring(z.read(n))
            fn = root.findtext("filename")
            img = n.split("/annotations/")[0] + "/images/" + fn
            if img not in names:
                cands = [c for c in names if c.startswith(n.split("/annotations/")[0] + "/images/" + os.path.splitext(fn)[0] + ".")]
                if not cands:
                    continue
                img = cands[0]
            counts = Counter(o.findtext("name") for o in root.findall("object"))
            out.append((img, n.split("/")[0].replace("drafter_", ""), dict(counts)))
    return out


def group_counts(counts):
    return {g: sum(counts.get(c, 0) for c in cs) for g, cs in CGHD_GROUPS.items()}


def make_circuit_rows(item, rng, freq=None):
    """One or two questions for one photo: a count (rank-uniform options), a presence check or a which-type.
    ``freq``: how often each group is present over the whole set; rare groups are asked about (and offered as
    options) in inverse proportion, so the option text alone does not tell which type is the present one."""
    img, drafter, counts = item
    gc = group_counts(counts)
    freq = freq or {g: 0.5 for g in CGHD_GROUPS}
    w_rare = lambda gs: [1 / max(freq[g], 0.02) for g in gs]  # noqa: E731
    w_common = lambda gs: [max(freq[g], 0.02) for g in gs]  # noqa: E731
    rows = []
    kinds = ["count", rng.choice(["present", "which"])]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "count":
            r = rng.randrange(4)
            pool = [g for g, c in gc.items() if 1 <= c <= 12] + ([g for g, c in gc.items() if c == 0][:0])
            zeros = [g for g in ("inductors or coils", "transformers", "switches", "transistors (BJT, FET or photo)") if gc[g] == 0]
            if zeros and rng.random() < 0.12:
                pool = zeros
            if not pool:
                continue
            for _ in range(20):
                g = rng.choice(pool)
                gold = gc[g]
                mistakes = [gc[o] for o in gc if o != g and gc[o] != gold] + [sum(gc.values())]
                q = _num_choice("How many %s are drawn in this circuit? Count every symbol of that kind." % g, gold, mistakes,
                                lambda v: "%d" % int(v), rng, [-3, -2, -1, 1, 2, 3], lo=0, r=r)
                if q is not None:
                    break
            if q is None:
                continue
            parts = ", ".join("%d %s" % (counts.get(c, 0), c) for c in CGHD_GROUPS[g] if counts.get(c, 0))
            why = "The annotation marks %d %s%s." % (gold, g.split(" (")[0], (" (%s)" % parts) if parts else "")
            rows.append((q[0], q[1], why, "count"))
        elif kind == "present":
            want = rng.random() < 0.5
            pool = [g for g, c in gc.items() if (c > 0) == want and g not in ("ground symbols",)]
            if not pool:
                continue
            g = rng.choices(pool, w_rare(pool) if want else w_common(pool))[0]
            q = _noul("Does this circuit contain at least one of: %s?" % g)
            rows.append((q, int(want), "The annotation marks %d %s." % (gc[g], g.split(" (")[0]), "present"))
        else:
            present = [g for g, c in gc.items() if c > 0 and g != "ground symbols"]
            absent = [g for g, c in gc.items() if c == 0]
            if not present or len(absent) < 3:
                continue
            g = rng.choices(present, w_rare(present))[0]
            opts, w = [g], w_common(absent)
            while len(opts) < 4:
                o = rng.choices(absent, w)[0]
                if o not in opts:
                    opts.append(o)
            q, lab = _mc("Which of these component types appears in the drawing?", opts, 0, rng)
            rows.append((q, lab, "The drawing has %d %s; none of the other three types is drawn." % (gc[g], g.split(" (")[0]), "which"))
    return rows


def _circuit_work(job):
    import io
    import zipfile

    from PIL import Image, ImageOps

    zip_path, item, out_dir, rel = job
    with zipfile.ZipFile(zip_path) as z:
        with Image.open(io.BytesIO(z.read(item[0]))) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((896, 896))
            im.save(os.path.join(out_dir, rel), "JPEG", quality=86)
    return rel


def build_circuits(root=DATA_ROOT, zip_path=CGHD_ZIP, workers=8, log=print, name="circuit_component_count", seed=0):
    from multiprocessing import get_context

    items = cghd_index(zip_path)
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    jobs, rel_of = [], {}
    for k, it in enumerate(items):
        rel_of[it[0]] = "images/%s.jpg" % it[0].replace("/images/", "-").rsplit(".", 1)[0].replace("/", "_")
        jobs.append((zip_path, it, tmp, rel_of[it[0]]))
    with get_context("spawn").Pool(min(workers, 8)) as pool:
        for k, _ in enumerate(pool.imap_unordered(_circuit_work, jobs, chunksize=4)):
            if (k + 1) % 500 == 0:
                log("%s: %d / %d photos" % (name, k + 1, len(jobs)))
    rng = random.Random(seed)
    freq = {g: sum(1 for it in items if group_counts(it[2])[g] > 0) / len(items) for g in CGHD_GROUPS}
    rows, kinds = {"train": [], "val": []}, {"train": Counter(), "val": Counter()}
    for it in items:
        split = "val" if it[1] in CGHD_VAL_DRAFTERS else "train"
        for q, lab, why, kind in make_circuit_rows(it, rng, freq):
            rows[split].append({"id": "%s-%s-%d" % (name, split, len(rows[split])), "image": rel_of[it[0]], "state_text": None, "question": q,
                                "label": lab, "rationale": why, "kind": kind, "image_license": "CC BY 4.0",
                                "attribution": dict(CGHD_CITE, file=it[0])})
            kinds[split][kind] += 1
    for split, rs in rows.items():
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"dataset": name, "source": CGHD_CITE["source"], "revision": "zenodo v16 (2025-10-25)", "doi": CGHD_CITE["doi"],
                "use": "count / detect component types in human hand-drawn circuit photos (human bounding boxes)",
                "license": {"annotations": "CC BY 4.0 (CGHD human bounding boxes; questions derived exactly from them, in-house)",
                            "images": "CC BY 4.0 (CGHD photos, downscaled)", "commercial": True, "attribution": CGHD_CITE,
                            "verified": "Zenodo record metadata license id cc-by-4.0, read 2026-09-30"},
                "split": "by drafter: val = drafters %s (all photos of one drawing share a drafter)" % sorted(CGHD_VAL_DRAFTERS),
                "counts": {s: len(r) for s, r in rows.items()}, "kinds": {s: dict(c) for s, c in kinds.items()},
                "label_counts": {s: {str(k): v for k, v in Counter(r["label"] for r in rs).items()} for s, rs in rows.items()},
                "question_only": text_baseline(rows["train"], rows["val"]), "photos": len(items)}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: %s; question-only %s" % (name, manifest["counts"], manifest["question_only"]))
    return manifest



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="dataset names from MAKERS, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--train", type=int, default=None)
    ap.add_argument("--val", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for n in (list(MAKERS) if a.names == ["all"] else a.names):
        if n == "circuit_component_count":
            build_circuits(a.root, workers=a.workers, log=lambda m: print(m, flush=True))
        else:
            build_dataset(n, a.root, a.train, a.val, a.workers, log=lambda m: print(m, flush=True))