"""Procedural game-screen image sets with exact, rule-computed labels (decisions that need the picture).

Every row is one rendered game position (PIL, free fonts, no external assets) and one typed question whose answer is
computed from the position by the rules of the game: line search (tic-tac-toe, Connect Four), breadth-first search
(mazes), enumeration of every consistent mine layout (Minesweeper), the slide-and-merge rule (2048), one- and two-move
simulation (Snake) and ray casting with blockers (chess). The position is never given as text (``state_text`` is
null); the only text inside an image is what the game itself shows: cell numbers, column numbers, tile values, mine
counts, option letters, S / E / K / D marks, file and rank labels, piece letters.

Datasets (``<root>/<name>/{train,val}.jsonl``, JPEG images under ``images/``):

* ``game_tictactoe_move``   (choice, 4) 3x3 or 4x4 board with numbered cells: the one cell that wins for X now, or the
                            one cell X must take to stop O. Hard rows: O threatens too (the winning move beats the
                            block), dead near-miss lines.
* ``game_connect4_win``     (noul) can the side to move connect four with one disc? (choice, 4) which column wins.
                            Positions come from legal alternating play under gravity. Hard rows: three in a row whose
                            fourth cell floats above an empty cell, the opponent's winning column, diagonal-only wins.
* ``game_maze_path``        (choice) shortest route length (rank-uniform options), first move of the shortest route,
                            shortest route with a key and a locked door; (noul) is the exit reachable?
* ``game_minesweeper_safe`` (choice A-D) which lettered covered cell is certainly a mine / certainly safe; (noul) is
                            cell A certainly a mine / safe? Forced cells come from enumerating every mine layout that
                            agrees with the numbers; hard rows cannot be read off one number.
* ``game_2048_move``        (choice) which of two moves merges more points, how many points a move scores, which move
                            leaves the board unchanged; (noul) does this move merge anything? (Left and right always
                            merge the same total, so "the best move" is asked between one horizontal and one vertical
                            move.)
* ``game_snake_safe``       (choice) the safe move that ends nearest the food, how many of the four moves are fatal;
                            (noul) is this move safe? after this move, is any safe move left (two-step)?
* ``game_chess_attack``     (noul) is the black king in check? (choice) which white piece attacks a named square, how
                            many white pieces attack the marked square. Kings, queens, rooks, bishops, knights only;
                            pieces block lines. Hard rows have a slider lined up with the square but blocked.

Balance: the kind of question, the wanted answer (yes / no, gold direction, gold count, gold piece) and the difficulty
are fixed by the row index, so every set is balanced by construction; option order is shuffled (gold position uniform)
and numeric options put the gold at a uniformly drawn rank. A row whose position was already used (in either split) is
drawn again from the next salt, so train and val never share a position.

    python -m laya.visionjev_games [--only name ...] [--root DIR] [--train N --val N]
"""
import argparse
import colorsys
import functools
import json
import os
import random
import shutil
import time
from collections import Counter, deque

from .visionjev_cad import text_baseline
from .visionjev_data import DATA_ROOT
from .visionjev_synth import _choice, _font, _noul

MAX_SIDE = 768
SS = 2  # every sheet is drawn at twice its size and reduced (smooth edges)
SIZES = (4000, 300)
LICENSE = {"annotations": "generated (VisionJev, exact rule-based labels)", "images": "rendered in-house with PIL and free fonts",
           "commercial": True, "verified": "generated in-house"}
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}  # (dx, dy), y grows downwards
DIR_NAMES = list(DIRS)

THEMES = [  # bg: sheet; panel: board / floor; line: grid lines; ink / dim: text; c1 / c2: the two sides; wall: blocks
    dict(bg=(246, 244, 238), panel=(255, 255, 255), line=(70, 70, 78), ink=(24, 24, 28), dim=(128, 128, 136),
         c1=(205, 52, 60), c2=(40, 96, 190), wall=(58, 62, 74), light=(238, 226, 200), dark=(160, 118, 84)),
    dict(bg=(30, 32, 40), panel=(92, 98, 116), line=(190, 194, 206), ink=(244, 244, 248), dim=(196, 200, 212),
         c1=(255, 150, 110), c2=(130, 215, 255), wall=(12, 12, 16), light=(196, 202, 214), dark=(104, 118, 150)),
    dict(bg=(228, 238, 244), panel=(246, 250, 252), line=(46, 82, 110), ink=(16, 40, 60), dim=(100, 130, 150),
         c1=(226, 120, 30), c2=(20, 130, 120), wall=(36, 70, 100), light=(222, 232, 238), dark=(110, 150, 178)),
    dict(bg=(240, 232, 214), panel=(252, 246, 230), line=(96, 72, 48), ink=(52, 36, 20), dim=(140, 118, 92),
         c1=(170, 40, 90), c2=(30, 110, 60), wall=(92, 64, 40), light=(240, 217, 181), dark=(181, 136, 99)),
    dict(bg=(236, 240, 232), panel=(250, 252, 246), line=(60, 90, 60), ink=(22, 44, 26), dim=(110, 134, 110),
         c1=(196, 60, 160), c2=(40, 60, 170), wall=(50, 84, 56), light=(234, 236, 208), dark=(118, 150, 86)),
    dict(bg=(255, 255, 255), panel=(238, 238, 242), line=(20, 20, 20), ink=(0, 0, 0), dim=(110, 110, 116),
         c1=(0, 0, 0), c2=(214, 40, 40), wall=(40, 40, 46), light=(232, 232, 236), dark=(148, 150, 160)),
]


# ---- drawing ------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _f(size, bold, family):
    return _font(size, bold, family)


def _ink_on(rgb):
    """Dark or light text, whichever reads on this fill."""
    return (30, 30, 34) if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 150 else (250, 250, 250)


def _mix(a, b, t):
    return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


class Sheet:
    """A canvas in final pixels: shapes are drawn at SS times the size and the sheet is reduced once at the end."""

    def __init__(self, w, h, bg):
        from PIL import Image, ImageDraw

        self.size = (int(w), int(h))
        self.im = Image.new("RGB", (self.size[0] * SS, self.size[1] * SS), bg)
        self.d = ImageDraw.Draw(self.im)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0, r=0.0):
        box = [x0 * SS, y0 * SS, x1 * SS, y1 * SS]
        w = max(1, int(round(width * SS)))
        if r > 0:
            self.d.rounded_rectangle(box, radius=r * SS, fill=fill, outline=outline, width=w)
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=w)

    def circle(self, cx, cy, rad, fill=None, outline=None, width=1.0):
        self.d.ellipse([(cx - rad) * SS, (cy - rad) * SS, (cx + rad) * SS, (cy + rad) * SS], fill=fill, outline=outline,
                       width=max(1, int(round(width * SS))))

    def line(self, pts, fill, width=1.0):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=max(1, int(round(width * SS))))

    def poly(self, pts, fill, outline=None):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=fill, outline=outline)

    def text(self, x, y, s, size, fill, bold=True, family=0, anchor="mm", stroke=0.0, stroke_fill=None):
        self.d.text((x * SS, y * SS), s, font=_f(max(6, int(round(size * SS))), bold, family), fill=fill, anchor=anchor,
                    stroke_width=int(round(stroke * SS)), stroke_fill=stroke_fill)

    def done(self):
        from PIL import Image

        return self.im.resize(self.size, Image.LANCZOS)


def _layout(rng, cols, rows, lo, hi, pad=(0, 0, 0, 0)):
    """Random cell size and margins for a cols x rows grid, long side <= MAX_SIDE. ``pad``: extra room (left, top,
    right, bottom) for labels. Returns the cell size, the grid origin and the sheet size."""
    m = [rng.randint(14, 54) + pad[k] for k in range(4)]
    cmax = min((MAX_SIDE - m[0] - m[2]) // cols, (MAX_SIDE - m[1] - m[3]) // rows, hi)
    cell = rng.randint(min(lo, cmax), cmax)
    return dict(cell=cell, x0=m[0], y0=m[1], w=m[0] + cols * cell + m[2], h=m[1] + rows * cell + m[3])


def _style(rng, cols, rows, lo, hi, pad=(0, 0, 0, 0)):
    return dict(_layout(rng, cols, rows, lo, hi, pad), th=rng.choice(THEMES), family=rng.randrange(4), v=rng.randrange(1 << 16))


def _num_options(gold, rng, named=(), lo=1, step=1, k=4, span=4):
    """k distinct numeric options with the gold at a uniformly drawn rank (so its rank says nothing); the named
    mistakes are offered first on their side of the gold. None if the gold is too small to take every rank."""
    if gold - (k - 1) * step < lo:
        return None
    r = rng.randrange(k)  # how many options are smaller than the gold
    out = [gold]
    for side, n in ((-1, r), (1, k - 1 - r)):
        first = [v for v in dict.fromkeys(named) if (v - gold) * side > 0 and v >= lo and (v - gold) % step == 0]
        rng.shuffle(first)
        pool = [gold + side * step * d for d in range(1, span + 1) if gold + side * step * d >= lo]
        rng.shuffle(pool)
        picks = []
        for v in first[:max(1, n - 1)] + pool:
            if v not in picks and len(picks) < n:
                picks.append(v)
        if len(picks) < n:
            return None
        out += picks
    return out


def _ranked(gold, others, rng, score, k=4):
    """k - 1 of the other candidates (cell or column numbers), the most tempting first, such that the gold's rank among
    the k options is drawn uniformly: the numbers offered say nothing about which one is right. None if impossible."""
    r = rng.randrange(k)
    scored = sorted(others, key=score, reverse=True)
    lo, hi = [c for c in scored if c < gold], [c for c in scored if c > gold]
    return lo[:r] + hi[:k - 1 - r] if len(lo) >= r and len(hi) >= k - 1 - r else None


# ---- tic-tac-toe --------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def ttt_lines(n):
    """Rows, columns and the two diagonals of an n x n board (cell index = row * n + column)."""
    L = [tuple(r * n + c for c in range(n)) for r in range(n)] + [tuple(r * n + c for r in range(n)) for c in range(n)]
    return L + [tuple(k * n + k for k in range(n)), tuple(k * n + n - 1 - k for k in range(n))]


def ttt_won(board, n, p):
    return any(all(board[j] == p for j in l) for l in ttt_lines(n))


def ttt_wins(board, n, p):
    """Empty cells that complete a full line for player p."""
    out = set()
    for l in ttt_lines(n):
        empty = [j for j in l if board[j] is None]
        if len(empty) == 1 and all(board[j] == p for j in l if j != empty[0]):
            out.add(empty[0])
    return sorted(out)


def ttt_dead(board, n):
    """Lines that hold n - 1 marks of one player and one of the other (a near miss that is already stopped)."""
    out = []
    for l in ttt_lines(n):
        c = Counter(board[j] for j in l)
        if (c["X"], c["O"]) in ((n - 1, 1), (1, n - 1)):
            out.append(l)
    return out


def draw_ttt(st, sty):
    n, th, c, x0, y0 = st["n"], sty["th"], sty["cell"], sty["x0"], sty["y0"]
    sh = Sheet(sty["w"], sty["h"], th["bg"])
    sh.rect(x0, y0, x0 + n * c, y0 + n * c, fill=th["panel"])
    lw = max(2.0, c * (0.03 + 0.02 * (sty["v"] % 3)))
    for k in range(n + 1):
        if 0 < k < n or sty["v"] % 2:  # with or without the outer frame
            sh.line([(x0 + k * c, y0), (x0 + k * c, y0 + n * c)], th["line"], lw)
            sh.line([(x0, y0 + k * c), (x0 + n * c, y0 + k * c)], th["line"], lw)
    for j, p in enumerate(st["board"]):
        cx, cy = x0 + (j % n + 0.5) * c, y0 + (j // n + 0.5) * c
        sh.text(x0 + (j % n) * c + c * 0.14, y0 + (j // n) * c + c * 0.15, str(j + 1), c * 0.17, th["dim"], family=sty["family"])
        r, w = c * 0.27, max(3.0, c * 0.09)
        if p == "X":
            sh.line([(cx - r, cy - r), (cx + r, cy + r)], th["c1"], w)
            sh.line([(cx - r, cy + r), (cx + r, cy - r)], th["c1"], w)
        elif p == "O":
            sh.circle(cx, cy, r * 1.1, outline=th["c2"], width=w)
    return sh.done()


def make_tictactoe(rng, i, render=True):
    kind, var = ("win", "block")[i % 2], i // 2
    hard = var % 2 == 1
    me, foe = ("X", "O") if kind == "win" else ("O", "X")
    while True:
        n = 3 if rng.random() < 0.12 else 4
        nx, no = (2, rng.choice([2, 3])) if n == 3 else (rng.randint(3, 6), 0)
        if n == 4:
            no = nx + rng.choice([0, 0, 1])
        board, lines = [None] * (n * n), list(ttt_lines(n))
        rng.shuffle(lines)
        seeds = [(lines[0], me, None)]  # the line that is one move from complete
        if hard and kind == "win":
            seeds.append((lines[1], foe, None))  # O threatens as well
        elif hard:
            a = rng.choice("XO")
            seeds.append((lines[1], a, "O" if a == "X" else "X"))  # a dead near-miss line
        for l, p, stop in seeds:
            free = [j for j in l if board[j] is None]
            if not free:
                continue
            hole = rng.choice(free)
            for j in free:
                board[j] = (stop if j == hole else p)
        left = {"X": nx - board.count("X"), "O": no - board.count("O")}
        empty = [j for j in range(n * n) if board[j] is None]
        rng.shuffle(empty)
        if left["X"] < 0 or left["O"] < 0 or len(empty) - left["X"] - left["O"] < 4:
            continue
        for p in "XO":
            for _ in range(left[p]):
                board[empty.pop()] = p
        if ttt_won(board, n, "X") or ttt_won(board, n, "O"):
            continue
        xw, ow = ttt_wins(board, n, "X"), ttt_wins(board, n, "O")
        if kind == "win" and (len(xw) != 1 or bool(set(ow) - set(xw)) != hard):
            continue
        if kind == "block" and (xw or len(ow) != 1 or bool(ttt_dead(board, n)) != hard):
            continue
        gold = (xw if kind == "win" else ow)[0]

        def tempting(j):  # the other side's winning cell, then cells on lines that are already half built
            s = 10 if j in ow or j in xw else 0
            for l in ttt_lines(n):
                if j in l:
                    c = Counter(board[q] for q in l)
                    s += 3 if max(c["X"], c["O"]) >= n - 2 and min(c["X"], c["O"]) == 0 else 0
            return s + rng.random() * 4

        others = _ranked(gold, [j for j in range(n * n) if board[j] is None and j != gold], rng, tempting)
        if others is None or (kind == "win" and hard and not set(ow) & set(others)):  # the tempting block must be offered
            continue
        break
    if kind == "win":
        ins = "Tic-tac-toe on a %dx%d board (%d in a line wins). It is X's turn. Which numbered cell lets X win immediately?" % (n, n, n)
    else:
        ins = ("Tic-tac-toe on a %dx%d board (%d in a line wins). It is X's turn and X cannot win this move. Which numbered "
               "cell must X take to stop O from winning next move?" % (n, n, n))
    q, label = _choice(ins, [str(j + 1) for j in [gold] + others], 0, rng)
    line = next(l for l in ttt_lines(n) if gold in l and all(board[j] == me for j in l if j != gold))
    why = "%s holds cells %s; cell %d completes that line." % (me, ", ".join(str(j + 1) for j in line if j != gold), gold + 1)
    if kind == "win" and hard:
        why += " O threatens cell %d, but winning now comes first." % (next(j for j in ow if j != gold) + 1)
    st = dict(n=n, board=board, gold=gold)
    sty = _style(rng, n, n, 84, 170)
    return dict(q=q, label=label, why=why, kind=kind + ("_hard" if hard else ""), meta=st, key="%d|%s" % (n, "".join(p or "." for p in board)),
                image=lambda: draw_ttt(st, sty))


# ---- Connect Four -------------------------------------------------------------------------------------------------
C4_W, C4_H = 7, 6
C4_WINDOWS = [(tuple((c + k * dc, r + k * dr) for k in range(4)), (dc, dr)) for c in range(C4_W) for r in range(C4_H)
              for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1))
              if 0 <= c + 3 * dc < C4_W and 0 <= r + 3 * dr < C4_H]
C4_NAME = {"R": "Red", "Y": "Yellow"}


def c4_at(cols, c, r):
    return cols[c][r] if r < len(cols[c]) else None


def c4_has4(cols, p):
    return any(all(c4_at(cols, c, r) == p for c, r in w) for w, _ in C4_WINDOWS)


def c4_threats(cols, p):
    """(playable, floating): for every window with three discs of p and one empty cell, the empty cell goes to
    ``playable`` ({column: [window directions]}) if a disc dropped now lands there, else to ``floating`` (set of cells)."""
    playable, floating = {}, set()
    for w, d in C4_WINDOWS:
        vals = [c4_at(cols, c, r) for c, r in w]
        if vals.count(p) == 3 and vals.count(None) == 1:
            c, r = w[vals.index(None)]
            if r == len(cols[c]):
                playable.setdefault(c, []).append(d)
            else:
                floating.add((c, r))
    return playable, floating


def c4_playout(rng, moves):
    """A position after ``moves`` legal alternating drops (either colour may have started) and the side to move."""
    cols, p = [[] for _ in range(C4_W)], rng.choice("RY")
    for _ in range(moves):
        c = rng.choice([c for c in range(C4_W) if len(cols[c]) < C4_H])
        cols[c].append(p)
        p = "Y" if p == "R" else "R"
    return cols, p


def draw_connect4(st, sty):
    th, c, x0, y0 = sty["th"], sty["cell"], sty["x0"], sty["y0"]
    sh = Sheet(sty["w"], sty["h"], th["bg"])
    board = [(34, 84, 190), (24, 110, 124), (70, 74, 90), (44, 60, 128), (30, 120, 84)][sty["v"] % 5]
    hole = th["bg"] if sum(th["bg"]) > 380 else (224, 226, 232)
    pad = c * 0.12
    sh.rect(x0 - pad, y0 - pad, x0 + C4_W * c + pad, y0 + C4_H * c + pad, fill=board, r=c * 0.2)
    fill = {"R": (222, 44, 44), "Y": (248, 208, 40)}
    for col in range(C4_W):
        sh.text(x0 + (col + 0.5) * c, y0 - pad - c * 0.3, str(col + 1), c * 0.3, th["ink"], family=sty["family"])
        for r in range(C4_H):
            p = c4_at(st["cols"], col, r)
            cx, cy = x0 + (col + 0.5) * c, y0 + (C4_H - 1 - r + 0.5) * c
            if p is None:
                sh.circle(cx, cy, c * 0.39, fill=hole)
            else:
                sh.circle(cx, cy, c * 0.39, fill=fill[p], outline=_mix(fill[p], (0, 0, 0), 0.3), width=max(1.5, c * 0.05))
    return sh.done()


def make_connect4(rng, i, render=True):
    kind, var = ("can_win", "column")[i % 2], i // 2
    want, hard = var % 2 == 1, (var // 2) % 2 == 1
    while True:
        cols, p = c4_playout(rng, rng.randint(7, 34))
        foe = "Y" if p == "R" else "R"
        if c4_has4(cols, "R") or c4_has4(cols, "Y"):
            continue
        wins, floating = c4_threats(cols, p)
        foe_wins = c4_threats(cols, foe)[0]
        open_cols = [c for c in range(C4_W) if len(cols[c]) < C4_H]
        diag_only = bool(wins) and all(d[0] and d[1] for ds in wins.values() for d in ds)
        if kind == "can_win":
            if bool(wins) != want or (diag_only if want else bool(floating)) != hard:
                continue
            q, label = _noul("Connect Four: it is %s's turn (discs fall to the lowest empty cell of a column). Can %s get four in a "
                             "row with this one move?" % (C4_NAME[p], C4_NAME[p])), int(want)
            if want:
                why = "A %s disc in column %s completes four in a row." % (C4_NAME[p].lower(), " or ".join(str(c + 1) for c in sorted(wins)))
            elif floating:
                why = "No column completes four: the open end%s of %s's three in a row (column %s) cannot be reached yet." % (
                    "s" if len(floating) > 1 else "", C4_NAME[p], ", ".join(str(c + 1) for c in sorted({c for c, _ in floating})))
            else:
                why = "No %s disc dropped now completes four in a row." % C4_NAME[p].lower()
            break
        decoys = sorted((set(foe_wins) | {c for c, _ in floating}) & set(open_cols) - set(wins))
        if len(wins) != 1 or len(open_cols) < 4 or (diag_only or bool(decoys)) != hard:
            continue
        gold = next(iter(wins))
        others = _ranked(gold, [c for c in open_cols if c != gold], rng, lambda c: (c in decoys) + rng.random() * 0.9)
        if others is None or (decoys and not diag_only and not set(decoys) & set(others)):  # a decoy column must be offered
            continue
        q, label = _choice("Connect Four: it is %s's turn (discs fall to the lowest empty cell of a column). Which column gives %s "
                           "four in a row at once?" % (C4_NAME[p], C4_NAME[p]), [str(c + 1) for c in [gold] + others], 0, rng)
        d = wins[gold][0]
        why = "A disc in column %d lands on row %d and completes a %s line of four." % (
            gold + 1, len(cols[gold]) + 1, "vertical" if d == (0, 1) else "horizontal" if d == (1, 0) else "diagonal")
        break
    st = dict(cols=cols, turn=p)
    sty = _style(rng, C4_W, C4_H, 56, 96, pad=(8, 44, 8, 8))
    return dict(q=q, label=label, why=why, kind=kind + ("_hard" if hard else ""), meta=st,
                key="%s|%s" % (p, "/".join("".join(c) for c in cols)), image=lambda: draw_connect4(st, sty))


# ---- mazes --------------------------------------------------------------------------------------------------------


def maze_bfs(W, H, walls, s):
    """{cell: number of moves from s} over open cells, 4-connected."""
    dist, todo = {s: 0}, deque([s])
    while todo:
        x, y = todo.popleft()
        for dx, dy in DIRS.values():
            n = (x + dx, y + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and n not in walls and n not in dist:
                dist[n] = dist[(x, y)] + 1
                todo.append(n)
    return dist


def maze_dist(W, H, walls, s, e, key=None, door=None):
    """Fewest moves from s to e, None if there is no route. The door cell can be entered only after the key cell has
    been visited (search over (cell, has key))."""
    start = (s, door is None or s == key)
    dist, todo = {start: 0}, deque([start])
    while todo:
        (x, y), has = cur = todo.popleft()
        if (x, y) == e:
            return dist[cur]
        for dx, dy in DIRS.values():
            n = (x + dx, y + dy)
            if not (0 <= n[0] < W and 0 <= n[1] < H) or n in walls or (n == door and not has):
                continue
            nxt = (n, has or n == key)
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                todo.append(nxt)
    return None


def draw_maze(st, sty):
    W, H, th, c, x0, y0 = st["W"], st["H"], sty["th"], sty["cell"], sty["x0"], sty["y0"]
    sh = Sheet(sty["w"], sty["h"], th["bg"])
    sh.rect(x0, y0, x0 + W * c, y0 + H * c, fill=th["panel"])
    inset = (sty["v"] % 3) * c * 0.03  # walls as full blocks or slightly inset tiles
    for (x, y) in st["walls"]:
        sh.rect(x0 + x * c + inset, y0 + y * c + inset, x0 + (x + 1) * c - inset, y0 + (y + 1) * c - inset, fill=th["wall"],
                r=inset * 2)
    grid = _mix(th["panel"], th["line"], 0.35)
    for k in range(1, W):
        sh.line([(x0 + k * c, y0), (x0 + k * c, y0 + H * c)], grid, 1)
    for k in range(1, H):
        sh.line([(x0, y0 + k * c), (x0 + W * c, y0 + k * c)], grid, 1)
    fw = max(3.0, c * 0.1)
    sh.rect(x0 - fw, y0 - fw, x0 + W * c + fw, y0 + H * c + fw, outline=th["wall"], width=fw)
    marks = [("S", st["s"], (36, 150, 70), "circle"), ("E", st["e"], (214, 50, 50), "square")]
    if st.get("key"):
        marks += [("K", st["key"], (240, 190, 30), "circle"), ("D", st["door"], (130, 80, 40), "door")]
    for ch, (x, y), col, shape in marks:
        cx, cy = x0 + (x + 0.5) * c, y0 + (y + 0.5) * c
        if shape == "circle":
            sh.circle(cx, cy, c * 0.36, fill=col, outline=_mix(col, (0, 0, 0), 0.4), width=max(1.5, c * 0.04))
        elif shape == "square":
            sh.rect(cx - c * 0.36, cy - c * 0.36, cx + c * 0.36, cy + c * 0.36, fill=col, outline=_mix(col, (0, 0, 0), 0.4),
                    width=max(1.5, c * 0.04), r=c * 0.06)
        else:
            sh.rect(cx - c * 0.5, cy - c * 0.5, cx + c * 0.5, cy + c * 0.5, fill=col)
            sh.rect(cx - c * 0.36, cy - c * 0.4, cx + c * 0.36, cy + c * 0.4, outline=(240, 220, 180), width=max(1.5, c * 0.05))
        sh.text(cx, cy, ch, c * 0.42, _ink_on(col), family=sty["family"])
    return sh.done()


MAZE_RULE = "Each move goes one cell up, down, left or right; wall blocks and the frame cannot be crossed."


def make_maze(rng, i, render=True):
    kind, var = ("length", "first", "reach", "key_length")[i % 4], i // 4
    while True:
        W, H = rng.randint(5, 9), rng.randint(5, 9)
        want = var % 2 == 1  # reach: the exit can be reached
        dens = rng.uniform(0.3, 0.46) if kind == "reach" and not want else rng.uniform(0.18, 0.4)
        cells = [(x, y) for y in range(H) for x in range(W)]
        walls = {c for c in cells if rng.random() < dens}
        free = [c for c in cells if c not in walls]
        if len(free) < 8:
            continue
        s, e = rng.sample(free, 2)
        man = abs(s[0] - e[0]) + abs(s[1] - e[1])
        if man < 2:
            continue
        from_e = maze_bfs(W, H, walls, e)
        d, st = from_e.get(s), dict(W=W, H=H, walls=sorted(walls), s=s, e=e)
        if kind == "length":
            hard = var % 2 == 1
            if d is None or d < 4 or (d > man) != hard:
                continue
            opts = _num_options(d, rng, named=[man])
            if opts is None:
                continue
            q, label = _choice("How many moves does the shortest route from S to E take? " + MAZE_RULE, [str(v) for v in opts], 0, rng)
            why = "Breadth-first search reaches E in %d moves%s." % (d, " (the straight-line count of %d is blocked by walls)" % man if d > man else "")
        elif kind == "first":
            gold, hard = DIR_NAMES[var % 4], (var // 4) % 2 == 1
            if d is None or d < 3:
                continue
            step = {n: (s[0] + v[0], s[1] + v[1]) for n, v in DIRS.items()}
            firsts = [n for n, c in step.items() if from_e.get(c) == d - 1]
            greedy = [n for n, c in step.items() if c in from_e and abs(c[0] - e[0]) + abs(c[1] - e[1]) < man]
            if firsts != [gold] or (greedy != [gold]) != hard:
                continue
            q, label = _choice("What is the first move of the shortest route from S to E? " + MAZE_RULE, DIR_NAMES, DIR_NAMES.index(gold), rng)
            why = "Every shortest route (%d moves) starts by going %s." % (d, gold)
        elif kind == "reach":
            hard = (var // 2) % 2 == 1
            if (d is not None) != want:
                continue
            if want and (d > man + 2) != hard:
                continue
            if not want and (min(len(from_e), len(maze_bfs(W, H, walls, s))) >= 4) != hard:
                continue
            q, label = _noul("Can E be reached from S? " + MAZE_RULE), int(want)
            why = "A route of %d moves exists." % d if want else "The walls separate S from E: no route exists."
        else:
            hard = True
            if d is None or d < 3:
                continue
            path = [s]  # one shortest route; the door goes on it
            while path[-1] != e:
                x, y = path[-1]
                path.append(rng.choice([(x + dx, y + dy) for dx, dy in DIRS.values() if from_e.get((x + dx, y + dy)) == from_e[path[-1]] - 1]))
            door = rng.choice(path[1:-1])
            key = rng.choice([c for c in free if c not in (s, e, door)])
            g = maze_dist(W, H, walls, s, e, key, door)
            if g is None or g <= d:
                continue
            closed = maze_dist(W, H, walls | {door}, s, e)
            if (closed == g) != (var % 2 == 1):  # every second row: the route around the door beats fetching the key
                continue
            opts = _num_options(g, rng, named=[d] + ([closed] if closed else []))
            if opts is None:
                continue
            st.update(key=key, door=door)
            q, label = _choice("The door D opens only after the key K has been picked up (stepping on K picks it up). How many moves does "
                               "the shortest route from S to E take? " + MAZE_RULE, [str(v) for v in opts], 0, rng)
            why = "Ignoring the door the route would take %d moves; %s takes %d." % (
                d, "walking around the door" if closed == g else "fetching the key first and passing the door", g)
        break
    sty = _style(rng, W, H, 52, 96)
    return dict(q=q, label=label, why=why, kind=kind + ("_hard" if hard and kind != "key_length" else ""), meta=st,
                key=json.dumps(st, sort_keys=True), image=lambda: draw_maze(st, sty))


# ---- Minesweeper --------------------------------------------------------------------------------------------------
MS_MAX = 22  # largest frontier component that is enumerated
MS_INK = {1: (25, 80, 210), 2: (30, 130, 50), 3: (215, 40, 40), 4: (20, 30, 130), 5: (130, 30, 30), 6: (20, 130, 130),
          7: (20, 20, 20), 8: (110, 110, 110)}
MS_SKINS = [dict(cov=(186, 186, 194), rev=(226, 226, 230), edge=(120, 120, 128)),
            dict(cov=(120, 180, 90), rev=(230, 212, 178), edge=(80, 130, 60)),
            dict(cov=(96, 140, 200), rev=(232, 238, 246), edge=(60, 96, 150)),
            dict(cov=(150, 150, 160), rev=(244, 244, 240), edge=(96, 96, 106))]


def ms_nb(c, W, H):
    return [(c[0] + dx, c[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and 0 <= c[0] + dx < W and 0 <= c[1] + dy < H]


def ms_constraints(W, H, shown, flags):
    """[(covered unflagged neighbours, mines still to place among them)] for every revealed number."""
    out = []
    for c, num in shown.items():
        nb = [n for n in ms_nb(c, W, H) if n not in shown]
        cov = tuple(n for n in nb if n not in flags)
        if cov:
            out.append((cov, num - (len(nb) - len(cov))))
    return out


def ms_forced(W, H, shown, flags=()):
    """{frontier cell: "mine" | "safe" | "open"}: a cell is a mine (safe) if it holds a mine in every (no) layout of
    mines on the frontier that agrees with all the numbers; the total number of mines is not known. None if a frontier
    component is too large to enumerate."""
    cons = ms_constraints(W, H, shown, set(flags))
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for cov, _ in cons:
        for c in cov:
            parent[find(c)] = find(cov[0])
    comps = {}
    for c in list(parent):
        comps.setdefault(find(c), []).append(c)
    out = {}
    for cells in comps.values():
        if len(cells) > MS_MAX:
            return None
        cells.sort(key=lambda c: (c[1], c[0]))
        mine = [(set(cov), need) for cov, need in cons if cov[0] in cells]
        of = {c: [k for k, (cov, _) in enumerate(mine) if c in cov] for c in cells}
        rem, free = [need for _, need in mine], [len(cov) for cov, _ in mine]
        hits, total, cur = dict.fromkeys(cells, 0), [0], {}

        def rec(k):
            if k == len(cells):
                total[0] += 1
                for c in cells:
                    hits[c] += cur[c]
                return
            c = cells[k]
            for v in (0, 1):
                ok = True
                for j in of[c]:
                    rem[j] -= v
                    free[j] -= 1
                    ok = ok and 0 <= rem[j] <= free[j]
                if ok:
                    cur[c] = v
                    rec(k + 1)
                for j in of[c]:
                    rem[j] += v
                    free[j] += 1

        rec(0)
        for c in cells:
            out[c] = "mine" if hits[c] == total[0] else "safe" if hits[c] == 0 else "open"
    return out


def ms_trivial(W, H, shown, flags=()):
    """What a single number settles on its own: all its covered neighbours are mines, or (given the flags) none is."""
    out = {}
    for cov, need in ms_constraints(W, H, shown, set(flags)):
        if need == 0 or need == len(cov):
            for c in cov:
                out[c] = "safe" if need == 0 else "mine"
    return out


def ms_board(rng):
    """A board after a first click on a zero (its region is revealed) and a few more safe reveals next to it."""
    W, H = rng.randint(6, 9), rng.randint(6, 9)
    cells = [(x, y) for y in range(H) for x in range(W)]
    mines = set(rng.sample(cells, int(round(W * H * rng.uniform(0.12, 0.22)))))
    num = {c: sum(n in mines for n in ms_nb(c, W, H)) for c in cells if c not in mines}
    zeros = [c for c in num if num[c] == 0]
    if not zeros:
        return None
    seen = set()

    def click(c):  # reveal a safe cell; a zero opens its whole region, as in the game
        todo = [c]
        while todo:
            c = todo.pop()
            if c not in seen:
                seen.add(c)
                todo += ms_nb(c, W, H) if num[c] == 0 else []

    click(rng.choice(zeros))
    for _ in range(rng.randint(0, 5)):
        edge = sorted(c for c in num if c not in seen and any(n in seen for n in ms_nb(c, W, H)))
        if edge:
            click(rng.choice(edge))
    return W, H, mines, {c: num[c] for c in sorted(seen)}


def draw_minesweeper(st, sty):
    W, H, c, x0, y0, th = st["W"], st["H"], sty["cell"], sty["x0"], sty["y0"], sty["th"]
    skin = MS_SKINS[sty["v"] % len(MS_SKINS)]
    sh = Sheet(sty["w"], sty["h"], th["bg"])
    shown, flags, marks = st["shown"], set(st["flags"]), st["marks"]
    b = max(2.0, c * 0.08)
    for y in range(H):
        for x in range(W):
            px, py = x0 + x * c, y0 + y * c
            if (x, y) in shown:
                sh.rect(px, py, px + c, py + c, fill=skin["rev"], outline=_mix(skin["rev"], (0, 0, 0), 0.25), width=1)
                if shown[(x, y)]:
                    sh.text(px + c / 2, py + c / 2, str(shown[(x, y)]), c * 0.56, MS_INK[shown[(x, y)]], family=sty["family"])
                continue
            sh.rect(px, py, px + c, py + c, fill=skin["edge"])  # a raised tile: light top-left, dark bottom-right
            sh.poly([(px, py), (px + c, py), (px, py + c)], fill=_mix(skin["cov"], (255, 255, 255), 0.55))
            sh.rect(px + b, py + b, px + c - b, py + c - b, fill=skin["cov"])
            if (x, y) in flags:
                sh.line([(px + c * 0.56, py + c * 0.22), (px + c * 0.56, py + c * 0.78)], (20, 20, 20), max(2.0, c * 0.06))
                sh.poly([(px + c * 0.58, py + c * 0.2), (px + c * 0.58, py + c * 0.52), (px + c * 0.24, py + c * 0.36)], fill=(225, 30, 30))
                sh.line([(px + c * 0.36, py + c * 0.8), (px + c * 0.74, py + c * 0.8)], (20, 20, 20), max(2.0, c * 0.07))
            elif (x, y) in marks:
                sh.text(px + c / 2, py + c / 2, marks[(x, y)], c * 0.56, (255, 255, 255), family=sty["family"],
                        stroke=max(1.5, c * 0.05), stroke_fill=(20, 20, 24))
    sh.rect(x0 - 2, y0 - 2, x0 + W * c + 2, y0 + H * c + 2, outline=skin["edge"], width=2)
    return sh.done()


MS_RULE = ("Minesweeper: each number counts the mines among the up to 8 cells around it; raised tiles are still covered, flags mark "
           "known mines and the total number of mines is unknown. ")


def make_minesweeper(rng, i, render=True):
    kind, var = ("mine", "safe", "check")[i % 3], i // 3
    if kind == "check":
        want, ask, hard = var % 2 == 1, ("mine", "safe")[(var // 2) % 2], (var // 4) % 2 == 1
    else:
        ask, hard = kind, var % 2 == 1
    while True:
        b = ms_board(rng)
        if b is None:
            continue
        W, H, mines, shown = b
        base = ms_forced(W, H, shown)
        if base is None:
            continue
        sure = sorted(c for c, v in base.items() if v == "mine")
        p_flag = rng.choice([0.0, 0.5, 0.9])  # a player flags only certain mines: none, some or most of them
        flags = sorted(c for c in sure if rng.random() < p_flag)
        status, easy = ms_forced(W, H, shown, flags), ms_trivial(W, H, shown, flags)
        covered = [(x, y) for y in range(H) for x in range(W) if (x, y) not in shown and (x, y) not in flags]
        golds = [c for c in covered if status.get(c) == ask and (easy.get(c) != ask) == hard]
        others = [c for c in covered if status.get(c, "open") != ask]
        near = [c for c in others if c in status]  # frontier cells that are not forced the asked way: the near misses
        if kind == "check":
            pool = golds if want else [c for c in others if (status.get(c) == "open") == hard]
            if not pool:
                continue
            cell = rng.choice(pool)
            marks = {cell: "A"}
            q, label = _noul(MS_RULE + "Is the cell marked A certainly %s?" % ("a mine" if ask == "mine" else "free of a mine")), int(want)
            state = status.get(cell, "open")
            why = ("Every mine layout that fits the numbers %s A." % ("puts a mine on" if state == "mine" else "leaves no mine on") if state != "open"
                   else "Some layouts that fit the numbers put a mine on A and some do not.")
            break
        if not golds or len(others) < 3 or len(near) < 2:
            continue
        gold = rng.choice(golds)
        rng.shuffle(near)
        near = near[:rng.randint(2, 3)]
        rest = [c for c in others if c not in near]
        rng.shuffle(rest)
        picks = [gold] + (near + rest)[:3]
        rng.shuffle(picks)
        marks = {c: "ABCD"[k] for k, c in enumerate(picks)}
        q, label = {"type": "choice", "instructions": MS_RULE + "Which lettered cell is certainly %s?" % (
            "a mine" if ask == "mine" else "safe (free of a mine)"), "criteria": list("ABCD")}, picks.index(gold)
        why = "Every mine layout that fits the numbers %s %s; each of the other lettered cells %s in at least one layout." % (
            "puts a mine on" if ask == "mine" else "leaves no mine on", marks[gold], "is free" if ask == "mine" else "holds a mine")
        break
    st = dict(W=W, H=H, shown=shown, flags=flags, marks=marks, mines=sorted(mines), ask=ask)
    sty = _style(rng, W, H, 50, 92)
    key = "%d,%d|%s|%s|%s" % (W, H, sorted(shown.items()), flags, sorted(marks.items()))
    return dict(q=q, label=label, why=why, kind=kind + "_" + ask[0] * (kind == "check") + ("hard" if hard else "easy"), meta=st, key=key,
                image=lambda: draw_minesweeper(st, sty))


# ---- 2048 ---------------------------------------------------------------------------------------------------------


def slide_2048(line):
    """One line pushed towards index 0: (new line, points). Equal neighbours merge once, nearest the wall first."""
    t, out, score, k = [v for v in line if v], [], 0, 0
    while k < len(t):
        if k + 1 < len(t) and t[k] == t[k + 1]:
            out.append(2 * t[k])
            score += 2 * t[k]
            k += 2
        else:
            out.append(t[k])
            k += 1
    return out + [0] * (len(line) - len(out)), score


def move_2048(board, d):
    """(board after the move, points) for a 4x4 board given as a tuple of rows; no new tile is spawned."""
    n = len(board)
    cols = d in ("up", "down")
    lines = [[board[k][j] if cols else board[j][k] for k in range(n)] for j in range(n)]
    out, score = [], 0
    for l in lines:
        new, s = slide_2048(l[::-1] if d in ("down", "right") else l)
        out.append(new[::-1] if d in ("down", "right") else new)
        score += s
    return tuple(tuple(out[k][j] if cols else out[j][k] for k in range(n)) for j in range(n)), score


def chain_2048(board, d):
    """The points a player counts who lets merged tiles merge again in the same move (the usual mistake)."""
    total = 0
    while True:
        board, s = move_2048(board, d)
        total += s
        if not s:
            return total


def gaps_2048(board, d):
    """True if some pair that merges under move d is not adjacent on the board (empty cells lie between)."""
    n, cols = len(board), d in ("up", "down")
    for j in range(n):
        l = [board[k][j] if cols else board[j][k] for k in range(n)]
        l = l[::-1] if d in ("down", "right") else l
        idx, k = [q for q in range(n) if l[q]], 0
        while k + 1 < len(idx):
            if l[idx[k]] == l[idx[k + 1]]:
                if idx[k + 1] - idx[k] > 1:
                    return True
                k += 2
            else:
                k += 1
    return False


def near_2048(board, d):
    """True if two equal tiles look mergeable under d but are not: in one line with another tile between them, or
    adjacent only across the line of movement."""
    n, cols = len(board), d in ("up", "down")
    for j in range(n):
        l = [v for v in ([board[k][j] if cols else board[j][k] for k in range(n)]) if v]
        if any(l[a] == l[b] for a in range(len(l)) for b in range(a + 2, len(l))):
            return True
    for y in range(n):
        for x in range(n):
            x2, y2 = (x + 1, y) if cols else (x, y + 1)
            if x2 < n and y2 < n and board[y][x] and board[y][x] == board[y2][x2]:
                return True
    return False


def draw_2048(st, sty):
    th, c, x0, y0, n = sty["th"], sty["cell"], sty["x0"], sty["y0"], 4
    v = sty["v"] % 4
    if v == 0:  # the classic warm skin
        frame, slot = (187, 173, 160), (205, 193, 180)
        skin = {2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59),
                128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)}
    else:  # one hue per power of two, on a neutral frame
        frame, slot = [((96, 104, 120), (124, 132, 148)), ((60, 62, 70), (88, 90, 100)), ((176, 184, 190), (204, 210, 214))][v - 1]
        h0 = (0.55, 0.3, 0.08)[v - 1]
        skin = {2 ** k: tuple(int(255 * u) for u in colorsys.hsv_to_rgb((h0 + 0.085 * k) % 1.0, 0.25 + 0.06 * (k % 5), 0.98 - 0.045 * (k % 6)))
                for k in range(1, 12)}
    sh = Sheet(sty["w"], sty["h"], th["bg"])
    g = c * 0.07
    sh.rect(x0 - g, y0 - g, x0 + n * c + g, y0 + n * c + g, fill=frame, r=c * 0.08)
    for y in range(n):
        for x in range(n):
            val = st["board"][y][x]
            fill = skin[val] if val else slot
            sh.rect(x0 + x * c + g, y0 + y * c + g, x0 + (x + 1) * c - g, y0 + (y + 1) * c - g, fill=fill, r=c * 0.06)
            if val:
                sh.text(x0 + (x + 0.5) * c, y0 + (y + 0.5) * c, str(val), c * (0.44, 0.44, 0.36, 0.29)[len(str(val)) - 1], _ink_on(fill),
                        family=sty["family"])
    return sh.done()


RULE_2048 = ("2048: a move slides every tile as far as it goes in that direction; two equal tiles that meet merge into one tile "
             "of twice the value (each tile merges at most once per move) and score that value. ")


def make_2048(rng, i, render=True):
    kind, var = ("best", "points", "stuck", "merges")[i % 4], i // 4
    while True:
        powers = rng.sample([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], rng.randint(2, 5))
        cells = rng.sample(range(16), rng.randint(5, 14))
        flat = [0] * 16
        for c in cells:
            flat[c] = rng.choice(powers)
        board = tuple(tuple(flat[r * 4:r * 4 + 4]) for r in range(4))
        res = {d: move_2048(board, d) for d in DIR_NAMES}
        if kind == "best":
            a, b = DIR_NAMES[2 + var % 2], DIR_NAMES[(var // 2) % 2]  # one horizontal and one vertical move
            gold, hard = (a, b)[(var // 4) % 2], (var // 8) % 2 == 1
            lose = b if gold == a else a
            if res[gold][1] <= res[lose][1] or (2 * res[lose][1] >= res[gold][1]) != hard:
                continue
            q, label = _choice(RULE_2048 + "Which of these two moves scores more points on this board?", [gold, lose], 0, rng)
            why = "Moving %s scores %d, moving %s scores %d." % (gold, res[gold][1], lose, res[lose][1])
        elif kind == "points":
            d, hard = DIR_NAMES[var % 4], (var // 4) % 2 == 1
            gold, chain = res[d][1], chain_2048(board, d)
            if (chain != gold or gaps_2048(board, d)) != hard:
                continue
            other = res["up" if d in ("left", "right") else "left"][1]
            opts = _num_options(gold, rng, named=[chain, other], lo=0, step=4)
            if opts is None:
                continue
            q, label = _choice(RULE_2048 + "How many points does moving %s score on this board?" % d, [str(v) for v in opts], 0, rng)
            why = "Moving %s merges tiles worth %d in total%s." % (d, gold, "; tiles made by a merge do not merge again (that would give %d)" % chain
                                                               if chain != gold else "")
        elif kind == "stuck":
            gold, hard = DIR_NAMES[var % 4], (var // 4) % 2 == 1
            for _ in range(4):  # pack the tiles against one side until nothing moves
                board = move_2048(board, gold)[0]
            res = {d: move_2048(board, d) for d in DIR_NAMES}
            n_tiles = sum(1 for r in board for v in r if v)
            if [d for d in DIR_NAMES if res[d][0] == board] != [gold] or (n_tiles >= 8) != hard:
                continue
            if max(v for r in board for v in r) > 2048:  # packing merged past the largest tile of the game
                continue
            q, label = _choice(RULE_2048 + "Which move is not possible here (it would leave the board unchanged)?", DIR_NAMES, DIR_NAMES.index(gold), rng)
            why = "No tile can slide or merge when moving %s; each of the other three moves changes the board." % gold
        else:
            d, want, hard = DIR_NAMES[var % 4], (var // 4) % 2 == 1, (var // 8) % 2 == 1
            yes = res[d][1] > 0
            if yes != want:
                continue
            adjacent = chain_2048(tuple(tuple(v or -(4 * y + x + 1) for x, v in enumerate(r)) for y, r in enumerate(board)), d) > 0
            if (not adjacent if want else near_2048(board, d)) != hard:  # yes: only across a gap; no: a near miss is on the board
                continue
            q, label = _noul(RULE_2048 + "Does moving %s merge at least one pair of tiles?" % d), int(want)
            why = "Moving %s merges tiles worth %d." % (d, res[d][1]) if want else "No two equal tiles meet when moving %s." % d
        break
    st = dict(board=board)
    sty = _style(rng, 4, 4, 84, 160)
    return dict(q=q, label=label, why=why, kind=kind + ("_hard" if hard else ""), meta=st, key="%s|%s" % (kind, board), image=lambda: draw_2048(st, sty))


# ---- Snake --------------------------------------------------------------------------------------------------------


def snake_hit(st, d, snake=None):
    """What the head runs into when it moves d: "border", "wall", "neck" (the segment behind the head), "body", "tail"
    (the tip, which is about to move away: makers avoid asking about it), or None for a safe move."""
    snake = st["snake"] if snake is None else snake
    c = (snake[0][0] + DIRS[d][0], snake[0][1] + DIRS[d][1])
    if not (0 <= c[0] < st["W"] and 0 <= c[1] < st["H"]):
        return "border"
    if c in st["walls"]:
        return "wall"
    if c in snake:
        return "neck" if c == snake[1] else "tail" if c == snake[-1] else "body"
    return None


def snake_step(st, d):
    """The snake after a safe move d (it grows by one when the head lands on the food, else the tail follows)."""
    head = (st["snake"][0][0] + DIRS[d][0], st["snake"][0][1] + DIRS[d][1])
    return [head] + (st["snake"] if head == st["food"] else st["snake"][:-1])


def snake_board(rng):
    W, H = rng.randint(7, 12), rng.randint(7, 12)
    body = [(rng.randrange(W), rng.randrange(H))]
    d = rng.choice(DIR_NAMES)
    for _ in range(rng.randint(3, 15)):  # a self-avoiding walk from the tail to the head, mostly straight
        opts = [n for n in DIR_NAMES if 0 <= body[-1][0] + DIRS[n][0] < W and 0 <= body[-1][1] + DIRS[n][1] < H
                and (body[-1][0] + DIRS[n][0], body[-1][1] + DIRS[n][1]) not in body]
        if not opts:
            return None
        d = d if d in opts and rng.random() < 0.55 else rng.choice(opts)
        body.append((body[-1][0] + DIRS[d][0], body[-1][1] + DIRS[d][1]))
    free = [(x, y) for y in range(H) for x in range(W) if (x, y) not in body]
    rng.shuffle(free)
    n_walls = rng.randint(0, max(1, W * H // 9))
    return dict(W=W, H=H, snake=body[::-1], walls=set(free[1:1 + n_walls]), food=free[0])


def draw_snake(st, sty):
    W, H, th, c, x0, y0 = st["W"], st["H"], sty["th"], sty["cell"], sty["x0"], sty["y0"]
    sh = Sheet(sty["w"], sty["h"], th["bg"])
    for y in range(H):
        for x in range(W):
            sh.rect(x0 + x * c, y0 + y * c, x0 + (x + 1) * c, y0 + (y + 1) * c,
                    fill=th["panel"] if (x + y) % 2 or sty["v"] % 2 else _mix(th["panel"], th["line"], 0.1))
    for (x, y) in st["walls"]:
        sh.rect(x0 + x * c + 1, y0 + y * c + 1, x0 + (x + 1) * c - 1, y0 + (y + 1) * c - 1, fill=th["wall"], r=c * 0.08)
    fw = max(3.0, c * 0.12)
    sh.rect(x0 - fw, y0 - fw, x0 + W * c + fw, y0 + H * c + fw, outline=th["wall"], width=fw)
    col = [(52, 168, 83), (66, 133, 244), (240, 160, 30), (170, 90, 210)][(sty["v"] // 2) % 4]
    dark, ctr = _mix(col, (0, 0, 0), 0.4), lambda p: (x0 + (p[0] + 0.5) * c, y0 + (p[1] + 0.5) * c)
    snake = st["snake"]
    for a, b in zip(snake, snake[1:]):  # links first, so the order of the segments can be followed
        sh.line([ctr(a), ctr(b)], dark, c * 0.5)
    for k, p in enumerate(snake):
        cx, cy = ctr(p)
        r = c * (0.42 if k == 0 else 0.26 if k == len(snake) - 1 else 0.36)
        sh.rect(cx - r, cy - r, cx + r, cy + r, fill=dark if k == 0 else col, outline=dark, width=max(1.5, c * 0.05), r=r * 0.45)
    hx, hy = ctr(snake[0])
    dx, dy = snake[0][0] - snake[1][0], snake[0][1] - snake[1][1]
    for s in (-1, 1):  # two eyes looking the way the head travels
        ex, ey = hx + dx * c * 0.16 - dy * s * c * 0.2, hy + dy * c * 0.16 + dx * s * c * 0.2
        sh.circle(ex, ey, c * 0.11, fill=(255, 255, 255))
        sh.circle(ex + dx * c * 0.04, ey + dy * c * 0.04, c * 0.055, fill=(10, 10, 10))
    fx, fy = ctr(st["food"])
    sh.circle(fx, fy + c * 0.04, c * 0.3, fill=(222, 44, 50), outline=(130, 16, 20), width=max(1.5, c * 0.04))
    sh.line([(fx, fy - c * 0.24), (fx + c * 0.1, fy - c * 0.42)], (70, 110, 40), max(2.0, c * 0.07))
    return sh.done()


SNAKE_RULE = ("Snake: the head is the segment with eyes and moves one cell per move. Moving into a wall block, off the board or into "
              "the snake's own body is fatal; the red apple is food. ")


def make_snake(rng, i, render=True):
    kind, var = ("best", "fatal_count", "safe", "trap")[i % 4], i // 4
    near = lambda c: [(c[0] + dx, c[1] + dy) for dx, dy in DIRS.values()]  # noqa: E731
    while True:
        st = snake_board(rng)
        if st is None or st["snake"][-1] in near(st["snake"][0]):
            continue
        head, food = st["snake"][0], st["food"]
        free = lambda c: 0 <= c[0] < st["W"] and 0 <= c[1] < st["H"] and c not in st["walls"] and c not in st["snake"]  # noqa: E731
        hit = {d: snake_hit(st, d) for d in DIR_NAMES}
        if kind == "best":
            gold, hard = DIR_NAMES[var % 4], (var // 4) % 2 == 1
            far = {d: abs(head[0] + v[0] - food[0]) + abs(head[1] + v[1] - food[1]) for d, v in DIRS.items()}
            safe = sorted((d for d in DIR_NAMES if hit[d] is None), key=far.get)
            if not safe or safe[0] != gold or (len(safe) > 1 and far[safe[1]] == far[gold]):
                continue
            if any(hit[d] and far[d] <= far[gold] for d in DIR_NAMES) != hard:  # the direct way to the food is deadly
                continue
            q, label = _choice(SNAKE_RULE + "Which move keeps the snake alive and leaves its head the fewest steps (rows plus columns) "
                               "from the food?", DIR_NAMES, DIR_NAMES.index(gold), rng)
            why = "Moving %s is safe and ends %d step(s) from the food; %s." % (gold, far[gold], "; ".join(
                "%s hits the %s" % (d, hit[d] if hit[d] in ("border", "wall") else "snake's body") for d in DIR_NAMES if hit[d]))
        elif kind == "fatal_count":
            gold, hard = 1 + var % 4, gold_hard(var)
            open_ = [d for d in DIR_NAMES if hit[d] is None]
            rng.shuffle(open_)
            while len(open_) > 4 - gold:  # wall off safe moves until the wanted number is left
                c = (head[0] + DIRS[open_[-1]][0], head[1] + DIRS[open_.pop()][1])
                if c == food:
                    break
                st["walls"].add(c)
            hit = {d: snake_hit(st, d) for d in DIR_NAMES}
            if sum(v is not None for v in hit.values()) != gold or ("body" in hit.values()) != hard:
                continue
            q, label = _choice(SNAKE_RULE + "How many of the four moves (up, down, left, right) are fatal for the snake right now?",
                               ["1", "2", "3", "4"], gold - 1, rng)
            why = "Fatal: %s." % ", ".join("%s (%s)" % (d, hit[d] if hit[d] in ("border", "wall") else "body") for d in DIR_NAMES if hit[d])
        elif kind == "safe":
            d, want, hard = DIR_NAMES[var % 4], (var // 4) % 2 == 1, (var // 8) % 2 == 1
            if (hit[d] is None) != want:
                continue
            around = sum(1 for c in near((head[0] + DIRS[d][0], head[1] + DIRS[d][1])) if c != head and not free(c))
            if (around >= 2 if want else hit[d] == "body") != hard:  # yes: a tight spot; no: a body segment that is not the neck
                continue
            q, label = _noul(SNAKE_RULE + "Is moving %s a safe move for the snake right now?" % d), int(want)
            why = "The cell %s of the head is free." % d if want else "Moving %s runs into the %s." % (d, hit[d] if hit[d] in ("border", "wall") else "snake's body")
        else:
            d, want, hard = DIR_NAMES[var % 4], (var // 4) % 2 == 1, True
            nh = (head[0] + DIRS[d][0], head[1] + DIRS[d][1])
            if hit[d] is not None or nh == food:
                continue
            exits = [c for c in near(nh) if free(c) and c != food]
            rng.shuffle(exits)
            keep = 0 if not want else rng.randint(1, 2) if rng.random() < 0.7 else len(exits)
            st["walls"].update(exits[keep:])  # close the pocket (all of it, or all but one or two cells)
            after = snake_step(st, d)
            if after[-1] in near(nh) or st["snake"][-1] in near(nh):  # the tail tip would decide: not asked
                continue
            if any(snake_hit(st, n, after) is None for n in DIR_NAMES) != want:
                continue
            q, label = _noul(SNAKE_RULE + "The snake now moves %s (a safe move). Will it then have at least one safe move left for the "
                             "move after?" % d), int(want)
            why = ("After moving %s the head can still go %s." % (d, " or ".join(n for n in DIR_NAMES if snake_hit(st, n, after) is None)) if want
                   else "After moving %s every neighbouring cell of the head is a wall, the border or the snake's body." % d)
        break
    st["walls"] = sorted(st["walls"])
    sty = _style(rng, st["W"], st["H"], 44, 80)
    return dict(q=q, label=label, why=why, kind=kind + ("_hard" if hard and kind != "trap" else ""), meta=st,
                key="%s|%s" % (kind, json.dumps(st, sort_keys=True)), image=lambda: draw_snake(st, sty))


def gold_hard(var):
    """fatal_count rows: a body hit (not the neck) is asked for on every second row with two or more fatal moves."""
    return var % 4 >= 1 and (var // 4) % 2 == 1


# ---- chess --------------------------------------------------------------------------------------------------------
ROOK_RAYS, BISHOP_RAYS = [(1, 0), (-1, 0), (0, 1), (0, -1)], [(1, 1), (1, -1), (-1, 1), (-1, -1)]
RAYS = {"R": ROOK_RAYS, "B": BISHOP_RAYS, "Q": ROOK_RAYS + BISHOP_RAYS}
KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
PIECE = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
GLYPH = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞"}  # solid chess glyphs of DejaVu Sans
MAX_COUNT = {"Q": 1, "R": 2, "B": 2, "N": 2, "K": 1}


def sq_name(sq):
    return "abcdefgh"[sq[0]] + str(sq[1] + 1)


def chess_line(board, src, dst):
    """For a queen, rook or bishop on src: None if dst is not on one of its lines, else the number of pieces standing
    strictly between src and dst (0 = it attacks dst)."""
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    if (dx, dy) == (0, 0) or not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
        return None
    step = ((dx > 0) - (dx < 0), (dy > 0) - (dy < 0))
    if step not in RAYS[board[src][1]]:
        return None
    n, c = 0, (src[0] + step[0], src[1] + step[1])
    while c != dst:
        n += c in board
        c = (c[0] + step[0], c[1] + step[1])
    return n


def chess_attacks(board, src, dst):
    """Does the piece on src attack dst (whatever stands on dst)? Sliding pieces are stopped by any piece between."""
    t, dx, dy = board[src][1], dst[0] - src[0], dst[1] - src[1]
    if t == "N":
        return (dx, dy) in KNIGHT
    if t == "K":
        return max(abs(dx), abs(dy)) == 1
    return chess_line(board, src, dst) == 0


def chess_attackers(board, dst, color="w"):
    return sorted(s for s, (c, _) in board.items() if c == color and s != dst and chess_attacks(board, s, dst))


def chess_blocked(board, dst, color="w"):
    """Sliding pieces of this colour that are lined up with dst but have a piece in the way."""
    return sorted(s for s, (c, t) in board.items() if c == color and t in RAYS and s != dst and (chess_line(board, s, dst) or 0) > 0)


def _chess_from(rng, board, dst, t, blocked=False):
    """A free square from which a piece of type t attacks dst on the current board; with ``blocked``, a slider's square
    and a free square between it and dst for the blocker."""
    if t == "N" or t == "K":
        cand = [(dst[0] + dx, dst[1] + dy) for dx, dy in (KNIGHT if t == "N" else ROOK_RAYS + BISHOP_RAYS)]
        cand = [c for c in cand if 0 <= c[0] < 8 and 0 <= c[1] < 8 and c not in board]
        return rng.choice(cand) if cand else None
    cand = []
    for dx, dy in RAYS[t]:
        c, between = (dst[0] + dx, dst[1] + dy), []
        while 0 <= c[0] < 8 and 0 <= c[1] < 8 and (blocked or c not in board):
            if c not in board:
                cand += [(c, b) for b in between] if blocked else [(c, None)]
                between.append(c)
            c = (c[0] + dx, c[1] + dy)
    if not cand:
        return None
    return rng.choice(cand) if blocked else rng.choice(cand)[0]


def chess_fill(rng, board, white, black, avoid=()):
    """Drop the remaining pieces on free squares; the kings never stand next to each other."""
    free = [(x, y) for x in range(8) for y in range(8) if (x, y) not in board and (x, y) not in avoid]
    rng.shuffle(free)
    for color, types in (("w", white), ("b", black)):
        for t in types:
            for k, s in enumerate(free):
                other = next((q for q, v in board.items() if v == ("b" if color == "w" else "w", "K")), None)
                if t == "K" and other and max(abs(other[0] - s[0]), abs(other[1] - s[1])) <= 1:
                    continue
                board[free.pop(k)] = (color, t)
                break
    return board


def draw_chess(st, sty):
    th, c, x0, y0, flip = sty["th"], sty["cell"], sty["x0"], sty["y0"], st["flip"]
    sh = Sheet(sty["w"], sty["h"], th["bg"])
    light, dark = [(th["light"], th["dark"]), ((240, 217, 181), (181, 136, 99)), ((234, 236, 208), (118, 150, 86)),
                   ((222, 227, 230), (140, 162, 173))][sty["v"] % 4]
    at = lambda s: (x0 + ((7 - s[0]) if flip else s[0]) * c, y0 + (s[1] if flip else 7 - s[1]) * c)  # noqa: E731
    for x in range(8):
        for y in range(8):
            px, py = at((x, y))
            sh.rect(px, py, px + c, py + c, fill=dark if (x + y) % 2 == 0 else light)
    sh.rect(x0 - 2, y0 - 2, x0 + 8 * c + 2, y0 + 8 * c + 2, outline=th["line"], width=2)
    for k in range(8):
        sh.text(at((k, 0))[0] + c / 2, y0 + 8 * c + c * 0.3, "abcdefgh"[k], c * 0.3, th["ink"], family=sty["family"])
        sh.text(x0 - c * 0.3, at((0, k))[1] + c / 2, str(k + 1), c * 0.3, th["ink"], family=sty["family"])
    if st.get("mark"):
        px, py = at(st["mark"])
        sh.rect(px + 2, py + 2, px + c - 2, py + c - 2, outline=(230, 30, 60), width=max(3.0, c * 0.09))
        sh.circle(px + c / 2, py + c / 2, c * 0.13, fill=(230, 30, 60))
    letters = (sty["v"] // 4) % 2 == 0
    for s, (color, t) in st["board"].items():
        px, py = at(s)
        fill, edge = ((250, 250, 246), (20, 20, 24)) if color == "w" else ((24, 24, 28), (236, 236, 236))
        if letters:  # a disc with the piece letter
            sh.circle(px + c / 2, py + c / 2, c * 0.38, fill=fill, outline=edge, width=max(2.0, c * 0.05))
            sh.text(px + c / 2, py + c / 2, t, c * 0.44, edge, family=sty["family"])
        else:  # the solid chess glyph, filled in the piece's colour and outlined in the other
            sh.text(px + c / 2, py + c / 2, GLYPH[t], c * 0.74, fill, bold=False, family=0, stroke=max(1.5, c * 0.035), stroke_fill=edge)
    return sh.done()


CHESS_RULE = "Chess diagram (kings, queens, rooks, bishops and knights only%s); a piece in the way blocks a queen, rook or bishop. "


def make_chess(rng, i, render=True):
    kind, var = ("check", "which", "count")[i % 3], i // 3
    sty = _style(rng, 8, 8, 52, 84, pad=(26, 0, 0, 26))
    letters = (sty["v"] // 4) % 2 == 0
    rule = CHESS_RULE % (": K, Q, R, B, N; white pieces are light, black pieces dark" if letters else "")
    while True:
        board, flip = {}, rng.random() < 0.25
        target = (rng.randrange(8), rng.randrange(8))
        extra_w = rng.sample(["Q", "R", "R", "B", "B", "N", "N"], rng.randint(1, 4))
        extra_b = rng.sample(["Q", "R", "B", "N", "N", "B"], rng.randint(0, 3))
        if kind == "check":
            want, hard = var % 2 == 1, (var // 2) % 2 == 1
            board[target] = ("b", "K")
            if want:
                t = rng.choice("QRBN")
                s = _chess_from(rng, board, target, t)
                board[s] = ("w", t)
            elif hard:
                t = rng.choice("QRB")
                s, b = _chess_from(rng, board, target, t, blocked=True)
                board[s], board[b] = ("w", t), rng.choice([("b", rng.choice("RBN")), ("w", rng.choice("BN"))])
            chess_fill(rng, board, ["K"] + extra_w, extra_b)
            att, blk = chess_attackers(board, target), chess_blocked(board, target)
            close = [s for s in att if board[s][1] != "N" and max(abs(s[0] - target[0]), abs(s[1] - target[1])) <= 2]
            if bool(att) != want or (not close if want else bool(blk)) != hard:
                continue
            q, label = _noul(rule + "Is the black king in check?"), int(want)
            why = ("The white %s attacks the black king on %s." % (" and ".join("%s on %s" % (PIECE[board[s][1]], sq_name(s)) for s in att), sq_name(target))
                   if want else "No white piece attacks %s%s." % (sq_name(target), "; the %s on %s is blocked" % (PIECE[board[blk[0]][1]], sq_name(blk[0])) if blk else ""))
        elif kind == "which":
            gold, hard = "QRBN"[var % 4], (var // 4) % 2 == 1
            s = _chess_from(rng, board, target, gold)
            board[s] = ("w", gold)
            rest = [t for t in "QRBN" if t != gold]
            if hard:
                t = rng.choice([t for t in rest if t != "N"])
                got = _chess_from(rng, board, target, t, blocked=True)
                if got is None:
                    continue
                board[got[0]], board[got[1]] = ("w", t), ("b", rng.choice("RBN"))
                rest.remove(t)
            chess_fill(rng, board, ["K"] + rest, ["K"] + extra_b[:2], avoid=[target])
            att = chess_attackers(board, target)
            if [board[s][1] for s in att] != [gold] or bool(chess_blocked(board, target)) != hard:
                continue
            q, label = _choice(rule + "Which white piece attacks the square %s?" % sq_name(target), ["queen", "rook", "bishop", "knight"], "QRBN".index(gold), rng)
            why = "Only the white %s on %s attacks %s." % (PIECE[gold], sq_name(att[0]), sq_name(target))
        else:
            gold, hard = var % 4, (var // 4) % 2 == 1
            pool = ["Q", "R", "R", "B", "B", "N", "N", "K"]
            rng.shuffle(pool)
            ok = True
            for _ in range(gold):
                s = _chess_from(rng, board, target, pool[-1])
                ok = ok and s is not None
                if s:
                    board[s] = ("w", pool.pop())
            for _ in range(rng.randint(1, 2) if hard else 0):
                t = next((t for t in pool if t in RAYS), None)
                got = _chess_from(rng, board, target, t, blocked=True) if t else None
                if got:
                    board[got[0]], board[got[1]] = ("w", t), ("b", rng.choice("RBN"))
                    pool.remove(t)
            n_more = rng.randint(0, 3)
            more = [t for t in pool if t != "K"][:n_more] + (["K"] if "K" in pool else [])
            chess_fill(rng, board, more, ["K"] + extra_b[:2], avoid=[target])
            att = chess_attackers(board, target)
            if not ok or len(att) != gold or bool(chess_blocked(board, target)) != hard:
                continue
            q, label = _choice(rule + "How many white pieces attack the square marked in red?", ["0", "1", "2", "3"], gold, rng)
            why = ("Attacking %s: %s." % (sq_name(target), ", ".join("%s %s" % (PIECE[board[s][1]], sq_name(s)) for s in att)) if att
                   else "No white piece attacks %s." % sq_name(target))
        kings = [s for s, v in board.items() if v[1] == "K"]
        if len(kings) != 2 or max(abs(kings[0][0] - kings[1][0]), abs(kings[0][1] - kings[1][1])) <= 1:
            continue
        if any(n > MAX_COUNT[t] for (_, t), n in Counter(board.values()).items()):  # no more pieces than a chess set has
            continue
        break
    st = dict(board=board, target=target, flip=flip, mark=target if kind == "count" else None)
    key = "%s|%s|%s" % (kind, sq_name(target), sorted((sq_name(s), v) for s, v in board.items()))
    return dict(q=q, label=label, why=why, kind=kind + ("_hard" if hard else ""), meta=st, key=key, image=lambda: draw_chess(st, sty))


# ---- build --------------------------------------------------------------------------------------------------------
MAKERS = {"game_tictactoe_move": make_tictactoe, "game_connect4_win": make_connect4, "game_maze_path": make_maze,
          "game_minesweeper_safe": make_minesweeper, "game_2048_move": make_2048, "game_snake_safe": make_snake,
          "game_chess_attack": make_chess}
USE = {"game_tictactoe_move": "find the winning or the blocking cell on a tic-tac-toe board",
       "game_connect4_win": "decide whether (and where) the side to move wins a Connect Four position at once",
       "game_maze_path": "shortest route, first move and reachability in a grid maze (with key and door)",
       "game_minesweeper_safe": "find the covered cell that the numbers force to be a mine or to be safe",
       "game_2048_move": "predict what a 2048 move does: points, the better move, the impossible move",
       "game_snake_safe": "safe and fatal moves of a Snake position, one and two moves ahead",
       "game_chess_attack": "attacks on a square of a chess diagram, with blocking pieces"}


def is_hard(kind):
    """Rows that need more than one glance: the *_hard kinds and the two-step kinds (key and door, snake trap)."""
    return kind.endswith("hard") or kind in ("key_length", "trap")


def make_row(name, split, i, render=True, salt=0):
    """Row i of a split: deterministic for (name, split, i, salt). ``image`` is a function that renders the position."""
    return MAKERS[name](random.Random("games-%s-%s-%d-%d" % (name, split, i, salt)), i, render)


def _work(job):
    name, split, i, out_dir, salt = job
    r = make_row(name, split, i, True, salt)
    rel = "images/%s-%06d.jpg" % (split, i)
    r["image"]().save(os.path.join(out_dir, rel), "JPEG", quality=90)
    rec = {"id": "%s-%s-%d" % (name, split, i), "image": rel, "state_text": None, "question": r["q"], "label": r["label"],
           "rationale": r["why"], "kind": r["kind"], "image_license": "generated (VisionJev)"}
    return split, i, rec, r["key"], salt


def build_dataset(name, root=DATA_ROOT, n_train=None, n_val=None, workers=8, log=print):
    from multiprocessing import get_context

    n_train = SIZES[0] if n_train is None else n_train
    n_val = SIZES[1] if n_val is None else n_val
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    jobs = [(name, "val", i, tmp, 0) for i in range(n_val)] + [(name, "train", i, tmp, 0) for i in range(n_train)]
    rows, keys, t0, redrawn = {"train": {}, "val": {}}, {}, time.time(), 0
    with get_context("spawn").Pool(min(workers, 8)) as pool:
        pending = jobs
        while pending:
            redo = []
            for split, i, rec, key, salt in pool.imap(_work, pending, chunksize=8):
                if keys.setdefault(key, (split, i)) != (split, i):  # this position is already a row: draw it again
                    redo.append((name, split, i, tmp, salt + 1))
                    continue
                rows[split][i] = rec
            redrawn += len(redo)
            pending = redo
    out = {s: [rows[s][i] for i in sorted(rows[s])] for s in ("train", "val")}
    for split, rs in out.items():
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"dataset": name, "source": "generated (laya.visionjev_games): positions drawn at random, labels computed by the rules of the game",
                "revision": "v1", "use": USE[name], "license": LICENSE, "seed": "random.Random('games-<name>-<split>-<i>-<salt>')",
                "counts": {s: len(rs) for s, rs in out.items()},
                "kinds": {s: dict(sorted(Counter(r["kind"] for r in rs).items())) for s, rs in out.items()},
                "hard_share": {s: round(sum(is_hard(r["kind"]) for r in rs) / max(1, len(rs)), 3) for s, rs in out.items()},
                "label_counts": {s: dict(sorted(Counter("%s:%s" % (r["question"]["type"], r["label"]) for r in rs).items())) for s, rs in out.items()},
                "question_only": text_baseline(out["train"], out["val"]), "unique_positions": len(keys), "redrawn_duplicates": redrawn,
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
    log("%s: %s in %.0f s; hard %s; question-only %s" % (name, manifest["counts"], time.time() - t0, manifest["hard_share"], manifest["question_only"]))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, choices=list(MAKERS), help="build only these sets (default: all)")
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--train", type=int, default=None)
    ap.add_argument("--val", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for n in (a.only or list(MAKERS)):
        build_dataset(n, a.root, a.train, a.val, a.workers, log=lambda m: print(m, flush=True))
