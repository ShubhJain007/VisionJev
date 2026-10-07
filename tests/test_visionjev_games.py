"""Game generators (``laya.visionjev_games``): the rule code is checked on hand-made positions with known answers, and
every generated label is recomputed here by code that does not use the generator's label logic (playing each option
out, brute force over every mine layout, relaxation instead of breadth-first search, a step-by-step ray walk). Also:
determinism for a fixed seed, balance (gold position, yes / no, difficulty), the record format and image files."""
import importlib.util
import itertools
import json
import os
import sys
from collections import Counter

import pytest

from laya import visionjev_games as G

DATA_VALIDATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_validator.py")
_ROWS = {}


def rows(name, n=160, split="test"):
    if (name, n, split) not in _ROWS:
        _ROWS[(name, n, split)] = [G.make_row(name, split, i, render=False) for i in range(n)]
    return _ROWS[(name, n, split)]


def gold(r):
    q = r["q"]
    return bool(r["label"]) if q["type"] == "noul" else q["criteria"][r["label"]]


# ---- tic-tac-toe --------------------------------------------------------------------------------------------------


def line_of(board, n, p):
    """Independent win test: scan every cell in the four directions."""
    for y in range(n):
        for x in range(n):
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(x + k * dx, y + k * dy) for k in range(n)]
                if all(0 <= cx < n and 0 <= cy < n and board[cy * n + cx] == p for cx, cy in cells):
                    return True
    return False


def test_tictactoe_rules_on_known_positions():
    X, O, _ = "X", "O", None
    b = [X, X, _,
         O, O, _,
         _, _, _]
    assert G.ttt_wins(b, 3, "X") == [2] and G.ttt_wins(b, 3, "O") == [5]
    b = [X, O, _,
         O, X, _,
         _, _, _]
    assert G.ttt_wins(b, 3, "X") == [8] and G.ttt_wins(b, 3, "O") == []  # the diagonal 1-5-9
    b = [X, X, X, O,   # a dead line: three X stopped by an O
         _, O, _, _,
         _, _, O, _,
         _, _, _, _]
    assert G.ttt_wins(b, 4, "X") == [] and G.ttt_wins(b, 4, "O") == [] and len(G.ttt_dead(b, 4)) == 1
    b[15] = "O"
    assert G.ttt_wins(b, 4, "O") == [] and not G.ttt_won(b, 4, "O")  # the O diagonal 1-6-11-16 starts with an X
    b[0], b[12], b[9] = "O", "O", "O"  # O on 4, 10, 13: the other diagonal needs cell 7
    assert G.ttt_wins(b, 4, "O") == [6]


def test_tictactoe_gold_is_the_only_option_that_works():
    for r in rows("game_tictactoe_move"):
        st, opts = r["meta"], [int(o) - 1 for o in r["q"]["criteria"]]
        n, board = st["n"], st["board"]
        assert len(opts) == 4 and all(board[j] is None for j in opts)
        assert board.count("O") - board.count("X") in (0, 1) and not line_of(board, n, "X") and not line_of(board, n, "O")
        mover = "X" if r["kind"].startswith("win") else "O"
        works = [j for j in range(n * n) if board[j] is None and line_of(board[:j] + [mover] + board[j + 1:], n, mover)]
        assert works == [opts[r["label"]]], (r["kind"], board)
        x_wins = [j for j in range(n * n) if board[j] is None and line_of(board[:j] + ["X"] + board[j + 1:], n, "X")]
        o_wins = [j for j in range(n * n) if board[j] is None and line_of(board[:j] + ["O"] + board[j + 1:], n, "O")]
        if r["kind"].startswith("block"):
            assert not x_wins  # "X cannot win this move" is true
        if r["kind"] == "win_hard":
            assert o_wins and set(o_wins) - set(works) and (set(o_wins) - set(works)) & set(opts)  # the block is offered
        if r["kind"] == "win":
            assert set(o_wins) <= set(works)  # no other cell tempts X to block


# ---- Connect Four -------------------------------------------------------------------------------------------------


def four(cols, p):
    at = lambda c, r: cols[c][r] if 0 <= c < 7 and 0 <= r < len(cols[c]) else None  # noqa: E731
    return any(all(at(c + k * dc, r + k * dr) == p for k in range(4))
               for c in range(7) for r in range(6) for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)))


def winning_columns(cols, p):
    return [c for c in range(7) if len(cols[c]) < 6 and four([col + [p] if k == c else col for k, col in enumerate(cols)], p)]


def test_connect4_rules_on_known_positions():
    cols = [list("R"), list("R"), list("R"), [], [], [], list("YYY")]
    assert set(G.c4_threats(cols, "R")[0]) == {3} and set(G.c4_threats(cols, "Y")[0]) == {6}
    assert not G.c4_has4(cols, "R") and winning_columns(cols, "R") == [3]
    cols = [list("YR"), list("YR"), list("YR"), [], [], [], []]  # three red on row 2: the fourth cell floats
    playable, floating = G.c4_threats(cols, "R")
    assert set(playable) == set() and floating == {(3, 1)} and set(G.c4_threats(cols, "Y")[0]) == {3}
    cols = [list("R"), list("YR"), list("YYR"), list("YYY"), [], [], []]  # a rising diagonal, completed on column 4
    assert set(G.c4_threats(cols, "R")[0]) == {3} and G.c4_threats(cols, "R")[0][3] == [(1, 1)]
    cols[3].append("R")
    assert G.c4_has4(cols, "R") and four(cols, "R")


def test_connect4_labels_match_playing_every_column():
    for r in rows("game_connect4_win"):
        cols, p = r["meta"]["cols"], r["meta"]["turn"]
        n = Counter(d for c in cols for d in c)
        assert all(len(c) <= 6 for c in cols) and len(cols) == 7 and not four(cols, "R") and not four(cols, "Y")
        assert n[p] in (n["RY".replace(p, "")], n["RY".replace(p, "")] - 1)  # alternating play, p moves next
        wins = winning_columns(cols, p)
        assert G.C4_NAME[p] in r["q"]["instructions"]
        if r["q"]["type"] == "noul":
            assert r["label"] == int(bool(wins))
        else:
            opts = [int(o) - 1 for o in r["q"]["criteria"]]
            assert wins == [opts[r["label"]]] and all(len(cols[c]) < 6 for c in opts) and len(set(opts)) == 4


# ---- mazes --------------------------------------------------------------------------------------------------------


def relax(W, H, walls, s, e, key=None, door=None):
    """Shortest route by repeated relaxation over (cell, has key) states (no queue, no early exit)."""
    INF = 10 ** 6
    cells = [(x, y) for x in range(W) for y in range(H) if (x, y) not in walls]
    dist = {(c, h): INF for c in cells for h in (False, True)}
    dist[(s, door is None)] = 0
    changed = True
    while changed:
        changed = False
        for (c, h), d in list(dist.items()):
            if d == INF:
                continue
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (c[0] + dx, c[1] + dy)
                if (n, h) not in dist or (n == door and not h):
                    continue
                state = (n, h or n == key)
                if d + 1 < dist[state]:
                    dist[state], changed = d + 1, True
    best = min(dist[(e, False)], dist[(e, True)])
    return None if best == INF else best


def test_maze_rules_on_known_positions():
    # S . #      the wall column forces the route around: 6 moves instead of 2... then a door on the only route
    # # . #
    # E . .
    walls = {(2, 0), (0, 1), (2, 1)}
    assert G.maze_dist(3, 3, walls, (0, 0), (0, 2)) == 4 and G.maze_bfs(3, 3, walls, (0, 2))[(0, 0)] == 4
    assert G.maze_dist(3, 3, walls | {(1, 1)}, (0, 0), (0, 2)) is None
    # door on (1, 1), key on (2, 2): the key cannot be fetched without passing the door
    assert G.maze_dist(3, 3, walls, (0, 0), (0, 2), key=(2, 2), door=(1, 1)) is None
    # S K .      key next to the start, door below the start: 2 moves without the door, 4 with it (S-K-S-D-E)
    # D # .      or 6 around the wall, so fetching the key wins
    # E . .
    walls = {(1, 1)}
    assert G.maze_dist(3, 3, walls, (0, 0), (0, 2)) == 2
    assert G.maze_dist(3, 3, walls, (0, 0), (0, 2), key=(1, 0), door=(0, 1)) == 4
    assert G.maze_dist(3, 3, walls | {(0, 1)}, (0, 0), (0, 2)) == 6
    assert G.maze_dist(3, 3, walls, (0, 0), (0, 2), key=(2, 0), door=(0, 1)) == 6  # key two cells away: 6 either way


def test_maze_labels_match_relaxation():
    seen = Counter()
    for r in rows("game_maze_path", 200):
        st = r["meta"]
        walls, s, e = set(map(tuple, st["walls"])), tuple(st["s"]), tuple(st["e"])
        assert s not in walls and e not in walls and s != e
        d = relax(st["W"], st["H"], walls, s, e, st.get("key"), st.get("door"))
        kind = r["kind"].replace("_hard", "")
        seen[kind] += 1
        if kind == "reach":
            assert r["label"] == int(d is not None)
        elif kind in ("length", "key_length"):
            assert gold(r) == str(d) and len(set(r["q"]["criteria"])) == 4
            if kind == "key_length":
                assert st["door"] not in walls and st["key"] not in walls and relax(st["W"], st["H"], walls, s, e) < d
        else:
            step = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
            ok = []
            for name, (dx, dy) in step.items():
                n = (s[0] + dx, s[1] + dy)
                if 0 <= n[0] < st["W"] and 0 <= n[1] < st["H"] and n not in walls:
                    rest = 0 if n == e else relax(st["W"], st["H"], walls, n, e)
                    if rest is not None and rest + 1 == d:
                        ok.append(name)
            assert ok == [gold(r)]
    assert set(seen) == {"length", "first", "reach", "key_length"}


# ---- Minesweeper --------------------------------------------------------------------------------------------------


def brute_force(W, H, shown, flags):
    """Status of every frontier cell from all 2^n mine layouts on the frontier."""
    frontier = sorted({n for c in shown for n in G.ms_nb(c, W, H) if n not in shown and n not in flags})
    always, never, total = set(frontier), set(frontier), 0
    for bits in itertools.product((0, 1), repeat=len(frontier)):
        mines = {c for c, b in zip(frontier, bits) if b} | set(flags)
        if all(sum(n in mines for n in G.ms_nb(c, W, H)) == v for c, v in shown.items()):
            total += 1
            always &= mines
            never -= mines
    assert total > 0
    return {c: "mine" if c in always else "safe" if c in never else "open" for c in frontier}


def test_minesweeper_rules_on_known_positions():
    # 3 x 3, everything revealed except the right column; the numbers 1 / 1 / 0 ... put the mine in the top corner
    shown = {(0, 0): 0, (0, 1): 0, (0, 2): 0, (1, 0): 1, (1, 1): 1, (1, 2): 0}
    assert G.ms_forced(3, 3, shown) == {(2, 0): "mine", (2, 1): "safe", (2, 2): "safe"} == brute_force(3, 3, shown, [])
    # 1-2-1 along a wall: mines under the two 1s, the middle cell is safe; no single number says so
    shown = {(0, 1): 1, (1, 1): 2, (2, 1): 1}
    assert G.ms_forced(3, 2, shown) == {(0, 0): "mine", (1, 0): "safe", (2, 0): "mine"}
    assert G.ms_trivial(3, 2, shown) == {}
    # 1-1-1-1 along a wall: the end 1s see two cells, the inner 1s three, which forces mine, safe, safe, mine
    shown = {(0, 1): 1, (1, 1): 1, (2, 1): 1, (3, 1): 1}
    assert G.ms_forced(4, 2, shown) == brute_force(4, 2, shown, []) == {(0, 0): "mine", (1, 0): "safe", (2, 0): "safe", (3, 0): "mine"}
    # a lone 1 above two covered cells: either may hold the mine
    assert G.ms_forced(2, 2, {(0, 1): 1, (1, 1): 1}) == {(0, 0): "open", (1, 0): "open"}
    # a flag satisfies the 1: the other covered neighbour is trivially safe
    shown = {(0, 1): 1, (1, 1): 1}
    assert G.ms_trivial(2, 2, shown, flags=[(0, 0)]) == {(1, 0): "safe"} and G.ms_forced(2, 2, shown, [(0, 0)]) == {(1, 0): "safe"}


def test_minesweeper_labels_match_brute_force():
    checked = 0
    for r in rows("game_minesweeper_safe", 150):
        st = r["meta"]
        W, H, shown, flags, mines = st["W"], st["H"], st["shown"], st["flags"], set(st["mines"])
        assert all(sum(n in mines for n in G.ms_nb(c, W, H)) == v for c, v in shown.items())  # the numbers are true
        assert set(flags) <= mines and not set(shown) & mines and not set(st["marks"]) & (set(shown) | set(flags))
        assert all(n in shown for c, v in shown.items() if v == 0 for n in G.ms_nb(c, W, H))  # a zero opens its neighbours
        letter = {v: k for k, v in st["marks"].items()}
        ask = st["ask"]
        truth = (lambda c: c in mines) if ask == "mine" else (lambda c: c not in mines)
        if r["q"]["type"] == "choice":
            assert sorted(letter) == list("ABCD") and r["q"]["criteria"] == list("ABCD") and truth(letter[gold(r)])
        elif r["label"]:
            assert truth(letter["A"])
        frontier = {n for c in shown for n in G.ms_nb(c, W, H) if n not in shown and n not in flags}
        if len(frontier) > 15:
            continue
        checked += 1
        status = brute_force(W, H, shown, flags)
        forced = {k for k, c in letter.items() if status.get(c, "open") == ask}
        assert forced == ({gold(r)} if r["q"]["type"] == "choice" else {"A"} if r["label"] else set()), (r["kind"], status)
        if r["kind"].endswith("hard") and (r["q"]["type"] == "choice" or r["label"]):  # no single number settles the gold
            cell = letter[gold(r)] if r["q"]["type"] == "choice" else letter["A"]
            for c, v in shown.items():
                cov = [n for n in G.ms_nb(c, W, H) if n not in shown and n not in flags]
                need = v - sum(n in flags for n in G.ms_nb(c, W, H))
                assert cell not in cov or 0 < need < len(cov)
    assert checked > 60


# ---- 2048 ---------------------------------------------------------------------------------------------------------


def play(board, d):
    """Independent 2048 move: tiles walk one at a time towards the wall and merge with an unmerged equal neighbour."""
    n = 4
    g = [list(r) for r in board]
    dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
    order = sorted(((x, y) for x in range(n) for y in range(n)), key=lambda c: -(c[0] * dx + c[1] * dy))
    merged, score = set(), 0
    for x, y in order:
        if not g[y][x]:
            continue
        while 0 <= x + dx < n and 0 <= y + dy < n:
            nx, ny = x + dx, y + dy
            if g[ny][nx] == 0:
                g[ny][nx], g[y][x] = g[y][x], 0
            elif g[ny][nx] == g[y][x] and (nx, ny) not in merged and (x, y) not in merged:
                g[ny][nx], g[y][x] = 2 * g[y][x], 0
                merged.add((nx, ny))
                score += g[ny][nx]
                break
            else:
                break
            x, y = nx, ny
    return tuple(map(tuple, g)), score


def test_2048_rules_on_known_positions():
    assert G.slide_2048([2, 2, 2, 0]) == ([4, 2, 0, 0], 4)
    assert G.slide_2048([2, 2, 4, 4]) == ([4, 8, 0, 0], 12)
    assert G.slide_2048([4, 4, 8, 0]) == ([8, 8, 0, 0], 8)  # the new 8 does not merge again in the same move
    assert G.slide_2048([2, 0, 0, 2]) == ([4, 0, 0, 0], 4) and G.slide_2048([2, 4, 2, 4]) == ([2, 4, 2, 4], 0)
    b = ((2, 0, 0, 2), (4, 4, 8, 0), (0, 0, 0, 0), (2, 0, 0, 0))
    assert G.move_2048(b, "left") == (((4, 0, 0, 0), (8, 8, 0, 0), (0, 0, 0, 0), (2, 0, 0, 0)), 12)
    assert G.move_2048(b, "right") == (((0, 0, 0, 4), (0, 0, 8, 8), (0, 0, 0, 0), (0, 0, 0, 2)), 12)
    assert G.move_2048(b, "down")[1] == 0 and G.move_2048(b, "down")[0][3] == (2, 4, 8, 2)
    assert G.chain_2048(b, "left") == 12 + 16 and G.gaps_2048(b, "left") and not G.gaps_2048(b, "up")
    for d in G.DIR_NAMES:
        assert G.move_2048(b, d) == play(b, d)


def test_2048_labels_match_an_independent_move():
    for r in rows("game_2048_move", 200):
        b, ins = r["meta"]["board"], r["q"]["instructions"]
        res = {d: play(b, d) for d in ("up", "down", "left", "right")}
        assert all(v == 0 or (v & (v - 1) == 0 and 2 <= v <= 2048) for row in b for v in row)
        kind = r["kind"].replace("_hard", "")
        if kind == "best":
            a, c = r["q"]["criteria"]
            assert {a, c} & {"left", "right"} and {a, c} & {"up", "down"} and res[a][1] != res[c][1]
            assert gold(r) == max((a, c), key=lambda d: res[d][1])
        elif kind == "points":
            d = next(d for d in res if "moving %s score" % d in ins)
            assert gold(r) == str(res[d][1]) and all(int(o) % 4 == 0 for o in r["q"]["criteria"])
        elif kind == "stuck":
            assert [d for d in res if res[d][0] == b] == [gold(r)]
        else:
            d = next(d for d in res if "moving %s merge" % d in ins)
            assert r["label"] == int(res[d][1] > 0)


# ---- Snake --------------------------------------------------------------------------------------------------------


def test_snake_rules_on_known_positions():
    # . . . . .      head H at (1, 1) looking right, neck to its left, body turning down and back right under the head
    # n H . # .      -> right is free, up is free, left is the neck, down is the body
    # b b b . .
    # . . t . .
    st = dict(W=5, H=4, snake=[(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3)], walls={(3, 1)}, food=(4, 0))
    assert [G.snake_hit(st, d) for d in ("up", "down", "left", "right")] == [None, "body", "neck", None]
    st["snake"] = [(2, 1)] + st["snake"][:-1]  # after moving right: wall ahead, tail tip (2, 2) below
    assert G.snake_step(dict(st, snake=[(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3)]), "right") == st["snake"]
    assert [G.snake_hit(st, d) for d in ("up", "down", "left", "right")] == [None, "tail", "neck", "wall"]
    st = dict(W=3, H=3, snake=[(0, 0), (1, 0), (2, 0)], walls=set(), food=(0, 1))
    assert G.snake_hit(st, "up") == "border" and G.snake_hit(st, "left") == "border" and G.snake_hit(st, "down") is None
    assert G.snake_step(st, "down") == [(0, 1), (0, 0), (1, 0), (2, 0)]  # eating the food: one segment longer


def fatal(st, snake, d):
    dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
    c = (snake[0][0] + dx, snake[0][1] + dy)
    return c[0] < 0 or c[1] < 0 or c[0] >= st["W"] or c[1] >= st["H"] or c in set(map(tuple, st["walls"])) or c in snake


def test_snake_labels_match_simulation():
    names = ("up", "down", "left", "right")
    for r in rows("game_snake_safe", 200):
        st, ins = r["meta"], r["q"]["instructions"]
        snake, food, walls = [tuple(c) for c in st["snake"]], tuple(st["food"]), set(map(tuple, st["walls"]))
        assert len(set(snake)) == len(snake) and not set(snake) & walls and food not in walls and food not in snake
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(snake, snake[1:]))  # a connected body
        assert abs(snake[0][0] - snake[-1][0]) + abs(snake[0][1] - snake[-1][1]) != 1  # the tail tip never decides
        dead = {d: fatal(st, snake, d) for d in names}
        kind = r["kind"].replace("_hard", "")
        if kind == "best":
            far = {d: abs(snake[0][0] + G.DIRS[d][0] - food[0]) + abs(snake[0][1] + G.DIRS[d][1] - food[1]) for d in names}
            alive = [d for d in names if not dead[d]]
            best = min(far[d] for d in alive)
            assert [d for d in alive if far[d] == best] == [gold(r)]
        elif kind == "fatal_count":
            assert gold(r) == str(sum(dead.values())) and sorted(r["q"]["criteria"]) == ["1", "2", "3", "4"]
        elif kind == "safe":
            d = next(d for d in names if "Is moving %s " % d in ins)
            assert r["label"] == int(not dead[d])
        else:
            d = next(d for d in names if "now moves %s " % d in ins)
            assert not dead[d]
            head = (snake[0][0] + G.DIRS[d][0], snake[0][1] + G.DIRS[d][1])
            after = [head] + snake[:-1]  # the food is never on that cell, so the tail follows
            assert head != food and r["label"] == int(any(not fatal(st, after, n) for n in names))


# ---- chess --------------------------------------------------------------------------------------------------------


def sq(name):
    return ("abcdefgh".index(name[0]), int(name[1]) - 1)


def walk_attacks(board, src, dst):
    """Independent attack test: walk outwards from the piece square by square."""
    t = board[src][1]
    if t == "N":
        return sorted((abs(src[0] - dst[0]), abs(src[1] - dst[1]))) == [1, 2]
    if t == "K":
        return src != dst and abs(src[0] - dst[0]) <= 1 and abs(src[1] - dst[1]) <= 1
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)] * (t in "RQ") + [(1, 1), (1, -1), (-1, 1), (-1, -1)] * (t in "BQ")
    for dx, dy in steps:
        c = (src[0] + dx, src[1] + dy)
        while 0 <= c[0] < 8 and 0 <= c[1] < 8:
            if c == dst:
                return True
            if c in board:
                break
            c = (c[0] + dx, c[1] + dy)
    return False


def test_chess_rules_on_known_positions():
    board = {sq("e1"): ("w", "R"), sq("e8"): ("b", "K"), sq("a1"): ("w", "K")}
    assert G.chess_attackers(board, sq("e8")) == [sq("e1")] and G.chess_blocked(board, sq("e8")) == []
    board[sq("e4")] = ("b", "N")  # a knight steps into the file: no check, the rook is blocked
    assert G.chess_attackers(board, sq("e8")) == [] and G.chess_blocked(board, sq("e8")) == [sq("e1")]
    board[sq("h5")] = ("w", "B")  # bishop on the h5-e8 diagonal
    board[sq("f6")] = ("w", "N")  # the knight gives check and also blocks nothing (g6, f7 are free)
    assert G.chess_attackers(board, sq("e8")) == sorted([sq("h5"), sq("f6")])
    board[sq("g6")] = ("b", "R")  # a black rook blocks the bishop
    assert G.chess_attackers(board, sq("e8")) == [sq("f6")] and G.chess_blocked(board, sq("e8")) == sorted([sq("e1"), sq("h5")])
    board[sq("d3")] = ("w", "Q")  # the queen attacks along file, rank and diagonals up to the first piece (e4), not past it
    assert G.chess_attacks(board, sq("d3"), sq("d8")) and G.chess_attacks(board, sq("d3"), sq("a6")) and G.chess_attacks(board, sq("d3"), sq("e4"))
    assert not G.chess_attacks(board, sq("d3"), sq("f5")) and not G.chess_attacks(board, sq("d3"), sq("e8")) and G.chess_blocked(board, sq("f5")) == [sq("d3")]
    assert not G.chess_attacks(board, sq("e1"), sq("h4")) and G.chess_line(board, sq("e1"), sq("h4")) is None  # no rook diagonal
    assert G.chess_attacks(board, sq("a1"), sq("b2")) and not G.chess_attacks(board, sq("a1"), sq("c3"))
    assert G.sq_name((4, 4)) == "e5"


def test_chess_labels_match_a_ray_walk():
    for r in rows("game_chess_attack", 200):
        st, ins = r["meta"], r["q"]["instructions"]
        board, target = st["board"], st["target"]
        kings = [s for s, v in board.items() if v[1] == "K"]
        assert sorted(board[s][0] for s in kings) == ["b", "w"] and max(abs(kings[0][0] - kings[1][0]), abs(kings[0][1] - kings[1][1])) > 1
        assert all(0 <= x < 8 and 0 <= y < 8 for x, y in board)
        for color in "wb":
            n = Counter(t for c, t in board.values() if c == color)
            assert all(n[t] <= G.MAX_COUNT[t] for t in n)
        att = [s for s, (c, _) in board.items() if c == "w" and s != target and walk_attacks(board, s, target)]
        kind = r["kind"].replace("_hard", "")
        if kind == "check":
            assert board[target] == ("b", "K") and r["label"] == int(bool(att)) and st["mark"] is None
        elif kind == "which":
            assert target not in board and G.sq_name(target) in ins and [G.PIECE[board[s][1]] for s in att] == [gold(r)]
            assert Counter(t for c, t in board.values() if c == "w") == Counter("KQRBN")  # each option names one piece
        else:
            assert target not in board and st["mark"] == target and gold(r) == str(len(att)) and G.sq_name(target) not in ins
        if r["kind"].endswith("_hard") and not (kind == "check" and r["label"]):  # a lined-up but blocked slider is on the board
            assert any(t in "QRB" and (s[0] == target[0] or s[1] == target[1] or abs(s[0] - target[0]) == abs(s[1] - target[1]))
                       and not walk_attacks(board, s, target) for s, (c, t) in board.items() if c == "w")


# ---- structure ----------------------------------------------------------------------------------------------------


def test_rows_are_deterministic_for_a_fixed_seed():
    import numpy as np

    for name in G.MAKERS:
        for i in (0, 7):
            a, b = G.make_row(name, "train", i, render=False), G.make_row(name, "train", i, render=False)
            assert (a["q"], a["label"], a["why"], a["kind"], a["key"]) == (b["q"], b["label"], b["why"], b["kind"], b["key"])
        x, y = G.make_row(name, "train", 3)["image"](), G.make_row(name, "train", 3)["image"]()
        assert x.size == y.size and np.array_equal(np.asarray(x), np.asarray(y))
        assert G.make_row(name, "train", 3, render=False)["key"] != G.make_row(name, "val", 3, render=False)["key"]
        assert G.make_row(name, "train", 3, render=False)["key"] != G.make_row(name, "train", 3, render=False, salt=1)["key"]


def test_balance_of_positions_answers_and_difficulty():
    for name in G.MAKERS:
        rs = rows(name, 480, split="balance")
        yes = [r["label"] for r in rs if r["q"]["type"] == "noul"]
        if yes:
            assert 0.4 <= sum(yes) / len(yes) <= 0.6, (name, sum(yes), len(yes))
        by_text = {}
        for r in rs:
            if r["q"]["type"] == "noul":
                by_text.setdefault(r["q"]["instructions"], []).append(r["label"])
        assert all(0.3 <= sum(v) / len(v) <= 0.7 for v in by_text.values() if len(v) >= 20), name
        for k in (2, 4):
            pos = Counter(r["label"] for r in rs if r["q"]["type"] == "choice" and len(r["q"]["criteria"]) == k)
            total = sum(pos.values())
            assert total == 0 or all(pos[j] / total > 0.6 / k for j in range(k)), (name, k, pos)
        assert sum(G.is_hard(r["kind"]) for r in rs) / len(rs) >= 1 / 3, name
        # the gold answer itself is spread out too (directions, pieces, counts, letters), not only its position
        texts = Counter(gold(r) for r in rs if r["q"]["type"] == "choice" and not gold(r).isdigit())
        assert not texts or min(texts.values()) / sum(texts.values()) > 0.4 / len(texts), (name, texts)


def test_numeric_golds_take_every_rank():
    for name, word in (("game_maze_path", "shortest route"), ("game_2048_move", "How many points")):
        rs = [r for r in rows(name, 480, split="balance") if word in r["q"]["instructions"] and r["q"]["criteria"][0].isdigit()]
        ranks = Counter(sorted(int(o) for o in r["q"]["criteria"]).index(int(gold(r))) for r in rs)
        assert len(rs) > 100 and all(ranks[k] / len(rs) > 0.15 for k in range(4)), (name, ranks)


def test_built_datasets_have_images_manifest_and_distinct_splits(tmp_path):
    from PIL import Image

    names = list(G.MAKERS)
    for name in names:
        man = G.build_dataset(name, str(tmp_path), n_train=12, n_val=6, workers=2, log=lambda m: None)
        d = tmp_path / name
        assert man["counts"] == {"train": 12, "val": 6} and (d / "_READY").exists() and man["unique_positions"] == 18
        lic = json.load(open(d / "manifest.json"))["license"]
        assert lic == {"annotations": "generated (VisionJev, exact rule-based labels)", "images": "rendered in-house with PIL and free fonts",
                       "commercial": True, "verified": "generated in-house"}
        for split in ("train", "val"):
            recs = [json.loads(l) for l in open(d / (split + ".jsonl"))]
            assert len(recs) == man["counts"][split] and len({r["image"] for r in recs}) == len(recs)
            for rec in recs:
                assert set(rec) == {"id", "image", "state_text", "question", "label", "rationale", "kind", "image_license"}
                assert rec["state_text"] is None and rec["rationale"] and rec["kind"] and rec["image"].endswith(".jpg")
                q = rec["question"]
                assert set(q) == {"type", "instructions", "criteria"} and q["type"] in ("choice", "noul")
                assert 0 <= rec["label"] < (2 if q["type"] == "noul" else len(q["criteria"]))
                with Image.open(d / rec["image"]) as im:
                    im.load()
                    assert im.format == "JPEG" and im.mode == "RGB" and 280 <= max(im.size) <= 768
                    assert len(set(im.resize((32, 32)).tobytes())) > 8  # not a blank sheet
    if os.path.exists(DATA_VALIDATOR):
        spec = importlib.util.spec_from_file_location("data_validator", DATA_VALIDATOR)
        dv = importlib.util.module_from_spec(spec)
        sys.modules["data_validator"] = dv
        spec.loader.exec_module(dv)
        rep, _ = dv.run([str(tmp_path / n) for n in names], quiet=True)
        assert rep.n_errors == 0, dict(rep.errors)


def test_duplicate_positions_are_drawn_again(tmp_path, monkeypatch):
    """Tic-tac-toe on 3x3 has few positions: a build of only 3x3 boards must still give distinct positions."""
    keys = [G.make_row("game_tictactoe_move", s, i, render=False)["key"] for s, n in (("val", 40), ("train", 400)) for i in range(n)]
    assert len(set(keys)) < len(keys)  # salt 0 alone does repeat positions ...
    G.build_dataset("game_tictactoe_move", str(tmp_path), n_train=400, n_val=40, workers=2, log=lambda m: None)
    man = json.load(open(tmp_path / "game_tictactoe_move" / "manifest.json"))
    assert man["unique_positions"] == 440 and man["redrawn_duplicates"] > 0  # ... and the build redraws them
