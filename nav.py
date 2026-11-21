"""Map model: surface heights, line of sight, and time-sliced A* on a 2 m grid."""
import heapq
import json
import math
import os
import time

import numpy as np

SQ2 = math.sqrt(2.0)


def _poly_mask(ring, x0, y0, nx, ny):
    """Even-odd fill of a flat [x0,y0,x1,y1,...] polygon on the 1 m vertex grid. Returns (j0,i0,mask) or None."""
    xs = np.asarray(ring[0::2], dtype=np.float64) - x0
    ys = np.asarray(ring[1::2], dtype=np.float64) - y0
    i0 = max(int(math.floor(xs.min())), 0); i1 = min(int(math.ceil(xs.max())), nx - 1)
    j0 = max(int(math.floor(ys.min())), 0); j1 = min(int(math.ceil(ys.max())), ny - 1)
    if i1 < i0 or j1 < j0:
        return None
    gx, gy = np.meshgrid(np.arange(i0, i1 + 1, dtype=np.float64), np.arange(j0, j1 + 1, dtype=np.float64))
    inside = np.zeros(gx.shape, dtype=bool)
    n = len(xs)
    for k in range(n):
        xa, ya, xb, yb = xs[k], ys[k], xs[(k + 1) % n], ys[(k + 1) % n]
        if ya == yb:
            continue
        c = (ya > gy) != (yb > gy)
        xint = xa + (gy - ya) * (xb - xa) / (yb - ya)
        inside ^= c & (gx < xint)
    return j0, i0, inside


class World:
    def __init__(self, map_dir):
        w = json.load(open(os.path.join(map_dir, 'v2', 'world.json')))
        T = w['terrain']
        self.nx, self.ny, self.x0, self.y0 = T['nx'], T['ny'], float(T['minX']), float(T['minY'])
        step = float(T.get('step', 1.0))
        assert abs(step - 1.0) < 1e-6
        ground = np.fromfile(os.path.join(map_dir, 'v2', T.get('file', 'terrain.i16')), dtype='<i2')
        ground = ground.reshape(self.ny, self.nx).astype(np.float32) * float(T.get('scale', 0.01))
        S = ground.copy()
        for p in w['prisms']:
            r = _poly_mask(p['ring'], self.x0, self.y0, self.nx, self.ny)
            if r is None:
                continue
            j0, i0, m = r
            for h in p.get('holes', []) or []:
                if len(h) >= 6:
                    rh = _poly_mask(h, self.x0, self.y0, self.nx, self.ny)
                    if rh is None:
                        continue
                    hj, hi, hm = rh
                    # clip hole into m's window
                    a0, b0 = max(hj, j0), max(hi, i0)
                    a1, b1 = min(hj + hm.shape[0], j0 + m.shape[0]), min(hi + hm.shape[1], i0 + m.shape[1])
                    if a1 > a0 and b1 > b0:
                        m[a0 - j0:a1 - j0, b0 - i0:b1 - i0] &= ~hm[a0 - hj:a1 - hj, b0 - hi:b1 - hi]
            sub = S[j0:j0 + m.shape[0], i0:i0 + m.shape[1]]
            g = ground[j0:j0 + m.shape[0], i0:i0 + m.shape[1]]
            sel = m & ((p['base'] - g) < 9.5) & (p['top'] > sub)
            sub[sel] = p['top']
        self.S = S
        self.ground = ground
        self.bounds = w['bounds']
        self._build_nav()

    # ---------- sampling ----------
    def h(self, x, y):
        i = int(round(x - self.x0)); j = int(round(y - self.y0))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return float(self.S[j, i])
        return 999.0

    def los(self, a, b, step=1.0):
        """True if the straight segment a->b (3D) clears all surfaces."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        L = math.hypot(dx, dy)
        n = int(L / step)
        if n < 2:
            return True
        t = np.linspace(0.0, 1.0, n + 1)[1:-1]
        ii = np.rint(a[0] + dx * t - self.x0).astype(np.int64)
        jj = np.rint(a[1] + dy * t - self.y0).astype(np.int64)
        np.clip(ii, 0, self.nx - 1, out=ii); np.clip(jj, 0, self.ny - 1, out=jj)
        z = a[2] + dz * t
        return not bool((self.S[jj, ii] > z).any())

    # ---------- navigation grid ----------
    def _build_nav(self):
        C = 2
        self.C = C
        ny2, nx2 = self.ny // C, self.nx // C
        S = self.S[:ny2 * C, :nx2 * C]
        H = S.reshape(ny2, C, nx2, C).max(axis=(1, 3))
        Lo = S.reshape(ny2, C, nx2, C).min(axis=(1, 3))
        self.nx2, self.ny2 = nx2, ny2
        self.H2 = H
        rough = (H - Lo) > 1.6  # a wall edge runs through this cell
        # wall-adjacent: some 8-neighbor is much higher than this cell
        Hp = np.pad(H, 1, mode='constant', constant_values=999.0)
        hi = np.zeros_like(H, dtype=bool)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == 0 and di == 0:
                    continue
                nb = Hp[1 + dj:1 + dj + ny2, 1 + di:1 + di + nx2]
                hi |= nb > H + 2.2
        blocked = rough | hi
        # keep a margin from the map edge
        m = 15 // C
        blocked[:m, :] = True; blocked[-m:, :] = True; blocked[:, :m] = True; blocked[:, -m:] = True
        # near-wall penalty (2 cells)
        bp = np.pad(blocked, 2, mode='constant', constant_values=True)
        near = np.zeros_like(blocked)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                near |= bp[2 + dj:2 + dj + ny2, 2 + di:2 + di + nx2]
        self.blocked = blocked
        self.Hf = H.ravel().astype(np.float64).tolist()
        self.Bf = blocked.ravel().tolist()
        self.Pf = np.where(near & ~blocked, 2.5, 1.0).ravel().tolist()

    def cell(self, x, y):
        i = int((x - self.x0) / self.C); j = int((y - self.y0) / self.C)
        i = min(max(i, 0), self.nx2 - 1); j = min(max(j, 0), self.ny2 - 1)
        return j * self.nx2 + i

    def cell_xy(self, c):
        j, i = divmod(c, self.nx2)
        return (self.x0 + (i + 0.5) * self.C, self.y0 + (j + 0.5) * self.C)

    def free(self, x, y):
        return not self.Bf[self.cell(x, y)]

    def nearest_free(self, x, y, maxr=12):
        c = self.cell(x, y)
        if not self.Bf[c]:
            return c
        j0, i0 = divmod(c, self.nx2)
        for r in range(1, maxr):
            best = None; bd = 1e9
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    j, i = j0 + dj, i0 + di
                    if 0 <= j < self.ny2 and 0 <= i < self.nx2:
                        cc = j * self.nx2 + i
                        if not self.Bf[cc]:
                            d = dj * dj + di * di
                            if d < bd:
                                bd, best = d, cc
            if best is not None:
                return best
        return c


class Planner:
    """Time-sliced A*. Call step(budget_s) each tick; .path is a list of (x, y) when done."""
    STEP_UP = 2.2

    def __init__(self, world):
        self.w = world
        self.job = None
        self.path = None
        self.goal_xy = None
        self.failed = False

    def request(self, start_xy, goal_xy):
        w = self.w
        s = w.cell(start_xy[0], start_xy[1])
        g = w.nearest_free(goal_xy[0], goal_xy[1])
        self.goal_xy = goal_xy
        self.job = self._astar(s, g)
        self.failed = False

    def busy(self):
        return self.job is not None

    def step(self, budget):
        if self.job is None:
            return
        deadline = time.perf_counter() + budget
        try:
            while True:
                r = next(self.job)
                if r is not None:
                    self.path = r
                    self.job = None
                    return
                if time.perf_counter() > deadline:
                    return
        except StopIteration:
            self.job = None
            self.failed = True

    def _astar(self, s, g):
        w = self.w
        nx = w.nx2
        H, B, P = w.Hf, w.Bf, w.Pf
        gj, gi = divmod(g, nx)
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))
        dist = {s: 0.0}
        came = {s: -1}
        openh = [(0.0, s)]
        closed = set()
        n = 0
        up = self.STEP_UP
        while openh:
            f, c = heapq.heappop(openh)
            if c in closed:
                continue
            if c == g:
                out = []
                while c != -1:
                    out.append(w.cell_xy(c))
                    c = came[c]
                out.reverse()
                yield out
                return
            closed.add(c)
            cj, ci = divmod(c, nx)
            hc = H[c]
            dc = dist[c]
            for di, dj, cost in nbrs:
                i, j = ci + di, cj + dj
                if i < 0 or j < 0 or i >= nx or j >= w.ny2:
                    continue
                nc = j * nx + i
                if B[nc] and nc != g:
                    continue
                if H[nc] - hc > up:
                    continue
                if di and dj and (B[cj * nx + i] or B[j * nx + ci]):
                    continue
                nd = dc + cost * P[nc]
                if nd < dist.get(nc, 1e18):
                    dist[nc] = nd
                    came[nc] = c
                    ax, ay = abs(i - gi), abs(j - gj)
                    hh = (ax + ay) + (SQ2 - 2) * min(ax, ay)
                    heapq.heappush(openh, (nd + hh * 1.05, nc))
            n += 1
            if n % 200 == 0:
                yield None
            if n > 60000:
                return
