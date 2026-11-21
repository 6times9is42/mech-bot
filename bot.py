"""Mech Royale bot."""
import math
import os
import random
import sys
import time

import mechfight as mf
from nav import World, Planner

TAU = 2 * math.pi


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


WEAPON_VALUE = {'laser': 3.0, 'lightning': 2.7, 'gun': 2.0, 'mortar': 1.4}
UPGRADE_PRIO = ["armor", "plating", "laser_damage", "lightning_efficiency", "nanites", "gun_damage",
                "laser_charge", "gun_rate", "lightning_chain", "servos", "gyros", "lightning_range",
                "capacitor", "reactor", "jets", "gun_mag", "gun_velocity", "sensors",
                "mortar_reload", "mortar_splash", "mortar_velocity", "laser_pierce"]

PARAMS = {
    'farm_hp': 0.6,        # min hp fraction to farm npcs/crates
    'retreat_hp': 0.30,    # below this, break line of sight
    'loot_radius': 70.0,
    'drop_enemy_clear': 70.0,
    'zone_frac': 0.55,     # stay within this fraction of the next circle's radius
}


class Bot:
    name = "yash-mech"

    def __init__(self, hello):
        t0 = time.perf_counter()
        cfg = hello['config']
        self.cfg = cfg
        self.slot = hello.get('slot', 0)
        random.seed(hello.get('seed', 0))
        md = hello.get('map', {}).get('dir', '/arena')
        self.w = World(md)
        self.plan = Planner(self.w)
        self.P = dict(PARAMS)
        env = os.environ.get('BOT_PARAMS')
        if env:
            for kv in env.split(','):
                k, v = kv.split('=')
                self.P[k] = float(v)
        self.mem = {}           # id -> dict(pos, vel, t, ...)
        self.last_t = 0.0
        self.strafe = 1
        self.strafe_until = 0.0
        self.pos_hist = []
        self.stuck_until = 0.0
        self.plan_goal = None
        self.plan_time = -99
        self.threat_bearing = None
        self.threat_t = -99
        self.last_damage_t = -99
        self.last_dealt_t = -99
        self.sound_look = None
        self.goal_label = ''
        self.taken_fail = {}
        mf.log(f"init {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------ helpers
    def scale(self, cubes):
        return 1 + self.cfg['cubes']['size_per_cube'] * (cubes or 0)

    def walk_clear(self, a, b):
        w = self.w
        L = d2(a, b)
        n = max(int(L / 1.5), 1)
        hprev = w.Hf[w.cell(a[0], a[1])]
        for k in range(1, n + 1):
            t = k / n
            c = w.cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if w.Bf[c]:
                return False
            hc = w.Hf[c]
            if hc - hprev > 2.2:
                return False
            hprev = hc
        return True

    def free_dir(self, pos, yaw, dist=5.0):
        x = pos[0] + math.cos(yaw) * dist; y = pos[1] + math.sin(yaw) * dist
        return self.walk_clear(pos, (x, y))

    def zone_now(self, z, t):
        if t <= z['shrink_start']:
            return z['center'], z['radius']
        f = min(1.0, (t - z['shrink_start']) / max(1e-6, z['shrink_end'] - z['shrink_start']))
        c = [z['center'][0] + (z['next_center'][0] - z['center'][0]) * f,
             z['center'][1] + (z['next_center'][1] - z['center'][1]) * f]
        return c, z['radius'] + (z['next_radius'] - z['radius']) * f

    def safe_point(self, obs, pos):
        """Where we should be with respect to the zone, or None if fine."""
        z = obs['zone']
        nc, nr = z['next_center'], z['next_radius']
        frac = self.P['zone_frac']
        t = obs['t']
        # hurry more as shrink approaches
        if t > z['shrink_start'] - 25:
            frac = min(frac, 0.6)
        lim = max(nr * frac, min(nr, 12.0))
        dd = d2(pos, nc)
        if dd <= lim:
            return None
        k = (dd - lim * 0.8) / dd
        return (pos[0] + (nc[0] - pos[0]) * k, pos[1] + (nc[1] - pos[1]) * k)

    # ------------------------------------------------------------------ movement
    def go_to(self, obs, pos, goal, act, urgent=False):
        """Steer toward goal using direct walk or A*."""
        now = obs['t']
        if d2(pos, goal) < 1.5:
            act['move'] = [0.0, 0.0]
            return None
        direct = d2(pos, goal) < 120 and self.walk_clear(pos, goal)
        target = goal
        if not direct:
            need = (self.plan_goal is None or d2(self.plan_goal, goal) > 12 or now - self.plan_time > 4.0
                    or (self.plan.path is None and not self.plan.busy()))
            if need and not self.plan.busy():
                self.plan.request(pos, goal)
                self.plan_goal = goal
                self.plan_time = now
                self.plan.path = None
            path = self.plan.path
            if path:
                # drop passed waypoints
                while len(path) > 1 and d2(pos, path[0]) < 4.0:
                    path.pop(0)
                best = path[0]
                for k in range(min(len(path), 14) - 1, 0, -1):
                    if self.walk_clear(pos, path[k]):
                        best = path[k]
                        break
                target = best
        yaw = math.atan2(target[1] - pos[1], target[0] - pos[0])
        act['move'] = [math.cos(yaw), math.sin(yaw)]
        if urgent:
            act['boost'] = True
        return yaw

    def handle_stuck(self, obs, act):
        me = obs['self']
        now = obs['t']
        self.pos_hist.append((now, me['pos'][0], me['pos'][1]))
        while self.pos_hist and now - self.pos_hist[0][0] > 1.2:
            self.pos_hist.pop(0)
        mv = act.get('move', [0, 0])
        if now < self.stuck_until:
            act['boost'] = True
            if me['grounded']:
                act['jump'] = True
            return
        if math.hypot(*mv) > 0.5 and len(self.pos_hist) > 8 and now - self.pos_hist[0][0] > 1.0:
            moved = math.hypot(me['pos'][0] - self.pos_hist[0][1], me['pos'][1] - self.pos_hist[0][2])
            if moved < 1.5 and me.get('energy', 0) > 25:
                self.stuck_until = now + 0.9
                act['jump'] = True
                act['boost'] = True
                self.plan_time = -99

    # ------------------------------------------------------------------ memory
    def update_mem(self, obs):
        now = obs['t']
        for m in obs['mechs']:
            e = self.mem.get(m['id'])
            if e is None:
                e = self.mem[m['id']] = {}
            e.update(m)
            e['t'] = now
        for k in [k for k, e in self.mem.items() if now - e['t'] > 20]:
            del self.mem[k]
        for ev in obs['events']:
            if ev['type'] == 'damaged':
                self.last_damage_t = now
                if ev.get('bearing') is not None:
                    self.threat_bearing = ev['bearing']
                    self.threat_t = now
            elif ev['type'] == 'hit':
                self.last_dealt_t = now
            elif ev['type'] == 'take_failed':
                self.taken_fail[ev['id']] = now
        best = None
        for s in obs['sounds']:
            if s['kind'] in ('gun', 'laser', 'laser_charge', 'lightning', 'mortar', 'step', 'boost', 'jet', 'reload'):
                pri = {'near': 0, 'mid': 1, 'far': 2}[s['dist']]
                if best is None or pri < best[0]:
                    best = (pri, s['bearing'], now)
        if best and best[0] <= 1:
            self.sound_look = best

    # ------------------------------------------------------------------ aiming
    def eye(self, me):
        return (me['pos'][0], me['pos'][1], me['pos'][2] + me['stats'].get('eye_height', 8))

    def muzzle(self, me):
        return (me['pos'][0], me['pos'][1], me['pos'][2] + me['stats'].get('muzzle_height', 7))

    def center(self, m):
        if m.get('kind') == 'npc':
            return (m['pos'][0], m['pos'][1], m['pos'][2] + 1.75)
        if 'cube_h' in m:
            return (m['pos'][0], m['pos'][1], m['pos'][2] + m['cube_h'])
        s = self.scale(m.get('cubes', 0))
        return (m['pos'][0], m['pos'][1], m['pos'][2] + 4.5 * s)

    def lead(self, src, tgt, vel, speed, extra=0.0):
        """Intercept point for a straight projectile."""
        px, py, pz = tgt
        vx, vy, vz = vel if vel else (0, 0, 0)
        vz = 0.0 if abs(vz) < 3 else vz * 0.5
        t = math.dist(src, tgt) / speed + extra
        for _ in range(3):
            q = (px + vx * t, py + vy * t, pz + vz * t)
            t = math.dist(src, q) / speed + extra
        return (px + vx * t, py + vy * t, pz + vz * t), t

    def aim_at(self, src, p):
        dx, dy, dz = p[0] - src[0], p[1] - src[1], p[2] - src[2]
        return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]

    def ballistic(self, src, p, v, g, high=False):
        dx, dy = p[0] - src[0], p[1] - src[1]
        d = math.hypot(dx, dy)
        h = p[2] - src[2]
        disc = v ** 4 - g * (g * d * d + 2 * h * v * v)
        if disc < 0 or d < 1:
            return None
        r = math.sqrt(disc)
        tan = (v * v + r) / (g * d) if high else (v * v - r) / (g * d)
        pitch = math.atan(tan)
        tflight = d / (v * math.cos(pitch))
        return [math.atan2(dy, dx), pitch], tflight

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t_start = time.perf_counter()
        me = obs['self']
        now = obs['t']
        pos = me['pos']
        act = {'tick': obs['tick'], 'fire': [False, False]}
        self.update_mem(obs)

        if obs.get('offer'):
            ch = obs['offer']['choices']
            ids = [c['id'] for c in ch]
            best = min(range(len(ids)), key=lambda i: UPGRADE_PRIO.index(ids[i]) if ids[i] in UPGRADE_PRIO else 99)
            act['pick'] = best

        weapons = me['weapons']
        kinds = [w['kind'] if w else None for w in weapons]
        hp_frac = me['hp'] / me['max_hp']
        eye = self.eye(me)
        muz = self.muzzle(me)

        # --- visible enemies
        enemies = [m for m in obs['mechs'] if m['kind'] == 'mech']
        npcs = [m for m in obs['mechs'] if m['kind'] == 'npc']
        for m in enemies:
            m['_d'] = d2(pos, m['pos'])
            m['_los'] = self.w.los(eye, self.center(m))

        target = None
        if enemies:
            def score(m):
                s = m['_d']
                s += m.get('hp', 1400) * 0.04
                if not m['_los']:
                    s += 120
                return s
            target = min(enemies, key=score)

        # --- weapon choice ranges
        has = set(k for k in kinds if k)
        if 'lightning' in has and not ({'laser', 'gun'} & has):
            pref_range = 25.0
        elif 'lightning' in has:
            pref_range = 60.0
        else:
            pref_range = 95.0

        mode = 'idle'
        aim = None
        move_yaw = None
        z = obs['zone']
        zc, zr = self.zone_now(z, now)
        out_of_zone = not me['in_zone']
        safe = self.safe_point(obs, pos)

        # ---------------- combat
        if target is not None and target['_d'] < 230:
            mode = 'fight'
            tc = self.center(target)
            tvel = target.get('vel', [0, 0, 0])
            d = target['_d']
            # movement: strafe + range keeping
            if now > self.strafe_until or not self.free_dir(pos, math.atan2(tc[1] - pos[1], tc[0] - pos[0]) + self.strafe * math.pi / 2, 5):
                self.strafe = -self.strafe if random.random() < 0.7 or now <= self.strafe_until else self.strafe
                self.strafe_until = now + random.uniform(0.5, 1.4)
            to = math.atan2(tc[1] - pos[1], tc[0] - pos[0])
            radial = 0.0
            want = pref_range
            if hp_frac < self.P['retreat_hp'] and target.get('hp', 1400) > me['hp'] * 1.3:
                want = 200.0
            if d > want * 1.25:
                radial = 0.8
            elif d < want * 0.75:
                radial = -0.8
            mvx = math.cos(to) * radial + math.cos(to + self.strafe * math.pi / 2)
            mvy = math.sin(to) * radial + math.sin(to + self.strafe * math.pi / 2)
            move_yaw = math.atan2(mvy, mvx)
            # avoid walls
            for k in (0, 1, -1, 2, -2, 3, -3, 4):
                yy = move_yaw + k * math.pi / 4
                if self.free_dir(pos, yy, 5):
                    move_yaw = yy
                    break
            # zone overrides when outside
            if out_of_zone or (safe and d2(pos, z['next_center']) > z['next_radius'] * 0.95 and now > z['shrink_start'] - 10):
                move_yaw = None
            # threatened by laser charge? boost
            if target.get('charging') and abs(wrap(target.get('aim', [0])[0] - math.atan2(pos[1] - target['pos'][1], pos[0] - target['pos'][0]))) < 0.2:
                act['boost'] = me['energy'] > 15
                if random.random() < 0.25:
                    self.strafe = -self.strafe
            elif 'lightning' not in has and me['energy'] > 50:
                act['boost'] = True
            # ---- fire control
            aim = self.fire_control(obs, me, target, act, kinds, weapons, muz)
        else:
            # no mech: maybe npc/crate targets for farming
            aim = None

        # ---------------- non-combat goals
        goal = None
        urgent = False
        if out_of_zone:
            goal = z['next_center'] if d2(pos, z['next_center']) < d2(pos, zc) else zc
            urgent = True
            self.goal_label = 'zone!'
        if mode != 'fight' or move_yaw is None:
            if goal is None and safe is not None and (now > z['shrink_start'] - 30 or d2(pos, z['next_center']) > z['next_radius']):
                goal = safe
                self.goal_label = 'zone'
            if goal is None:
                goal, lab = self.pick_goal(obs, me, kinds, hp_frac, act)
                self.goal_label = lab
            if goal is None and safe is not None:
                goal = safe
                self.goal_label = 'zone-drift'
            if goal is not None and mode != 'fight':
                move_yaw = self.go_to(obs, pos, goal, act, urgent=urgent)
            elif goal is not None:
                move_yaw = self.go_to(obs, pos, goal, act, urgent=True)
        if move_yaw is not None and 'move' not in act:
            act['move'] = [math.cos(move_yaw), math.sin(move_yaw)]

        # ---------------- pickups (opportunistic, within reach)
        self.try_take(obs, me, kinds, act)

        # ---------------- farm / look around
        if aim is None:
            aim = self.farm_fire(obs, me, npcs, kinds, weapons, act, hp_frac)
        if aim is None:
            aim = self.look_around(obs, me)
        act['aim'] = [aim[0], max(me['stats'].get('min_pitch', -0.7), min(me['stats'].get('max_pitch', 1.3), aim[1]))]

        # gun reload when idle
        for i, w in enumerate(weapons):
            if w and w['kind'] == 'gun' and w.get('reload', 0) == 0:
                if w.get('ammo', 1) == 0 or (mode != 'fight' and w.get('ammo', 12) < w['stats']['mag'] * 0.5 and not act['fire'][i]):
                    act['reload'] = True

        self.handle_stuck(obs, act)

        # path planning slices with remaining budget
        spent = time.perf_counter() - t_start
        self.plan.step(max(0.0, 0.022 - spent))
        mf.draw.label(f"{mode}/{self.goal_label} hp{me['hp']:.0f} c{me['cubes']}")
        return act

    # ------------------------------------------------------------------ fire control
    def fire_control(self, obs, me, target, act, kinds, weapons, muz):
        tc = self.center(target)
        tvel = target.get('vel', [0, 0, 0])
        d = target['_d']
        los = target['_los']
        cur = me['aim']
        aims = []
        s = self.scale(target.get('cubes', 0))
        rad = 2.5 * s
        primary = None
        for i, w in enumerate(weapons):
            if not w:
                continue
            k = w['kind']; st = w['stats']
            if k == 'gun':
                p, tt = self.lead(muz, tc, tvel, st['speed'], 0.05)
                a = self.aim_at(muz, p)
                err = math.hypot(wrap(cur[0] - a[0]) * math.cos(a[1]), cur[1] - a[1])
                tol = math.atan2(rad * 0.9, max(d, 1))
                if los and d < st['range'] and err < tol + 0.01 and w.get('reload', 0) == 0 and w.get('ammo', 1) > 0:
                    act['fire'][i] = True
                aims.append((2, a))
            elif k == 'laser':
                p, _ = self.lead(muz, tc, tvel, 1e9, 0.1)
                a = self.aim_at(muz, p)
                err = math.hypot(wrap(cur[0] - a[0]) * math.cos(a[1]), cur[1] - a[1])
                tol = math.atan2(rad * 0.7, max(d, 1))
                charge = w.get('charge', 0) or 0
                full = st['charge']
                if charge > 0:
                    if charge >= full - 0.05 and los and err < tol:
                        act['fire'][i] = False  # release
                    elif charge >= full + st.get('hold_s', 2) - 0.3 and los:
                        act['fire'][i] = False
                    else:
                        act['fire'][i] = True
                elif w.get('cooldown', 0) <= 0.05 and los and d < st['range'] * 0.95:
                    act['fire'][i] = True
                aims.append((3, a))
            elif k == 'lightning':
                a = self.aim_at(muz, tc)
                yerr = abs(wrap(cur[0] - a[0]))
                if los and d < st['range'] * 0.95 and yerr < math.radians(st['cone_deg']) * 0.5 and me['energy'] >= st['energy'] + 2:
                    act['fire'][i] = True
                aims.append((4 if d < st['range'] else 0, a))
            elif k == 'mortar':
                src = (me['pos'][0], me['pos'][1], me['pos'][2] + me['stats'].get('height', 9) + 0.5)
                tgt_ground = (tc[0], tc[1], target['pos'][2])
                r = None
                if d > 15:
                    r = self.ballistic(src, tgt_ground, st['speed'], st['gravity'], high=not los)
                    if r:
                        _, tf = r
                        p = (tgt_ground[0] + tvel[0] * tf, tgt_ground[1] + tvel[1] * tf, tgt_ground[2])
                        r = self.ballistic(src, p, st['speed'], st['gravity'], high=not los)
                if r:
                    a, tf = r
                    err = math.hypot(wrap(cur[0] - a[0]), cur[1] - a[1])
                    if err < 0.04 and w.get('cooldown', 0) <= 0:
                        act['fire'][i] = True
                    aims.append((1, a))
        if not aims:
            return None
        aims.sort(key=lambda x: -x[0])
        # lightning in range wins aim; else laser; else gun
        return aims[0][1]

    def farm_fire(self, obs, me, npcs, kinds, weapons, act, hp_frac):
        if hp_frac < 0.35:
            return None
        pos = me['pos']
        eye = self.eye(me)
        muz = self.muzzle(me)
        cands = []
        for n in npcs:
            dd = d2(pos, n['pos'])
            if dd < 110 and self.w.los(eye, self.center(n)):
                cands.append((dd, n, 'npc'))
        for c in obs['crates']:
            dd = d2(pos, c['pos'])
            if dd < 70:
                cc = {'pos': c['pos'], 'vel': [0, 0, 0], 'kind': 'npc', 'cube_h': 1.2}
                if self.w.los(eye, (c['pos'][0], c['pos'][1], c['pos'][2] + 1.2)):
                    cands.append((dd + 15, cc, 'crate'))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        dd, tgt, _ = cands[0]
        tgt = dict(tgt)
        tgt['_d'] = dd
        tgt['_los'] = True
        return self.fire_control(obs, me, tgt, act, kinds, weapons, muz)

    def look_around(self, obs, me):
        now = obs['t']
        if self.threat_bearing is not None and now - self.threat_t < 3.0:
            return [self.threat_bearing, 0.0]
        if self.sound_look and now - self.sound_look[2] < 2.0:
            return [self.sound_look[1], 0.0]
        # remembered enemy
        best = None
        for k, e in self.mem.items():
            if e.get('kind') == 'mech' and now - e['t'] < 8:
                dd = d2(me['pos'], e['pos'])
                if best is None or dd < best[0]:
                    best = (dd, e)
        if best and best[0] < 250 and now - best[1]['t'] < 4:
            e = best[1]
            return [math.atan2(e['pos'][1] - me['pos'][1], e['pos'][0] - me['pos'][0]), 0.0]
        return [wrap(now * 2.6 + self.slot), 0.0]

    # ------------------------------------------------------------------ goals
    def want_pickup(self, kinds, kind):
        if kind in kinds:
            return None
        v = WEAPON_VALUE.get(kind, 0)
        if None in kinds:
            if v >= 1.4:
                return kinds.index(None)
            return None
        vals = [WEAPON_VALUE.get(k, 0) for k in kinds]
        lo = min(range(2), key=lambda i: vals[i])
        if v > vals[lo] + 0.6:
            return lo
        return None

    def try_take(self, obs, me, kinds, act):
        pos = me['pos']
        reach = self.cfg['loot']['pickup_radius']
        for p in obs['pickups']:
            if d2(pos, p['pos']) < reach - 0.3 and abs(p['pos'][2] - pos[2]) < 3.8:
                slot = self.want_pickup(kinds, p['kind'])
                if slot is None:
                    continue
                w = me['weapons'][slot] if slot < len(me['weapons']) else None
                if w and w['kind'] == 'laser' and (w.get('charge') or 0) > 0:
                    continue
                if act['fire'][slot]:
                    continue
                act['take'] = p['id']
                if kinds[slot] is not None:
                    act['drop'] = slot
                return

    def pick_goal(self, obs, me, kinds, hp_frac, act):
        pos = me['pos']
        now = obs['t']
        P = self.P
        known = [e for e in self.mem.values() if e.get('kind') == 'mech' and now - e['t'] < 6]
        # supply drops
        for dr in obs.get('drops', []):
            dd = d2(pos, dr['pos'])
            eta = dd / 12.0
            if eta < dr['lands_in'] + 25 and dd < 260:
                close = [e for e in known if d2(e['pos'], dr['pos']) < P['drop_enemy_clear']]
                if not close and (dr.get('holder') in (None, me['id'])):
                    if dd < 6 or (dr['lands_in'] > 0 and dd < 30):
                        act['move'] = [0, 0]  # hold position on the drop
                        return (pos[0], pos[1]), 'drop-hold'
                    return tuple(dr['pos'][:2]), 'drop'
        # low hp: break line of sight / heal
        if hp_frac < 0.55 and known:
            cover = self.find_cover(pos, known)
            if cover:
                return cover, 'cover'
        # cubes
        best = None
        for c in obs['cubes']:
            dd = d2(pos, c['pos'])
            if dd < P['loot_radius'] and abs(c['pos'][2] - pos[2]) < 3.5 and (best is None or dd < best[0]):
                best = (dd, tuple(c['pos'][:2]), 'cube')
        if best:
            return best[1], best[2]
        # weapon pickups
        for p in obs['pickups']:
            dd = d2(pos, p['pos'])
            if dd < P['loot_radius'] and abs(p['pos'][2] - pos[2]) < 3.5 and self.want_pickup(kinds, p['kind']) is not None:
                if now - self.taken_fail.get(p['id'], -99) < 5:
                    continue
                v = WEAPON_VALUE.get(p['kind'], 0)
                s = dd - v * 15
                if best is None or s < best[0]:
                    best = (s, tuple(p['pos'][:2]), 'loot:' + p['kind'])
        if best:
            return best[1], best[2]
        # npcs to farm: approach within 70 m
        if hp_frac > P['farm_hp'] and now < 300:
            bn = None
            for n in obs['mechs']:
                if n['kind'] == 'npc':
                    dd = d2(pos, n['pos'])
                    if dd < 160 and (bn is None or dd < bn[0]):
                        bn = (dd, n)
            if bn and bn[0] > 60:
                return tuple(bn[1]['pos'][:2]), 'npc'
            if bn:
                return None, 'npc-shoot'
        # crates visible
        bc = None
        for c in obs['crates']:
            dd = d2(pos, c['pos'])
            if dd < 150 and abs(c['pos'][2] - pos[2]) < 3 and (bc is None or dd < bc[0]):
                bc = (dd, c)
        if bc and bc[0] > 50:
            return tuple(bc[1]['pos'][:2]), 'crate'
        # explore toward weapons we saw? otherwise drift toward zone center
        z = obs['zone']
        nc = z['next_center']
        if d2(pos, nc) > z['next_radius'] * 0.35:
            return tuple(nc), 'center'
        return None, 'hold'

    def find_cover(self, pos, enemies):
        best = None
        for r in (12, 24, 36):
            for k in range(12):
                a = k * TAU / 12
                c = (pos[0] + math.cos(a) * r, pos[1] + math.sin(a) * r)
                if not self.w.free(*c):
                    continue
                z = self.w.h(*c)
                hidden = True
                for e in enemies:
                    ee = (e['pos'][0], e['pos'][1], e['pos'][2] + 8)
                    if self.w.los(ee, (c[0], c[1], z + 8)):
                        hidden = False
                        break
                if hidden and self.walk_clear(pos, c):
                    s = r
                    if best is None or s < best[0]:
                        best = (s, c)
            if best:
                return best[1]
        return None


if __name__ == '__main__':
    mf.run(Bot)
