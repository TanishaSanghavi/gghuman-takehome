"""Frame renderer: Timeline -> RGB frames.  Pure function of (timeline, frame index)."""
import math, bisect
import numpy as np, cv2
from PIL import Image, ImageDraw
from .config import W, H, FPS, CREAM, INK, CAPTION_Y, TABLE_TOP, snap
from .art import chars as C, props as P, text as T, paper as PA, games as G
from .art.text import font
from .render_util import prem, blit, place, tint, vignette, scanlines, ease_out_back, ease_out_bounce, smooth

BUCKETS = (0.3, 0.5, 0.75, 1.0, 1.3)
NIGHT_MUL = (0.20, 0.30, 0.50)
TRANS_DUR = {"whip": 0.16, "flash": 0.12, "dip": 0.22, "iris": 0.28}
WT = dict(cx=360, cy=775, rx=335, ry=150)   # wide-shot table


def wide_top(x):
    return WT["cy"] - WT["ry"] * math.sqrt(max(0.0, 1 - ((x - WT["cx"]) / WT["rx"]) ** 2))


def close_top(x):
    return 1620 - 800 * math.sqrt(max(0.0, 1 - ((x - 360) / 900) ** 2))


class Renderer:
    def __init__(self, tl_obj, spec=None):
        self.tl = tl_obj.data
        self.clips = tl_obj.clips
        self.shots = self.tl["shots"]
        self.t0s = [s["t0"] for s in self.shots]
        self.cast = self.tl["meta"]["cast"]
        self.hook = self.tl["meta"]["hook"]
        self.total = self.tl["meta"]["duration"]
        self.seats = (spec or {}).get("seats") or self.cast
        self.board_seed = int((spec or {}).get("board_seed", 0) or 0)
        self.pm = {}
        self._chunks()
        self._persist()
        h = self.hook
        self.hook_until = self.shots[0]["t1"] if "until" not in h else h["until"]

    # ------------------------------------------------------------ prep
    def _chunks(self):
        caps = sorted(self.tl["captions"], key=lambda c: c["t0"])
        for i, c in enumerate(caps):
            nxt = caps[i + 1]["t0"] if i + 1 < len(caps) else 1e9
            c["disp_end"] = min(nxt, c["t1"] + 0.14) if nxt - c["t1"] < 0.3 else c["t1"] + 0.12
        self.caps = caps
        self.cap_t0 = [c["t0"] for c in caps]

    def _persist(self):
        out, self.out = set(), {}
        for s in self.shots:
            for ch, st in s["spec"].get("state", {}).items():
                if st.get("out") is True:
                    out.add(ch)
                elif st.get("out") is False:
                    out.discard(ch)
            self.out[s["i"]] = set(out)
            if s["type"] == "secret" and s["spec"].get("action") == "eliminate" and not s["spec"].get("fails"):
                out.add(s["spec"]["target"])
            if s["type"] == "secret_split":
                for pn in s["spec"]["panels"]:
                    if pn.get("action") == "eliminate" and not pn.get("fails"):
                        out.add(pn["target"])

    # ------------------------------------------------------------ sprites
    def _bucket(self, target):
        for b in BUCKETS:
            if b >= target * 0.85:
                return b
        return BUCKETS[-1]

    def head(self, who, ex, target, boil, layer="all", night=False):
        b = self._bucket(target)
        k = ("h", who, ex, b, boil, layer, night)
        if k not in self.pm:
            img, ox, oy = C.head_sprite(who, ex, b, boil, layer)
            pm = prem(img)
            if night and layer == "all":
                pm = tint(pm, NIGHT_MUL)
            self.pm[k] = (pm, ox, oy, b)
        return self.pm[k]

    def torso(self, who, target, boil, night=False):
        b = self._bucket(target)
        k = ("b", who, b, boil, night)
        if k not in self.pm:
            img, ox, oy = C.body_sprite(who, b, boil)
            pm = prem(img)
            if night:
                pm = tint(pm, NIGHT_MUL)
            self.pm[k] = (pm, ox, oy, b)
        return self.pm[k]

    def arm(self, who, arm, target, boil, night=False):
        b = self._bucket(target)
        k = ("a", who, arm, b, boil, night)
        if k not in self.pm:
            img, ox, oy = C.arm_sprite(who, arm, b, boil)
            pm = prem(img)
            if night:
                pm = tint(pm, NIGHT_MUL)
            self.pm[k] = (pm, ox, oy, b)
        return self.pm[k]

    def hat(self, who, target):
        b = self._bucket(target)
        k = ("hat", who, b)
        if k not in self.pm:
            img, ox, oy = C.hat_sprite(who, b)
            self.pm[k] = (prem(img), ox, oy, b)
        return self.pm[k]

    def prop(self, key, fn):
        if key not in self.pm:
            self.pm[key] = prem(fn())
        return self.pm[key]

    # ------------------------------------------------------------ state
    def active_line(self, shot, t):
        for l in shot["lines"]:
            if l["t0"] - 1e-6 <= t < l["t1"]:
                return l
        return None

    def state(self, shot, who, t):
        st = dict(expr="neutral", arm=None, gaze=None, anim=None, anim_t=-9.0, arm_t=None, out=who in self.out[shot["i"]])
        for k, v in shot["spec"].get("state", {}).get(who, {}).items():
            if k in st:
                st[k] = v
        evs = [(l["t0"], {"expr": l["expr"]}) for l in shot["lines"] if l["who"] == who and l["expr"]]
        evs += [(b["t"], b["set"]) for b in shot["beats"] if b["who"] == who]
        for te, d in sorted(evs, key=lambda e: e[0]):
            if te <= t + 1e-6:
                st.update(d)
                if "anim" in d:
                    st["anim_t"] = te
                if "arm" in d:
                    st["arm_t"] = te
        if st["out"]:
            st["expr"] = "dead"
            st["arm"] = None
        return st

    def mouth_level(self, line, t):
        clip = self.clips[line["id"]]
        i = int((t - line["t0"] + line.get("onset", 0.0)) / 0.01)
        e = float(clip.env[min(max(i, 0), len(clip.env) - 1)])
        return 0 if e < 0.10 else 1 if e < 0.32 else 2 if e < 0.62 else 3

    # ------------------------------------------------------------ one character
    def draw_char(self, dst, shot, who, t, neck, k, night=False, speaker=None, phase=0.0, order="both", head_scale_boost=1.0, force_expr=None):
        """Draw torso+head at neck=(x,y) with display scale k (px per design unit).
        Returns a deferred-arm callable (drawn after the table)."""
        st = self.state(shot, who, t)
        line = self.active_line(shot, t)
        speaking = line is not None and line["who"] == who
        frozen = st["anim"] == "freeze" and t >= st["anim_t"]
        boil = 0 if frozen else (int(t * 8 + phase * 3) % 3)
        lvl = self.mouth_level(line, t) if speaking else 0
        gx, gy = 0, 0
        if st["gaze"]:
            gx, gy = st["gaze"]
        elif speaker is not None and not speaking and not st["out"]:
            gx = int(np.sign(speaker[0] - neck[0]))
        blink = (not frozen) and ((t * 1.0 + phase * 2.3) % 3.1) < 0.11
        name = force_expr or st["expr"]
        ex = C.expr_key(name, lvl, gx, gy, ())
        if blink and ex[0] in ("dot", "wide") and not st["out"] and name != "shock":
            ex = ("closed",) + ex[1:]
        if st["out"]:
            ex = C.expr_key("dead")
        # procedural motion
        dx = dy = rot = 0.0
        sc = 1.0
        bob = 0.0 if frozen else math.sin(2 * math.pi * 0.7 * t + phase * 6) * 4 * k / 0.6 * (0.6 if speaking else 0.35)
        if speaking:
            bob += math.sin(2 * math.pi * 3.3 * t) * 3.0 * min(1, k + 0.3) * (0.4 + 0.2 * lvl)
        dt = t - st["anim_t"]
        a = st["anim"]
        if a == "hop" and 0 <= dt < 0.28:
            dy -= 26 * k * math.sin(math.pi * dt / 0.28) / 0.6
            sc = 1 + 0.06 * math.sin(math.pi * dt / 0.28)
        elif a == "shake" and dt >= 0:
            dx += math.sin(t * 90) * 4.5 * k / 0.6
        elif a == "lean" and dt >= 0:
            sc = 1 + 0.07 * smooth(dt / 0.35)
            dy += 14 * smooth(dt / 0.35) * k / 0.6
        elif a == "recoil" and dt >= 0:
            p = smooth(dt / 0.18)
            dx += 20 * p * k / 0.6
            rot = -6 * p
        if speaker is None and speaking and line is not None:
            dt2 = t - line["t0"]
            if 0 <= dt2 < 0.2:
                sc *= 1 + 0.045 * math.sin(math.pi * dt2 / 0.2)
        if st["out"]:
            rot, dy = 24, 55 * k / 0.6
        pose_arm = None
        # torso
        tb, bx, by, bb = self.torso(who, k, boil, night)
        if order in ("both", "body"):
            place(dst, tb, bx, by, neck[0], neck[1] + (14 * k if st["out"] else 0), k / bb, k / bb * (1 - 0.01 * math.sin(t * 4 + phase)))
        if order in ("both", "head"):
            hp, hx, hy, hb = self.head(who, ex, k * sc, boil, "all", night)
            place(dst, hp, hx, hy, neck[0] + dx, neck[1] + dy + bob, k * sc * head_scale_boost / hb, rot=rot)
            if night:
                gk = ("glow", who, ex, self._bucket(k * sc), boil)
                if gk not in self.pm:
                    gl, gx_, gy_, gb = self.head(who, ex, k * sc, boil, "eyes", False)
                    halo = cv2.GaussianBlur(gl, (0, 0), 9 * gb + 2)
                    halo[..., 3] = np.clip(halo[..., 3].astype(np.float32) * 2.2, 0, 255).astype(np.uint8)
                    halo[..., :3] = np.clip(halo[..., :3].astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
                    self.pm[gk] = (gl, halo, gx_, gy_, gb)
                gl, halo, gx_, gy_, gb = self.pm[gk]
                place(dst, halo, gx_, gy_, neck[0] + dx, neck[1] + dy + bob, k * sc / gb, rot=rot, opacity=0.8)
                place(dst, gl, gx_, gy_, neck[0] + dx, neck[1] + dy + bob, k * sc / gb, rot=rot)
        return st, (dx, dy + bob)

    def arm_target(self, who, arm_s, neck, k, targets, side_hint=1):
        """'point:gemini' / 'point_r' / 'point_l' / 'up' / 'both_up' -> arm tuple (px-space angle, reach)."""
        if not arm_s:
            return None
        if arm_s in ("up", "both_up"):
            return ("up", side_hint) if arm_s == "up" else ("both_up",)
        if arm_s.startswith("point:") and arm_s[6:] in targets:
            tx, ty = targets[arm_s[6:]]
            side = 1 if tx >= neck[0] else -1
            sx, sy = neck[0] + 300 * k * side, neck[1] + 130 * k
            ang = math.degrees(math.atan2(ty - sy, tx - sx))
            dist = math.hypot(tx - sx, ty - sy) / k
            return ("point", int(round(ang / 5) * 5), side, int(min(380, max(260, round((dist - 60) / 40) * 40))))
        if arm_s == "point_r":
            return ("point", -12, 1, 400)
        if arm_s == "point_l":
            return ("point", 192, -1, 400)
        if arm_s == "point_down":
            return ("point", 90, 1, 300)
        return None

    def draw_arm(self, dst, who, arm, neck, k, boil, night=False, t=0.0, st=None):
        if arm is None:
            return
        pm, ax, ay, b = self.arm(who, arm, k, boil, night)
        grow = 1.0
        if st and st.get("arm_t") is not None:
            grow = ease_out_back((t - st["arm_t"]) / 0.18, 1.2)
        place(dst, pm, ax, ay, neck[0], neck[1], k / b * max(0.05, grow), k / b * max(0.05, grow))

    # ================================================================ scenes
    def _bg(self):
        return PA.paper().copy()

    def _table_close(self, dst, dark=False):
        k = ("tc", dark)
        if k not in self.pm:
            self.pm[k] = prem(PA.table_close(dark))
        blit(dst, self.pm[k], 0, 0)

    def _table_wide(self, dst, dark=False, color=None):
        k = ("tw", dark, color)
        if k not in self.pm:
            self.pm[k] = prem(PA.table_wide(WT["cx"], WT["cy"], WT["rx"], WT["ry"], dark, 1.0, color))
        blit(dst, self.pm[k], 0, 0)

    SEATS = {5: [(70, .36), (200, .30), (360, .31), (520, .30), (650, .36)], 4: [(100, .37), (260, .33), (460, .33), (620, .37)],
             3: [(130, .40), (360, .35), (590, .40)], 2: [(200, .46), (520, .46)], 1: [(360, .5)]}

    def seat_layout(self, seats=None):
        seats = [c for c in (seats or self.seats) if c in self.cast]
        spec = self.SEATS[min(5, len(seats))]
        return {who: (float(x), wide_top(x) + 3, k) for who, (x, k) in zip(seats, spec)}

    def scene_wide(self, shot, t, night=False, heads_down=False, show_names=True, vote=None):
        img = self._bg() if not night else np.full((H, W, 3), (14, 18, 30), np.uint8)
        pos = self.seat_layout()
        line = self.active_line(shot, t)
        speaker = None
        if line and line["who"] in pos:
            sx, sy, sk = pos[line["who"]]
            speaker = (sx, sy)
        order = sorted(pos, key=lambda w: pos[w][1])
        arms = []
        heads = {w: (pos[w][0], pos[w][1] - 230 * pos[w][2]) for w in pos}
        for who in order:
            x, y, k = pos[who]
            st = self.state(shot, who, t)
            fe = "sleep" if heads_down else None
            neck = (x, y + (30 * k / 0.3 if heads_down else 0))
            _, off = self.draw_char(img, shot, who, t, neck, k, night=night, speaker=speaker, phase=hash(who) % 7 / 7.0, force_expr=fe)
            armspec = st["arm"]
            if vote is not None and who in vote:
                armspec = "point:" + vote[who]
            if armspec and not st["out"]:
                at = self.arm_target(who, armspec, neck, k, heads, 1 if x < 360 else -1)
                arms.append((who, at, neck, k, st))
        self._table_wide(img, dark=night)
        # little card deck on the table
        if not night:
            cb = self.prop(("cb", 0), lambda: P.card_back(0))
            for j, (dx, dy, r) in enumerate(((-18, 0, -8), (0, -4, 3), (16, 2, 10))):
                place(img, cb, 150, 200, 360 + dx, WT["cy"] + 4 + dy, 0.2, 0.2, r)
        boil = int(t * 8) % 3
        for who, at, neck, k, st in arms:
            self.draw_arm(img, who, at, neck, k, boil, night, t, st)
        for who in pos:
            x, y, k = pos[who]
            if night:
                continue
            if show_names:
                nm = T.plain(C.CAST[who]["name"], 30, (60, 54, 50))
                blit(img, prem(nm), int(x - nm.shape[1] / 2), int(WT["cy"] + WT["ry"] + 14 + 30 * (abs(x - 360) < 100)))
            if self.state(shot, who, t)["out"]:
                tag = self.prop(("out", 0), lambda: P.out_tag(0))
                place(img, tag, 100, 45, x, y - 230 * k * 2.1, 0.55, 0.55, -8)
        return img

    def scene_two(self, shot, t, night=False):
        L, R = shot["spec"]["pair"]
        img = self._bg()
        k = 0.66
        xs = {L: 190.0, R: 530.0}
        line = self.active_line(shot, t)
        speaker = (xs[line["who"]], 0) if line and line["who"] in xs else None
        arms = []
        heads = {w: (xs[w], close_top(xs[w]) - 230 * k - 14 * k) for w in xs}
        for who in (L, R):
            x = xs[who]
            neck = (x, close_top(x) - 14 * k)
            kk = k * (1.06 if line and line["who"] == who else 0.98)
            st, _ = self.draw_char(img, shot, who, t, neck, kk, speaker=speaker, phase=hash(who) % 7 / 7.0)
            if st["arm"] and not st["out"]:
                at = self.arm_target(who, st["arm"], neck, kk, heads, 1 if x < 360 else -1)
                arms.append((who, at, neck, kk, st))
        self._table_close(img)
        for who, at, neck, kk, st in arms:
            self.draw_arm(img, who, at, neck, kk, int(t * 8) % 3, False, t, st)
        for who in (L, R):
            if self.state(shot, who, t)["out"]:
                tag = self.prop(("out", 0), lambda: P.out_tag(0))
                place(img, tag, 100, 45, xs[who], 470, 0.9, 0.9, -8)
        return img

    def scene_close(self, shot, t):
        who = shot["spec"]["who"]
        if isinstance(who, list):
            who = who[0]
        img = self._bg()
        k = 1.18
        neck = (360.0, close_top(360) - 18 * k)
        st, _ = self.draw_char(img, shot, who, t, neck, k, speaker=None, phase=hash(who) % 7 / 7.0)
        self._table_close(img)
        if st["arm"] and not st["out"]:
            at = self.arm_target(who, st["arm"], neck, k, {}, 1)
            self.draw_arm(img, who, at, neck, k * 0.8, int(t * 8) % 3, False, t, st)
        if st["out"]:
            tag = self.prop(("out", 0), lambda: P.out_tag(0))
            place(img, tag, 100, 45, 360, 250, 1.4, 1.4, -8)
        if shot["spec"].get("mark"):
            m = shot["spec"]["mark"]
            mk = self.prop(("mark", m["kind"]), lambda: P.mark(m["kind"]))
            p = ease_out_back((t - shot["t0"] - m.get("at", 0.2)) / 0.25)
            if p > 0:
                place(img, mk, 70, 80, m.get("x", 560), m.get("y", 260), 1.3 * p, 1.3 * p, m.get("rot", 10))
        return img

    def scene_blackout(self, shot, t):
        local = t - shot["t0"]
        if local < 0.07:
            img = self.scene_wide(shot, t)
            return img
        img = self.scene_wide(shot, t, night=True, heads_down=True, show_names=False)
        img = (img.astype(np.float32) * vignette(H, W, 0.5, 0.52)).astype(np.uint8)
        pos = self.seat_layout()
        for j, who in enumerate(pos):
            x, y, k = pos[who]
            z = self.prop(("zzz", 0), lambda: P.mark("zzz"))
            ph = (local * 1.2 + j * 0.37) % 1.0
            place(img, z, 70, 80, x + 55, y - 190 * k * 1.5 - 70 * ph, 0.5, 0.5, 8, opacity=0.8 * (1 - ph))
        p = ease_out_back((local - 0.5) / 0.3)
        if p > 0:
            txt = T.chunky("LIGHTS OUT", 150, (250, 244, 228), (10, 10, 18), 12)
            flick = 1.0 if (local < 1.0 or int(local * 14) % 9) else 0.5
            place(img, prem(txt), txt.shape[1] / 2, txt.shape[0] / 2, 360, 300, p, p, -4, opacity=flick)
        return img

    def glow_prop(self, key, fn, night=True):
        k = (key, night)
        if k not in self.pm:
            pm = prem(fn())
            self.pm[k] = tint(pm, (0.55, 0.7, 0.9)) if night else pm
        return self.pm[k]

    # ================================================================ secret camera
    def _noise_bank(self):
        if "noise" not in self.pm:
            rng = np.random.default_rng(7)
            self.pm["noise"] = [cv2.resize(rng.normal(0, 7, (H // 3, W // 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) for _ in range(6)]
        return self.pm["noise"]

    def secret_panel(self, shot, t, who, action, target, stamp_t, pw, ph, u, label="MAFIA CAM", idx=0, show_iris=True, fails=False, hand=None, hud_t=None):
        local = t - shot["t0"]
        rows = np.linspace(0, 1, ph)[:, None]
        base = (np.array((9, 16, 22)) * (1 - rows) + np.array((20, 36, 40)) * rows).astype(np.uint8)
        cv = np.repeat(base[:, None, :], pw, axis=1).copy()
        # moonlight beam
        beam = np.zeros((ph, pw, 3), np.uint8)
        cv2.fillConvexPoly(beam, np.array([[pw * .78, 0], [pw * 1.0, 0], [pw * .75, ph * .8], [pw * .35, ph * .8]], np.int32), (22, 30, 28), cv2.LINE_AA)
        cv = cv2.add(cv, beam)
        cv2.ellipse(cv, (pw // 2, int(ph * 1.17)), (int(pw * 0.98), int(ph * 0.45)), 0, 0, 360, (32, 27, 33), -1, cv2.LINE_AA)
        cv2.ellipse(cv, (pw // 2, int(ph * 1.17)), (int(pw * 0.98), int(ph * 0.45)), 0, 200, 340, (6, 8, 12), max(2, int(6 * u)), cv2.LINE_AA)
        peek = action == "peek"
        hand_mode = action == "hand"
        ax = pw * (0.5 if (peek or hand_mode) else 0.36)
        k = (0.72 if hand_mode else 0.95) * u
        neck = (ax, ph * (0.37 if hand_mode else 0.56))
        fp = stamp_t - shot["t0"]
        done = t >= stamp_t
        expr = ("evil" if done else "sus") if not (peek or hand_mode) else ("evil" if t >= stamp_t + (0.15 if hand_mode else 0.12) else "neutral")
        # card + arm geometry
        card_c = (pw * (0.5 if peek else 0.66), ph * 0.80)
        st, _ = self.draw_char(cv, shot, who, t, neck, k, night=True, phase=0.3 + idx, force_expr=expr)
        if hand_mode:
            n = len(hand)
            sc = (1.08 if n <= 2 else 0.72 if n <= 3 else 0.56) * u
            w0 = 200 * sc
            sp = min(w0 * 1.1, (pw * 0.9 - w0) / max(1, n - 1)) if n > 1 else 0
            rise = smooth(local / 0.45)
            back = self.prop(("cb", 0), lambda: P.card_back(0))
            for i, code in enumerate(hand):
                off = i - (n - 1) / 2
                cx_ = pw / 2 + off * sp
                cy_ = ph * 0.665 + (1 - rise) * 330 * u + abs(off) * (6 * u if n > 2 else 0)
                p = (t - stamp_t - 0.09 * i) / 0.28
                sx = abs(math.cos(math.pi * min(max(p, 0), 1)))
                if p >= 0.5:
                    face = self.prop(("cardface", code), lambda code=code: G.card_face(code, 0))
                    face = tint(face, (0.86, 0.95, 1.0))
                else:
                    face = tint(back, (0.62, 0.74, 0.95))
                place(cv, face, 100, 140, cx_, cy_, sc * max(sx, 0.03), sc, off * (4.5 if n > 2 else 0) + (0 if n > 2 else off * 3))
            if t >= stamp_t + 0.1:
                halo = np.zeros_like(cv)
                cv2.circle(halo, (int(pw / 2), int(ph * 0.665)), int(260 * u), (70, 60, 10), -1)
                halo = cv2.GaussianBlur(halo, (0, 0), 50 * u)
                cv = cv2.add(cv, (halo * 0.55).astype(np.uint8))
        elif peek:
            rise = smooth(local / 0.45)
            cy = card_c[1] + (1 - rise) * 320 * u
            p = (t - stamp_t) / 0.32
            back = self.prop(("cb", 0), lambda: P.card_back(0))
            role = "MAFIA"
            front = self.prop(("role", role), lambda: P.role_card(role, 0))
            sx = abs(math.cos(math.pi * min(max(p, 0), 1)))
            face = front if p >= 0.5 else back
            face = tint(face, (0.62, 0.74, 0.95))
            place(cv, face, 150, 200, card_c[0], cy, 1.05 * u * max(sx, 0.03), 1.05 * u)
            if p >= 1:
                pulse = 0.5 + 0.5 * math.sin((t - stamp_t) * 6)
                halo = np.zeros_like(cv)
                cv2.circle(halo, (int(card_c[0]), int(cy)), int(220 * u), (60, 20, 20), -1)
                halo = cv2.GaussianBlur(halo, (0, 0), 40 * u)
                cv = cv2.add(cv, (halo * (0.5 + 0.3 * pulse)).astype(np.uint8))
        else:
            pc = self.prop(("pc", target), lambda: P.portrait_card(target, 1.0, 0, False))
            pcd = self.prop(("pcd", target), lambda: P.portrait_card(target, 1.0, 0, True))
            rise = smooth(local / 0.45)
            cy = card_c[1] + (1 - rise) * 320 * u
            card = tint(pcd if (done and not fails) else pc, (0.5, 0.58, 0.78) if done else (0.62, 0.74, 0.95))
            place(cv, card, 125, 165, card_c[0], cy, 1.15 * u, 1.15 * u, -3)
            # reaching arm
            arm_on = stamp_t - 0.6
            if t >= arm_on:
                heads = {"__c": (card_c[0] - 40 * u, card_c[1] - 215 * u)}
                at = self.arm_target(who, "point:__c", neck, k, heads, 1)
                tmp = dict(st)
                tmp["arm_t"] = arm_on
                self.draw_arm(cv, who, at, neck, k, int(t * 8) % 3, True, t, tmp)
            if t >= stamp_t:
                xm = self.prop(("x", 1.0), lambda: P.x_mark(1.0, 220, 0))
                pr = min(1.0, (t - stamp_t) / 0.22)
                xp = self.prop(("xp", round(pr * 6)), lambda pr=pr: P.x_mark(round(pr * 6) / 6, 220, 0))
                place(cv, xp, 110, 110, card_c[0], cy, 1.3 * u, 1.3 * u, -4)
                flash = math.exp(-(t - stamp_t) * 9)
                if flash > 0.02:
                    cv = cv2.addWeighted(cv, 1, np.full_like(cv, (70, 14, 14)), float(flash), 0)
        # grade (one multiply + one add on the full panel)
        gk = ("grade", ph, pw)
        if gk not in self.pm:
            self.pm[gk] = (vignette(ph, pw, 0.7, 0.5) * scanlines(ph, pw) * np.array([0.98, 1.06, 1.0], np.float32)).astype(np.float32)
        f = cv.astype(np.float32) * self.pm[gk] * (1 + 0.025 * math.sin(t * 61))
        f += self._noise_bank()[int(t * 30) % 6][:ph, :pw, None]
        out = np.clip(f, 0, 255).astype(np.uint8)
        # HUD
        pil = Image.fromarray(out)
        d = ImageDraw.Draw(pil)
        fm = font("Nunito.ttf", max(14, int(30 * u)), 800)
        mono = font("PatrickHand-Regular.ttf", max(14, int(30 * u)))
        mg = int(26 * u)
        if int(t * 2) % 2 == 0:
            d.ellipse([mg, mg + 4 * u, mg + 22 * u, mg + 26 * u], fill=(235, 50, 50))
        d.text((mg + 32 * u, mg - 2 * u), f"REC  {label}", font=fm, fill=(235, 240, 235))
        tc = 134 + local
        d.text((pw - mg - 170 * u, mg - 2 * u), f"02:{int(tc // 60) % 60:02d}:{int(tc) % 60:02d}:{int((tc % 1) * 30):02d}", font=fm, fill=(210, 225, 215))
        hud = None
        if hand_mode:
            hh = hud_t or ["HIDDEN FROM YOU", "HAND REVEALED"]
            hud = hh[1] if t >= stamp_t + 0.25 else hh[0]
        elif peek:
            hud = "ROLE: MAFIA" if t >= stamp_t + 0.2 else ("ROLE: ???" if local > 0.4 else "")
        elif t >= stamp_t - 0.6:
            hud = f"TARGET: {C.CAST[target]['name']}" if not done else ("KILL REJECTED" if fails else f"{C.CAST[target]['name']}  ELIMINATED")
        if hud:
            col = (255, 214, 90) if (done and fails) else (255, 90, 90) if (done and not peek) or (peek and t >= stamp_t + 0.2) else (255, 214, 90)
            d.text((mg, mg + 44 * u), hud, font=mono, fill=col)
        # corner brackets
        L = int(46 * u)
        wcol = (230, 236, 230)
        b = int(10 * u)
        for (x0, y0, sx_, sy_) in ((b, b, 1, 1), (pw - b, b, -1, 1), (b, ph - b, 1, -1), (pw - b, ph - b, -1, -1)):
            d.line([(x0, y0 + sy_ * L), (x0, y0), (x0 + sx_ * L, y0)], fill=wcol, width=max(2, int(4 * u)))
        out = np.asarray(pil).copy()
        if show_iris and local < 0.24:
            r = int(math.hypot(pw, ph) / 2 * smooth(local / 0.24))
            mask = np.zeros((ph, pw), np.uint8)
            cv2.circle(mask, (pw // 2, ph // 2), max(r, 1), 255, -1)
            out[mask == 0] = (4, 6, 8)
        return out

    def scene_secret(self, shot, t):
        s = shot["spec"]
        img = np.full((H, W, 3), (6, 8, 12), np.uint8)
        panel = self.secret_panel(shot, t, s["who"], s["action"], s.get("target"), shot["stamp_t"][0], W, H, 1.0, fails=s.get("fails", False))
        img[:] = panel
        return img

    def scene_split(self, shot, t):
        s = shot["spec"]
        img = np.full((H, W, 3), (6, 8, 12), np.uint8)
        ph, pw = 500, 704
        for i, pn in enumerate(s["panels"]):
            y0 = 40 + i * 520
            panel = self.secret_panel(shot, t, pn["who"], pn["action"], pn.get("target"), shot["stamp_t"][i], pw, ph, 0.5,
                                      label=f"CAM {i + 1}", idx=i, fails=pn.get("fails", False))
            img[y0:y0 + ph, 8:8 + pw] = panel
        cv2.line(img, (0, 540), (W, 540), (4, 5, 8), 5)
        return img

    # ================================================================ vote / reveal
    def scene_vote(self, shot, t):
        s = shot["spec"]
        votes, vts = s.get("votes", []), shot.get("vote_t", [])
        cast_votes, counts = {}, {}
        for v, tt in zip(votes, vts):
            if t >= tt - 1e-6:
                cast_votes[v["from"]] = v["to"]
                counts[v["to"]] = counts.get(v["to"], 0) + 1
        img = self.scene_wide(shot, t, vote=cast_votes)
        names = tuple(c for c in self.seats if c in self.cast and c not in self.out[shot["i"]])
        cnt = tuple(counts.get(n, 0) for n in names)
        win = None
        if vts and t >= vts[-1] + 0.35 and counts:
            win = max(counts, key=counts.get)
        board = self.prop(("tally", names, cnt, win), lambda: P.tally_board(names, cnt, win, 0))
        place(img, board, board.shape[1] / 2, 0, 360, 36, 0.8, 0.8)
        return img

    def scene_reveal(self, shot, t):
        s = shot["spec"]
        local = t - shot["t0"]
        img = self._bg()
        P.burst(img, 360, 600, t, alpha=0.5 * min(1, local / 0.15))
        whos = s["who"] if isinstance(s["who"], list) else [s["who"]]
        n = len(whos)
        spots = [(360.0, 925.0, 0.95)] if n == 1 else [(190.0, 925.0, 0.64), (530.0, 925.0, 0.64)]
        for i, (who, (x, y, k)) in enumerate(zip(whos, spots)):
            shake = math.sin(local * 70) * 6 * math.exp(-local * 4) if local < 1.2 else 0
            neck = (x + shake, y)
            self.draw_char(img, shot, who, t, neck, k, force_expr=s.get("expr", "busted"), phase=i * 0.4)
            hp, hx, hy, hb = self.hat(who, k)
            hp_t = (local - 0.10 - 0.07 * i) / 0.5
            if hp_t > 0:
                fall = (1 - ease_out_bounce(hp_t)) * 760 * k
                place(img, hp, hx, hy, neck[0], neck[1] - fall, k / hb, rot=(1 - min(1, hp_t)) * -14)
        lines = s["title"].split("\n")
        cols = [(255, 214, 64), (232, 56, 43), (255, 255, 255)]
        for j, ln in enumerate(lines):
            tt = (local - 0.05 - 0.1 * j) / 0.3
            if tt <= 0:
                continue
            pr = ease_out_back(tt, 2.4)
            size = 190 if len(ln) <= 9 else 150 if len(ln) <= 12 else 118
            bm = T.chunky(ln, size, cols[j % 3], (20, 16, 14), 14, max_w=660)
            sh = math.sin(local * 55 + j) * 4 * math.exp(-local * 3)
            place(img, prem(bm), bm.shape[1] / 2, bm.shape[0] / 2, 360 + sh, 215 + j * 185, pr, pr, (-3 if j % 2 == 0 else 3) * (1 if local < 1 else 0.6))
        return img

    # ================================================================ camera + overlays
    def apply_cam(self, img, shot, t):
        local = t - shot["t0"]
        dur = max(0.1, shot["t1"] - shot["t0"])
        cam = shot["spec"].get("cam", {})
        ty = shot["type"]
        z0, z1 = cam.get("zoom", {"close": (1.0, 1.07), "two": (1.0, 1.05), "wide": (1.08, 1.14), "table": (1.0, 1.06), "insert": (1.0, 1.04), "private": (1.0, 1.03), "scream": (1.18, 1.0), "reveal": (1.12, 1.0), "vote": (1.08, 1.12)}.get(ty, (1.0, 1.02)))
        z = z0 + (z1 - z0) * smooth(local / dur)
        fx, fy = cam.get("focus", (360, 640) if ty not in ("wide", "vote") else (360, 600))
        dx, dy = cam.get("pan", (0, 0))
        dx, dy = dx * local / dur, dy * local / dur
        amp = cam.get("shake", 0) * math.exp(-local * 6)
        if ty == "reveal":
            amp += 14 * math.exp(-local * 7)
        if ty == "scream":
            amp += 16 * math.exp(-local * 4)
        for st in shot.get("stamp_t", []):
            if t >= st:
                amp += 9 * math.exp(-(t - st) * 11)
        if amp > 0.3:
            dx += math.sin(t * 97) * amp
            dy += math.cos(t * 83) * amp
        if abs(z - 1) < 1e-4 and abs(dx) < .2 and abs(dy) < .2:
            return img
        M = np.array([[z, 0, fx - z * fx + dx], [0, z, fy - z * fy + dy]], np.float32)
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def scene(self, shot, t):
        ty = shot["type"]
        return {"wide": self.scene_wide, "two": self.scene_two, "close": self.scene_close, "blackout": self.scene_blackout,
                "secret": self.scene_secret, "secret_split": self.scene_split, "vote": self.scene_vote, "reveal": self.scene_reveal, "table": self.scene_table, "insert": self.scene_insert,
                "private": self.scene_private, "scream": self.scene_scream, "endcard": self.scene_endcard}[ty](shot, t)

    def shot_img(self, shot, t):
        return self.apply_cam(self.scene(shot, t), shot, t)

    def transition(self, a, b, kind, p):
        if kind == "whip":
            sh = int(W * smooth(p))
            both = np.concatenate([a, b], axis=1)
            out = both[:, sh:sh + W].copy()
            kx = int(5 + 50 * math.sin(math.pi * p))
            return cv2.blur(out, (kx | 1, 1))
        if kind == "flash":
            return cv2.addWeighted(b, 1 - (1 - p) * 0.95, np.full_like(b, 255), (1 - p) * 0.95, 0)
        if kind == "dip":
            return (a * (1 - 2 * p)).astype(np.uint8) if p < 0.5 else (b * (2 * p - 1)).astype(np.uint8)
        if kind == "iris":
            m = np.zeros((H, W), np.uint8)
            cv2.circle(m, (W // 2, H // 2), int(math.hypot(W, H) / 2 * smooth(p)), 255, -1)
            out = a.copy()
            out[m > 0] = b[m > 0]
            return out
        return b

    def overlay(self, img, shot, t):
        if shot["type"] == "endcard":
            return img
        dark = shot["type"] in ("secret", "secret_split", "blackout", "private")
        local = t - shot["t0"]
        # --- POV hook (handwritten, top)
        if t < self.hook_until and shot["type"] not in ("secret", "secret_split"):
            fade = min(1.0, (self.hook_until - t) / 0.2)
            col = (232, 226, 214) if dark else (60, 54, 50)
            pv = T.plain(self.hook.get("pov", "POV:"), 62, col)
            blit(img, prem(pv), int(360 - pv.shape[1] / 2), 80, fade)
            import textwrap
            body = "\n".join(textwrap.wrap(self.hook["text"], 36))
            tb = T.plain(body, 46, col)
            blit(img, prem(tb), int(360 - tb.shape[1] / 2), 152, fade)
        elif t < self.hook_until and shot["type"] in ("secret", "secret_split") and False:
            pass
        # --- captions
        i = bisect.bisect_right(self.cap_t0, t + 1e-6) - 1
        if i >= 0 and t < self.caps[i]["disp_end"]:
            c = self.caps[i]
            y = 550 if shot["type"] == "secret_split" else (1035 if shot["type"] == "private" else CAPTION_Y)
            if shot["type"] == "reveal":
                y = 1010 if False else CAPTION_Y
            fill = (206, 224, 255) if c.get("style") == "whisper" else (255, 255, 255)
            cap = T.outlined(c["text"].upper(), 72, fill, (20, 16, 14), 8, 660)
            p = ease_out_back((t - c["t0"]) / 0.12, 2.6)
            sc = 0.75 + 0.25 * p
            place(img, prem(cap), cap.shape[1] / 2, cap.shape[0] / 2, 360, y, sc, sc, 0)
            if shot["type"] in ("wide", "vote", "blackout") and c["speaker"] in C.CAST:
                nm = T.outlined(C.CAST[c["speaker"]]["name"], 34, (255, 224, 140), (20, 16, 14), 6, 300)
                blit(img, prem(nm), int(360 - nm.shape[1] / 2), int(y - 78 - nm.shape[0] / 2))
        # --- end-card CTA (comment bait)
        cta = shot["spec"].get("cta")
        if cta and t >= shot["t1"] - 1.25:
            p = ease_out_back((t - (shot["t1"] - 1.25)) / 0.25, 1.8)
            tb = T.plain(cta, 62, (40, 34, 30), "PatrickHand-Regular.ttf")
            bgc = prem(np.dstack([np.full(tb.shape[:2], 250, np.uint8), np.full(tb.shape[:2], 246, np.uint8), np.full(tb.shape[:2], 236, np.uint8), np.full(tb.shape[:2], 235, np.uint8)]))
            place(img, bgc, bgc.shape[1] / 2, bgc.shape[0] / 2, 360, CAPTION_Y + 6, p * 1.12, p * 1.12, 0)
            place(img, prem(tb), tb.shape[1] / 2, tb.shape[0] / 2, 360, CAPTION_Y + 6, p, p, -2)
        # --- reaction words
        for r in shot["react"]:
            if r["t0"] - 1e-6 <= t < r["t1"]:
                lt = t - r["t0"]
                pr = ease_out_back(lt / 0.18, 2.4)
                fade = 1.0 if r["t1"] - t > 0.12 else (r["t1"] - t) / 0.12
                col = tuple(r.get("color", (255, 214, 64)))
                bm = T.chunky(r["text"], r.get("size", 124), col, (20, 16, 14), 11)
                x, y = r.get("pos", (360, 330))
                rot = r.get("rot", -6) + 2.5 * math.sin(lt * 18)
                place(img, prem(bm), bm.shape[1] / 2, bm.shape[0] / 2, x, y, pr, pr, rot, opacity=fade)
        # --- series furniture: round tag (top-left) and running tally (top-right)
        tg = shot["spec"].get("tag")
        if tg:
            tb = self.prop(("tag", tg), lambda: prem(T.chunky(tg, 62, (226, 106, 52), (20, 16, 14), 8, max_w=360)))
            place(img, tb, tb.shape[1] / 2, tb.shape[0] / 2, 22 + tb.shape[1] / 2, 62, 1.0, 1.0, -3)
        ty = shot["spec"].get("tally")
        if ty:
            rows = ty["rows"]
            if ty.get("to") and (t - shot["t0"]) >= ty.get("at", 1.0):
                rows = ty["to"]
            rk = tuple((r[0], r[1], r[2]) for r in rows)
            pan = self.prop(("tally", ty.get("title", "WINS"), rk), lambda: G.score_panel(rk, ty.get("title", "WINS"), None, 0))
            k = 0.46
            place(img, pan, pan.shape[1] / 2, 0, 720 - 16 - pan.shape[1] * k / 2, 14, k, k)
        # --- brand footer
        key = ("brand", dark)
        if key not in self.pm:
            b = T.brand().copy()
            if dark:
                b[..., :3] = np.where(b[..., :3] < 90, 235, b[..., :3] + 120).clip(0, 255).astype(np.uint8)
            self.pm[key] = prem(b)
        blit(img, self.pm[key], 22, 1082, 0.92 if dark else 1.0)
        return img

    def frame(self, fi):
        t = fi / FPS + 1e-6
        i = max(0, bisect.bisect_right(self.t0s, t) - 1)
        shot = self.shots[i]
        img = self.shot_img(shot, t)
        tr = shot["trans"]
        local = t - shot["t0"]
        if tr != "cut" and i > 0 and local < TRANS_DUR[tr]:
            prev = self.shot_img(self.shots[i - 1], shot["t0"] - 1 / FPS + 1e-6)
            img = self.transition(prev, img, tr, local / TRANS_DUR[tr])
        img = self.overlay(img, shot, t)
        return img

    # ================================================================ GGhuman-series scenes
    FELT = (112, 130, 108)

    def scene_table(self, shot, t):
        s = shot["spec"]
        L, R = s["pair"]
        poker = s["game"] == "poker"
        img = self._bg()
        xs = {L: 190.0, R: 530.0}
        line = self.active_line(shot, t)
        speaker = (xs[line["who"]], 0) if line and line["who"] in xs else None
        arms = []
        for who in (L, R):
            x = xs[who]
            neck = (x, wide_top(x) + 3)
            st, _ = self.draw_char(img, shot, who, t, neck, 0.5, speaker=speaker, phase=hash(who) % 7 / 7.0)
            if st["arm"] and not st["out"]:
                arms.append((who, self.arm_target(who, st["arm"], neck, 0.5, {}, 1 if x < 360 else -1), neck, st))
        self._table_wide(img, color=self.FELT if poker else None)
        cy = WT["cy"] + 4
        if poker:
            board = s.get("table_cards") or [None] * 5
            back = self.prop(("cb", 0), lambda: P.card_back(0))
            for i, code in enumerate(board):
                sp = ease_out_back((t - shot["t0"] - 0.05 * i) / 0.2)
                face = back if code is None else self.prop(("cardface", code), lambda code=code: G.card_face(code, 0))
                place(img, face, 100, 140, 360 + (i - 2) * 66, cy, 0.27 * max(sp, .05), 0.27 * max(sp, .05), (i - 2) * 1.5)
            ch = self.prop(("chips", 5, 0), lambda: G.chip_stack(5, (214, 52, 44), 0))
            ch2 = self.prop(("chips", 4, 1), lambda: G.chip_stack(4, (40, 110, 200), 0))
            place(img, ch, 80, 200, 200, cy + 55, 0.5, 0.5)
            place(img, ch2, 80, 200, 520, cy + 55, 0.5, 0.5)
        else:
            bd = self.prop(("hexboard", self.board_seed), lambda: G.hex_board(0, 62, self.board_seed))
            sc = 235.0 / bd.shape[0]
            place(img, bd, bd.shape[1] / 2, bd.shape[0] / 2, 360, cy - 4, sc, sc)
        boil = int(t * 8) % 3
        for who, at, neck, st in arms:
            self.draw_arm(img, who, at, neck, 0.5, boil, False, t, st)
        return img

    def scene_insert(self, shot, t):
        s = shot["spec"]["prop"]
        local = t - shot["t0"]
        img = self._bg()
        kind = s["kind"]
        pop = ease_out_back(local / 0.25)
        if kind == "poker" and s.get("hands"):
            board = s.get("board", [])
            for i, code in enumerate(board):
                pp = ease_out_back((local - 0.07 * i) / 0.2)
                face = self.prop(("cardface", code), lambda c=code: G.card_face(c, 0))
                place(img, face, 100, 140, 360 + (i - 2) * 118, 330, 0.52 * max(pp, .05), 0.52 * max(pp, .05), (i - 2) * 2)
            xs = (190, 530)
            win_at = s.get("win_at", 1.0)
            for hi_, hd in enumerate(s["hands"]):
                lab = T.plain(hd["label"], 46, (60, 54, 50))
                blit(img, prem(lab), int(xs[hi_] - lab.shape[1] / 2), 500)
                for j, code in enumerate(hd["cards"]):
                    pp = ease_out_back((local - 0.3 - 0.08 * j - 0.1 * hi_) / 0.25)
                    face = self.prop(("cardface", code), lambda c=code: G.card_face(c, 0))
                    place(img, face, 100, 140, xs[hi_] + (j - 0.5) * 140, 690, 0.7 * max(pp, .05), 0.7 * max(pp, .05), (j - 0.5) * 5)
            if local >= win_at:
                for hi_, hd in enumerate(s["hands"]):
                    x0, x1, y0, y1 = int(xs[hi_] - 160), int(xs[hi_] + 160), 578, 802
                    if hd.get("win"):
                        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 190, 40), 9, cv2.LINE_AA)
                    else:
                        ov = img.copy()
                        cv2.rectangle(ov, (x0, y0), (x1, y1), (236, 230, 216), -1)
                        img[:] = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        elif kind == "poker":
            board = s.get("board", [])
            slots = s.get("slots", 5)
            has_hand = bool(s.get("hand"))
            by = 370 if has_hand else 560
            sc = 0.56 if has_hand else 0.6
            back = self.prop(("slot", 0), lambda: None) if False else None
            for i in range(slots):
                x = 360 + (i - (slots - 1) / 2) * 128 * (sc / 0.56)
                if i < len(board):
                    pp = ease_out_back((local - 0.09 * i) / 0.22)
                    face = self.prop(("cardface", board[i]), lambda c=board[i]: G.card_face(c, 0))
                    place(img, face, 100, 140, x, by, sc * max(pp, .05), sc * max(pp, .05), (i - 2) * 2)
                else:
                    cv2.rectangle(img, (int(x - 100 * sc), int(by - 140 * sc)), (int(x + 100 * sc), int(by + 140 * sc)), (150, 140, 124), 3, cv2.LINE_AA)
            if has_hand:
                lab = T.plain(s.get("hand_label", "THE AI"), 52, (60, 54, 50))
                blit(img, prem(lab), int(360 - lab.shape[1] / 2), 505)
                n = len(s["hand"])
                for i, code in enumerate(s["hand"]):
                    pp = ease_out_back((local - 0.35 - 0.1 * i) / 0.25)
                    face = self.prop(("cardface", code), lambda c=code: G.card_face(c, 0))
                    place(img, face, 100, 140, 360 + (i - (n - 1) / 2) * 200, 725, 0.88 * max(pp, .05), 0.88 * max(pp, .05), (i - (n - 1) / 2) * 4)
        elif kind == "chips":
            stk = s.get("stacks") or (7, 9, 6)
            sc_ = s.get("scale", 1.3 if not s.get("stacks") else 1.0)
            for j, (n, col, dx, dy) in enumerate(((stk[0], (214, 52, 44), -150, 30), (stk[1], (40, 110, 200), 0, -10), (stk[2], (214, 52, 44), 150, 50))):
                ch = self.prop(("chips", n, col), lambda n=n, col=col: G.chip_stack(n, col, 0))
                pp = ease_out_back((local - 0.07 * j) / 0.25)
                place(img, ch, 80, ch.shape[0], 360 + dx, (800 if s.get("stacks") else 780) + dy, sc_ * max(pp, .05), sc_ * max(pp, .05))
            if s.get("label"):
                lb = T.chunky(s["label"], 150, (226, 106, 52), (20, 16, 14), 9)
                pl = ease_out_back((local - 0.2) / 0.25, 2.2)
                place(img, prem(lb), lb.shape[1] / 2, lb.shape[0] / 2, 340, s.get("label_y", 235 if s.get("stacks") else 150), pl, pl, -4)
        elif kind == "score":
            rows = s["rows"]
            vals = [r[2] for r in rows]
            to = s.get("to")
            if to and local >= s.get("flip_at", 0.5):
                pr = min(1.0, (local - s.get("flip_at", 0.5)) / 0.55)
                vals = [int(round(a + (b - a) * pr)) for a, b in zip(vals, to)]
            hi = s.get("hi") if (to and local >= s.get("flip_at", 0.5) + 0.55) else None
            key = ("score", tuple((r[0], r[1]) for r in rows), tuple(vals), hi)
            panel = self.prop(key, lambda: G.score_panel(tuple((r[0], r[1], v) for r, v in zip(rows, vals)), "SCORE", hi, 0))
            place(img, panel, panel.shape[1] / 2, panel.shape[0] / 2, 360, 560, 1.1 * max(pop, .05), 1.1 * max(pop, .05))
        elif kind == "board":
            bd = self.prop(("hexboard", self.board_seed), lambda: G.hex_board(0, 62, self.board_seed))
            sc = 600.0 / bd.shape[1]
            place(img, bd, bd.shape[1] / 2, bd.shape[0] / 2, 360, 560, sc * max(pop, .05), sc * max(pop, .05), math.sin(local * 2) * 0.6)
        return img

    def scene_private(self, shot, t):
        s = shot["spec"]
        return self.secret_panel(shot, t, s["who"], "hand", None, shot["stamp_t"][0], W, H, 1.0, label=s.get("label", "AI PRIVATE VIEW"),
                                 hand=s["hand"], hud_t=s.get("hud"))

    def scene_scream(self, shot, t):
        local = t - shot["t0"]
        who = shot["spec"].get("who", "wren")
        img = self._bg()
        P.burst(img, 360, 560, t, n=16, col=(255, 236, 170), alpha=0.35 * min(1, local / 0.1))
        shake = math.sin(local * 85) * 9 * math.exp(-local * 2.2)
        st = shot["spec"].setdefault("state", {})
        st.setdefault(who, {"expr": "shock"})
        self.draw_char(img, shot, who, t, (360 + shake, 1030 + shake * 0.5), 1.55, force_expr="shock", phase=0.1)
        return img

    def scene_endcard(self, shot, t):
        s = shot["spec"]
        ec = shot["ec"]
        local = t - shot["t0"]
        img = self._bg()

        def typed(txt, t0, t1, y, size=64):
            n = int(len(txt) * min(1.0, max(0.0, (local - t0) / (t1 - t0))) + 0.001)
            if n <= 0:
                return
            im = T.plain(txt[:n], size, (50, 44, 40), max_w=680)
            full = T.plain(txt, size, (50, 44, 40), max_w=680)
            blit(img, prem(im), int(360 - full.shape[1] / 2) + (full.shape[1] - im.shape[1]) // 2 * 0, y)
        typed(s.get("line1", "Think you can beat"), 0.05, 0.5, 250)
        typed(s.get("line2", "leading AI at"), 0.45, 0.85, 330)
        pn = ease_out_back((local - ec["name"]) / 0.3, 2.0)
        name = s["game"].upper() + "?"
        nm = T.chunky(name, 190, (226, 106, 52), (40, 30, 26), 6, shadow=False)
        if pn > 0:
            place(img, prem(nm), nm.shape[1] / 2, nm.shape[0] / 2, 360, 500, pn, pn, -2)
        # mini table scene
        pc = ease_out_back((local - ec["name"] - 0.15) / 0.3)
        if pc > 0:
            poker = s["game"] == "poker"
            tc = (112, 130, 108) if poker else (139, 108, 94)
            ov = img.copy()
            cv2.ellipse(ov, (360, 820), (255 , 92), 0, 0, 360, tc[::-1] if False else tc, -1, cv2.LINE_AA)
            cv2.ellipse(ov, (360, 820), (255, 92), 0, 0, 360, INK, 6, cv2.LINE_AA)
            img[:] = cv2.addWeighted(ov, min(1, pc), img, 1 - min(1, pc), 0)
            ai = s.get("ai", "gpt")
            shot_stub = shot
            self.draw_char(img, shot_stub, "wren", t, (225, 790), 0.33 * min(1.0, pc), speaker=None, phase=0.2, force_expr="neutral")
            self.draw_char(img, shot_stub, ai, t, (495, 790), 0.33 * min(1.0, pc), speaker=None, phase=0.6, force_expr="neutral")
            self._table_ellipse_fix = None
            if poker:
                back = self.prop(("cb", 0), lambda: P.card_back(0))
                for i in range(3):
                    place(img, back, 100, 140, 330 + i * 30, 826, 0.17 * pc, 0.17 * pc, (i - 1) * 6)
            else:
                bd = self.prop(("hexboard", self.board_seed), lambda: G.hex_board(0, 62, self.board_seed))
                sc = 105.0 / bd.shape[0]
                place(img, bd, bd.shape[1] / 2, bd.shape[0] / 2, 360, 822, sc * pc, sc * pc)
            vs = self.prop(("vs", 0), lambda: G.vs_circle())
            place(img, vs, 60, 60, 360, 690, 0.55 * pc, 0.55 * pc)
        # button + cursor
        pb = ease_out_back((local - ec["button"]) / 0.3, 2.0)
        pressed = ec["click"] <= local < ec["click"] + 0.14
        if pb > 0:
            btn = self.prop(("btn", pressed), lambda: G.cta_button("gghuman.ai", pressed, 0))
            sc = max(pb, .05) * (0.96 if pressed else 1.0)
            place(img, btn, 235, 65, 360, 1010, sc, sc)
        if local >= ec["cursor"]:
            k = smooth((local - ec["cursor"]) / max(0.2, ec["click"] - ec["cursor"]))
            cx_ = 640 + (500 - 640) * k
            cy_ = 1200 + (1030 - 1200) * k
            cu = self.prop(("cursor", 0), lambda: G.cursor())
            place(img, cu, 8, 6, cx_, cy_, 1.0, 1.0)
            if ec["click"] <= local < ec["click"] + 0.35:
                rr = int(14 + 70 * (local - ec["click"]) / 0.35)
                ov = img.copy()
                cv2.circle(ov, (int(cx_), int(cy_)), rr, (255, 255, 255), 5, cv2.LINE_AA)
                img[:] = cv2.addWeighted(ov, 0.7 * (1 - (local - ec["click"]) / 0.35), img, 1 - 0.7 * (1 - (local - ec["click"]) / 0.35), 0)
        wm = self.prop(("wordmark", 0), lambda: G.wordmark(56))
        blit(img, wm, int(360 - wm.shape[1] / 2), 1150)
        return img
