"""Shared motif module for the vertical reel: THE TOKEN MATCH-CUT.

Every beat file starts with

    from style import *
    from motif import *

and every object that crosses a cut is built ONLY by the functions in here, so
adjacent beats match by construction (see handoff_state()).

Contents (details in each docstring):
  constants ........ AC, X_C, STAGE, TTZ, KICKER_Y, HUD_NUM_TOP, HUD_LABEL_Y, STEP_TOP, ...
  math/easing ...... lerp, ramp, clip01, story_frame, shake, rgba, ease_in_cubic, ...
  text ............. kick, card, below, hud_number, hud_label, hud_mult, step_readout, step_str
  animations ....... Drive, CARD_IN, CARD_OUT, SLAM, POP, slam_scale, pop_scale
  frame-exact text . TextSlot, LiveText, TextTrack, Hud
  units/tokens ..... unit, token_glyph, unit_with_glyph, handoff_token (Token)
  camera ........... expand_cam, EXPAND / Expand, CamRig, Collapse / COLLAPSE
  effects .......... ShockRing, entry_ring, FrameFlash, add_behind
  hook grid ........ HookGrid, hook_state, b01_worker_idx, b09_worker_idx
  handoffs ......... handoff_state, enter, exit_to, BEAT_ENTRY, BEAT_EXIT

Typical beat (b03-like):

    class B03Cpu(BeatScene):
        def beat(self):
            st = enter(self)                    # E02 token + live HUD (pixel-exact frame 0)
            hud = st.hud
            world = VGroup(...)                 # build at REST positions
            exp = Expand(world, slot, 0.95, 0.70, hero=hero, token='core')   # draws e=0 now
            add_behind(self, world)             # content above dust, behind token/HUD
            entry_ring(self)                    # ring behind content, from frame 1
            self.remove(st.token)               # the hero now draws the token
            tr = TextTrack(self); self.add(tr)  # text on/off schedule (frame exact)
            tr.on(kick('SCALE 2/6 · ONE CPU'), 0.20, 3.80)
            hud.slam(2.20, '256', mult='×16')   # HUD events are scheduled, not played
            self.cue('hit'); self.play(exp)
            ...                                 # your own updaters use self.now / sframe(self)
            col = COLLAPSE(package, world, 'cpu')
            self.play(col.anim(0.41)); self.play(col.settle_anim())
            exit_to(self)                       # canonical E03, >= 3 frames before the end

Cairo gotcha (why the helpers look the way they do): manim computes the list of
"moving" mobjects (flattened families) when a play()/wait() starts. Removing or
swapping a submobject inside an updater mid-play does NOT stop the old object
from being drawn until that play ends. So: never add/remove/swap children in an
updater; mutate points/colours in place (that is what TextSlot, HookGrid,
CamRig and Collapse do) and do structural changes (scene.remove, exit_to)
between plays.

Static-wait gotcha: Scene.__init__ sets self.always_update_mobjects = False on the
instance (shadowing BeatScene's class attribute), so a wait() renders frame by frame
only because some mobject has a TIME-BASED updater (the dust has one). Every updater
in this module takes (m, dt) for that reason; write yours as lambda m, dt: ... too.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Sequence

import numpy as np

from style import *  # noqa: F401,F403
from style import (AMBER, BG, CYAN, DIM, DIMMER, FONT_BODY, FONT_DISPLAY, FRAME_H,
                   FRAME_W, INK, LIME, MAGENTA, assert_safe, ease_in_out_expo,
                   ease_out_back, ease_out_cubic, ease_out_expo, fit_width, fmt_int,
                   beat_duration, kicker, mono, token)

# ==========================================================================
# Constants
# ==========================================================================
AC = np.array([-0.35, 0.45, 0.0])          # anchor centre: every token sits here
X_C = -0.35                                 # text column centre


class Box(NamedTuple):
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, 0.0])

    def contains(self, mob, tol: float = 1e-6) -> bool:
        return (mob.get_left()[0] >= self.x0 - tol and mob.get_right()[0] <= self.x1 + tol
                and mob.get_bottom()[1] >= self.y0 - tol and mob.get_top()[1] <= self.y1 + tol)


TEXT_W = 6.9                                # max width of any text line
STAGE = Box(-3.85, 3.15, -2.40, 3.25)       # where the visuals live
TTZ_BOTTOM, TTZ_TOP = 3.35, 5.72
TTZ = Box(X_C - TEXT_W / 2, X_C + TEXT_W / 2, TTZ_BOTTOM, TTZ_TOP)   # top text zone
KICKER_Y = 6.04
HUD_NUM_TOP = -2.52
HUD_LABEL_Y = -4.00
STEP_TOP = -2.60
CARD_TOP = TTZ_TOP
CARD_GAP = 0.14

TOKEN_SIZE = 1.2
TOKEN_CORNER = 0.22                         # ratio -> radius 0.264
TOKEN_FILL = 0.18
TOKEN_STROKE = 4.0
SETTLE_FROM = 0.94                          # collapse lands at 0.94, settles to 1.0
SETTLE_TIME = 0.17
COLLAPSE_SIZE = TOKEN_SIZE * SETTLE_FROM    # 1.128

EPS = 1e-6                                  # tolerance for time comparisons

# ==========================================================================
# Math / easing helpers (all pure; safe inside updaters)
# ==========================================================================


def clip01(x: float) -> float:
    return 0.0 if x <= 0 else (1.0 if x >= 1 else float(x))


def ramp(x: float, a: float, b: float) -> float:
    """0 before a, 1 after b, linear in between (b <= a means a hard step at a)."""
    if b <= a:
        return 1.0 if x >= a - EPS else 0.0
    return clip01((x - a) / (b - a))


def ramp_arr(x, a, b):
    """Vectorised ramp (x may be an array; a, b scalars or arrays)."""
    return np.clip((np.asarray(x, float) - a) / np.maximum(np.asarray(b, float) - a, 1e-12), 0.0, 1.0)


def lerp(a, b, t):
    """a*(1-t) + b*t: exact at t=0 and t=1 (works on numpy arrays)."""
    return a * (1 - t) + b * t


def story_frame(t: float) -> int:
    """Storyboard frame index (60 fps grid) of a local time: round(t*60)."""
    return int(round(t * 60))


def sframe(scene) -> int:
    """Story frame of the frame being drawn: round(scene.now*60), except that the beat's
    final rendered frame always maps to the beat's final 60-fps frame (e.g. b09 -> f209), so a
    30-fps preview ends on the same state as the 60-fps final (and on the next beat's frame 0).
    Use it for frame-clock logic (the b01/b09 crawl)."""
    f = story_frame(scene.now)
    dur = beat_duration(type(scene).__name__)
    if dur is not None and scene.now >= dur - 1.0 / config.frame_rate - EPS:
        return int(round(dur * 60)) - 1
    return f


def ease_in_cubic(t: float) -> float:
    return t * t * t


def ease_in_quad(t: float) -> float:
    return t * t


def ease_out_quad(t: float) -> float:
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


def ease_in_sine(t: float) -> float:
    return 1 - math.cos(t * math.pi / 2)


def ease_out_sine(t: float) -> float:
    return math.sin(t * math.pi / 2)


def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2


def rgba(color, opacity: float = 1.0) -> np.ndarray:
    """(4,) float array, the format VMobject.fill_rgbas rows use."""
    return np.array([*ManimColor(color).to_rgb(), opacity], dtype=float)


def shake(t_since: float, amp: float = 0.06, dur: float = 0.20, seed: int = 0) -> np.ndarray:
    """Seeded impact-shake offset (3-vector). Amplitude decays amp->0 over dur;
    the direction is re-drawn every 60 fps story frame (so 30/60 fps agree)."""
    if t_since < -EPS or t_since >= dur - EPS:
        return np.zeros(3)
    f = story_frame(max(t_since, 0.0))
    d = np.random.default_rng((seed, f)).uniform(-1, 1, 2) * amp * (1 - max(t_since, 0) / dur)
    return np.array([d[0], d[1], 0.0])


def jitter(f: int, amp: float, seed: int = 0) -> float:
    """Seeded uniform(-amp, amp) for story frame f (e.g. headline x-jitter)."""
    return float(np.random.default_rng((seed, int(f))).uniform(-amp, amp))


def _hex(c) -> str:
    return ManimColor(c).to_hex()


# ==========================================================================
# Text
# ==========================================================================
_TCACHE: dict = {}


def _cached(key, build):
    t = _TCACHE.get(key)
    if t is None:
        t = build()
        t._glyphs = t.family_members_with_points()
        _TCACHE[key] = t
    return t


def cap_box(m: Mobject) -> tuple[float, float]:
    """(bottom, top) of the cap/digit band of a text: medians over its tall glyphs,
    so '/', '·', ',' and round-letter overshoot do not move it."""
    gl = m.family_members_with_points()
    tops = np.array([g.get_top()[1] for g in gl])
    bots = np.array([g.get_bottom()[1] for g in gl])
    hs = tops - bots
    sel = hs >= 0.75 * hs.max()
    return float(np.median(bots[sel])), float(np.median(tops[sel]))


def _check_width(m: Mobject, what: str, max_w: float = TEXT_W) -> None:
    if m.width > max_w + 1e-6:
        raise AssertionError(f"{what} is {m.width:.3f} wide (> {max_w})")


def kick(text: str, color=CYAN, tracking: int = 150, size: float = 34) -> MarkupText:
    """Kicker: style.kicker(text, 34, color, tracking) (uppercased), centred on
    x = X_C with its cap band centred on y = KICKER_Y (6.04). Asserts safe + <= 6.9 wide.
    'THE CATCH · AMDAHL'S LAW' needs tracking=80."""
    m = kicker(text, size, color, tracking)
    b, t = cap_box(m)
    m.shift([X_C - m.get_center()[0], KICKER_Y - (b + t) / 2, 0])
    _check_width(m, f"kick({text!r})")
    assert_safe(m)
    return m


# style name -> (font, size, weight, minimum effective size)
CARD_STYLES = {
    "H96": (FONT_DISPLAY, 96, HEAVY, 65),
    "H84": (FONT_DISPLAY, 84, HEAVY, 65),
    "H150": (FONT_DISPLAY, 150, HEAVY, 65),
    "S56": (FONT_DISPLAY, 56, BOLD, 39),
}


class Card(VGroup):
    """VGroup of card lines (card[i] is the i-th Text). Each line has .style,
    .eff_size and .line_scale attributes."""


def _parse_line(spec):
    if isinstance(spec, str):
        spec = (spec,)
    spec = tuple(spec) + (None,) * (4 - len(spec))
    text, style, color, t2c = spec[:4]
    return text, style or "H96", color if color is not None else INK, t2c or {}


def card(lines, top: float = CARD_TOP, gap: float = CARD_GAP, x: float = X_C,
         max_width: float = TEXT_W, check: bool = True) -> Card:
    """Headline card. lines: list of (text, style, color, t2c) (style default 'H96',
    color default INK, t2c optional {substring: color}); a bare str is allowed.
    Styles: H96/H84/H150 = Inter Display HEAVY (Black) 96/84/150, S56 = Inter Display BOLD 56.
    Lines of the same style share ONE scale s = min(1, 6.9 / widest line of that style).
    The first line's TOP edge sits at `top` (5.72), each next line's top 0.14 under the
    previous bottom, all centred on x = X_C.
    Asserts (check=True): bottom >= 3.35 when the card starts inside the TTZ,
    style.assert_safe on every line, effective size >= 65 (H) / >= 39 (S)."""
    specs = [_parse_line(s) for s in lines]
    texts = []
    for text, style, color, t2c in specs:
        font, size, weight, _ = CARD_STYLES[style]
        texts.append(Text(text, font=font, font_size=size, color=color, weight=weight, t2c=t2c))
    widest: dict[str, float] = {}
    for (_, style, _, _), t in zip(specs, texts):
        widest[style] = max(widest.get(style, 0.0), t.width)
    scales = {st: min(1.0, max_width / w) for st, w in widest.items()}
    y = top
    for (text, style, _, _), t in zip(specs, texts):
        s = scales[style]
        if s != 1.0:
            t.scale(s)
        t.style = style
        t.line_scale = s
        t.eff_size = CARD_STYLES[style][1] * s
        t.move_to([x, y - t.height / 2, 0])
        y = t.get_bottom()[1] - gap
    c = Card(*texts)
    c.scales = scales
    if check:
        for (text, style, _, _), t in zip(specs, texts):
            if t.eff_size < CARD_STYLES[style][3] - 1e-6:
                raise AssertionError(f"card line {text!r}: effective size {t.eff_size:.1f} "
                                     f"< {CARD_STYLES[style][3]} ({style})")
            assert_safe(t)
        if top >= TTZ_BOTTOM - 1e-9 and c.get_bottom()[1] < TTZ_BOTTOM - 1e-6:
            raise AssertionError(f"card {[s[0] for s in specs]} bottom {c.get_bottom()[1]:.3f} "
                                 f"< TTZ bottom {TTZ_BOTTOM}")
    return c


def below(mob: Mobject, gap: float = CARD_GAP) -> float:
    """Top y for a line/card that continues the stack under `mob`."""
    return mob.get_bottom()[1] - gap


# ---- HUD / readout templates (cached by string; public builders return copies)
def _hud_number_tmpl(s: str, color=INK) -> Text:
    def build():
        t = mono(s, 96, color, BOLD)
        fit_width(t, TEXT_W)
        t.move_to([X_C, HUD_NUM_TOP - t.height / 2, 0])
        assert_safe(t)
        b, top = cap_box(t)
        t._about = np.array([X_C, (b + top) / 2, 0.0])   # pulse/slam anchor (digit band centre)
        return t
    return _cached(("hud_number", s, _hex(color)), build)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hud_label_tmpl(s: str, color=INK, opacity: float = 0.70) -> MarkupText:
    def build():
        t = MarkupText(f'<span letter_spacing="150">{_esc(s)}</span>', font=FONT_BODY,
                       font_size=34, color=color, weight=BOLD)
        t.set_fill(color, opacity=opacity)
        b, top = cap_box(t)
        t.shift([X_C - t.get_center()[0], HUD_LABEL_Y - (b + top) / 2, 0])
        _check_width(t, f"hud_label({s!r})")
        assert_safe(t)
        return t
    return _cached(("hud_label", s, _hex(color), round(opacity, 6)), build)


def _hud_mult_tmpl(s: str, color=LIME) -> Text:
    def build():
        t = mono(s, 34, color, BOLD)
        b, top = cap_box(t)
        t.shift([X_C - t.get_center()[0], HUD_LABEL_Y - (b + top) / 2, 0])
        assert_safe(t)
        return t
    return _cached(("hud_mult", s, _hex(color)), build)


def _step_tmpl(s: str, color=AMBER) -> Text:
    def build():
        t = mono(s, 52, color, BOLD)
        fit_width(t, TEXT_W)
        _, top = cap_box(t)
        t.shift([X_C - t.get_center()[0], STEP_TOP - top, 0])
        assert_safe(t)
        b, top = cap_box(t)
        t._about = np.array([X_C, (b + top) / 2, 0.0])
        return t
    return _cached(("step", s, _hex(color)), build)


def hud_number(s: str, color=INK) -> Text:
    """JetBrains Mono BOLD 96, fit_width(6.9), TOP edge at -2.52, centred on X_C (top-aligned
    so commas never make it jump). Cached per string (returns a copy)."""
    return _hud_number_tmpl(s, color).copy()


def hud_label(s: str, color=INK, opacity: float = 0.70) -> MarkupText:
    """Inter BOLD 34, letter_spacing 150, no auto-uppercase, cap band centred on
    (X_C, -4.00), fill opacity `opacity`. Cached per string (returns a copy)."""
    return _hud_label_tmpl(s, color, opacity).copy()


def hud_mult(s: str, color=LIME) -> Text:
    """LIME JetBrains Mono BOLD 34 multiplier ('×16'), digit band centred on (X_C, -4.00)."""
    return _hud_mult_tmpl(s, color).copy()


def step_readout(s: str, color=AMBER) -> Text:
    """JetBrains Mono BOLD 52, fit_width(6.9), centred on X_C; the cap/digit band's top sits
    at STEP_TOP (-2.60) (the '/' glyph pokes 0.06 above it)."""
    return _step_tmpl(s, color).copy()


def step_str(idx: int) -> str:
    """'STEP 032 / 320'."""
    return f"STEP {int(idx):03d} / 320"


# ==========================================================================
# Generic frame-exact animation + text animation conventions
# ==========================================================================
class Drive(Animation):
    """play()-able animation that calls fn(rate_func(alpha)) every frame.
    No starting copy is made and the mobject's updaters keep running. The final
    fn(rate_func(1)) is applied by finish(), i.e. it is first SEEN on the frame after
    the play (manim draws alpha = 0, 1/n, ..., (n-1)/n)."""

    def __init__(self, mobject: Mobject, fn: Callable[[float], None], run_time: float = 1.0,
                 rate_func: Callable[[float], float] = linear, **kw):
        super().__init__(mobject, run_time=run_time, rate_func=rate_func,
                         suspend_mobject_updating=False, **kw)
        self.fn = fn

    def begin(self) -> None:
        self.interpolate(0)

    def finish(self) -> None:
        self.interpolate(1)

    def update_mobjects(self, dt: float) -> None:
        pass

    def interpolate_mobject(self, alpha: float) -> None:
        self.fn(self.rate_func(alpha))


def CARD_IN(*mobs, run_time: float = 0.25, shift=DOWN * 0.2):
    """FadeIn(shift=DOWN*0.2) over 0.25 s, ease_out_expo (one mob -> FadeIn, several -> group)."""
    anims = [FadeIn(m, shift=shift, run_time=run_time, rate_func=ease_out_expo) for m in mobs]
    return anims[0] if len(anims) == 1 else AnimationGroup(*anims)


def CARD_OUT(*mobs, run_time: float = 0.20, shift=UP * 0.15):
    """FadeOut(shift=UP*0.15) over 0.20 s (smooth). Removes the mobs at the end."""
    anims = [FadeOut(m, shift=shift, run_time=run_time, rate_func=smooth) for m in mobs]
    return anims[0] if len(anims) == 1 else AnimationGroup(*anims)


def slam_scale(t_since: float, s0: float = 1.3, dur: float = 0.20) -> float:
    """SLAM scale factor t_since seconds after the hard cut (1.3 -> 1.0, ease_out_back)."""
    if t_since < 0 or t_since >= dur:
        return 1.0
    return lerp(s0, 1.0, ease_out_back(t_since / dur))


def pop_scale(t_since: float, s0: float = 1.08, dur: float = 0.20) -> float:
    """POP scale factor (1.08 -> 1.0, ease_out_expo)."""
    if t_since < 0 or t_since >= dur:
        return 1.0
    return lerp(s0, 1.0, ease_out_expo(t_since / dur))


def _scale_in(mob, s0, ease, run_time, about):
    mems = mob.family_members_with_points()
    rest = [m.points.copy() for m in mems]
    c = np.asarray(about, float) if about is not None else mob.get_center()

    def fn(a):
        s = lerp(s0, 1.0, ease(a))
        for m, p in zip(mems, rest):
            m.points = p.copy() if s == 1.0 else (p - c) * s + c
    return Drive(mob, fn, run_time=run_time, rate_func=linear)


def SLAM(mob: Mobject, run_time: float = 0.20, scale: float = 1.3, about=None) -> Drive:
    """Hard cut on (visible at 1.3x on the first frame), then 1.3 -> 1.0 in 0.20 s,
    ease_out_back, about the mob's centre (or `about`). Build it right when you play it."""
    return _scale_in(mob, scale, ease_out_back, run_time, about)


def POP(mob: Mobject, run_time: float = 0.20, scale: float = 1.08, about=None) -> Drive:
    """Hard cut on, then 1.08 -> 1.0 in 0.20 s, ease_out_expo, no opacity fade."""
    return _scale_in(mob, scale, ease_out_expo, run_time, about)


# ==========================================================================
# Frame-exact text: glyph pools, live text, schedules, HUD
# ==========================================================================
_STYLE_ATTRS = ("stroke_width", "background_stroke_width", "sheen_factor", "joint_type", "cap_style")


def _copy_style(dst: VMobject, src: VMobject, alpha: float = 1.0, tint=None) -> None:
    dst.fill_rgbas = src.fill_rgbas.copy()
    dst.stroke_rgbas = src.stroke_rgbas.copy()
    dst.background_stroke_rgbas = src.background_stroke_rgbas.copy()
    for a in _STYLE_ATTRS:
        if hasattr(src, a):
            setattr(dst, a, getattr(src, a))
    dst.sheen_direction = np.array(src.sheen_direction)
    if alpha != 1.0:
        dst.fill_rgbas[:, 3] *= alpha
        dst.stroke_rgbas[:, 3] *= alpha
        dst.background_stroke_rgbas[:, 3] *= alpha
    if tint is not None:
        rgb = np.array(ManimColor(tint).to_rgb())
        dst.fill_rgbas[:, :3] = rgb
        dst.stroke_rgbas[:, :3] = rgb


_EMPTY = np.zeros((0, 3))


class TextSlot(VGroup):
    """A pool of VMobjects that displays any (cached, positioned) Text in place:
    display(tmpl, alpha, scale, about, shift, tint) copies glyph points + style, unused pool members get no points (not drawn).
    Renders pixel-identically to the template itself; safe to change every frame
    mid-play (nothing is added/removed from the scene)."""

    def __init__(self):
        super().__init__()
        self._key = None

    def clear_slot(self) -> None:
        if self._key == "empty":
            return
        for m in self.submobjects:
            m.points = _EMPTY.copy()
        self._key = "empty"

    def display(self, tmpl: Mobject, alpha: float = 1.0, scale: float = 1.0, about=None,
             shift=None, tint=None) -> None:
        if alpha <= 0:
            self.clear_slot()
            return
        sh = None if shift is None or not np.any(shift) else np.asarray(shift, float)
        ab = None if scale == 1.0 else (np.asarray(about, float) if about is not None
                                          else getattr(tmpl, "_about", tmpl.get_center()))
        key = (id(tmpl), round(alpha, 9), round(scale, 9),
               None if ab is None else tuple(np.round(ab, 9)),
               None if sh is None else tuple(np.round(sh, 9)), None if tint is None else _hex(tint))
        if key == self._key:
            return
        self._key = key
        glyphs = getattr(tmpl, "_glyphs", None) or tmpl.family_members_with_points()
        while len(self.submobjects) < len(glyphs):
            self.add(VMobject())
        for m, g in zip(self.submobjects, glyphs):
            p = g.points
            if ab is not None:
                p = (p - ab) * scale + ab
            if sh is not None:
                p = p + sh
            m.points = p.copy() if p is g.points else p
            _copy_style(m, g, alpha, tint)
        for m in self.submobjects[len(glyphs):]:
            m.points = _EMPTY.copy()


class LiveText(VGroup):
    """Text whose string is a function of time, e.g. the b01/b09 step readout:

        ro = LiveText(lambda now: step_str(b01_worker_idx(story_frame(now))),
                      lambda s: step_readout_tmpl(s, AMBER), scene=self)

    fn(now) -> str or None (hidden). build(s) -> positioned Mobject; built once per string.
    Optional callables of `now`: alpha(now), scale(now) (about the template's centre
    or `about`), shift(now) -> 3-vector. Renders immediately and every frame once bound."""

    def __init__(self, fn, build, scene=None, alpha=None, scale=None, shift=None, about=None):
        super().__init__()
        self.fn, self.build = fn, build
        self.alpha_fn, self.scale_fn, self.shift_fn, self.about = alpha, scale, shift, about
        self.slot = TextSlot()
        self.add(self.slot)
        self._cache: dict = {}
        self.render_at(0.0)
        if scene is not None:
            self.bind(scene)

    def bind(self, scene) -> "LiveText":
        self.add_updater(lambda m, dt: m.render_at(scene.now))
        return self

    def template(self, s: str) -> Mobject:
        t = self._cache.get(s)
        if t is None:
            t = self.build(s)
            t._glyphs = t.family_members_with_points()
            self._cache[s] = t
        return t

    def render_at(self, now: float) -> None:
        s = self.fn(now)
        if s is None:
            self.slot.clear_slot()
            return
        a = self.alpha_fn(now) if self.alpha_fn else 1.0
        sc = self.scale_fn(now) if self.scale_fn else 1.0
        sh = self.shift_fn(now) if self.shift_fn else None
        self.slot.display(self.template(s), a, sc, self.about, sh)


def step_readout_tmpl(s: str, color=AMBER) -> Text:
    """The cached (shared, do not mutate) template behind step_readout(); for LiveText."""
    return _step_tmpl(s, color)


class _TrackItem:
    __slots__ = ("mob", "mems", "rest", "t_in", "t_out", "enter", "exit", "enter_dur",
                 "exit_dur", "alpha_fn", "shift_fn", "about", "key")


class TextTrack(VGroup):
    """Frame-exact on/off schedule for static mobjects (cards, kickers, labels), driven by
    scene.now, so you can run one long play/wait and still land every t_in/t_out exactly.

        tr = TextTrack(self)                       # add it once: self.add(tr)
        tr.on(kick('SCALE 1/6 · ONE CORE'), 0.10, 4.10)              # CARD_IN / CARD_OUT
        tr.on(card([...])[0], 1.50, 2.60, enter='slam')              # SLAM in, CARD_OUT
        tr.on(head, 0.0, 2.35, exit='cut', alpha=lambda now: 1 if now < 1 else 0.4)

    enter: 'in' (CARD_IN 0.25 s), 'slam' (1.3->1 0.20 s), 'pop' (1.08->1 0.20 s), 'cut'.
    exit : 'out' (CARD_OUT 0.20 s, starting at t_out), 'cut' (gone at t_out), None = stays.
    alpha(now) multiplies opacity, shift(now) adds an offset (e.g. jitter). Items are
    invisible (opacity 0) outside their window. Register mobs at their rest position."""

    ENTER_DUR = {"in": 0.25, "slam": 0.20, "pop": 0.20, "cut": 0.0}
    EXIT_DUR = {"out": 0.20, "cut": 0.0, None: 0.0}

    def __init__(self, scene=None):
        super().__init__()
        self.items: list[_TrackItem] = []
        self._scene = scene
        if scene is not None:
            self.add_updater(lambda m, dt: m.render_at(scene.now))

    def on(self, mob: Mobject, t_in: float, t_out: float | None = None, enter: str = "in",
             exit: str | None = "out", alpha=None, shift=None, about=None,
             enter_dur: float | None = None, exit_dur: float | None = None) -> Mobject:
        it = _TrackItem()
        it.mob, it.t_in, it.t_out, it.enter, it.exit = mob, t_in, t_out, enter, exit
        it.enter_dur = self.ENTER_DUR[enter] if enter_dur is None else enter_dur
        it.exit_dur = self.EXIT_DUR[exit] if exit_dur is None else exit_dur
        it.alpha_fn, it.shift_fn = alpha, shift
        it.mems = mob.family_members_with_points()
        it.rest = [(m.points.copy(), m.fill_rgbas.copy(), m.stroke_rgbas.copy(),
                    m.background_stroke_rgbas.copy()) for m in it.mems]
        it.about = np.asarray(about, float) if about is not None else mob.get_center()
        it.key = None
        self.items.append(it)
        self.add(mob)
        if self._scene is not None:
            self._render_item(it, self._scene.now)
        else:
            self._render_item(it, 0.0)
        return mob

    def render_at(self, now: float) -> None:
        for it in self.items:
            self._render_item(it, now)

    @staticmethod
    def _state(it: _TrackItem, now: float):
        if now < it.t_in - EPS:
            return 0.0, 1.0, None
        if it.t_out is not None and it.exit is not None and now >= it.t_out + it.exit_dur - EPS:
            return 0.0, 1.0, None
        a, s, sh = 1.0, 1.0, np.zeros(3)
        d = now - it.t_in
        if it.enter_dur > 0 and d < it.enter_dur - EPS:
            p = max(d, 0.0) / it.enter_dur
            if it.enter == "in":
                e = ease_out_expo(p)
                a, sh = e, UP * 0.2 * (1 - e)
            elif it.enter == "slam":
                s = lerp(1.3, 1.0, ease_out_back(p))
            elif it.enter == "pop":
                s = lerp(1.08, 1.0, ease_out_expo(p))
        if it.t_out is not None and it.exit == "out" and now >= it.t_out - EPS:
            e = smooth(clip01((now - it.t_out) / it.exit_dur))
            a *= 1 - e
            sh = sh + UP * 0.15 * e
        if it.alpha_fn is not None:
            a *= it.alpha_fn(now)
        if it.shift_fn is not None:
            sh = sh + np.asarray(it.shift_fn(now), float)
        return a, s, (sh if np.any(sh) else None)

    def _render_item(self, it: _TrackItem, now: float) -> None:
        a, s, sh = self._state(it, now)
        key = (round(a, 9), round(s, 9), None if sh is None else tuple(np.round(sh, 9)))
        if key == it.key:
            return
        it.key = key
        for m, (p, fr, sr, br) in zip(it.mems, it.rest):
            q = p
            if s != 1.0:
                q = (q - it.about) * s + it.about
            if sh is not None:
                q = q + sh
            m.points = q.copy() if q is p else q
            if a == 1.0:
                m.fill_rgbas, m.stroke_rgbas, m.background_stroke_rgbas = fr.copy(), sr.copy(), br.copy()
            else:
                for attr, r in (("fill_rgbas", fr), ("stroke_rgbas", sr), ("background_stroke_rgbas", br)):
                    v = r.copy()
                    v[:, 3] *= a
                    setattr(m, attr, v)


class Hud(VGroup):
    """The bottom counter: number (hud_number, top edge -2.52) + label (hud_label) and the
    LIME multiplier (hud_mult) at y -4.00. A pure function of the beat's self.now:
    you schedule events (local beat seconds) once, it renders the right frame every frame.

        hud = Hud('256', 'CALCULATIONS AT ONCE', scene=self)   # or enter(self).hud
        hud.slam(2.20, '4,096', mult='×16')                    # hard cut + 1.3->1 + ×16 swap
        hud.roll(0.40, 1.30, 256, 16896)                       # log10-space comma roll
        hud.slam(1.30, '16,896', mult='×66')

    Number events (the latest one with t <= now wins; before the first: the initial number):
      cut(t, s, color)                    hard cut
      slam(t, s, color, mult=None)        hard cut + SLAM scale; mult -> mult(t, mult)
      pop(t, s=None, color)               (optional hard cut) + POP scale
      roll(t0, t1, v0, v1, ease, log=True, fmt=fmt_int, suffix='', color)  holds fmt(v1) after t1
      drive(t, fn, color)                 fn(now) -> str or (str, color), from t on
      swap(t, s, dur=0.25, dist=0.35)     slot machine: old slides down+fades, new drops in
    Other channels:
      pulse(t, amp=1.1, dur=0.2)          number scale amp->1 (ease_out_cubic), no change
      mult(t, s, hold=1.0, fade=0.12)     label->multiplier swap 0.12 s (label out, then mult in), hold, back
      label(t, s, color=INK, opacity=0.70, fade=0.0)   new label (crossfade if fade > 0)
      fade_to(t0, t1, o0, o1, shift0, shift1, ease)    whole-HUD opacity/offset envelope
      hide_at(t) / show_at(t)             instant envelope 0 / 1
      glitch(t0, t1, amp=0.06, tint=AMBER, split_frames=(), seed=8)  per-frame x-jitter,
                                          tint on odd story frames, RGB-split copies on the
                                          listed story-frame offsets; nothing on frame t0 itself
    hud.post = fn(hud, now) runs after each render for anything bespoke.
    With no events (or before them) it renders exactly hud_number(number) + hud_label(label),
    so Hud(...) at rest == the handoff_state HUD pixel for pixel."""

    def __init__(self, number: str | None = None, label: str | None = None, color=INK,
                 label_color=INK, label_opacity: float = 0.70, scene=None):
        super().__init__()
        self.s_split = VGroup(*[TextSlot() for _ in range(4)])
        self.s_num_old, self.s_num = TextSlot(), TextSlot()
        self.s_lab_old, self.s_lab, self.s_mult = TextSlot(), TextSlot(), TextSlot()
        self.add(self.s_split, self.s_num_old, self.s_num, self.s_lab_old, self.s_lab, self.s_mult)
        self._init_num = (number, color)
        self._init_lab = (label, label_color, label_opacity)
        self._num: list = []
        self._scl: list = []
        self._mult: list = []
        self._lab: list = []
        self._env: list = []
        self._gl: list = []
        self.post = None
        self.scene = None
        self.render_at(0.0)
        if scene is not None:
            self.bind(scene)

    # ---- binding
    def bind(self, scene) -> "Hud":
        if self.scene is None:
            self.scene = scene
            self.add_updater(lambda m, dt: m.render_at(scene.now))
        return self

    @staticmethod
    def _ins(lst, ev):
        lst.append(ev)
        lst.sort(key=lambda e: e[0])

    # ---- number events
    def cut(self, t, s, color=None):
        self._ins(self._num, (t, "set", (s, color)))
        return self

    def slam(self, t, s, color=None, mult=None, scale=1.3, dur=0.20):
        self.cut(t, s, color)
        self._ins(self._scl, (t, scale, dur, ease_out_back))
        if mult:
            self.mult(t, mult)
        return self

    def pop(self, t, s=None, color=None, scale=1.08, dur=0.20):
        if s is not None:
            self.cut(t, s, color)
        self._ins(self._scl, (t, scale, dur, ease_out_expo))
        return self

    def pulse(self, t, amp=1.1, dur=0.20, ease=ease_out_cubic):
        self._ins(self._scl, (t, amp, dur, ease))
        return self

    def roll(self, t0, t1, v0, v1, ease=None, log=True, fmt=fmt_int, suffix="", color=None):
        ease = ease or ease_in_out_cubic
        self._ins(self._num, (t0, "roll", (t1, float(v0), float(v1), ease, log, fmt, suffix, color)))
        return self

    def drive(self, t, fn, color=None):
        self._ins(self._num, (t, "drive", (fn, color)))
        return self

    def swap(self, t, s, dur=0.25, dist=0.35, color=None):
        self._ins(self._num, (t, "swap", (s, color, dur, dist)))
        return self

    # ---- other channels
    def mult(self, t, s, hold=1.0, fade=0.12, color=LIME):
        self._ins(self._mult, (t, s, hold, fade, color))
        return self

    def label(self, t, s, color=INK, opacity=0.70, fade=0.0):
        self._ins(self._lab, (t, s, color, opacity, fade))
        return self

    def fade_to(self, t0, t1, o0, o1, shift0=ORIGIN, shift1=ORIGIN, ease=None):
        self._ins(self._env, (t0, t1, o0, o1, np.asarray(shift0, float), np.asarray(shift1, float),
                              ease or ease_out_cubic))
        return self

    def hide_at(self, t):
        return self.fade_to(t, t, 0.0, 0.0)

    def show_at(self, t):
        return self.fade_to(t, t, 1.0, 1.0)

    def glitch(self, t0, t1, amp=0.06, tint=AMBER, split_frames=(), seed=8):
        self._ins(self._gl, (t0, t1, amp, tint, tuple(split_frames), seed))
        return self

    # ---- state evaluation
    def _num_eval(self, i: int, now: float):
        if i < 0:
            return self._init_num
        t, kind, d = self._num[i]
        if kind == "set":
            s, c = d
            return s, (c if c is not None else self._num_eval(i - 1, t)[1] or INK)
        if kind == "swap":
            return d[0], (d[1] if d[1] is not None else self._num_eval(i - 1, t)[1] or INK)
        if kind == "roll":
            t1, v0, v1, ease, log, fmt, suffix, c = d
            p = 1.0 if t1 <= t else clip01((now - t) / (t1 - t))
            e = ease(p)
            if e >= 1.0:
                v = v1
            elif log:
                v = 10 ** lerp(math.log10(v0), math.log10(v1), e)
            else:
                v = lerp(v0, v1, e)
            return fmt(v) + suffix, (c if c is not None else self._num_eval(i - 1, t)[1] or INK)
        if kind == "drive":
            fn, c = d
            r = fn(now)
            if isinstance(r, tuple):
                return r
            return r, (c if c is not None else self._num_eval(i - 1, t)[1] or INK)
        raise ValueError(kind)

    def _num_index(self, now: float) -> int:
        i = -1
        for j, ev in enumerate(self._num):
            if ev[0] <= now + EPS:
                i = j
        return i

    def number_at(self, now: float):
        """(string, color) the HUD number shows at local time `now`."""
        return self._num_eval(self._num_index(now), now)

    def _scale_at(self, now: float) -> float:
        s = 1.0
        for t, s0, dur, ease in self._scl:
            d = now - t
            if -EPS <= d < dur - EPS:
                s *= lerp(s0, 1.0, ease(clip01(d / dur)))
        return s

    def _env_at(self, now: float):
        if not self._env:
            return 1.0, None
        last = None
        for ev in self._env:
            if ev[0] <= now + EPS:
                last = ev
        if last is None:
            ev = self._env[0]
            return ev[2], ev[4]
        t0, t1, o0, o1, sh0, sh1, ease = last
        if t1 <= t0 or now >= t1 - EPS:
            return o1, sh1
        e = ease(clip01((now - t0) / (t1 - t0)))
        return lerp(o0, o1, e), lerp(sh0, sh1, e)

    def _mult_at(self, now: float):
        best = (0.0, None, None)
        for t, s, hold, fade, color in self._mult:
            d = now - t
            if d < -EPS:
                continue
            if d < fade:
                m = ease_in_out_cubic(clip01(d / fade))
            elif d < fade + hold:
                m = 1.0
            elif d < 2 * fade + hold:
                m = 1.0 - ease_in_out_cubic(clip01((d - fade - hold) / fade))
            else:
                m = 0.0
            if m > 0:
                best = (m, s, color)
        return best

    def _lab_at(self, now: float):
        cur, prev, a = self._init_lab, None, 1.0
        for t, s, color, op, fade in self._lab:
            if t <= now + EPS:
                prev, cur = cur, (s, color, op)
                a = 1.0 if fade <= 0 else clip01((now - t) / fade)
                a = ease_in_out_cubic(a) if a < 1.0 else 1.0
        return cur, (prev if a < 1.0 else None), a

    def _glitch_at(self, now: float):
        for t0, t1, amp, tint, split, seed in self._gl:
            if t0 + EPS < now < t1 - EPS:
                f = story_frame(now - t0)
                if f <= 0:
                    return 0.0, None, False
                return jitter(f, amp, seed), (tint if (tint is not None and f % 2 == 1) else None), f in split
        return 0.0, None, False

    # ---- render
    def render_at(self, now: float) -> None:
        env, env_sh = self._env_at(now)
        gx, tint, split = self._glitch_at(now)
        shift = np.zeros(3) if env_sh is None else np.array(env_sh, float)
        shift[0] += gx
        if env <= 0:
            for sl in (self.s_num_old, self.s_num, self.s_lab_old, self.s_lab, self.s_mult, *self.s_split):
                sl.clear_slot()
            return
        # number
        i = self._num_index(now)
        s, col = self._num_eval(i, now)
        scale = self._scale_at(now)
        num_tmpl = None
        swapping = i >= 0 and self._num[i][1] == "swap" and now - self._num[i][0] < self._num[i][2][2] - EPS
        if swapping:
            t, _, (s_new, _, dur, dist) = self._num[i]
            a = clip01((now - t) / dur)
            old_s, old_c = self._num_eval(i - 1, t)
            e1, e2 = ease_in_out_cubic(a), ease_out_cubic(a)
            if old_s:
                self.s_num_old.display(_hud_number_tmpl(old_s, old_c), env * (1 - e1), 1.0, None,
                                    shift + DOWN * dist * e1, tint)
            else:
                self.s_num_old.clear_slot()
            num_tmpl = _hud_number_tmpl(s, col)
            self.s_num.display(num_tmpl, env * e2, scale, None, shift + UP * dist * (1 - e2), tint)
        else:
            self.s_num_old.clear_slot()
            if s:
                num_tmpl = _hud_number_tmpl(s, col)
                self.s_num.display(num_tmpl, env, scale, None, shift, tint)
            else:
                self.s_num.clear_slot()
        # label + multiplier
        m, ms, mcol = self._mult_at(now)
        cur, prev, a = self._lab_at(now)
        # Sequential swap (integration fix): the label is gone by the half-way point of the
        # 0.12 s fade and the multiplier only appears after it, so the two strings never
        # double-expose on the same line (they did at every ×N jump with a plain crossfade).
        la = env * (1 - ramp(m, 0.0, 0.5))
        ma = ramp(m, 0.5, 1.0)
        lab_tmpl = _hud_label_tmpl(*cur) if cur[0] else None
        if prev is not None and prev[0]:
            self.s_lab_old.display(_hud_label_tmpl(*prev), la * (1 - a), 1.0, None, shift, tint)
        else:
            self.s_lab_old.clear_slot()
        if lab_tmpl is not None:
            self.s_lab.display(lab_tmpl, la * (a if prev is not None else 1.0), 1.0, None, shift, tint)
        else:
            self.s_lab.clear_slot()
        if m > 0 and ms:
            self.s_mult.display(_hud_mult_tmpl(ms, mcol), env * ma, 1.0, None, shift, tint)
        else:
            self.s_mult.clear_slot()
        # RGB split copies (behind)
        if split:
            for k, (tm, dx, c) in enumerate(((num_tmpl, -0.05, CYAN), (num_tmpl, 0.05, MAGENTA),
                                              (lab_tmpl, -0.05, CYAN), (lab_tmpl, 0.05, MAGENTA))):
                if tm is None:
                    self.s_split[k].clear_slot()
                else:
                    self.s_split[k].display(tm, 0.5 * env, scale if k < 2 else 1.0, None,
                                         shift + RIGHT * dx, c)
        else:
            for sl in self.s_split:
                sl.clear_slot()
        if self.post is not None:
            self.post(self, now)


def amdahl(n: float, serial: float = 0.1) -> float:
    """Amdahl speedup S(N) = 1 / (s + (1-s)/N)."""
    return 1.0 / (serial + (1.0 - serial) / n)


def speedup_str(s: float) -> str:
    """Floor-truncated '{:.1f}×' (never shows 10.0 for finite N): 1.8×, 3.0×, 9.9×."""
    return f"{math.floor(10 * s + 1e-9) / 10:.1f}×"


# ==========================================================================
# Units and handoff tokens
# ==========================================================================
def unit(size: float, color=CYAN, fill: float = 0.15, stroke: float = 3) -> RoundedRectangle:
    """Every core/GPU/server: style.token(size, color, fill, stroke, corner=0.22),
    centred at the origin. A unit scaled to 1.2 is exactly the token outline."""
    return token(size, color, fill, stroke, corner=TOKEN_CORNER)


_GPU_R = np.random.default_rng(5).random((8, 8))


def token_glyph(kind: str, color=CYAN, center=AC, scale: float = 1.0) -> VGroup:
    """The glyph of handoff_token(kind), scaled `scale` about its token centre and moved so
    that centre is `center` (stroke widths are NOT scaled). E.g. b03's cores carry
    token_glyph('core', center=core_center, scale=0.95/1.2)."""
    g = VGroup()
    ax, ay = AC[0], AC[1]
    if kind == "plain":
        pass
    elif kind == "core":
        for k in range(16):
            r = Rectangle(width=0.036, height=0.36, stroke_width=0, fill_color=color, fill_opacity=0.65)
            r.move_to([ax + (k - 7.5) * 0.056, ay - 0.19, 0])
            g.add(r)
    elif kind == "cpu":
        offs = (-0.375, -0.125, 0.125, 0.375)
        for row in range(4):
            for col in range(4):
                r = RoundedRectangle(width=0.20, height=0.20, corner_radius=0.04,
                                     stroke_color=color, stroke_width=1.5, stroke_opacity=0.9,
                                     fill_color=color, fill_opacity=0.15)
                r.move_to([ax + offs[col], ay + offs[3 - row], 0])
                g.add(r)
    elif kind == "gpu":
        for row in range(8):
            for col in range(8):
                sq = Square(side_length=0.085, stroke_width=0, fill_color=color,
                            fill_opacity=0.25 + 0.65 * _GPU_R[row][col])
                sq.move_to([ax + (col - 3.5) * 0.115, ay + (3.5 - row) * 0.115, 0])
                g.add(sq)
    elif kind == "node":
        for yo in (0.345, 0.115, -0.115, -0.345):
            for xo in (-0.24, 0.24):
                r = RoundedRectangle(width=0.30, height=0.18, corner_radius=0.04,
                                     stroke_color=color, stroke_width=1.5,
                                     fill_color=color, fill_opacity=0.2)
                r.move_to([ax + xo, ay + yo, 0])
                g.add(r)
        g.add(Line([ax, ay - 0.45, 0], [ax, ay + 0.45, 0], stroke_color=MAGENTA,
                   stroke_width=2.5, stroke_opacity=0.9))
    elif kind == "cluster":
        for row in range(7):
            for col in range(7):
                sq = Square(side_length=0.06, stroke_width=0, fill_color=color, fill_opacity=0.65)
                sq.move_to([ax + (col - 3) * 0.13, ay + (3 - row) * 0.13, 0])
                g.add(sq)
    else:
        raise ValueError(f"unknown token kind {kind!r}")
    if scale != 1.0:
        g.scale(scale, about_point=AC)
    c = np.asarray(center, float)
    if np.any(c != AC):
        g.shift(c - AC)
    return g


class Token(VGroup):
    """handoff_token(): VGroup(base, glyph) with .base, .glyph, .kind, .token_color
    (never assign .color on a mobject: it is a property that recolours the family)."""


def handoff_token(kind: str = "plain", color=CYAN) -> Token:
    """The handoff unit at AC: base = style.token(1.2, color, fill 0.18, stroke 4,
    corner 0.22 -> radius 0.264) with token_glyph(kind) drawn above it.
    kinds: plain, core, cpu, gpu, node, cluster (glyph colour = `color`, node spine MAGENTA)."""
    base = token(TOKEN_SIZE, color, TOKEN_FILL, TOKEN_STROKE, corner=TOKEN_CORNER).move_to(AC)
    glyph = token_glyph(kind, color)
    t = Token(base, glyph)
    t.base, t.glyph, t.kind, t.token_color = base, glyph, kind, color
    return t


def unit_with_glyph(kind: str, size: float, center, color=CYAN, fill: float = 0.15,
                    stroke: float = 3, glyph_fill: float | None = None) -> VGroup:
    """VGroup(unit, glyph) at `center`: a core/GPU/server that carries the token glyph scaled
    size/1.2 (so EXPAND's hero at e=0 is exactly handoff_token(kind)). glyph_fill overrides the
    glyph members' fill opacity (e.g. 0.35 for b03's idle cores). Attributes .unit, .glyph."""
    u = unit(size, color, fill, stroke).move_to(center)
    g = token_glyph(kind, color, center=center, scale=size / TOKEN_SIZE)
    if glyph_fill is not None:
        for m in g.family_members_with_points():
            if kind == "node" and isinstance(m, Line):
                continue
            m.set_fill(opacity=glyph_fill)
    grp = VGroup(u, g)
    grp.unit, grp.glyph = u, g
    return grp


# ==========================================================================
# Camera rules: EXPAND (pull-back from a token) and COLLAPSE (into a token)
# ==========================================================================
def expand_cam(slot, u: float, e: float) -> tuple[float, np.ndarray]:
    """EXPAND camera as p -> k*p + b. k0 = 1.2/u; the at-rest layout is scaled by
    k0^(1-e) about `slot`, then shifted by (AC - slot)*(1-e). e=0: the unit of size u at
    `slot` is exactly the token at AC; e=1: identity (k = 1, b = 0 exactly)."""
    slot = np.asarray(slot, float)
    if slot.shape[0] == 2:
        slot = np.array([slot[0], slot[1], 0.0])
    k = (TOKEN_SIZE / u) ** (1 - e)
    b = slot * (1 - k) + (AC - slot) * (1 - e)
    return k, b


def cam_apply(p, k: float, b) -> np.ndarray:
    """Map world point(s) p (shape (3,) or (n,3)) through the camera."""
    return np.asarray(p, float) * k + b


class _Rest:
    __slots__ = ("m", "p", "fr", "sr", "br", "sw", "bsw")

    def __init__(self, m):
        self.m = m
        self.p = m.points.copy()
        self.fr = m.fill_rgbas.copy()
        self.sr = m.stroke_rgbas.copy()
        self.br = m.background_stroke_rgbas.copy()
        self.sw = float(m.stroke_width)
        self.bsw = float(m.background_stroke_width)


class CamRig:
    """Stores a group's at-rest geometry + style ONCE and redraws it through a 2D camera
    p -> k*p + b every frame (pure; no drift). Use for any camera move:

        rig = CamRig(world)                 # world at rest (final layout)
        rig.apply(k, b, alpha=..., scale_strokes=True)

    alpha: None, a scalar, or a per-member array (len(rig.members)) multiplying opacity.
    k == 1 and b == 0 reproduce the rest state exactly. The group's structure must not
    change after the rig is built (rebuild the rig if it does)."""

    def __init__(self, world: Mobject):
        self.world = world
        self.members = world.family_members_with_points()
        self.rest = [_Rest(m) for m in self.members]
        self.index = {id(m): i for i, m in enumerate(self.members)}

    def indices_of(self, mob: Mobject) -> list[int]:
        """Member indices belonging to `mob` (any submobject of the world)."""
        return [self.index[id(m)] for m in mob.family_members_with_points() if id(m) in self.index]

    def apply(self, k: float, b, alpha=None, scale_strokes: bool = True) -> None:
        b = np.asarray(b, float)
        n = len(self.rest)
        al = np.ones(n) if alpha is None else (np.full(n, float(alpha)) if np.isscalar(alpha) else alpha)
        for r, a in zip(self.rest, al):
            m = r.m
            m.points = r.p * k + b
            _set_alpha(m, r, a)
            m.stroke_width = r.sw * k if scale_strokes else r.sw
            m.background_stroke_width = r.bsw * k if scale_strokes else r.bsw

    def restore(self) -> None:
        self.apply(1.0, np.zeros(3), None, False)


def _set_alpha(m, r: _Rest, a: float) -> None:
    if a == 1.0:
        m.fill_rgbas, m.stroke_rgbas, m.background_stroke_rgbas = r.fr.copy(), r.sr.copy(), r.br.copy()
    else:
        for attr, v0 in (("fill_rgbas", r.fr), ("stroke_rgbas", r.sr), ("background_stroke_rgbas", r.br)):
            v = v0.copy()
            v[:, 3] *= a
            setattr(m, attr, v)


class _ExpandRig(CamRig):
    def __init__(self, world, hero, token_ref, fades, fade):
        super().__init__(world)
        n = len(self.members)
        self.win = np.tile(np.array([fade[0], fade[1], 0.0, 1.0]), (n, 1))
        self.pair: dict[int, _Rest] = {}
        if hero is not None:
            hidx = self.indices_of(hero)
            for i in hidx:
                self.win[i] = (0.0, 1.0, 1.0, 1.0)
            if token_ref is not None:
                tok = handoff_token(token_ref) if isinstance(token_ref, str) else token_ref
                tm = tok.family_members_with_points()
                for i, t in zip(hidx, tm):
                    self.pair[i] = _Rest(t)
        for mob, w in (fades or {}).items():
            w = tuple(w) + ((0.0, 1.0) if len(w) == 2 else ())
            for i in self.indices_of(mob):
                self.win[i] = w

    def at(self, slot, u, e, scale_strokes=True):
        k, b = expand_cam(slot, u, e)
        for i, r in enumerate(self.rest):
            m = r.m
            m.points = r.p * k + b
            e0, e1, a0, a1 = self.win[i]
            a = lerp(a0, a1, ramp(e, e0, e1))
            tp = self.pair.get(i)
            if tp is not None:      # hero: style tweens token -> own
                m.fill_rgbas = lerp(tp.fr, r.fr, e) if tp.fr.shape == r.fr.shape else r.fr.copy()
                m.stroke_rgbas = lerp(tp.sr, r.sr, e) if tp.sr.shape == r.sr.shape else r.sr.copy()
                m.background_stroke_rgbas = r.br.copy()
                m.stroke_width = lerp(tp.sw, r.sw, e)
                if a != 1.0:
                    m.fill_rgbas[:, 3] *= a
                    m.stroke_rgbas[:, 3] *= a
            else:
                _set_alpha(m, r, a)
                m.stroke_width = r.sw * k if scale_strokes else r.sw
            m.background_stroke_width = r.bsw


def EXPAND(world: Mobject, slot, u: float, e: float, hero: Mobject | None = None,
           token: str | Mobject | None = None, fades: dict | None = None,
           fade: tuple = (0.0, 0.6), scale_strokes: bool = True) -> None:
    """Camera pull-back, as a pure function of e (0 -> 1; pass the EASED value).
    world: the at-rest layout (built at its final positions; captured on the first call).
    slot/u: centre and size of the hero unit. At e=0 the hero is exactly the incoming
    token at AC; its family members' style tweens from `token` (a kind string or a Token,
    paired member-by-member, e.g. unit_with_glyph(...) <-> handoff_token(kind)) to its own.
    Every other member's opacity goes 0 -> 1 over e in `fade` (= [0, 0.6]).
    fades = {submobject: (e0, e1) | (e0, e1, a0, a1)} overrides opacity windows
    (e.g. b03 labels (0.3, 0.8); b06 hero glyph (0, 0.35, 1, 0)).
    Strokes of non-hero members scale with the camera (scale_strokes).
    Usable in updaters: world.add_updater(lambda m, dt: EXPAND(m, slot, u, ease_out_expo(...)))."""
    rig = getattr(world, "_motif_expand", None)
    if rig is None:
        rig = _ExpandRig(world, hero, token, fades, fade)
        world._motif_expand = rig
    rig.at(slot, u, e, scale_strokes)


def Expand(world: Mobject, slot, u: float, run_time: float = 0.70, rate_func=ease_out_expo, **kw) -> Drive:
    """play()-able EXPAND (default 0.70 s ease_out_expo). kw -> EXPAND (hero, token, fades...)."""
    EXPAND(world, slot, u, 0.0, **kw)       # capture rest + draw e=0 now
    return Drive(world, lambda e: EXPAND(world, slot, u, e, **kw), run_time=run_time, rate_func=rate_func)


class Collapse:
    """COLLAPSE(outline, contents, kind): the outline morphs (points + style) into
    handoff_token(kind).base at scale 0.94; contents follow
        p -> lerp(C_out, AC, e) + s*(p - C_out),  s = (1.128 / max(w, h))^e
    (strokes scale by s too) and fade 1 -> 0 over e in [0.45, 0.80]; the token glyph
    (self.glyph, scale 0.94) fades in over e in [0.55, 1.0]. Then SETTLE scales outline +
    glyph 0.94 -> 1.00 about AC (0.17 s, ease_out_back).

        c = COLLAPSE(frame, contents, 'core')         # captures rest state now
        self.play(c.anim(0.65))                       # ease_in_out_expo; adds c.glyph
        self.play(c.settle_anim())                    # 0.17 s
        exit_to(self)                                 # swap in the canonical exit state

    Updater use: self.add(c.glyph) once, then c.at(ease_in_out_expo(p)) / c.settle(p).
    outline must be ONE VMobject (e.g. the frame RoundedRectangle); contents any Mobject
    (or None)."""

    def __init__(self, outline: VMobject, contents: Mobject | None, kind: str = "plain",
                 color=CYAN, contents_fade=(0.45, 0.80), glyph_fade=(0.55, 1.0),
                 scale_strokes: bool = True):
        self.kind, self.color = kind, color
        self.outline, self.contents = outline, contents
        self.contents_fade, self.glyph_fade, self.scale_strokes = contents_fade, glyph_fade, scale_strokes
        tok = handoff_token(kind, color)
        target = tok.base.copy().scale(SETTLE_FROM, about_point=AC)
        src = outline.copy()
        if len(src.points) != len(target.points):
            src.align_points(target)
            outline.points = src.points.copy()
        self._o = _Rest(outline)
        self._t = _Rest(target)
        self.C = outline.get_center()
        self.s1 = COLLAPSE_SIZE / max(outline.width, outline.height)
        self._c = [_Rest(m) for m in contents.family_members_with_points()] if contents is not None else []
        self.glyph = tok.glyph.copy().scale(SETTLE_FROM, about_point=AC)
        self._g = [_Rest(m) for m in self.glyph.family_members_with_points()]
        for r in self._g:
            _set_alpha(r.m, r, 0.0)
        self._e = 0.0

    def at(self, e: float) -> None:
        """Collapse state at eased progress e in [0, 1]."""
        self._e = e
        o, r0, r1 = self.outline, self._o, self._t
        o.points = lerp(r0.p, r1.p, e)
        o.fill_rgbas = lerp(r0.fr, r1.fr, e)
        o.stroke_rgbas = lerp(r0.sr, r1.sr, e)
        o.stroke_width = lerp(r0.sw, r1.sw, e)
        c = lerp(self.C, AC, e)
        s = self.s1 ** e
        a = 1.0 - ramp(e, *self.contents_fade)
        for r in self._c:
            r.m.points = (r.p - self.C) * s + c
            _set_alpha(r.m, r, a)
            r.m.stroke_width = r.sw * s if self.scale_strokes else r.sw
        ga = ramp(e, *self.glyph_fade)
        for r in self._g:
            r.m.points = r.p.copy()
            _set_alpha(r.m, r, ga)

    def settle(self, p: float) -> None:
        """SETTLE at linear progress p in [0, 1] (scale 0.94 -> 1.0, ease_out_back, about AC)."""
        if self._e < 1.0:
            self.at(1.0)
        s = lerp(SETTLE_FROM, 1.0, ease_out_back(clip01(p))) / SETTLE_FROM
        self.outline.points = (self._t.p - AC) * s + AC
        for r in self._g:
            r.m.points = (r.p - AC) * s + AC

    def anim(self, run_time: float, rate_func=ease_in_out_expo) -> Drive:
        return Drive(self.glyph, self.at, run_time=run_time, rate_func=rate_func)

    def settle_anim(self, run_time: float = SETTLE_TIME) -> Drive:
        return Drive(self.glyph, self.settle, run_time=run_time, rate_func=linear)

    def token(self) -> Token:
        """A fresh canonical handoff_token(kind, color) (what the settled state equals)."""
        return handoff_token(self.kind, self.color)


def COLLAPSE(outline: VMobject, contents: Mobject | None, kind: str = "plain", color=CYAN, **kw) -> Collapse:
    """Alias that reads like the storyboard: returns a Collapse (see there)."""
    return Collapse(outline, contents, kind, color, **kw)


# ==========================================================================
# Effects: shock rings, entry ring, full-frame flash
# ==========================================================================
_UNIT_CIRCLE = Circle(radius=1.0).points.copy()


class ShockRing(Circle):
    """Expanding ring as a pure function of scene.now: from t0 over dur, radius r0 -> r1,
    stroke width w0 -> w1, opacity op0 -> op1, all on ease(p). Invisible outside
    [t0, t0+dur). Stays in the scene (remove it between plays if you like)."""

    def __init__(self, scene, t0: float, center=AC, r0: float = 0.2, r1: float = 6.5,
                 w0: float = 6.0, op0: float = 1.0, dur: float = 0.5, color=INK,
                 ease=ease_out_cubic, w1: float = 0.0, op1: float = 0.0):
        super().__init__(radius=1.0, stroke_color=color, stroke_width=0, fill_opacity=0)
        self.t0, self.c, self.r0, self.r1, self.w0, self.w1 = t0, np.asarray(center, float), r0, r1, w0, w1
        self.op0, self.op1, self.dur, self.ring_color, self.ease = op0, op1, dur, color, ease
        self.render_at(scene.now if scene is not None else 0.0)
        if scene is not None:
            self.add_updater(lambda m, dt: m.render_at(scene.now))

    def render_at(self, now: float) -> None:
        p = (now - self.t0) / self.dur
        if p < -EPS or p >= 1.0 - EPS:
            self.points = _UNIT_CIRCLE * self.r0 + self.c
            self.set_stroke(self.ring_color, width=0, opacity=0)
            return
        e = self.ease(clip01(p))
        self.points = _UNIT_CIRCLE * lerp(self.r0, self.r1, e) + self.c
        self.set_stroke(self.ring_color, width=lerp(self.w0, self.w1, e), opacity=lerp(self.op0, self.op1, e))


def add_behind(scene, *mobs) -> None:
    """Add mobs to the scene directly above the dust (behind all content). Call between plays."""
    scene.add(*mobs)
    for m in mobs:
        scene.mobjects.remove(m)
    dust = getattr(scene, "dust", None)
    i = scene.mobjects.index(dust) + 1 if dust is not None and dust in scene.mobjects else 0
    for j, m in enumerate(mobs):
        scene.mobjects.insert(i + j, m)


def entry_ring(scene, behind: bool = True) -> ShockRing:
    """The b02-b07 entry ring: Circle at AC, r 0.85 -> 3.2, stroke INK w8 -> 0, opacity
    0.8 -> 0, 0.45 s ease_out_cubic, starting at story frame 1 (t = 1/60), never frame 0.
    Added behind content (above the dust). Call at the very start of beat()."""
    ring = ShockRing(scene, 1 / 60, AC, 0.85, 3.2, 8.0, 0.8, 0.45, INK, ease_out_cubic)
    if behind:
        add_behind(scene, ring)
    else:
        scene.add(ring)
    return ring


class FrameFlash(Rectangle):
    """Full-frame colour flash driven by scene.now: opacity 0 -> peak over `rise` s, then
    peak -> 0 linearly until t0 + dur. b01: FrameFlash(self, 1.0, 0.22, 0.12);
    b08: FrameFlash(self, 5.90, 0.12, 0.30, rise=0.15, color=AMBER)."""

    def __init__(self, scene, t0: float, peak: float = 0.22, dur: float = 0.12,
                 rise: float = 0.0, color=INK):
        super().__init__(width=FRAME_W + 0.4, height=FRAME_H + 0.4, stroke_width=0,
                         fill_color=color, fill_opacity=0)
        self.t0, self.peak, self.dur, self.rise = t0, peak, dur, rise
        self.render_at(scene.now if scene is not None else 0.0)
        if scene is not None:
            self.add_updater(lambda m, dt: m.render_at(scene.now))

    def opacity_at(self, now: float) -> float:
        d = now - self.t0
        if d < -EPS or d >= self.dur - EPS:
            return 0.0
        if self.rise > 0 and d < self.rise:
            return self.peak * clip01(d / self.rise)
        return self.peak * (1 - clip01((d - self.rise) / max(self.dur - self.rise, 1e-9)))

    def render_at(self, now: float) -> None:
        self.set_fill(opacity=self.opacity_at(now))


# ==========================================================================
# The hook grid (b01 + b09)
# ==========================================================================
HOOK_COLS, HOOK_ROWS = 20, 16
HOOK_N = HOOK_COLS * HOOK_ROWS              # 320
HOOK_CELL = 0.27
HOOK_CORNER = 0.05
HOOK_CORNER_RATIO = HOOK_CORNER / HOOK_CELL  # 0.185
HOOK_PITCH = 0.34
HOOK_WORKER_SCALE = 1.15
HOOK_PUSH_SCALE = COLLAPSE_SIZE / HOOK_CELL  # 4.178: b01 push-in end scale


def hook_rc(idx: int) -> tuple[int, int]:
    """Serpentine work order: row = idx//20; col = idx%20 on even rows, 19 - idx%20 on odd."""
    r = idx // HOOK_COLS
    c = idx % HOOK_COLS if r % 2 == 0 else HOOK_COLS - 1 - idx % HOOK_COLS
    return r, c


HOOK_RC = np.array([hook_rc(i) for i in range(HOOK_N)])
HOOK_CENTERS = np.stack([X_C + (HOOK_RC[:, 1] - 9.5) * HOOK_PITCH,
                         AC[1] + (7.5 - HOOK_RC[:, 0]) * HOOK_PITCH,
                         np.zeros(HOOK_N)], axis=1)


def hook_center(idx: int) -> np.ndarray:
    """Centre of work-order cell idx (idx 0 = (-3.58, 3.00), 32 = (-1.20, 2.66), 49 = (-0.52, 2.32))."""
    return HOOK_CENTERS[idx].copy()


HOOK_IDLE_FILL = rgba(DIMMER, 0.9)
HOOK_IDLE_STROKE = rgba(DIM, 0.35)
HOOK_DONE_STROKE = rgba(AMBER, 0.5)
HOOK_WORKER_FILL = rgba(AMBER, 1.0)
HOOK_WORKER_STROKE = rgba(INK, 1.0)
HOOK_LIT_STROKE = rgba(CYAN, 0.6)


def hook_trail_opacity(age):
    """DONE-cell fill opacity for age = worker_idx - idx (>= 1): 0.28 + 0.55*max(0, 1-(age-1)/6)."""
    return 0.28 + 0.55 * np.maximum(0.0, 1.0 - (np.asarray(age, float) - 1.0) / 6.0)


_RR_TMPL: dict = {}


def _rr_unit(ratio: float) -> np.ndarray:
    key = round(float(ratio), 7)
    p = _RR_TMPL.get(key)
    if p is None:
        p = RoundedRectangle(width=1.0, height=1.0, corner_radius=key).points.copy()
        _RR_TMPL[key] = p
    return p


class HookGrid(VGroup):
    """320 RoundedRectangle cells in WORK ORDER (self[idx]); one vectorised draw().

    draw(centers=None, size=None, fill=None, stroke=None, sw=None, corner=None):
      centers (320,3) (default HOOK_CENTERS), size (320,) multiplier of the 0.27 cell,
      fill/stroke (320,4) RGBA rows, sw (320,) stroke widths,
      corner {idx: corner_ratio} overrides (default 0.05/0.27; the token is 0.22).
    crawl_arrays(w) -> dict(fill, stroke, sw, size) for the crawl rule (DONE trail / WORKER /
    IDLE); edit the copies (LIT, flashes, push-in) and draw() them.
    set_crawl(w) == draw(**crawl_arrays(w)) at rest: exactly hook_state(w)."""

    def __init__(self):
        super().__init__(*[RoundedRectangle(width=HOOK_CELL, height=HOOK_CELL, corner_radius=HOOK_CORNER)
                           for _ in range(HOOK_N)])
        self.worker_idx = None

    @staticmethod
    def crawl_arrays(worker_idx: int) -> dict:
        idx = np.arange(HOOK_N)
        fill = np.tile(HOOK_IDLE_FILL, (HOOK_N, 1))
        stroke = np.tile(HOOK_IDLE_STROKE, (HOOK_N, 1))
        sw = np.ones(HOOK_N)
        size = np.ones(HOOK_N)
        done = idx < worker_idx
        if np.any(done):
            fill[done, :3] = rgba(AMBER)[:3]
            fill[done, 3] = hook_trail_opacity(worker_idx - idx[done])
            stroke[done] = HOOK_DONE_STROKE
        if 0 <= worker_idx < HOOK_N:
            fill[worker_idx] = HOOK_WORKER_FILL
            stroke[worker_idx] = HOOK_WORKER_STROKE
            sw[worker_idx] = 2.0
            size[worker_idx] = HOOK_WORKER_SCALE
        return {"fill": fill, "stroke": stroke, "sw": sw, "size": size}

    def draw(self, centers=None, size=None, fill=None, stroke=None, sw=None, corner=None) -> "HookGrid":
        centers = HOOK_CENTERS if centers is None else np.asarray(centers, float)
        size = np.ones(HOOK_N) if size is None else np.asarray(size, float)
        base = _rr_unit(HOOK_CORNER_RATIO)
        corner = corner or {}
        for i, cell in enumerate(self.submobjects):
            P = base if i not in corner else _rr_unit(corner[i])
            cell.points = P * (HOOK_CELL * size[i]) + centers[i]
            if fill is not None:
                cell.fill_rgbas = fill[i:i + 1].copy()
            if stroke is not None:
                cell.stroke_rgbas = stroke[i:i + 1].copy()
            if sw is not None:
                cell.stroke_width = float(sw[i])
        return self

    def set_crawl(self, worker_idx: int) -> "HookGrid":
        self.worker_idx = worker_idx
        return self.draw(**self.crawl_arrays(worker_idx))


def hook_state(worker_idx: int) -> HookGrid:
    """Pure function: the hook grid at rest with the crawl styles for worker_idx
    (idx < w DONE amber comet trail, w = WORKER (x1.15, AMBER, INK w2), rest IDLE)."""
    return HookGrid().set_crawl(worker_idx)


def b01_worker_idx(f: int) -> int:
    """b01 crawl: 32 + floor((f+1)/3) for f0..f50 (hops at f2, f5, ...), frozen at 49 after."""
    return 32 + min((max(f, 0) + 1) // 3, 17)


def b09_worker_idx(f: int) -> int:
    """b09 crawl: floor((f - 113)/3) from f113 (idx 0) to f209 (idx 32); 0 before f113."""
    return max(0, (f - 113) // 3)


# ==========================================================================
# Handoff states: the contract between beats
# ==========================================================================
HANDOFF_NAMES = ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "LOOP")
CALC_LABEL = "CALCULATIONS AT ONCE"

_HO_TOKEN = {"E01": ("plain", CYAN), "E02": ("core", CYAN), "E03": ("cpu", CYAN),
             "E04": ("gpu", CYAN), "E05": ("node", CYAN), "E06": ("cluster", CYAN),
             "E08": ("plain", AMBER)}
_HO_HUD = {"E02": ("16", INK, CALC_LABEL, INK, 0.70),
           "E03": ("256", INK, CALC_LABEL, INK, 0.70),
           "E04": ("16,896", INK, CALC_LABEL, INK, 0.70),
           "E05": ("135,168", INK, CALC_LABEL, INK, 0.70),
           "E06": ("1,689,600,000+", INK, CALC_LABEL, INK, 0.70),
           "E07": ("350,000+", INK, "VOLUNTEER GPUs · 2020", CYAN, 0.90),
           "E08": ("10×", AMBER, "MAX SPEEDUP · 10% SERIAL", AMBER, 0.85)}

BEAT_ENTRY = {"B01Hook": "LOOP", "B02Simd": "E01", "B03Cpu": "E02", "B04Gpu": "E03",
              "B05Node": "E04", "B06DataCenter": "E05", "B07Planet": "E06",
              "B08Amdahl": "E07", "B09Loop": "E08"}
BEAT_EXIT = {"B01Hook": "E01", "B02Simd": "E02", "B03Cpu": "E03", "B04Gpu": "E04",
             "B05Node": "E05", "B06DataCenter": "E06", "B07Planet": "E07",
             "B08Amdahl": "E08", "B09Loop": "LOOP"}


class Handoff(VGroup):
    """A boundary frame. Attributes (None when absent): name, token (Token), hud (Hud,
    unbound), grid (HookGrid), card (Card), headline (card[0]), readout (Text)."""


def handoff_state(name: str) -> Handoff:
    """EXACT boundary frame builder ('E01'..'E08' = exit of b0N = entry of b0N+1;
    'LOOP' (aliases 'E09', 'E00') = b09's exit = b01's frame 0). Dust is not included.
      E01  handoff_token('plain')
      E02  token('core')    + HUD '16'             / CALCULATIONS AT ONCE (INK 0.70)
      E03  token('cpu')     + HUD '256'            / ...
      E04  token('gpu')     + HUD '16,896'         / ...
      E05  token('node')    + HUD '135,168'        / ...
      E06  token('cluster') + HUD '1,689,600,000+' / ... (fit 0.627)
      E07  no token; HUD '350,000+' / 'VOLUNTEER GPUs · 2020' (CYAN 0.90)
      E08  token('plain', AMBER) + HUD '10×' AMBER / 'MAX SPEEDUP · 10% SERIAL' (AMBER 0.85)
      LOOP hook_state(32) + card 'One at a time.' (H96 INK) + step_readout('STEP 032 / 320', AMBER)
    Z-order: token, then HUD (grid, card, readout for LOOP)."""
    key = {"E09": "LOOP", "E00": "LOOP"}.get(name, name)
    if key not in HANDOFF_NAMES:
        raise ValueError(f"unknown handoff state {name!r}; known: {HANDOFF_NAMES}")
    st = Handoff()
    st.name, st.token, st.hud, st.grid, st.card, st.headline, st.readout = key, None, None, None, None, None, None
    if key == "LOOP":
        st.grid = hook_state(32)
        st.card = card([("One at a time.", "H96", INK)])
        st.headline = st.card[0]
        st.readout = step_readout(step_str(32), AMBER)
        st.add(st.grid, st.card, st.readout)
        return st
    if key in _HO_TOKEN:
        st.token = handoff_token(*_HO_TOKEN[key])
        st.add(st.token)
    if key in _HO_HUD:
        n, nc, lab, lc, lo = _HO_HUD[key]
        st.hud = Hud(n, lab, nc, lc, lo)
        st.add(st.hud)
    return st


def enter(scene, name: str | None = None) -> Handoff:
    """Put the beat's entry state on screen (call first thing in beat()). Each part is added
    as its own top-level mobject (so scene.remove(st.token) works), and st.hud is bound to
    the scene (schedule events on it). name defaults to BEAT_ENTRY[type(scene).__name__]."""
    st = handoff_state(name or BEAT_ENTRY[type(scene).__name__])
    scene.add(*st.submobjects)
    if st.hud is not None:
        st.hud.bind(scene)
    return st


def exit_to(scene, name: str | None = None) -> Handoff:
    """Swap in the canonical exit state: removes every mobject except the dust and adds
    handoff_state(name) (default BEAT_EXIT[type(scene).__name__]). Call BETWEEN plays,
    right after your settle, at least 0.05 s (3 frames at 60 fps) before the beat ends;
    BeatScene pads the rest with a static hold."""
    dust = getattr(scene, "dust", None)
    for m in list(scene.mobjects):
        if m is not dust:
            scene.remove(m)
    st = handoff_state(name or BEAT_EXIT[type(scene).__name__])
    scene.add(st)
    return st
