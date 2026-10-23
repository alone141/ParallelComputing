from style import *

class FoundationTest(BeatScene):
    def beat(self):
        k = kicker("Scale 03 · GPU", color=CYAN).move_to([SAFE_CENTER[0], 5.6, 0])
        h = fit_width(headline("21,760 cores", size=110)).next_to(k, DOWN, buff=0.35)
        b = fit_width(body("one instruction, many numbers", size=44, color=DIM)).next_to(h, DOWN, buff=0.3)
        w = VGroup(*[headline(f"{wt}", size=60, weight=wt) for wt in [MEDIUM, SEMIBOLD, BOLD, ULTRABOLD, HEAVY]]).arrange(DOWN, buff=0.15).move_to([SAFE_CENTER[0], 0, 0])
        tk = token(1.2, MAGENTA)
        g = glow(tk, MAGENTA).move_to([SAFE_CENTER[0]-2.2, -3, 0])
        dot = glow_dot([SAFE_CENTER[0]+2, -3, 0], 0.1, LIME)
        val = ValueTracker(1)
        ctr = counter_text(lambda: val.get_value(), size=80, color=AMBER, anchor=[SAFE_CENTER[0], -3, 0])
        assert_safe(k, h, b, w)
        self.cue("hit")
        self.play(FadeIn(k, shift=DOWN*0.3), FadeIn(h, scale=1.15), rate_func=ease_out_expo, run_time=0.6)
        self.add(b, w, g, dot, ctr, chip("WARP 32", LIME).move_to([SAFE_CENTER[0], 3.2, 0]))
        self.cue("riser", dt=0.0)
        self.play(val.animate.set_value(21760), run_time=1.6, rate_func=ease_in_out_expo)
