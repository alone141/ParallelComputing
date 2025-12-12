from manim import *
from manim_slides import Slide

class WhyParallelSlides(Slide):
    def construct(self):
        # -----------------------------------------
        # SLIDE 1: Title
        # -----------------------------------------
        title = Text("Why do we need it?", font_size=48, color=BLUE)
        self.play(Write(title))
        
        # Pause for intro (Click 1)
        self.next_slide()
        
        self.play(title.animate.to_edge(UP))

        # -----------------------------------------
        # SLIDE 2: The Limits (Power/Heat)
        # -----------------------------------------
        # Concept: A graph showing Frequency increasing then flattening
        
# Create Axes
        # X: Clock Speed (Frequency)
        # Y: Power (Watts/Heat)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 100, 20],
            axis_config={"include_numbers": False, "tip_shape": StealthTip},
            x_length=6,
            y_length=5
        ).shift(DOWN * 0.5 + LEFT * 0.5)

        # Labels
        x_label = axes.get_x_axis_label("Clock Speed (GHz)")
        y_label = axes.get_y_axis_label("Power / Heat (Watts)")

        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Pause: "Let's look at the relationship between speed and heat."
        self.next_slide()

        # -----------------------------------------
        # STEP 1: The Exponential Curve
        # -----------------------------------------
        
        # Power is roughly proportional to Frequency^3 (P ~ V^2 * f)
        # We scale it to fit the graph nicely
        graph = axes.plot(lambda x: 1.5 * x**3, color=WHITE, x_range=[0, 4.1])
        
        # Color the curve with a gradient: Green (Safe) -> Yellow -> Red (Danger)
        graph.set_color_by_gradient(GREEN, YELLOW, RED)
        
        label_curve = Text("P ∝ f³", font_size=36, color=YELLOW).move_to(axes.c2p(2, 60))
        
        self.play(Create(graph, run_time=2), FadeIn(label_curve))

        # Pause: "As you can see, power doesn't rise linearly."
        self.next_slide()

        # -----------------------------------------
        # STEP 2: The "Safe Zone" vs "Danger Zone"
        # -----------------------------------------
        
        # Add a "Cooling Limit" Line (e.g., at Y=80)
        limit_line = DashedLine(
            start=axes.c2p(0, 80), 
            end=axes.c2p(5, 80), 
            color=RED
        )
        limit_text = Text("Cooling Limit", font_size=20, color=RED).next_to(limit_line, UP, aligned_edge=RIGHT)
        
        self.play(Create(limit_line), Write(limit_text))
        
        # Animate a Dot moving up the curve
        dot = Dot(color=WHITE)
        dot.move_to(axes.c2p(0, 0))
        self.add(dot)
        
        # Trace path
        path = MoveAlongPath(dot, graph, run_time=3, rate_func=linear)
        
        # Dynamic Text showing values
        val_tracker = ValueTracker(0)
        
        # Move dot
        self.play(
            path, 
            val_tracker.animate.set_value(4), 
            run_time=3
        )
        
        # Pause: "We hit the ceiling."
        self.next_slide()

        # -----------------------------------------
        # STEP 3: The Cost of Speed
        # -----------------------------------------
        
        # Show comparison:
        # Point A: 3 GHz (Moderate Power)
        # Point B: 4 GHz (Massive Power)
        
        pt_a = Dot(color=GREEN).move_to(axes.c2p(3, 1.5 * 3**3)) # 3GHz, ~40W
        pt_b = Dot(color=RED).move_to(axes.c2p(4, 1.5 * 4**3))   # 4GHz, ~96W
        
        line_a = axes.get_lines_to_point(pt_a.get_center(), color=GREEN)
        line_b = axes.get_lines_to_point(pt_b.get_center(), color=RED)
        
        self.play(FadeIn(pt_a), Create(line_a))
        label_a = Text("3 GHz", font_size=20, color=GREEN).next_to(pt_a, LEFT)
        self.play(Write(label_a))
        
        self.next_slide()
        
        self.play(FadeIn(pt_b), Create(line_b))
        label_b = Text("4 GHz", font_size=20, color=RED).next_to(pt_b, LEFT)
        self.play(Write(label_b))

        # Annotate the gap
        brace = BraceBetweenPoints(axes.c2p(4.2, 40), axes.c2p(4.2, 96), direction=RIGHT)
        text_gain = Text("+33% Speed", font_size=20, color=GREEN).next_to(pt_b, UP)
        text_cost = Text("+140% Heat!", font_size=24, color=RED).next_to(brace, RIGHT)
        
        self.play(Create(brace), Write(text_cost))

        # Pause: "Small speed gain = Huge heat cost."
        self.next_slide()

        # Cleanup
        self.play(
            FadeOut(graph), FadeOut(axes), FadeOut(dot), FadeOut(limit_line), 
            FadeOut(limit_text), FadeOut(pt_a), FadeOut(pt_b), 
            FadeOut(line_a), FadeOut(line_b), FadeOut(brace), 
            FadeOut(text_cost), FadeOut(title), 
            FadeOut(x_label), FadeOut(y_label), FadeOut(label_curve), FadeOut(label_a), FadeOut(label_b)
        )
        # -----------------------------------------
        # SLIDE 4: Efficiency
        # -----------------------------------------
        point3_text = Text("3. Efficiency (Perf/Watt)", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(point3_text))

        # Visual: A battery bar or scale
        # Simple equation visual
        
        eff_eq = MathTex(r"\text{Performance} \propto \frac{\text{Work}}{\text{Energy}}").scale(1.5)
        self.play(Write(eff_eq))
        
        # Highlight: Parallelism is the path
        sub_text = Text("Parallelism is the path to better efficiency.", color=YELLOW, font_size=24).next_to(eff_eq, DOWN, buff=1)
        self.play(FadeIn(sub_text))

        # Pause to explain energy constraints (Click 4)
        self.next_slide()
        
        self.play(FadeOut(eff_eq), FadeOut(sub_text), FadeOut(point3_text))

        # -----------------------------------------
        # SLIDE 5: Bridge Line
        # -----------------------------------------
        # Visual: Single Fast Core -> Many Slower Cores
        
        bridge_text = Paragraph(
            "“So performance growth shifted from",
            "faster cores to more cores",
            "and specialized hardware.”",
            alignment="center",
            font_size=36
        )
        bridge_text.set_color_by_gradient(BLUE, TEAL)
        
        self.play(Write(bridge_text), run_time=2)
        
        # Final Pause (Click 5)
        self.next_slide()
        
        self.play(FadeOut(bridge_text), FadeOut(title))