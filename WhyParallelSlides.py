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
        
        # Draw Axes
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 5, 1],
            axis_config={"include_numbers": False},
            x_length=6,
            y_length=4
        ).shift(DOWN * 0.5)
        
        labels = axes.get_axis_labels(Tex("Time").scale(0.7), Text("Clock Speed").scale(0.45).shift(LEFT * 2))
        
        # The Curve: Linear rise then plateau
        graph = axes.plot(lambda x: 4 * (1 - np.exp(-x)), color=YELLOW)
        
        # The "Heat" Limit Line
        limit_line = DashedLine(
            start=axes.c2p(0, 4), 
            end=axes.c2p(6, 4), 
            color=RED
        )
        limit_text = Text("Power & Heat Wall", font_size=24, color=RED).next_to(limit_line, UP)

        point_text = Text("1. Single-core limits", font_size=32).next_to(title, DOWN, buff=0.5)

        self.play(Write(point_text))
        self.play(Create(axes), Write(labels))
        self.play(Create(graph), run_time=2)
        self.play(Create(limit_line), FadeIn(limit_text))
        
        # Pause to explain Dennard Scaling/Moore's Law limits (Click 2)
        self.next_slide()

        self.play(FadeOut(axes), FadeOut(labels), FadeOut(graph), FadeOut(limit_line), FadeOut(limit_text), FadeOut(point_text))

        # -----------------------------------------
        # SLIDE 3: Workload Explosion
        # -----------------------------------------
        point2_text = Text("2. Workloads Exploded", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(point2_text))

        # Visual: A small data circle growing huge and spawning children
        center_blob = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        center_label = Text("Data", font_size=24).move_to(center_blob)
        blob_group = VGroup(center_blob, center_label).move_to(ORIGIN)

        #self.play(FadeIn(blob_group))
        
        # Animate explosion
        new_blob = Circle(radius=25, color=BLUE, fill_opacity=0.2)
        
        # Satellite bubbles
        ai_bubble = Circle(radius=10, color=PURPLE, fill_opacity=0.2).move_to(new_blob.get_center()+ LEFT*5)
        ai_text = Text("AI", font_size=24).move_to(ai_bubble)
        
        rt_bubble = Circle(radius=10, color=TEAL, fill_opacity=0.2).move_to(new_blob.get_center() +RIGHT*5)
        rt_text = Text("Real-time", font_size=20).move_to(rt_bubble)
        
        sim_bubble = Circle(radius=10, color=ORANGE, fill_opacity=0.2).move_to(rt_bubble.get_center() +RIGHT*5)
        sim_text = Text("Simulations", font_size=20).move_to(sim_bubble)

        self.play(
            GrowFromCenter(new_blob), Write(center_label),
            GrowFromCenter(ai_bubble), Write(ai_text),
            GrowFromCenter(rt_bubble), Write(rt_text),
            GrowFromCenter(sim_bubble), Write(sim_text),
        )

        # Pause to discuss the needs of modern computing (Click 3)
        self.next_slide()
        
        self.play(FadeOut(new_blob), FadeOut(center_label), FadeOut(ai_bubble), FadeOut(ai_text), FadeOut(rt_bubble), FadeOut(rt_text), FadeOut(sim_bubble), FadeOut(sim_text), FadeOut(point2_text))

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