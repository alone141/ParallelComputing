from manim import *
from manim_slides import Slide

class ParallelComputingSlides(Slide):
    def construct(self):
        # -----------------------------------------
        # SLIDE 1: Title & Definition
        # -----------------------------------------
        title = Text("What is Parallel Computing?", font_size=48, color=BLUE)
        self.play(Write(title))
        
        # Pause here waiting for user input (Click 1)
        self.next_slide() 
        
        self.play(title.animate.to_edge(UP))

        # Definition Text
        def_text = Text(
            "Solving a problem by doing multiple parts\nof the work at the same time.",
            font_size=36
        )
        def_text.set_color_by_gradient(WHITE, YELLOW)
        
        self.play(Write(def_text))
        
        # Pause for user to read definition (Click 2)
        self.next_slide()
        
        self.play(FadeOut(def_text))

        # -----------------------------------------
        # SLIDE 2: Scales of Parallelism (Intro)
        # -----------------------------------------
        scale_header = Text("Exists Across Scales", font_size=40, color=TEAL).next_to(title, DOWN)
        self.play(FadeIn(scale_header))
        
        # Pause before listing scales (Click 3)
        self.next_slide()

        # -----------------------------------------
        # SLIDE 3: Scale 1 - SIMD
        # -----------------------------------------
        s1_text = Text("1. Inside a CPU Core (SIMD)", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Visual: Vector Block
        vec_box = Rectangle(width=4, height=0.8, color=GREEN)
        vec_data = VGroup(*[Square(side_length=0.6, fill_opacity=0.5, fill_color=GREEN).move_to(vec_box.get_left() + RIGHT*(0.5 + i)) for i in range(4)])
        vec_label = Text("Vector Instruction", font_size=20).next_to(vec_box, UP)
        simd_group = VGroup(vec_box, vec_data, vec_label).next_to(s1_text, DOWN)

        self.play(Write(s1_text))
        self.play(Create(vec_box), FadeIn(vec_data), Write(vec_label))
        
        # Pause to explain SIMD (Click 4)
        self.next_slide()
        
        self.play(FadeOut(simd_group), FadeOut(s1_text))

        # -----------------------------------------
        # SLIDE 4: Scale 2 - Multicore
        # -----------------------------------------
        s2_text = Text("2. Across CPU Cores (Threads)", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Visual: CPU Chip
        chip_bg = Square(side_length=3, color=GREY, fill_opacity=0.2)
        cores = VGroup(
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(UL) + DOWN*0.8 + RIGHT*0.8),
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(UR) + DOWN*0.8 + LEFT*0.8),
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(DL) + UP*0.8 + RIGHT*0.8),
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(DR) + UP*0.8 + LEFT*0.8),
        )
        core_labels = VGroup(*[Text("Core", font_size=16).move_to(c.get_center()) for c in cores])
        cpu_group = VGroup(chip_bg, cores, core_labels).next_to(s2_text, DOWN)

        self.play(Write(s2_text))
        self.play(Create(chip_bg), GrowFromCenter(cores), Write(core_labels))
        
        # Pause to explain Multicore (Click 5)
        self.next_slide()

        self.play(FadeOut(cpu_group), FadeOut(s2_text))

        # -----------------------------------------
        # SLIDE 5: Scale 3 - GPU
        # -----------------------------------------
        s3_text = Text("3. Across Many GPU Cores", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Visual: GPU Grid
        gpu_bg = Rectangle(width=5, height=3, color=GREEN_E, fill_opacity=0.2)
        small_cores = VGroup()
        for x in range(10):
            for y in range(6):
                dot = Dot(radius=0.08, color=GREEN_B)
                dot.move_to(gpu_bg.get_corner(UL) + RIGHT*(0.25 + x*0.5) + DOWN*(0.25 + y*0.5))
                small_cores.add(dot)
        gpu_group = VGroup(gpu_bg, small_cores).next_to(s3_text, DOWN)

        self.play(Write(s3_text))
        self.play(Create(gpu_bg), ShowIncreasingSubsets(small_cores))
        
        # Pause to explain GPU (Click 6)
        self.next_slide()

        self.play(FadeOut(gpu_group), FadeOut(s3_text))

        # -----------------------------------------
        # SLIDE 6: Scale 4 - Distributed
        # -----------------------------------------
        s4_text = Text("4. Across Multiple Machines (Distributed)", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Visual: Servers
        server1 = RoundedRectangle(height=1.5, width=1, corner_radius=0.2, color=WHITE).shift(LEFT*2.5)
        server2 = RoundedRectangle(height=1.5, width=1, corner_radius=0.2, color=WHITE)
        server3 = RoundedRectangle(height=1.5, width=1, corner_radius=0.2, color=WHITE).shift(RIGHT*2.5)
        lines = VGroup(
            Line(server1.get_right(), server2.get_left(), color=YELLOW),
            Line(server2.get_right(), server3.get_left(), color=YELLOW)
        )
        dist_group = VGroup(server1, server2, server3, lines).next_to(s4_text, DOWN)

        self.play(Write(s4_text))
        self.play(DrawBorderThenFill(server1), DrawBorderThenFill(server2), DrawBorderThenFill(server3))
        self.play(Create(lines))
        
        # Pause to explain Distributed (Click 7)
        self.next_slide()

        self.play(FadeOut(dist_group), FadeOut(s4_text), FadeOut(scale_header))

        # -----------------------------------------
        # SLIDE 7: Anchor Sentence
        # -----------------------------------------
        anchor_text = Paragraph(
            "“Parallel computing is a strategy to reduce",
            "time-to-solution by increasing simultaneous",
            "work, from a single chip to a data center.”",
            alignment="center",
            font_size=34
        )
        anchor_text.set_color(YELLOW)
        anchor_text.next_to(title, DOWN, buff=1.5)

        self.play(Write(anchor_text), run_time=3)
        
        # Final pause before ending (Click 8)
        self.next_slide()
        
        self.play(FadeOut(anchor_text), FadeOut(title))