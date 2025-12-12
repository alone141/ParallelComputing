from manim import *
from manim_slides import Slide
import numpy as np

class ParallelComputingPresentation(Slide):
    def construct(self):
        # The Narrative Flow
        self.chapter_1_intro()      # from ParallelComputingViz.py
        self.chapter_2_why()        # from WhyParallelSlides.py
        self.chapter_3_apps()       # from ApplicationsSlides.py
        self.chapter_4_how()        # from HowToParallelizeSlides.py
        self.chapter_5_hardware()   # from HardwareArchitectures.py
        self.chapter_6_cuda()       # from CudaHistorySlides.py
        self.chapter_7_perf()       # from PerformanceRecommendations.py

    def chapter_1_intro(self):
        # -----------------------------------------
        # SLIDE 1: Title & Definition
        # -----------------------------------------
        title = Text("What is Parallel Computing?", font_size=48, color=BLUE)
        self.play(Write(title))
        
        self.next_slide() 
        
        self.play(title.animate.to_edge(UP))

        # Definition Text
        def_text = Text(
            "Solving a problem by doing multiple parts\nof the work at the same time.",
            font_size=36
        )
        def_text.set_color_by_gradient(WHITE, YELLOW)
        
        self.play(Write(def_text))
        
        self.next_slide()
        
        self.play(FadeOut(def_text))

        # -----------------------------------------
        # SLIDE 2: Scales of Parallelism (Intro)
        # -----------------------------------------
        scale_header = Text("Exists Across Scales", font_size=40, color=TEAL).next_to(title, DOWN)
        self.play(FadeIn(scale_header))
        
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
        
        self.next_slide()
        
        self.play(FadeOut(anchor_text), FadeOut(title))
        self.wait(0.5)

    def chapter_2_why(self):
        # -----------------------------------------
        # SLIDE 1: Title
        # -----------------------------------------
        title = Text("Why do we need it?", font_size=48, color=BLUE)
        self.play(Write(title))
        
        self.next_slide()
        
        self.play(title.animate.to_edge(UP))

        # -----------------------------------------
        # SLIDE 2: The Limits (Power/Heat)
        # -----------------------------------------
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 100, 20],
            axis_config={"include_numbers": False, "tip_shape": StealthTip},
            x_length=6,
            y_length=5
        ).shift(DOWN * 0.5 + LEFT * 0.5)

        x_label = axes.get_x_axis_label("Clock Speed (GHz)")
        y_label = axes.get_y_axis_label("Power / Heat (Watts)")

        self.play(Create(axes), Write(x_label), Write(y_label))
        
        self.next_slide()

        # -----------------------------------------
        # STEP 1: The Exponential Curve
        # -----------------------------------------
        graph = axes.plot(lambda x: 1.5 * x**3, color=WHITE, x_range=[0, 4.1])
        graph.set_color_by_gradient(GREEN, YELLOW, RED)
        
        label_curve = Text("P ∝ f³", font_size=36, color=YELLOW).move_to(axes.c2p(2, 60))
        
        self.play(Create(graph, run_time=2), FadeIn(label_curve))

        #self.next_slide()

        # -----------------------------------------
        # STEP 2: The "Safe Zone" vs "Danger Zone"
        # -----------------------------------------
        limit_line = DashedLine(
            start=axes.c2p(0, 80), 
            end=axes.c2p(5, 80), 
            color=RED
        )
        limit_text = Text("Cooling Limit", font_size=20, color=RED).next_to(limit_line, UP, aligned_edge=RIGHT)
        
        self.play(Create(limit_line), Write(limit_text))
        
        dot = Dot(color=WHITE)
        dot.move_to(axes.c2p(0, 0))
        self.add(dot)
        
        path = MoveAlongPath(dot, graph, run_time=3, rate_func=linear)
        val_tracker = ValueTracker(0)
        
        self.play(
            path, 
            val_tracker.animate.set_value(4), 
            run_time=3
        )
        
        self.next_slide()

        # -----------------------------------------
        # STEP 3: The Cost of Speed
        # -----------------------------------------
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

        brace = BraceBetweenPoints(axes.c2p(4.2, 40), axes.c2p(4.2, 96), direction=RIGHT)
        text_gain = Text("+33% Speed", font_size=20, color=GREEN).next_to(pt_b, UP)
        text_cost = Text("+140% Heat!", font_size=24, color=RED).next_to(brace, RIGHT)
        
        self.play(Create(brace), Write(text_cost))

        self.next_slide()

        self.play(
            FadeOut(graph), FadeOut(axes), FadeOut(dot), FadeOut(limit_line), 
            FadeOut(limit_text), FadeOut(pt_a), FadeOut(pt_b), 
            FadeOut(line_a), FadeOut(line_b), FadeOut(brace), 
            FadeOut(text_cost), FadeOut(title), 
            FadeOut(x_label), FadeOut(y_label), FadeOut(label_curve), FadeOut(label_a), FadeOut(label_b)
        )
        
        # Re-title for continuity if desired, or just new section
        title = Text("Why do we need it?", font_size=40, color=BLUE).to_edge(UP)

        # -----------------------------------------
        # SLIDE 4: Efficiency
        # -----------------------------------------
        point3_text = Text("3. Efficiency (Perf/Watt)", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(title), Write(point3_text))

        eff_eq = MathTex(r"\text{Performance} \propto \frac{\text{Work}}{\text{Energy}}").scale(1.5)
        self.play(Write(eff_eq))
        
        sub_text = Text("Parallelism is the path to better efficiency.", color=YELLOW, font_size=24).next_to(eff_eq, DOWN, buff=1)
        self.play(FadeIn(sub_text))

        self.next_slide()
        
        self.play(FadeOut(eff_eq), FadeOut(sub_text), FadeOut(point3_text))

        # -----------------------------------------
        # SLIDE 5: Bridge Line
        # -----------------------------------------
        bridge_text = Paragraph(
            "“So performance growth shifted from",
            "faster cores to more cores",
            "and specialized hardware.”",
            alignment="center",
            font_size=36
        )
        bridge_text.set_color_by_gradient(BLUE, TEAL)
        
        self.play(Write(bridge_text), run_time=2)
        
        self.next_slide()
        
        self.play(FadeOut(bridge_text), FadeOut(title))
        self.wait(0.5)

    def chapter_3_apps(self):
        # -----------------------------------------
        # SETUP: Title and Central Hub
        # -----------------------------------------
        title = Text("3. What can we do with it?", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))

        # Central Hub
        hub_circle = Circle(radius=1.2, color=WHITE, fill_opacity=0.1)
        hub_text = Paragraph("Parallel", "Computing", alignment="center", font_size=24)
        hub = VGroup(hub_circle, hub_text).move_to(ORIGIN)
        
        self.play(DrawBorderThenFill(hub_circle), Write(hub_text))
        
        self.next_slide()

        # -----------------------------------------
        # DEFINING THE 6 CATEGORIES
        # -----------------------------------------
        radius = 3.0
        angles = [150, 90, 30, 210, 270, 330] # Degrees
        positions = [
            radius * np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0]) 
            for a in angles
        ]

        # --- 1. Scientific Computing (Mesh/Grid) ---
        science_pos = positions[0]
        science_grid = VGroup()
        for i in range(3):
            for j in range(3):
                sq = Square(side_length=0.4, color=BLUE, fill_opacity=0.5)
                sq.move_to(science_pos + np.array([(i-1)*0.4, (j-1)*0.4, 0]))
                science_grid.add(sq)
        science_label = Text("Scientific\n(CFD/PDEs)", font_size=18).next_to(science_grid, UP)
        line_1 = Line(hub.get_left(), science_grid.get_bottom(), color=GREY).set_opacity(0.5)

        # --- 3. AI/ML (Neural Net) ---
        ai_pos = positions[2]
        nodes_L1 = VGroup(*[Dot(radius=0.1, color=PURPLE).move_to(ai_pos + LEFT*0.5 + UP*(i-0.5)*0.5) for i in range(2)])
        nodes_L2 = VGroup(*[Dot(radius=0.1, color=PURPLE).move_to(ai_pos + RIGHT*0.5 + UP*(i-1)*0.5) for i in range(3)])
        edges = VGroup()
        for n1 in nodes_L1:
            for n2 in nodes_L2:
                edges.add(Line(n1.get_center(), n2.get_center(), stroke_width=1, color=PURPLE_A))
        ai_group = VGroup(edges, nodes_L1, nodes_L2)
        ai_label = Text("AI / ML", font_size=18).next_to(ai_group, UP)
        line_3 = Line(hub.get_right(), ai_group.get_bottom(), color=GREY).set_opacity(0.5)

        # --- 4. Media (Pixel/Image) ---
        media_pos = positions[3]
        media_pixels = VGroup()
        colors = [RED, GREEN, BLUE, YELLOW]
        for i in range(2):
            for j in range(2):
                c = colors[(i+j)%4]
                p = Square(side_length=0.5, color=c, fill_opacity=0.8, stroke_width=0)
                p.move_to(media_pos + np.array([(i-0.5)*0.5, (j-0.5)*0.5, 0]))
                media_pixels.add(p)
        media_label = Text("Media\n(Video/Img)", font_size=18).next_to(media_pixels, DOWN)
        line_4 = Line(hub.get_left(), media_pixels.get_top(), color=GREY).set_opacity(0.5)

        # --- 5. Signal Processing (Wave) ---
        sig_pos = positions[4]
        sine_wave = FunctionGraph(
            lambda t: 0.5 * np.sin(3*t),
            x_range=[-1, 1],
            color=TEAL
        ).move_to(sig_pos)
        sig_label = Text("Signal Proc.\n(Radar/FFT)", font_size=18).next_to(sine_wave, DOWN)
        line_5 = Line(hub.get_bottom(), sine_wave.get_top(), color=GREY).set_opacity(0.5)

        # --- 6. Finance (Stock Chart) ---
        fin_pos = positions[5]
        fin_line = VMobject().set_color(GREEN)
        pts = [
            [-1, -0.5, 0], [-0.5, -0.2, 0], [0, -0.6, 0], 
            [0.5, 0.2, 0], [1, 0.8, 0]
        ]
        real_pts = [np.array(p) * 0.8 + fin_pos for p in pts]
        fin_line.set_points_as_corners(real_pts)
        fin_label = Text("Finance", font_size=18).next_to(fin_line, DOWN)
        line_6 = Line(hub.get_right(), fin_line.get_top(), color=GREY).set_opacity(0.5)

        # -----------------------------------------
        # ANIMATION SEQUENCE
        # -----------------------------------------
        
        # 1. Science
        self.play(Create(line_1), FadeIn(science_grid), Write(science_label))
        self.play(science_grid.animate.set_color(RED), run_time=0.5) 
        #self.next_slide()

        # 3. AI
        self.play(Create(line_3), Create(ai_group), Write(ai_label))
        self.play(ShowPassingFlash(edges.copy().set_color(WHITE), time_width=0.5), run_time=1)
        #self.next_slide()

        # 4. Media
        self.play(Create(line_4), FadeIn(media_pixels), Write(media_label))
        self.play(Rotate(media_pixels, PI/2), run_time=0.5) 
        #self.next_slide()

        # 5. Signal
        self.play(Create(line_5), Create(sine_wave), Write(sig_label))
        self.play(sine_wave.animate.stretch(1.5, 0), run_time=0.5)
        #self.next_slide()

        # 6. Finance
        self.play(Create(line_6), Create(fin_line), Write(fin_label))
        self.play(Wiggle(fin_line), run_time=0.5) 
        self.next_slide()

        # -----------------------------------------
        # CLOSING MESSAGE
        # -----------------------------------------
        final_box = Rectangle(width=10, height=2, color=BLACK, fill_opacity=0.8).to_edge(DOWN)
        final_text = Text(
            "“Many modern workloads are naturally data-parallel\nor can be restructured to be.”",
            font_size=28, color=YELLOW, slant=ITALIC
        ).move_to(final_box)

        self.play(FadeIn(final_box), Write(final_text))
        
        self.next_slide()
        
        # Cleanup
        all_objs = VGroup(
            hub, title, 
            science_grid, science_label, line_1,
            ai_group, ai_label, line_3,
            media_pixels, media_label, line_4,
            sine_wave, sig_label, line_5,
            fin_line, fin_label, line_6,
            final_box, final_text
        )
        self.play(FadeOut(all_objs))
        self.wait(0.5)

    def chapter_4_how(self):
        # -----------------------------------------
        # SETUP: Title
        # -----------------------------------------
        title = Text("4. How can we do it?", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # CONCEPT 1: Data Parallelism
        # -----------------------------------------
        header_data = Text("Data Parallelism", font_size=32, color=YELLOW).next_to(title, DOWN)
        self.play(FadeIn(header_data))

        array_group = VGroup(*[Square(side_length=0.5, color=WHITE) for _ in range(8)]).arrange(RIGHT, buff=0)
        array_group.move_to(ORIGIN)
        
        self.play(Create(array_group))
        
        chunk1 = VGroup(array_group[0], array_group[1])
        chunk2 = VGroup(array_group[2], array_group[3])
        chunk3 = VGroup(array_group[4], array_group[5])
        chunk4 = VGroup(array_group[6], array_group[7])
        
        self.play(
            chunk1.animate.shift(LEFT*1.5 + UP*0.5),
            chunk2.animate.shift(LEFT*0.5 + DOWN*0.5),
            chunk3.animate.shift(RIGHT*0.5 + UP*0.5),
            chunk4.animate.shift(RIGHT*1.5 + DOWN*0.5),
        )
        
        self.play(
            chunk1.animate.set_fill(GREEN, opacity=0.5),
            chunk2.animate.set_fill(GREEN, opacity=0.5),
            chunk3.animate.set_fill(GREEN, opacity=0.5),
            chunk4.animate.set_fill(GREEN, opacity=0.5),
            run_time=0.5
        )
        
        text_data = Text("Same function, different data", font_size=24).next_to(array_group, DOWN, buff=1.5)
        self.play(Write(text_data))
        
        self.next_slide()
        self.play(FadeOut(array_group), FadeOut(header_data), FadeOut(text_data), FadeOut(chunk1), FadeOut(chunk2), FadeOut(chunk3), FadeOut(chunk4))

        # -----------------------------------------
        # CONCEPT 2: Task Parallelism
        # -----------------------------------------
        header_task = Text("Task Parallelism", font_size=32, color=ORANGE).next_to(title, DOWN)
        self.play(FadeIn(header_task))
        
        task1 = Circle(radius=0.4, color=RED).shift(LEFT*2)
        task2 = Triangle(color=BLUE).scale(0.5)
        task3 = Square(side_length=0.8, color=GREEN).shift(RIGHT*2)
        
        tasks = VGroup(task1, task2, task3)
        self.play(Create(tasks))
        
        l1 = Text("UI Thread", font_size=16).next_to(task1, DOWN)
        l2 = Text("Network", font_size=16).next_to(task2, DOWN)
        l3 = Text("Compute", font_size=16).next_to(task3, DOWN)
        labels = VGroup(l1, l2, l3)
        self.play(Write(labels))
        
        self.play(
            Rotate(task1),
            Wiggle(task2),
            ScaleInPlace(task3, 1.2),
            rate_func=there_and_back,
            run_time=1
        )
        
        text_task = Text("Different functions simultaneously", font_size=24).next_to(labels, DOWN, buff=0.5)
        self.play(Write(text_task))

        self.next_slide()
        self.play(FadeOut(tasks), FadeOut(labels), FadeOut(header_task), FadeOut(text_task))

        # -----------------------------------------
        # CONCEPT 3: Pipeline Parallelism
        # -----------------------------------------
        header_pipe = Text("Pipeline Parallelism", font_size=32, color=TEAL).next_to(title, DOWN)
        self.play(FadeIn(header_pipe))
        
        stage1 = Square(color=WHITE).shift(LEFT*3)
        stage2 = Square(color=WHITE)
        stage3 = Square(color=WHITE).shift(RIGHT*3)
        stages = VGroup(stage1, stage2, stage3)
        
        arrow1 = Arrow(stage1.get_right(), stage2.get_left(), buff=0.1)
        arrow2 = Arrow(stage2.get_right(), stage3.get_left(), buff=0.1)
        
        self.play(Create(stages), Create(arrow1), Create(arrow2))
        
        item1 = Dot(color=YELLOW).move_to(stage1)
        item2 = Dot(color=YELLOW).move_to(stage1).shift(LEFT*2) 
        
        self.play(FadeIn(item1), FadeIn(item2))
        
        self.play(
            item1.animate.move_to(stage2),
            item2.animate.move_to(stage1),
            run_time=1
        )
        
        self.play(
            item1.animate.move_to(stage3),
            item2.animate.move_to(stage2),
            run_time=1
        )
        
        text_pipe = Text("Stream processing (Assembly Line)", font_size=24).next_to(stages, DOWN)
        self.play(Write(text_pipe))

        self.next_slide()
        self.play(FadeOut(stages), FadeOut(arrow1), FadeOut(arrow2), FadeOut(item1), FadeOut(item2), FadeOut(header_pipe), FadeOut(text_pipe))

        # -----------------------------------------
        # CONCEPT 4: Reduction
        # -----------------------------------------
        header_red = Text("Reduction Patterns", font_size=32, color=PURPLE).next_to(title, DOWN)
        self.play(FadeIn(header_red))
        
        l1_nodes = VGroup(*[Circle(radius=0.3, color=WHITE).move_to(LEFT*1.5 + RIGHT*i + UP*0.5) for i in range(4)])
        l2_nodes = VGroup(
            Circle(radius=0.3, color=BLUE).move_to(l1_nodes[0].get_center() + RIGHT*0.5 + DOWN*1.5),
            Circle(radius=0.3, color=BLUE).move_to(l1_nodes[2].get_center() + RIGHT*0.5 + DOWN*1.5)
        )
        l3_node = Circle(radius=0.3, color=GREEN).move_to(DOWN*2.5) 
        
        lines = VGroup()
        lines.add(Line(l1_nodes[0].get_bottom(), l2_nodes[0].get_top()))
        lines.add(Line(l1_nodes[1].get_bottom(), l2_nodes[0].get_top()))
        lines.add(Line(l1_nodes[2].get_bottom(), l2_nodes[1].get_top()))
        lines.add(Line(l1_nodes[3].get_bottom(), l2_nodes[1].get_top()))
        lines.add(Line(l2_nodes[0].get_bottom(), l3_node.get_top()))
        lines.add(Line(l2_nodes[1].get_bottom(), l3_node.get_top()))
        
        self.play(Create(l1_nodes))
        self.play(Create(lines), Create(l2_nodes))
        self.play(Create(l3_node))
        
        text_red = Text("Combine results (Map -> Reduce)", font_size=24).next_to(l3_node, RIGHT)
        self.play(Write(text_red))
        
        self.next_slide()
        self.play(FadeOut(l1_nodes), FadeOut(l2_nodes), FadeOut(l3_node), FadeOut(lines), FadeOut(header_red), FadeOut(text_red))

        # -----------------------------------------
        # BRIDGE LINE
        # -----------------------------------------
        final_text = Paragraph(
            "“The architecture you choose should",
            "match the type of parallelism",
            "your problem exposes.”",
            alignment="center",
            font_size=34, color=YELLOW
        )
        self.play(Write(final_text))
        
        self.next_slide()
        self.play(FadeOut(final_text), FadeOut(title))
        self.wait(0.5)

    def chapter_5_hardware(self):
        # -----------------------------------------
        # SETUP: Title
        # -----------------------------------------
        title = Text("5. What hardware do we need?", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # CONCEPT 1: CPU
        # -----------------------------------------
        header_cpu = Text("CPU: Low Latency / Complex Logic", font_size=32, color=BLUE).next_to(title, DOWN)
        
        cpu_group = VGroup()
        for i in range(4):
            core = Square(side_length=1.5, color=BLUE, fill_opacity=0.2)
            if i % 2 == 0:
                inner = Circle(radius=0.4, color=WHITE).move_to(core)
            else:
                inner = Triangle(color=WHITE).scale(0.4).move_to(core)
            
            group = VGroup(core, inner)
            x = (i % 2) * 1.8 - 0.9
            y = (i // 2) * 1.8 - 0.9
            group.move_to([x, y - 0.5, 0])
            cpu_group.add(group)

        self.play(FadeIn(header_cpu), Create(cpu_group))
        
        self.play(
            Rotate(cpu_group[0][1], angle=PI),
            Wiggle(cpu_group[1][1]),
            ScaleInPlace(cpu_group[2][1], 1.2),
            Flash(cpu_group[3], color=WHITE, run_time=1),
        )
        
        desc_cpu = Text("Great for: Mixed workloads, Control flow", font_size=24, color=GREY).next_to(cpu_group, DOWN, buff=0.5)
        self.play(Write(desc_cpu))
        
        self.next_slide()
        self.play(FadeOut(cpu_group), FadeOut(header_cpu), FadeOut(desc_cpu))

        # -----------------------------------------
        # CONCEPT 2: GPU
        # -----------------------------------------
        header_gpu = Text("GPU: High Throughput / Data Parallel", font_size=32, color=GREEN).next_to(title, DOWN)
        
        gpu_group = VGroup()
        for x in range(8):
            for y in range(6):
                core = Square(side_length=0.3, color=GREEN, fill_opacity=0.5)
                core.move_to(np.array([(x-3.5)*0.5, (y-2.5)*0.5 - 0.5, 0]))
                gpu_group.add(core)
                
        self.play(FadeIn(header_gpu), ShowIncreasingSubsets(gpu_group, run_time=1))
        
        self.play(
            gpu_group.animate.set_color(YELLOW),
            run_time=0.5,
            rate_func=there_and_back
        )
        
        desc_gpu = Text("Great for: Massive identical tasks (Pixels, Matrices)", font_size=24, color=GREY).next_to(gpu_group, DOWN, buff=0.5)
        self.play(Write(desc_gpu))
        
        self.next_slide()
        self.play(FadeOut(gpu_group), FadeOut(header_gpu), FadeOut(desc_gpu))

        # -----------------------------------------
        # CONCEPT 3: Heterogeneous
        # -----------------------------------------
        header_het = Text("Heterogeneous (CPU + GPU)", font_size=32, color=PURPLE).next_to(title, DOWN)
        
        cpu_icon = Square(side_length=1.5, color=BLUE, fill_opacity=0.5).move_to(LEFT*3 + DOWN*0.5)
        cpu_lbl = Text("Host (CPU)", font_size=20).next_to(cpu_icon, UP)
        
        gpu_icon = Rectangle(width=3, height=2, color=GREEN, fill_opacity=0.5).move_to(RIGHT*2 + DOWN*0.5)
        gpu_grid = VGroup(*[Square(side_length=0.2, color=WHITE, fill_opacity=0.2).move_to(gpu_icon.get_corner(UL) + RIGHT*(0.25+i*0.3) + DOWN*(0.25+j*0.3)) for i in range(8) for j in range(5)])
        gpu_lbl = Text("Device (GPU)", font_size=20).next_to(gpu_icon, UP)
        
        bus = DoubleArrow(cpu_icon.get_right(), gpu_icon.get_left(), color=WHITE)
        
        self.play(FadeIn(header_het), DrawBorderThenFill(cpu_icon), Write(cpu_lbl))
        self.play(DrawBorderThenFill(gpu_icon), FadeIn(gpu_grid), Write(gpu_lbl))
        self.play(GrowFromCenter(bus))
        
        data_packet = Dot(color=YELLOW).move_to(cpu_icon.get_center())
        self.play(data_packet.animate.move_to(gpu_icon.get_center()), run_time=1)
        self.play(Wiggle(gpu_icon))
        
        self.next_slide()
        self.play(FadeOut(cpu_icon), FadeOut(cpu_lbl), FadeOut(gpu_icon), FadeOut(gpu_grid), FadeOut(gpu_lbl), FadeOut(bus), FadeOut(data_packet), FadeOut(header_het))

        # -----------------------------------------
        # CONCEPT 4: Clusters
        # -----------------------------------------
        header_cluster = Text("Clusters (Distributed)", font_size=32, color=ORANGE).next_to(title, DOWN)
        
        servers = VGroup()
        for i in range(3):
            rack = RoundedRectangle(corner_radius=0.1, height=1.5, width=1.0, color=WHITE)
            lights = VGroup(*[Dot(radius=0.05, color=GREEN).move_to(rack.get_top() + DOWN*(0.2 + k*0.3) + LEFT*0.2) for k in range(3)])
            server = VGroup(rack, lights)
            server.move_to(LEFT*3 + RIGHT*(i*3) + DOWN*0.5)
            servers.add(server)
            
        lines = VGroup(
            Line(servers[0].get_right(), servers[1].get_left(), color=YELLOW),
            Line(servers[1].get_right(), servers[2].get_left(), color=YELLOW)
        )
        
        self.play(FadeIn(header_cluster), Create(servers))
        self.play(ShowPassingFlash(lines.copy().set_color(YELLOW), time_width=0.5), run_time=1.5)
        
        desc_cluster = Text("Critical when one machine isn't enough.", font_size=24, color=GREY).next_to(servers, DOWN, buff=0.5)
        self.play(Write(desc_cluster))

        self.next_slide()
        self.play(FadeOut(servers), FadeOut(lines), FadeOut(header_cluster), FadeOut(desc_cluster))

        # -----------------------------------------
        # FINAL MESSAGE
        # -----------------------------------------
        final_text = Paragraph(
            "“Hardware is not just faster silicon;",
            "it is different execution models.”",
            alignment="center",
            font_size=36, color=YELLOW
        )
        self.play(Write(final_text))
        
        self.next_slide()
        self.play(FadeOut(final_text), FadeOut(title))
        self.wait(0.5)

    def chapter_6_cuda(self):
        # -----------------------------------------
        # SETUP: Title
        # -----------------------------------------
        title = Text("Emergence & History of CUDA", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # PHASE 1: The Graphics Era (The "Pre-History")
        # -----------------------------------------
        header_gfx = Text("Origin: Built for Graphics", font_size=32, color=BLUE).next_to(title, DOWN)
        
        screen_border = Rectangle(width=4, height=3, color=WHITE)
        triangle = Triangle(color=RED, fill_opacity=0.8).scale(0.8)
        
        gpu_chip = Square(side_length=1.5, color=GREEN, fill_opacity=0.5).shift(DOWN*2)
        gpu_label = Text("GPU", font_size=20).move_to(gpu_chip)
        
        wire = Line(gpu_chip.get_top(), screen_border.get_bottom(), color=GREEN)
        
        self.play(FadeIn(header_gfx), Create(screen_border), FadeIn(triangle), DrawBorderThenFill(gpu_chip), Write(gpu_label), Create(wire))
        
        self.play(Rotate(triangle, angle=TAU, run_time=2))
        
        self.next_slide()

        # -----------------------------------------
        # PHASE 2: The Realization (GPGPU)
        # -----------------------------------------
        header_gpgpu = Text("The Insight: It's just math!", font_size=32, color=YELLOW).next_to(title, DOWN)
        
        matrix_nums = VGroup()
        for i in range(3):
            for j in range(3):
                num = Text(str(np.random.randint(0, 9)), font_size=24, color=YELLOW)
                num.move_to(triangle.get_center() + np.array([(i-1)*0.5, (j-1)*0.5, 0]))
                matrix_nums.add(num)
        
        self.play(
            ReplacementTransform(header_gfx, header_gpgpu),
            ReplacementTransform(triangle, matrix_nums)
        )
        
        self.play(Indicate(matrix_nums, color=WHITE))
        
        self.next_slide()
        self.play(FadeOut(screen_border), FadeOut(matrix_nums), FadeOut(wire), FadeOut(header_gpgpu))

        # -----------------------------------------
        # PHASE 3: CUDA (The Bridge)
        # -----------------------------------------
        header_cuda = Text("CUDA: Making it Programmable", font_size=32, color=GREEN).next_to(title, DOWN)
        self.play(FadeIn(header_cuda))
        
        code_bg_left = Rectangle(width=3, height=3.5, color=GREY, fill_opacity=0.2).shift(LEFT*3)
        code_gfx = Text("glBegin();\nglVertex3f();\nTexture();\n// Hacky!", font_size=18, font="Monospace", color=RED).move_to(code_bg_left)
        label_left = Text("Pre-2007 (Graphics API)", font_size=20, color=RED).next_to(code_bg_left, DOWN)
        
        code_bg_right = Rectangle(width=4, height=3.5, color=GREEN, fill_opacity=0.2).shift(RIGHT*3)
        code_cuda = Text("__global__ void\nkernel(float* x) {\n  int i = threadIdx.x;\n  x[i] = ...;\n}", font_size=18, font="Monospace", color=WHITE).move_to(code_bg_right)
        label_right = Text("CUDA (C++ style)", font_size=20, color=GREEN).next_to(code_bg_right, DOWN)
        
        arrow = Arrow(code_bg_left.get_right(), code_bg_right.get_left(), color=WHITE)
        
        self.play(
            FadeIn(code_bg_left), Write(code_gfx), Write(label_left)
        )
        self.next_slide()
        
        self.play(
            GrowArrow(arrow),
            FadeIn(code_bg_right), Write(code_cuda), Write(label_right)
        )
        
        self.next_slide()
        self.play(FadeOut(code_bg_left), FadeOut(code_gfx), FadeOut(label_left), FadeOut(arrow), FadeOut(code_bg_right), FadeOut(code_cuda), FadeOut(label_right), FadeOut(header_cuda))

        # -----------------------------------------
        # PHASE 4: Ecosystem Growth
        # -----------------------------------------
        header_eco = Text("Ecosystem Growth", font_size=32, color=BLUE).next_to(title, DOWN)
        
        gpu_chip.move_to(DOWN*2.5) 
        
        layer_cuda = Rectangle(width=4, height=0.8, color=GREEN, fill_opacity=0.8).next_to(gpu_chip, UP, buff=0.1)
        txt_cuda = Text("CUDA Core", font_size=24, color=BLACK).move_to(layer_cuda)
        
        libs = ["cuBLAS", "cuDNN", "Thrust", "TensorRT"]
        lib_blocks = VGroup()
        for i, lib in enumerate(libs):
            blk = Rectangle(width=1.5, height=0.6, color=BLUE, fill_opacity=0.6)
            x = (i % 2) * 1.6 - 0.8
            y = (i // 2) * 0.7
            blk.move_to(layer_cuda.get_top() + UP*(0.5 + y) + RIGHT*x)
            
            lbl = Text(lib, font_size=18).move_to(blk)
            lib_blocks.add(VGroup(blk, lbl))
            
        app_layer = Ellipse(width=5, height=1, color=YELLOW, fill_opacity=0.3).next_to(lib_blocks, UP, buff=0.2)
        app_txt = Text("Modern AI & Simulation", font_size=24, color=YELLOW).move_to(app_layer)
        
        self.play(FadeIn(header_eco))
        # FadeIn gpu_chip in case it was lost in transitions
        self.play(FadeIn(gpu_chip), Write(gpu_label))
        self.play(DrawBorderThenFill(layer_cuda), Write(txt_cuda))
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN) for b in lib_blocks], lag_ratio=0.2))
        self.play(GrowFromCenter(app_layer), Write(app_txt))
        
        self.next_slide()
        self.play(FadeOut(gpu_chip), FadeOut(gpu_label), FadeOut(layer_cuda), FadeOut(txt_cuda), FadeOut(lib_blocks), FadeOut(app_layer), FadeOut(app_txt), FadeOut(header_eco))

        # -----------------------------------------
        # FINAL MESSAGE
        # -----------------------------------------
        final_text = Paragraph(
            "“CUDA is not the start of parallel computing,",
            "but a major inflection point for",
            "practical developer-friendly GPU computing.”",
            alignment="center",
            font_size=32, color=YELLOW
        )
        self.play(Write(final_text))
        
        self.next_slide()
        self.play(FadeOut(final_text), FadeOut(title))
        self.wait(0.5)

    def chapter_7_perf(self):
        # ---------------------------------------------------------
        # 1. Data Setup
        # ---------------------------------------------------------
        x_start_pow = 9
        x_vals = list(range(x_start_pow, 23)) 

        COL_CPU = YELLOW
        COL_GPU_E2E = TEAL
        COL_GPU_COMP = PINK

        data = {
            "fft": {
                "cpu": [0.016, 0.023, 0.068, 0.066, 0.174, 0.345, 0.810, 1.471, 3.388, 8.543, 15.662, 59.188, 125.848, 265.588],
                "gpu_e2e": [0.179, 0.182, 0.389, 0.265, 0.433, 0.717, 1.010, 1.855, 3.852, 7.042, 14.154, 43.264, 62.733, 174.906],
                "gpu_comp": [0.066, 0.079, 0.093, 0.071, 0.087, 0.128, 0.079, 0.088, 0.098, 0.143, 0.215, 0.407, 0.701, 1.378]
            },
            "lfilter": {
                "cpu": [0.196, 0.239, 0.571, 0.672, 1.296, 2.097, 4.726, 6.924, 15.916, 32.764, 65.532, 130.715, 271.957, 451.610],
                "gpu_e2e": [0.612, 0.605, 0.949, 0.714, 0.855, 1.121, 1.463, 2.515, 5.116, 9.375, 17.713, 50.246, 70.972, 147.096],
                "gpu_comp": [0.459, 0.446, 0.611, 0.463, 0.485, 0.472, 0.544, 0.842, 1.285, 2.251, 4.176, 8.139, 15.702, 19.481]
            },
            "fftconv": {
                "cpu": [0.105, 0.140, 0.324, 0.226, 0.315, 0.597, 1.079, 2.269, 4.420, 9.219, 21.636, 70.197, 102.753, 259.647],
                "gpu_e2e": [0.465, 0.588, 0.517, 0.562, 0.766, 1.025, 1.299, 2.117, 3.831, 7.418, 14.624, 44.084, 62.361, 121.661],
                "gpu_comp": [0.330, 0.283, 0.523, 0.313, 0.360, 0.365, 0.357, 0.404, 0.402, 0.389, 0.568, 1.271, 2.394, 3.894]
            }
        }

        # ---------------------------------------------------------
        # 2. Axes Setup
        # ---------------------------------------------------------
        y_min_exp = -2
        y_max_exp = 3.5
        
        axes = Axes(
            x_range=[9, 22.5, 1],    
            y_range=[y_min_exp, y_max_exp, 1], 
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True, "color": GREY, "stroke_width": 4},
            y_axis_config={"include_numbers": False}, 
            x_axis_config={"include_numbers": False},
        ).to_edge(DOWN).shift(LEFT * 0.5)

        x_labels = VGroup()
        for i in range(9, 23):
            label = MathTex(r"2^{" + str(i) + r"}")
            label.scale(0.5)
            label.next_to(axes.c2p(i, y_min_exp), DOWN, buff=0.2)
            x_labels.add(label)

        y_labels = VGroup()
        for i in range(int(y_min_exp), 4): 
            label = MathTex(r"10^{" + str(i) + r"}")
            label.scale(0.5)
            label.next_to(axes.c2p(9, i), LEFT, buff=0.2)
            y_labels.add(label)

        x_title = Tex(r"Input Size ($N$)").scale(0.5).next_to(axes.x_axis, DOWN, buff=0.08)
        y_title = Tex("Time (ms) - Log Scale").scale(0.5).next_to(axes.y_axis, UP, buff=0.02).shift(RIGHT*1.5)

        grid_lines = VGroup()
        for x in x_vals:
            line = DashedLine(
                start=axes.c2p(x, y_min_exp),
                end=axes.c2p(x, y_max_exp),
                color=GREY, stroke_width=2, stroke_opacity=0.3, dash_length=0.1
            )
            grid_lines.add(line)

        axes_group = VGroup(axes, grid_lines, x_labels, y_labels, x_title, y_title)

        # ---------------------------------------------------------
        # 3. Legend
        # ---------------------------------------------------------
        main_title = Tex("Performance: FFT").scale(1.2).to_edge(UP).shift(LEFT * 2)

        legend_items = [
            (COL_CPU, "CPU Time"),
            (COL_GPU_E2E, "GPU E2E Time"),
            (COL_GPU_COMP, "GPU Compute Time")
        ]
        
        legend = VGroup()
        for color, text in legend_items:
            item = VGroup(
                Line(color=color, stroke_width=6).set_length(0.6),
                Tex(text, color=color).scale(0.6)
            ).arrange(RIGHT, buff=0.1)
            legend.add(item)
        
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(UP + RIGHT, buff=0.5)
        legend_bg = SurroundingRectangle(legend, color=WHITE, fill_color=BLACK, fill_opacity=0.85, buff=0.1)
        legend_group = VGroup(legend_bg, legend)

        # ---------------------------------------------------------
        # 4. Helpers
        # ---------------------------------------------------------
        def get_intersection_point(cpu_vals, gpu_vals):
            for i in range(len(cpu_vals) - 1):
                c1, c2 = np.log10(cpu_vals[i]), np.log10(cpu_vals[i+1])
                g1, g2 = np.log10(gpu_vals[i]), np.log10(gpu_vals[i+1])
                diff1 = c1 - g1
                diff2 = c2 - g2
                if np.sign(diff1) != np.sign(diff2):
                    t = abs(diff1) / (abs(diff1) + abs(diff2))
                    x_idx = x_start_pow + i + t
                    y_log = c1 * (1-t) + c2 * t
                    return axes.c2p(x_idx, y_log)
            return None

        def get_scene_elements(key):
            curves = VGroup()
            end_labels = VGroup()
            metrics = [("cpu", COL_CPU), ("gpu_e2e", COL_GPU_E2E), ("gpu_comp", COL_GPU_COMP)]
            for metric, color in metrics:
                raw_y_values = data[key][metric]
                points = [axes.c2p(x, np.log10(y)) for x, y in zip(x_vals, raw_y_values)]
                line = VMobject().set_points_smoothly(points)
                line.set_color(color).set_stroke(width=6)
                curves.add(line)
                label_text = f"{raw_y_values[-1]:.1f} ms"
                lbl = Tex(label_text, color=color).scale(0.5)
                lbl.next_to(points[-1], RIGHT, buff=0.15)
                end_labels.add(lbl)
            
            crossover_pt = get_intersection_point(data[key]["cpu"], data[key]["gpu_e2e"])
            crossover_group = VGroup()
            if crossover_pt is not None:
                dot = Dot(point=crossover_pt, color=WHITE, radius=0.08)
                glow = Dot(point=crossover_pt, color=COL_GPU_E2E, radius=0.15).set_opacity(0.5)
                text = Tex("GPU Wins", color=WHITE).scale(0.5)
                text.next_to(dot, UP + LEFT, buff=0.1)
                arrow = Arrow(start=text.get_bottom(), end=dot.get_top(), color=WHITE, buff=0.05, stroke_width=2, tip_length=0.15)
                crossover_group.add(glow, dot, text, arrow)
            return curves, end_labels, crossover_group

        # ---------------------------------------------------------
        # 5. Animation Sequence
        # ---------------------------------------------------------
        keys = ["fft", "lfilter", "fftconv"]
        titles = ["FFT", "LFilter", "FFTConv"]
        
        current_curves = VGroup()
        current_labels = VGroup()
        current_crossover = VGroup()
        
        self.play(Create(axes_group), Write(main_title), FadeIn(legend_group), run_time=1.5)
        
        for i, key in enumerate(keys):
            new_curves, new_labels, new_crossover = get_scene_elements(key)
            new_title_text = f"Performance: {titles[i]}"
            
            if i == 0:
                current_curves = new_curves
                current_labels = new_labels
                current_crossover = new_crossover
                
                self.play(Create(current_curves), Write(current_labels), run_time=2)
                if len(current_crossover) > 0:
                    self.play(FadeIn(current_crossover))
                
                self.next_slide()

            else:
                target_title = Tex(new_title_text).scale(1).move_to(main_title)
                
                self.play(
                    Transform(current_curves, new_curves),
                    Transform(current_labels, new_labels),
                    Transform(main_title, target_title),
                    FadeOut(current_crossover), 
                    run_time=2
                )
                
                current_crossover = new_crossover
                if len(current_crossover) > 0:
                    self.play(FadeIn(current_crossover))
                
                self.next_slide()
        
        self.wait(1)