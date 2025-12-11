from manim import *
from manim_slides import Slide

class PC(Slide):
    def construct(self):
        # --- SLIDE 1: Introduction & Definition ---
        self.intro_concept()
        
        # This command creates a "break" in the presentation.
        # In the PPTX, the video will pause here or move to the next slide.
        self.next_slide() 

        self.clear()

        # --- SLIDE 2: Scales of Parallelism ---
        self.show_scales()
        self.next_slide()

        self.clear()

        # --- SLIDE 3: The Anchor Sentence ---
        self.anchor_sentence()
        self.next_slide()

    def intro_concept(self):
        # Title
        title = Text("What is Parallel Computing?", font_size=48).to_edge(UP)
        self.play(Write(title))
        self.next_slide()
        # --- SERIAL REPRESENTATION ---
        serial_label = Text("Serial Processing", font_size=24, color=RED).shift(UP*1.5 + LEFT*3)
        task_bar = Rectangle(width=4, height=0.5, color=WHITE, fill_opacity=0.1).next_to(serial_label, DOWN)
        worker_dot = Dot(color=RED).move_to(task_bar.get_left())
        
        self.play(FadeIn(serial_label), Create(task_bar), FadeIn(worker_dot))
        self.next_slide()
        self.play(worker_dot.animate.move_to(task_bar.get_right()), run_time=2, rate_func=linear)
        self.play(worker_dot.animate.move_to(task_bar.get_left()), run_time=2, rate_func=linear)
        self.next_slide()
        # --- PARALLEL REPRESENTATION ---
        parallel_label = Text("Parallel Computing", font_size=24, color=GREEN).shift(UP*1.5 + RIGHT*3)
        chunk_group = VGroup(*[
            Rectangle(width=1, height=0.5, color=GREEN, fill_opacity=0.3) 
            for _ in range(4)
        ]).arrange(RIGHT, buff=0).next_to(parallel_label, DOWN)
        workers = VGroup(*[Dot(color=GREEN).move_to(chunk.get_left()) for chunk in chunk_group])

        self.play(FadeIn(parallel_label), Create(chunk_group), FadeIn(workers))
        self.play(
            *[worker.animate.move_to(chunk.get_right()) for worker, chunk in zip(workers, chunk_group)],
            run_time=0.5, rate_func=linear
        )
        
        def_text = Text("Doing multiple parts of the work\nat the same time.", font_size=32).next_to(chunk_group, DOWN, buff=1)
        self.play(Write(def_text))

    def show_scales(self):
        scale_title = Text("Scales of Parallelism", font_size=40).to_edge(UP)
        self.play(FadeIn(scale_title))

        # 1. SIMD
        core_box = Square(side_length=2, color=BLUE)
        core_label = Text("CPU Core", font_size=20).next_to(core_box, UP)
        vector_data = VGroup(*[Square(side_length=0.4, fill_opacity=0.5, fill_color=YELLOW) for _ in range(4)]).arrange(RIGHT, buff=0.1).move_to(core_box.get_center())
        simd_text = Text("SIMD / Vectorization", font_size=24, color=YELLOW).next_to(core_box, DOWN)

        self.play(Create(core_box), Write(core_label), FadeIn(vector_data), Write(simd_text))
        self.play(vector_data.animate.set_color(GREEN), run_time=0.5)
        self.wait(0.5)

        # 2. Multicore
        cpu_package = Square(side_length=4.5, color=GREY)
        cores_group = VGroup(core_box, core_box.copy(), core_box.copy(), core_box.copy()).arrange_in_grid(2, 2, buff=0.2).move_to(cpu_package.get_center())
        multicore_text = Text("Multicore (Threads)", font_size=24, color=BLUE).next_to(cpu_package, DOWN)

        self.play(
            FadeOut(simd_text), FadeOut(core_label), FadeOut(vector_data),
            Transform(core_box, cores_group[0])
        )
        self.play(Create(cpu_package), FadeIn(cores_group[1:]), Write(multicore_text))
        self.wait(0.5)

        # 3. GPU
        gpu_package = Rectangle(width=6, height=4, color=GREEN)
        gpu_grid = VGroup(*[
            Square(side_length=0.2, fill_opacity=0.6, fill_color=GREEN, stroke_width=1) 
            for _ in range(100)
        ]).arrange_in_grid(10, 10, buff=0.05).move_to(gpu_package.get_center())
        gpu_text = Text("Many GPU Cores", font_size=24, color=GREEN).next_to(gpu_package, DOWN)

        self.play(
            FadeOut(multicore_text),
            ReplacementTransform(cpu_package, gpu_package),
            ReplacementTransform(cores_group, gpu_grid),
            Write(gpu_text)
        )
        self.wait(0.5)

        # 4. Distributed
        node1 = VGroup(gpu_package, gpu_grid).copy()
        nodes = VGroup(node1, node1.copy(), node1.copy()).arrange(RIGHT, buff=1).scale(0.4)
        connections = VGroup(Line(nodes[0].get_right(), nodes[1].get_left()), Line(nodes[1].get_right(), nodes[2].get_left()))
        dist_text = Text("Distributed (Data Center)", font_size=24, color=PURPLE).next_to(nodes, DOWN)

        self.play(
            FadeOut(gpu_text),
            ReplacementTransform(VGroup(gpu_package, gpu_grid), nodes[0]),
            FadeIn(nodes[1:]), Create(connections), Write(dist_text)
        )

    def anchor_sentence(self):
        quote = Text(
            "“Parallel computing is a strategy to reduce\n"
            "time-to-solution by increasing simultaneous work,\n"
            "from a single chip to a data center.”",
            font_size=34,
            t2c={"time-to-solution": YELLOW, "simultaneous work": GREEN, "single chip": BLUE, "data center": PURPLE}
        )
        self.play(Write(quote, run_time=3))