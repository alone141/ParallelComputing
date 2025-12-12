from manim import *
from manim_slides import Slide
import numpy as np

class HardwareArchitectures(Slide):
    def construct(self):
        # -----------------------------------------
        # SETUP: Title
        # -----------------------------------------
        title = Text("5. What hardware do we need?", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # CONCEPT 1: CPU (The Mastermind)
        # -----------------------------------------
        header_cpu = Text("CPU: Low Latency / Complex Logic", font_size=32, color=BLUE).next_to(title, DOWN)
        
        # Create 4 "Cores"
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
        # CONCEPT 2: GPU (The Swarm)
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
        # CONCEPT 3: Heterogeneous (The Team)
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
        # CONCEPT 4: Clusters (Scaling Out)
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
        
        # FIX: Apply color to the copy of the object, not as a keyword arg
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