from manim import *
from manim_slides import Slide
import numpy as np

class CudaHistorySlides(Slide):
    def construct(self):
        # -----------------------------------------
        # SETUP: Title
        # -----------------------------------------
        title = Text("Emergence & History of CUDA", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # PHASE 1: The Graphics Era (The "Pre-History")
        # -----------------------------------------
        # Visual: A GPU rendering a triangle (Graphics)
        
        header_gfx = Text("Origin: Built for Graphics", font_size=32, color=BLUE).next_to(title, DOWN)
        
        # A "Screen" showing a rotating 3D triangle
        screen_border = Rectangle(width=4, height=3, color=WHITE)
        triangle = Triangle(color=RED, fill_opacity=0.8).scale(0.8)
        
        gpu_chip = Square(side_length=1.5, color=GREEN, fill_opacity=0.5).shift(DOWN*2)
        gpu_label = Text("GPU", font_size=20).move_to(gpu_chip)
        
        # Connection
        wire = Line(gpu_chip.get_top(), screen_border.get_bottom(), color=GREEN)
        
        group_gfx = VGroup(screen_border, triangle, gpu_chip, gpu_label, wire)
        
        self.play(FadeIn(header_gfx), Create(screen_border), FadeIn(triangle), DrawBorderThenFill(gpu_chip), Write(gpu_label), Create(wire))
        
        # Animate triangle spinning
        self.play(Rotate(triangle, angle=TAU, run_time=2))
        
        self.next_slide()

        # -----------------------------------------
        # PHASE 2: The Realization (GPGPU)
        # -----------------------------------------
        # Visual: The Triangle transforms into a Matrix of Numbers
        
        header_gpgpu = Text("The Insight: It's just math!", font_size=32, color=YELLOW).next_to(title, DOWN)
        
        # Create a matrix grid
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
        
        # "Pixels are just numbers"
        self.play(Indicate(matrix_nums, color=WHITE))
        
        self.next_slide()
        self.play(FadeOut(screen_border), FadeOut(matrix_nums), FadeOut(wire), FadeOut(header_gpgpu))

        # -----------------------------------------
        # PHASE 3: CUDA (The Bridge)
        # -----------------------------------------
        # Visual: Turning messy "Shader Hacks" into "C++ Code"
        
        header_cuda = Text("CUDA: Making it Programmable", font_size=32, color=GREEN).next_to(title, DOWN)
        self.play(FadeIn(header_cuda))
        
        # Left: Messy Shader Code
        code_bg_left = Rectangle(width=3, height=3.5, color=GREY, fill_opacity=0.2).shift(LEFT*3)
        code_gfx = Text("glBegin();\nglVertex3f();\nTexture();\n// Hacky!", font_size=18, font="Monospace", color=RED).move_to(code_bg_left)
        label_left = Text("Pre-2007 (Graphics API)", font_size=20, color=RED).next_to(code_bg_left, DOWN)
        
        # Right: Clean C++ Code
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
        # Visual: Building a structure
        
        header_eco = Text("Ecosystem Growth", font_size=32, color=BLUE).next_to(title, DOWN)
        
        # Base: Hardware (Already have gpu_chip)
        gpu_chip.move_to(DOWN*2.5) # Ensure it's at bottom
        #self.add(gpu_chip, gpu_label) # Make sure it's visible
        
        # Layer 1: CUDA Driver
        layer_cuda = Rectangle(width=4, height=0.8, color=GREEN, fill_opacity=0.8).next_to(gpu_chip, UP, buff=0.1)
        txt_cuda = Text("CUDA Core", font_size=24, color=BLACK).move_to(layer_cuda)
        
        # Layer 2: Libraries (Blocks piling up)
        libs = ["cuBLAS", "cuDNN", "Thrust", "TensorRT"]
        lib_blocks = VGroup()
        for i, lib in enumerate(libs):
            blk = Rectangle(width=1.5, height=0.6, color=BLUE, fill_opacity=0.6)
            # Position in a 2x2 pile on top
            x = (i % 2) * 1.6 - 0.8
            y = (i // 2) * 0.7
            blk.move_to(layer_cuda.get_top() + UP*(0.5 + y) + RIGHT*x)
            
            lbl = Text(lib, font_size=18).move_to(blk)
            lib_blocks.add(VGroup(blk, lbl))
            
        # Layer 3: Apps (Top)
        app_layer = Ellipse(width=5, height=1, color=YELLOW, fill_opacity=0.3).next_to(lib_blocks, UP, buff=0.2)
        app_txt = Text("Modern AI & Simulation", font_size=24, color=YELLOW).move_to(app_layer)
        
        self.play(FadeIn(header_eco))
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