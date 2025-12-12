from manim import *
from manim_slides import Slide
import numpy as np

class ApplicationsSlides(Slide):
    def construct(self):
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
        # We will place them in a circle around the hub
        # Positions:
        # 1. Science (Top Left)
        # 2. Data (Top)
        # 3. AI (Top Right)
        # 4. Media (Bottom Left)
        # 5. Signal (Bottom)
        # 6. Finance (Bottom Right)
        
        radius = 3.0
        angles = [150, 90, 30, 210, 270, 330] # Degrees
        positions = [
            radius * np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0]) 
            for a in angles
        ]

        # --- 1. Scientific Computing (Mesh/Grid) ---
        # Visual: A 3x3 grid where cells change color
        science_pos = positions[0]
        science_grid = VGroup()
        for i in range(3):
            for j in range(3):
                sq = Square(side_length=0.4, color=BLUE, fill_opacity=0.5)
                sq.move_to(science_pos + np.array([(i-1)*0.4, (j-1)*0.4, 0]))
                science_grid.add(sq)
        science_label = Text("Scientific\n(CFD/PDEs)", font_size=18).next_to(science_grid, UP)
        line_1 = Line(hub.get_left(), science_grid.get_bottom(), color=GREY).set_opacity(0.5)

        # --- 2. Data Processing (Bar Chart) ---
        # Visual: 3 bars growing
        #data_pos = positions[1]
        #data_bars = VGroup()
        #for i in range(4):
        #    bar = Rectangle(width=0.3, height=1.0 + i*0.2, color=YELLOW, fill_opacity=0.8)
        #    bar.align_to(data_pos + DOWN*0.5, DOWN)
        #    bar.shift(RIGHT * (i-1.5) * 0.4)
        #    data_bars.add(bar)
        #data_label = Text("Big Data", font_size=18).next_to(data_bars, UP)
        #line_2 = Line(hub.get_top(), data_bars.get_bottom(), color=GREY).set_opacity(0.5)

        # --- 3. AI/ML (Neural Net) ---
        # Visual: Nodes connected by lines
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
        # Visual: A pixelated square
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
        # Visual: A sine wave
        sig_pos = positions[4]
        # We draw a small sine wave
        sine_wave = FunctionGraph(
            lambda t: 0.5 * np.sin(3*t),
            x_range=[-1, 1],
            color=TEAL
        ).move_to(sig_pos)
        sig_label = Text("Signal Proc.\n(Radar/FFT)", font_size=18).next_to(sine_wave, DOWN)
        line_5 = Line(hub.get_bottom(), sine_wave.get_top(), color=GREY).set_opacity(0.5)

        # --- 6. Finance (Stock Chart) ---
        # Visual: Jagged line going up
        fin_pos = positions[5]
        fin_line = VMobject().set_color(GREEN)
        pts = [
            [-1, -0.5, 0], [-0.5, -0.2, 0], [0, -0.6, 0], 
            [0.5, 0.2, 0], [1, 0.8, 0]
        ]
        # Adjust points relative to fin_pos
        real_pts = [np.array(p) * 0.8 + fin_pos for p in pts]
        fin_line.set_points_as_corners(real_pts)
        fin_label = Text("Finance", font_size=18).next_to(fin_line, DOWN)
        line_6 = Line(hub.get_right(), fin_line.get_top(), color=GREY).set_opacity(0.5)

        # -----------------------------------------
        # ANIMATION SEQUENCE
        # -----------------------------------------
        
        # 1. Science
        self.play(Create(line_1), FadeIn(science_grid), Write(science_label))
        self.play(science_grid.animate.set_color(RED), run_time=0.5) # Heat simulation
        self.next_slide()

        # 2. Data
        #self.play(Create(line_2), FadeIn(data_bars), Write(data_label))
        #self.play(data_bars[0].animate.stretch_to_fit_height(1.5), run_time=0.5) # Dynamic data
        self.next_slide()

        # 3. AI
        self.play(Create(line_3), Create(ai_group), Write(ai_label))
        self.play(ShowPassingFlash(edges.copy().set_color(WHITE), time_width=0.5), run_time=1)
        self.next_slide()

        # 4. Media
        self.play(Create(line_4), FadeIn(media_pixels), Write(media_label))
        self.play(Rotate(media_pixels, PI/2), run_time=0.5) # Image rotation
        self.next_slide()

        # 5. Signal
        self.play(Create(line_5), Create(sine_wave), Write(sig_label))
        self.play(sine_wave.animate.stretch(1.5, 0), run_time=0.5) # Wave stretching
        self.next_slide()

        # 6. Finance
        self.play(Create(line_6), Create(fin_line), Write(fin_label))
        self.play(Wiggle(fin_line), run_time=0.5) # Market volatility
        self.next_slide()

        # -----------------------------------------
        # CLOSING MESSAGE
        # -----------------------------------------
        
        # Fade out hub to make room or just overlay
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