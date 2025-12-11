from manim import *
from manim_slides import Slide # pip install manim-slides

class PerformanceRecommendations(Slide):
    def construct(self):
        # ---------------------------------------------------------
        # 1. Data Setup (Same as before)
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
        # 5. Animation Sequence (Manim Slides)
        # ---------------------------------------------------------
        
        # Initial Setup
        
        #self.play(Write(main_title), FadeIn(legend_group))
        
        keys = ["fft", "lfilter", "fftconv"]
        titles = ["FFT", "LFilter", "FFTConv"]
        
        current_curves = VGroup()
        current_labels = VGroup()
        current_crossover = VGroup()
        self.play(Create(axes_group),Write(main_title), FadeIn(legend_group),Create(current_curves), Write(current_labels), run_time=2.0)
        for i, key in enumerate(keys):
            new_curves, new_labels, new_crossover = get_scene_elements(key)
            new_title_text = f"Performance: {titles[i]}"
            
            if i == 0:
                # --- SLIDE 1 ---
                current_curves = new_curves
                current_labels = new_labels
                current_crossover = new_crossover
                
                self.play(Create(current_curves), Write(current_labels), run_time=2)
                if len(current_crossover) > 0:
                    self.play(FadeIn(current_crossover))
                
                # Marks the end of Slide 1. Presentation will pause here.
                self.next_slide()

            else:
                # --- SLIDE 2, 3... ---
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
                
                # Marks the end of Slide 2, 3...
                self.next_slide()
        
        # Keep the last slide on screen briefly before closing
        self.wait(1)