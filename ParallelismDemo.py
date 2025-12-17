from manim import *
from manim_slides import Slide  # 1. Import Slide

class ParallelismDemo(Slide):   # 2. Inherit from Slide instead of Scene
    def construct(self):
        # --- 1. SETUP LAYOUT ---
        separator = Line(UP * 3.5, DOWN * 3.5, color=GREY)
        self.add(separator)

        # Titles
        serial_title = Text("Single Core", font_size=36, color=BLUE).to_edge(UP).shift(LEFT * 3.5)
        parallel_title = Text("Multi Core (4x)", font_size=36, color=GREEN).to_edge(UP).shift(RIGHT * 3.5)
        
        self.add(serial_title, parallel_title)

        # --- 2. CREATE CORES ---
        def create_core(label, position, color=BLUE):
            square = Square(side_length=1.0, color=color, fill_opacity=0.2)
            lbl = Text(label, font_size=16).move_to(square)
            return VGroup(square, lbl).move_to(position)

        serial_core = create_core("CPU 1", LEFT * 3.5)
        
        p_cores = VGroup()
        for i in range(4):
            core = create_core(f"C{i+1}", RIGHT * (1.5 + i * 1.3), color=GREEN)
            p_cores.add(core)
        p_cores.move_to(RIGHT * 3.5)

        self.add(serial_core, p_cores)

        # --- 3. CREATE TASKS ---
        total_tasks = 8
        
        def create_task_stack(location, color=YELLOW):
            stack = VGroup(*[
                Rectangle(height=0.25, width=0.8, fill_color=color, fill_opacity=1, stroke_color=WHITE, stroke_width=2)
                for _ in range(total_tasks)
            ])
            stack.arrange(DOWN, buff=0.05)
            stack.next_to(location, DOWN, buff=1.0)
            return stack

        serial_tasks = create_task_stack(serial_core, color=BLUE_A)
        parallel_tasks = create_task_stack(p_cores, color=GREEN_A)

        self.add(serial_tasks, parallel_tasks)

        # --- SLIDE BREAK 1 ---
        # The presentation will pause here. 
        # You can explain the setup (Single vs Multi core) before starting the race.
        self.next_slide() 

        # --- 5. ANIMATION LOOP ---
        batch_size = 4
        num_batches = total_tasks // batch_size
        process_duration = 0.5 
        
        for batch in range(num_batches):
            anims_move_in = []
            anims_process = []
            anims_move_out = []
            
            # A. SETUP PARALLEL MOVES
            current_p_tasks = parallel_tasks[0:4] 
            for i, task in enumerate(current_p_tasks):
                target = p_cores[i]
                anims_move_in.append(task.animate.move_to(target.get_center()))
            
            # B. SETUP SERIAL MOVE
            current_s_task = serial_tasks[0] 
            anims_move_in.append(current_s_task.animate.move_to(serial_core.get_center()))
            
            # PLAY MOVE IN
            self.play(*anims_move_in, run_time=0.8)
            
            # C. PROCESSING
            for t in current_p_tasks:
                anims_process.append(t.animate.set_color(RED))
            anims_process.append(current_s_task.animate.set_color(RED))
            
            self.play(*anims_process, run_time=process_duration)
            
            # D. MOVE OUT
            p_done_pos = p_cores.get_top() + UP * 1.5
            for t in current_p_tasks:
                anims_move_out.append(FadeOut(t, target_position=p_done_pos, scale=0.5))
            
            s_done_pos = serial_core.get_top() + UP * 1.5
            anims_move_out.append(FadeOut(current_s_task, target_position=s_done_pos, scale=0.5))
            
            self.play(*anims_move_out, run_time=0.6)
            
            # CLEANUP & SHIFT
            parallel_tasks.remove(*current_p_tasks)
            serial_tasks.remove(current_s_task)
            
            if len(serial_tasks) > 0:
                self.play(
                    serial_tasks.animate.shift(UP * 0.3), 
                    parallel_tasks.animate.shift(UP * 1.2), 
                    run_time=0.2
                )

        # --- SLIDE BREAK 2 ---
        # Pause again before revealing the final "Remaining" count
        self.next_slide()

        # --- 6. FINAL COMPARISON ---
        leftover_count = len(serial_tasks)
        fail_text = Text(f"{leftover_count} Tasks Remaining...", color=RED, font_size=24).next_to(serial_tasks, DOWN)
        self.play(Write(fail_text))
        
        # End slide
        self.next_slide()