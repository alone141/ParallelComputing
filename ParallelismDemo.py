from manim import *
from manim_slides import Slide

class ParallelismDemo(Slide):
    def construct(self):
        # Title that stays at the top
        title = Text("CUDA Kernel Configurations", font_size=48).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # We will cycle through these configurations
        # Format: (blocks, threads, code_string)
        configs = [
            (1, 1, "printHelloGPU<<<1, 1>>>();"),
            (1, 5, "printHelloGPU<<<1, 5>>>();"),
            (5, 1, "printHelloGPU<<<5, 1>>>();"),
            (5, 5, "printHelloGPU<<<5, 5>>>();"),
        ]

        # Keep track of previous objects to transform/fade out
        prev_group = VGroup()
        prev_code = VGroup()
        prev_stats = VGroup()

        for blocks, threads, code_str in configs:
            # 1. Create the Code Snippet
            # CORRECTED: Used 'code_string' instead of 'code'
            code = Code(
                code_string=code_str,
                tab_width=4,
                background="window",
                language="cpp",
            ).next_to(title, DOWN, buff=0.5)

            # 2. Create the Visual Representation (Blocks and Threads)
            gpu_grid = VGroup()
            
            for b in range(blocks):
                # Create a Block (Container)
                block_box = Square(side_length=1.5, color=BLUE, fill_opacity=0.1)
                
                # Create Threads inside the Block
                block_threads = VGroup()
                for t in range(threads):
                    thread_dot = Dot(color=GREEN, radius=0.1)
                    block_threads.add(thread_dot)
                
                # Arrange threads inside the block
                if threads > 1:
                    # Arrange in a grid if many, or line if few. 
                    block_threads.arrange(RIGHT, buff=0.15)
                    # If threads don't fit in the box, wrap them (simple logic)
                    if block_threads.width > 1.2:
                         block_threads.arrange_in_grid(cols=3, buff=0.15)

                
                # Group box and threads
                block_group = VGroup(block_box, block_threads)
                
                # Add label for the block ID
                block_label = Text(f"Block {b}", font_size=16, color=BLUE_B).next_to(block_box, UP, buff=0.1)
                block_group.add(block_label)
                
                gpu_grid.add(block_group)

            # Arrange the blocks horizontally
            gpu_grid.arrange(RIGHT, buff=0.5)
            
            # If the grid is too wide (case 5,5), scale it down
            if gpu_grid.width > 12:
                gpu_grid.scale_to_fit_width(12)
            
            gpu_grid.move_to(ORIGIN)

            # 3. Create Explanatory Text (The Math)
            total_threads = blocks * threads
            explanation = VGroup(
                Text(f"Grid Configuration: {blocks} Block(s)", font_size=24, color=BLUE),
                Text(f"Block Configuration: {threads} Thread(s) per Block", font_size=24, color=GREEN),
                Text(f"Total Execution: {blocks} x {threads} = {total_threads} Run(s)", font_size=30, color=WHITE)
            ).arrange(DOWN, buff=0.2).next_to(gpu_grid, DOWN, buff=1.0)

            # 4. Animation Sequence
            
            # Transition Code
            if len(prev_code) > 0:
                # Fade out old code, fade in new (Code objects can be tricky to Transform directly)
                self.play(FadeOut(prev_code), run_time=0.3)
                self.play(FadeIn(code), run_time=0.3)
            else:
                self.play(FadeIn(code))
            
            prev_code = code

            # Transition Visuals
            self.play(
                FadeOut(prev_group),
                FadeOut(prev_stats),
                run_time=0.5
            )
            
            self.play(
                Create(gpu_grid),
                run_time=1.5
            )
            
            self.play(Write(explanation))

            # Store current references for next loop
            prev_group = gpu_grid
            prev_stats = explanation

            # Pause for slide
            self.next_slide()

        # Clear screen at end
        self.play(FadeOut(prev_group), FadeOut(prev_stats), FadeOut(prev_code), FadeOut(title))