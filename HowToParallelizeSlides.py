from manim import *
from manim_slides import Slide
import numpy as np

class HowToParallelizeSlides(Slide):
    def construct(self):
        # -----------------------------------------
        # SETUP: Title
        # -----------------------------------------
        title = Text("4. How can we do it?", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # CONCEPT 1: Data Parallelism
        # -----------------------------------------
        # Metaphor: Cutting a long loaf of bread (Array) and eating slices simultaneously.
        
        header_data = Text("Data Parallelism", font_size=32, color=YELLOW).next_to(title, DOWN)
        self.play(FadeIn(header_data))

        # A long array
        array_group = VGroup(*[Square(side_length=0.5, color=WHITE) for _ in range(8)]).arrange(RIGHT, buff=0)
        array_group.move_to(ORIGIN)
        
        self.play(Create(array_group))
        
        # Split it into 4 chunks (simulating 4 cores)
        # We physically move them apart
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
        
        # Process them (Change color)
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
        # Metaphor: Different workers doing different jobs (Shape sorting)
        
        header_task = Text("Task Parallelism", font_size=32, color=ORANGE).next_to(title, DOWN)
        self.play(FadeIn(header_task))
        
        # 3 Distinct Tasks
        task1 = Circle(radius=0.4, color=RED).shift(LEFT*2)
        task2 = Triangle(color=BLUE).scale(0.5)
        task3 = Square(side_length=0.8, color=GREEN).shift(RIGHT*2)
        
        tasks = VGroup(task1, task2, task3)
        self.play(Create(tasks))
        
        # Labels
        l1 = Text("UI Thread", font_size=16).next_to(task1, DOWN)
        l2 = Text("Network", font_size=16).next_to(task2, DOWN)
        l3 = Text("Compute", font_size=16).next_to(task3, DOWN)
        labels = VGroup(l1, l2, l3)
        self.play(Write(labels))
        
        # Animate simultaneous activity
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
        # Metaphor: Assembly Line
        
        header_pipe = Text("Pipeline Parallelism", font_size=32, color=TEAL).next_to(title, DOWN)
        self.play(FadeIn(header_pipe))
        
        # 3 Stages
        stage1 = Square(color=WHITE).shift(LEFT*3)
        stage2 = Square(color=WHITE)
        stage3 = Square(color=WHITE).shift(RIGHT*3)
        stages = VGroup(stage1, stage2, stage3)
        
        arrow1 = Arrow(stage1.get_right(), stage2.get_left(), buff=0.1)
        arrow2 = Arrow(stage2.get_right(), stage3.get_left(), buff=0.1)
        
        self.play(Create(stages), Create(arrow1), Create(arrow2))
        
        # Data items flowing
        # Item A moves 1->2 while Item B moves Start->1
        item1 = Dot(color=YELLOW).move_to(stage1)
        item2 = Dot(color=YELLOW).move_to(stage1).shift(LEFT*2) # Waiting
        
        self.play(FadeIn(item1), FadeIn(item2))
        
        # Step 1: Item 1 moves to Stage 2, Item 2 moves to Stage 1
        self.play(
            item1.animate.move_to(stage2),
            item2.animate.move_to(stage1),
            run_time=1
        )
        
        # Step 2: Item 1 moves to Stage 3, Item 2 moves to Stage 2
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
        # Metaphor: A Tree summing up numbers
        
        header_red = Text("Reduction Patterns", font_size=32, color=PURPLE).next_to(title, DOWN)
        self.play(FadeIn(header_red))
        
        # Level 1: 4 nodes
        l1_nodes = VGroup(*[Circle(radius=0.3, color=WHITE).move_to(LEFT*1.5 + RIGHT*i + UP*0.5) for i in range(4)])
        # Level 2: 2 nodes
        l2_nodes = VGroup(
            Circle(radius=0.3, color=BLUE).move_to(l1_nodes[0].get_center() + RIGHT*0.5 + DOWN*1.5),
            Circle(radius=0.3, color=BLUE).move_to(l1_nodes[2].get_center() + RIGHT*0.5 + DOWN*1.5)
        )
        # Level 3: 1 node
        l3_node = Circle(radius=0.3, color=GREEN).move_to(DOWN*2.5) # Center bottomish
        
        # Lines
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