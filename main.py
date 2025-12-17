from manim import *
from manim_slides import Slide
import numpy as np

class ParallelComputingPresentation(Slide):
    def construct(self):
        # Hikaye Akışı  
        self.chapter_1_intro()      
        self.chapter_2_why()        
        self.chapter_3_apps()       
        self.chapter_4_how()        
        self.chapter_6_cuda()
        self.chapter_6_5_memory_model()    # Host vs Device Visuals
        self.chapter_8_cudathread()        # Deep dive into threadIdx/blockIdx
        self.chapter_kernel_configs()      # <<<Blocks, Threads>>> Visualizer
        self.chapter_9_code_walkthrough()  # Code Examples (CPU vs GPU)
        self.chapter_7_perf()       
        self.chapter_10_future()
    def chapter_1_intro(self):
        # -----------------------------------------
        # SLIDE 1: Başlık & Tanım
        # -----------------------------------------
        title = Text("Paralel Hesaplama Nedir?", font_size=48, color=BLUE)
        self.play(Write(title))
        
        self.next_slide() 
        
        self.play(title.animate.to_edge(UP))

        # Tanım Metni
        def_text = Text(
            "Bir problemi, işin birden fazla parçasını\naynı anda yaparak çözmektir.",
            font_size=36
        )
        def_text.set_color_by_gradient(WHITE, YELLOW)
        
        self.play(Write(def_text))
        
        self.next_slide()
        
        self.play(FadeOut(def_text))

        # -----------------------------------------
        # SLIDE 2: Ölçekler (Giriş)
        # -----------------------------------------
        scale_header = Text("Farklı Ölçeklerde Mevcuttur", font_size=40, color=TEAL).next_to(title, DOWN)
        self.play(FadeIn(scale_header))
        
        self.next_slide()

        # -----------------------------------------
        # SLIDE 3: Ölçek 1 - SIMD
        # -----------------------------------------
        s1_text = Text("1. CPU Çekirdeği İçi (SIMD)", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Görsel: Vektör Bloğu
        vec_box = Rectangle(width=4, height=0.8, color=GREEN)
        vec_data = VGroup(*[Square(side_length=0.6, fill_opacity=0.5, fill_color=GREEN).move_to(vec_box.get_left() + RIGHT*(0.5 + i)) for i in range(4)])
        vec_label = Text("Vektör Komutu", font_size=20).next_to(vec_box, UP)
        simd_group = VGroup(vec_box, vec_data, vec_label).next_to(s1_text, DOWN)

        self.play(Write(s1_text))
        self.play(Create(vec_box), FadeIn(vec_data), Write(vec_label))
        
        self.next_slide()
        
        self.play(FadeOut(simd_group), FadeOut(s1_text))

        # -----------------------------------------
        # SLIDE 4: Ölçek 2 - Çok Çekirdekli (Multicore)
        # -----------------------------------------
        s2_text = Text("2. CPU Çekirdekleri Arası (Thread'ler)", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Görsel: CPU Çipi
        chip_bg = Square(side_length=3, color=GREY, fill_opacity=0.2)
        cores = VGroup(
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(UL) + DOWN*0.8 + RIGHT*0.8),
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(UR) + DOWN*0.8 + LEFT*0.8),
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(DL) + UP*0.8 + RIGHT*0.8),
            Square(side_length=1, color=BLUE, fill_opacity=0.5).move_to(chip_bg.get_corner(DR) + UP*0.8 + LEFT*0.8),
        )
        core_labels = VGroup(*[Text("Çekirdek", font_size=16).move_to(c.get_center()) for c in cores])
        cpu_group = VGroup(chip_bg, cores, core_labels).next_to(s2_text, DOWN)

        self.play(Write(s2_text))
        self.play(Create(chip_bg), GrowFromCenter(cores), Write(core_labels))
        
        self.next_slide()

        self.play(FadeOut(s2_text))

        # -----------------------------------------
        # SLIDE 5: Ölçek 3 - GPU
        # -----------------------------------------
        s3_text = Text("3. Çoklu GPU Çekirdekleri Arası", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Görsel: GPU Izgarası
        gpu_bg = Rectangle(width=5, height=3, color=GREEN_E, fill_opacity=0.2)
        small_cores = VGroup()
        for x in range(10):
            for y in range(6):
                dot = Dot(radius=0.08, color=GREEN_B)
                dot.move_to(gpu_bg.get_corner(UL) + RIGHT*(0.25 + x*0.5) + DOWN*(0.25 + y*0.5))
                small_cores.add(dot)
        gpu_group = VGroup(gpu_bg, small_cores).next_to(s3_text, DOWN)

        self.play(Write(s3_text))
        self.play(Transform(cpu_group, gpu_group), Create(gpu_bg), ShowIncreasingSubsets(small_cores))
        
        self.next_slide()

        self.play(FadeOut(cpu_group), FadeOut(s3_text))

        # -----------------------------------------
        # SLIDE 6: Ölçek 4 - Dağıtık (Distributed)
        # -----------------------------------------
        s4_text = Text("4. Birden Çok Makine Arası (Dağıtık)", font_size=32).next_to(scale_header, DOWN, buff=1)
        
        # Görsel: Sunucular
        server1 = RoundedRectangle(height=1.5, width=1, corner_radius=0.2, color=WHITE).shift(LEFT*2.5)
        server2 = RoundedRectangle(height=1.5, width=1, corner_radius=0.2, color=WHITE)
        server3 = RoundedRectangle(height=1.5, width=1, corner_radius=0.2, color=WHITE).shift(RIGHT*2.5)
        lines = VGroup(
            Line(server1.get_right(), server2.get_left(), color=YELLOW),
            Line(server2.get_right(), server3.get_left(), color=YELLOW)
        )
        dist_group = VGroup(server1, server2, server3, lines).next_to(s4_text, DOWN)

        self.play(Write(s4_text))
        self.play(Transform(gpu_group, server1), DrawBorderThenFill(server2), DrawBorderThenFill(server3))
        self.play(Create(lines))
        
        self.next_slide()

        self.play(FadeOut(gpu_group), FadeOut(dist_group), FadeOut(s4_text), FadeOut(scale_header))

        # -----------------------------------------
        # SLIDE 7: Çapa Cümlesi (Anchor)
        # -----------------------------------------
        anchor_text = Paragraph(
            "“Paralel hesaplama, tek bir çipten veri merkezine kadar,",
            "eşzamanlı çalışmayı artırarak sonuca ulaşma",
            "süresini azaltma stratejisidir.”",
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
        # SLIDE 1: Başlık
        # -----------------------------------------
        title = Text("Neden İhtiyacımız Var?", font_size=48, color=BLUE)
        self.play(Write(title))
        
        self.next_slide()
        
        self.play(title.animate.to_edge(UP))

        # -----------------------------------------
        # SLIDE 2: Sınırlar (Güç/Isı)
        # -----------------------------------------
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 100, 20],
            axis_config={"include_numbers": False, "tip_shape": StealthTip},
            x_length=6,
            y_length=5
        ).shift(DOWN * 0.5 + LEFT * 0.5)

        x_label = axes.get_x_axis_label("Saat Hızı (GHz)")
        y_label = axes.get_y_axis_label(Text("Güç / Isı (Watt)"))

        self.play(Create(axes), Write(x_label), Write(y_label))
        
        #self.next_slide()

        # -----------------------------------------
        # ADIM 1: Üstel Eğri
        # -----------------------------------------
        graph = axes.plot(lambda x: 1.5 * x**3, color=WHITE, x_range=[0, 4.1])
        graph.set_color_by_gradient(GREEN, YELLOW, RED)
        
        label_curve = Text("P ∝ f³", font_size=36, color=YELLOW).move_to(axes.c2p(2, 60))
        
        self.play(Create(graph, run_time=2), FadeIn(label_curve))

        # -----------------------------------------
        # ADIM 2: "Güvenli Bölge" vs "Tehlike Bölgesi"
        # -----------------------------------------
        limit_line = DashedLine(
            start=axes.c2p(0, 80), 
            end=axes.c2p(5, 80), 
            color=RED
        )
        limit_text = Text("Soğutma Limiti", font_size=20, color=RED).next_to(limit_line, UP, aligned_edge=RIGHT)
        
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
        
        #self.next_slide()

        # -----------------------------------------
        # ADIM 3: Hızın Maliyeti
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
        text_gain = Text("+%33 Hız", font_size=20, color=GREEN).next_to(pt_b, UP)
        text_cost = Text("+%140 Isı!", font_size=24, color=RED).next_to(brace, RIGHT)
        
        self.play(Create(brace), Write(text_cost))

        #self.next_slide()

        self.play(
            FadeOut(graph), FadeOut(axes), FadeOut(dot), FadeOut(limit_line), 
            FadeOut(limit_text), FadeOut(pt_a), FadeOut(pt_b), 
            FadeOut(line_a), FadeOut(line_b), FadeOut(brace), 
            FadeOut(text_cost), FadeOut(title), 
            FadeOut(x_label), FadeOut(y_label), FadeOut(label_curve), FadeOut(label_a), FadeOut(label_b)
        )
        
        # Başlığı yeniden yaz
        title = Text("Neden İhtiyacımız Var?", font_size=40, color=BLUE).to_edge(UP)

        # -----------------------------------------
        # SLIDE 4: Verimlilik
        # -----------------------------------------
        point3_text = Text("3. Verimlilik (Perf/Watt)", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(title), Write(point3_text))

        eff_eq = MathTex(r"\text{Performans} \propto \frac{\text{İş}}{\text{Enerji}}").scale(1.5)
        self.play(Write(eff_eq))
        
        sub_text = Text("Paralellik, daha iyi verimliliğe giden yoldur.", color=YELLOW, font_size=24).next_to(eff_eq, DOWN, buff=1)
        self.play(FadeIn(sub_text))

        self.next_slide()
        
        self.play(FadeOut(eff_eq), FadeOut(sub_text), FadeOut(point3_text))

        # -----------------------------------------
        # SLIDE 5: Köprü Cümlesi
        # -----------------------------------------
        bridge_text = Paragraph(
            "“Böylece performans artışı, daha hızlı çekirdeklerden",
            "daha fazla çekirdeğe",
            "ve özelleşmiş donanımlara kaydı.”",
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
        # KURULUM: Başlık ve Merkez Hub
        # -----------------------------------------
        title = Text("3. Bununla Ne Yapabiliriz?", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))

        # Central Hub
        hub_circle = Circle(radius=1.2, color=WHITE, fill_opacity=0.1)
        hub_text = Paragraph("Paralel", "Hesaplama", alignment="center", font_size=24)
        hub = VGroup(hub_circle, hub_text).move_to(ORIGIN)
        
        self.play(DrawBorderThenFill(hub_circle), Write(hub_text))
        
        self.next_slide()

        # -----------------------------------------
        # 6 KATEGORİYİ TANIMLAMA
        # -----------------------------------------
        radius = 3.0
        angles = [150, 90, 30, 210, 270, 330] # Derece
        positions = [
            radius * np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0]) 
            for a in angles
        ]

        # --- 1. Bilimsel Hesaplama (Mesh/Grid) ---
        science_pos = positions[0]
        science_grid = VGroup()
        for i in range(3):
            for j in range(3):
                sq = Square(side_length=0.4, color=BLUE, fill_opacity=0.5)
                sq.move_to(science_pos + np.array([(i-1)*0.4, (j-1)*0.4, 0]))
                science_grid.add(sq)
        science_label = Text("Bilimsel\n(CFD/PDE)", font_size=18).next_to(science_grid, UP)
        line_1 = Line(hub.get_left(), science_grid.get_bottom(), color=GREY).set_opacity(0.5)

        # --- 3. AI/ML (Sinir Ağı) ---
        ai_pos = positions[2]
        nodes_L1 = VGroup(*[Dot(radius=0.1, color=PURPLE).move_to(ai_pos + LEFT*0.5 + UP*(i-0.5)*0.5) for i in range(2)])
        nodes_L2 = VGroup(*[Dot(radius=0.1, color=PURPLE).move_to(ai_pos + RIGHT*0.5 + UP*(i-1)*0.5) for i in range(3)])
        edges = VGroup()
        for n1 in nodes_L1:
            for n2 in nodes_L2:
                edges.add(Line(n1.get_center(), n2.get_center(), stroke_width=1, color=PURPLE_A))
        ai_group = VGroup(edges, nodes_L1, nodes_L2)
        ai_label = Text("Yapay Zeka / ML", font_size=18).next_to(ai_group, UP)
        line_3 = Line(hub.get_right(), ai_group.get_bottom(), color=GREY).set_opacity(0.5)

        # --- 4. Medya (Piksel/Resim) ---
        media_pos = positions[3]
        media_pixels = VGroup()
        colors = [RED, GREEN, BLUE, YELLOW]
        for i in range(2):
            for j in range(2):
                c = colors[(i+j)%4]
                p = Square(side_length=0.5, color=c, fill_opacity=0.8, stroke_width=0)
                p.move_to(media_pos + np.array([(i-0.5)*0.5, (j-0.5)*0.5, 0]))
                media_pixels.add(p)
        media_label = Text("Medya\n(Video/Resim)", font_size=18).next_to(media_pixels, DOWN)
        line_4 = Line(hub.get_left(), media_pixels.get_top(), color=GREY).set_opacity(0.5)

        # --- 5. Sinyal İşleme (Dalga) ---
        sig_pos = positions[4]
        sine_wave = FunctionGraph(
            lambda t: 0.5 * np.sin(3*t),
            x_range=[-1, 1],
            color=TEAL
        ).move_to(sig_pos)
        sig_label = Text("Sinyal İşleme\n(Radar/FFT)", font_size=18).next_to(sine_wave, DOWN)
        line_5 = Line(hub.get_bottom(), sine_wave.get_top(), color=GREY).set_opacity(0.5)

        # --- 6. Finans (Borsa Grafiği) ---
        fin_pos = positions[5]
        fin_line = VMobject().set_color(GREEN)
        pts = [
            [-1, -0.5, 0], [-0.5, -0.2, 0], [0, -0.6, 0], 
            [0.5, 0.2, 0], [1, 0.8, 0]
        ]
        real_pts = [np.array(p) * 0.8 + fin_pos for p in pts]
        fin_line.set_points_as_corners(real_pts)
        fin_label = Text("Finans", font_size=18).next_to(fin_line, DOWN)
        line_6 = Line(hub.get_right(), fin_line.get_top(), color=GREY).set_opacity(0.5)

        # -----------------------------------------
        # ANİMASYON SIRASI
        # -----------------------------------------
        
        # 1. Bilim
        self.play(Create(line_1), FadeIn(science_grid), Write(science_label))
        self.play(science_grid.animate.set_color(RED), run_time=0.5) 

        # 3. YZ
        self.play(Create(line_3), Create(ai_group), Write(ai_label))
        self.play(ShowPassingFlash(edges.copy().set_color(WHITE), time_width=0.5), run_time=1)

        # 4. Medya
        self.play(Create(line_4), FadeIn(media_pixels), Write(media_label))
        self.play(Rotate(media_pixels, PI/2), run_time=0.5) 

        # 5. Sinyal
        self.play(Create(line_5), Create(sine_wave), Write(sig_label))
        self.play(sine_wave.animate.stretch(1.5, 0), run_time=0.5)

        # 6. Finans
        self.play(Create(line_6), Create(fin_line), Write(fin_label))
        self.play(Wiggle(fin_line), run_time=0.5) 
        self.next_slide()

        # -----------------------------------------
        # KAPANIŞ MESAJI
        # -----------------------------------------
        final_box = Rectangle(width=10, height=2, color=BLACK, fill_opacity=0.8).to_edge(DOWN)
        final_text = Text(
            "“Çoğu modern iş yükü doğası gereği veri paraleldir\nveya buna uygun şekilde yeniden yapılandırılabilir.”",
            font_size=28, color=YELLOW, slant=ITALIC
        ).move_to(final_box)

        self.play(FadeIn(final_box), Write(final_text))
        
        self.next_slide()
        
        # Temizlik
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
        # KURULUM: Başlık
        # -----------------------------------------
        title = Text("4. Bunu Nasıl Yapabiliriz?", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # KAVRAM 1: Veri Paralelliği
        # -----------------------------------------
        header_data = Text("Veri Paralelliği (Data Parallelism)", font_size=32, color=YELLOW).next_to(title, DOWN)
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
        
        text_data = Text("Aynı fonksiyon, farklı veri", font_size=24).next_to(array_group, DOWN, buff=1.5)
        self.play(Write(text_data))
        
        self.next_slide()
        self.play(FadeOut(array_group), FadeOut(header_data), FadeOut(text_data), FadeOut(chunk1), FadeOut(chunk2), FadeOut(chunk3), FadeOut(chunk4))

        # -----------------------------------------
        # KAVRAM 2: Görev Paralelliği
        # -----------------------------------------
        header_task = Text("Görev Paralelliği (Task Parallelism)", font_size=32, color=ORANGE).next_to(title, DOWN)
        self.play(FadeIn(header_task))
        
        task1 = Circle(radius=0.4, color=RED).shift(LEFT*2)
        task2 = Triangle(color=BLUE).scale(0.5)
        task3 = Square(side_length=0.8, color=GREEN).shift(RIGHT*2)
        
        tasks = VGroup(task1, task2, task3)
        self.play(Create(tasks))
        
        l1 = Text("Arayüz", font_size=16).next_to(task1, DOWN)
        l2 = Text("Ağ", font_size=16).next_to(task2, DOWN)
        l3 = Text("Hesaplama", font_size=16).next_to(task3, DOWN)
        labels = VGroup(l1, l2, l3)
        self.play(Write(labels))
        
        self.play(
            Rotate(task1),
            Wiggle(task2),
            ScaleInPlace(task3, 1.2),
            rate_func=there_and_back,
            run_time=1
        )
        
        text_task = Text("Farklı fonksiyonlar eşzamanlı", font_size=24).next_to(labels, DOWN, buff=0.5)
        self.play(Write(text_task))

        self.next_slide()
        self.play(FadeOut(tasks), FadeOut(labels), FadeOut(header_task), FadeOut(text_task))

        # -----------------------------------------
        # KAVRAM 3: Pipeline Paralelliği
        # -----------------------------------------
        header_pipe = Text("Boru Hattı (Pipeline) Paralelliği", font_size=32, color=TEAL).next_to(title, DOWN)
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
        
        text_pipe = Text("Akış işleme (Montaj Hattı)", font_size=24).next_to(stages, DOWN)
        self.play(Write(text_pipe))

        self.next_slide()
        self.play(FadeOut(stages), FadeOut(arrow1), FadeOut(arrow2), FadeOut(item1), FadeOut(item2), FadeOut(header_pipe), FadeOut(text_pipe))

        # -----------------------------------------
        # KAVRAM 4: İndirgeme
        # -----------------------------------------
        header_red = Text("İndirgeme (Reduction) Desenleri", font_size=32, color=PURPLE).next_to(title, DOWN)
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
        
        text_red = Text("Sonuçları birleştirme (Map -> Reduce)", font_size=24).next_to(l3_node, RIGHT)
        self.play(Write(text_red))
        
        self.next_slide()
        self.play(FadeOut(l1_nodes), FadeOut(l2_nodes), FadeOut(l3_node), FadeOut(lines), FadeOut(header_red), FadeOut(text_red))

        # -----------------------------------------
        # KÖPRÜ MESAJI
        # -----------------------------------------
        final_text = Paragraph(
            "“Seçtiğiniz mimari, probleminizin sunduğu",
            "paralellik türüyle",
            "eşleşmelidir.”",
            alignment="center",
            font_size=34, color=YELLOW
        )
        self.play(Write(final_text))
        
        self.next_slide()
        self.play(FadeOut(final_text), FadeOut(title))
        self.wait(0.5)

    def chapter_6_cuda(self):
        # -----------------------------------------
        # KURULUM: Başlık
        # -----------------------------------------
        title = Text("CUDA'nın Ortaya Çıkışı ve Tarihi", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # -----------------------------------------
        # FAZ 1: Grafik Çağı (Tarih Öncesi)
        # -----------------------------------------
        header_gfx = Text("Aslında grafikler için Üretildi", font_size=32, color=BLUE).next_to(title, DOWN)
        
        screen_border = Rectangle(width=4, height=3, color=WHITE)
        triangle = Triangle(color=RED, fill_opacity=0.8).scale(0.8)
        
        gpu_chip = Square(side_length=1.05, color=GREEN, fill_opacity=0.5).shift(DOWN*2)
        gpu_label = Text("GPU", font_size=20).move_to(screen_border.get_top()+UP*0.2)
        
        wire = Line(gpu_chip.get_top(), screen_border.get_bottom(), color=GREEN)
        
        self.play(FadeIn(header_gfx), Create(screen_border), FadeIn(triangle), Write(gpu_label), Create(wire))
        
        self.play(Rotate(triangle, angle=TAU, run_time=2))
        
        self.next_slide()

        # -----------------------------------------
        # FAZ 2: Farkındalık (GPGPU)
        # -----------------------------------------
        header_gpgpu = Text("İşin Sonunda Matematik Değil Mi?", font_size=32, color=YELLOW).next_to(title, DOWN)
        
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
        self.play(FadeOut(screen_border), FadeOut(matrix_nums), FadeOut(wire), FadeOut(header_gpgpu),FadeOut(gpu_label))

        # -----------------------------------------
        # FAZ 3: CUDA (Köprü)
        # -----------------------------------------
        header_cuda = Text("CUDA: Programlanabilir Hale Getirmek", font_size=32, color=GREEN).next_to(title, DOWN)
        self.play(FadeIn(header_cuda))
        
        code_bg_left = Rectangle(width=3, height=3.5, color=GREY, fill_opacity=0.2).shift(LEFT*3)
        code_gfx = Text("glBegin();\nglVertex3f();\nTexture();\n// ???!", font_size=18,  color=RED).move_to(code_bg_left)
        label_left = Text("2007 Öncesi (Grafik API)", font_size=20, color=RED).next_to(code_bg_left, DOWN)
        
        code_bg_right = Rectangle(width=4, height=3.5, color=GREEN, fill_opacity=0.2).shift(RIGHT*3)
        code_cuda = Text("__global__ void\nkernel(float* x) {\n  int i = threadIdx.x;\n  x[i] = ...;\n}", font_size=18,  color=WHITE).move_to(code_bg_right)
        label_right = Text("CUDA (C++ tarzı)", font_size=20, color=GREEN).next_to(code_bg_right, DOWN)
        
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
        # FAZ 4: Ekosistem Büyümesi
        # -----------------------------------------
        header_eco = Text("Ekosistem Büyümesi", font_size=32, color=BLUE).next_to(title, DOWN)
        
        #gpu_chip.move_to(DOWN*2.5) 
        
        layer_cuda = Rectangle(width=4, height=0.8, color=GREEN, fill_opacity=0.8).next_to(gpu_chip, UP, buff=0.1)
        txt_cuda = Text("CUDA Çekirdeği", font_size=24, color=BLACK).move_to(layer_cuda)
        
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
        app_txt = Text("Modern YZ ve Simülasyon", font_size=24, color=YELLOW).move_to(app_layer)
        
        self.play(FadeIn(header_eco))
        # FadeIn gpu_chip in case it was lost in transitions
        #self.play(Write(gpu_label))
        self.play(DrawBorderThenFill(layer_cuda), Write(txt_cuda))
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN) for b in lib_blocks], lag_ratio=0.2))
        self.play(GrowFromCenter(app_layer), Write(app_txt))
        
        self.next_slide()
        self.play(FadeOut(layer_cuda), FadeOut(txt_cuda), FadeOut(lib_blocks), FadeOut(app_layer), FadeOut(app_txt), FadeOut(header_eco))

        # -----------------------------------------
        # SON MESAJ
        # -----------------------------------------
        final_text = Paragraph(
            "“CUDA paralel hesaplamanın başlangıcı değil,",
            "ancak pratik ve geliştirici dostu GPU hesaplama",
            "için büyük bir dönüm noktasıdır.”",
            alignment="center",
            font_size=32, color=YELLOW
        )
        self.play(Write(final_text))
        
        self.next_slide()
        self.play(FadeOut(final_text), FadeOut(title))
        self.wait(0.5)

    def chapter_6_5_memory_model(self):
        # -----------------------------------------
        # NEW SECTION: HOST VS DEVICE VISUALIZATION
        # -----------------------------------------
        title = Text("1. Host vs. Device Model", font_size=40).to_edge(UP)
        self.play(Write(title))

        # --- Draw CPU (Host) ---
        cpu_box = Rectangle(height=4, width=3, color=BLUE).to_edge(LEFT, buff=1)
        cpu_label = Text("CPU (Host)", font_size=24, color=BLUE).next_to(cpu_box, UP)
        ram_label = Text("System RAM", font_size=20).move_to(cpu_box.get_center())
        
        # --- Draw GPU (Device) ---
        gpu_box = Rectangle(height=4, width=3, color=GREEN).to_edge(RIGHT, buff=1)
        gpu_label = Text("GPU (Device)", font_size=24, color=GREEN).next_to(gpu_box, UP)
        vram_label = Text("Global Memory", font_size=20).move_to(gpu_box.get_center())

        self.play(
            Create(cpu_box), Write(cpu_label), Write(ram_label),
            Create(gpu_box), Write(gpu_label), Write(vram_label)
        )
        self.next_slide()  # Wait for user

        # --- Data Transfer (PCIe Bus) ---
        arrow_to_gpu = Arrow(cpu_box.get_right(), gpu_box.get_left(), buff=0.1, color=YELLOW)
        memcpy_label = Text("cudaMemcpy\n(HostToDevice)", font_size=16, color=YELLOW).next_to(arrow_to_gpu, UP)
        
        data_packet = Square(side_length=0.5, color=WHITE, fill_opacity=0.5).move_to(cpu_box.get_center())
        
        self.play(FadeIn(data_packet))
        self.play(
            Create(arrow_to_gpu), 
            Write(memcpy_label),
            data_packet.animate.move_to(gpu_box.get_center())
        )
        self.next_slide() # Wait for user

        # --- Kernel Launch ---
        kernel_text = Text("Kernel Launch\n(Compute)", font_size=20, color=RED).next_to(arrow_to_gpu, DOWN)
        processing_flash = Flash(gpu_box, color=RED, flash_radius=1.5)
        
        self.play(Write(kernel_text))
        self.play(processing_flash)
        self.play(data_packet.animate.set_color(RED)) # Data processed
        self.next_slide() # Wait for user

        # --- Copy Back ---
        arrow_to_cpu = Arrow(gpu_box.get_left(), cpu_box.get_right(), buff=0.1, color=YELLOW).shift(DOWN * 0.5)
        # Shift original arrow up to make room
        self.play(
            arrow_to_gpu.animate.shift(UP * 0.5),
            memcpy_label.animate.shift(UP * 0.5),
            Create(arrow_to_cpu)
        )
        
        memcpy_back_label = Text("cudaMemcpy\n(DeviceToHost)", font_size=16, color=YELLOW).next_to(arrow_to_cpu, DOWN)
        self.play(Write(memcpy_back_label))
        
        self.play(data_packet.animate.move_to(cpu_box.get_center()))
        self.next_slide() # Wait for user

        # Cleanup
        self.play(FadeOut(Group(title, cpu_box, cpu_label, ram_label, gpu_box, gpu_label, vram_label, arrow_to_gpu, memcpy_label, data_packet, kernel_text, arrow_to_cpu, memcpy_back_label)))

    def chapter_kernel_configs(self):
        # -----------------------------------------
        # NEW SECTION: KERNEL CONFIGS DEMO
        # -----------------------------------------
        
        # Title that stays at the top
        title = Text("CUDA Thread Manipülasyonu", font_size=48).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # We will cycle through these configurations
        # Format: (blocks, threads, code_string)
        configs = [
            (1, 1, "VectorAdd<<<1, 1>>>();"),
            (1, 5, "VectorAdd<<<1, 5>>>();"),
            (5, 1, "VectorAdd<<<5, 1>>>();"),
            (5, 5, "VectorAdd<<<5, 5>>>();"),
        ]

        # Keep track of previous objects to transform/fade out
        prev_group = VGroup()
        prev_code = VGroup()
        prev_stats = VGroup()

        for blocks, threads, code_str in configs:
            # 1. Create the Code Snippet
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
                Text(f"Grid Yapısı: Her gridde {blocks} Block", font_size=24, color=BLUE),
                Text(f"Block Yapısı:  Her Blokda {threads} Thread", font_size=24, color=GREEN),
                Text(f"Her geçişte {blocks} x {threads} = {total_threads} İşlem", font_size=30, color=WHITE)
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
            if len(prev_group) > 0:
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

    def chapter_8_cudathread(self):
        # ---------------------------------------------------------
        # SECTION 1: The Basic Unit (The Thread)
        # ---------------------------------------------------------
        title = Text("CUDA Thread Yapısı", font_size=48).to_edge(UP)
        self.play(Write(title))
        
        # Create a single thread representation
        thread_box = Square(side_length=0.8, color=BLUE, fill_opacity=0.5)
        thread_label = Text("Thread", font_size=20).next_to(thread_box, DOWN)
        
        self.play(DrawBorderThenFill(thread_box), Write(thread_label))
        
        # Pause: Explain that a thread is the smallest unit of execution
        self.next_slide() 
        
        # ---------------------------------------------------------
        # SECTION 2: Thread Block (Grouping Threads)
        # ---------------------------------------------------------
        # Move original thread to a position to start the block formation
        self.play(FadeOut(thread_label), thread_box.animate.move_to(LEFT * 4))
        
        # Create a Block of 4 threads (1D Block for simplicity)
        # We duplicate the original thread 3 times
        block_threads = VGroup(thread_box)
        for i in range(3):
            new_thread = thread_box.copy().next_to(block_threads[-1], RIGHT, buff=0.1)
            block_threads.add(new_thread)
            
        self.play(Create(block_threads[1:]))
        
        # Group them visually with a rectangle
        block_rect = SurroundingRectangle(block_threads, color=GREEN, buff=0.2)
        block_label = Text("Thread Block", color=GREEN, font_size=24).next_to(block_rect, UP)
        
        self.play(Create(block_rect), Write(block_label))
        
        # Add indices to threads (threadIdx.x)
        indices = VGroup()
        for i, t in enumerate(block_threads):
            idx = Integer(i, font_size=20).move_to(t.get_center())
            indices.add(idx)
        
        self.play(Write(indices))
        
        code_tidx = Text("threadIdx.x", font_size=24).next_to(block_rect, DOWN)
        self.play(FadeIn(code_tidx))
        
        # Pause: Explain blockDim and threadIdx
        self.next_slide()

        # ---------------------------------------------------------
        # SECTION 3: The Grid (Grouping Blocks)
        # ---------------------------------------------------------
        # Group the entire block conceptual object
        full_block_group = VGroup(block_threads, block_rect, indices, block_label, code_tidx)
        
        # Zoom out / Scale down to fit the Grid
        self.play(
            full_block_group.animate.scale(0.6).to_edge(LEFT, buff=1)
        )
        
        # Create 2 more blocks to form a Grid
        grid_group = VGroup(full_block_group)
        for i in range(2):
            new_block = full_block_group.copy().next_to(grid_group[-1], RIGHT, buff=0.5)
            # Update the block label for visual clarity (optional, but good for details)
            grid_group.add(new_block)
            
        self.play(FadeIn(grid_group[1:]))
        
        grid_rect = SurroundingRectangle(grid_group, color=RED, buff=0.2)
        grid_label = Text("Grid", color=RED, font_size=30).next_to(grid_rect, UP)
        
        self.play(Create(grid_rect), Write(grid_label))
        
        # Pause: Explain gridDim and blockIdx
        self.next_slide()
        
        # ---------------------------------------------------------
        # SECTION 4: Global Index Calculation
        # ---------------------------------------------------------
        # Clear noise to focus on the formula
        self.play(
            FadeOut(title), FadeOut(grid_label), FadeOut(grid_rect),
            grid_group.animate.move_to(UP * 0.5)
        )
        
        # The CUDA formula
        formula = MathTex(
            r"i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}",
            font_size=36
        ).move_to(DOWN * 2)
        
        self.play(Write(formula))
        
        # Demonstrate calculation for a specific thread
        # Let's pick: Block 2 (index 1), Thread 3 (index 2) -> Global Index 6
        target_block = grid_group[1] # 2nd block
        # Access the VGroup structure we built: [threads, rect, indices, label, code]
        # threads is at index 0 inside the block group
        target_thread = target_block[0][2] # 3rd thread
        
        self.play(
            target_thread.animate.set_fill(YELLOW, opacity=0.8),
            Indicate(target_block[1], color=YELLOW) # Highlight block rect
        )
        
        # Show calculation text
        calc_text = MathTex(
            r"i = 1 \times 4 + 2 = 6",
            color=YELLOW, font_size=36
        ).next_to(formula, DOWN)
        
        self.play(TransformFromCopy(formula, calc_text))
        
        self.next_slide()
        
        # Clean up
        self.play(FadeOut(Group(*self.mobjects)))

    def chapter_9_code_walkthrough(self):
            # -----------------------------------------
            # NEW SECTION: CODE WALKTHROUGH
            # -----------------------------------------
            # Title
            title = Text("Kısa Örnek: Vector Add", font_size=40, color=BLUE).to_edge(UP)
            self.play(Write(title))
            self.next_slide()

            # --- PART 1: The CPU Way ---
            cpu_code_str = """void vectorAdd(int *a, int *b, int *c, int n) {
        for (int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }"""
            cpu_code = Code(
                code_string=cpu_code_str,
                tab_width=4,
                background="window",
                language="cpp",
                
            ).move_to(ORIGIN)
            
            lbl_cpu = Text("Standard C++ (CPU)", font_size=24, color=GREY).next_to(cpu_code, UP)

            self.play(FadeIn(cpu_code), Write(lbl_cpu))
            self.next_slide()

            # --- PART 2: Transforming to GPU ---
            gpu_code_str = """__global__ void vectorAdd(int *a, int *b, int *c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }"""
            gpu_code = Code(
                code_string=gpu_code_str,
                tab_width=4,
                background="window",
                language="cpp",
                
            ).move_to(ORIGIN)
            
            lbl_gpu = Text("CUDA Kernel (GPU)", font_size=24, color=GREEN).next_to(gpu_code, UP)

            # Transform CPU code to GPU code
            self.play(
                ReplacementTransform(cpu_code, gpu_code),
                ReplacementTransform(lbl_cpu, lbl_gpu)
            )
            self.next_slide()
            
            # --- HIGHLIGHTING SECTIONS (FIXED INDICES) ---
            # Note: We use gpu_code[2] to access the code lines VGroup
            
            # 1. Highlight __global__ (Line 0)
            rect_global = SurroundingRectangle(gpu_code[2][0], color=YELLOW, buff=0.05)
            txt_global = Text("Device üstünde çalışır, Host tarafından çağırılır", font_size=20, color=YELLOW).next_to(rect_global, UP, buff=0.2)
            
            self.play(Create(rect_global), Write(txt_global))
            self.next_slide()
            self.play(FadeOut(rect_global), FadeOut(txt_global))
            
            # 2. Highlight Index Calculation (Line 1)
            rect_idx = SurroundingRectangle(gpu_code[2][1], color=YELLOW, buff=0.05)
            txt_idx = Text("Threadin index hesaplaması", font_size=20, color=YELLOW).next_to(rect_idx.get_center(), RIGHT + DOWN, buff=0.2)
            
            self.play(Create(rect_idx), Write(txt_idx))
            self.next_slide()
            self.play(FadeOut(rect_idx), FadeOut(txt_idx))
            
            # 3. Highlight Guard Check (Line 2)
            rect_guard = SurroundingRectangle(gpu_code[2][2], color=YELLOW, buff=0.05)
            txt_guard = Text("Boundary Check (Safety)", font_size=20, color=YELLOW).next_to(rect_guard, RIGHT, buff=0.2)
            
            self.play(Create(rect_guard), Write(txt_guard))
            self.next_slide()
            self.play(FadeOut(rect_guard), FadeOut(txt_guard), FadeOut(gpu_code), FadeOut(lbl_gpu))

            # --- PART 3: The Host API ---
            host_title = Text("Host(CPU) tarafında olanlar (Main)", font_size=32, color=BLUE).next_to(title, DOWN)
            self.play(Write(host_title))
            
            host_code_str = """int main() {
        cudaMalloc(&d_a, bytes);

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);

        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
    }"""
            host_code = Code(
                code_string=host_code_str,
                tab_width=4,
                background="window",
                language="cpp",
                
                
            ).scale(0.8).next_to(host_title, DOWN)
            
            self.play(FadeIn(host_code))
            self.next_slide()
            
            # Step through important lines
            # Indices manually checked against the string above (0-indexed)
            steps = [
                (1, "Reserve VRAM"),     # cudaMalloc
                (3, "Send Data"),        # cudaMemcpy H2D
                (5, "Execute"),         # Kernel Launch
                (7, "Get Results"),     # cudaMemcpy D2H
                (9, "Cleanup")          # cudaFree
            ]
            
            for line_idx, note in steps:
                # Safely access the line using [2] for the code block
                target_line = host_code[2][line_idx]
                
                arrow = Arrow(LEFT, RIGHT, color=YELLOW).next_to(target_line, LEFT)
                txt = Text(note, font_size=15, color=YELLOW).next_to(arrow, LEFT)
                
                self.play(Create(arrow), Write(txt), run_time=0.5)
                self.next_slide()
                self.play(FadeOut(arrow), FadeOut(txt), run_time=0.3)
                
            self.play(FadeOut(host_code), FadeOut(host_title), FadeOut(title))
    def chapter_7_perf(self):
            # ---------------------------------------------------------
            # 1. Veri Kurulumu
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
            # 2. Eksen Kurulumu
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
                # Sayılarda sorun yok, MathTex kalabilir
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

            # DÜZELTME: Tex yerine Text kullanıldı (Türkçe karakterler için)
            x_title = Text("Girdi Boyutu (N)", font_size=24).next_to(axes.x_axis, DOWN, buff=0.08)
            y_title = Text("Süre (ms) - Log Ölçeği", font_size=24).next_to(axes.y_axis, UP, buff=0.02).shift(RIGHT*1.5)

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
            # 3. Lejant
            # ---------------------------------------------------------
            # DÜZELTME: Tex yerine Text kullanıldı
            main_title = Text("Performans: FFT", font_size=40).to_edge(UP).shift(LEFT * 2)

            legend_items = [
                (COL_CPU, "CPU Süresi"),
                (COL_GPU_E2E, "GPU Uçtan Uca Süre"),
                (COL_GPU_COMP, "GPU Hesaplama Süresi")
            ]
            
            legend = VGroup()
            for color, text in legend_items:
                # DÜZELTME: Tex yerine Text kullanıldı
                item = VGroup(
                    Line(color=color, stroke_width=6).set_length(0.6),
                    Text(text, color=color, font_size=20)
                ).arrange(RIGHT, buff=0.1)
                legend.add(item)
            
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            legend.to_corner(UP + RIGHT, buff=0.5)
            legend_bg = SurroundingRectangle(legend, color=WHITE, fill_color=BLACK, fill_opacity=0.85, buff=0.1)
            legend_group = VGroup(legend_bg, legend)

            # ---------------------------------------------------------
            # 4. Yardımcı Fonksiyonlar
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
                    # Rakamlar ASCII olduğu için Text/Tex fark etmez ama tutarlılık için Text yapıyoruz
                    lbl = Text(label_text, color=color, font_size=18)
                    lbl.next_to(points[-1], RIGHT, buff=0.15)
                    end_labels.add(lbl)
                
                crossover_pt = get_intersection_point(data[key]["cpu"], data[key]["gpu_e2e"])
                crossover_group = VGroup()
                if crossover_pt is not None:
                    dot = Dot(point=crossover_pt, color=WHITE, radius=0.08)
                    glow = Dot(point=crossover_pt, color=COL_GPU_E2E, radius=0.15).set_opacity(0.5)
                    # DÜZELTME: Tex yerine Text kullanıldı (Türkçe: Kazanır)
                    text = Text("GPU Wins", color=WHITE, font_size=20)
                    text.next_to(dot, UP + LEFT, buff=0.1)
                    arrow = Arrow(start=text.get_bottom(), end=dot.get_top(), color=WHITE, buff=0.05, stroke_width=2, tip_length=0.15)
                    crossover_group.add(glow, dot, text, arrow)
                return curves, end_labels, crossover_group

            # ---------------------------------------------------------
            # 5. Animasyon Dizisi
            # ---------------------------------------------------------
            keys = ["fft", "lfilter", "fftconv"]
            titles = ["FFT", "LFilter", "FFTConv"]
            
            current_curves = VGroup()
            current_labels = VGroup()
            current_crossover = VGroup()
            
            self.play(Create(axes_group), Write(main_title), FadeIn(legend_group), run_time=1.5)
            
            for i, key in enumerate(keys):
                new_curves, new_labels, new_crossover = get_scene_elements(key)
                new_title_text = f"Performans: {titles[i]}"
                
                if i == 0:
                    current_curves = new_curves
                    current_labels = new_labels
                    current_crossover = new_crossover
                    
                    self.play(Create(current_curves), Write(current_labels), run_time=2)
                    if len(current_crossover) > 0:
                        self.play(FadeIn(current_crossover))
                    
                    self.next_slide()

                else:
                    # DÜZELTME: Tex yerine Text kullanıldı
                    target_title = Text(new_title_text, font_size=40).move_to(main_title)
                    
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

    def chapter_10_future(self):
            # -----------------------------------------
            # SLIDE 1: Buzdağının Görünen Kısmı
            # -----------------------------------------
            title = Text("Daha Gidecek Çok Yol Var...", font_size=40, color=BLUE).to_edge(UP)
            self.play(Write(title))
            self.next_slide()

            # --- Topic 1: Memory (Bellek) ---
            # Visual: A grid representing memory banks or coalescing
            mem_icon = VGroup(*[Square(side_length=0.2, fill_opacity=0.8, color=GREEN) for _ in range(9)])
            mem_icon.arrange_in_grid(3, 3, buff=0.05)
            
            mem_title = Text("Gelişmiş Bellek", font_size=24, color=GREEN).next_to(mem_icon, DOWN)
            mem_details = Text(
                "• Shared Memory\n• Memory Coalescing\n• Bank Conflicts",
                font_size=16, color=GREY_A
            ).next_to(mem_title, DOWN)
            
            mem_group = VGroup(mem_icon, mem_title, mem_details)

            # --- Topic 2: Streams (Akışlar) ---
            # Visual: Parallel arrows showing async execution
            stream_icon = VGroup(
                Arrow(start=LEFT, end=RIGHT, color=YELLOW, buff=0, stroke_width=4).scale(0.4),
                Arrow(start=LEFT, end=RIGHT, color=YELLOW, buff=0, stroke_width=4).scale(0.4),
                Arrow(start=LEFT, end=RIGHT, color=YELLOW, buff=0, stroke_width=4).scale(0.4)
            ).arrange(DOWN, buff=0.15)
            
            stream_title = Text("Eşzamanlılık (Streams)", font_size=24, color=YELLOW).next_to(stream_icon, DOWN)
            stream_details = Text(
                "• Async Data Transfer\n• Kernel Overlapping\n• Multi-GPU",
                font_size=16, color=GREY_A
            ).next_to(stream_title, DOWN)
            
            stream_group = VGroup(stream_icon, stream_title, stream_details)

            # --- Topic 3: Optimization (Optimizasyon) ---
            # Visual: A gauge or gear representing tuning
            opt_icon = VGroup(
                Circle(radius=0.4, color=RED, stroke_width=4),
                Line(ORIGIN, UP*0.3, color=RED, stroke_width=4).rotate(-45*DEGREES)
            )
            # Animate the needle later
            
            opt_title = Text("Mikro Optimizasyon", font_size=24, color=RED).next_to(opt_icon, DOWN)
            opt_details = Text(
                "• Warp Divergence\n• Occupancy Calculator\n• Instruction Level Opt.",
                font_size=16, color=GREY_A
            ).next_to(opt_title, DOWN)
            
            opt_group = VGroup(opt_icon, opt_title, opt_details)

            # --- Arrange and Animate ---
            # Position the three groups horizontally
            all_topics = VGroup(mem_group, stream_group, opt_group).arrange(RIGHT, buff=1.5)
            
            # 1. Memory
            self.play(FadeIn(mem_group, shift=UP))
            self.play(mem_icon.animate.set_color(WHITE), run_time=0.5)
            
            # 2. Streams
            self.play(FadeIn(stream_group, shift=UP))
            self.play(LaggedStart(
                *[s.animate.shift(RIGHT*0.2) for s in stream_icon], 
                lag_ratio=0.2, run_time=1
            ))
            
            # 3. Optimization
            self.play(FadeIn(opt_group, shift=UP))
            self.play(Rotate(opt_icon[1], angle=90*DEGREES, about_point=opt_icon[0].get_center()))

            self.next_slide()

            # -----------------------------------------
            # SLIDE 2: Kapanış
            # -----------------------------------------
            self.play(
                FadeOut(all_topics),
                FadeOut(title)
            )

            final_text = Text("Dinlediğiniz İçin Teşekkürler", font_size=48, color=BLUE)
            contact_info = Text("@meliksahsagun", font_size=24, color=GREY).next_to(final_text, DOWN, buff=0.5)

            self.play(Write(final_text))
            self.play(FadeIn(contact_info))
            
            self.wait(2)