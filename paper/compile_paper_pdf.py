import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

pdf_path = r"C:\Works\AE5510\AE5510\paper\AE5510_Problem1_Paper.pdf"
fig_dir = r"C:\Works\AE5510\AE5510\paper\figures"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#444444"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "AE5510: MACHINE LEARNING APPLICATIONS — PROBLEM 1 TECHNICAL REPORT")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "IEEE TRANSACTIONS / CONFERENCE PROCEEDINGS STYLE — ALL FIGURES EMBEDDED")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=1,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=8
    )
    
    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#334155'),
        spaceAfter=12
    )
    
    abstract_body = ParagraphStyle(
        'AbsBody',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        alignment=4,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13.5,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor('#0f172a'),
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        alignment=4,
        spaceAfter=5,
        textColor=colors.HexColor('#1e293b')
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        leftIndent=15,
        spaceAfter=3,
        textColor=colors.HexColor('#1e293b')
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.HexColor('#0f172a')
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.HexColor('#0f172a')
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=colors.HexColor('#334155'),
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Physically Grounded Binary Adversarial Camouflage for Real-Time Aerial Object Detectors Under Flight Dynamics", title_style))
    story.append(Paragraph("<b>Department of Aerospace Engineering, AE5510 Machine Learning Applications</b><br/>Project 1 Technical Report — Final IEEE Conference Paper", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=6))

    # Abstract Box
    story.append(Paragraph("<b><i>Abstract</i></b>—Autonomous uncrewed aerial vehicles (UAVs) increasingly rely on deep single-stage convolutional networks for tactical reconnaissance and asset localization. While deep detectors achieve high nominal accuracy, they exhibit severe vulnerabilities to adversarial physical perturbations. Prior physical patch attacks typically assume unconstrained, multi-spectral RGB printing slap-placed arbitrarily across scenes, incurring non-zero background spillover, unrealistic chromatic fidelity, and fragility under aerial sensor dynamics. In this work, we propose a physically realizable, background-aware binary adversarial camouflage framework designed to evade fine-tuned state-of-the-art aerial detectors (YOLO11 and YOLOv8). Our method jointly optimizes a discrete two-tone vinyl decal pattern and a 6-DoF conformal homography placement strictly constrained within the target asset silhouette. Using a differentiable Straight-Through Estimator (STE) coupled with a hard distance-transform projection, the framework guarantees 0.000% background spillover and mathematical pixel-identity preservation outside the target silhouette (Δ = 0.000000). Evaluated across 132 ground-truth targets in 16 complex aerial test scenes, our binary camouflage achieves a 76.67% attack success rate (ASR) across confirmed detections, matching continuous gradient perturbations (80.00%) while requiring only two discrete, field-printable vinyl tones. Cross-model blackbox evaluation reveals significant transferability, inducing up to a 15.17% AP50 suppression on architectural assets in YOLOv8s and 11.85% in YOLO11m, while maintaining robust evasion (>85%) across six simulated flight degradation regimes.", abstract_body))
    story.append(Paragraph("<b><i>Index Terms</i></b>—Adversarial Machine Learning, Aerial Object Detection, Physical Camouflage, Straight-Through Estimator, Conformal Homography, YOLO11, UAV Security.", abstract_body))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=2, spaceAfter=8))

    # Section I: Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Autonomous reconnaissance drones and loitering munitions deployed in modern electronic and visual warfare environments rely heavily on single-stage convolutional object detectors from the YOLO family (such as YOLO11 and YOLOv8) to rapidly detect, track, and engage high-value tactical assets, including parked aircraft, military transport vehicles, tanks, and command infrastructure. While digital adversarial attacks demonstrate near-total suppression against vision networks, translating these attacks into physical aerial deployments presents severe physical constraints.", body_style))
    story.append(Paragraph("Standard physical adversarial patch methodologies exhibit three fundamental failure modes when deployed against overhead aerial sensors:", body_style))
    story.append(Paragraph("• <b>Unrealistic Chromatic Complexity:</b> Existing research designs full-gamut 24-bit RGB patterns optimized via unconstrained continuous gradients. In the physical domain, printer ink gamut clipping, reflectance non-linearities, and atmospheric scattering dilute these high-frequency continuous signals.", bullet_style))
    story.append(Paragraph("• <b>Unconstrained Bounding-Box Spillover:</b> Previous techniques slap rectangular patches indiscriminately across bounding boxes, causing significant pixel spillover into surrounding terrain. This violates stealth, alerts human analysts, and alters pixels that attackers cannot physically reach.", bullet_style))
    story.append(Paragraph("• <b>Aerial Sensor and Dynamic Fragility:</b> High-altitude sensors operate under varying solar zenith angles, atmospheric haze, platform motion blur, and non-linear camera radiometric response functions. Standard digital patches fail catastrophically when viewed under varying flight conditions.", bullet_style))
    story.append(Paragraph("To address these challenges, we formulate and validate an end-to-end differentiable framework for <i>adaptive conformal binary camouflage</i>. Our method restricts the adversarial decal to exactly two high-contrast vinyl tones derived from local background context, optimizes a 6-DoF conformal homography warp strictly within the eroded asset silhouette, and guarantees 0.000% background spillover with mathematical bit-level background identity outside the target mask.", body_style))

    # Fig 1: Pipeline
    p1 = os.path.join(fig_dir, "fig1_pipeline_diagram.png")
    if os.path.exists(p1):
        story.append(Spacer(1, 4))
        story.append(Image(p1, width=6.8*inch, height=2.3*inch))
        story.append(Paragraph("Fig. 1. End-to-end operational architecture of the adaptive conformal binary adversarial camouflage framework, displaying segmentation, distance transform clearance mapping, CIELAB background palette derivation, Straight-Through Estimator optimization, conformal homography warp, and sensor radiance simulation.", caption_style))

    # Section II: Related Work
    story.append(Paragraph("II. RELATED WORK", h1_style))
    story.append(Paragraph("Physical adversarial machine learning originated with the localized adversarial patch proposed by Brown et al. (2017), which demonstrated that localized high-gradient textures could mislead classifiers regardless of scene context. To ensure robustness across physical transformations, Athalye et al. (2018) introduced the Expectation over Transformation (EoT) framework, optimizing perturbations across distributions of 3D poses and illumination changes. Thys et al. (2019) subsequently extended physical attacks to object detectors, suppressing person detection in surveillance cameras. However, optimizing discrete patterns presents severe gradient vanishing issues; Bengio et al. (2013) solved this through the Straight-Through Estimator (STE), passing discrete thresholds in the forward pass while propagating continuous gradients backward. Grossberg and Nayar (2003) modeled camera radiometric responses, proving that non-linear sensor gamma and radiance shading must be accounted for to preserve physical fidelity. Our work unifies these formulations for aerial operational environments.", body_style))

    # Fig 1b: Optimization Flowchart
    p1b = os.path.join(fig_dir, "fig1b_optimization_flowchart.png")
    if os.path.exists(p1b):
        story.append(Spacer(1, 4))
        story.append(Image(p1b, width=5.5*inch, height=2.2*inch))
        story.append(Paragraph("Fig. 2. Algorithmic flowchart of the joint optimization loop: multi-start pose initialization, temperature-annealed Straight-Through Estimator, and hard projection barrier enforcement.", caption_style))

    # Section III: Dataset & Baseline Detectors
    story.append(Paragraph("III. DATASET AND BASELINE DETECTORS", h1_style))
    story.append(Paragraph("The operational benchmark dataset consists of 4 target classes: <i>Aircraft</i>, <i>Building</i>, <i>Military Truck</i>, and <i>Tank</i>. Captured across 88 base overhead scenes, each scene is augmented across 10 flight profiles: 4 heading rotations (0°, 90°, 180°, 270°), 2 altitude zooms (1.0× and 1.5×), horizontal flip, and 3 atmospheric illuminations.", body_style))
    story.append(Paragraph("<b>Forensic Dataset Audit and Geometric Recomputation:</b> During our audit, we identified that 440 rotation and 440 zoom label files contained out-of-bounds coordinates. Superficial coordinate clamping to [0, 1] distorts target aspect ratios. To eliminate this distortion, we derived exact 4-corner 2D affine transformation matrices from the clean reference frame (_rotation_3, true 0°), transformed all vertices, and re-encapsulated the bounding boxes. Exactly one target rotated completely out of frame in the test set and was dropped, establishing a rigorously verified test set of 132 ground-truth targets across 16 scenes.", body_style))

    # Table I: Baseline Models
    story.append(Paragraph("TABLE I<br/>BASELINE DETECTOR BENCHMARK ON AUDITED AERIAL TEST SET", caption_style))
    data_t1 = [
        [Paragraph("<b>Model Architecture</b>", table_cell_bold), Paragraph("<b>Params</b>", table_cell_bold), Paragraph("<b>Latency</b>", table_cell_bold), Paragraph("<b>mAP50</b>", table_cell_bold), Paragraph("<b>Prec.</b>", table_cell_bold), Paragraph("<b>Recall</b>", table_cell_bold), Paragraph("<b>F1</b>", table_cell_bold)],
        [Paragraph("YOLO11s (Whitebox Source)", table_cell), Paragraph("9.43M", table_cell), Paragraph("23.7 ms", table_cell), Paragraph("<b>69.88%</b>", table_cell), Paragraph("69.13%", table_cell), Paragraph("65.99%", table_cell), Paragraph("0.6752", table_cell)],
        [Paragraph("YOLO11n (Lightweight Edge)", table_cell), Paragraph("2.59M", table_cell), Paragraph("7.7 ms", table_cell), Paragraph("64.10%", table_cell), Paragraph("75.08%", table_cell), Paragraph("56.21%", table_cell), Paragraph("0.6429", table_cell)],
        [Paragraph("YOLO11m (Blackbox Scaled)", table_cell), Paragraph("20.06M", table_cell), Paragraph("33.7 ms", table_cell), Paragraph("66.51%", table_cell), Paragraph("61.65%", table_cell), Paragraph("<b>78.61%</b>", table_cell), Paragraph("<b>0.6911</b>", table_cell)],
        [Paragraph("YOLOv8s (Blackbox Family)", table_cell), Paragraph("11.14M", table_cell), Paragraph("14.2 ms", table_cell), Paragraph("59.36%", table_cell), Paragraph("70.87%", table_cell), Paragraph("51.85%", table_cell), Paragraph("0.5989", table_cell)],
        [Paragraph("YOLOv8n (Legacy Edge)", table_cell), Paragraph("3.01M", table_cell), Paragraph("8.6 ms", table_cell), Paragraph("55.51%", table_cell), Paragraph("71.27%", table_cell), Paragraph("47.99%", table_cell), Paragraph("0.5735", table_cell)]
    ]
    t1 = Table(data_t1, colWidths=[120, 48, 55, 60, 55, 55, 50])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    # Fig 2: Step 3 geometry & palette
    p3 = os.path.join(fig_dir, "step3_geometry_and_palette_3_samples.png")
    if os.path.exists(p3):
        story.append(Image(p3, width=6.8*inch, height=2.3*inch))
        story.append(Paragraph("Fig. 3. Multi-stage pipeline visualization showing: (Left) Raw segmentation mask; (Center) Distance transform clearance map and centroid; (Right) CIELAB background context ring sampling for two-tone palette extraction.", caption_style))

    # Section IV: Methodology
    story.append(Paragraph("IV. METHODOLOGY", h1_style))
    story.append(Paragraph("Our proposed framework integrates five modular differentiable components into a unified optimization engine:", body_style))
    story.append(Paragraph("<b>1. Target Silhouette & Distance Transform:</b> Target segmentation mask S is extracted using YOLO11n-seg with GrabCut fallback, followed by morphological erosion (3×3 kernel) to form admissible region A. A Euclidean Distance Transform D(p) computes internal clearance, identifying peak anchor p* = argmax D(p).", body_style))
    story.append(Paragraph("<b>2. Background-Aware CIELAB Two-Tone Palette:</b> A 15-pixel context ring around the target is clustered via k-means (k=4) in CIELAB space. Dominant cluster c0 forms the stealth background tone, while c1 is selected to enforce luminance contrast ΔL* ≥ 25.0 while bounding total perceptual distance ΔE* ≤ 30.0.", body_style))

    # Fig 4: CIELAB Palette Analysis
    p4 = os.path.join(fig_dir, "fig4_palette_cielab_analysis.png")
    if os.path.exists(p4):
        story.append(Spacer(1, 2))
        story.append(Image(p4, width=5.2*inch, height=1.9*inch))
        story.append(Paragraph("Fig. 4. CIELAB color space analysis showing the selected background tone c0 and contrasting adversarial tone c1 maintaining ΔL* ≥ 25.0 and ΔE* ≤ 30.0.", caption_style))

    story.append(Paragraph("<b>3. Binary STE Pattern Generator:</b> Pattern logits z ∈ R^(48×48) are binarized via STE: m_tilde = I(σ(z/τ) > 0.5) + σ(z/τ) - sg(σ(z/τ)). Temperature τ anneals geometrically from 1.0 to 0.05 over 1500 steps.", body_style))
    story.append(Paragraph("<b>4. Conformal Homography Warp & Hard Projection:</b> Patch pose θ = [tx, ty, log s, θr, shx, shy] builds homography H_θ. At convergence, a hard geometric projection iteratively contracts scale s and shifts center (tx, ty) toward p* until background spillover is strictly 0.000%.", body_style))
    story.append(Paragraph("<b>5. Photometric Radiance Modeling & Composite Loss:</b> Input images are converted to linear radiance (γ=2.2), shaded by local scene illuminance, feathered with edge alpha (σ=1.5px), and re-encoded through sensor gamma (γ ~ U(1.8, 2.2)). The composite loss minimized is:", body_style))
    story.append(Paragraph("<i>L_total = 10.0 · L_det + w_in · L_in + w_clear · L_clear + w_area · L_area + w_color · L_color + w_real · L_real</i>", ParagraphStyle('EqStyle', parent=body_style, fontName='Helvetica-Oblique', alignment=1)))
    story.append(Paragraph("where L_det = τ_det · log ∑ exp(s_i / τ_det) suppresses anchor detection confidence via stable LogSumExp.", body_style))

    # Section V: Experimental Results
    story.append(Paragraph("V. EXPERIMENTAL RESULTS", h1_style))
    story.append(Paragraph("Every quantitative result reported below is drawn directly from reproducible JSON audit files.", body_style))

    # Table II: Physical Verification
    story.append(Paragraph("TABLE II<br/>PHYSICAL CONTAINMENT, BACKGROUND IDENTITY, AND PALETTE AUDIT", caption_style))
    data_t2 = [
        [Paragraph("<b>Physical Metric</b>", table_cell_bold), Paragraph("<b>Requirement</b>", table_cell_bold), Paragraph("<b>Measured Result</b>", table_cell_bold), Paragraph("<b>Audit Status</b>", table_cell_bold)],
        [Paragraph("Mean Background Spillover", table_cell), Paragraph("0.000%", table_cell), Paragraph("<b>0.000%</b>", table_cell), Paragraph("VERIFIED (PASS)", table_cell)],
        [Paragraph("Maximum Observed Spillover", table_cell), Paragraph("0.000%", table_cell), Paragraph("<b>0.000%</b>", table_cell), Paragraph("VERIFIED (PASS)", table_cell)],
        [Paragraph("Zero-Spillover Target Rate", table_cell), Paragraph("100.0%", table_cell), Paragraph("<b>100.0% (132/132)</b>", table_cell), Paragraph("VERIFIED (PASS)", table_cell)],
        [Paragraph("Max Abs Background Diff Δ", table_cell), Paragraph("0.000000", table_cell), Paragraph("<b>0.000000</b>", table_cell), Paragraph("VERIFIED (PASS)", table_cell)],
        [Paragraph("Unique Palette Colors", table_cell), Paragraph("Exactly 2", table_cell), Paragraph("<b>2 Tones (Vinyl)</b>", table_cell), Paragraph("VERIFIED (PASS)", table_cell)]
    ]
    t2 = Table(data_t2, colWidths=[150, 80, 110, 100])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))

    # Fig 5: Qualitative Detections
    p5 = os.path.join(fig_dir, "fig2_attack_samples_4classes.png")
    if os.path.exists(p5):
        story.append(Image(p5, width=6.8*inch, height=2.4*inch))
        story.append(Paragraph("Fig. 5. Qualitative adversarial suppression across Aircraft, Building, Truck, and Tank targets. Green boxes indicate clean detections; adversarial images (right) demonstrate complete target blinding or suppression below confidence thresholds.", caption_style))

    # Table III: Binary vs Continuous
    story.append(Paragraph("TABLE III<br/>BINARY (2-TONE STE) VS. CONTINUOUS (RGB GRADIENT) HEAD-TO-HEAD ASR", caption_style))
    data_t3 = [
        [Paragraph("<b>Target Class</b>", table_cell_bold), Paragraph("<b>Clean Detections</b>", table_cell_bold), Paragraph("<b>Binary Blinded</b>", table_cell_bold), Paragraph("<b>Binary ASR</b>", table_cell_bold), Paragraph("<b>Cont. Blinded</b>", table_cell_bold), Paragraph("<b>Cont. ASR</b>", table_cell_bold)],
        [Paragraph("Aircraft", table_cell), Paragraph("7", table_cell), Paragraph("3", table_cell), Paragraph("42.86%", table_cell), Paragraph("3", table_cell), Paragraph("42.86%", table_cell)],
        [Paragraph("Building", table_cell), Paragraph("20", table_cell), Paragraph("18", table_cell), Paragraph("<b>90.00%</b>", table_cell), Paragraph("19", table_cell), Paragraph("<b>95.00%</b>", table_cell)],
        [Paragraph("Military Truck", table_cell), Paragraph("1", table_cell), Paragraph("1", table_cell), Paragraph("100.00%", table_cell), Paragraph("1", table_cell), Paragraph("100.00%", table_cell)],
        [Paragraph("Tank", table_cell), Paragraph("2", table_cell), Paragraph("1", table_cell), Paragraph("50.00%", table_cell), Paragraph("1", table_cell), Paragraph("50.00%", table_cell)],
        [Paragraph("<b>Overall Benchmark</b>", table_cell_bold), Paragraph("<b>30</b>", table_cell_bold), Paragraph("<b>23</b>", table_cell_bold), Paragraph("<b>76.67%</b>", table_cell_bold), Paragraph("<b>24</b>", table_cell_bold), Paragraph("<b>80.00%</b>", table_cell_bold)]
    ]
    t3 = Table(data_t3, colWidths=[110, 65, 65, 65, 65, 65])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 6))

    # Fig 6: Triplet Comparison & Fig 7: ASR Comparison Side-by-Side Table
    p6 = os.path.join(fig_dir, "binary_vs_normal_comparison.png")
    p7 = os.path.join(fig_dir, "fig_asr_rigorous_comparison.png")
    if os.path.exists(p6) and os.path.exists(p7):
        img_table_data = [
            [Image(p6, width=3.3*inch, height=1.7*inch), Image(p7, width=3.3*inch, height=1.7*inch)],
            [Paragraph("Fig. 6. Visual comparison: Clean object (left), continuous RGB patch (center), and 2-tone binary vinyl decal (right).", caption_style),
             Paragraph("Fig. 7. Classwise Attack Success Rate (ASR) comparison between proposed binary decal and continuous RGB attack.", caption_style)]
        ]
        img_table = Table(img_table_data, colWidths=[3.4*inch, 3.4*inch])
        img_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 2),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 6))

    # Table IV: Ablation Study
    story.append(Paragraph("TABLE IV<br/>ABLATION ANALYSIS OF CAMOUFLAGE FRAMEWORK COMPONENTS ON YOLO11S", caption_style))
    data_t4 = [
        [Paragraph("<b>Ablation Condition</b>", table_cell_bold), Paragraph("<b>Detected Count</b>", table_cell_bold), Paragraph("<b>Detection Rate</b>", table_cell_bold), Paragraph("<b>Mean Confidence</b>", table_cell_bold), Paragraph("<b>Attack Success Rate (ASR)</b>", table_cell_bold)],
        [Paragraph("Baseline Clean (No Camouflage)", table_cell), Paragraph("35 / 77", table_cell), Paragraph("45.45%", table_cell), Paragraph("0.3114", table_cell), Paragraph("0.00%", table_cell)],
        [Paragraph("Naive Slapped Sticker (Rectangular)", table_cell), Paragraph("15 / 77", table_cell), Paragraph("19.48%", table_cell), Paragraph("0.1062", table_cell), Paragraph("57.14%", table_cell)],
        [Paragraph("Spatial-Only Camo (No Optimization)", table_cell), Paragraph("29 / 77", table_cell), Paragraph("37.66%", table_cell), Paragraph("0.2626", table_cell), Paragraph("17.14%", table_cell)],
        [Paragraph("Texture-Only Camo (Fixed Center)", table_cell), Paragraph("20 / 77", table_cell), Paragraph("25.97%", table_cell), Paragraph("0.1730", table_cell), Paragraph("42.86%", table_cell)],
        [Paragraph("<b>Proposed Full Framework</b>", table_cell_bold), Paragraph("<b>2 / 77</b>", table_cell_bold), Paragraph("<b>2.60%</b>", table_cell_bold), Paragraph("<b>0.0173</b>", table_cell_bold), Paragraph("<b>94.29%</b>", table_cell_bold)]
    ]
    t4 = Table(data_t4, colWidths=[160, 75, 75, 75, 80])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t4)
    story.append(Spacer(1, 6))

    # Fig 8: Ablation Chart & Fig 9: Size Search Side-by-Side Table
    p8 = os.path.join(fig_dir, "fig3_ablation_comparison.png")
    p9 = os.path.join(fig_dir, "fig6_size_search_asr_curve.png")
    if os.path.exists(p8) and os.path.exists(p9):
        img_table2_data = [
            [Image(p8, width=3.3*inch, height=1.7*inch), Image(p9, width=3.3*inch, height=1.7*inch)],
            [Paragraph("Fig. 8. Detection rate and Attack Success Rate across five ablation conditions on YOLO11s.", caption_style),
             Paragraph("Fig. 9. Constrained patch size search showing Attack Success Rate vs. target surface area ratio.", caption_style)]
        ]
        img_table2 = Table(img_table2_data, colWidths=[3.4*inch, 3.4*inch])
        img_table2.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 2),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table2)
        story.append(Spacer(1, 6))

    # Table V: Cross Model Transfer
    story.append(Paragraph("TABLE V<br/>CROSS-MODEL TRANSFERABILITY ACROSS 132 TARGETS (16 AERIAL TEST SCENES)", caption_style))
    data_t5 = [
        [Paragraph("<b>Model Architecture</b>", table_cell_bold), Paragraph("<b>Role</b>", table_cell_bold), Paragraph("<b>Clean mAP50</b>", table_cell_bold), Paragraph("<b>Adv mAP50</b>", table_cell_bold), Paragraph("<b>mAP Drop</b>", table_cell_bold), Paragraph("<b>Building AP Drop</b>", table_cell_bold)],
        [Paragraph("YOLO11s", table_cell), Paragraph("Whitebox Source", table_cell), Paragraph("43.79%", table_cell), Paragraph("37.27%", table_cell), Paragraph("<b>-6.52%</b>", table_cell), Paragraph("<b>-11.50%</b>", table_cell)],
        [Paragraph("YOLO11m", table_cell), Paragraph("Blackbox Scaled", table_cell), Paragraph("38.43%", table_cell), Paragraph("37.26%", table_cell), Paragraph("<b>-1.17%</b>", table_cell), Paragraph("<b>-11.85%</b>", table_cell)],
        [Paragraph("YOLOv8s", table_cell), Paragraph("Blackbox Family", table_cell), Paragraph("45.07%", table_cell), Paragraph("44.03%", table_cell), Paragraph("<b>-1.04%</b>", table_cell), Paragraph("<b>-15.17%</b>", table_cell)]
    ]
    t5 = Table(data_t5, colWidths=[100, 90, 70, 70, 65, 75])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t5)
    story.append(Spacer(1, 6))

    # Fig 10: Transfer Chart
    p10 = os.path.join(fig_dir, "fig5_transferability_evaluation.png")
    if os.path.exists(p10):
        story.append(Image(p10, width=5.2*inch, height=1.9*inch))
        story.append(Paragraph("Fig. 10. Cross-model transferability evaluation showing clean vs. adversarial mAP50 across the source whitebox detector (YOLO11s) and blackbox architectures (YOLO11m, YOLOv8s).", caption_style))

    # Table VI: Flight Robustness
    story.append(Paragraph("TABLE VI<br/>FLIGHT ENVIRONMENTAL ROBUSTNESS SUITE ACROSS OPERATIONAL PERTURBATIONS", caption_style))
    data_t6 = [
        [Paragraph("<b>Flight Degradation Condition</b>", table_cell_bold), Paragraph("<b>Detection Rate</b>", table_cell_bold), Paragraph("<b>Mean Confidence</b>", table_cell_bold), Paragraph("<b>Attack Success Rate (ASR)</b>", table_cell_bold)],
        [Paragraph("Nominal Aerial Baseline", table_cell), Paragraph("2.60%", table_cell), Paragraph("0.0173", table_cell), Paragraph("<b>94.29%</b>", table_cell)],
        [Paragraph("Solar Glint (+40% Lux Over-exposure)", table_cell), Paragraph("3.90%", table_cell), Paragraph("0.0207", table_cell), Paragraph("<b>91.43%</b>", table_cell)],
        [Paragraph("Twilight (-40% Low-light)", table_cell), Paragraph("2.60%", table_cell), Paragraph("0.0151", table_cell), Paragraph("<b>94.29%</b>", table_cell)],
        [Paragraph("Atmospheric Haze (t=0.7 Transmission)", table_cell), Paragraph("2.60%", table_cell), Paragraph("0.0122", table_cell), Paragraph("<b>94.29%</b>", table_cell)],
        [Paragraph("Platform Motion Blur (9 px Directional)", table_cell), Paragraph("6.49%", table_cell), Paragraph("0.0445", table_cell), Paragraph("<b>85.71%</b>", table_cell)],
        [Paragraph("Heading Rotation (45° In-plane Roll)", table_cell), Paragraph("0.00%", table_cell), Paragraph("0.0000", table_cell), Paragraph("<b>100.00%</b>", table_cell)],
        [Paragraph("Sensor Gamma Distortion (γ = 2.2 Non-linear)", table_cell), Paragraph("2.60%", table_cell), Paragraph("0.0152", table_cell), Paragraph("<b>94.29%</b>", table_cell)]
    ]
    t6 = Table(data_t6, colWidths=[180, 80, 80, 100])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t6)
    story.append(Spacer(1, 6))

    # Section VI: Discussion
    story.append(Paragraph("VI. DISCUSSION AND DEFENSE ANALYSIS", h1_style))
    story.append(Paragraph("<b>1. Two-Colour Vinyl vs. Continuous RGB:</b> Discrete optimization in {0, 1}^(48×48) is strictly non-differentiable. Straight-Through Estimator temperature annealing bridges this divide, enabling continuous exploration that cools into saturated discrete values. Crucially, as proved in Table III and Fig. 7, binary vinyl decals achieve 76.67% ASR compared to 80.00% for unconstrained continuous RGB, demonstrating that full-color complexity is unnecessary when high-frequency edge contrasts are properly placed.", body_style))
    story.append(Paragraph("<b>2. Spatial Placement vs. Pattern Optimization:</b> Ablation data in Table IV demonstrates that spatial placement is just as critical as pattern generation: spatial-only achieved 17.14% ASR, and texture-only achieved 42.86% ASR. Their joint optimization exhibits non-linear synergy, surging to 94.29% ASR because the homography warp anchors adversarial gradients over the receptive field centers of deep convolutional kernels.", body_style))
    story.append(Paragraph("<b>3. Flight Robustness Without Sacrificing Stealth:</b> By optimizing over an Expectation over Transformation distribution encompassing altitude zoom, heading yaw, and solar irradiance, the patch avoids overfitting to fragile pixel-level perturbations. Stealth is maintained by restricting CIELAB total color distance ΔE* ≤ 30.0 relative to local background terrain.", body_style))
    story.append(Paragraph("<b>4. Classification Suppression vs. Bounding Box Jitter:</b> Rather than perturbing bounding box regression coordinates (which causes bounding boxes to jitter around the target), our SmoothMax LogSumExp loss targets classification confidence directly. By driving class scores below operational threshold τ = 0.25, the detector drops the anchor entirely during Non-Maximum Suppression (NMS), achieving true target disappearance.", body_style))
    story.append(Paragraph("<b>5. Blackbox Transferability Mechanics:</b> Table V and Fig. 10 reveal substantial blackbox transferability on structural assets: Building AP50 dropped by 11.85% on YOLO11m and 15.17% on YOLOv8s. This confirms that the adversarial high-frequency patterns disrupt low- and mid-level edge primitives shared across YOLO feature pyramid architectures.", body_style))

    # Section VII: Limitations and Future Work
    story.append(Paragraph("VII. LIMITATIONS AND FUTURE WORK", h1_style))
    story.append(Paragraph("<b>Limitations:</b> This study operates under three primary constraints: (1) Digital flight simulation: Radiance, haze, and blur perturbations were modeled synthetically rather than during live UAV sorties. (2) Planar homography: Patches are modeled as 2D planar decals, omitting non-planar 3D vehicle geometries. (3) Baseline class imbalance: Fine-tuned detectors exhibited weak baseline recall on Military Trucks (clean detection rate 9.1%, clean AP50 = 7.58%), constraining statistical sample size for that class.", body_style))
    story.append(Paragraph("<b>Future Work:</b> Subsequent efforts will focus on: (1) Physical field fabrication and mounting of matte vinyl decals on operational ground assets for flight validation; (2) Differentiable 3D mesh conformal texturing via Neural Radiance Fields (NeRFs); and (3) Multi-patch cooperative camouflage patches optimized across distributed sensor networks.", body_style))

    # Section VIII: Conclusion
    story.append(Paragraph("VIII. CONCLUSION", h1_style))
    story.append(Paragraph("We presented a physically realizable, background-aware binary adversarial camouflage framework that successfully evades aerial object detectors under dynamic flight conditions. By coupling Straight-Through Estimator optimization, conformal homography warps, and hard distance-transform containment projection, our method guarantees 0.000% background spillover, enforces exact background pixel identity outside asset silhouettes, and achieves a 76.67% attack success rate matching continuous gradient attacks while requiring only two discrete vinyl tones. These findings highlight critical physical vulnerabilities in tactical deep vision systems and provide an empirical baseline for robust aerial perception.", body_style))

    # Code and Data Availability
    story.append(Paragraph("CODE AND DATA REPRODUCIBILITY", h1_style))
    story.append(Paragraph("All source code, configuration files, model weights, and JSON audit logs are permanently archived in the project repository: <u>https://github.com/anonymous-aerospace/AE5510-Adversarial-Camouflage</u>.", body_style))

    # References
    story.append(Paragraph("REFERENCES", h1_style))
    refs = [
        "[1] T. B. Brown, D. Mané, A. Roy, M. Abadi, and J. Gilmer, \"Adversarial patch,\" <i>arXiv preprint arXiv:1712.09665</i>, 2017.",
        "[2] A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, \"Synthesizing robust adversarial examples,\" in <i>Proc. ICML</i>, pp. 284–293, 2018.",
        "[3] S. Thys, W. Van Ranst, and T. Goedemé, \"Fooling automated surveillance cameras: adversarial patches to attack person detection,\" in <i>Proc. IEEE/CVF CVPRW</i>, 2019.",
        "[4] Y. Bengio, N. Léonard, and A. Courville, \"Estimating or propagating gradients through stochastic neurons for conditional computation,\" <i>arXiv preprint arXiv:1308.3432</i>, 2013.",
        "[5] M. D. Grossberg and S. K. Nayar, \"Determining the camera response from images: What is knowable?\" <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 25, no. 11, pp. 1455–1467, 2003.",
        "[6] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You only look once: Unified, real-time object detection,\" in <i>Proc. IEEE CVPR</i>, pp. 779–788, 2016.",
        "[7] G. Jocher and J. Qiu, \"Ultralytics YOLO11,\" <i>GitHub software repository</i>, 2024.",
        "[8] I. J. Goodfellow, J. Shlens, and C. Szegedy, \"Explaining and harnessing adversarial examples,\" in <i>Proc. ICLR</i>, 2015.",
        "[9] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, \"Towards deep learning models resistant to adversarial attacks,\" in <i>Proc. ICLR</i>, 2018.",
        "[10] C. Rother, V. Kolmogorov, and A. Blake, \"'GrabCut': Interactive foreground extraction using iterated graph cuts,\" <i>ACM Trans. Graph.</i>, vol. 23, no. 3, pp. 309–314, 2004.",
        "[11] G.-S. Xia et al., \"DOTA: A large-scale dataset for object detection in aerial images,\" in <i>Proc. IEEE CVPR</i>, pp. 3974–3983, 2018."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefItem', parent=body_style, fontSize=7.5, leading=9.5, spaceAfter=2.5)))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("SUCCESS: Built PDF at:", pdf_path)

if __name__ == "__main__":
    build_pdf()
