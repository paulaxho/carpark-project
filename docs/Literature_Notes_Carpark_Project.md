# Literature Notes — Carpark Project
<!-- Single source of truth. Copy the relevant template below for each new source you read. -->
<!-- Per-paper template now includes two added fields (see Core Papers): -->
<!--   "My critical take (beyond what they admit)" and "Design decision this justifies" -->

---

# SECTION 1 — Foundational Concepts
*Read these first to build your vocabulary*

---

## What Is a Convolutional Neural Network?

**Source:**
uk.mathworks.com/discovery/convolutional-neural-network

**Date Read:**
1 June 2026

**Definition in my own words:**
A CNN is a type of neural network designed to learn patterns directly from image data. It passes an image through many layers, each one picking up on increasingly complex features — starting from simple things like edges, building up to recognising whole objects.

**The bit that helped me understand it:**
The analogy of filters being applied at different resolutions, where each layer's output becomes the next layer's input — like peeling back layers of detail. Also the three key operations: mathworks

Convolution — scans the image for features using filters
ReLU — keeps only the useful signals, throws away negatives
Pooling — shrinks the data down so the network stays manageable

**How it connects to my project:**
CNNs are particularly useful for finding patterns in images to recognise objects, classes, and categories. YOLOv8 uses a CNN as its backbone — it's what actually "sees" the cars in my aerial images before the detection head draws the boxes. mathworks
The point about shared weights meaning the network can detect the same feature wherever it appears in the image is directly relevant — a car in the top-left corner of my aerial image should be detected just as well as one in the bottom-right. mathworks
Also important for my project: fine-tuning a pretrained network with transfer learning is typically much faster and easier than training from scratch, and requires the least amount of data and computational resources. This is exactly the approach I'll take — starting from YOLOv8's pretrained weights rather than training from zero

**One thing I'm still unsure about:** ❓

---

## What Is Object Detection?

**Source:** MathWorks — uk.mathworks.com/discovery/object-detection.html
**Date Read:** 1 June 2026

**Definition in my own words:**
Object detection is a computer vision technique for locating instances of objects in images or videos. It goes beyond simply asking "is there a car in this image?" — it also answers "where exactly is it?" and draws a bounding box around it.

**Difference between detection, classification and segmentation:**
|Task|What it does|Example|
|---|---|---|
|Classification|Is there a car? Yes/No|"This image contains a car"|
|Detection|Where is the car?|Bounding box drawn around it|
|Instance segmentation|Exact pixel outline of the car|Precise shape, not just a box|

Detection is what my project uses — bounding boxes around each vehicle.

You can either use pretrained object detectors that can detect common objects without further training, or create a custom object detector using transfer learning — building on a pretrained network and refining it for your specific application. mathworks
My project starts with option 1 (pretrained YOLOv8 — which gave 0 detections today), and will move to option 2 (fine-tuned on aerial imagery) as the next phase.

When choosing between machine learning and deep learning, consider whether you have a powerful GPU and lots of labelled training images. Deep learning techniques tend to work better when you have more images, and GPUs decrease the time needed to train the model. mathworks
This is a constraint I need to discuss with Toby — how much labelled training data can we realistically produce, and do I have GPU access?

**How it connects to my project:**
YOLO, SSD, and R-CNN are popular deep learning approaches using CNNs that automatically learn to detect objects within images. My project uses YOLOv8, which is the latest YOLO generation. This page confirms that YOLO is a standard, well-established choice for this type of task — not experimental.

**One thing I'm still unsure about:** ❓
What is the practical difference between YOLO, SSD, and R-CNN in terms of speed vs accuracy? Which would be best for aerial car detection specifically — worth asking Toby.

---

## Intersection over Union (IoU)

**Source:** Rosebrock, A. — PyImageSearch, pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection
**Date Read:** 1 June 2026

**Definition in my own words:**
IoU measures how well a predicted bounding box aligns with the ground truth box — the box a human has drawn around the actual object. It gives you a single number between 0 and 1 telling you how accurate the detection was.

**The formula (in plain English, not maths):**

Divide the overlapping area between the predicted box and the ground truth box by the total combined area of both boxes. Medium
Two squares drawn on paper — one by the human, one by the AI. IoU asks: "What fraction of the total area covered by both squares is the bit where they overlap?"

IoU = 0 → the boxes don't touch at all — completely wrong
IoU = 0.5 → half the combined area overlaps — acceptable
IoU = 1 → the boxes match perfectly — ideal

**What a good IoU score looks like:**
In standard object detection workflows, a threshold of 0.50 or 0.75 is commonly used — if the overlap score exceeds this number, the detection is counted as correct.

**How it connects to my project:**
Two places IoU appears in my pipeline:

During training — when fine-tuning YOLOv8, IoU tells the model how close its predicted boxes are to my hand-drawn labels. This is how the model learns to improve.
During evaluation — after training, IoU is used to decide whether each detection is a true positive or false positive, which feeds directly into mAP.

IoU is also used for duplicate removal — during inference, a model might identify the same object multiple times with slightly different boxes. This is called Non-Maximum Suppression (NMS) and it's why the iou=0.45 parameter existed in my test pipeline script.

---

## Mean Average Precision (mAP)

**Source:** <!-- Hui -->
**Date Read:**

**Definition in my own words:**

**Why we use mAP instead of just accuracy:**

**What a good mAP score looks like for object detection:**

**How it connects to my project:**

---

## Anchor Boxes

**Source:**
**Date Read:**

**Definition in my own words:**

**Why they matter for detecting small objects (like cars from above):**

**How it connects to my project:**

---

---

# SECTION 2 — Core Papers
*The academic literature directly relevant to your dissertation*

<!-- ===== PER-PAPER TEMPLATE (copy this block for each new paper) =====
## [Title]
**Author(s):**
**Year:**
**Date Read:**

**Problem this paper solves (1 sentence):**

**How they solved it (1-2 sentences):**

**Key results/numbers:**

**Dataset details:**
- Number of images:
- Resolution:
- Locations/cities covered:
- How cars were labelled:

**Relevant to my project because:**

**Limitations they mention:**            <- the authors' OWN admitted limitations

**My critical take (beyond what they admit):**   <- NEW: your own critique = critical-engagement marks

**Design decision this justifies:**              <- NEW: which choice of MINE does this back up?

**Possible citation:**

**One thing I'm still unsure about:** ❓
====================================================================== -->

---

## Predicting Car Park Occupancy Rates from Satellite Images with a Deep Learning Model

**Author(s):** Leblanc, M.; Van Dijk, J.; Rains, T.; Jack, H. (UCL × Sainsbury's, via the CDRC)
**Year:** 2021 *(verify — appears to be an MSc / CDRC project abstract, not a peer-reviewed article; cite accordingly)*
**Date Read:** 2 June 2026

**Problem this paper solves (1 sentence):**
Whether open-access satellite imagery plus a deep learning detector can count vehicles in UK supermarket car parks and estimate occupancy as a proxy for in-store retail intensity.

**How they solved it (1-2 sentences):**
Built a library of 110 Google Earth Engine satellite images across 19 Sainsbury's stores, cropped and tilted them, then ran a pretrained YOLOv4 (no domain retraining) to detect and count cars, and regressed detected counts against manually observed counts.

**Key results/numbers:**
Pipeline functional but severely under-detected. Detected counts explained only ~10.8% of observed vehicles (R² ≈ 0.108, per Figure 2). Wide variation in accuracy between sites; small sample (19 stores). Thresholds used: conf = 0.1, NMS = 0.3.

**Dataset details:**
- Number of images: 110
- Resolution: ~0.6–1.0 m/pixel *(coarse — far lower than my ~25 cm Getmapping aerial)*
- Locations/cities covered: 19 Sainsbury's stores across the UK
- Source: Google Earth Engine (open access, free)
- How cars were labelled: manual observed vehicle counts used as ground truth for the regression

**Relevant to my project because:**
Closest UK precedent to my dissertation — same task, same retail-proxy framing, same model family (YOLO), and a shared retailer (Sainsbury's, one of my site types). Gives me an industry-driven justification for the whole premise (car park occupancy ↔ in-store trade) for my business-context section.

**Limitations they mention:**
Imagery resolution/format unsuitable for the model; high number of clustered/occluded cars; small sample (not conclusive); pretrained model not adapted to the domain; manual steps hurt reproducibility.

**My critical take (beyond what they admit):**
Detection and counting are conflated — only a count-regression R² is reported, with no precision/recall/mAP, so detection quality and counting quality can't be separated. The abstract also gives two incompatible R² values for the same relationship (≈0.108 in the figure/text vs 0.59 elsewhere), so the headline fit is ambiguous — do NOT cite the 0.59 uncritically. "Open-access" Google Earth imagery has unknown/variable capture dates, which undermines temporal comparability.

**Design decision this justifies:**
- Using ~25 cm Getmapping aerial imagery instead of coarse open satellite → directly targets their main failure cause. *(My core positioning.)*
- Resolution-matching COWC/VEDAI + optional fine-tuning → they failed using pretrained YOLO off-the-shelf.
- Reporting precision/recall/F1/mAP AND MAE/percentage error separately → fixes their conflation and delivers the evaluation they called for as future work.
- Using YOLOv8 over their YOLOv4 (2021) → "building on and improving prior UK work."

**Possible citation:**
Leblanc et al. (2021) — UCL–Sainsbury's study framing satellite-derived car park occupancy as an indicator of in-store retail intensity *(confirm citation details / find fuller version).*

**One thing I'm still unsure about:** ❓
Is there a fuller dissertation behind this abstract that reports proper detection metrics (precision/recall/mAP) I could contrast my results against?

---

## A Large Contextual Dataset for Classification, Detection and Counting of Cars with Deep Learning (COWC)

**Author(s):** Mundhenk, T.N.; Konjevod, G.; Sakla, W.A.; Boakye, K. (Lawrence Livermore National Laboratory)
**Year:** 2016 (ECCV 2016, Part III, LNCS 9907, pp. 785–800; DOI 10.1007/978-3-319-46487-9_48)
**Date Read:** 2 June 2026

> NB: Like VEDAI, this paper IS the citation for one of my training datasets (COWC) — core paper + dataset reference.

**Problem this paper solves (1 sentence):**
Existing overhead-car datasets were too small and too narrow — OIRDS (180 cars) and VEDAI (2,950 cars) cover essentially one region / one sensor — so there was no large, diverse, deliberately difficult public dataset for training deep learners to classify, detect and count cars from above.

**How they solved it (1-2 sentences):**
Built COWC (Cars Overhead With Context): 32,716 unique cars from six geographically distinct sources captured by different imagers, plus 58,247 hand-picked hard negatives (boats, trailers, bushes, A/C units) and surrounding context. They also introduce a "ResCeption" network and a one-look counting method (count cars in a patch directly, then scan with a large stride) — counting without localising.

**Key results/numbers:**
Classification ~99% correct (ResCeption 99.14%). Detection on held-out 2048×2048 scenes: F-score ≈94.3%. One-look scene counting: best model ≈5–6% mean absolute error (down to ~5.15% with offset-averaged strides), with very low total error (~0.2–0.5%) over 20 scenes; counts ~1 km²/sec (AlexNet). Cars span 24–48 px at the standardised resolution.

**Dataset details:**
- Number of images: 32,716 unique cars + 58,247 negatives; ~309k training / ~79k testing classification patches (256×256); held-out test = 2048×2048 scenes (~307 m across)
- Resolution: **15 cm/pixel** (standardised from mixed sources) ← sharper than my ~25 cm Getmapping
- Locations: Toronto (CA), Selwyn (NZ), Potsdam & Vaihingen (DE), Columbus & Utah (US) — 4 RGB, 2 grayscale. **No UK source.**
- How cars were labelled: **single-pixel DOT on each car centre — NOT bounding boxes.** Large trucks omitted; vans/pickups count as cars; boats/trailers/construction = negatives

**Relevant to my project because:**
This is my primary training set, and it directly fixes VEDAI's weaknesses: far larger (32,716 cars), genuinely multi-region / multi-sensor, and deliberately difficult with hard negatives. Crucially, **COWC DOES include dense parking-lot scenes** (the paper notes patches "necessarily overlap" in crowded lots, and held-out scenes include dense lots) — i.e. it covers the dense/occluded case VEDAI deliberately excluded. It also anchors my business context: the intro cites a commercial product counting cars in parking lots so investors can monitor retailers' business volume, and the conclusion frames counts as a proxy for shopping-centre footfall — exactly my premise.

**Limitations they mention:**
Counting has no perfect fix for cars split across stride boundaries (mitigated by stride tuning / offset averaging); detection error is inflated by splits and mergers; context only helps in ~1–2% of hard (occluded) cases; large trucks excluded by design.

**My critical take (beyond what they admit):**
- The networks here (AlexNet/Inception/ResCeption, one-look counting) are 2016 *classification/counting* models, not modern detectors — cite COWC as a **dataset**, treat its methods as historical context, not the approach I replicate (I use YOLOv8 detection).
- **Annotation mismatch:** dot-only labels mean I must convert points → bounding boxes to train YOLO. Box size is an assumption (cars 24–48 px, varied orientation), which injects label noise — a real preprocessing limitation to document.
- Their detection F-score is US 15 cm imagery scored with a bespoke "half-the-car-in-box" rule and fixed 48 px boxes — NOT comparable to my mAP@0.5/0.75 on 25 cm UK imagery.
- Six sources but none British → domain gap to UK retail persists (different car models, parking layouts, building styles) — reinforces why I still need my own UK validation set.
- 15 cm > my 25 cm, so downsampling needed (resolution matching again).

**Design decision this justifies:**
- **Why COWC as primary training data:** large, diverse, multi-sensor, deliberately hard with confounders — the strongest available justification, and it answers the gap I flagged: COWC, unlike VEDAI, contains dense car-park scenes, so training on it gives the model exposure to the occluded/clustered case.
- Add a **dot→bounding-box conversion** step to my data-prep methodology (fixed box sized to ~max car length).
- Downsample COWC 15 cm → ~25 cm to match my imagery (resolution matching).
- Include hard negatives in my own annotation, mirroring COWC's confounder strategy, to cut false positives.
- Second business-context citation (with Leblanc) for "car counts as a retail-activity proxy."
- Lets me contrast counting paradigms: detection-based counting (my YOLO approach, gives locations for occupancy mapping) vs COWC's one-look / density alternatives — material for my occupancy-estimation chapter.

**Possible citation:**
Mundhenk et al. (2016, ECCV) — introduces COWC, a large (32,716-car), geographically diverse, deliberately difficult overhead-car dataset at 15 cm, with contextual hard negatives; also demonstrates efficient one-look car counting.

**One thing I'm still unsure about:** ❓ What fixed box size should I use when converting COWC's centre-point dots to YOLO bounding boxes, given cars run 24–48 px and rotate?

---

## Vehicle Detection in Aerial Imagery: A Small Target Detection Benchmark (VEDAI)

**Author(s):** Razakarivony, S. & Jurie, F. (Sagem + University of Caen / CNRS / ENSICAEN)
**Year:** 2015 (HAL technical report; also published in *J. Visual Communication and Image Representation*, 2016 — *check which version to cite*)
**Date Read:** 2 June 2026

> NB: This paper IS the source/citation for the VEDAI dataset I use for training and benchmarking — so it does double duty (core paper + dataset reference).

**Problem this paper solves (1 sentence):**
There was no reproducible public benchmark for *small*-vehicle detection in aerial imagery — mainstream detection datasets (PASCAL VOC, ImageNet) contain large, centred objects, and the only prior aerial set (OIRDS) had no defined evaluation protocol, so results weren't comparable.

**How they solved it (1-2 sentences):**
Introduced VEDAI: a public aerial-imagery benchmark with multiple vehicle classes, several spectral bands and two resolutions, plus a precisely defined, reproducible evaluation protocol (fixed 10-fold cross-validation and defined metrics) and baseline results characterising how hard the task is.

**Key results/numbers:**
~1,210 images (1024×1024, with downscaled 512×512 duplicates); 3,700+ annotated targets; ~5.5 vehicles per image; vehicles occupy only ~0.7% of image pixels (i.e. genuinely tiny). Baselines (all *pre-deep-learning*: SVM+HOG/LBP/LTP, DPM, template matching, Hough forest) performed poorly — the authors conclude that on this data "none of the detectors gives usable results"; best 'car' result was recall ≈13% at 0.01 FPPI, i.e. most cars missed at a realistic operating point.

**Dataset details:**
- Number of images: ~1,210 (4 subsets: large/small × colour/infrared)
- Resolution: **12.5 cm/pixel** (1024×1024) and **25 cm/pixel** (512×512 downscaled) ← the 25 cm subset matches my Getmapping imagery
- Source/location: Utah AGRC HRO 2012 6-inch orthophotos, USA, spring 2012
- Channels: RGB colour + near-infrared
- Classes: 9 (plane, boat, camping car, car, pick-up, tractor, truck, van, other); meta-classes "small/large land vehicles"
- Labelling: per target — centre coords, orientation, 4 corner coords, class, occlusion flag, in-image flag
- **Deliberately EXCLUDED images with many vehicles (e.g. large car parks)** — see critical take

**Why small target detection is relevant to my project:**
At 25 cm a car is only a handful of pixels (~0.7% of the image here), so this paper is direct evidence that vehicle-from-above is a *small-target* problem, not ordinary object detection — it justifies a dedicated literature thread on small-object difficulty and supports choices like higher input resolution (`imgsz`) and resolution-matched training.

**How their approach compares to standard YOLO:**
No YOLO here — this is a 2015 paper, so the baselines are classical hand-crafted features + SVM (HOG/LBP) and DPM. Their matching criterion is also non-standard: a "centre-in-ellipse" test, not IoU/Jaccard. So their AP numbers are NOT directly comparable to the COCO-style mAP@0.5/0.75 I'll report.

**My critical take (beyond what they admit):**
- The "none of the detectors work" verdict is about *2015 hand-crafted methods*, not modern CNNs — I must NOT present it as the current state of the art, or I'll misrepresent the field. Frame it as: the task was hard for pre-deep-learning methods, motivating the CNN/YOLO approaches I use.
- **Biggest catch for me:** VEDAI deliberately excludes dense car parks because an algorithm guessing random positions would score well there. So VEDAI scenes are *sparse* (cars far apart; the authors note NMS rarely matters) — the opposite of my dense retail car parks, where clustering and occlusion are the core problem (exactly what sank Leblanc). VEDAI teaches the model what an overhead car looks like, but its scene statistics differ sharply from my deployment domain.
- Center-in-ellipse matching ≠ IoU, so comparability with my metrics is limited — note this when I cite their numbers.
- US imagery, single ground distance, no oblique views → the authors admit this makes it "easier" than operational imagery; another domain-gap caveat for UK retail.

**Design decision this justifies:**
- Using VEDAI for training/benchmarking — it's the standard, reproducible public small-vehicle aerial benchmark (strong "why VEDAI" justification).
- **Train/evaluate preferentially on VEDAI's 25 cm subset (or downsample the 12.5 cm set)** to match my ~25 cm Getmapping imagery — concrete, citable basis for my resolution-matching plan.
- Because VEDAI excludes dense parking lots, I *cannot* rely on it alone for the dense-occlusion problem → justifies both my own UK retail annotation set and using COWC (which includes contextual/denser scenes) alongside it.
- Their PR-curve / AP / recall-at-FPPI discussion and IoU/Jaccard treatment → background for my evaluation-metrics chapter.

**Possible citation:**
Razakarivony & Jurie (2015/2016) — introduces VEDAI, the standard reproducible benchmark for small-vehicle detection in aerial imagery; baseline results show the difficulty of the task for pre-deep-learning methods.

**One thing I'm still unsure about:** ❓ Should I cite the 2015 HAL technical report or the 2016 JVCIR journal version — which is the canonical reference?

---

## SOD-YOLO: Small-Object-Detection Algorithm Based on Improved YOLOv8 for UAV Images

**Author(s):** Li, Y.; Li, Q.; Pan, J.; Zhou, Y.; Zhu, H.; Wei, H.; Liu, C. (Qilu Aerospace Information Research Institute, Jinan + Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing)
**Year:** 2024 (*Remote Sensing* 16(16), 3057; DOI 10.3390/rs16163057; MDPI, open access CC BY. Received 9 Jul, published 20 Aug 2024)
**Date Read:** 2 June 2026

> NB: This is the **recent YOLOv8-era aerial paper** I flagged as missing in my Themes section — it directly fills the "modern detector" gap that Leblanc (YOLOv4, 2021), VEDAI (2015 classical) and COWC (2016 classification) leave open.

**Problem this paper solves (1 sentence):**
Stock YOLOv8 detects objects well from a horizontal viewpoint but performs poorly on UAV/drone imagery, where targets are tiny with little information, scale varies wildly across viewpoints (overhead/oblique/side), spatial distribution is dense/sparse/clustered, backgrounds are cluttered, and motion blur is common.

**How they solved it (1-2 sentences):**
Modified the YOLOv8 backbone and neck: (1) an **RFCBAM** downsampling module (receptive-field convolution + CBAM-style attention) replacing plain stride-2 conv, to extract features more efficiently and reduce the spatial-information loss that downsampling causes; and (2) a new neck, **BSSI-FPN** (Balanced Spatial and Semantic Information Fusion Pyramid Network), which leans on large-scale (high-resolution) feature maps, fuses across scales more frequently, and uses dynamic upsampling (DySample). Net effect: a higher-resolution detection path (outputs at 160×160, 80×80, 40×40) tuned for small targets, with far fewer parameters.

**Key results/numbers:** *(VisDrone2019; trained from scratch, no pretrained weights)*
- **SOD-YOLO-s:** mAP50 **42.0%** vs YOLOv8s 39.0% (**+3.0 pts**); params 11.1M → **1.75M** (−84.2%); GFLOPs 28.7 → 20.1 (−~30%); FPS 148 → 126.
- **SOD-YOLO-l:** mAP50 **51.5%** vs YOLOv8l 43.8% (**+7.7 pts**); params 43.6M → 17.6M (−59.6%).
- **AP-small (COCO metric):** SOD-YOLO-l 21.9% vs YOLOv8l 15.5% (**+6.4 pts**) — the gain concentrates exactly on small objects.
- Gap **widens with scale** (n: +1.0 pt; l: +7.7 pts). Beats YOLOv9e by 4.9 pts mAP50 with ~30% of its parameters; beats Drone-YOLO with ~¼ the parameters.
- **Generalisation (SODA-D):** SOD-YOLO-s 64.1 mAP50 vs YOLOv8s 61.8; APrS (relatively-small) +4.1 pts.

**Dataset details:**
- **Primary — VisDrone2019:** 10,209 still aerial images (from 288 UAV video clips), **ten** categories (pedestrians, people, vehicles, bicycles, etc.); train 6,471 / val 548 / test 1,610. AISKYEYE team, Tianjin University. City streets, rural fields, construction sites; **varied drone viewpoints + low altitude** (not nadir orthophoto).
- **Secondary — SODA-D:** 24,828 traffic-scene images, nine categories, 278,433 instances (used only to test generalisation).
- Setup: input 640×640, 150 epochs, SGD (LR 0.01→0.0001), single RTX 3090, Ultralytics v8.0.202, **no pretrained weights** (deliberate, for fair architecture comparison).

**Relevant to my project because:**
- It's my **citable, current (2024) evidence** that small-vehicle-from-above is a distinct *small-object* problem that stock YOLOv8 handles poorly — and that the fix is architectural (high-resolution head + better neck fusion), not just more data. This directly explains why my **off-the-shelf YOLOv8 gave 0 detections** on day one.
- Confirms YOLOv8 is the right modern baseline to build on — these authors started from the same place I did, in 2024.
- AP-small is the metric that moved most → backs my plan to report small-object AP separately, not just headline mAP.
- Establishes the **upgrade path** if fine-tuning stock YOLOv8 still under-detects my small cars: add a higher-resolution (P2-style, 160×160) detection head and improved multi-scale fusion.

**Limitations they mention:**
The improved network "lacks a head component" and omits mainstream attention mechanisms (room for further gains); main evaluation confined to VisDrone2019, so they call for testing on a wider range of UAV platforms and more diverse datasets.

**My critical take (beyond what they admit):**
- **Domain gap is large and runs the *opposite* way to my project.** VisDrone is *low-altitude, multi-angle* drone footage of streets (pedestrians, motorbikes, oblique views, motion blur). My imagery is *high-altitude nadir orthophoto* (~25 cm Getmapping) of retail car parks. SOD-YOLO is optimised for scale/viewpoint *variation*; my scenes are scale-*consistent* and top-down. So its architecture is well-motivated for VisDrone but I cannot assume the +7.7 pt gain transfers to my fixed-nadir car task — note this explicitly if I cite the numbers.
- **Absolute mAP is low (42–51%).** Even this 2024 SOTA only reaches ~51.5 mAP50 on VisDrone. Useful **expectation-setting**: I should not be alarmed if my own small-car mAP looks modest — small-object UAV detection is genuinely hard, and that's the field-wide baseline, not a failure of my pipeline.
- **"Trained from scratch" ≠ advice for me.** They drop pretrained weights *only* to isolate the architecture's contribution in a fair benchmark. My CNN/Object-Detection notes commit me to **transfer learning** (fine-tune pretrained YOLOv8) — the recommended route with limited UK data. So their from-scratch numbers are **not directly comparable** to results I'll get by fine-tuning, and I must not present "train from scratch" as a takeaway.
- **Vehicles are one of ten classes** — VisDrone results are not car-only, so I can't read across a clean "car detection accuracy" figure. SODA-D is closest to my vehicle/traffic focus.
- Complexity cost: BSSI-FPN + RFCBAM is a **custom architecture**, not a config flag. Realistically my project uses stock YOLOv8 fine-tuning; SOD-YOLO is best framed as the *evidence-backed extension* I'd reach for if stock falls short, not my Phase-1 method.

**Design decision this justifies:**
- **Reporting AP-small / mAP@0.5 and 0.5:0.95 separately** (their headline gain was on AP-small) → strengthens my evaluation-metrics chapter.
- **A dedicated small-object literature thread**, now anchored by a 2024 YOLOv8 paper rather than only pre-deep-learning work — closes the "no modern detector cited" weakness.
- **Higher input resolution / `imgsz` and resolution-matched training** — their whole gain comes from preserving high-resolution spatial detail, mirroring the VEDAI/COWC "cars are a handful of pixels" argument.
- **A concrete, citable Phase-2 / future-work path:** if fine-tuned stock YOLOv8 under-detects small cars, add a P2-level high-resolution detection head + improved neck fusion — with quantified evidence (+6.4 pts AP-small) that this is the established fix.
- Lets me **set realistic mAP targets** in my methodology and discussion.

**Possible citation:**
Li et al. (2024, *Remote Sensing*) — SOD-YOLO, a YOLOv8-based small-object detector for UAV imagery; an RFCBAM downsampling backbone and BSSI-FPN neck raise small-object AP substantially (AP-small +6.4 pts over YOLOv8l) while cutting parameters by 60–84%, evidencing that stock YOLOv8 under-performs on small aerial targets and that high-resolution feature fusion is the remedy.

**One thing I'm still unsure about:** ❓ Does VisDrone contain enough top-down/nadir car-park-like scenes for SOD-YOLO's gains to plausibly transfer to my fixed-nadir retail imagery — or is it dominated by oblique street-level drone footage that makes the comparison loose? (Worth a look at the Figure 9 samples / VisDrone docs before I lean on its numbers.)

---

## Influence of Aerial Image Resolution on Vehicle Detection Accuracy

**Author(s):** Gliaubičiūtė, D.; Janavičius, R.; Gadeikytė, A.; Paulauskas, L. (Kaunas University of Technology, Lithuania)
**Year:** 2023 (*IVUS'2023*, 28th Conf. on Information Society and University Studies, Kaunas; published in CEUR Workshop Proceedings, Vol. 3575, ISSN 1613-0073, CC BY 4.0)

> NB: **This may be the single most directly relevant paper to my design so far.** It uses *both* my datasets (COWC + VEDAI), *my* model (YOLOv8), *my* exact method (transfer learning from pretrained weights), to answer *my* core question — how much does accuracy fall as resolution coarsens toward ~25 cm? Treat as a methods/positioning anchor, but mind the venue (MSc applied-research project, workshop proceedings — not high-tier peer review).

**Problem this paper solves (1 sentence):**
How far can you degrade the resolution of aerial vehicle imagery before one-stage detectors stop working well — i.e. what is the minimum pixel ratio that still gives acceptable car-detection accuracy?

**How they solved it (1-2 sentences):**
Took COWC and VEDAI, reduced everything to a single "Car" class, then progressively downsampled each dataset to 15, 20, 25, 27.5, 30, and 32.5 cm/pixel and fine-tuned YOLOv5, YOLOv7 and YOLOv8 (from pretrained weights, transfer learning, 100 epochs) at each resolution, recording P/R/mAP/speed to map the accuracy-vs-resolution curve.

**Key results/numbers:**
- **Degradation is gentle in the band I care about.** Coarsening every 5 cm from 12.5 → 27.5 cm/pixel cost only **~3.51% mAP on average** across all models.
- **YOLOv8 was the most resolution-robust** — on COWC it dropped just **0.24%** mAP from 12.5 → 27.5 cm. (YOLOv5 was worst, ~9.6% drop on COWC.)
- **At 25 cm/pixel (my resolution): YOLOv8 mAP@50 = 0.93 on COWC and 0.92 on VEDAI** — barely below the native-resolution scores. (Table 3.)
- **Cliff edge past 30 cm.** Beyond 30 cm degradation accelerates sharply, *especially on VEDAI*: at 32.5 cm the drop averaged ~3.23% (COWC) but ~72.84% (VEDAI); YOLOv7 on VEDAI collapsed to mAP@50 = 0.03.
- YOLOv7 was fastest (lowest inference ms) across the board; all models scored much higher on COWC than VEDAI.

**Dataset details:**
- COWC: reduced to "Car" only, negatives removed; 32,810 annotations, avg 18.8 boxes/image, 2–49 cars/image (dense). Native 15 cm.
- VEDAI: reduced to "Car" only (dropped plane/boat/camping car/tractor/truck/other); 2,807 annotations, avg **2.2 boxes/image**, mostly 2–4 cars/image (sparse). Native 12.5 cm.
- Splits 70/20/10 train/val/test; Roboflow for downsampling; pretrained weights `yolov5s/yolov7/yolov8s.pt`; single Nvidia T4 GPU.

**Relevant to my project because:**
- It is **the quantitative backbone for my whole "use ~25 cm Getmapping, not coarse satellite" thesis.** It converts my qualitative claim ("higher resolution detects better") into a citable curve, and crucially shows **25 cm is still firmly in the high-accuracy zone** (YOLOv8 mAP@50 ≈0.92–0.93), not a compromise.
- It is the **direct rebuttal to Leblanc's failure**: Leblanc used 0.6–1.0 m satellite and got R²≈0.11; this paper shows the accuracy cliff sits around 30–32.5 cm, so Leblanc was operating *well past* the usable-resolution edge. Strong "this is why I changed the imagery" evidence.
- Uses **my exact pipeline** (pretrained YOLOv8 + transfer learning) — a citable precedent that my approach is standard and works, and (unlike SOD-YOLO's from-scratch training) directly comparable to what I'll do.
- Justifies treating ~27.5 cm as my practical worst-case resolution budget and avoiding anything coarser.

**Limitations they mention:**
VEDAI's far smaller annotation count (2,807 vs 32,810) and its labelling of partially-hidden/edge-clipped vehicles likely depressed and destabilised its scores; training instabilities observed (YOLOv8 on COWC at 15 cm; YOLOv7 on VEDAI at 20 cm — metrics collapsed mid-run); they call for more models (incl. two-stage) and finer resolution steps.

**My critical take (beyond what they admit):**
- **The headline mAP numbers (0.86–0.97) are NOT comparable to SOD-YOLO's 42–51%.** This is *single-class* "Car" detection on relatively sparse scenes (VEDAI ≈2.2 cars/image); SOD-YOLO is 10-class dense VisDrone. So I must not put these scores side by side — they measure very different task difficulties. The *trend* (accuracy vs resolution) is the citable contribution, not the absolute values.
- **The VEDAI "collapse" is largely a dataset artifact, not pure resolution physics.** With only 2,807 boxes, ~2 cars/image, and edge/occluded labels, VEDAI is starved of training signal and brittle at low res; the authors half-acknowledge this but still fold the 72.84% VEDAI drop into headline framing. I should attribute the cliff to *data quantity + annotation quality interacting with resolution*, and lean on the **COWC** curve (denser, larger, closer to my dense car parks) as the more trustworthy guide.
- **Reporting is inconsistent.** The abstract says 30 → 32.5 cm drops "2.33% (COWC) and 42.4% (VEDAI)", but the conclusion gives 30 cm = 1.42%/11.96% and 32.5 cm = 3.23%/72.84% — these don't reconcile cleanly, and "%" slides between percentage-points and relative-% throughout. **Cite the curve's shape and the 25 cm operating point, not the loose aggregate percentages.**
- **Single T4, 100 epochs, one run per cell, tiny test split (10%)** → limited statistical weight; treat as indicative, not definitive.
- Native resolutions differ (VEDAI 12.5 cm, COWC 15 cm), so the 12.5 cm COWC row is upsampled — the authors flag the two datasets aren't comparable at 12.5 cm.
- Still non-UK, non-retail, and the COWC scenes (residential/parking) aren't UK supermarket car parks — domain gap persists alongside the resolution point.

**Design decision this justifies:**
- **My core positioning, now quantified:** ~25 cm imagery sits in the stable, high-accuracy band for YOLOv8 (mAP@50 ≈0.92–0.93) → strongest single citation for "why ~25 cm Getmapping is sufficient and why coarse satellite (Leblanc) failed."
- **Set ~27.5–30 cm as a hard resolution floor** in my methodology; anything coarser risks the cliff. Backs my resolution-matching / downsampling plan with numbers.
- **Choosing YOLOv8** over v5/v7 — it was the most resolution-robust here.
- **Transfer learning from pretrained YOLOv8 weights** is the validated route (reconciles the SOD-YOLO "from scratch" caveat — that was a benchmarking choice; *this* paper shows fine-tuning is what practitioners actually do and it works).
- **Lean on COWC, not VEDAI, for my resolution argument** given VEDAI's sparsity/annotation issues — consistent with my earlier note that COWC carries the dense-scene load.
- Reinforces reporting P/R/mAP@50/mAP@50-95 + speed (same suite).

**Possible citation:**
Gliaubičiūtė et al. (2023, IVUS/CEUR) — fine-tuned YOLOv5/v7/v8 on COWC and VEDAI across 12.5–32.5 cm/pixel; detection accuracy is stable down to ~27.5 cm (YOLOv8 mAP@50 ≈0.92–0.93 at 25 cm) and degrades sharply beyond 30 cm, with YOLOv8 the most resolution-robust — evidence that ~25 cm aerial imagery is sufficient for car detection while coarse satellite resolutions are not.

**One thing I'm still unsure about:** ❓ Is the sharp VEDAI drop mostly a resolution effect or mostly a small-dataset/annotation artifact? If I can establish it's the latter, I can cite the (more reassuring) COWC curve as the better guide for my dense UK car parks — worth a sentence in my discussion.

---

## Smart Parking Space Availability Detection

**Author(s):** H et al.
**Year:**
**Date Read:**

**Problem this paper solves (1 sentence):**

**How they solved it (1-2 sentences):**

**Key results/numbers:**

**Is this ground-level or aerial detection?**
<!-- Important distinction — ground-level cameras are very different to aerial -->

**Relevant to my project because:**

**Limitations compared to my approach:**

**My critical take (beyond what they admit):**

**Design decision this justifies:**

**Possible citation:**

**One thing I'm still unsure about:** ❓

---

## Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

**Author(s):** Ren et al.
**Year:** 2015
**Date Read:**

**Problem this paper solves (1 sentence):**

**How it differs from YOLO:**
<!-- Both are object detectors but work differently — worth understanding the distinction -->

**Key results/numbers:**

**Why this is relevant background for a YOLO project:**

**My critical take (beyond what they admit):**

**Design decision this justifies:**

**Possible citation:**

**One thing I'm still unsure about:** ❓

---

---

# SECTION 3 — Broader Context

---

## The View from Above: Applications of Satellite and Aerial Imagery

**Author(s):** Donaldson et al.
**Year:**
**Date Read:**

**Main argument of this paper (2-3 sentences):**

**Examples of how aerial imagery has been used in research:**

**How this situates my project in the wider field:**
<!-- This is useful for your introduction chapter -->

**Possible citation:**

---

---

# SECTION 4 — Tools & Documentation
*Read these when you need them, not all at once*

---

## Ultralytics YOLOv8 Documentation

**Date Read:**
**URL:**

**What YOLOv8 does that earlier versions didn't:**

**Key parameters I need to understand:**
- `conf`:
- `iou`:
- `epochs`:
- `imgsz`:

**How to fine-tune on custom data (summary):**

**Gotchas / things to watch out for:**

---

## QGIS Training Manual

**Date Read:**

**What I will use QGIS for in this project:**

**Key things I learned:**

**Gotchas / things to watch out for:**

---

---

# My Open Questions
*Running list — bring these to your next Toby meeting*

- ❓ Do I have GPU access for training/fine-tuning, and how much?
- ❓ How much labelled UK retail training/test data can we realistically produce?
- ❓ Practical trade-off (speed vs accuracy) between YOLO, SSD and R-CNN — and which suits small aerial cars best?
- ❓ Where does total parking capacity come from for the occupancy denominator (site info vs manual bay counting)?
- ❓ How do I document Digimap image capture dates to avoid the temporal-validity critique that affects open-satellite studies?
- ❓ Can I find a fuller version of the Leblanc et al. study with formal detection metrics to contrast against?
- ❓ Cite VEDAI as the 2015 HAL report or the 2016 JVCIR journal version?
- ✅ ANSWERED: Does COWC include dense/occluded car-park scenes? — YES. COWC contains dense parking lots (VEDAI excluded them), so COWC carries the dense-case training load. My UK set still needed for UK-specific domain match.
- ❓ What fixed box size to use when converting COWC's centre-point dot annotations into YOLO bounding boxes (cars 24–48 px, varied rotation)?
- ❓ Is the sharp VEDAI accuracy drop at low resolution (Gliaubičiūtė et al.) mostly a resolution effect or a small-dataset/annotation artifact (only 2,807 boxes, edge/occluded labels)? If the latter, I can foreground the steadier COWC curve as the better guide for my dense car parks.
- ❓ Does VisDrone (SOD-YOLO's dataset) contain top-down/nadir car-park scenes, or mostly oblique street-level drone footage? Decides whether SOD-YOLO's gains plausibly transfer to my fixed-nadir imagery.
- ❓ Decision for Toby: stick with stock YOLOv8 fine-tuning, or budget for a P2 high-resolution detection head if AP-small is poor? SOD-YOLO shows +6.4 pts AP-small from exactly this kind of change, but it's a custom architecture, not a config flag.
- ❓ SOD-YOLO trained *from scratch* (no pretrained weights) for fair benchmarking — does that mean my transfer-learning (fine-tuned) results aren't directly comparable to their published numbers?

---

# Themes Across the Literature
*Fill this in after you've read at least 3 papers — useful for writing your literature review chapter*

**Common challenges mentioned across papers:**
- **Resolution / small-target difficulty** — Leblanc (coarse 0.6–1.0 m satellite), VEDAI (cars ≈0.7% of pixels), COWC (cars 24–48 px at 15 cm) and SOD-YOLO (whole architecture rebuilt around preserving high-resolution spatial detail) all frame vehicle-from-above as a *small-object* problem where resolution / spatial detail is decisive. A clear field-wide theme spanning classical (2015) to modern YOLOv8 (2024) work. SOD-YOLO adds the key modern lesson: even 2024 SOTA only reaches ~42–51 mAP50 on small UAV targets — absolute accuracy stays low, so realistic expectation-setting matters. **Gliaubičiūtė et al. (2023) now quantifies the resolution axis directly:** fine-tuned YOLOv8 stays stable down to ~27.5 cm (mAP@50 ≈0.92–0.93 at 25 cm) and only falls off a cliff beyond 30 cm. So the theme sharpens into a concrete claim — ~25 cm is usable, ~0.6–1 m satellite (Leblanc) is past the cliff — which is the spine of my positioning.
- **Density, clustering and occlusion** — Leblanc blamed clustered cars for missed detections; VEDAI *excluded* dense car parks from its benchmark; COWC explicitly *includes* dense lots and notes occlusion/merging as the hard cases for counting. So density is acknowledged across the field as the difficulty — but treated very differently by each dataset.
- **Domain shift** — every dataset is from elsewhere (US/EU/NZ/CA, not UK) and at a different resolution to my imagery; all three implicitly or explicitly raise generalisation to new regions/sensors.

**Gap in the literature my project addresses:**
- COWC narrows (but doesn't close) the gap: it gives dense-car-park training data at 15 cm from six non-UK regions. SOD-YOLO (2024) closes the *"modern detector"* side of the gap — it shows what a current YOLOv8 architecture achieves on small aerial targets — but on *low-altitude, multi-angle drone* imagery (VisDrone), not nadir orthophoto, and not UK retail. Gliaubičiūtė et al. (2023) closes the *"is ~25 cm enough?"* side — yes, for single-class car detection on COWC/VEDAI. So what remains untested is still the specific combination I target — **dense UK retail car parks at ~25 cm fixed-nadir, with the detector domain-adapted to that resolution and validated on UK ground truth.** Prior work is coarse and non-adapted (Leblanc), sparse/non-UK (VEDAI), dense-but-US one-look counting (COWC), modern-but-oblique-drone (SOD-YOLO), or a resolution-sweep on non-UK data without the occupancy-estimation step (Gliaubičiūtė). None occupies my niche.

**Methods that seem most promising for aerial car detection:**
- Deep CNN detectors (YOLO family) over the classical SVM/HOG/DPM baselines VEDAI tested — the 2015 "nothing works" verdict predates modern detectors and is motivation, not a ceiling. SOD-YOLO (2024) confirms YOLOv8 is the live baseline researchers build on, and shows the specific architectural levers that raise small-object accuracy: a higher-resolution (P2-style) detection head and stronger multi-scale neck fusion (+6.4 pts AP-small over stock YOLOv8l). My concrete fallback if stock fine-tuning under-detects small cars.
- **Counting paradigm is a live design choice:** COWC counts by one-look regression (no localisation); I count by detection (localise then tally). My approach is justified because I need *locations* for occupancy mapping and interpretability — worth arguing explicitly in the occupancy chapter against the density/one-look alternatives.

**Cross-cutting business-context anchor:**
- Both Leblanc and COWC explicitly frame overhead car counting as a proxy for retail activity / footfall (COWC even cites a commercial product for investors monitoring retailer volume). Two independent, citable anchors for my project's premise.
