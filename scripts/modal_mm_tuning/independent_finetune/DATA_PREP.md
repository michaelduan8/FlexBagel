# Data preparation: RadImageNet-VQA and ROCOv2

How the training data for `radimagenet.py` and `rocov2.py` was built. Both are image–caption
(alignment-style) datasets for CT/MRI, prepared into the same single-turn JSONL schema as the
other prepare scripts:

```json
{"id": "...", "orig_images": ["<source id or filename>"], "images": ["<absolute image path>"],
 "conversation": [{"role": "user", "content": "<prompt>", "img_loc": "before"},
                  {"role": "assistant", "content": "<caption>", "img_loc": null}]}
```

Local working copies live under `/scratch2/duanm/data/` on CARC (scratch, not backed up).

---

## RadImageNet-VQA (alignment subset)

**Source:** Hugging Face `raidium/RadImageNet-VQA` (gated; paper arXiv:2512.17396, MIDL 2026),
config `alignment`. Built from the CT/MRI part of RadImageNet: each image has one expert label for
modality, body region and pathology, and the caption is a template filled from those labels.

**License:** RadImageNet Research Use Agreement — non-commercial research only, no redistribution;
other users must request access themselves. Keep the Modal volume private.

### Steps

1. **Download** `alignment/` (59 train + 7 val parquet shards, ~38 GB; PNG bytes embedded) to
   `/scratch2/duanm/data/radimagenet_vqa/alignment/`.
2. **Extract images + build JSONL** with
   `data_processing/prepare/radimagenet_vqa.py --raw_data_dir <dir> --split {train,val} --output <jsonl>`.
   - Images are written to `images/<split>/<shard:05d>/<row:05d>_<orig_name>.png`. The row prefix is
     needed because original filenames are **not unique** (403 repeat, mostly `lung*.png`).
   - The `<image>` token is stripped from the prompt (placement is carried by `img_loc`); the
     response role `template` is mapped to `assistant`. Metadata (modality/location/pathology) is dropped.
3. **Remove benchmark-overlapping images** → `*_nobench.jsonl`. The benchmark (`benchmark/test`,
   1,000 images / 9,000 QAs, drawn from RadImageNet's *test* split) was compared against every
   alignment image (32×32 grayscale thumbnails, normalized correlation). 58 alignment rows whose image
   matched a benchmark image at corr ≥ 0.99 were removed: 57 byte-identical (all ankle/foot MRI) plus
   1 visually identical abdomen CT. (One-off analysis; not a repo script.)
4. **Modal versions** (`*_modal.jsonl`): image paths rewritten from the local prefix to
   `/mnt/radimagenet/radimagenet/` with `data_processing/remap_rrg24_paths_modal.py
   --orig_prefix /scratch2/duanm/data/radimagenet_vqa/ --new_prefix /mnt/radimagenet/radimagenet/`
   (the extra `radimagenet/` level keeps the volume root clean).
5. **Transfer zip** `radimagenet_images.zip` (36 GB, 833,677 PNGs, stored/uncompressed) containing
   `images/...`. It was written directly from the parquet bytes (much faster than zipping 834k small
   files on BeeGFS); entry paths match the JSONLs and spot-checked byte-identical to the extracted files.

### Final files used for training

| File | Rows |
|---|---|
| `radimagenet_vqa_alignment_train_nobench_modal.jsonl` | 749,956 |
| `radimagenet_vqa_alignment_val_nobench_modal.jsonl` | 83,663 |

Unfiltered versions (`radimagenet_vqa_alignment_{train,val}_modal.jsonl`, 750,009 / 83,668 rows)
are also kept.

### What's in it

- 833,677 images, all 224×224 RGB. 73% MRI / 27% CT. Regions: abdomen 25% (only region with both
  CT and MRI), ankle/foot 19%, knee 19%, lung 12% (CT), spine 8%, shoulder 6%, hip 6%, brain 5% (MRI).
- 97 pathology labels, 26% "normal"; very imbalanced (abdomen-normal alone ≈128k rows; 30 classes
  have <500 rows, 7 have <100).
- Captions are templated, 8–18 words (median 10), e.g. *"MRI scan of the knee knee joint with findings
  consistent with meniscal abnormality"*. 65 distinct prompts. Captions carry only the three labels —
  good for learning modality/region/coarse pathology, not for descriptive language. Some templates
  omit the pathology name ("…reveals pathological alterations"); some have doubled words.

### Known issues (left in place — no deduplication by choice)

- **Exact duplicate images:** 10.5% of rows (43,669 groups, almost all size 2), concentrated in
  ankle/foot (52.8% of its rows), so ankle/foot is effectively double-weighted. 97% of duplicate pairs
  share labels and differ only in caption wording; 1,287 groups carry different labels — mostly the
  same image stored once per finding (RadImageNet is single-label), plus 9 with conflicting body regions.
- **Train/val overlap:** 8,002 val rows (9.6%) are byte-identical to train images. We evaluate on the
  official benchmark instead of val.
- **Blank images:** one all-black image ×135 (labelled abdomen-normal) and one uniform-gray image ×53
  (labelled brain-normal).
- **Possible patient overlap with the benchmark:** beyond the removed exact copies, some benchmark
  images have near-identical neighbouring slices in train (184 at corr ≥ 0.95). RadImageNet exposes no
  patient IDs, so this can't be verified or removed; treat benchmark scores as relative comparisons.

### Not used

- `instruct/` (also downloaded): the **same images** in the same order with ~9 templated QA turns per
  image. Prepared (`data_processing/prepare/radimagenet_vqa_instruct.py`, reuses the extracted images)
  but not used because turns leak answers to each other (e.g. turn 1 "abdomen" answers turns 2–4),
  turn position predicts yes/no answers, and MC answers are option text rather than letters.
- `benchmark/test-00000-of-00001.parquet` is downloaded but not yet converted to an eval JSONL.

---

## ROCOv2 (CT + MRI subset)

**Source:** Zenodo record 10821435 — *ROCOv2: Radiology Objects in COntext Version 2* (v2.0.1).
Radiology figures and their captions from PubMed Central open-access articles, with UMLS concept
annotations. **License:** CC BY-NC 4.0 (per-image attribution in `raw/license_information.csv`).

### Steps

1. **Download** all 14 files to `/scratch2/duanm/data/rocov2/raw/` (image zips per split + captions,
   concepts, manual concepts, CUI mapping, licenses); all verified against the Zenodo MD5s.
2. **Extract** the three image zips to `images/{train,valid,test}/ROCOv2_2023_<split>_<n>.jpg`
   (59,958 / 9,904 / 9,927 images, matching the caption counts).
3. **Build JSONL** with
   `data_processing/prepare/rocov2.py --raw_data_dir <dir> --split {train,valid,test} --modality ct mri --output <jsonl>`.
   - ROCOv2 has captions only, so each caption is paired with one of 8 generic captioning prompts
     ("Describe this medical image.", …), chosen deterministically from the image ID (~even split).
   - `--modality` filters on the **manual** concept annotations: every annotated image carries exactly
     one modality CUI (CT `C0040405`, MRI `C0024485`, X-ray, ultrasound, angiography, PET); none is both
     CT and MRI. Images with no manual concepts (~500 in train) are excluded since modality is unknown.
   - One valid row was dropped for an empty caption (a lone narrow no-break space).
4. **Modal versions** via `remap_rrg24_paths_modal.py --orig_prefix /scratch2/duanm/data/rocov2/
   --new_prefix /mnt/rocov2/rocov2/`.
5. **Transfer zip** `rocov2_images.zip` (6.8 GB, all 79,789 images, all modalities) containing
   `images/{train,valid,test}/...`, repacked from the three Zenodo zips with an `images/` prefix.
6. **Length filter at 4,096 tokens** → `rocov2_ctmri_{split}_w_length.jsonl` (+ `_modal`), using the
   measurement in `data_processing/length_filtering.py` (Qwen2.5-VL-3B processor; image tokens + prompt
   + caption). Lengths before filtering: p50 ≈ 530–620, p99 ≈ 1.2–1.3k, max up to 16.4k. Nothing falls
   between ~3.5k and 4,096, so the cutoff only removes the extreme outliers — very large images
   (up to 3799×5842, 8k–16k tokens) that could OOM at batch size 8: 10 train, 2 valid, 2 test.
   Removed rows are kept in `rocov2_ctmri_{split}_removed_long_examples.jsonl`. The `_w_length` files
   carry extra `vlm_length`/`vlm_text_length`/`vlm_image_token_estimate` columns (ignored by training).
   (Run via a small driver around the script's functions, since its save lines are commented out.)

### Final files used for training

| File | Rows |
|---|---|
| `rocov2_ctmri_train_w_length_modal.jsonl` (used by `rocov2.py`) | 30,239 |
| `rocov2_ctmri_valid_w_length_modal.jsonl` | 5,115 |
| `rocov2_ctmri_test_w_length_modal.jsonl` | 5,036 |

Before length filtering (`rocov2_ctmri_{split}_modal.jsonl`): 30,249 / 5,117 / 5,038 rows
(CT 20,459 / 3,673 / 3,615; MRI 9,790 / 1,444 / 1,423). All-modality versions (`rocov2_{train,valid,test}_modal.jsonl`, 59,958 / 9,903 / 9,927 rows) are also
kept and work with the same zip.

### What's in it

- Real figure captions, median 17 words (up to ~400), with specific findings, sizes and locations
  (e.g. *"CT transversal image showing a 7 cm tumor in the upper pole of the left kidney with central
  necrosis."*). Complements RadImageNet's templated captions.
- Images vary in size and aspect (median ≈650×560; 10% strongly non-square; 2% with short side <256).

### Known caption/image noise (not filtered)

| Pattern (CT/MRI train captions) | Share |
|---|---|
| Mentions arrows / arrowheads / asterisks | 32% |
| Refers to a coloured marker ("blue arrow") | 13% |
| Follow-up / pre-/post-op timing not visible in the image | 6% |
| Very short (<6 words, e.g. "CT.", "Liver.") | 4% |
| Patient/case numbers; panel letters / "Fig. 2" | ~2% each |

Images can contain drawn arrows, numbered labels, burned-in scanner text, and occasionally a photo of
a monitor. Possible light filters if needed: drop captions under ~4 words; strip marker references
like "(arrow)".

---

## Modal volume layout

JSONLs at the volume root; images one level down in a same-named folder:

```
/mnt/radimagenet/                                   (volume "radimagenet")
├── radimagenet_vqa_alignment_{train,val}_nobench_modal.jsonl
└── radimagenet/images/{train,val}/<shard>/<row>_<name>.png

/mnt/rocov2/                                        (volume "rocov2")
├── rocov2_ctmri_{train,valid,test}_w_length_modal.jsonl
└── rocov2/images/{train,valid,test}/ROCOv2_2023_<split>_<n>.jpg
```

Unpack the zips into the dataset folder (they already contain the top-level `images/`):

```bash
unzip -q radimagenet_images.zip -d radimagenet
unzip -q rocov2_images.zip -d rocov2
```

RadImageNet is not length-filtered: every example is 64 image tokens (224×224 needs no resizing)
plus ~30 text tokens. ROCOv2 is filtered at 4,096 tokens (see above).

All images in both datasets are 8 bits per channel (checked from every file header): RadImageNet PNGs
are 8-bit RGB; ROCOv2 JPEGs are 8-bit (29,007 grayscale, 50,782 RGB). The Qwen processor converts to
RGB (lossless for 8-bit grayscale) and only rounds sizes to multiples of 28.

---

## CT-Bench (lesion training data)

**Source:** Kaggle `zhuqingqingqq/ct-bench` (CT-Bench, arXiv:2602.14879; built on NIH DeepLesion),
downloaded with `kagglehub` to `/scratch2/duanm/data/ct_bench/raw/`. **License:** the Kaggle listing says
CC BY-NC-SA 4.0 while the bundled `License.txt` says CC BY-SA 4.0 — treat as non-commercial.

**Release contents used:** `lesion_{nobox,bbox}/lesion_<idx>.png` (key slices, 512×512 8-bit soft-tissue
window; the bbox version has a green box) and the merged metadata CSV (description, size, split,
patient/study ids, DeepLesion measurements, ontology `type`/`bodypart`/`property`). The QA benchmark
(`qa_llm.json`, 950 val / 1,900 test MCQs) is used only for evaluation.

**Leakage check:** the official split is patient-disjoint (0 of 3,793 patients span splits) and every
benchmark question uses val/test lesions — no benchmark lesion, study or patient is in train. Note that
report wording is formulaic, so 340/1,200 correct description options match some train description verbatim.

### Steps

`data_processing/prepare/ct_bench.py --data_dir /scratch2/duanm/data/ct_bench --split {train,val}`:

- **A. Captioning** — boxed and unboxed slice → report description (8 prompt variants). Unboxed rows are
  skipped for the 197 slices that hold several lesions (ambiguous "the lesion").
- **B. Size** — → report size normalized to cm (`"1.3 cm x 1.0 cm"`; mm converted, "TRV/AP" stripped,
  3-D sizes kept, ranges/single values skipped). Each report size is cross-checked against DeepLesion's
  RECIST measurement (99.9% agree); 19 train rows disagreeing by >1.5× are dropped (incl. 5 mm-labelled-as-cm).
- **C. Attributes** — location (`bodypart`), type (`type` minus body-part/property terms — the benchmark's
  type answers never contain those), property (24% of lesions). Answers list all values, rarest
  (most specific) first.
- B and C use the boxed or unboxed slice 50/50 per lesion (boxed for multi-lesion slices); the prompt
  mentions the green box when present.

Modal JSONLs via `remap_rrg24_paths_modal.py --orig_prefix /scratch2/duanm/data/ct_bench/raw/
--new_prefix /mnt/ctbench/ctbench/images/`; `ctbench_images.zip` (5.8 GB) holds the 33,838 train+val
slices as `images/lesion_{nobox,bbox}/...`.

### Final files

| File | Rows | caption | size | location | type | property |
|---|---|---|---|---|---|---|
| `ct_bench_train_modal.jsonl` (used by `ctbench.py`) | 77,500 | 30,362 | 15,355 | 15,148 | 12,998 | 3,637 |
| `ct_bench_val_modal.jsonl` | 8,823 | 3,476 | 1,757 | 1,733 | 1,453 | 404 |

Volume layout (volume `ctbench`): JSONLs at `/mnt/ctbench/`, images under `/mnt/ctbench/ctbench/images/`
(`unzip -q ctbench_images.zip -d ctbench`). Sequences are ~330 image tokens + short text (a few 1024×1024
slices ≈ 1.3k image tokens), so no length filtering was needed.

---

## ECG-Grounding (variants A and AB)

**Source:** HF `LANSG/ECG-Grounding` (GEM, arXiv:2503.06073), downloaded to
`/scratch2/duanm/data/ecg_grounding/raw/`. Card license Apache-2.0; the underlying signals come from
PhysioNet MIMIC-IV-ECG / PTB-XL, which carry their own terms. Training data is
`grounding_train_30k.json`: 30,000 MIMIC-IV-ECG ECGs, one 12-lead image each (ECG-image-kit render,
2200×1700), one instruction ("Interpret the provided ECG image…") and a ~270-word GPT-4o-generated,
evidence-based interpretation. Also included: per-beat machine features and the machine report.

**Relation to the GEM paper:** GEM trains in one stage on ECGInstruct (1.16M) + these 30k, with a raw-signal
encoder and a 336×336 CLIP image (center crop). The paper reports no grounding-only training. Here
ECGInstruct is a separate silo; these runs are image-only Qwen2.5-VL-3B.

### Steps

`data_processing/prepare/ecg_grounding.py --data_dir /scratch2/duanm/data/ecg_grounding --variant {A,AB}
--image_scale 0.5 --extra_images <mimic test json> <ptbxl test json>`:

- **Images** resized ×0.5 to 1100×850 RGB (≈1.2k image tokens instead of ≈4.8k; waveforms and the major
  grid stay legible, the 1 mm grid becomes faint) into `images_x0.5/gen_images/...`; the MIMIC-IV and
  PTB-XL grounding test images are resized identically for evaluation.
- **Val split:** 1% of patients (206 / 20,606), patient-disjoint from train.
- **A** — the released interpretation rows (prompt with `<image>` stripped).
- **AB** — A plus one short QA row per ECG, seeded choice among the tasks its metadata supports:
  `rhythm` (first rhythm statement of the machine report), `heart_rate` (machine rate, kept only when it
  agrees with the report's brady/tachy/sinus wording — it disagrees for ~14% of ECGs), `verdict`
  (Normal / Borderline / Abnormal / Normal except for rate), `report` (report statements minus boilerplate).
  Per-lead beat features are not used: they are automatically extracted and visibly noisy.

**Known data caveats:** the per-lead features are noisy, and the "RR Interval (ms)" field is actually in
samples (2× off). GEM's interpretations were generated from these features, so some quoted numbers in A
inherit that noise. ~50% of the MIMIC-IV grounding test rows come from patients also in train (no shared
studies); PTB-XL and ECG-Bench come from other databases.

Modal JSONLs via `remap_rrg24_paths_modal.py --orig_prefix /scratch2/duanm/data/ecg_grounding/
--new_prefix /mnt/ecggrounding/ecggrounding/`; `ecggrounding_images.zip` (12 GB) holds the 30,000
train+val images as `images_x0.5/gen_images/mimic_gen/...`.

### Final files

| File | Rows |
|---|---|
| `ecg_grounding_A_train_modal.jsonl` (used by `ecg_grounding_a.py`) | 29,720 |
| `ecg_grounding_A_val_modal.jsonl` | 280 |
| `ecg_grounding_AB_train_modal.jsonl` (used by `ecg_grounding_ab.py`) | 59,440 (29,720 interpretation + 8,050 rhythm + 8,139 report + 7,810 heart rate + 5,721 verdict) |
| `ecg_grounding_AB_val_modal.jsonl` | 560 |

Volume layout (volume `ecggrounding`): JSONLs at `/mnt/ecggrounding/`, images under
`/mnt/ecggrounding/ecggrounding/` (`unzip -q ecggrounding_images.zip -d ecggrounding`). Sequences are
1,375 tokens median, 1,673 max — no length filtering needed.

---

## SurgΣ-DB v0.1 (CholecT50 + GraSP image tasks)

**Source:** HF `SurgSigma/SurgSigma-DB` (arXiv:2603.16822), v0.1 = `SurgSigma_v0.1_{T50,GraSP}subset.json`
in `/scratch2/duanm/data/surgsigma_db/annotations/`. Annotations only (CC BY-NC-SA 4.0); the media
come from the official sources and keep their own terms: **CholecT50** (CAMMA request form, full
release, 1 fps 854×480 PNGs, CC BY-NC-SA 4.0) and **GraSP** (`GraSP_1fps/frames` + `annotations` from
the GraSP GitHub's Drive folder, 1280×800 JPEGs), extracted to `raw_data/{CholecT50,GraSP/GraSP_1fps}/`.
Every annotation image path resolves (CholecT50 paths mix `VID1`/`VID01` and are normalized), except one
GraSP row that points one frame past the end of CASE001 (skipped).

### Steps

`data_processing/prepare/surgsigma.py --data_dir /scratch2/duanm/data/surgsigma_db --subset {cholect50,grasp}
--split {train,val,test} --output <jsonl>`:

- **Tasks kept** (default `--tasks 0 3 5 6 7 8 10`): instrument, tissue, phase, step, action, triplet,
  safety, plus the combined instrument+phase and phase+step rows. Dropped: dense prediction / generation
  (image outputs), video tasks, localization (1) and captions (11) (selectable via `--tasks`), and the
  `<thinking>` traces (answers only).
- **Splits by video/case:** CholecT50 official `standard` split from the CAMMA loader (`--cholect50_split
  challenge` also available); GraSP official train/test cases (no val). CholecT50 VID05/12/26/27/31 are
  excluded from every split: their Cholec80 frames are in the EndoChat eval (`eval/vlm/eval/endochat/all.jsonl`).
- **Prompts/answers verbatim** (mixed formats: phrases, sentences, multiple-choice JSON when the prompt asks
  for it, "## Surgical Answers" blocks), except GraSP phase/step answers, which are JSON although their
  prompts never ask for it — unwrapped to `J. Seminal Vesicles`. Exact (image, question, answer)
  duplicates removed.
- **Per-frame cap** (`--max_rows_per_frame_task 1`, train only → `*_train_capped.jsonl`): the annotator groups
  (SJTU terse/single-template, NUS sentence/MC-JSON/multi-question) restate each frame's CholecT50 label several
  times; one seeded phrasing per (frame, task) is kept, both styles survive across frames. Rows carry an `annotator` field.
- **GraSP safety** is 99.85% "criteria not met" (175 of 116,516 positive): all positives kept plus at most
  10× negatives (`--grasp_safety_neg_ratio`). CholecT50 safety (84k no / 19k yes) is kept as-is.
- **Images** re-encoded to JPEG q95 under `images/{CholecT50/VIDxx,GraSP/CASExxx}/<frame>.jpg`: CholecT50
  at native 854×480 (510 image tokens), GraSP resized ×0.5 to 640×400 (322 instead of ~1.3k tokens).

SurgΣ released test-video annotations for only some tasks, so the test files are partial: CholecT50 test =
instrument/action/triplet; GraSP test = action/triplet/safety. Tissue appears only in CholecT50 train.

Modal JSONLs via `remap_rrg24_paths_modal.py --orig_prefix /scratch2/duanm/data/surgsigma_db/
--new_prefix /mnt/surgsigma/surgsigma/` (pass absolute input paths); `surgsigma_images.zip` (~9 GB,
stored) holds all 94,444 images of every split as `images/...`.

### Final files

| File | Rows | Tasks |
|---|---|---|
| `surgsigma_cholect50_train_capped_modal.jsonl` (used by `surgsigma.py`) | 302,439 | one phrasing per (frame, task), safety per (frame, question): safety 101k, triplet 40k, action 40k, instrument 40k, phase 38k, tissue 30k, instrument+phase 14k |
| `surgsigma_cholect50_train_modal.jsonl` (uncapped) | 427,407 | 31 videos: safety 101k, triplet 92k, instrument 64k, action 64k, phase 63k, tissue 30k, instrument+phase 14k |
| `surgsigma_cholect50_val_modal.jsonl` | 31,414 | 4 videos; all but tissue |
| `surgsigma_cholect50_test_modal.jsonl` | 41,337 | 10 videos; instrument, action, triplet |
| `surgsigma_grasp_train_capped_modal.jsonl` (used by `surgsigma.py`) | 45,676 | capped as above |
| `surgsigma_grasp_train_modal.jsonl` (uncapped) | 46,369 | 8 cases: triplet 11k, phase 10k, step 10k, phase+step 10k, instrument 2.5k, action 1.8k, safety 0.9k |
| `surgsigma_grasp_test_modal.jsonl` | 8,935 | 5 cases; action, triplet, safety |

Volume layout (volume `surgsigma`): JSONLs at `/mnt/surgsigma/`, images under `/mnt/surgsigma/surgsigma/`
(`unzip -q surgsigma_images.zip -d surgsigma`). Sequences are ≤ ~860 tokens (510/322 image tokens + text
p99 ~300), so no length filtering. CholecT50 is ~9× GraSP by rows; pass one `--sample_size` per dataset to rebalance.

### SurgΣ video clips (frame lists)

`data_processing/prepare/surgsigma_video.py --data_dir /scratch2/duanm/data/surgsigma_db --subset {cholect50,grasp}
--split {train,val,test} --output <jsonl>` rebuilds SurgΣ's video tasks from the 1 fps frames in `raw_data/` (frame
number = second); the annotations' mp4s (Cholec80 / GraSP_30fps) are only used for their time windows.

- **Tasks:** CholecT50 action (7), video caption (12; 25 fps indices → seconds), action remaining time (13), next
  action (14; single moment → the 8 s ending at it); GraSP video caption (12). Dropped: GraSP 7/13/14 (time steps run
  up to 1.5× past the last frame in 6 of 8 train cases — unknown time base), the 1–3-frame phase/triplet clips
  (duplicates of the image rows) and video generation (17).
- **Rows:** `{"videos": [[frame paths]], "video_fps": [fps], ...}` for `VideoAwareVLMCollator`. Every second up to 32
  frames; longer clips (22% of CholecT50) use stride 2 and `video_fps` 0.5. Frames rendered to
  `video_frames/{CholecT50,GraSP}/<video>/<frame>.jpg` at 448×252 / 448×280 (144 / 160 tokens per 2 frames).
- Same splits and EndoChat exclusion as the image rows. Verified: all 72,940 frames resolve; a mixed image+video
  batch runs through `preprocess_dataset` + `VideoAwareVLMCollator` (second_per_grid_ts 2.0 at 1 fps).
- `_modal` paths `/mnt/surgsigma/surgsigma/video_frames/...` (remapped inline: `remap_rrg24_paths_modal.py` only
  handles `images`); `surgsigma_video_frames.zip` (stored, 72,940 frames) unzipped on the volume with
  `scripts/modal_mm_tuning/unzip_volume.py --zip-path surgsigma_video_frames.zip --dest surgsigma --delete-zip`.

| File | Clips | Tokens/clip (mean / max) |
|---|---|---|
| `surgsigma_cholect50_video_train_modal.jsonl` (used by `surgsigma.py`) | 5,530 (action 1,433 · remaining 1,433 · next action 1,433 · caption 1,231) | 997 / 2,399 |
| `surgsigma_grasp_video_train_modal.jsonl` (used by `surgsigma.py`) | 1,264 captions | 1,042 / 2,675 |
| `surgsigma_cholect50_video_{val,test}_modal.jsonl` | 540 / 1,968 | |
| `surgsigma_grasp_video_test_modal.jsonl` | 736 | |

Video is ~2% of training rows / ~3% of tokens; repeat the video JSONLs in `--datasets` to upweight.
