# Construction AI Estimation

Automated quantity take-off and cost estimation from construction PDF drawings.

Upload a plan, and the engine reads the drawing, finds the structural elements
(columns, beams, slabs, walls), recovers the reinforcement schedule, measures
real-world quantities, prices them against a rate card, and hands back a BOQ
with downloadable CSV and PDF reports.

> **Version 2.0.0** — rewritten around exact PDF vector parsing. The engine no
> longer relies on a trained object detector to do the take-off (see
> [Honest limitations](#honest-limitations)).

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Web UI](#web-ui)
- [CLI](#cli)
- [HTTP API](#http-api)
- [Rate card](#rate-card)
- [Configuration](#configuration)
- [Accuracy and limitations](#accuracy-and-limitations)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Deployment (Render / Docker)](#deployment-render)

---

## Quick start

**Requirements:** Python 3.10+ on Windows, Linux or macOS.

```bash
git clone <repo-url>
cd Construction-AI-estimation-main

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# 1. sanity check the engine on a generated sample drawing
python tools/selftest.py

# 2. start the web UI
python main.py --serve
```

Open <http://127.0.0.1:8000>, drop in a PDF, press **Analyse drawing**.

No system-level dependencies: PDF rendering uses **PyMuPDF** (a pure wheel), so
Poppler/ImageMagick are *not* needed.

The core install is deliberately small — the engine needs no deep-learning
framework. If you want to train or run the optional YOLO detector, install the
extra (it pulls in torch, ~2 GB):

```bash
pip install ultralytics torch torchvision
```

---

## How it works

Real construction drawings are CAD exports, so the PDF already contains the
exact geometry and text of every line. v2 reads that directly instead of
guessing from pixels.

```
PDF ──► PyMuPDF ──► vector geometry ──► classification ──► quantities ──► BOQ
             │            (lines, rects, text)      (column/beam/slab/wall)
             │
             └──► raster fallback (OpenCV) for scanned / image-only pages
```

### 1. Ingest — `src/pdf_processor.py`
Each page is parsed twice over:

| Channel | What it gives | When it is used |
|---|---|---|
| **Vector** (`page.get_drawings()`, `page.get_text("words")`) | Exact line segments, stroke widths, rectangles, and every text run with its position | Always — this is the primary source |
| **Raster** (page rendered to an image) | Pixels | Only for pages with no usable vector content (`cv_mode: auto`) |

Rendering is capped at `max_render_dimension` (5200 px) so an A1 sheet at
300 DPI (~70 megapixels) stays workable. The vision pass then works at
`cv_max_dimension` (2200 px).

### 2. Scale — `src/blueprint/scale.py`
The denominator is resolved from, in order of trust:

1. **Title block / notes** — `SCALE 1:100`, `SCALE - 1:50`
2. **Dimension lines** — a labelled dimension matched against the line it
   annotates, cross-checked against the sheet size
3. **Graphical scale bar** — a bar of known length
4. **Manual override** — the `Drawing scale` dropdown in the UI, or `--scale`
   on the CLI

Consolidation (`consolidate_scales`) picks the best-supported value and records
which method won, so the report always states its own provenance.

### 3. Element detection — `src/blueprint/elements.py`
Vector geometry is classified by shape and drafting convention:

- **Columns** — closed small rectangles, or a dense cluster of coincident lines
- **Beams** — heavy strokes along grid/zone edges
- **Slabs** — large enclosed zones, merged where a zone is split by a beam
- **Walls** — parallel double lines (`convention: double-line`)

Two rules keep annotation panels out of the take-off:

- **Annotation panel detection** (`find_annotation_panels`) — clusters of dense
  text (title blocks, BBS tables, legends) are masked out before measuring.
- **Stroke-width rules** — thin strokes are grid/axis references, heavy strokes
  on a zone edge are beams.

### 4. Reinforcement — `src/blueprint/schedule.py`
If the drawing contains a **bar bending schedule**, it is parsed directly
(`MEMBER | DIA | LENGTH | NOS | SPACING`) and is treated as authoritative
steel. If no schedule exists, steel is derived from the concrete volume using
the per-member reinforcement ratios in `config.yaml`.

### 5. Quantities & BOQ — `src/estimator.py`, `src/rates.py`
Measured geometry becomes quantities (m³ concrete, m² formwork, m³ masonry,
t reinforcement), and quantities are priced against the rate card and a market
multiplier.

### 6. Reports — `src/report.py`, `src/annotate.py`
- `estimate.csv` — full BOQ, quantities, rates, amounts, totals
- `estimate.pdf` — formatted report with the annotated drawing
- Annotated page images — every detected element outlined and labelled by type

---

## Web UI

```bash
python main.py --serve --host 127.0.0.1 --port 8000
```

- Drag-and-drop or browse for a PDF (limit configurable, default 80 MB)
- Optional **project name**, **drawing scale** (leave on *Auto*), and
  **rate market**
- Live progress: parsing → scale & elements → quantities → reports
- Results tabs: **BOQ**, **Reinforcement schedule**, **Annotated drawings**,
  **Assumptions**
- Download the CSV and PDF reports

---

## CLI

```bash
# analyse a drawing and print the BOQ
python main.py --pdf drawing.pdf --name "Block A"

# override scale and rate market
python main.py --pdf drawing.pdf --scale 50 --market gcc

# start the web UI
python main.py --serve --host 0.0.0.0 --port 8000

# train the (optional) YOLO detector
python main.py --train

# run detection on a single image with trained weights
python main.py --infer page.png
```

Outputs land in `output/estimates/<job_id>/`:

```
output/estimates/<job_id>/
├── <job_id>_estimate.csv
├── <job_id>_estimate.pdf
├── page_001_annotated.jpg
└── result.json
```

---

## HTTP API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Liveness + engine version |
| `GET` | `/api/markets` | Available rate markets and multipliers |
| `POST` | `/api/upload` | Upload a PDF (`multipart/form-data`), returns a job id |
| `GET` | `/api/job/<job_id>` | Poll job status and progress |
| `GET` | `/api/result/<job_id>` | Full result payload |
| `GET` | `/api/download/<job_id>/<csv\|pdf>` | Download a report |
| `GET` | `/api/page/<job_id>/<page_no>` | Annotated page image |

```bash
curl -F "file=@drawing.pdf" -F "project_name=Block A" -F "market=india_urban" \
     http://127.0.0.1:8000/api/upload
```

---

## Rate card

All rates are **₹ (INR)** and live in `config.yaml`. Edit them freely — they
are the only thing standing between a drawing and a quote.

### Material rates

| Item | Rate | Unit |
|---|---|---|
| Concrete M20 / M25 / M30 / M35 | 6,500 / 7,200 / 7,900 / 8,600 | ₹/m³ |
| Steel Fe415 / Fe500 / Fe550 | 66,000 / 72,000 / 78,000 | ₹/tonne |
| Formwork — slab / beam / column / wall | 520 / 640 / 700 / 560 | ₹/m² |
| Brick masonry in cement mortar 1:5 | 5,400 | ₹/m³ |

### Market multipliers

| Market | Multiplier |
|---|---|
| `india_urban` | 1.00 |
| `india_rural` | 0.88 |
| `gcc` (Gulf) | 1.26 |
| `msec` (Middle East / Africa) | 1.08 |
| `international` | 1.85 |

### Geometric assumptions

| Assumption | Default | Config key |
|---|---|---|
| Storey height | 3.0 m | `estimation.storey_height_m` |
| Slab thickness | 125 mm | `estimation.slab_thickness_mm` |
| Beam | 230 × 450 mm | `estimation.beam_width_mm` / `beam_depth_mm` |
| Column | 300 mm | `estimation.column_size_mm` |
| Wall thickness | 230 mm | `estimation.wall_thickness_mm` |
| Wall type | masonry | `estimation.wall_type` (`masonry` \| `rcc`) |
| Cover | 40 mm | `estimation.cover_mm` |

Every assumption actually used is listed in the **Assumptions** tab of the UI
and in the generated PDF, so an estimate is never silently guessed.

---

## Configuration

Everything is driven by `config.yaml` at the repo root — no hard-coded values
in the pipeline. The most useful knobs:

```yaml
pdf_processing:
  dpi: 300                  # render resolution
  max_render_dimension: 5200
  cv_mode: "auto"           # auto | always | off
  cv_max_dimension: 2200

scale:
  default: 100              # fallback when nothing is detected
  detect_from_text: true

estimation:
  wall_type: "masonry"
  steel_ratio_kg_per_m3: {slab: 80, beam: 110, column: 160, ...}

rates:
  market: "india_urban"
  currency_symbol: "₹"

deployment:
  api_port: 8000
  upload_limit_mb: 80
```

---

## Accuracy and limitations

### Honest limitations

- **There is no trained object-detection model in this repository.** The YOLO
  path is wired end-to-end (training script, dataset YAML, inference CLI), but
  `models/yolo_weights/best.pt` and a labelled dataset are **not** included,
  so the YOLO branch is inactive by default. The take-off is carried by the
  vector-geometry engine and the reinforcement-schedule parser, which is more
  accurate on real CAD exports anyway.
- **OCR is not enabled.** Tesseract is not installed; scanned pages therefore
  rely on the OpenCV vision pass, which is considerably weaker than the vector
  path. Vector PDFs are strongly recommended.
- **Scale detection is the single biggest source of error.** If the drawing has
  no title block, no dimension lines and no scale bar, the fallback
  `scale.default` is used and the estimate will be wrong by the square of the
  scale error. Check the reported scale before trusting the numbers.
- **Quantities are geometric, not a substitute for a bill of quantities.** The
  engine measures what is drawn. It does not know about openings, drips, laps,
  wastage, water tanks, staircases, or builder's openings.
- **Formwork** is derived from member surface areas, not from a bar-schedule
  count of shuttering panels.
- **A single plan sheet describes one floor.** Multi-storey buildings are
  estimated for the geometry present on the drawing; storey count is read from
  the notes and used for wall/masonry height, not to multiply the structure.

### Typical confidence

The engine reports a confidence score. Roughly:

| Confidence | Meaning |
|---|---|
| **High (≥ 70%)** | Scale was confirmed and a reinforcement schedule was found |
| **Medium (45–70%)** | Scale detected from a single source, or steel derived from ratios |
| **Low (< 45%)** | Scale fell back to the default, or the vision pass was used on a scan |

Always review the **Assumptions** and **Annotated drawings** tabs before
relying on a number.

---

## Project layout

```
Construction-AI-estimation-main/
├── main.py                    # CLI entry point
├── config.yaml                # single source of truth for every knob
├── requirements.txt
├── render.yaml                # Render Blueprint (deploy config)
├── Procfile                   # gunicorn start command
├── Dockerfile                 # production container image
├── .dockerignore              # keeps the build context small
├── .python-version            # pins Python 3.11.9
├── .gitattributes             # normalises line endings to LF
├── src/
│   ├── config.py              # config loading + path resolution
│   ├── pdf_processor.py       # PyMuPDF page parsing (vector + raster)
│   ├── blueprint/
│   │   ├── models.py          # Element / RebarItem / PageAnalysis dataclasses
│   │   ├── vector.py          # segment/rect extraction, paper-mm conversion
│   │   ├── scale.py           # scale detection + consolidation
│   │   ├── schedule.py        # BBS / reinforcement schedule parsing
│   │   ├── elements.py        # geometry → element classification
│   │   ├── cv_elements.py     # OpenCV fallback for scanned pages
│   │   └── detector.py        # hybrid orchestrator
│   ├── estimator.py           # BOQ maths
│   ├── rates.py               # rate card + market multipliers
│   ├── report.py              # CSV + PDF reports
│   ├── annotate.py            # drawing overlays
│   ├── pipeline.py            # job orchestration
│   ├── deployment.py          # Flask API
│   └── yolo_trainer.py        # optional YOLO training
├── web/
│   ├── templates/index.html
│   └── static/{css,js}/
├── tools/
│   ├── selftest.py            # 15 end-to-end engine checks
│   ├── apitest.py             # API smoke tests
│   └── make_sample_drawing.py # generates a known-answer A1 test drawing
├── data/                      # uploads, rendered pages, dataset (git-ignored)
└── output/                    # estimates and reports (git-ignored)
```

---

## Testing

```bash
python tools/selftest.py    # engine: parsing, detection, quantities, totals
python tools/apitest.py     # HTTP API: upload, poll, result, downloads
```

`tools/selftest.py` builds a synthetic A1 drawing with a known layout (a 4 × 3
column grid, 6 rooms, 7 grid beams, and an 8-row reinforcement schedule) and
asserts the engine reproduces the hand-verified quantities:

| Quantity | Expected |
|---|---|
| Columns / slabs / walls / beams | 12 / 6 / 8 / 7 |
| Concrete | 36.05 m³ |
| Reinforcement | 2.574 t (4 diameter groups) |
| Formwork | 350.1 m² |
| Masonry | 49.68 m³ |

---

## Deployment (Render)

The repo ships with everything Render needs:

| File | Purpose |
|---|---|
| `render.yaml` | One-click Blueprint: build command, start command, health check, env vars |
| `Procfile` | `web: gunicorn src.deployment:app ...` for non-Blueprint deploys |
| `.python-version` | Pins Python 3.11.9 |
| `Dockerfile` | Optional container image (Render, Railway, Fly, any VPS) |
| `gunicorn` | Listing in `requirements.txt` (installs on Windows too, so local setup is unaffected) |

### 1. Push to GitHub

Render deploys from a Git repo:

```bash
git init
git add .
git commit -m "Construction AI Estimation v2.0.0"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### 2. Deploy

**Blueprint (recommended)** — Render Dashboard → **New → Blueprint** → pick the
repo. Render reads `render.yaml` and configures everything automatically,
including `healthCheckPath: /health`.

**Manual** — Render Dashboard → **New → Web Service** → pick the repo, then:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start command | `gunicorn src.deployment:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:$PORT` |
| Health check path | `/health` |

No environment variables are required — `PORT` is injected by Render and read
by the app.

### Docker

Prefer a container? The `Dockerfile` is a `python:3.11-slim` image that runs
as an unprivileged user (uid 10001), installs only `libgomp1` + `libglib2.0-0`
(the OpenCV/NumPy runtime deps), and serves with gunicorn:

```bash
docker build -t construction-ai .
docker run --rm -p 8000:8000 construction-ai
# http://127.0.0.1:8000
```

| Build arg / env | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Bind port; Render injects its own |
| `WEB_THREADS` | `8` | gunicorn `--threads` — raise this to scale |
| `WEB_TIMEOUT` | `120` | gunicorn worker timeout (seconds) |

`--workers` is **hard-locked to 1** inside the image. Setting an environment
variable cannot override it, because multi-worker state splitting would break
job polling (see below).

To use the image on Render, swap the `runtime: python` service in `render.yaml`
for the commented `runtime: docker` block at the bottom of that file.

> **Not built in this repo's CI.** No Docker daemon was available on the
> authoring machine, so the image is validated by static inspection only
> (instruction parsing, `COPY` sources, `CMD` expansion, health-check payload
> compilation). Run `docker build` once on your machine before trusting it in
> production.

### The `--workers 1` rule

Job state lives in memory (`_jobs` dict) and on the instance's local disk
(`output/estimates/<job_id>/`). Running **more than one worker would split that
state**, so a poll request could hit a worker that has never heard of the job.
Keep `--workers 1` and scale with `--threads` instead.

### Free-tier notes

- **Cold starts.** The free plan spins the instance down after ~15 minutes of
  inactivity; the next request takes ~30–60 s while the container boots.
- **Ephemeral disk.** `data/` and `output/` are wiped on every deploy/restart,
  so in-flight and past jobs disappear. Download the CSV/PDF while the result
  is on screen. For durable results, point `paths.*` at object storage.
- **512 MB RAM.** Fine for single-sheet drawings. Very large multi-page PDFs
  may need a paid plan (raise `pdf_processing.cv_max_dimension` down /
  `max_render_dimension` down to reduce memory).
- **No trained model.** `yolo_available` is `false`; the vector + schedule
  engine handles the take-off. See [Honest limitations](#honest-limitations).

### Verify a deploy

```bash
curl https://<your-app>.onrender.com/health
# {"status":"ok","version":"2.0.0","yolo_available":false,...}
```

Then open the URL, drop in a PDF, and confirm the BOQ renders.

---

## License

MIT — see [LICENSE](LICENSE).
