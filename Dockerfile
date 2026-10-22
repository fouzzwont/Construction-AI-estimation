# syntax=docker/dockerfile:1
#
# Construction AI Estimation - production image
#
#   docker build -t construction-ai .
#   docker run --rm -p 8000:8000 construction-ai
#
# Deploying on Render with this image? Use the commented "runtime: docker"
# block in render.yaml, or create the service manually and pick Docker.
#
# The app needs no external binaries: PyMuPDF renders PDFs natively, so
# Poppler/ImageMagick/Tesseract are deliberately NOT installed.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    WEB_THREADS=8 \
    WEB_TIMEOUT=120

WORKDIR /app

# libgomp1     -> OpenMP runtime pulled in by the numpy/opencv wheels
# libglib2.0-0 -> glib, linked by some OpenCV manylinux builds
# No libGL / libgtk: this project uses opencv-python-headless.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so editing source does not invalidate the wheel layer
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Runtime write targets. Hosts mount these onto ephemeral storage, so they
# must exist and be owned by the unprivileged user before the app starts.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p data/raw_pdfs data/processed_images data/uploads data/dataset \
             output/estimates output/reports logs models/yolo_weights \
 && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health', timeout=4).read()"

# --workers is hard-locked to 1 on purpose. Job state lives in an in-memory
# dict plus the instance-local disk; extra workers would split that state and
# poll requests would randomly 404. Scale concurrency with WEB_THREADS instead.
CMD exec gunicorn src.deployment:app \
      --workers 1 \
      --threads "${WEB_THREADS:-8}" \
      --timeout "${WEB_TIMEOUT:-120}" \
      --bind "0.0.0.0:${PORT:-8000}" \
      --access-logfile - \
      --error-logfile -
