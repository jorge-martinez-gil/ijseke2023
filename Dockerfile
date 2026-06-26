# Reproducible environment for the semantic similarity ensemble benchmark.
#
#   docker build -t simbench .
#   docker run --rm -v "$PWD/results:/app/results" simbench
#
# The default command regenerates results/ (tables + figures). LR, LGP and TGP
# run out of the box; CGP runs only if `tengp` installs on your platform.
FROM python:3.10-slim

WORKDIR /app

# System deps for matplotlib / scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install the scientific stack; allow CGP's optional dependency to fail without
# breaking the image (the harness degrades gracefully).
RUN pip install --no-cache-dir numpy==1.25.2 pandas==1.5.3 scipy==1.9.1 \
        "scikit-learn==1.3.0" gplearn==0.4.2 "matplotlib>=3.5,<4" \
    && pip install --no-cache-dir tengp==0.4 || echo "tengp unavailable: CGP will be skipped"

COPY . .

ENV MPLCONFIGDIR=/tmp/mpl PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-m", "bench"]
