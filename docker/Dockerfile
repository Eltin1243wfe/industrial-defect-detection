# kept this on slim instead of alpine - torch wheels + alpine's musl libc
# has bitten me before, not worth the smaller image size headache
FROM python:3.11-slim

WORKDIR /app

# system deps needed for opencv/pillow image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

# checkpoints get mounted in at runtime, not baked into the image -
# keeps the image reusable across model versions without rebuilding
RUN mkdir -p checkpoints

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
