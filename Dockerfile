FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git cmake build-essential \
    libgl1 libglib2.0-0 awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt tyro

# Copy training code + retro integration data (ROM excluded)
COPY train.py pretrain.py evaluate.py ./
COPY envs/ envs/
COPY retro_data/FE776-Snes/data.json retro_data/FE776-Snes/scenario.json \
     retro_data/FE776-Snes/metadata.json retro_data/FE776-Snes/rom.sha \
     retro_data/FE776-Snes/

# Save states are copied in; ROM is downloaded from S3 at runtime
COPY retro_data/FE776-Snes/*.state retro_data/FE776-Snes/

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
