# Portable GPU image — same behavior on 4060 local and cloud A100/H100.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl wget build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

# Copy requirements first for layer caching
COPY requirements/ /workspace/requirements/

RUN pip install --upgrade pip wheel setuptools \
    && pip install -r requirements/base.txt \
    && pip install -r requirements/pytorch.txt \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
    && pip install -r requirements/tensorflow.txt \
    && pip install -r requirements/ai_harness.txt \
    && pip install -r requirements/ml_extra.txt \
    && pip install -r requirements/dev.txt

COPY . /workspace

CMD ["/bin/bash"]
