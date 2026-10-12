# syntax=docker/dockerfile:1
#
# Outis API server. VARIANT selects the PyTorch build:
#   cpu  CPU-only PyTorch from the PyTorch index (linux/amd64 and linux/arm64)
#   gpu  the locked PyPI PyTorch with its CUDA libraries (linux/amd64; needs an
#        NVIDIA driver that supports the locked CUDA version and the NVIDIA
#        Container Toolkit on the host)
#
# Every other dependency comes from uv.lock. The pinned detector checkpoint is
# baked into the image, and the container runs with HF_HUB_OFFLINE=1, so it
# never reaches the network. See docs/deployment.md.

ARG VARIANT=cpu

FROM ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 AS uv

FROM python:3.13-slim-trixie@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS build
ARG VARIANT
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/outis
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN uv venv "$VIRTUAL_ENV"
# The runtime closure from the lockfile, hash-checked. The CPU image drops the
# locked torch and its CUDA packages and installs the same torch version from
# the CPU index instead (that index publishes no upload times, so --no-config
# skips the project's exclude-newer cooldown, as CI does).
RUN <<EOF
set -eu
case "$VARIANT" in
    cpu) excluded=$(python - <<'PY'
import re, tomllib
for package in tomllib.load(open("uv.lock", "rb"))["package"]:
    if re.fullmatch(r"torch|triton|nvidia-.*|cuda-.*", package["name"]):
        print(f"--no-emit-package={package['name']}")
    if package["name"] == "torch":
        print(f"torch=={package['version']}", file=open("torch.txt", "w"))
PY
    ) ;;
    gpu) excluded="" ;;
    *) echo "VARIANT must be cpu or gpu, not $VARIANT" >&2; exit 1 ;;
esac
uv export --frozen --no-dev --extra pdf --extra parquet --no-emit-project --no-header --no-annotate $excluded -o requirements.txt
if [ "$VARIANT" = cpu ]; then
    uv pip install --no-config --no-deps -r torch.txt --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --no-deps --require-hashes -r requirements.txt
uv pip install --no-deps .
EOF
# Bake in the pinned default checkpoint so the container needs no network.
ENV HF_HOME=/opt/hf
# huggingface_hub writes its tree listing owner-only; without it the non-root
# runtime user would try to fetch the listing again and fail offline.
RUN /opt/outis/bin/python -c 'from outis.paths import resolve_model_dir; print(resolve_model_dir())' \
    && chmod -R a+rX /opt/hf

FROM python:3.13-slim-trixie@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
ARG VARIANT
LABEL org.opencontainers.image.title="outis" \
      org.opencontainers.image.description="Local clinical PHI detection, redaction and consistent synthetic replacement" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/ewhauser/outis" \
      io.github.ewhauser.outis.variant="${VARIANT}"
RUN useradd --system --uid 10001 --home-dir /home/outis --create-home outis
COPY --from=build /opt/outis /opt/outis
COPY --from=build /opt/hf /opt/hf
ENV PATH=/opt/outis/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1
USER outis
WORKDIR /home/outis
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
# Binding beyond loopback requires OUTIS_API_KEY; the server refuses to start
# without it.
ENTRYPOINT ["outis"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
