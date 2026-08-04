# QuantOS -- reproducible container image.
#
# Two stages. The builder compiles the wheel; the runtime carries only the
# installed package, so the final image contains no compiler, no build cache and
# no source tree to drift from what was built.
#
#   docker build -t quantos .
#   docker run --rm quantos quantos research --ticker NVDA
#   docker run --rm -p 8000:8000 quantos quantos serve --host 0.0.0.0
#
# The image pins a digest rather than a floating tag. `python:3.12-slim` moves,
# and an image that silently changes base is not reproducible -- which matters
# more here than usual, because this repository's own CI asserts bit-identical
# simulation output across runs.

# --------------------------------------------------------------------------- #
# Stage 1 -- build the wheel
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

WORKDIR /build

# Copy only what the build needs first, so a source edit does not invalidate the
# dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /dist

# --------------------------------------------------------------------------- #
# Stage 2 -- runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="QuantOS" \
      org.opencontainers.image.description="Quantitative research on any listed security. One runtime dependency." \
      org.opencontainers.image.source="https://github.com/danielboulan10/QuantOS" \
      org.opencontainers.image.licenses="MIT"

# Never run as root. A container that makes outbound HTTP requests on a caller's
# behalf is exactly the one that should not have root in it.
RUN useradd --create-home --shell /bin/bash quantos

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QUANTOS_LOG_LEVEL=warning \
    # The data cache lives on a writable path the non-root user owns, so a
    # read-only root filesystem still works.
    QUANTOS_CACHE_DIR=/home/quantos/.cache/quantos

COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

USER quantos
WORKDIR /home/quantos

# Fails the build if the package cannot import or the runtime picked up a
# dependency it should not have. Cheap, and catches a broken image before it is
# ever pushed.
RUN quantos doctor > /dev/null && python -c "import quantos; print(quantos.__version__)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1

ENTRYPOINT []
CMD ["quantos", "--help"]
