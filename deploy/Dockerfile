FROM python@sha256:ec948fa5f90f4f8907e89f4800cfd2d2e91e391a4bce4a6afa77ba265bc3a2fe

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    BAGCOUNTER_DATA_DIR=/data \
    OPENCV_FFMPEG_LOGLEVEL=8 \
    BAGCOUNTER_MODEL_DIR=/app/models/bag_v8s_int8

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY models/ models/

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin bagcounter \
 && mkdir -p /data && chown bagcounter:bagcounter /data
USER bagcounter
VOLUME ["/data"]
EXPOSE 8080 9101

ENTRYPOINT ["python", "-m", "bagcounter.cli"]
CMD ["api"]
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 CMD ["python", "-m", "bagcounter.cli", "healthcheck", "api"]
