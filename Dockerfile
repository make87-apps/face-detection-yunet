ARG VERSION
FROM ghcr.io/make87/python3-base:$VERSION

WORKDIR /app

COPY . .

RUN python3 -m pip install -U pip && \
    python3 -m pip install .

CMD python3 -m app.main
