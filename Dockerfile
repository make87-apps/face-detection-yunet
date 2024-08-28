FROM debian:bullseye-slim AS download-whl-artifacts

RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y))

RUN mkdir -p -m 755 /etc/apt/keyrings && \
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt update && \
    apt install gh -y

RUN --mount=type=secret,id=GITHUB_TOKEN \
    gh auth login --with-token < /run/secrets/GITHUB_TOKEN

WORKDIR /tmp
RUN gh release download --repo make87/make87 py3/lib/0.0.0-dev -p 'make87-0.0.1-py3-none-any.whl'
RUN gh release download --repo make87/make87 py3/msg/0.0.0-dev -p 'make87_messages-0.0.1-py3-none-any.whl'

FROM python:3.11-slim-bullseye AS build-python-base-image

COPY --from=download-whl-artifacts /tmp/make87-0.0.1-py3-none-any.whl /tmp/make87-0.0.1-py3-none-any.whl
COPY --from=download-whl-artifacts /tmp/make87_messages-0.0.1-py3-none-any.whl /tmp/make87_messages-0.0.1-py3-none-any.whl


ARG VIRTUAL_ENV=/app/venv

RUN python3 -m venv --copies ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /app

# Required to have `cmake` for armv7 build (otherwise `ninja` install fails)
RUN apt update && apt install -y \
    build-essential \
    cmake

RUN  python3 -m pip install -U pip setuptools wheel && \
     python3 -m pip install \
        /tmp/make87-0.0.1-py3-none-any.whl \
        /tmp/make87_messages-0.0.1-py3-none-any.whl

FROM python:3.11-slim-bullseye AS python-app-image
LABEL org.opencontainers.image.source=https://github.com/make87/make87

ARG VIRTUAL_ENV=/app/venv

COPY --from=build-python-base-image ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /app

COPY . .

RUN python3 -m pip install -U pip && \
    python3 -m pip install .

CMD python3 -m app.main
