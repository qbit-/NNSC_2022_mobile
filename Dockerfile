FROM mambaorg/micromamba:0.19.1

COPY --chown=$MAMBA_USER:$MAMBA_USER env_docker.yaml /tmp/env.yaml
COPY bin/adb-linux /usr/local/bin/adb
USER root
RUN chmod +x /usr/local/bin/adb
USER mambauser

RUN micromamba install -y -f /tmp/env.yaml && \
    micromamba clean --all --yes

