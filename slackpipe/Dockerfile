# Solver-only development image; Megatron uses slackpipe-dev separately.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ cmake ninja-build make git clang-format \
    python3 python3-pytest python3-numpy python3-pandas python3-matplotlib \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/Megatron-LM
CMD ["/bin/bash"]
