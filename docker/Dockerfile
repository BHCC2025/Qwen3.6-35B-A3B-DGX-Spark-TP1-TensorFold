# docker/Dockerfile — the image this recipe runs: TensorFold v0.6.2 installed from its upstream repository
# (https://github.com/ashhart/TensorFold, Apache-2.0) at a pinned commit, unmodified, inside NVIDIA's PyTorch container,
# as TensorFold's README describes for NVIDIA GPUs. ./setup.sh builds it once on the head and copies it to the workers.
# The checkout stays in /opt/tensorfold, so TensorFold's own tools (bench_concurrent.py) run from the same commit.
FROM nvcr.io/nvidia/pytorch:26.07-py3
ARG TENSORFOLD_REPO=https://github.com/ashhart/TensorFold.git
ARG TENSORFOLD_COMMIT=56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21
RUN git clone --quiet --filter=blob:none "$TENSORFOLD_REPO" /opt/tensorfold \
 && git -C /opt/tensorfold checkout --quiet --detach "$TENSORFOLD_COMMIT" \
 && python -m pip install --no-cache-dir /opt/tensorfold \
 && python -c "import torch, tensorfold; print('torch', torch.__version__, 'tensorfold', tensorfold.__version__)"
ENV TENSORFOLD_NO_UPDATE_CHECK=1
