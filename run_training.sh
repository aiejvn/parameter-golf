#!/bin/bash

# Small script that activates venv, trains an h-net, and closes the remote instance.
source .venv/bin/activate
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1 MAX_WALLCLOCK_SECONDS=172800 MODEL_DIM=320  python train_gpt_byte_hnet.py
shutdown -h now