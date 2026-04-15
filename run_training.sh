#!/bin/bash

# Small script that activates venv, trains an h-net, and closes the remote instance.
source .venv/bin/activate
TORCHINDUCTOR_FX_GRAPH_CACHE=1 MAX_WALLCLOCK_SECONDS=172800 COMPILE_MODEL=0 NUM_LAYERS=24 OUTER_LAYERS=4 MODEL_DIM=192 NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=2 python train_gpt_byte_hnet.py
shutdown -h now