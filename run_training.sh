#!/bin/bash

# Small script that activates venv, trains an h-net, and closes the remote instance.
source .venv/bin/activate
python train_gpt_byte_hnet.py
shutdown -h now