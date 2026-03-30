import numpy as np
import struct
# read header of a shard to understand format
with open('./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin', 'rb') as f:
    header = np.frombuffer(f.read(256*4), dtype=np.int32)
    print('magic:', header[0], hex(header[0]))
    print('version:', header[1])
    print('num_tokens:', header[2])
    # read a few tokens
    toks = np.frombuffer(f.read(20), dtype=np.uint16)
    print('first 10 tokens:', toks)

    toks = np.frombuffer(f.read(199999961), dtype=np.uint16)
    toks = np.frombuffer(f.read(20), dtype=np.uint16)
    print('last 10 tokens:', toks)
