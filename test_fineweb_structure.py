import numpy as np
import struct
# read header of a shard to understand format
with open('./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin', 'rb') as f:
    header = np.frombuffer(f.read(256*4), dtype=np.int32)
    print('magic:', header[0], hex(header[0]))
    print('version:', header[1])
    print('num_tokens:', header[2])
    # read a few tokens
    # toks = np.frombuffer(f.read(20), dtype=np.uint16)
    # print('first 10 tokens:', toks)

    # toks = np.frombuffer(f.read(199999961), dtype=np.uint16)
    # toks = np.frombuffer(f.read(20), dtype=np.uint16)
    # print('last 10 tokens:', toks)

    toks = np.frombuffer(f.read(200000000), dtype=np.uint16)

    # Count 1s
    print('num 1s:', np.count_nonzero(toks == 1))

    # Indices of 1s
    ones_idx = np.flatnonzero(toks == 1)

    # Distances between consecutive 1s
    distances = np.diff(ones_idx)

    if distances.size > 0:
        mean = distances.mean()
        median = np.median(distances)

        # Mode (NumPy-only)
        vals, counts = np.unique(distances, return_counts=True)
        mode = vals[np.argmax(counts)]

        # Quartiles
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)

        min_d, max_d = min(distances), max(distances)

        print(f"mean distance: {mean}")
        print(f"median distance: {median}")
        print(f"mode distance: {mode}")
        print(f"Q1 (25%): {q1}")
        print(f"Q3 (75%): {q3}")
        print(f"Min: {min_d}")
        print(f"Max: {max_d}")
    else:
        print("Not enough 1s to compute distances.")