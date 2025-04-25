#!/usr/bin/env python3
import os
import torch
import numpy as np

# ——— Configuration ———
PTH_FILE   = "fake_quant_fp_lenet_weight.pth"
OUT_DIR    = "quant-data"
# layer names in order
LAYERS = [
    ("conv1", True),
    ("conv2", True),
    ("fc1",   False),
    ("fc2",   False),
    ("fc3",   False),
]
# ——— End configuration ———

os.makedirs(OUT_DIR, exist_ok=True)

state = torch.load(PTH_FILE, map_location="cpu")

def write_txt(fname, arr):
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write(f"# Shape: {tuple(arr.shape)}\n")
        f.write("# C-style row major order\n")
        flat = arr.flatten()
        f.write(",".join(map(str, flat)))

for layer, is_conv in LAYERS:
    # fetch and quantize weight
    w      = state[f"{layer}.weight"].numpy()
    scale  = state[f"{layer}.weight_fake_quant.scale"].numpy()
    zp     = state[f"{layer}.weight_fake_quant.zero_point"].numpy()
    q_w    = np.zeros_like(w, dtype=np.int8)

    if is_conv:
        # conv: per-out-channel quant
        OC = w.shape[0]
        for oc in range(OC):
            # broadcast over [IC,H,W]
            q = np.round((w[oc] - zp[oc]) / scale[oc])
            q = np.clip(q, -128, 127)
            q_w[oc] = q
    else:
        # FC: each row has its own scale/zp
        OD = w.shape[0]
        for r in range(OD):
            q = np.round((w[r] - zp[r]) / scale[r])
            q = np.clip(q, -128, 127)
            q_w[r] = q

    write_txt(f"{layer}_weight_int8.txt", q_w)

    # fetch and quantize bias
    b         = state[f"{layer}.bias"].numpy()
    post_scale= state[f"{layer}.activation_post_process.scale"].item()
    # same per-channel rule as weight
    q_b = np.zeros_like(b, dtype=np.int8)
    C  = b.shape[0]
    for oc in range(C):
        denom = scale[oc] * post_scale
        q   = int(round(b[oc] / denom))
        q_b[oc] = np.clip(q, -128, 127)

    write_txt(f"{layer}_bias_int8.txt", q_b)

print(f"All files written into ./{OUT_DIR}/")    
