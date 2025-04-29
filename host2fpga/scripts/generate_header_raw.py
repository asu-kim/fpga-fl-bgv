#!/usr/bin/env python3
import os, re
import numpy as np

RAW_DIR = "quant_data/raw_params"
OUT_HDR = "data/weights_bias_raw_params.h"

def parse_shape(line):
    # "# Shape: (6,1,5,5)" -> (6,1,5,5)
    nums = re.findall(r"\d+", line)
    return tuple(map(int, nums))

def load_file(path):
    txt = open(path, "r").read()
    # look for a shape comment
    first = txt.splitlines()[0].strip()
    if first.startswith("# Shape:"):
        shape = parse_shape(first)
    else:
        shape = None

    # strip out comment-lines entirely, then brackets/commas
    data_lines = [
        ln for ln in txt.splitlines()
        if not ln.strip().startswith("#")
    ]
    blob = " ".join(data_lines)
    blob = re.sub(r"[\[\],]", " ", blob)

    # parse floats
    arr = np.fromstring(blob, sep=" ", dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr, arr.shape

def dtype_for(name):
    if name.endswith("_zp"):
        return "int32_t"
    # everything else float
    return "float"

def format_c(name, dtype, data, shape):
    out = []
    dims = ", ".join(str(d) for d in shape)
    out.append(f"static const int   {name}_SHAPE[] = {{ {dims} }};")
    out.append(f"static const {dtype} {name}_DATA[{data.size}] = {{")
    flat = data.flatten()
    for i in range(0, flat.size, 16):
        chunk = flat[i:i+16]
        if dtype == "float":
            vals = ", ".join(f"{x:.6e}" for x in chunk)
        else:
            vals = ", ".join(str(int(x)) for x in chunk)
        out.append("    " + vals + ("," if i+16 < flat.size else ""))
    out.append("};\n")
    return out

def main():
    lines = [
        "/* AUTO-GENERATED: raw weights, biases & quant params */",
        "#pragma once\n"
    ]

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".txt"):
            continue
        base = fname[:-4]           # strip .txt
        name = base.upper().replace("-", "_")
        path = os.path.join(RAW_DIR, fname)

        data, shape = load_file(path)
        dt = dtype_for(base)
        lines += format_c(name, dt, data, shape)

    with open(OUT_HDR, "w") as f:
        f.write("\n".join(lines))

    print(f"→ wrote {OUT_HDR}")

if __name__ == "__main__":
    main()
