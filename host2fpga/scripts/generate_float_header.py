#!/usr/bin/env python2
"""
scripts/generate_header.py

Auto-generate a combined C header from weight/bias .txt files under quant_data/,
robust to trailing commas in shape comments like “# Shape: (6,)”.
"""

import os
import glob
import re
import sys

def normalize_shape_line(line):
    # remove any comma immediately before the closing ')'
    return re.sub(r',\s*\)', ')', line)

def parse_shape_line(line):
    """Extracts integer dims from a '# Shape: (...)' line."""
    # first normalize trailing commas
    line = normalize_shape_line(line)
    nums = re.findall(
        r'\(\s*([0-9]+)'
        r'(?:\s*,\s*([0-9]+))?'
        r'(?:\s*,\s*([0-9]+))?'
        r'(?:\s*,\s*([0-9]+))?'
        r'\s*\)',
        line
    )
    if not nums:
        return None
    # take first match and filter out empty captures
    return [float(x) for x in nums[0] if x]

def generate_header(txt_files, header_path):
    lines_out = []
    lines_out.append("// Auto-generated weights & biases\n")
    lines_out.append("#ifndef WEIGHTS_BIAS_FLOAT_H\n#define WEIGHTS_BIAS_FLOAT_H\n\n")

    for txt in sorted(txt_files):
        with open(txt) as f:
            file_lines = f.read().splitlines()

        # Find the shape comment
        shape = None
        for l in file_lines:
            if l.strip().startswith("# Shape"):
                shape = parse_shape_line(l)
                break
        if shape is None:
            print(f"Error: no shape found in {txt}", file=sys.stderr)
            sys.exit(1)

        # Gather numeric data
        data = []
        for l in file_lines:
            if not l.strip() or l.strip().startswith("#"):
                continue
            for part in l.split(","):
                part = part.strip()
                if part:
                    data.append(float(part))

        # Verify count matches shape product
        expected = 1
        for d in shape:
            expected *= d
        if len(data) != expected:
            print(f"Error: {txt} has {len(data)} values but shape product is {expected}", file=sys.stderr)
            sys.exit(1)

        # Build variable names
        stem = os.path.splitext(os.path.basename(txt))[0]
        var = re.sub(r'[^0-9A-Za-z_]', '_', stem).upper()

        # Emit DATA array
        lines_out.append(f"static const int32_t {var}_DATA[{expected}] = {{\n")
        for i, v in enumerate(data):
            lines_out.append(f"  {v},")
            if (i + 1) % 10 == 0:
                lines_out.append("\n")
        lines_out.append("\n};\n")

        # Emit SHAPE array
        dims = ", ".join(str(d) for d in shape)
        lines_out.append(f"static const int {var}_SHAPE[] = {{ {dims} }};\n\n")

    lines_out.append("#endif // WEIGHTS_BIAS_FLOAT_H\n")

    # Write out
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("".join(lines_out))

    print(f"Generated {header_path} from: {', '.join(txt_files)}")

def main():
    folder = "float-data"
    patterns = [
        f"{folder}/conv*_weight*.txt",
        f"{folder}/conv*_bias*.txt",
        f"{folder}/fc*_weight*.txt",
        f"{folder}/fc*_bias*.txt",
    ]
    txts = []
    for p in patterns:
        txts.extend(glob.glob(p))
    if not txts:
        print("No matching TXT files found under quant_data/", file=sys.stderr)
        sys.exit(1)
    generate_header(txts, "data/weights_bias_float.h")

if __name__ == "__main__":
    main()
