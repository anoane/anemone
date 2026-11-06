#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Writes the C table that puts the kernel modules inside the program.

    embed_kernel_images.py <output .c> <build/kernels/<architecture>/<kernel name>.cubin> ...

Every module becomes one array of bytes and one entry of cuda_kernel_images, the table declared in
source/backend_cuda_kernel_images.h: architecture is the name of the module's directory, kernel_name its file name
without the suffix. The output is a build product (it lies under build/ and never enters the repository); it is
written to a temporary file first, so that a failed run leaves no half table.

Build tool: not part of the engine.
"""
import os
import sys

MODULE_SUFFIX = ".cubin"
BYTES_PER_LINE = 24
MINIMUM_ARGUMENT_COUNT = 2
EXIT_STATUS_USAGE = 2


def main():
    if len(sys.argv) < MINIMUM_ARGUMENT_COUNT:
        print(__doc__, file = sys.stderr)
        return EXIT_STATUS_USAGE
    output_path, module_paths = sys.argv[1], sorted(sys.argv[2:])
    lines = ["/* SPDX-License-Identifier: AGPL-3.0-or-later */",
             "/* Written by tools/build/embed_kernel_images.py: a build product, never edited. */",
             '#include "backend_cuda_kernel_images.h"', ""]
    entries = []
    for index, path in enumerate(module_paths):
        architecture = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)
        if not file_name.endswith(MODULE_SUFFIX) or not architecture:
            print(f"{path}: not <architecture>/<kernel name>{MODULE_SUFFIX}", file = sys.stderr)
            return 1
        with open(path, "rb") as handle:
            content = handle.read()
        if not content:
            print(f"{path}: an empty module", file = sys.stderr)
            return 1
        lines.append(f"static const unsigned char kernel_image_{index}[] = {{")
        for start in range(0, len(content), BYTES_PER_LINE):
            lines.append("    " + ", ".join(str(byte) for byte in content[start:start + BYTES_PER_LINE]) + ",")
        lines.append("};")
        entries.append(f'    {{"{architecture}", "{file_name[:-len(MODULE_SUFFIX)]}", kernel_image_{index}, '
                       f"sizeof(kernel_image_{index})}},")
    lines.append("")
    # A table needs one entry to be valid C: with no kernel the count is 0 and the entry is never read.
    lines.append("const struct cuda_kernel_image cuda_kernel_images[] = {")
    lines += entries if entries else ['    {"", "", 0, 0},']
    lines.append("};")
    lines.append(f"const uint32_t cuda_kernel_image_count = {len(entries)};")
    temporary_path = output_path + ".partial"
    with open(temporary_path, "w", encoding = "ascii") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(temporary_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
