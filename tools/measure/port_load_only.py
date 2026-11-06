#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loads the model with today's engine in a given layout and exits: the load that load_phase_sampler.py watches.

    run_with_port.sh port_load_only.py --layers-per-device 12,18,10 --ram-layers 33..39

Measurement tool: not part of the engine.
"""
import argparse
import sys

import torch

from kernel_time_trace import DEFAULT_CORPUS_DIRECTORY, DEFAULT_MODEL_DIRECTORY, load_model


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default = DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--corpus", default = DEFAULT_CORPUS_DIRECTORY)
    parser.add_argument("--ram-layers", default = "0")
    parser.add_argument("--ram-split", default = "0")
    parser.add_argument("--layers-per-device", default = None)
    parser.add_argument("--gigabytes-per-device", default = None)
    parser.add_argument("--cache-tokens", type = int, default = 65536)
    with torch.inference_mode():
        load_model(parser.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
