#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The token ids anemone generate printed against a capture's greedy reference (batch_four_card.sh, batch_five_card.sh).

    compare_generated_ids.py <capture folder> <generate log>

The prompt's ids ("prompt ids" line) must be the capture's input.token_ids, else no comparison and exit 1. Then each
generated id ("token <index> <id> <logit> <seconds>" lines) against generate.token_ids up to the first difference:
the reference's top-2 margin there and at every step before it, and the logit the engine gives each token against the
reference's for the same token at the same step (logits.all_rows). Exit 0 when the prompt is the capture's, whatever
the tokens: the VERDICT line says where they part. Measurement tool: not part of the engine.
"""
import json, pathlib, re, sys
import numpy

folder, log = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
records = {record["name"]: record for record in manifest["tensors"]}

def array(name):
    record = records[name]
    count = int(numpy.prod(record["shape"]))
    return numpy.fromfile(folder / "tensors.bin", dtype=record["dtype"], count=count,
                          offset=record["byte_offset"]).reshape(record["shape"])

text = log.read_text(encoding="utf8", errors="replace")
tokens = [(int(m.group(2)), float(m.group(3)), float(m.group(4)))
          for m in re.finditer(r"^token (\d+) (-?\d+) (\S+) (\S+)$", text, re.MULTILINE)]
prompt_match = re.search(r"^prompt ids((?: \d+)*)$", text, re.MULTILINE)
prompt = int(manifest["results"]["prompt_tokens"])
reference = array("generate.token_ids")
reference_prompt = [int(value) for value in array("input.token_ids")[0][:prompt]]
logits = array("logits.all_rows")[0]
engine_prompt = [int(value) for value in prompt_match.group(1).split()] if prompt_match else []
print(f"COMPARE prompt={prompt} engine_prompt={len(engine_prompt)} same_prompt_ids="
      f"{'yes' if engine_prompt == reference_prompt else 'no'} generated={len(tokens)} "
      f"reference_generated={len(reference)}")
if engine_prompt != reference_prompt or not tokens:
    print("VERDICT no comparison: the engine's prompt ids are not the capture's, or it generated nothing")
    sys.exit(1)
first_difference = None
for step, (engine_id, engine_logit, seconds) in enumerate(tokens[:len(reference)]):
    row = logits[prompt - 1 + step]
    order = numpy.argsort(-row, kind="stable")
    best, second = int(order[0]), int(order[1])
    margin = float(row[best] - row[second])
    same = step < len(reference) and engine_id == int(reference[step])
    print(f"STEP {step} engine={engine_id} reference={int(reference[step])} same={'yes' if same else 'no'} "
          f"reference_top2={best}:{row[best]:.6f},{second}:{row[second]:.6f} margin={margin:.6f} "
          f"engine_logit={engine_logit:.6f} reference_logit_of_engine_token={float(row[engine_id]):.6f} "
          f"seconds={seconds:.6f}")
    if not same:
        first_difference = (step, margin, int(reference[step]), engine_id)
        break
if first_difference is None:
    print(f"VERDICT same: the {min(len(tokens), len(reference))} generated ids the capture holds are the reference's")
else:
    step, margin, wanted, given = first_difference
    print(f"VERDICT differs at step {step}: reference {wanted}, engine {given}; the reference's top-2 margin "
          f"there {margin:.6f}")
