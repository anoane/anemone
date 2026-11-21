#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Embed the reviewed catalog; fail when a CLI flag disappears from the catalog."""
import argparse
import ast
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "tools/cli/KNOBS.md"
OUTPUT = ROOT / "generated/engine/cli/cli_knobs.inc"


def generate():
    rows = []
    for line in CATALOG.read_text().split("<!-- generated-axis-table -->", 1)[0].splitlines():
        if line.startswith("| ANEMONE_"):
            name, values, scope, evidence = [field.strip() for field in line.strip("|").split("|")]
            rows.append([name, values, "unset", "Diagnostic environment switch; never enabled by an axis.",
                         "environment, generate", scope, "draft-only" if "draft-only" in scope else "same bits",
                         "diagnostic only; no strict CLI range/duplicate validation", evidence,
                         "Unmeasured overhead; " + evidence])
        if re.match(r"\| --[a-z]", line):
            fields = [field.strip() for field in line.strip("|").split("|")]
            if len(fields) != 10:
                raise ValueError(f"catalog row has {len(fields)} columns: {fields[0]}")
            rows.append(fields)
    names = {row[0] for row in rows if "unavailable in S" not in row[4]}
    source = (ROOT / "source/engine/cli/commands.c").read_text()
    table = source.split("static const struct number_option options[] = {", 1)[1].split("\n\t};", 1)[0]
    inventory = json.loads((ROOT / "tools/cli/inventory.json").read_text())
    actual_numbers = []
    for entry in re.findall(r"\{([^{}]+)\}", table):
        fields = [field.strip() for field in entry.replace("\n", " ").split(",")]
        actual_numbers.append(dict(zip(("name", "variable", "least", "most", "count", "commands"),
                                       [fields[0].strip('"').strip("-="), *fields[1:]])))
    expected_numbers = [{key: value for key, value in entry.items() if key != "line"} for entry in inventory]
    renamed_identifiers = {'ANEMONE_BACKEND_MAXIMUM_CARDS': 'BACKEND_MAXIMUM_CARDS', 'ANEMONE_CHAT_TEMPLATE_HIGHEST_EFFORT': 'DS41F_CHAT_TEMPLATE_HIGHEST_EFFORT', 'ANEMONE_DENSE_STAGED_MOST_OUTPUTS': 'DENSE_STAGED_MOST_OUTPUTS', 'ANEMONE_DENSE_STAGED_OUTPUTS': 'DENSE_STAGED_OUTPUTS', 'ANEMONE_DRAFT_DEPTH_ADAPTIVE': 'DRAFT_DEPTH_ADAPTIVE', 'ANEMONE_DRAFT_DEPTH_FIXED': 'DRAFT_DEPTH_FIXED', 'ANEMONE_DRAFT_MOST_COST_US': 'DRAFT_MOST_COST_US', 'ANEMONE_DRAFT_MOST_PROBE_ROUNDS': 'DRAFT_MOST_PROBE_ROUNDS', 'ANEMONE_DRAFT_PROBABILITY_SCALE': 'DRAFT_PROBABILITY_SCALE', 'ANEMONE_EXL3_TC_ROTATION_THREE_WAY': 'DS41F_EXL3_TC_ROTATION_THREE_WAY', 'ANEMONE_LINEAR_MAXIMUM_ROWS': 'LINEAR_MAXIMUM_ROWS', 'ANEMONE_LINEAR_ORDER_REFERENCE': 'LINEAR_ORDER_REFERENCE', 'ANEMONE_LINEAR_PROFILE_SPEED': 'LINEAR_PROFILE_SPEED', 'ANEMONE_LIVE_STATS_MOST_INTERVAL_MS': 'LIVE_STATS_MOST_INTERVAL_MS', 'ANEMONE_LOADER_MAXIMUM_READER_THREADS_PER_COPY': 'LOADER_MAXIMUM_READER_THREADS_PER_COPY', 'ANEMONE_MODEL_CONTEXT_MOST': 'DS41F_CONTEXT_MOST', 'ANEMONE_DS41F_DRAFT_BLOCK': 'DS41F_DRAFT_BLOCK', 'ANEMONE_MODEL_WIDE_MOST_ROWS': 'STEP_WIDE_MOST_ROWS', 'ANEMONE_ORDERED_PUSH_BF16': 'ORDERED_PUSH_BF16', 'ANEMONE_ORDERED_PUSH_FP32': 'ORDERED_PUSH_FP32', 'ANEMONE_PREFILL_DECODE_MOST_ROUNDS': 'PREFILL_DECODE_MOST_ROUNDS', 'ANEMONE_PREFILL_ORDER_SHORT_ROWS': 'PREFILL_ORDER_SHORT_ROWS', 'ANEMONE_PROGRAM_BATCH_MOST_ROWS': 'PROGRAM_BATCH_MOST_ROWS', 'ANEMONE_SERVER_CACHE_GLOBAL': 'SERVER_CACHE_GLOBAL', 'ANEMONE_SERVER_CACHE_SESSION': 'SERVER_CACHE_SESSION', 'ANEMONE_SERVER_MOST_GRACE_SECONDS': 'SERVER_MOST_GRACE_SECONDS', 'ANEMONE_SERVER_MOST_IDLE_SAVE_SECONDS': 'SERVER_MOST_IDLE_SAVE_SECONDS', 'ANEMONE_SERVER_MOST_PORT': 'SERVER_MOST_PORT', 'ANEMONE_SERVER_MOST_WAITING': 'SERVER_MOST_WAITING', 'ANEMONE_SERVER_PREFIX_COLD_PLAN': 'SERVER_PREFIX_COLD_PLAN', 'ANEMONE_SERVING_BATCH_COSTS': 'SERVING_BATCH_COSTS', 'ANEMONE_SERVING_CONE_MARGIN': 'SERVING_CONE_MARGIN', 'ANEMONE_SERVING_DRAFT_MOST': 'SERVING_DRAFT_MOST', 'ANEMONE_SERVING_SLOTS_MOST': 'SERVING_SLOTS_MOST', 'ANEMONE_STEP_BATCH_MAXIMUM_SESSIONS': 'STEP_BATCH_MAXIMUM_SESSIONS', 'ANEMONE_STEP_DRAFT_MOST': 'STEP_DRAFT_MOST', 'ANEMONE_STEP_PIPELINE_MOST': 'STEP_PIPELINE_MOST', 'ANEMONE_TIMING_HOOKS_ENABLED': 'TIMING_HOOKS_ENABLED', 'ANEMONE_TP_LOAD_MAXIMUM_WORKERS': 'TP_LOAD_MAXIMUM_WORKERS', 'ANEMONE_TP_MOST_GRAPH_PARTS': 'DS41F_TP_MOST_GRAPH_PARTS', 'ANEMONE_V41_GATHER_BALANCE_WEIGHTED': 'DS41F_GATHER_BALANCE_WEIGHTED', 'ANEMONE_V41_SPLIT_JOINT_DOWN': 'DS41F_SPLIT_JOINT_DOWN', 'ANEMONE_V41_TP_EXCHANGE_CE_RINGS': 'DS41F_TP_EXCHANGE_CE_RINGS', 'ANEMONE_V41_TP_SPLIT_MOST_ROWS': 'DS41F_TP_SPLIT_MOST_ROWS'}
    for entry in expected_numbers:
        for key, value in entry.items():
            entry[key] = re.sub(r'\b[A-Za-z_][A-Za-z_0-9]*\b', lambda match: renamed_identifiers.get(match.group(), match.group()), value)
    if actual_numbers != expected_numbers:
        raise ValueError("numeric parser changed; review catalog values/gates and refresh tools/cli/inventory.json")
    source += (ROOT / "source/engine/cli/cli_axes.c").read_text()
    source += (ROOT / "generated/engine/cli/cli_axes.inc").read_text()
    for filename in ("cli_stats_client.c", "server_keys.c", "replicas.c", "storage_autotune.c",
                     "storage_autotune_load.c", "engram_autotune.c", "cli_golden_capture.c"):
        source += next((ROOT / "source").rglob(filename)).read_text()
    actual = (set(re.findall(r'"(--[a-z][a-z_0-9-]*)=', source)) |
              set(re.findall(r'"(--[a-z][a-z_0-9-]*)"', source)) | {"--once", "--user"})
    missing = actual - names
    if missing:
        raise ValueError(f"uncatalogued CLI flags: {sorted(missing)}")
    # REUSE-IgnoreStart: the tag below is the generated file's, not this program's
    lines = ["/* SPDX-License-Identifier: AGPL-3.0-or-later */",
             # REUSE-IgnoreEnd
             "/* Generated by tools/cli/generate_help.py from tools/cli/KNOBS.md; do not edit. */",
             "#ifndef CLI_KNOBS_INC", "#define CLI_KNOBS_INC", "",
             "struct cli_knob_help { const char *name, *detail; };", "",
             "static const struct cli_knob_help cli_knobs[] = {"]
    labels = ("values", "default", "meaning", "commands", "placement/recipe", "exactness", "gates",
              "interactions", "measured effect/source")
    for fields in rows:
        lines.append("\t{ " + json.dumps(fields[0]) + ",")
        for label, value in zip(labels, fields[1:]):
            text = f"  {label}: {value}\n"
            # Adjacent C literals keep every physical line and literal bounded.
            parts = textwrap.wrap(text[:-1], width=95, break_long_words=True, break_on_hyphens=False,
                                  replace_whitespace=False, drop_whitespace=False)
            for index, part in enumerate(parts):
                lines.append("\t  " + json.dumps(part + ("\n" if index == len(parts) - 1 else "")))
        lines[-1] += " },"
    lines += ["};", "", "#endif", ""]
    renamed_environment = {'ANEMONE_DIAG_DRAFT_PROBE': 'DIAG_DRAFT_PROBE', 'ANEMONE_DIAG_DRAFT_SKIP_FFN': 'DIAG_DRAFT_SKIP_FFN', 'ANEMONE_DIAG_DRAFT_STAGES': 'DIAG_DRAFT_STAGES', 'ANEMONE_DIAG_STATE_DIGESTS': 'DIAG_STATE_DIGESTS', 'ANEMONE_SOCKET_DESCRIPTOR': 'INFERENCE_SOCKET_FD', 'ANEMONE_TRANSPORT_CHECK_ONLY': 'DEVICE_TRANSPORT_CHECK'}
    emitted = "\n".join(lines)
    for old, name in renamed_environment.items():
        emitted = re.sub(r'\b' + re.escape(old) + r'\b', name, emitted)
    return emitted, len(rows), len(actual)


def c_content(source):
    """Compare generated C content while accepting clang-format layout and literal splitting."""
    tokens = re.findall(r'"(?:\\.|[^"\\])*"(?:\s*"(?:\\.|[^"\\])*")*|[^\s]', source)
    return [("literal", ast.literal_eval("(" + token + ")")) if token.startswith('"')
            else ("token", token) for token in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated, count, actual = generate()
    if args.check:
        if not OUTPUT.exists() or c_content(OUTPUT.read_text()) != c_content(generated):
            raise SystemExit("generated knob help is stale; run tools/cli/generate_help.py")
    elif OUTPUT.exists() and c_content(OUTPUT.read_text()) == c_content(generated):
        OUTPUT.touch()
    else:
        OUTPUT.write_text(generated)
    print(f"knob catalog: {count} rows; {actual} CLI names covered; generated help current")


if __name__ == "__main__":
    main()
