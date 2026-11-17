<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Golden conditional-head scorer

`golden_score.py` compares a candidate's complete conditional F32 heads with a retained T=0 golden generation. It needs Python 3 and NumPy; the qualified workstation versions are Python 3.12.15 and NumPy 2.5.3. Run from the release root with `PYTHON` set to your Python executable containing NumPy. The scorer accepts caller-supplied capture directories; it has no dependency on a machine-specific corpus path.

The golden reference is the Hugging Face dataset
[anoane/anemone-ds41f1017-golden](https://huggingface.co/datasets/anoane/anemone-ds41f1017-golden): 435 cases
and 66,409 positions, each case a golden capture directory in the contract below, with its full F32 heads
(34.3 GB in all) and its compact reference `topk64.bin` (53 MB in all). Raw benchmark data and capture
observer/provenance artifacts are not bundled. These optional Python tools are separate from the Python-free
release build and inference executable.

```sh
"$PYTHON" tools/golden_score.py \
  GOLDEN_CASE CANDIDATE_CASE \
  --certain-margin=1 --free-token-ids=CANDIDATE_FREE_IDS_JSON --output=NEW_SCORE_JSON
```

The output is create-only. Omitting `--free-token-ids` uses `free_token_ids.json` in the candidate directory when present. The JSON includes per-position metrics, aggregate values and percentile distributions. Generation is greedy, but NLL, perplexity and KL use the ordinary unit-temperature distribution of the full logits. NLL is in nats; aggregate perplexity is `exp(mean NLL)`. These are likelihoods of the golden tokens.

Each capture directory contains:

| File | Contract |
|---|---|
| `metadata.json` | Schema version 1, vocabulary size, generated position count, explicit temperature 0 and alignment hashes/mode |
| `prompt_ids.json` | Complete native input ID array |
| `token_ids.json` | Golden continuation IDs, including a generated EOS when present |
| `conditioning_ids.json` | Actual consumed prompt followed by the golden continuation except its last ID |
| `logits.f32` | Little-endian F32, contiguous `[generated_positions, vocabulary_size]` |
| `native_drawn_ids.json` | Optional native candidate argmax IDs; checked against the full heads |
| `free_token_ids.json` | Optional independent free-generation continuation, used only for first-divergence reporting |

`alignment.mode` is `autoregressive_greedy` for a golden and `teacher_forced` for an aligned candidate. Golden self-comparison accepts the same directory. The alignment object binds `prompt_ids_sha256`, `continuation_ids_sha256` and `conditioning_ids_sha256`. Each hash is SHA256 of the compact ASCII JSON ID array (`json.dumps(ids, separators=(',', ':'), ensure_ascii=True)`, without a newline). The scorer checks complete shapes, finite values, ID ranges, exact conditioning, hashes and vocabulary equality. Producer records must establish that the conditioning witness was actually observed. A declaration or hash alone cannot prove execution.


`anemone capture` (docs/USAGE.md section 10.1) writes candidates in this layout from the engine's own production
path: each case one forced continuation of the golden tokens, every landed row's full F32 head as the sampler read it,
`conditioning_ids.json` the driver's committed history read at the case's last landing, and per row a witness (its
position and input token, the draw the lowest-ID argmax, the draw's logit bits). Its metadata adds a `capture` object
(the executable's SHA-256, the step's rows, placement, map) beside the alignment. `tools/golden/golden_capture_score.py`
scores every case of a capture root, in full or `--compact` mode, and aggregates them by position into one record.

Candidate row `i` predicts golden ID `i` after the complete prompt and golden IDs before `i`. Free-run logits after a divergent token do not have that conditioning. Record them separately. The R demonstration uses independent cold full-prefix commands and one native draw per command, with a resident model. Its producer checks every actual prefix and every rank before forming the scorer's canonical conditioning witness. This evaluates conditional heads; it does not claim a continuous candidate cache trajectory.

Top-1 ties select the lowest ID. The certain bucket uses a strict golden top-1 minus top-2 logit margin greater than `--certain-margin` (default 1). It reports the bucket's share and agreement within both certain and uncertain buckets. This threshold is a specified margin, not a calibration claim. Free divergence is zero-based in generated-token coordinates; a length mismatch is distinguished from equal completed continuations.

Full F32 data preserves tail probabilities for KL and all vocabulary values for RMS/max deltas. Vocabulary 129,280 requires 517,120 bytes per generated position, or 126.25 MiB for 256 positions. No top-k list plus logsumexp recovers the full-vocabulary KL or the RMS/max errors.

## Compact reference

`--write-compact` writes a golden's compact reference once, and `--compact` scores a candidate against it. The
golden directory then needs `topk64.bin` instead of `logits.f32`; the candidate still supplies its full heads.

```sh
"$PYTHON" tools/golden_score.py GOLDEN_CASE --write-compact=GOLDEN_CASE/topk64.bin
"$PYTHON" tools/golden_score.py GOLDEN_CASE CANDIDATE_CASE --compact --certain-margin=1 --output=NEW_SCORE_JSON
```

`topk64.bin` is little-endian: a 64-byte header, then six arrays, each contiguous over the positions.

| Part | Type | Contents |
|---|---|---|
| header | `8s` + 6 × `u32` + 32 zero bytes | magic `ANEMTOPK`, version 1, top_k (64, or the vocabulary when smaller), positions, vocabulary size, header bytes (64), 0 |
| `top_ids` | `int32 [positions, top_k]` | the highest golden logits first, ties by lowest ID, so `top_ids[i][0]` is the greedy token |
| `top_logprobs` | `float64 [positions, top_k]` | their log probabilities, the full mode's own F64 values |
| `logsumexp` | `float64 [positions]` | the row's maximum plus the log of its max-shifted exponential sum |
| `golden_logprob` | `float64 [positions]` | the golden token's log probability |
| `top1_margin` | `float64 [positions]` | the top-1 minus top-2 golden logit, as the full mode computes it |
| `tail_logprob` | `float64 [positions]` | the log of the probability outside `top_ids`, `-inf` when there is none |

The reader checks the header against the metadata, the size, the ID order, finite values and a total mass of 1 within
1e-9. The candidate's NLL and perplexity, top-1 agreement and the certain share and agreements are the full mode's
exact values. The compact mode adds three of its own:
- `top_k_agreement`: the mean overlap of the golden and candidate top-k ID sets, and how often the golden token is in
  the candidate's top k;
- `kl_top_k_tail`: KL(golden || candidate) over the golden's top-k IDs and one bucket for the rest of the vocabulary.
  It is a lower bound of the full-vocabulary KL, and equals it when top_k covers the vocabulary;
- the per-position rows of both.
The max/RMS logit deltas and the full-vocabulary KL need the full golden heads.
