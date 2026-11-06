<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Golden conditional-head scorer

`golden_score.py` compares a candidate's complete conditional F32 heads with a retained T=0 golden generation. It needs Python 3 and NumPy; the qualified workstation versions are Python 3.12.15 and NumPy 2.5.3. Run from the release root with `PYTHON` set to your Python executable containing NumPy. The scorer accepts caller-supplied capture directories; it has no dependency on a machine-specific corpus path.

The scorer source and this manual ship together. The external golden
corpus (about 68 GB), raw benchmark data and capture observer/provenance artifacts
are not bundled, and this manual specifies no public download for them. Supply
captures satisfying the contract below. These optional Python tools are separate
from the Python-free release build and inference executable.

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

Candidate row `i` predicts golden ID `i` after the complete prompt and golden IDs before `i`. Free-run logits after a divergent token do not have that conditioning. Record them separately. The R demonstration uses independent cold full-prefix commands and one native draw per command, with a resident model. Its producer checks every actual prefix and every rank before forming the scorer's canonical conditioning witness. This evaluates conditional heads; it does not claim a continuous candidate cache trajectory.

Top-1 ties select the lowest ID. The certain bucket uses a strict golden top-1 minus top-2 logit margin greater than `--certain-margin` (default 1). It reports the bucket's share and agreement within both certain and uncertain buckets. This threshold is a specified margin, not a calibration claim. Free divergence is zero-based in generated-token coordinates; a length mismatch is distinguished from equal completed continuations.

Full F32 data preserves tail probabilities for KL and all vocabulary values for RMS/max deltas. Vocabulary 129,280 requires 517,120 bytes per generated position, or 126.25 MiB for 256 positions. A top-1024 list plus logsumexp cannot recover exact full-vocabulary tail KL or RMS/max errors.
