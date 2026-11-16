<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Prometheus metrics

Enable `--metrics=1` on `anemone serve` and scrape `GET /metrics` on the serving
listener. The exporter defaults to the vLLM V1 format described below. The
endpoint remains disabled by default (`--metrics=0`). When disabled, GET, POST
and GET-with-body requests to `/metrics` behave exactly like an unknown path,
including JSON statistics deltas. With `--key_file`, the enabled endpoint uses
serving bearer authentication and the same `--stats_users` ACL as JSON stats.
Without a key file, it is anonymous. No separate listener is introduced.

The response is Prometheus text exposition 0.0.4. Invalid UTF-8 or raw carriage
returns in the served model identity refuse startup when metrics are enabled.
Exporter failures return HTTP 500 without entering generation or per-user
failure accounting. Labels never contain prompts, credentials, user IDs,
request IDs, session IDs or arbitrary request paths.

## Pinned compatibility target

The intended mainstream engine is inferred to be **vLLM**, based on the requested
vLLM-style metrics and its OpenAI/Anthropic API combination. This inference does
not identify an unnamed caller with certainty. The compatibility target is
[vLLM V1 at 1ecc9cc4ce6626352df39e4364579ec376c10761](https://github.com/vllm-project/vllm/tree/1ecc9cc4ce6626352df39e4364579ec376c10761),
retrieved 2026-11-07, with default histogram buckets and one API server. Sources:

- [Production metrics](https://docs.vllm.ai/en/stable/usage/metrics/) and the
  [pinned documentation](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/docs/usage/metrics.md).
- [Metric definitions](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/vllm/v1/metrics/loggers.py),
  [buckets](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/vllm/v1/metrics/buckets.py),
  [speculation](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/vllm/v1/spec_decode/metrics.py),
  [expected metrics test](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/tests/entrypoints/serve/instrumentator/test_metrics.py),
  [Grafana dashboard](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/examples/observability/prometheus_grafana/grafana.json).
- [HTTP instrumentator 8.0.0](https://github.com/trallnag/prometheus-fastapi-instrumentator/blob/2f841527277a21ac9ea622a9f923a5f9078234c4/src/prometheus_fastapi_instrumentator/metrics.py)
  and [Python process collector 0.24.1](https://github.com/prometheus/client_python/blob/f417f6ea8f058165a1934e368fed245e91aafc14/prometheus_client/process_collector.py).

Supported families have the same names, TYPE, verbatim HELP text, label keys and
numeric bucket boundaries as those sources. The machine-readable
[contract](../tools/metrics/vllm_contract.json) records every family, label and
bucket. Metadata is attributed there to the respective upstream projects;
no upstream implementation is incorporated.

Counters use `_total` samples and metadata, as `prometheus_client` does in text
0.0.4. Counter, histogram and summary `_created` gauges are included because
the normal single API server registry emits them. They identify collector
initialization in Unix seconds. vLLM's multiple API server registry suppresses
these gauges; that alternate registry profile is not the default target here.
Process collectors do not add `_created`. Sample order and numeric value
precision are not a byte-for-byte copy of another process's scrape. Bucket label spelling follows prometheus_client, including `1e+06`; max-concurrency configuration uses float strings such as `1.0`.

## Supported engine families

Except where specified below, engine families have exactly `model_name` (the
served alias, or canonical name without an alias) and `engine="0"`. Histogram
buckets additionally have `le`; sums, counts and created gauges do not.

| Families (all prefixed `vllm:`) | Type and semantics |
| --- | --- |
| `num_requests_running`, `num_requests_waiting` | Gauges of dispatched generation outside the engine queue, and HTTP plus engine waiters. |
| `num_requests_waiting_by_reason` | Gauge with `reason="capacity|deferred"`; snapshot prerequisites are deferred, other waiters await capacity. Reasons sum to waiting. |
| `prompt_tokens_total`, `generation_tokens_total`, `prompt_tokens_cached_total` | Engine lifetime token counters: submitted prompt including reuse, emitted engine generation including EOS/control/reasoning and partial cancelled work, and reused prompt tokens. |
| `prompt_tokens_by_source_total` | Counter with `source="local_compute|local_cache_hit|external_kv_transfer"`; actual computed prompt tokens, reused tokens, and zero external transfer. During unfinished prefill, computation can lag submitted prompt tokens. |
| `prefix_cache_queries_total`, `prefix_cache_hits_total` | Token counters: prompt tokens considered at engine admission and tokens actually retained/reused when prefix reuse is enabled; both remain zero when it is disabled. No block lookup or cross-instance cache is claimed. |
| `request_success_total` | Terminal engine results, adding `finished_reason="stop|length|abort|error|repetition"`, matching vLLM's FinishReason strings despite its historical HELP wording. Tools map to stop. Client disconnects are removed before vLLM terminal accounting, as in the pinned output processor; abort stays zero for that path. Native cancellation counters still advance. Repetition remains zero because there is no repetition detector. HTTP refusals before admission are excluded. |
| `num_preemptions_total` | Zero: the engine never preempts a running request. Queuing, backpressure and cancellation are not preemptions. |
| `admission_rejections_total` | Only `model_name` and `reason`; `max_num_queued_reqs` counts global/per-user queue limit refusals. `max_num_queued_tokens` is zero because no such token limit exists. |
| `kv_cache_usage_perc` | Gauge: committed positions referenced by active slots / total allocated position capacity. Idle retained sessions count as free, following vLLM's unreferenced cached-block policy. This is position occupancy rather than block-rounded or physical byte utilization. Native retained-context gauges continue to include idle sessions. |
| `cache_config_info` | Gauge 1 with `engine`, `enable_prefix_caching` (`True|False`), `kv_cache_size_tokens`, `kv_cache_max_concurrency`. Only configuration labels with truthful equivalents are present; there is no `model_name` label on this family. |

All time histograms are in seconds. TTFT is measured from socket accept to the
first token observed on the poller. Each later engine token event contributes
one inter-token latency, using a fresh poller monotonic timestamp; speculative
bursts can therefore have very small consecutive gaps. This is a per-token
observation boundary rather than vLLM's batch-output boundary. Terminal request
TPOT is the poller first-to-last token interval divided by generated tokens
minus one, or zero for fewer than two tokens. Prefill is scheduled engine start
to its first token; decode is engine first-to-last token; inference is their
sum. Queue includes HTTP and engine queues. E2E ends at terminal generation
accounting, before socket drain. Times of phases never entered are zero.
Engine errors contribute terminal histograms. Client aborts skip terminal histograms, including n and actual maximum output, as in vLLM; already observed TTFT, ITL and token counters remain. Native outcomes include every cancellation.

| Histograms (all prefixed `vllm:`) | Bucket family / observation |
| --- | --- |
| `time_to_first_token_seconds` | TTFT buckets; one observation when the first engine token arrives. |
| `inter_token_latency_seconds`, `request_time_per_output_token_seconds` | Inter-token buckets; subsequent token gaps / one terminal request TPOT. |
| `e2e_request_latency_seconds`, `request_queue_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds`, `request_inference_time_seconds` | Request latency buckets; one observation per terminal admitted request. |
| `request_prompt_tokens`, `request_generation_tokens`, `request_prefill_kv_computed_tokens`, `request_params_max_tokens` | Token buckets; terminal prompt, engine output, prompt minus retained tokens, requested max tokens. |
| `request_max_num_generation_tokens` | Token buckets; actual generated output length at successful or error finish. The pinned ParentRequest observer records the maximum actual output across children; only n=1 is supported here. |
| `request_params_n` | N buckets; 1 when the parent request finishes, excluding client aborts. |
| `request_num_preemptions` | Preemption buckets; 0 for each terminal request. |
| `iteration_tokens_total` | Iteration buckets; computed prompt tokens plus emitted generation tokens per completed engine step, including the first generation token, excluding rejected speculative candidates. Every landing in checkpoint-restore, session-priority and image-drain paths has its own observation. Snapshot/clone operations themselves are excluded. |

Exact finite bounds, each followed by `+Inf`:

| Bucket family | Bounds |
| --- | --- |
| TTFT | .001, .005, .01, .02, .04, .06, .08, .1, .25, .5, .75, 1, 2.5, 5, 7.5, 10, 20, 40, 80, 160, 640, 2560 |
| Inter-token | .01, .025, .05, .075, .1, .15, .2, .3, .4, .5, .75, 1, 2.5, 5, 7.5, 10, 20, 40, 80 |
| Request latency | .3, .5, .8, 1, 1.5, 2, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 120, 240, 480, 960, 1920, 7680 |
| Tokens | `build_1_2_5_buckets(context_token_limit)`: 1, 2, 5 times increasing powers of ten, retaining values at most the configured context length. |
| Iteration | 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 |
| N | 1, 2, 5, 10, 20 |
| Preemptions | 1, 2, 3, 4, 5, 10, 20 |

With speculation enabled, `spec_decode_num_drafts_total` counts verified draft
rounds; `spec_decode_num_draft_tokens_total` sums proposed tokens;
`spec_decode_num_accepted_tokens_total` sums matching non-EOS prefix tokens.
`spec_decode_num_accepted_tokens_per_pos_total` adds zero-based `position` labels
up to the configured verification depth. Counts aggregate DSpark and lookup
sources when hybrid speculation is selected. Accepted tokens exclude the
ordinary anchor/bonus token, so vLLM's mean acceptance length is
`1 + rate(accepted_total[1m]) / rate(drafts_total[1m])`. These are verifier
outcomes, which can differ from user-visible output when a limit truncates a
round. Existing per-position engine statistics supply the values; collection
does not add worker token hooks.

The terminal accounting definitions also follow the same pinned revision's
[parallel sampling observer](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/vllm/v1/engine/parallel_sampling.py),
[client abort processing](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/vllm/v1/engine/output_processor.py), and
[KV cache manager](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/vllm/v1/core/kv_cache_manager.py).
Retrieved 7 November 2026.

## HTTP, process and native families

`http_requests_total` has `handler`, `method`, `status` (2xx–5xx).
`http_request_duration_seconds` has `handler`, `method`, low-resolution
buckets .1, .5, 1, +Inf. `http_request_duration_highr_seconds` has no API labels
and bounds .01, .025, .05, .075, .1, .25, .5, .75, 1, 1.5, 2, 2.5, 3, 3.5, 4,
4.5, 5, 7.5, 10, 30, 60, +Inf. Durations end when output drains, or at disconnect.

`http_request_size_bytes` and `http_response_size_bytes` are **summaries** with
only `handler`, sum/count and created samples, without quantiles. They respect
Content-Length headers, including zero for streamed responses without a length,
matching the instrumentator. Handler values are fixed route templates or
`none` for unmatched paths and method mismatches (including automatic HEAD requests, which this server does not implement). Methods use GET/POST/HEAD/PUT/DELETE/PATCH/OPTIONS;
other methods collapse to `OTHER` to keep cardinality bounded; upstream preserves arbitrary method strings. The pinned regex exclusions search the path for `/metrics`, `/health`, `/load`, `/ping`, `/version` and `/server_info`, including within an unmatched path. Query strings are excluded from this search. This server matches raw path bytes; ASGI-style percent-decoding is not performed, so encoded paths remain unmatched. HTTP protocol rejections before request acceptance are excluded, including invalid request lines, malformed header fields and invalid or duplicate Content-Length. The parser reports field-framing errors to metrics only when enabled; serving refusal statuses are unchanged. Validly framed application refusals (including 404/405/411/413/417/501 and this server's Host checks) remain counted. This follows the upstream transport boundary ([uvicorn 0.38.0](https://github.com/encode/uvicorn/blob/0.38.0/uvicorn/protocols/http/httptools_impl.py), retrieved 7 November 2026). These transport statistics include HTTP
refusals without reclassifying them as engine requests. Size sums saturate at UINT64_MAX for hostile declared lengths. Labelled HTTP children and their `_created` timestamps appear at first use; without children their `_created` HELP/TYPE records are omitted, as in prometheus_client.

The Linux process collector emits `process_cpu_seconds_total`,
`process_resident_memory_bytes`, `process_virtual_memory_bytes`,
`process_start_time_seconds`, `process_open_fds`, `process_max_fds` without labels.
It reads bounded `/proc/self` data and the process's file-descriptor limit at scrape time. Boot time and sysconf constants are read once at metrics startup. An unavailable source omits its dependent process families and leaves the rest of the scrape available. CPU is total user plus system seconds; start time is Unix epoch seconds;
memory is bytes; descriptors are counts. No GPU/device query is added.

Native families retain `model_name`, `engine` and their original meanings:

| Family | Type / extra label keys |
| --- | --- |
| `anemone_requests_admitted_total` | Counter; none. |
| `anemone_requests_finished_total` | Counter; `finished_reason`: stop, length, tool_calls, error, cancelled. |
| `anemone_request_refusals_total` | Counter; `status_class`: 4xx, 5xx. A refused admitted waiter can also have a terminal error; do not add these counters. |
| `anemone_prefill_tokens_total` | Counter of actual new prompt computation; none. |
| `anemone_cache_context_tokens`, `anemone_cache_capacity_tokens` | Gauges of retained committed positions / total slot capacity; none. |
| `anemone_model_ready` | Gauge; 0 on engine failure or shutdown; none. |
| `anemone_process_uptime_seconds` | Gauge of monotonic engine uptime; none. |
| `anemone_build_info` | Gauge 1; `build_id` is the ELF build ID or unavailable. |
| `anemone_speculation_proposed_tokens_total`, `anemone_speculation_accepted_tokens_total` | Counters when speculation is enabled; none. |

Engine token/speculation counters are engine lifetime; HTTP counters and request
histograms are server lifetime. The supported lifecycle creates one server per
engine, and a new process resets both. Metrics add no per-request or per-token
record allocation: fixed accounting records are poller-owned; one scrape buffer
is allocated at server startup only when enabled. Iteration accounting uses one
fixed histogram under the publication lock after each landing, enabled with the exporter; image drains publish each step separately.
The request path does not acquire a new lock; the worker gets no new token hook.
Iteration collection is enabled before the first engine command admission;
attempting to enable it later returns BUSY_VALUE. Disabling remains legal.

Unlisted families are omitted. There is no equivalent for vLLM's paged KV block
lifetime/idle/reuse histograms, connector transfers or external caches,
multimodal item caches, LoRA, watermarking, diffusion, MFU, CUDA graph metrics or
NaN-specific corruption counters. Tier-specific cached-token source counters
are omitted because restored snapshot provenance is not retained in cumulative
engine totals. Engine sleep-state reporting is omitted because sleep mode is
not implemented. Python runtime/GC families have no native-runtime equivalent
and are omitted. No fabricated measurements are exported for these features.

llama.cpp is the second plausible engine. Its
[pinned handler](https://github.com/ggml-org/llama.cpp/blob/80d2c0c2c56e4a1bbf74be98f64b99927bf230bc/tools/server/server-task.cpp)
uses distinct `llamacpp:` names. An optional second format is not implemented;
the default and only format here is vLLM.

## Host verification

`server_metrics_test` exercises real CPU fixture HTTP traffic, default-off
identity, tokens/latencies/iterations/parameters, errors/abort accounting,
concurrent scrapes, startup labels and restart resets. To compare its saved
exposition and execute all 32 PromQL expressions from the pinned example
Grafana dashboard:

```sh
./build/release/server_metrics_test /tmp/metrics.prom
python3 tools/check/metrics_exposition_test.py /tmp/metrics.prom
python3 tools/check/metrics_vllm_test.py /tmp/metrics.prom --context 4096 \
  --model "$(printf '%s\n%s' 'fixture\"' 'model')" --promtool /path/to/promtool
```

The fixture deliberately uses an alias containing a backslash, quote and newline; `--model` checks its exact label value. The checker also rejects invalid label values and noncanonical bucket spellings.

The PromQL test uses a zero baseline followed by the measured scrape as a
synthetic history. Each expression must return finite data under real promtool
rule evaluation. This proves dashboard query compatibility, not native model
quality or overhead. Native qualification is required before deployment.

## Storage replica information

The optional Prometheus exporter includes this gauge family when serving metrics
are enabled with `--metrics=1`:

```text
# HELP anemone_storage_replica_info Startup replica selection by physical device and role.
# TYPE anemone_storage_replica_info gauge
anemone_storage_replica_info{mount="/srv",role="load",status="included",reason="verified_identity_and_bytes"} 1
anemone_storage_replica_info{mount="/srv",role="lookup",status="included",reason="verified_identity_and_bytes"} 1
```

`role` is `load` or `lookup`; `status` is `included` or `excluded`. `reason` is
one of the engine's fixed replica reason codes. Examples include
`insufficient_space`, `external_device_requires_override`,
`network_filesystem_or_transport`, `rotational_device`,
`duplicate_physical_device`, `device_disappeared_or_unknown`,
`hardware_identity_mismatch_retune_required`, `zero_tuned_weight` and
`replica_missing_short_or_unverified`. The value 1 identifies a decision, not a
throughput measurement, byte count or live I/O-health gauge. A retained replica with no share for one role is excluded
for that role (`zero_tuned_weight`); an explicitly overridden action is `action_disabled`.

Labels are bounded by the startup's device candidates: at most 64 retained
physical/mount representatives and two roles (128 samples). Physical overlaps
and duplicate mounts are deduplicated, preferring the included representative.
There are no request/user/key/session/prompt/token labels. Mount label escaping
prevents exposition injection. The snapshot retains exclusions even if every
replica fails and configuration falls back; disabled metrics emit no family.

The snapshot is generated once at startup/profile reload and lives unchanged
through serving. Scrapes allocate no new storage buffers and perform no disk,
udev/sysfs or model lookup. A later device failure is reflected after explicit
reload/restart; this family does not poll hot-plug health. Product stderr lines
and `anemone storage-report PROFILE` show the saved and current policy without
starting a model. `storage-report PROFILE --metrics=1` exports the same info
snapshot directly; full replica verification in that report is real startup I/O,
not a benchmark and not a model load.

The storage implementation supplies an immutable `server_options` info snapshot.
The optional metrics endpoint appends it only on the metrics-enabled path,
retaining that endpoint's authentication/ACL, limits, content type and disabled
routing behavior. Other inference metrics and their timing/accounting contracts
belong to the endpoint documentation. No deployment, unit or live profile is
changed by enabling this family.
