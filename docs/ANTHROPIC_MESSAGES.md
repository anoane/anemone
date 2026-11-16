<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Anthropic Messages adapter

Opt in with `--anthropic_messages=1` on an otherwise admitted `anemone serve`
command. The default is 0. The native listener serves `POST /v1/messages`
through the same engine, per-user queues, session slots, cache isolation and
cancellation paths as Chat and Responses. Existing OpenAI routes and their
defaults are unchanged. With the option disabled, Messages-specific head parsing is disabled too, including for malformed requests; the path uses the ordinary unknown-route response bytes.

This is a text/client-tool compatibility subset for model `anemone`. It does
not provide Claude model identity, model quality, billing or hosted tools.

Supply one `anthropic-version` header: `2023-06-01` or `2026-10-21`.
With server authentication configured, send either `x-api-key` or
`Authorization: Bearer` using an existing native user key. Combined or
repeated credentials are refused. The key selects the same native username
and queue; no additional account system is created.

Require `model: "anemone"`, a positive integer `max_tokens`, and a nonempty
`messages` array beginning and ending with a user turn. Accept string or
text-block content, a top-level string/text-block `system`, assistant
`tool_use` object inputs, and immediately matching user `tool_result` blocks.
Text precedes calls; results precede ordinary user text. Results may contain
strings or text blocks; `is_error:true` adds `Error: ` to the model-facing
result. IDs must be unique within a pending turn and 1–128 ASCII letters,
digits, underscores or hyphens. Names use the same alphabet, at most 64 bytes.
At most 128 tools/calls per turn and 4096 messages/content blocks are admitted,
subject to the shared body, normalized-message and context limits.

Tools accept `name`, optional string `description`, an object `input_schema`
whose type is `object`, and optional `strict:false`. Tool choice is `auto` or
`none`. Tools are not executed by the server. Schema-constrained output,
forced calls and a single-call constraint are unsupported. Empty/null SDK
replay fields `citations`, `caller` and `toolset_name` are allowed only where
they carry no additional semantics; nonempty values are refused.

For version `2023-06-01`, temperature is 0–1 (default 1) and top_p is (0,1]
(default 1). top_k is an integer from0 to2097152 (default0/off), also accepted as an OpenAI
Chat extension. OpenAI falsy values retain the existing unset semantics; Messages requires the documented integer type. It keeps the highest raw logits, breaking ties by lowest token
ID, before the existing temperature/nucleus/keyed-race policy. Temperature0
remains greedy. Zero or a count at least the vocabulary retains the old set.
Positive truncation defines a distinct sampling reference, and does not imply
equality to an untruncated original-reference trajectory.
Version `2026-10-21` refuses all three sampling fields, following the current
[official version history](https://platform.claude.com/docs/en/api/versioning).
`stream` is boolean. Up to four nonempty `stop_sequences`, each at most 256
UTF-8 bytes, use the shared content-only stop matcher.

Thinking supports `disabled`, `enabled` with integer `budget_tokens` (at least
1024 and below `max_tokens`), and `adaptive` with `output_config.effort`.
Low maps to DeepSeek effort50; medium/high/xhigh map75; max maps100. These are
the model's existing native levels. Enabled uses the requested hard cap.
Adaptive reserves an answer token and caps reasoning at one quarter of the
available output at low, three quarters at medium/high/xhigh, and all remaining
output at max. Available output is the smaller of max_tokens and the actual context room after tokenization. An explicit enabled budget that cannot fit is refused; adaptive is clamped to that room. The model may finish reasoning sooner. The cap includes the
closing native token. The engine commits the actual `</think>` token when
needed and continues decoding from it, suppressing speculation in reasoning
so a sampled successor cannot cross that forced boundary. Closing preserves complete UTF-8 characters and may occur up to three tokens early. A token that finishes one character and starts another can be reduced to the reserved continuation byte tokens before closing. The decoder recognizes the actual closing control token; literal text spelling it does not close reasoning. Tool calls inside budgeted reasoning are malformed output and fail only that request.

The Messages template keeps supplied prior thinking through the encoder's
existing keep-thinking option. Each prior turn retains its own mode/effort;
the next turn's effort instruction is placed after the shared history. Thus
changing the next mode or budget does not rewrite the cached history. Pass
thinking blocks back unchanged, including their opaque local signature
annotation. This annotation records the prior native effort; it is not an
Anthropic cryptographic signature or Claude provenance. Thinking SSE includes
thinking and signature deltas. Usage includes actual reasoning-token counts.

Images, documents, nonempty citations, cache directives, persistent state,
metadata, beta features, null controls, redacted/encrypted thinking, thinking
display summaries and unknown fields remain unsupported. There is no
`max_tokens:0` prompt-cache warming.

Responses contain Message thinking/text/tool_use blocks and stop reasons `end_turn`,
`max_tokens`, `model_context_window_exceeded`, `tool_use` or `stop_sequence`, with the matched string for the
last case. Usage reports local rendered prompt and actual generated token
counts, including EOS and tool delimiters. It omits Anthropic billing/cache
fields. Malformed generated tool JSON produces `api_error` if no complete call exists. Once a valid call is complete, a malformed or trailing DSML suffix is discarded; completed calls remain replayable and the reason is tool_use at EOS or max_tokens at the output limit. Context exhaustion uses model_context_window_exceeded. These reasons are accepted for both supported versions.

SSE uses named message/content-block events, text or JSON deltas, cumulative
output usage in `message_delta` and terminal `message_stop`; there is no
`[DONE]`. Errors after the stream starts emit `event:error` without a success
terminal. Errors use `{type:"error",error:{type,message}}`. Queue saturation
retains native HTTP 503 with `Retry-After` and `overloaded_error`.

The contract was reviewed against the official
[Messages](https://platform.claude.com/docs/en/api/messages/create),
[streaming](https://platform.claude.com/docs/en/build-with-claude/streaming),
[errors](https://platform.claude.com/docs/en/api/errors),
[tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)
and [thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
documentation on 2026-11-07.

CPU validation on Linux (the separate sampler_top_k_card_test is a prepared GPU gate and must use normal explicit GPU admission):

```sh
make -j4 build/release/server_messages_test
./build/release/server_messages_test
python3 tools/serving/messages_contract_test.py build/release/server_messages_test
```

The Python suite uses real sockets and the native CPU fixture. If installed
in that Python environment, the official `anthropic` SDK also checks complete
responses, stream accumulation and tool-result replay. CPU evidence does not
qualify native-model correctness, tool quality, full-context operation or
performance. Real-model tests require the operator's normal guarded admission.

Resource layout: the shared sampler settings record is20 bytes (16 before top_k,32 in the earlier adapter). The four-rank exchange retains two physical phases plus one512-byte optional entry barrier (2560 bytes per card in total), rather than100 phases/102400 bytes. The512-byte entry area is appended after every existing stage workspace; the entry-address table uses spare space in the original256-byte peer allocation, preserving all main offsets; extra top-k/nucleus reductions reuse those slots with a separate even tag and clear each consumed optional record before the next publication barrier. The entry barrier and an odd final phase protect rows that skip calls. Optional payload/tag publication uses system release/acquire ordering; the default path keeps its existing tags and exchanges. top_k unset retains the existing reduction sequence and arithmetic. Formatter state is inline and borrows the poll thread's existing prompt scratch. The retained bounded frontend storage deviation is documented in STYLE.md. Input/normalized body limits return413; string/complexity limits return400; actual allocation failures return500.
