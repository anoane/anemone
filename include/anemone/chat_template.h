/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * chat_template.h: a conversation laid out as the model reads it, byte for byte what its reference encoder writes.
 *
 * Contract:
 *   - The input is the request's own JSON: the messages array and, when the request has tools, the tools array,
 *     which goes on the first message as the reference's harness puts it (encoding.py, load_cases). When nonempty
 *     request tools are supplied and the first message is not system, prepend an empty system message carrying
 *     those tools, as if {"role":"system","content":""} had been prepended before calling the reference.
 *     The developer role is normalized to system before all checks, merges and rendering, at any position.
 *     Existing system-first conversations, absent/empty tools and message-local tools are unchanged. The output is
 *     the prompt's text, what encoding.py's encode_messages returns for json.loads of the same JSON with the begin
 *     token first (the encoder both checkpoints' repositories carry since 2026-10-16, git blob 3c7c357d); the
 *     tokenizer then turns it into ids. The options stand for the encoder's arguments, and continues_final_message
 *     for its wo_eos on the last message, an assistant's, which is then left open.
 *   - Of a message, role, content, reasoning_content, tool_calls, tool_call_id, tools, response_format and task are
 *     read; other members are left alone; a member given twice counts with its last value, as in Python. As the
 *     encoder does it: tool results merge into user messages, in the order of the calls before them; consecutive
 *     user messages merge; with drops_thinking, reasoning before the user's latest turn is dropped, and tool
 *     results alone open no turn. Any nonempty tools declaration disables dropping, as in the encoder. In thinking
 *     mode, retained empty, null or absent reasoning renders the same empty reasoning before </think>; in chat mode
 *     reasoning is omitted. Tools and calls are written in DSML, JSON values as json.dumps(ensure_ascii=False)
 *     spells them (", " and ": ", a float's shortest repr, NaN and Infinity); an image part is its placeholder, the
 *     images in the order of their placeholders in the prompt, tool results' after their sort (the pixels are the
 *     vision piece's).
 *   - Refused with STATUS_INVALID_ARGUMENT, nothing written then to be used:
 *       JSON that json.loads refuses, invalid UTF-8, a surrogate escape without its other half, nesting deeper than
 *       source/json.h allows (128 arrays and objects), in the request or in a call's arguments string that the
 *       encoder would decode again or write (one that holds an object or a string);
 *       what the encoder refuses: an unknown role or task, the image placeholder in text (render only),
 *       a tool or call without
 *       its function, name or arguments, a name with "::" twice, an image without a source, a reasoning effort
 *       outside 1 to 100, tools on no message, a user content that is null;
 *       what the encoder would write only through Python's str() of a value of another type: a content, text or
 *       reasoning that is no string (or null, or an array for a content);
 *       by this contract, though the encoder takes them, because a request of the OpenAI-compatible API cannot
 *       carry them: a namespace on a tool or a call (a name with one "::" is written as it is), a content part
 *       that is not text or image_url (the encoder's own tool_result and image parts among them), the encoder's
 *       own message members content_blocks and wo_eos (continues_final_message stands for wo_eos), a call's id
 *       that is truthy and no string (the encoder refuses a list or an object), a tool_call_id that is neither a
 *       string nor null;
 *       by this contract, where Python keeps the last value in the place of the first: an object written as JSON (a
 *       tool's function, a response format, a call's arguments, and every object inside them) with a key twice;
 *       more than CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS calls in a message or tool results in a merged one;
 *       continues_final_message when the last message is not an assistant's.
 *     A task or role the encoder would never read is refused all the same.
 *   - A prompt above prompt_capacity, in a request refused for nothing else, is STATUS_UNSUPPORTED. A prompt
 *     has at least its bytes over the longest token's in tokens: with a capacity of the context's tokens times the
 *     longest token's bytes, UNSUPPORTED is a prompt longer than the context, not an invalid request.
 *   - Work and memory are linear in the input: while the call runs, five times its bytes of scratch, 48 bytes for
 *     each message plus one possible synthetic system entry and 56 KiB; render retains nothing after. The
 *     render_spans variant additionally returns the caller-owned origin map described below.
 *     Fault point: chat_template_memory.
 */
#ifndef CHAT_TEMPLATE_H
#define CHAT_TEMPLATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/status.h"
#include "anemone/image.h"
#include "anemone/tokenizer.h"

#define CHAT_TEMPLATE_MAXIMUM_TOOL_CALLS 1024
#define CHAT_TEMPLATE_MAXIMUM_INPUT_BYTES \
	1073741824ull /* 1 GiB of messages and tools together */
#define CHAT_TEMPLATE_LOWEST_EFFORT 1
#define CHAT_TEMPLATE_HIGHEST_EFFORT 100

struct chat_template_options {
	uint32_t reasoning_effort; /* 1 to 100 (the encoder's names: low 50, high 75, max 100) */
	bool is_thinking; /* false is the encoder's "chat" mode: thinking off */
	bool drops_thinking; /* the encoder's drop_thinking: earlier reasoning is not kept */
	bool continues_final_message; /* the last message is the start of the answer to continue */
};

/* Image source offsets refer to the messages_json given to render_spans, including the URL's JSON quotes.
 * prompt_byte_offset identifies the placeholder written by the template, after tool-result sorting. */
struct chat_template_image {
	size_t url_begin, url_end;
	uint32_t prompt_byte_offset;
};

/* Owned result of render_spans; free(spans) after encoding. No prompt or JSON bytes are borrowed. */
struct chat_template_spans {
	struct tokenizer_span *spans;
	uint32_t count;
	struct chat_template_image images[IMAGE_MAX_IMAGES];
	uint32_t image_count;
};

/* First users: the server, for every request, and the tests. tools_json is NULL when the request has no tools.
 * On failure prompt_byte_count_out is untouched. */
MUST_CHECK result_status chat_template_render(
	const char *messages_json, size_t messages_byte_count,
	const char *tools_json, size_t tools_byte_count,
	const struct chat_template_options *options, char *prompt_out,
	size_t prompt_capacity, size_t *prompt_byte_count_out);

/* Same prompt bytes as render, plus coalesced spans written by the template itself. It additionally accepts the
 * image-placeholder spelling in text/replayed reasoning as literal payload. Image parts have trusted placeholders
 * and source metadata in prompt order; more than IMAGE_MAX_IMAGES is INVALID_ARGUMENT.
 * All other refusals are render's. Message content and text parts of every role, retained reasoning, names,
 * schemas, descriptions, response formats and decoded
 * call arguments are payload, including their canonical JSON serialization. Feed these spans to tokenizer's
 * encode_spans/encode_cached_spans. Ordinary text is never split merely because its origin changes.
 * trusted_out must be non-NULL; on success it receives a fresh allocation owned by the caller (no previous
 * allocation is freed). On failure both scalar/result outputs are untouched. The template bounds the range
 * offsets to tokenizer's MAXIMUM_TEXT_BYTES; larger output is UNSUPPORTED. Extra span memory is linear in the
 * number of alternating template/payload writes. Fault point: chat_template_memory. */
MUST_CHECK result_status chat_template_render_spans(
	const char *messages_json, size_t messages_byte_count,
	const char *tools_json, size_t tools_byte_count,
	const struct chat_template_options *options, char *prompt_out,
	size_t prompt_capacity, size_t *prompt_byte_count_out,
	struct chat_template_spans *trusted_out);

#endif
