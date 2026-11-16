/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* SOURCE ONLY: bounded Anthropic Messages adapter over the shared Chat decoder. */
#ifndef SERVER_MESSAGES_H
#define SERVER_MESSAGES_H
#include "anemone/chat_completions.h"
#include "json.h"
#define MESSAGES_ADAPTIVE_DIVISOR 4u
#define MESSAGES_MODEL "anemone"
#define MESSAGES_LEGACY_VERSION "2023-06-01"
#define MESSAGES_CURRENT_VERSION "2026-10-21"
#define MESSAGES_MOST_TOOLS 128u
#define MESSAGES_NAME_BYTES 64u
#define MESSAGES_ID_BYTES 128u
MUST_CHECK uint32_t server_messages_read(const char *body, size_t count,
					 bool legacy_sampling,
					 const struct chat_request *defaults,
					 struct chat_request *request,
					 struct chat_buffer *converted,
					 struct json_scratch *workspace,
					 struct chat_refusal *refusal);
void server_messages_refuse(struct chat_buffer *output,
			    const struct chat_refusal *refusal);
void server_messages_begin(struct chat_answer *answer);
void server_messages_text(struct chat_answer *answer, const char *bytes,
			  size_t count);
void server_messages_call(struct chat_answer *answer, struct json_bytes name,
			  const struct chat_buffer *arguments);
void server_messages_finish(struct chat_answer *answer, const char *reason);
void server_messages_fail(struct chat_answer *answer);
void server_messages_free(struct chat_answer *answer);
#endif
