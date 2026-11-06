/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* SOURCE ONLY: stateless Responses input and output, sharing the Chat renderer and token decoder. */
#ifndef SERVER_RESPONSES_H
#define SERVER_RESPONSES_H

#include "anemone/chat_completions.h"
#include "json.h"

/* Converted spans belong to chat_body until rendering ends. Refusal spans always name the original body. */
MUST_CHECK uint32_t server_responses_read(const char *body, size_t count,
					  const char *model, const char *alias,
					  const struct chat_request *defaults,
					  struct chat_request *request,
					  struct chat_buffer *chat_body,
					  struct chat_refusal *refusal);
/* Server form: retain echoes separately from history; attach fields to answer.pending_responses_fields before begin. */
MUST_CHECK uint32_t server_responses_read_fields(
	const char *body, size_t count, const char *model, const char *alias,
	const struct chat_request *defaults, struct chat_request *request,
	struct chat_buffer *chat_body, struct chat_buffer *fields,
	struct chat_refusal *refusal);
void server_responses_begin(struct chat_answer *answer, const char *model,
			    int64_t created);
void server_responses_text(struct chat_answer *answer, const char *bytes,
			   size_t count);
void server_responses_call(struct chat_answer *answer, struct json_bytes name,
			   const struct chat_buffer *arguments);
void server_responses_finish(struct chat_answer *answer, const char *reason);
void server_responses_fail(struct chat_answer *answer);
void server_responses_free(struct chat_answer *answer);

#endif
