/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_serve.c: the serve loop's stop, request failures and scheduling usage (split from commands.c by whole items).
 */
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include "common/hooks.h"
#include "engine/storage/load/loader.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/runtime/placement/placement.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/engram/engram_disk.h"
#include "engine/engram/engram_fit.h"
#include "engine/storage/io/local_device.h"
#include "engine/storage/profile/storage_profile.h"
#include "engine/storage/autotune/storage_autotune.h"
#include "engine/server/server.h"
#include "engine/runtime/placement/session_fit.h"
#include "cli_stats_client.h"
#include "cli_policy.h"
#include "engine/storage/io/storage_exl3.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "cli_model_folder.h"
#include "models/model_registry.h"
#include "engine/runtime/scheduler/serving_cursor.h"
#include "cli_axes.h"
#include "engine/runtime/step/step.h"
#include "engine/runtime/step/step_images.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "engine/runtime/speculation/draft_lookup.h"
#include "engine/formats/pack/tokenizer_tables_pack.h"
#include "common/json.h"
#include "engine/engram/engram_step.h"
#include "engine/engram/engram_hot.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_model.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_peer.h"
#include "commands.h"
#include "cli_golden_capture.h"

#include "cli_internal.h"

#define PIPE_ENDS 2 /* of a pipe: read, write */
#define STOP_SIGNALS 3 /* of serve: SIGINT, SIGTERM, SIGHUP */
static int stop_pipe[PIPE_ENDS] = {
	-1, -1
}; /* serve's: a signal writes, the server polls, serve closes */

/* serve's signals: one byte into the pipe the server polls. */
static void stop_requested(int signal_number)
{
	const int saved_errno = errno;
	const unsigned char byte = (unsigned char)signal_number;
	const ssize_t written =
		write(stop_pipe[1], &byte,
		      sizeof(byte)); /* a full pipe already says it */
	(void)written;
	errno = saved_errno;
}

/* serve: the stop pipe and its signals, the line that says the server is up, the requests until a signal; then the
 * signals' handlers as they were, and the pipe closed (left open if a handler could not be given back: it writes). */
MUST_CHECK bool requests_failed(const struct text_tokenizer *tokenizer)
{
	static const int stop_signals[STOP_SIGNALS] = { SIGINT, SIGTERM,
							SIGHUP };
	struct sigaction action = { .sa_handler = stop_requested },
			 before[STOP_SIGNALS];
	uint32_t taken = 0;
	bool is_ready = sigemptyset(&action.sa_mask) == 0 &&
			pipe2(stop_pipe, O_CLOEXEC | O_NONBLOCK) == 0;
	for (; is_ready && taken < STOP_SIGNALS; taken += is_ready ? 1u : 0u) {
		is_ready = sigaction(stop_signals[taken], &action,
				     &before[taken]) == 0;
	}
	if (is_ready) {
		(void)printf("serving %s on %s port %" PRIu32 "\n",
			     pack_header_of(pack)->model_name, bind_address,
			     port);
	}
	result_status info_status = is_ready ? storage_metrics_prepare() :
					       STATUS_INPUT_OUTPUT_ERROR;
	if (info_status == STATUS_OK)
		info_status = server_storage_info(server, storage_replica_info,
						  storage_replica_info_bytes);
	const result_status status =
		info_status == STATUS_OK ?
			server_run(server, engine, tokenizer, stop_pipe[0]) :
			info_status;
	bool is_restored = true;
	for (uint32_t index = 0; index < taken; index++) {
		is_restored = sigaction(stop_signals[index], &before[index],
					NULL) == 0 &&
			      is_restored;
	}
	for (uint32_t end = 0; is_restored && end < PIPE_ENDS; end++) {
		if (stop_pipe[end] >= 0) {
			(void)close(stop_pipe[end]);
		}
	}
	if (!is_restored) {
		LOG_ERROR(
			"SIGINT or SIGTERM could not be given its handler back: the stop pipe stays open");
	}
	return step_failed("serve", status == STATUS_OK && !is_restored ?
					    STATUS_INPUT_OUTPUT_ERROR :
					    status);
}

/* A separate literal keeps the complete scheduling examples below C11's 4,095-byte translation limit. */
void scheduling_usage(FILE *output)
{
	(void)fputs(
		"Scheduling: anemone serve --help-scheduling\n"
		"A ready decoder runs its ordinary decode program between exact whole-granule prompt chunks.\n"
		"--prefill_decode_rows=0|256..2048 caps chunks only while another stream decodes; 0 keeps old"
		" behavior.\n"
		"--prefill_decode_rounds=1..64 sets ordinary decode rounds at each interval (old value 1).\n"
		"--prefill_decode_waiters=1 uses original chunks and one round while multiple prompts wait;\n"
		"  the configured cap and burst apply to the final pending prompt. 0 keeps a fixed policy.\n"
		"--prefill_decode_every=0 uses one chunk per interval with the cap; N>0 permits N chunks.\n"
		"--prefill_chunk_rows=0|256..2048 caps chunks even without a decoder (default 0).\n"
		"Caps are multiples of 256, require --chunks_in_flight=1 and the exact prefill plan, and retain\n"
		"the original final tail. --prefill_order=0 uses turns; 1 oldest; 2 shortest with bounded oldest"
		" wait.\n"
		"Order 3 shares short tails in the measured row-independent EXL3 BATCH recipe only.\n"
		"--prefill_prompt_chunks=1 keeps runnable prompt chunks out of decode BATCH steps; 0 restores mixing.\n"
		"It defaults to 0; the measured code C8 cost is 1.12% stream decode / 3.00% aggregate.\n"
		"BATCH default: 1024 rows / 1 round / waiters0 / order2 with --prefill_first=1 and wide_rows=1024,\n"
		"BF16 column-owner sums, thin decode bits and the exact chunk plan. THIN retains old defaults.\n"
		"Explicit options override defaults; --prefill_decode_rows=0 also keeps rounds1/waiters0/order0.\n"
		"Examples: $BATCH stands for your normal BATCH recipe options; TABLES and MAP are model paths.\n"
		"Place these scheduling options before the model paths; do not repeat options already in $BATCH.\n"
		"  anemone serve $BATCH TABLES MAP                         # default: 1024/1, waiters0, order2\n"
		"  anemone serve $BATCH --prefill_decode_rows=1024 --prefill_decode_rounds=1"
		" --prefill_decode_waiters=0 --prefill_order=2 TABLES MAP    # full chunks, "
		"one decode round (see table)\n"
		"  anemone serve $BATCH --prefill_decode_rows=1024 --prefill_decode_rounds=4"
		" --prefill_decode_waiters=0 --prefill_order=2 TABLES MAP    # faster reply completion\n"
		"  anemone serve $BATCH --prefill_decode_rows=256 --prefill_decode_rounds=8"
		" --prefill_decode_waiters=0 --prefill_order=2 TABLES MAP    # shortest measured pauses\n"
		"  anemone serve $BATCH --prefill_chunk_rows=0 --prefill_decode_rows=0 --prefill_decode_rounds=1"
		" --prefill_decode_waiters=0 --prefill_decode_every=0 --prefill_order=0 TABLES MAP # old defaults\n"
		"\nMeasured 4-CMP BATCH order2, prefix reuse off: 30K prompt + four short streams, T=0, max64 tokens\n"
		"(one short ends at EOS50). Seconds; short TTFT/completion are medians, pause is largest SSE content"
		" gap.\n"
		"Completion is time after the first token. Fraser replies are 2,048 tokens; aggregate includes"
		" first-token wait.\n"
		"Policy       ShortTTFT Complete Pause LongTTFT Fraser C4/C8 median TTFT  C4/C8 aggregate tok/s\n"
		"old/order2      3.75    11.53  5.45   14.29       3.94/7.13                 321/384\n"
		"512/1           4.10     5.70  0.29   16.16       4.40/8.38                 312/358\n"
		"1024/4          4.36     2.82  0.46   15.99       4.61/9.25                 316/356\n"
		"256/8           5.08     1.27  0.21   17.34       7.93/18.77                266/274\n"
		"waiter512/4     3.92     3.84  0.47   15.32       4.08/7.61                 316/368\n"
		"Waiter row: one pilot Fraser batch; mixed table uses the together arrival.\n"
		"1024/1          3.92     7.77  0.48   15.38       4.10/7.64                 322/373\n"
		"1024/1 row: one final pilot batch per Fraser concurrency.\n"
		"Old and 512/1 have three Fraser batches; 1024/4 and 256/8 are one-batch pilots. Mixed texts equal"
		" solo.\n"
		"Shorter caps and larger bursts reduce pauses/completion time but add first-token work and can lower\n"
		"aggregate throughput. These are workload measurements, not latency bounds.\n"
		"See the RND_M evidence log.\n",
		output);
}
