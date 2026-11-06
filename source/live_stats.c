/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Live JSON and text views share one snapshot schema. Formatting and sorting happen only at refresh. */
#include "anemone/live_stats.h"

#include <elf.h>
#include <inttypes.h>
#include <link.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "json.h"

#define LIVE_SCHEMA_VERSION 2u
#define NANOSECONDS_PER_SECOND 1000000000.0
#define NANOSECONDS_PER_MILLISECOND 1000000.0
#define MILLISECONDS_PER_SECOND 1000.0
#define MICROSECONDS_PER_MILLISECOND 1000.0
#define PERCENT_SCALE 100.0
#define PERCENTILE_HIGH 95u
#define PERCENTILE_MIDDLE 50u
#define PERCENTILE_SCALE 100u
#define NUMBER_BYTES 96u
#define JSON_STRING_EXPANSION 6u
#define JSON_QUOTES_BYTES 2u
#define INDENT_COLUMNS 2u
#define MAXIMUM_INDENT_COLUMNS 20u
#define NOTE_ALIGNMENT 4u
#define HEX_RADIX 16u
#define HEX_DIGITS_PER_BYTE 2u
#define BUILD_NOTE_NAME_BYTES 4u
#define LAYER_MASK_BITS 64u
#define NUMBER_SIGNIFICANT_DIGITS 9
#define WALL_SIGNIFICANT_DIGITS 17
#define WALL_TIME_BYTES 32u
#define WALL_LAST_SECOND 253402300799.0

struct live_writer {
	struct chat_buffer *output;
	bool first;
};

static void write_text(struct chat_buffer *output, const char *text)
{
	chat_buffer_write(output, text, strlen(text));
}

static void write_string_bytes(struct chat_buffer *output, const char *text,
			       size_t length)
{
	const size_t capacity =
		length * JSON_STRING_EXPANSION + JSON_QUOTES_BYTES;
	char *const room = chat_buffer_room(output, capacity);
	if (room != NULL) {
		struct json_writer writer = { room, capacity, 0 };
		json_write_string(&writer, (struct json_bytes){ text, length });
		output->count += writer.length;
	}
}

static void member(struct live_writer *writer, const char *name)
{
	write_text(writer->output, writer->first ? "" : ",");
	writer->first = false;
	write_string_bytes(writer->output, name, strlen(name));
	write_text(writer->output, ":");
}

static void integer(struct live_writer *writer, const char *name,
		    uint64_t value)
{
	char number[NUMBER_BYTES];
	const int count = snprintf(number, sizeof(number), "%" PRIu64, value);
	member(writer, name);
	if (count > 0 && (size_t)count < sizeof(number)) {
		chat_buffer_write(writer->output, number, (size_t)count);
	} else {
		writer->output->is_short = true;
	}
}

static void real_precision(struct live_writer *writer, const char *name,
			   double value, int precision)
{
	char number[NUMBER_BYTES];
	const int count = isfinite(value) ? snprintf(number, sizeof(number),
						     "%.*g", precision, value) :
					    0;
	member(writer, name);
	if (count > 0 && (size_t)count < sizeof(number)) {
		chat_buffer_write(writer->output, number, (size_t)count);
	} else {
		write_text(writer->output, "null");
	}
}

static void real(struct live_writer *writer, const char *name, double value)
{
	real_precision(writer, name, value, NUMBER_SIGNIFICANT_DIGITS);
}

static void boolean(struct live_writer *writer, const char *name, bool value)
{
	member(writer, name);
	write_text(writer->output, value ? "true" : "false");
}

static void string(struct live_writer *writer, const char *name,
		   const char *value, size_t capacity)
{
	member(writer, name);
	write_string_bytes(writer->output, value, strnlen(value, capacity));
}

static struct live_writer object(struct live_writer *writer, const char *name)
{
	member(writer, name);
	write_text(writer->output, "{");
	return (struct live_writer){ writer->output, true };
}

static void counts(struct live_writer *writer, const char *name,
		   const uint64_t *values, size_t count)
{
	member(writer, name);
	write_text(writer->output, "[");
	for (size_t index = 0; index < count; index++) {
		char number[NUMBER_BYTES];
		const int length = snprintf(number, sizeof(number), "%" PRIu64,
					    values[index]);
		write_text(writer->output, index == 0 ? "" : ",");
		if (length > 0 && (size_t)length < sizeof(number)) {
			chat_buffer_write(writer->output, number,
					  (size_t)length);
		} else {
			writer->output->is_short = true;
		}
	}
	write_text(writer->output, "]");
}

static void rate(struct live_writer *writer, const char *name, uint64_t count,
		 double seconds)
{
	real(writer, name, seconds > 0 ? (double)count / seconds : (double)NAN);
}

static void milliseconds_per_token(struct live_writer *writer, const char *name,
				   uint64_t count, double seconds)
{
	real(writer, name,
	     count > 0 ? seconds * MILLISECONDS_PER_SECOND / (double)count :
			 (double)NAN);
}

static double measurement_rate(const struct live_measurement *value)
{
	return value->elapsed_ns > 0 && value->count >= 0 ?
		       value->count * NANOSECONDS_PER_SECOND /
			       (double)value->elapsed_ns :
		       (double)NAN;
}

static double measurement_ms(const struct live_measurement *value)
{
	return value->elapsed_ns > 0 && value->count > 0 ?
		       (double)value->elapsed_ns / NANOSECONDS_PER_MILLISECOND /
			       value->count :
		       (double)NAN;
}

static void token_window(struct live_writer *writer,
			 const struct live_measurement *value,
			 const char *rate_name, const char *ms_name,
			 const char *seconds_name)
{
	real(writer, rate_name, measurement_rate(value));
	if (ms_name != NULL) {
		real(writer, ms_name, measurement_ms(value));
	}
	if (seconds_name != NULL) {
		real(writer, seconds_name,
		     (double)value->elapsed_ns / NANOSECONDS_PER_SECOND);
	}
}

static uint64_t sum_counts(const uint64_t *values, size_t count)
{
	uint64_t total = 0;
	for (size_t index = 0; index < count; index++) {
		total += values[index];
	}
	return total;
}

static int compare_times(const void *first, const void *second)
{
	const uint64_t first_value = *(const uint64_t *)first;
	const uint64_t second_value = *(const uint64_t *)second;
	return first_value < second_value ? -1 :
	       first_value > second_value ? 1 :
					    0;
}

static uint64_t percentile(const uint64_t *sorted, uint32_t count,
			   uint32_t fraction)
{
	return count == 0 ? 0 :
			    sorted[(count * fraction + PERCENTILE_SCALE - 1u) /
					   PERCENTILE_SCALE -
				   1u];
}

static void latencies(struct live_writer *writer, const char *name,
		      uint64_t *values, uint32_t count)
{
	struct live_writer latency = object(writer, name);
	qsort(values, count, sizeof(values[0]), compare_times);
	uint64_t total = sum_counts(values, count);
	integer(&latency, "samples", count);
	real(&latency, "mean_ms",
	     count > 0 ? (double)total /
				 ((double)count * NANOSECONDS_PER_MILLISECOND) :
			 (double)NAN);
	real(&latency, "min_ms",
	     count > 0 ? (double)values[0] / NANOSECONDS_PER_MILLISECOND :
			 (double)NAN);
	real(&latency, "p50_ms",
	     count > 0 ? (double)percentile(values, count, PERCENTILE_MIDDLE) /
				 NANOSECONDS_PER_MILLISECOND :
			 (double)NAN);
	real(&latency, "p95_ms",
	     count > 0 ? (double)percentile(values, count, PERCENTILE_HIGH) /
				 NANOSECONDS_PER_MILLISECOND :
			 (double)NAN);
	real(&latency, "max_ms",
	     count > 0 ?
		     (double)values[count - 1u] / NANOSECONDS_PER_MILLISECOND :
		     (double)NAN);
	write_text(writer->output, "}");
}

static void speculation(struct live_writer *writer,
			const struct draft_statistics *value, uint32_t mode,
			uint64_t probes, uint64_t tails)
{
	static const char *const modes[] = { "off", "lookup", "dspark",
					     "hybrid" };
	struct live_writer result = object(writer, "speculation");
	const size_t mode_count = sizeof(modes) / sizeof(modes[0]);
	string(&result, "mode", mode < mode_count ? modes[mode] : "unknown",
	       NUMBER_BYTES);
	integer(&result, "drawing_rounds", value->drawing_rounds);
	counts(&result, "chosen_depth_histogram", value->depth_rounds,
	       DRAFT_DEPTH_COUNTS);
	uint64_t rounds = 0, proposed = 0, reached = 0, accepted = 0;
	for (uint32_t source = 0; source < DRAFT_SOURCES; source++) {
		const struct draft_source_statistics *const source_value =
			&value->source[source];
		rounds += source_value->rounds;
		proposed += sum_counts(source_value->proposed,
				       DRAFT_STAT_POSITIONS);
		reached +=
			sum_counts(source_value->reached, DRAFT_STAT_POSITIONS);
		accepted += sum_counts(source_value->accepted,
				       DRAFT_STAT_POSITIONS);
	}
	integer(&result, "rounds", rounds);
	integer(&result, "proposals", proposed);
	integer(&result, "reached", reached);
	integer(&result, "accepted", accepted);
	real(&result, "acceptance_percent",
	     reached > 0 ? PERCENT_SCALE * (double)accepted / (double)reached :
			   (double)NAN);
	rate(&result, "mean_accepted_per_round", accepted, (double)rounds);
	rate(&result, "tokens_per_round", value->drawing_rounds + accepted,
	     (double)value->drawing_rounds);
	integer(&result, "probes", probes);
	integer(&result, "dspark_tails_run", tails);
	for (uint32_t source = 0; source < DRAFT_SOURCES; source++) {
		const struct draft_source_statistics *const source_value =
			&value->source[source];
		struct live_writer item = object(
			&result, source == DRAFT_LOOKUP ? "lookup" : "dspark");
		integer(&item, "rounds", source_value->rounds);
		counts(&item, "proposed", source_value->proposed,
		       DRAFT_STAT_POSITIONS);
		counts(&item, "reached", source_value->reached,
		       DRAFT_STAT_POSITIONS);
		counts(&item, "accepted", source_value->accepted,
		       DRAFT_STAT_POSITIONS);
		write_text(writer->output, "}");
	}
	member(&result, "confidence_outcomes");
	write_text(writer->output, "[");
	for (uint32_t sample_mode = 0; sample_mode < DRAFT_SAMPLING_MODES;
	     sample_mode++) {
		write_text(writer->output, sample_mode == 0 ? "[" : ",[");
		for (uint32_t position = 0;
		     position < DRAFT_CALIBRATION_POSITIONS; position++) {
			write_text(writer->output, position == 0 ? "[" : ",[");
			for (uint32_t bin = 0; bin < DRAFT_CONFIDENCE_BINS;
			     bin++) {
				const struct draft_outcomes *const outcome =
					&value->confidence[sample_mode]
							  [position][bin];
				const uint64_t array[] = { outcome->produced,
							   outcome->verified,
							   outcome->reached,
							   outcome->matched };
				struct live_writer temporary = { writer->output,
								 true };
				write_text(writer->output, bin == 0 ? "" : ",");
				/* Counts writes a member; an array element uses the same numeric formatter below. */
				write_text(writer->output, "[");
				for (size_t index = 0;
				     index < sizeof(array) / sizeof(array[0]);
				     index++) {
					char number[NUMBER_BYTES];
					const int length = snprintf(
						number, sizeof(number),
						"%" PRIu64, array[index]);
					write_text(writer->output,
						   index == 0 ? "" : ",");
					if (length > 0 &&
					    (size_t)length < sizeof(number)) {
						chat_buffer_write(
							temporary.output,
							number, (size_t)length);
					} else {
						temporary.output->is_short =
							true;
					}
				}
				write_text(writer->output, "]");
			}
			write_text(writer->output, "]");
		}
		write_text(writer->output, "]");
	}
	write_text(writer->output, "]}");
}

static void chunks(struct live_writer *writer, const uint64_t *values)
{
	struct live_writer kinds = object(writer, "steps_by_kind");
	static const char *const names[STATS_STEP_KINDS] = {
		"thin_1", "thin_2", "thin_3", "thin_4", "thin_5", "thin_6",
		"thin_7", "thin_8", "wide",   "full",	"drafts"
	};
	for (uint32_t kind = 0; kind < STATS_STEP_KINDS; kind++) {
		integer(&kinds, names[kind], values[kind]);
	}
	write_text(writer->output, "}");
}

/* The step plan's BATCH programs, when it has any: each one's rows, verifiers, steps, mean step wall time
 * (submission to landing) and the plan's current cost of a step. */
static void batch_programs(struct live_writer *writer,
			   const struct engine_stats *engine)
{
	if (engine->batch_program_count == 0) {
		return;
	}
	member(writer, "batch_programs");
	write_text(writer->output, "[");
	for (uint32_t index = 0; index < engine->batch_program_count &&
				 index < STATS_BATCH_PROGRAMS;
	     index++) {
		const struct stats_batch_program *const program =
			&engine->batch_programs[index];
		write_text(writer->output, index == 0 ? "{" : ",{");
		struct live_writer item = { writer->output, true };
		integer(&item, "rows", program->rows);
		integer(&item, "verifiers", program->verifiers);
		integer(&item, "steps", program->steps);
		real(&item, "mean_ms",
		     program->steps != 0 ?
			     (double)program->step_ns /
				     (double)program->steps /
				     NANOSECONDS_PER_MILLISECOND :
			     0.0);
		real(&item, "cost_ms",
		     (double)program->cost_us / MICROSECONDS_PER_MILLISECOND);
		write_text(writer->output, "}");
	}
	write_text(writer->output, "]");
}

static void card_placement(struct live_writer *writer,
			   const struct engine_card_stats *card)
{
	uint64_t layers[LAYER_MASK_BITS] = { 0 };
	size_t count = 0;
	for (uint32_t layer = 0; layer < LAYER_MASK_BITS; layer++) {
		if ((card->layer_mask & (1ull << layer)) != 0) {
			layers[count++] = layer;
		}
	}
	counts(writer, "layers", layers, count);
	member(writer, "heads");
	write_text(writer->output, "[");
	static const char *const names[] = { "language_model", "draft_markov",
					     "draft_confidence" };
	bool first = true;
	for (size_t index = 0; index < sizeof(names) / sizeof(names[0]);
	     index++) {
		if ((card->head_mask & (1ull << index)) != 0) {
			write_text(writer->output, first ? "" : ",");
			write_string_bytes(writer->output, names[index],
					   strlen(names[index]));
			first = false;
		}
	}
	write_text(writer->output, "]");
}

static void request_engine(struct live_writer *writer,
			   const struct live_snapshot *snapshot,
			   const struct live_request_record *request)
{
	const struct engine_slot_stats *const value = &request->engine;
	const uint64_t endpoint = value->finished_ns != 0 ?
					  value->finished_ns :
					  snapshot->engine.now_ns;
	const double prefill_seconds =
		(double)value->prefill_ns / NANOSECONDS_PER_SECOND;
	double decode_seconds =
		value->first_token_ns > 0 && endpoint >= value->first_token_ns ?
			(double)(endpoint - value->first_token_ns) /
				NANOSECONDS_PER_SECOND :
			0;
	if (value->finished_ns == 0 && request->finished_seconds > 0) {
		const double first_seconds =
			(double)value->first_token_ns / NANOSECONDS_PER_SECOND;
		decode_seconds =
			value->first_token_ns > 0 &&
					request->finished_seconds >=
						first_seconds ?
				request->finished_seconds - first_seconds :
				0;
	}
	const struct engine_slot_stats *const current_slot =
		request->slot < snapshot->engine.slots_total &&
				request->slot < STATS_SLOT_CAPACITY ?
			&snapshot->engine.slots[request->slot] :
			NULL;
	const bool same_request =
		current_slot != NULL && request->finished_seconds == 0 &&
		value->finished_ns == 0 &&
		current_slot->request_tag == value->request_tag &&
		current_slot->submitted_ns == value->submitted_ns;
	const struct live_slot_window empty = { 0 };
	const struct live_slot_window *const short_window =
		same_request ? &snapshot->windows[LIVE_WINDOW_SHORT]
					.slots[request->slot] :
			       &empty;
	const struct live_slot_window *const long_window =
		same_request ? &snapshot->windows[LIVE_WINDOW_LONG]
					.slots[request->slot] :
			       &empty;
	struct live_writer prompt = object(writer, "prompt");
	integer(&prompt, "rendered_tokens", request->rendered_prompt_tokens);
	integer(&prompt, "reused_tokens", request->reused_tokens);
	integer(&prompt, "reused_slot_prefix_tokens",
		request->reused_slot_tokens);
	integer(&prompt, "reused_snapshot_tokens",
		request->reused_snapshot_tokens);
	integer(&prompt, "reused_prompt_checkpoint_tokens",
		request->reused_checkpoint_tokens);
	integer(&prompt, "processed_tokens", value->prompt_processed);
	const uint64_t remaining =
		value->prompt_tokens > value->prompt_processed ?
			value->prompt_tokens - value->prompt_processed :
			0;
	integer(&prompt, "remaining_tokens", remaining);
	real(&prompt, "progress_percent",
	     value->prompt_tokens > 0 ?
		     PERCENT_SCALE * (double)value->prompt_processed /
			     (double)value->prompt_tokens :
		     (double)NAN);
	real(&prompt, "prefill_elapsed_ms",
	     prefill_seconds * MILLISECONDS_PER_SECOND);
	const uint64_t newly_processed =
		value->prompt_processed > value->reused_tokens ?
			value->prompt_processed - value->reused_tokens :
			0;
	rate(&prompt, "prefill_tokens_per_second", newly_processed,
	     prefill_seconds);
	milliseconds_per_token(&prompt, "prefill_ms_per_token", newly_processed,
			       prefill_seconds);
	token_window(&prompt, &short_window->prefill,
		     "prefill_tokens_per_second_10s", NULL,
		     "window_seconds_10s");
	token_window(&prompt, &long_window->prefill,
		     "prefill_tokens_per_second_60s", NULL,
		     "window_seconds_60s");
	real(&prompt, "time_to_first_token_ms",
	     value->first_token_ns > 0 ?
		     ((double)value->first_token_ns / NANOSECONDS_PER_SECOND -
		      request->arrival_seconds) *
			     MILLISECONDS_PER_SECOND :
		     (double)NAN);
	chunks(&prompt, value->chunks);
	write_text(writer->output, "}");
	struct live_writer decode = object(writer, "decode");
	integer(&decode, "generated_tokens", value->generated_tokens);
	integer(&decode, "reasoning_tokens", request->reasoning_tokens);
	integer(&decode, "content_tokens", request->content_tokens);
	integer(&decode, "tool_call_tokens", request->tool_call_tokens);
	real(&decode, "elapsed_ms", decode_seconds * MILLISECONDS_PER_SECOND);
	const uint64_t timed_tokens =
		value->generated_tokens > 0 ? value->generated_tokens - 1u : 0;
	rate(&decode, "tokens_per_second_since_first", timed_tokens,
	     decode_seconds);
	milliseconds_per_token(&decode, "ms_per_token_since_first",
			       timed_tokens, decode_seconds);
	token_window(&decode, &short_window->decode,
		     "tokens_per_second_interval", "ms_per_token_interval",
		     NULL);
	token_window(&decode, &short_window->decode, "tokens_per_second_10s",
		     "ms_per_token_10s", "window_seconds_10s");
	token_window(&decode, &long_window->decode, "tokens_per_second_60s",
		     "ms_per_token_60s", "window_seconds_60s");
	uint64_t durations[STATS_TOKEN_RING] = { 0 };
	uint32_t count = 0;
	const uint32_t ring_count = value->token_ring_count < STATS_TOKEN_RING ?
					    value->token_ring_count :
					    STATS_TOKEN_RING;
	for (uint32_t index = 1; index < ring_count; index++) {
		const uint32_t first =
			(value->token_ring_next + STATS_TOKEN_RING -
			 ring_count + index - 1u) %
			STATS_TOKEN_RING;
		const uint32_t second = (first + 1u) % STATS_TOKEN_RING;
		if (value->token_times[second] >= value->token_times[first]) {
			durations[count++] = value->token_times[second] -
					     value->token_times[first];
		}
	}
	latencies(&decode, "inter_token_latency", durations, count);
	write_text(writer->output, "}");
	speculation(writer, &value->speculation, value->speculation_mode,
		    value->probes, value->dspark_tails);
	struct live_writer current = object(writer, "current_speculation");
	integer(&current, "chosen_depth", value->chosen_depth);
	string(&current, "source",
	       value->draft_source == DRAFT_LOOKUP ? "lookup" : "dspark",
	       NUMBER_BYTES);
	boolean(&current, "has_confidence", value->has_confidence);
	uint64_t bins[DRAFT_CALIBRATION_POSITIONS];
	for (uint32_t position = 0; position < DRAFT_CALIBRATION_POSITIONS;
	     position++) {
		bins[position] = value->confidence_bins[position];
	}
	counts(&current, "last_confidence_bins", bins,
	       DRAFT_CALIBRATION_POSITIONS);
	write_text(writer->output, "}");
	struct live_writer state = object(writer, "state");
	integer(&state, "position", value->position);
	integer(&state, "context_capacity", value->context_capacity);
	real(&state, "context_used_percent",
	     value->context_capacity > 0 ?
		     PERCENT_SCALE * (double)value->position /
			     (double)value->context_capacity :
		     (double)NAN);
	integer(&state, "reach", value->reach);
	integer(&state, "decoder_from", value->decoder_from);
	integer(&state, "checkpoint_G", value->checkpoint);
	integer(&state, "kept_lag", value->kept_lag);
	integer(&state, "reach_lag", value->reach_lag);
	write_text(writer->output, "}");
}

static void request(struct live_writer *writer,
		    const struct live_snapshot *snapshot,
		    const struct live_request_record *value)
{
	struct live_writer result = { writer->output, true };
	write_text(writer->output, "{");
	struct live_writer identity = object(&result, "identity");
	string(&identity, "user", value->user[0] != '\0' ? value->user : "-",
	       SERVER_USER_BYTES);
	if (value->slot == UINT32_MAX) {
		member(&identity, "slot");
		write_text(writer->output, "null");
	} else {
		integer(&identity, "slot", value->slot);
	}
	integer(&identity, "request_tag", value->request_tag);
	string(&identity, "request_id", value->identifier,
	       sizeof(value->identifier));
	string(&identity, "client", value->client, sizeof(value->client));
	string(&identity, "model", value->model, sizeof(value->model));
	boolean(&identity, "stream", value->options.is_streamed);
	boolean(&identity, "tools_declared", value->tools_declared);
	integer(&identity, "reasoning_effort", value->options.reasoning_effort);
	real(&identity, "temperature", value->options.temperature);
	real(&identity, "top_p", value->options.top_p);
	integer(&identity, "seed", value->options.seed);
	integer(&identity, "max_tokens", value->options.max_tokens);
	member(&identity, "stop_sequences");
	write_text(writer->output, "[");
	for (uint32_t index = 0;
	     index < value->options.stop_count && index < CHAT_MOST_STOPS;
	     index++) {
		write_text(writer->output, index == 0 ? "" : ",");
		const size_t length =
			value->options.stop_byte_counts[index] <
					CHAT_MOST_STOP_BYTES ?
				value->options.stop_byte_counts[index] :
				CHAT_MOST_STOP_BYTES;
		write_string_bytes(writer->output, value->options.stops[index],
				   length);
	}
	write_text(writer->output, "]}");
	struct live_writer timeline = object(&result, "timeline");
	const double endpoint = value->finished_seconds > 0 ?
					value->finished_seconds :
					snapshot->now_seconds;
	string(&timeline, "phase", value->phase, sizeof(value->phase));
	real(&timeline, "phase_elapsed_ms",
	     (endpoint - value->phase_seconds) * MILLISECONDS_PER_SECOND);
	real(&timeline, "elapsed_ms",
	     (endpoint - value->arrival_seconds) * MILLISECONDS_PER_SECOND);
	double queue_wait = value->queue_wait_seconds;
	if (value->slot == UINT32_MAX && strcmp(value->phase, "queued") == 0 &&
	    endpoint >= value->phase_seconds) {
		queue_wait += endpoint - value->phase_seconds;
	} else if (value->slot != UINT32_MAX &&
		   value->engine.submitted_ns > 0) {
		const uint64_t queue_end = value->engine.started_ns > 0 ?
						   value->engine.started_ns :
					   value->engine.phase == STATS_QUEUED ?
						   snapshot->engine.now_ns :
						   value->engine.submitted_ns;
		if (queue_end >= value->engine.submitted_ns) {
			queue_wait += (double)(queue_end -
					       value->engine.submitted_ns) /
				      NANOSECONDS_PER_SECOND;
		}
	}
	real(&timeline, "queue_wait_ms", queue_wait * MILLISECONDS_PER_SECOND);
	write_text(writer->output, "}");
	request_engine(&result, snapshot, value);
	struct live_writer output = object(&result, "output");
	integer(&output, "bytes_sent", value->bytes_sent);
	integer(&output, "events_sent", value->events_sent);
	string(&output, "finish_reason", value->finish_reason,
	       sizeof(value->finish_reason));
	string(&output, "last_error", value->last_error,
	       sizeof(value->last_error));
	write_text(writer->output, "}}");
}

static void sessions(struct live_writer *writer,
		     const struct live_snapshot *snapshot)
{
	static const char *const phases[] = { "idle", "queued", "prompt",
					      "decode", "finishing" };
	member(writer, "sessions");
	write_text(writer->output, "[");
	for (uint32_t slot = 0; slot < snapshot->engine.slots_total; slot++) {
		const struct engine_session_stats *const state =
			&snapshot->engine.sessions[slot];
		const struct live_session *const retained =
			&snapshot->sessions[slot];
		const struct live_request_record *current = NULL;
		for (uint32_t index = 0;
		     state->active && state->operation == 0 &&
		     index < snapshot->request_count;
		     index++) {
			const struct live_request_record *const candidate =
				&snapshot->requests[index];
			if (candidate->slot == slot &&
			    candidate->finished_seconds == 0 &&
			    candidate->request_tag ==
				    snapshot->engine.slots[slot].request_tag &&
			    candidate->engine.submitted_ns ==
				    snapshot->engine.slots[slot].submitted_ns) {
				current = candidate;
				break;
			}
		}
		const char *const phase =
			!state->active ? "idle" :
			state->operation == SESSION_SAVE ?
					 "snapshot_save" :
			state->operation == SESSION_LOAD ?
					 "snapshot_load" :
			state->operation == SESSION_CLONE ?
					 "session_clone" :
			state->phase < sizeof(phases) / sizeof(phases[0]) ?
					 phases[state->phase] :
					 "unknown";
		write_text(writer->output, slot == 0 ? "{" : ",{");
		struct live_writer item = { writer->output, true };
		integer(&item, "slot", slot);
		boolean(&item, "active", state->active);
		string(&item, "state", phase, LIVE_STATS_PHASE_BYTES);
		real(&item, "idle_seconds",
		     !state->active && snapshot->engine.now_ns >=
					       state->idle_since_ns ?
			     (double)(snapshot->engine.now_ns -
				      state->idle_since_ns) /
				     NANOSECONDS_PER_SECOND :
			     0);
		const char *const user =
			retained->user[0] != '\0' ? retained->user :
			current != NULL && current->user[0] != '\0' ?
						    current->user :
						    "-";
		string(&item, "user", user, SERVER_USER_BYTES);
		integer(&item, "position", state->position);
		integer(&item, "reach", state->reach);
		integer(&item, "context_capacity", state->context_capacity);
		real(&item, "context_used_percent",
		     state->context_capacity > 0 ?
			     PERCENT_SCALE * (double)state->position /
				     (double)state->context_capacity :
			     (double)NAN);
		member(&item, "current_request");
		if (current != NULL) {
			request(&item, snapshot, current);
		} else {
			write_text(writer->output, "null");
		}
		member(&item, "last_request");
		if (retained->has_last_request) {
			request(&item, snapshot, &retained->last_request);
		} else {
			write_text(writer->output, "null");
		}
		struct live_writer transfers = object(&item, "snapshots");
		integer(&transfers, "saved", state->snapshots_saved);
		integer(&transfers, "loaded", state->snapshots_loaded);
		integer(&transfers, "save_bytes", state->snapshot_save_bytes);
		integer(&transfers, "load_bytes", state->snapshot_load_bytes);
		real(&transfers, "save_ms",
		     (double)state->snapshot_save_ns /
			     NANOSECONDS_PER_MILLISECOND);
		real(&transfers, "load_ms",
		     (double)state->snapshot_load_ns /
			     NANOSECONDS_PER_MILLISECOND);
		write_text(writer->output, "}}");
	}
	write_text(writer->output, "]");
}

result_status live_stats_json_entry(const struct live_snapshot *snapshot,
				    struct chat_buffer *output)
{
	if (snapshot == NULL || output == NULL ||
	    snapshot->request_count > LIVE_STATS_REQUEST_CAPACITY ||
	    snapshot->finished_count > LIVE_STATS_FINISHED_CAPACITY ||
	    snapshot->user_count > SERVER_MOST_USERS ||
	    snapshot->engine.card_count > BACKEND_MAXIMUM_CARDS ||
	    snapshot->engine.slots_total > STATS_SLOT_CAPACITY) {
		return STATUS_INVALID_ARGUMENT;
	}
	output->count = 0;
	output->sent = 0;
	output->is_short = false;
	write_text(output, "{");
	struct live_writer root = { output, true };
	integer(&root, "version", LIVE_SCHEMA_VERSION);
	struct live_writer global = object(&root, "global");
	real_precision(&global, "wall_seconds", snapshot->wall_seconds,
		       WALL_SIGNIFICANT_DIGITS);
	char wall_time[WALL_TIME_BYTES] = "unavailable";
	if (isfinite(snapshot->wall_seconds) && snapshot->wall_seconds >= 0 &&
	    snapshot->wall_seconds <= WALL_LAST_SECOND) {
		const time_t seconds = (time_t)snapshot->wall_seconds;
		struct tm calendar;
		if (gmtime_r(&seconds, &calendar) != NULL &&
		    strftime(wall_time, sizeof(wall_time), "%Y-%m-%dT%H:%M:%SZ",
			     &calendar) == 0) {
			(void)snprintf(wall_time, sizeof(wall_time), "%s",
				       "unavailable");
		}
	}
	string(&global, "wall_time", wall_time, sizeof(wall_time));
	real(&global, "uptime_seconds",
	     snapshot->now_seconds - snapshot->started_seconds);
	real(&global, "interval_seconds",
	     (double)snapshot->windows[LIVE_WINDOW_SHORT].decode.elapsed_ns /
		     NANOSECONDS_PER_SECOND);
	string(&global, "build_id", snapshot->build_id,
	       sizeof(snapshot->build_id));
	string(&global, "model_alias", snapshot->model_alias,
	       sizeof(snapshot->model_alias));
	string(&global, "card_order", snapshot->card_order,
	       sizeof(snapshot->card_order));
	string(&global, "placement", snapshot->placement,
	       sizeof(snapshot->placement));
	const struct engine_stats *const engine = &snapshot->engine;
	const struct live_window_record *const short_window =
		&snapshot->windows[LIVE_WINDOW_SHORT];
	const struct live_window_record *const long_window =
		&snapshot->windows[LIVE_WINDOW_LONG];
	integer(&global, "slots_total", engine->slots_total);
	integer(&global, "slots_busy", engine->slots_busy);
	integer(&global, "slots_idle",
		engine->slots_total >= engine->slots_busy ?
			engine->slots_total - engine->slots_busy :
			0);
	integer(&global, "concurrent_requests", snapshot->request_count);
	integer(&global, "queued_requests", snapshot->queued_requests);
	integer(&global, "queue_depth", snapshot->queue_depth);
	integer(&global, "user_queue_depth", snapshot->user_queue_depth);
	integer(&global, "open_connections", snapshot->open_connections);
	integer(&global, "served_requests", snapshot->served_requests);
	integer(&global, "refused_503", snapshot->refused_503);
	integer(&global, "refused_400", snapshot->refused_400);
	integer(&global, "refused_auth", snapshot->refused_auth);
	integer(&global, "failed_requests", snapshot->failed_requests);
	integer(&global, "cancelled_requests", snapshot->cancelled_requests);
	integer(&global, "prompt_tokens", engine->prompt_tokens);
	integer(&global, "cached_tokens", engine->cached_tokens);
	integer(&global, "generated_tokens", engine->generated_tokens);
	member(&global, "users");
	write_text(output, "[");
	for (uint32_t index = 0;
	     index < snapshot->user_count && index < SERVER_MOST_USERS;
	     index++) {
		const struct live_user *const user = &snapshot->users[index];
		write_text(output, index == 0 ? "{" : ",{");
		struct live_writer item = { output, true };
		string(&item, "user", user->user, sizeof(user->user));
		integer(&item, "concurrent_requests", user->concurrent_requests);
		integer(&item, "queued_requests", user->queued_requests);
		integer(&item, "served_requests", user->served_requests);
		integer(&item, "prompt_tokens", user->prompt_tokens);
		integer(&item, "cached_tokens", user->cached_tokens);
		integer(&item, "generated_tokens", user->generated_tokens);
		integer(&item, "refused", user->refused);
		write_text(output, "}");
	}
	write_text(output, "]");
	const double uptime =
		engine->now_ns >= engine->started_ns ?
			(double)(engine->now_ns - engine->started_ns) /
				NANOSECONDS_PER_SECOND :
			0;
	token_window(&global, &short_window->decode,
		     "decode_tokens_per_second_interval",
		     "decode_ms_per_token_interval", NULL);
	token_window(&global, &short_window->decode,
		     "decode_tokens_per_second_10s", "decode_ms_per_token_10s",
		     "window_seconds_10s");
	token_window(&global, &long_window->decode,
		     "decode_tokens_per_second_60s", "decode_ms_per_token_60s",
		     "window_seconds_60s");
	rate(&global, "decode_tokens_per_second_since_start",
	     engine->generated_tokens, uptime);
	milliseconds_per_token(&global, "decode_ms_per_token_since_start",
			       engine->generated_tokens, uptime);
	token_window(&global, &short_window->prefill,
		     "prefill_tokens_per_second_interval", NULL, NULL);
	token_window(&global, &short_window->prefill,
		     "prefill_tokens_per_second_10s", NULL, NULL);
	token_window(&global, &long_window->prefill,
		     "prefill_tokens_per_second_60s", NULL, NULL);
	rate(&global, "prefill_tokens_per_second_since_start",
	     engine->prefill_tokens, uptime);
	integer(&global, "flights_in_air", engine->flights);
	real(&global, "steps_landed_per_second",
	     measurement_rate(&short_window->steps));
	const uint64_t window_start =
		engine->now_ns >= short_window->steps.elapsed_ns ?
			engine->now_ns - short_window->steps.elapsed_ns :
			0;
	uint64_t step_latencies[STATS_STEP_RING] = { 0 };
	uint32_t sample_count = 0;
	for (uint32_t index = 0;
	     index < engine->step_ring_count && index < STATS_STEP_RING;
	     index++) {
		if (engine->step_samples[index].landed_ns > window_start) {
			step_latencies[sample_count++] =
				engine->step_samples[index].latency_ns;
		}
	}
	latencies(&global, "step_latency", step_latencies, sample_count);
	boolean(&global, "step_latency_ring_truncated",
		short_window->steps.count > (double)sample_count);
	chunks(&global, engine->steps);
	batch_programs(&global, engine);
	integer(&global, "prompt_checkpoints_saved", engine->checkpoints_saved);
	integer(&global, "prompt_checkpoints_restored",
		engine->checkpoints_restored);
	integer(&global, "snapshots_saved", engine->snapshots_saved);
	integer(&global, "snapshots_loaded", engine->snapshots_loaded);
	integer(&global, "snapshot_save_bytes", engine->snapshot_save_bytes);
	integer(&global, "snapshot_load_bytes", engine->snapshot_load_bytes);
	real(&global, "snapshot_save_ms",
	     (double)engine->snapshot_save_ns / NANOSECONDS_PER_MILLISECOND);
	real(&global, "snapshot_load_ms",
	     (double)engine->snapshot_load_ns / NANOSECONDS_PER_MILLISECOND);
	integer(&global, "event_queue_depth", engine->event_queue_depth);
	integer(&global, "event_backpressure_stalls",
		engine->backpressure_stalls);
	speculation(&global, &engine->speculation, engine->speculation_mode,
		    engine->probes, engine->dspark_tails);
	write_text(output, "}");
	member(&root, "cards");
	write_text(output, "[");
	for (uint32_t index = 0; index < engine->card_count; index++) {
		const struct engine_card_stats *const value =
			&engine->cards[index];
		struct live_writer card = { output, true };
		write_text(output, index == 0 ? "{" : ",{");
		integer(&card, "engine_index", value->engine_index);
		integer(&card, "backend_index", value->backend_index);
		string(&card, "name", value->description.name,
		       sizeof(value->description.name));
		string(&card, "pci_bus_id", value->description.bus_address,
		       sizeof(value->description.bus_address));
		integer(&card, "arithmetic_class",
			value->description.arithmetic_class);
		integer(&card, "layer_mask", value->layer_mask);
		integer(&card, "head_mask", value->head_mask);
		card_placement(&card, value);
		boolean(&card, "holds_draft", value->holds_draft);
		integer(&card, "arena_bytes", value->arena_bytes);
		boolean(&card, "free_memory_available",
			value->free_memory_available);
		if (value->free_memory_available) {
			integer(&card, "free_device_bytes",
				value->description.memory_free_bytes);
		} else {
			member(&card, "free_device_bytes");
			write_text(output, "null");
		}
		integer(&card, "steps", value->steps);
		integer(&card, "visits", value->visits);
		boolean(&card, "busy_time_available", value->busy_available);
		integer(&card, "busy_observed_monotonic_ns",
			value->busy_observed_ns);
		real(&card, "busy_ms_total",
		     value->busy_available ?
			     (double)value->busy_ns /
				     NANOSECONDS_PER_MILLISECOND :
			     (double)NAN);
		const struct live_measurement empty = { 0 };
		const struct live_measurement *const short_busy =
			value->busy_available ? &short_window->cards[index] :
						&empty;
		const struct live_measurement *const long_busy =
			value->busy_available ? &long_window->cards[index] :
						&empty;
		const double short_ms =
			short_busy->elapsed_ns > 0 ?
				short_busy->count /
					NANOSECONDS_PER_MILLISECOND :
				(double)NAN;
		const double long_ms =
			long_busy->elapsed_ns > 0 ?
				long_busy->count / NANOSECONDS_PER_MILLISECOND :
				(double)NAN;
		const double short_percent =
			short_busy->elapsed_ns > 0 ?
				PERCENT_SCALE * short_busy->count /
					(double)short_busy->elapsed_ns :
				(double)NAN;
		const double long_percent =
			long_busy->elapsed_ns > 0 ?
				PERCENT_SCALE * long_busy->count /
					(double)long_busy->elapsed_ns :
				(double)NAN;
		real(&card, "busy_ms_interval", short_ms);
		real(&card, "busy_percent_interval", short_percent);
		real(&card, "busy_ms_10s", short_ms);
		real(&card, "busy_percent_10s", short_percent);
		real(&card, "window_seconds_10s",
		     (double)short_busy->elapsed_ns / NANOSECONDS_PER_SECOND);
		real(&card, "busy_ms_60s", long_ms);
		real(&card, "busy_percent_60s", long_percent);
		real(&card, "window_seconds_60s",
		     (double)long_busy->elapsed_ns / NANOSECONDS_PER_SECOND);
		real(&card, "busy_percent_since_start",
		     value->busy_available && value->busy_observed_ns >
						      engine->started_ns ?
			     PERCENT_SCALE * (double)value->busy_ns /
				     (double)(value->busy_observed_ns -
					      engine->started_ns) :
			     (double)NAN);
		write_text(output, "}");
	}
	write_text(output, "]");
	sessions(&root, snapshot);
	member(&root, "requests");
	write_text(output, "[");
	for (uint32_t index = 0; index < snapshot->request_count; index++) {
		write_text(output, index == 0 ? "" : ",");
		request(&root, snapshot, &snapshot->requests[index]);
	}
	write_text(output, "]");
	member(&root, "finished");
	write_text(output, "[");
	for (uint32_t index = 0; index < snapshot->finished_count; index++) {
		write_text(output, index == 0 ? "" : ",");
		request(&root, snapshot, &snapshot->finished[index]);
	}
	write_text(output, "]}");
	return output->is_short ? STATUS_OUT_OF_MEMORY : STATUS_OK;
}

struct text_writer {
	struct chat_buffer *output;
	uint32_t columns, column, indent;
};

static void text_indent(struct text_writer *writer)
{
	for (uint32_t column = 0; column < writer->indent; column++) {
		write_text(writer->output, " ");
	}
	writer->column = writer->indent;
}

static void text_bytes(struct text_writer *writer, const char *bytes,
		       size_t count)
{
	for (size_t position = 0; position < count;) {
		if (bytes[position] == '\n') {
			write_text(writer->output, "\n");
			writer->column = 0;
			position++;
			continue;
		}
		bool is_valid = false;
		const size_t sequence_bytes = tokenizer_utf8_sequence_bytes(
			bytes, position, count, &is_valid);
		const size_t copied_bytes =
			is_valid && sequence_bytes > 0 ? sequence_bytes : 1u;
		if (writer->column + copied_bytes > writer->columns) {
			write_text(writer->output, "\n");
			writer->column = 0;
		}
		if (writer->column == 0) {
			text_indent(writer);
		}
		chat_buffer_write(writer->output, bytes + position,
				  copied_bytes);
		writer->column += (uint32_t)copied_bytes;
		position += copied_bytes;
	}
}

static void text_word(struct text_writer *writer, const char *word)
{
	text_bytes(writer, word, strlen(word));
}

static bool array_has_no_objects(struct json_value value)
{
	size_t cursor = value.begin;
	struct json_value item;
	while (json_next(value, &cursor, NULL, &item)) {
		if (item.text[item.begin] == '{' ||
		    (item.text[item.begin] == '[' &&
		     !array_has_no_objects(item))) {
			return false;
		}
	}
	return true;
}

static void text_value(struct text_writer *writer, struct json_value value,
		       uint32_t depth)
{
	if (depth >= JSON_MAXIMUM_DEPTH) {
		return; /* the checked document cannot contain a deeper container */
	}
	const bool is_object = value.text[value.begin] == '{';
	const bool is_array = value.text[value.begin] == '[';
	if (!is_object && !is_array) {
		text_bytes(writer, value.text + value.begin,
			   value.end - value.begin);
		text_word(writer, "\n");
		return;
	}
	if (is_array && array_has_no_objects(value)) {
		text_bytes(writer, value.text + value.begin,
			   value.end - value.begin);
		text_word(writer, "\n");
		return;
	}
	size_t cursor = value.begin;
	struct json_value key, item;
	uint64_t index = 0;
	while (json_next(value, &cursor, &key, &item)) {
		if (is_object) {
			text_bytes(writer, key.text + key.begin + 1u,
				   key.end - key.begin - JSON_QUOTES_BYTES);
		} else {
			char number[NUMBER_BYTES];
			const int count = snprintf(number, sizeof(number),
						   "[%" PRIu64 "]", index);
			if (count > 0 && (size_t)count < sizeof(number)) {
				text_bytes(writer, number, (size_t)count);
			} else {
				writer->output->is_short = true;
			}
		}
		text_word(writer, ": ");
		const uint32_t saved_indent = writer->indent;
		const bool nested = item.text[item.begin] == '{' ||
				    (item.text[item.begin] == '[' &&
				     !array_has_no_objects(item));
		if (nested) {
			text_word(writer, "\n");
			writer->indent = saved_indent < MAXIMUM_INDENT_COLUMNS ?
						 saved_indent + INDENT_COLUMNS :
						 saved_indent;
		}
		text_value(writer, item, depth + 1u);
		writer->indent = saved_indent;
		index++;
	}
	if (index == 0) {
		text_word(writer, is_object ? "{}\n" : "[]\n");
	}
}

static struct json_value find_member(struct json_value object_value,
				     const char *name)
{
	if (object_value.text == NULL ||
	    object_value.text[object_value.begin] != '{') {
		return (struct json_value){ NULL, 0, 0 };
	}
	size_t cursor = object_value.begin;
	struct json_value key, value;
	const size_t name_length = strlen(name);
	while (json_next(object_value, &cursor, &key, &value)) {
		if (key.end - key.begin == name_length + JSON_QUOTES_BYTES &&
		    memcmp(key.text + key.begin + 1u, name, name_length) == 0) {
			return value;
		}
	}
	return (struct json_value){ NULL, 0, 0 };
}

#define TABLE_WIDE_COLUMNS 140u
#define TABLE_MOST_COLUMNS 160u
#define TABLE_CELL_BYTES 96u
#define TABLE_NUMBER_PRECISION 3
#define TABLE_SCALE 1000.0
#define TABLE_MOST_SCALE 6u
#define TABLE_CLOCK_OFFSET 11u
#define TABLE_CLOCK_BYTES 8u
#define TABLE_COMPACT_USER_COLUMNS 16u
#define TABLE_FIRST_METRIC_COLUMN 3u
#define TABLE_CARD_NUMBER_COLUMNS 5u

enum table_global_metric {
	TABLE_CONCURRENT,
	TABLE_SLOTS,
	TABLE_QUEUED,
	TABLE_QUEUE_DEPTH,
	TABLE_DECODE,
	TABLE_MS_PER_TOKEN,
	TABLE_PREFILL,
	TABLE_FLIGHTS
};

struct table_column {
	const char *group, *name, *heading, *compact;
	uint32_t width, narrow;
	double divisor;
};

/* The compact view shares saved username width across numeric measurements. */
static const struct table_column table_columns[] = {
	{ "identity", "user", "user", "user", 32, 32, 1 },
	{ "identity", "slot", "slot", "sl", 4, 2, 1 },
	{ "timeline", "phase", "phase", "p", 7, 1, 1 },
	{ "prompt", "rendered_tokens", "prompt", "pr", 7, 2, 1 },
	{ "decode", "generated_tokens", "generated", "gn", 8, 2, 1 },
	{ "decode", "tokens_per_second_10s", "d10/s", "d10", 8, 2, 1 },
	{ "decode", "tokens_per_second_since_first", "davg/s", "dav", 8, 2, 1 },
	{ "decode", "ms_per_token_10s", "ms10", "m10", 8, 2, 1 },
	{ "decode", "ms_per_token_since_first", "msavg", "mav", 8, 2, 1 },
	{ "prompt", "prefill_tokens_per_second_10s", "p10/s", "p10", 8, 2, 1 },
	{ "prompt", "prefill_tokens_per_second", "pavg/s", "pav", 8, 2, 1 },
	{ "prompt", "time_to_first_token_ms", "TTFT/ms", "tt", 7, 2, 1 },
	{ "latency", "p50_ms", "ITL50ms", "50", 7, 2, 1 },
	{ "latency", "p95_ms", "ITL95ms", "95", 7, 2, 1 },
	{ "speculation", "mean_accepted_per_round", "acc/r", "ar", 6, 2, 1 },
	{ "current_speculation", "chosen_depth", "depth", "d", 5, 1, 1 },
	{ "state", "context_used_percent", "ctx%", "cx", 6, 2, 1 }
};

#define TABLE_COLUMN_COUNT (sizeof(table_columns) / sizeof(table_columns[0]))

static void table_number(struct json_value value, double divisor,
			 uint32_t width, char result[TABLE_CELL_BYTES])
{
	char input[NUMBER_BYTES];
	(void)snprintf(result, TABLE_CELL_BYTES, "%s", "-");
	if (value.text == NULL || value.end - value.begin >= sizeof(input)) {
		return;
	}
	const size_t length = value.end - value.begin;
	memcpy(input, value.text + value.begin, length);
	input[length] = '\0';
	char *end = NULL;
	double number = strtod(input, &end) / divisor;
	if (end != input + length || end == input || !isfinite(number) ||
	    number < 0) {
		return;
	}
	/* Keep small integers exact; compact large values only as needed to fit. */
	int count = snprintf(result, TABLE_CELL_BYTES, "%.*g",
			     NUMBER_SIGNIFICANT_DIGITS, number);
	if (count > 0 && (uint32_t)count <= width) {
		return;
	}
	static const char suffix[] = " KMGTPE";
	uint32_t scale = 0;
	while (number >= TABLE_SCALE && scale < TABLE_MOST_SCALE) {
		number /= TABLE_SCALE;
		scale++;
	}
	for (int precision = TABLE_NUMBER_PRECISION; precision > 0;
	     precision--) {
		count = scale == 0 ?
				snprintf(result, TABLE_CELL_BYTES, "%.*g",
					 precision, number) :
				snprintf(result, TABLE_CELL_BYTES, "%.*g%c",
					 precision, number, suffix[scale]);
		if (count > 0 && (uint32_t)count <= width) {
			return;
		}
	}
	(void)snprintf(result, TABLE_CELL_BYTES, "%s", number < 1 ? "<1" : ">");
}

static void table_line(struct chat_buffer *output, const char *line,
		       uint32_t columns)
{
	const size_t length = strlen(line);
	const size_t kept = length < columns ? length : columns;
	chat_buffer_write(output, line, kept);
	if (length > columns && !output->is_short) {
		output->bytes[output->count - 1u] = '~';
	}
	write_text(output, "\n");
}

static void table_cell(char line[TABLE_MOST_COLUMNS + 1u], size_t *used,
		       const char *text, uint32_t width)
{
	const int count = snprintf(line + *used,
				   TABLE_MOST_COLUMNS + 1u - *used, "%s%*.*s",
				   *used == 0 ? "" : " ", (int)width,
				   (int)width, text);
	if (count > 0 && (size_t)count < TABLE_MOST_COLUMNS + 1u - *used) {
		*used += (size_t)count;
	}
}

static void table_value(struct json_value value, double divisor, uint32_t width,
			bool mark, char cell[TABLE_CELL_BYTES])
{
	if (value.text == NULL || value.text[value.begin] != '"') {
		table_number(value, divisor, width, cell);
		return;
	}
	const size_t length = value.end - value.begin - JSON_QUOTES_BYTES;
	const size_t kept = length < width ? length : width;
	memcpy(cell, value.text + value.begin + 1u, kept);
	if (mark && length > width && kept > 0) {
		cell[kept - 1u] = '~';
	}
	cell[kept] =
		'\0'; /* retain JSON escaping rather than unescaping terminal controls */
}

static void table_cards(struct json_value cards, uint32_t columns, bool wide,
			struct chat_buffer *output)
{
	char line[TABLE_MOST_COLUMNS + 1u] = "cards10%";
	size_t used = strlen(line);
	struct json_value card;
	for (size_t cursor = cards.begin;
	     cards.text != NULL && cards.text[cards.begin] == '[' &&
	     json_next(cards, &cursor, NULL, &card);) {
		char index[TABLE_CELL_BYTES], busy[TABLE_CELL_BYTES];
		table_number(find_member(card, "engine_index"), 1,
			     TABLE_NUMBER_PRECISION, index);
		table_number(find_member(card, "busy_percent_10s"), 1,
			     wide ? TABLE_CARD_NUMBER_COLUMNS :
				    TABLE_NUMBER_PRECISION,
			     busy);
		const int written = snprintf(line + used, sizeof(line) - used,
					     " c%.3s=%.5s%%", index, busy);
		if (written <= 0 || (size_t)written >= sizeof(line) - used) {
			break;
		}
		used += (size_t)written;
	}
	if (used == strlen("cards10%")) {
		(void)snprintf(line + used, sizeof(line) - used, "%s", " -");
	}
	table_line(output, line, columns);
}

static void table_request(struct json_value request_value,
			  struct json_value session,
			  const uint32_t widths[TABLE_COLUMN_COUNT],
			  uint32_t columns, struct chat_buffer *output)
{
	char line[TABLE_MOST_COLUMNS + 1u] = "";
	size_t used = 0;
	for (size_t index = 0; index < TABLE_COLUMN_COUNT; index++) {
		const struct table_column *const column = &table_columns[index];
		const bool latency = strcmp(column->group, "latency") == 0;
		struct json_value group = find_member(
			request_value, latency ? "decode" : column->group);
		if (latency) {
			group = find_member(group, "inter_token_latency");
		}
		struct json_value value = find_member(group, column->name);
		if (request_value.text == NULL ||
		    request_value.text[request_value.begin] != '{') {
			value = index == 0 ?
					find_member(session, "user") :
				index == 1 ?
					find_member(session, "slot") :
				strcmp(column->name, "phase") == 0 ?
					find_member(session, "state") :
				strcmp(column->name, "context_used_percent") ==
						0 ?
					find_member(session,
						    "context_used_percent") :
					value;
		}
		char cell[TABLE_CELL_BYTES];
		table_value(value, column->divisor, widths[index], index == 0,
			    cell);
		if (strcmp(column->name, "phase") == 0 && value.text != NULL) {
			const char *const begin = value.text + value.begin;
			const size_t length = value.end - value.begin;
			if (length == strlen("\"snapshot_save\"") &&
			    memcmp(begin, "\"snapshot_save\"", length) == 0) {
				(void)snprintf(cell, sizeof(cell), "%s",
					       widths[index] > 1 ? "save" :
								   "s");
			} else if (length == strlen("\"snapshot_load\"") &&
				   memcmp(begin, "\"snapshot_load\"", length) ==
					   0) {
				(void)snprintf(cell, sizeof(cell), "%s",
					       widths[index] > 1 ? "load" :
								   "l");
			}
		}
		table_cell(line, &used, cell, widths[index]);
	}
	table_line(output, line, columns);
}

static void table_idle(struct json_value session, uint32_t user_width,
		       uint32_t columns, struct chat_buffer *output)
{
	char slot[TABLE_CELL_BYTES], user[TABLE_CELL_BYTES],
		seconds[TABLE_CELL_BYTES];
	char position[TABLE_CELL_BYTES], capacity[TABLE_CELL_BYTES],
		reach[TABLE_CELL_BYTES], percent[TABLE_CELL_BYTES];
	table_number(find_member(session, "slot"), 1, TABLE_NUMBER_PRECISION,
		     slot);
	table_value(find_member(session, "user"), 1, user_width, true, user);
	table_number(find_member(session, "idle_seconds"), 1,
		     TABLE_CARD_NUMBER_COLUMNS, seconds);
	table_number(find_member(session, "position"), 1,
		     TABLE_CARD_NUMBER_COLUMNS, position);
	table_number(find_member(session, "context_capacity"), 1,
		     TABLE_CARD_NUMBER_COLUMNS, capacity);
	table_number(find_member(session, "reach"), 1,
		     TABLE_CARD_NUMBER_COLUMNS, reach);
	table_number(find_member(session, "context_used_percent"), 1,
		     TABLE_CARD_NUMBER_COLUMNS, percent);
	char line[TABLE_MOST_COLUMNS + 1u];
	(void)snprintf(
		line, sizeof(line),
		"idle sl=%.3s user=%.32s for=%.5ss ctx=%.5s/%.5s reach=%.5s %.5s%%",
		slot, user, seconds, position, capacity, reach, percent);
	table_line(output, line, columns);
}

static void render_table(struct json_value root, uint32_t columns,
			 struct chat_buffer *output)
{
	const bool wide = columns >= TABLE_WIDE_COLUMNS;
	const uint32_t limit =
		columns < TABLE_MOST_COLUMNS ? columns : TABLE_MOST_COLUMNS;
	const struct json_value global = find_member(root, "global");
	const struct json_value wall = find_member(global, "wall_time");
	char clock[TABLE_CELL_BYTES] = "unknown";
	if (wall.text != NULL && wall.text[wall.begin] == '"' &&
	    wall.end - wall.begin >
		    TABLE_CLOCK_OFFSET + TABLE_CLOCK_BYTES + 1u) {
		memcpy(clock, wall.text + wall.begin + 1u + TABLE_CLOCK_OFFSET,
		       TABLE_CLOCK_BYTES);
		clock[TABLE_CLOCK_BYTES] = '\0';
	}
	static const char *const names[] = { "concurrent_requests",
					     "slots_total",
					     "queued_requests",
					     "queue_depth",
					     "decode_tokens_per_second_10s",
					     "decode_ms_per_token_10s",
					     "prefill_tokens_per_second_10s",
					     "flights_in_air" };
	char figures[sizeof(names) / sizeof(names[0])][TABLE_CELL_BYTES];
	for (size_t index = 0; index < sizeof(names) / sizeof(names[0]);
	     index++) {
		table_number(find_member(global, names[index]), 1,
			     wide ? TABLE_CLOCK_BYTES : TABLE_NUMBER_PRECISION,
			     figures[index]);
	}
	char line[TABLE_MOST_COLUMNS + 1u];
	(void)snprintf(
		line, sizeof(line),
		"[%.8sZ] %s%.8s/%.8s %s%.8s/%.8s %s%.8s/s %.8sms/t %s%.8s/s %s%.8s",
		clock, wide ? "concurrent=" : "r=", figures[TABLE_CONCURRENT],
		figures[TABLE_SLOTS],
		wide ? "queued=" : "q=", figures[TABLE_QUEUED],
		figures[TABLE_QUEUE_DEPTH],
		wide ? "decode=" : "d=", figures[TABLE_DECODE],
		figures[TABLE_MS_PER_TOKEN],
		wide ? "prefill=" : "p=", figures[TABLE_PREFILL],
		wide ? "flights=" : "f=", figures[TABLE_FLIGHTS]);
	table_line(output, line, limit);
	table_cards(find_member(root, "cards"), limit, wide, output);

	const struct json_value requests = find_member(root, "requests");
	const struct json_value session_list = find_member(root, "sessions");
	uint32_t widths[TABLE_COLUMN_COUNT];
	for (size_t index = 0; index < TABLE_COLUMN_COUNT; index++) {
		widths[index] = wide ? table_columns[index].width :
				       table_columns[index].narrow;
	}
	if (!wide) {
		size_t longest = strlen(table_columns[0].compact);
		struct json_value request;
		for (size_t cursor = requests.begin;
		     requests.text != NULL &&
		     requests.text[requests.begin] == '[' &&
		     json_next(requests, &cursor, NULL, &request);) {
			const struct json_value user = find_member(
				find_member(request, "identity"), "user");
			if (user.text != NULL && user.text[user.begin] == '"') {
				const size_t length = user.end - user.begin -
						      JSON_QUOTES_BYTES;
				longest = length > longest ? length : longest;
			}
		}
		for (size_t cursor = session_list.begin;
		     session_list.text != NULL &&
		     session_list.text[session_list.begin] == '[' &&
		     json_next(session_list, &cursor, NULL, &request);) {
			const struct json_value user =
				find_member(request, "user");
			if (user.text != NULL && user.text[user.begin] == '"') {
				const size_t length = user.end - user.begin -
						      JSON_QUOTES_BYTES;
				longest = length > longest ? length : longest;
			}
		}
		widths[0] = longest < TABLE_COMPACT_USER_COLUMNS ?
				    (uint32_t)longest :
				    TABLE_COMPACT_USER_COLUMNS;
		uint32_t spare = table_columns[0].narrow - widths[0];
		/* Share the saved username space among metrics; slot, phase and depth already fit. */
		for (uint32_t pass = 0;
		     pass < table_columns[0].narrow && spare > 0; pass++) {
			for (size_t index = TABLE_FIRST_METRIC_COLUMN;
			     index < TABLE_COLUMN_COUNT && spare > 0; index++) {
				if (table_columns[index].narrow > 1 &&
				    widths[index] <
					    table_columns[index].width) {
					widths[index]++;
					spare--;
				}
			}
		}
	}
	size_t used = 0;
	line[0] = '\0';
	for (size_t index = 0;
	     index < sizeof(table_columns) / sizeof(table_columns[0]);
	     index++) {
		const struct table_column *const column = &table_columns[index];
		table_cell(line, &used,
			   wide ? column->heading : column->compact,
			   widths[index]);
	}
	table_line(output, line, limit);
	struct json_value item;
	for (size_t cursor = session_list.begin;
	     session_list.text != NULL &&
	     session_list.text[session_list.begin] == '[' &&
	     json_next(session_list, &cursor, NULL, &item);) {
		const struct json_value active = find_member(item, "active");
		if (active.text != NULL && active.text[active.begin] == 't') {
			table_request(find_member(item, "current_request"),
				      item, widths, limit, output);
		}
	}
	for (size_t cursor = requests.begin;
	     requests.text != NULL && requests.text[requests.begin] == '[' &&
	     json_next(requests, &cursor, NULL, &item);) {
		const struct json_value slot =
			find_member(find_member(item, "identity"), "slot");
		if (session_list.text == NULL || slot.text == NULL ||
		    slot.text[slot.begin] == 'n') {
			table_request(item, (struct json_value){ 0 }, widths,
				      limit, output);
		}
	}
	for (size_t cursor = session_list.begin;
	     session_list.text != NULL &&
	     session_list.text[session_list.begin] == '[' &&
	     json_next(session_list, &cursor, NULL, &item);) {
		const struct json_value active = find_member(item, "active");
		if (active.text == NULL || active.text[active.begin] != 't') {
			table_idle(item, widths[0], limit, output);
		}
	}
}

MUST_CHECK static result_status render_text(const char *json, size_t byte_count,
					    uint32_t columns, bool is_tty,
					    bool include_dump,
					    struct chat_buffer *output)
{
	struct json_value root;
	if (json == NULL || output == NULL ||
	    columns < LIVE_STATS_LEAST_COLUMNS ||
	    columns > LIVE_STATS_MOST_COLUMNS ||
	    json_read_document(json, byte_count, &root) !=
		    JSON_DOCUMENT_VALID ||
	    root.text[root.begin] != '{') {
		return STATUS_INVALID_ARGUMENT;
	}
	output->count = 0;
	output->sent = 0;
	output->is_short = false;
	if (is_tty) {
		write_text(output, "\033[H\033[J");
	}
	render_table(root, columns, output);
	if (include_dump) {
		struct text_writer writer = { output, columns, 0, 0 };
		text_value(&writer, root, 0);
	}
	return output->is_short ? STATUS_OUT_OF_MEMORY : STATUS_OK;
}

result_status live_stats_text(const char *json, size_t byte_count,
			      uint32_t columns, bool is_tty,
			      struct chat_buffer *output)
{
	return render_text(json, byte_count, columns, is_tty, true, output);
}

result_status live_stats_table_entry(const char *json, size_t byte_count,
				     uint32_t columns, bool is_tty,
				     struct chat_buffer *output)
{
	return render_text(json, byte_count, columns, is_tty, false, output);
}

struct build_identity {
	char *bytes;
	bool found;
};

static int build_note(struct dl_phdr_info *information,
		      size_t information_bytes, void *opaque)
{
	struct build_identity *const result = opaque;
	(void)information_bytes;
	if (information->dlpi_name != NULL &&
	    information->dlpi_name[0] != '\0') {
		return 0;
	}
	static const char hexadecimal[] = "0123456789abcdef";
	for (ElfW(Half) index = 0; index < information->dlpi_phnum; index++) {
		const ElfW(Phdr) *const segment =
			&information->dlpi_phdr[index];
		if (segment->p_type != PT_NOTE) {
			continue;
		}
		const unsigned char *const bytes =
			(const unsigned char *)(information->dlpi_addr +
						segment->p_vaddr);
		const size_t length = (size_t)segment->p_memsz;
		for (size_t offset = 0;
		     offset <= length &&
		     sizeof(ElfW(Nhdr)) <= length - offset;) {
			ElfW(Nhdr) note;
			memcpy(&note, bytes + offset, sizeof(note));
			offset += sizeof(note);
			const size_t name_bytes =
				((size_t)note.n_namesz + NOTE_ALIGNMENT - 1u) &
				~(size_t)(NOTE_ALIGNMENT - 1u);
			const size_t descriptor_bytes =
				((size_t)note.n_descsz + NOTE_ALIGNMENT - 1u) &
				~(size_t)(NOTE_ALIGNMENT - 1u);
			if (name_bytes > length - offset ||
			    descriptor_bytes > length - offset - name_bytes) {
				break;
			}
			if (note.n_type == NT_GNU_BUILD_ID &&
			    note.n_namesz == BUILD_NOTE_NAME_BYTES &&
			    memcmp(bytes + offset, "GNU",
				   BUILD_NOTE_NAME_BYTES) == 0 &&
			    note.n_descsz > 0 &&
			    note.n_descsz <= (LIVE_STATS_BUILD_BYTES - 1u) /
						     HEX_DIGITS_PER_BYTE) {
				for (uint32_t byte = 0; byte < note.n_descsz;
				     byte++) {
					const unsigned char value =
						bytes[offset + name_bytes +
						      byte];
					result->bytes[(size_t)HEX_DIGITS_PER_BYTE *
						      byte] =
						hexadecimal[value / HEX_RADIX];
					result->bytes[(size_t)HEX_DIGITS_PER_BYTE *
							      byte +
						      1u] =
						hexadecimal[value % HEX_RADIX];
				}
				result->bytes[(size_t)HEX_DIGITS_PER_BYTE *
					      note.n_descsz] = '\0';
				result->found = true;
				return 1;
			}
			offset += name_bytes + descriptor_bytes;
		}
	}
	return 0;
}

result_status live_stats_build_id(char output[LIVE_STATS_BUILD_BYTES])
{
	if (output == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct build_identity identity = { output, false };
	const int result = dl_iterate_phdr(build_note, &identity);
	if (result == 0 && !identity.found) {
		(void)snprintf(output, LIVE_STATS_BUILD_BYTES, "%s",
			       "unavailable");
	}
	return STATUS_OK;
}
