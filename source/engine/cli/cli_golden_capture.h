/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_golden_capture.h: anemone capture, generate's whole path (parse, axes, load, plan, programs, driver, engine)
 * whose front end teacher-forces each case of a golden dataset (engine_submit_forced) and keeps the head's F32 rows the
 * sampler read, in the scorer's candidate layout (tools/golden_score.md). SOURCE ONLY.
 *
 * Contract:
 *   - Inputs, all read before the first command: the keys file (one set/id a line, run order), and per case
 *     ROOT/cases/<set>/<id>/ prompt_ids.json, token_ids.json (canonical or not, parsed as --prompt_token_ids) and
 *     metadata.json: schema_version 1, temperature 0, mode autoregressive_greedy, generated_positions the
 *     continuation's count, and the prompt and continuation hashes the SHA-256 of the IDs' canonical compact JSON
 *     (json.dumps(ids, separators=(',', ':'))). At run: its vocabulary_size the model's, and the prompt and the
 *     positions but the last within the context. A key with "..", a slash other than one between set and id, or
 *     any byte but [A-Za-z0-9_.-] is refused, and a key listed twice.
 *   - OUT, new or a capture of the same plan (CAPTURE_MANIFEST.json's plan_sha256: the executable's, the argv's, the
 *     keys file's and ROOT/MANIFEST.json's SHA-256, null when it has none; the manifest also holds those digests, the
 *     cases, the positions kept, the bytes of their rows, the argv's words and the capture object of metadata.json;
 *     a digest that cannot be made is a refusal naming it, before any text is built): OUT/cases/<set>/<id>/ holds
 *     metadata.json (alignment teacher_forced, the three hashes; capture: the executable, version, argv digest,
 *     placement, weights, map name, pack header digest, the EXL3 map's descriptor digest, rows, context, truncated,
 *     logits digest, witness count),
 *     prompt_ids.json and token_ids.json (canonical: their SHA-256 is the alignment hash; with --capture_positions
 *     the first that many tokens, truncated), conditioning_ids.json (the driver's committed history read at the
 *     last landing: prompt + tokens[:-1] or the case fails), logits.f32 (little-endian F32 [P][V], row k the head
 *     after prompt + tokens[:k]), native_drawn_ids.json (the sampler's draws) and witness.json (per position: its
 *     row's position and input token, the step's rows, the row's place, the draw and its logit's bits); with
 *     --capture_free=1 also free_token_ids.json and free_logits.json (an ordinary T=0 command of the prompt, limit
 *     P: its tokens and their events' logit bits). Rows stream into <id>.partial/ (logits.f32 opened O_EXCL) and the
 *     case ends with fsync, the JSON files, a rename to <id>/, a directory fsync and one line of cases.jsonl.
 *   - Resume: a case with its folder and its cases.jsonl line is skipped; a .partial folder, or a folder without its
 *     line, loses the capture's own files and is run again. Refused: an OUT holding anything else, another plan, a
 *     cases, set or case folder that is a link (nothing is removed through one), and free space below the remaining
 *     rows' bytes and GOLDEN_CAPTURE_FLOOR_BYTES.
 *   - Each landed row is a witness: its position and input token, every logit finite, the draw the row's lowest-ID
 *     argmax, the draw's logit the row's at the draw bit for bit. A failure fails the case and the run, naming it.
 *   - A refusal or a case's failure is one line on stderr ("capture refused: WHY", "capture: case KEY failed (STATUS):
 *     position K: WHY"), in every build: a product output, as commands.c's.
 */
#ifndef CLI_GOLDEN_CAPTURE_H
#define CLI_GOLDEN_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>

#include "engine/runtime/scheduler/front_end_seam.h"
#include "common/status.h"

#define GOLDEN_CAPTURE_FLOOR_BYTES (1ull << 30u)

struct step_driver;
struct execution_program;
struct sampler_logits_part;

enum golden_capture_given {
	CAPTURE_GOLDEN_GIVEN = 1,
	CAPTURE_CASES_GIVEN = 2,
	CAPTURE_OUT_GIVEN = 4,
	CAPTURE_REQUIRED = 7,
	CAPTURE_ROWS_GIVEN = 8,
	CAPTURE_POSITIONS_GIVEN = 16,
	CAPTURE_FREE_GIVEN = 32
};
/* The options as parse took them: NULL or 0 until given. rows 0: speculative_depth + 1 (1 without speculation). */
struct golden_capture_options {
	const char *golden, *cases, *out;
	uint32_t rows, positions, free_run, given;
};
extern struct golden_capture_options golden_capture_options;

/* Whether argument is one of capture's options, taken; is_valid_out false for a value out of its range or an option
 * given twice. */
bool golden_capture_option_taken(const char *argument, bool *is_valid_out);

/* Where a capture reads the cards and OUT's disk: by default (NULL) the driver's arena (step_arena_read) and committed
 * tokens (step_history_token, slot 0), statvfs, and the landed rows as the engine gives them (rows_seen NULL); its
 * host test reads through its own and changes a row the witnesses see. */
struct golden_capture_io {
	MUST_CHECK result_status (*arena_read)(void *context, uint32_t card,
					       uint64_t offset,
					       uint64_t byte_count,
					       void *bytes_out);
	uint32_t (*history_token)(void *context, uint32_t position);
	bool (*free_bytes)(void *context, const char *path,
			   uint64_t *bytes_out); /* false: unknown */
	const struct step_row_record *(*rows_seen)(
		void *context, const struct step_row_record *rows,
		uint32_t step_row_count);
	void *context;
};

struct golden_capture_run {
	struct inference_engine
		*engine; /* one slot, speculative_depth as described */
	struct step_driver *driver;
	const struct execution_program *
		program; /* a thin program of the driver's: the samplers' logits parts */
	const struct golden_capture_io *io;
	int argument_count;
	char *const *arguments; /* the expanded argv: digested and recorded */
	const char *version, *placement, *weights,
		*map; /* recorded; map a file name, no path */
	const void *pack_header;
	uint64_t pack_header_bytes;
	const char *
		executable; /* the file digested as the executable: NULL, /proc/self/exe */
	/* an EXL3 map as the run names it, else NULL: descriptor_sha256 is its sidecar's (exl3_descriptor.h: the real
	 * path's EXL3_DESCRIPTOR_SUFFIX), null without one or without a map */
	const char *map_path;
	/* NULL: sampler_logits_parts of program; else these (the host test's parts) */
	const struct sampler_logits_part *parts;
	uint32_t part_count;
};
/* Captures every case of golden_capture_options; one line a case on stdout; a refusal or a failure one error line. */
MUST_CHECK result_status
golden_capture_run(const struct golden_capture_run *run);

#endif
