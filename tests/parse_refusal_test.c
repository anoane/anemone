/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Every refusal of the command line names the refused option and why, in one line before the usage: a case for each
 * kind of refusal (no command, the option scan's four, the parser's own checks and each refusal table, capture's each
 * refusal and option value; generate takes no capture option), each through the real parser in a fresh child, so
 * every case starts from commands.c's defaults. Then an explicit BATCH on the
 * TP4_EXL3 base (the documented EXL3 serve without its --pipeline_steps=1): the axes' inserted --pipeline_steps=2
 * yields to it, the parse runs one pipeline step and says so, and an explicit --pipeline_steps=2 is still refused. */
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "startup_axes.h"
#include "commands.h"

#define TEST_WORDS 64u
#define TEST_LOG_BYTES 16384u

struct refusal_case {
	const char *words[TEST_WORDS];
	const char *message; /* the line the child's stderr must hold */
	bool parse_expected;
	bool axes; /* expanded as main.c does, the axis lines logged */
};

static unsigned cases_run;

/* The documented EXL3 BATCH serve (docs/SERVER_PROFILE_COMMANDS.json's exl3_balanced) without --pipeline_steps=1,
 * then depth words and the two data files. */
#define EXL3_BATCH_SERVE                                                       \
	"serve", "--placement=tp4_exl3", "--numerics=relaxed",                 \
		"--tp_wire_bf16=1", "--cmp_order=reference",                   \
		"--linear_order=1", "--linear_profile=0", "--pro_emulation=0", \
		"--exl3_expert_parts=4", "--exl3_tc_moe=1", "--tp_head_mma=1", \
		"--tp_trunk_mma=1", "--tp_ordered_wide=1",                     \
		"--tp_head_shards=1", "--tp_head_stream=1",                    \
		"--tp_draft_shards=1", "--rows=8", "--full_steps=1",           \
		"--prefill_cone=1", "--chunks_in_flight=1", "--step_graphs=1", \
		"--speculative=2", "--draft=dspark", "--draft_block=5",        \
		"--draft_positions=5", "--card_memory_kept_free_mib=0",        \
		"--wide_rows=1024", "--batch_rows=8,8,12,12,16,24,32,48",      \
		"--batch_verifiers=1,2,2,4,4,4,8,8", "--batch_wide_head=1",    \
		"--batch_draft_rows=1", "--tp_draft_layers=1",                 \
		"--draft_learned_acceptance=1",                                \
		"--draft_session_acceptance=0", "--draft_base_us=14300",       \
		"--draft_tail_us=0", "--draft_row_us=690",                     \
		"--exl3_tc_product_blocks=0", "--tp_decode_round_once=0",      \
		"--reasoning_effort=0", "--context_tokens=65536",              \
		"--generated_tokens=65536", "--session_slots=8"
/* capture's three required options */
#define CAPTURE_WORDS                                                     \
	"capture", "--capture_golden=golden", "--capture_cases=keys.txt", \
		"--capture_out=out"
#define CAPTURE_GREEDY "a capture draws greedily: the witnesses are T=0 draws\n"
#define CAPTURE_VALUE ": a value out of its range, or given twice\n"
#define YIELD_LINE \
	" --pipeline_steps (yields to the user's --batch_rows: BATCH runs one pipeline step)"

static bool child_case(const struct refusal_case *test)
{
	char program[] = "anemone";
	char *arguments[TEST_WORDS + 1u] = { program };
	char *copies[TEST_WORDS] = { 0 };
	int count = 1;
	bool passed = true;
	for (size_t index = 0; index < TEST_WORDS && test->words[index] != NULL;
	     index++) {
		copies[index] = strdup(test->words[index]);
		passed = passed && copies[index] != NULL;
		arguments[count++] = copies[index];
	}
	FILE *const diagnostic = tmpfile();
	const int saved = dup(STDERR_FILENO);
	passed = passed && diagnostic != NULL && saved >= 0 &&
		 dup2(fileno(diagnostic), STDERR_FILENO) == STDERR_FILENO;
	struct axis_arguments_record expanded = { 0 };
	const bool parsed = passed &&
			    (!test->axes ?
				     parse_arguments(count, arguments) :
				     axis_expand(count, arguments, &expanded) &&
					     parse_arguments(expanded.count,
							     expanded.values));
	if (test->axes)
		axis_log(&expanded, stderr);
	/* the explicit BATCH runs one pipeline step, never the base's two */
	passed = passed && parsed == test->parse_expected &&
		 (!parsed || !test->axes ||
		  (pipeline_steps == 1u && batch_rows[0] != 0));
	(void)fflush(stderr);
	passed = saved >= 0 && dup2(saved, STDERR_FILENO) == STDERR_FILENO &&
		 passed;
	char text[TEST_LOG_BYTES] = "";
	if (diagnostic != NULL) {
		passed = fseek(diagnostic, 0L, SEEK_SET) == 0 && passed;
		clearerr(diagnostic);
		text[fread(text, 1u, sizeof(text) - 1u, diagnostic)] = '\0';
		(void)fclose(diagnostic);
	}
	passed = passed && strstr(text, test->message) != NULL &&
		 (!test->axes ||
		  strstr(text, "axes set --pipeline_steps=2") == NULL);
	if (!passed)
		(void)fprintf(stderr, "stderr was:\n%s", text);
	axis_arguments_free(&expanded);
	for (size_t index = 0; index < TEST_WORDS; index++)
		free(copies[index]);
	return passed;
}

int main(void)
{
	static const struct refusal_case tests[] = {
		{ { NULL }, "refused: the command: none given\n", false, false },
		{ { "frobnicate", "tables", "pack.anemone" },
		  "refused: frobnicate: not a command\n",
		  false,
		  false },
		{ { "serve", "--no_such_option=1", "tables", "pack.anemone" },
		  "refused: --no_such_option=1: no such option",
		  false,
		  false },
		{ { "load", "--metrics=1", "pack.anemone" },
		  "refused: --metrics=1: not an option of this command\n",
		  false,
		  false },
		{ { "serve", "--seed=1", "--seed=2", "tables", "pack.anemone" },
		  "refused: --seed=2: given twice\n",
		  false,
		  false },
		{ { "serve", "--port=0", "tables", "pack.anemone" },
		  "refused: --port=0: takes a number 1 to 65535\n",
		  false,
		  false },
		{ { "serve", "--card_order=1,1", "tables", "pack.anemone" },
		  "refused: --card_order=1,1: takes a list of distinct numbers, at most 16 values, each 0 to 15\n",
		  false,
		  false },
		{ { "serve", "--batch_rows=8,8,8,8,8,8,8,8,8", "tables",
		    "pack.anemone" },
		  "refused: --batch_rows=8,8,8,8,8,8,8,8,8: takes a list of numbers, at most 8 values, each 1 to 64\n",
		  false,
		  false },
		{ { "serve", "--session_slots=max", "--session_slots=max",
		    "tables", "pack.anemone" },
		  "refused: --session_slots=max: given twice\n",
		  false,
		  false },
		{ { "serve", "--engram_ram_percent=auto",
		    "--engram_ram_percent=auto", "tables", "pack.anemone" },
		  "refused: --engram_ram_percent=auto: given twice\n",
		  false,
		  false },
		{ { "serve", "--bind_address=::1", "--bind_address=::1",
		    "tables", "pack.anemone" },
		  "refused: --bind_address=::1: a value, a command or a repeat this option refuses",
		  false,
		  false },
		{ { "serve", "--batch_step_us=0,5", "tables", "pack.anemone" },
		  "refused: --batch_step_us=0,5: a single 0 turns the planning off",
		  false,
		  false },
		{ { "serve", "--draft=lookup", "--speculative=2", "tables",
		    "pack.anemone" },
		  "refused: --speculative: conflicts with the --draft given\n",
		  false,
		  false },
		{ { "generate", "--speculative=2", "--draft_block=5",
		    "--draft_positions=6", "messages", "tables",
		    "pack.anemone" },
		  "refused: --draft_positions: above --draft_block\n",
		  false,
		  false },
		{ { "serve", "--engram_ram_percent=auto",
		    "--engram_ram_percent=50", "tables", "pack.anemone" },
		  "refused: --engram_ram_percent: auto or a percent, not both\n",
		  false,
		  false },
		{ { "serve", "--session_slots=max", "--session_slots=2",
		    "tables", "pack.anemone" },
		  "refused: --session_slots: max or a count, not both\n",
		  false,
		  false },
		{ { "generate", "--ignore_eos=1", "--speculative=1", "messages",
		    "tables", "pack.anemone" },
		  "refused: --ignore_eos: non-speculative THIN only",
		  false,
		  false },
		{ { "serve", "--prefill_order=3", "tables", "pack.anemone" },
		  "refused: --prefill_order: concurrent prompt rows need",
		  false,
		  false },
		{ { "serve", "--engram_ram_percent=0", "--engram_hot_set=hot",
		    "tables", "pack.anemone" },
		  "refused: --engram_hot_set: taken only with its recorded --engram_hot_sha256",
		  false,
		  false },
		/* the first refusal table: its first row, a middle one and its last */
		{ { "serve", "--queue_depth=2", "--user_queue_depth=3",
		    "tables", "pack.anemone" },
		  "refused: --user_queue_depth: above --queue_depth\n",
		  false,
		  false },
		{ { "serve", "--live_stats_format=xml", "tables",
		    "pack.anemone" },
		  "refused: --live_stats_format: takes text, json or table\n",
		  false,
		  false },
		{ { "serve", "--tp_core_heads=3", "tables", "pack.anemone" },
		  "refused: --tp_core_heads: a multiple of the attention core's tile of heads\n",
		  false,
		  false },
		{ { "serve", "--batch_wide_head=1", "tables", "pack.anemone" },
		  "refused: --batch_draft_rows, --batch_wide_head, --batch_context_slabs, --batch_step_us and "
		  "--batch_verifiers: BATCH's own: give --batch_rows\n",
		  false,
		  false },
		{ { "serve", "--placement=tp2_pp2_pro", "tables",
		    "pack.anemone" },
		  "refused: --numerics: exact needs the reference CMP order",
		  false,
		  false },
		{ { "serve", "--placement=tp4_pro", "--layout=vision: card 0",
		    "tables", "pack.anemone" },
		  "refused: --layout: PP5's alone\n",
		  false,
		  false },
		/* the tensor table: its first row and its last */
		{ { "serve", "--placement=tp4_exl3", "--exl3_expert_parts=5",
		    "tables", "pack.exl3map" },
		  "refused: --exl3_expert_parts: 4 (the reference partition) or 8\n",
		  false,
		  false },
		{ { "serve", "--placement=tp4_exl3", "--tp_draft_layers=1",
		    "tables", "pack.exl3map" },
		  "refused: --tp_draft_layers: needs TP4_EXL3 with the BF16 wire",
		  false,
		  false },
		{ { "serve", "--session_fork=1", "--cache_scope=1", "tables",
		    "pack.anemone" },
		  "refused: --session_fork: needs --cache_scope=0\n",
		  false,
		  false },
		{ { "serve", "--prefill_prefix_stable=1", "tables",
		    "pack.anemone" },
		  "refused: --prefill_prefix_stable: the TP4_EXL3 BF16 column-owner THIN recipe alone",
		  false,
		  false },
		{ { "serve", "--prefix_reuse_exact=1", "tables",
		    "pack.anemone" },
		  "refused: --prefix_reuse_exact: needs TP4_EXL3 column owners",
		  false,
		  false },
		{ { "generate", "--model_dir=folder" },
		  "refused: --model_dir: generate still takes MESSAGES",
		  false,
		  false },
		/* the file table: no pack, then one too many for the token ids */
		{ { "serve", "tables" },
		  "refused: the pack: none given (or --model_dir)\n",
		  false,
		  false },
		{ { "generate", "--prompt_token_ids=ids.json", "tables",
		    "first.anemone", "second.anemone" },
		  "refused: --prompt_token_ids: takes one pack\n",
		  false,
		  false },
		{ { "generate", "--draft_calibration=/nonexistent/calibration",
		    "messages", "tables", "pack.anemone" },
		  "refused: --draft_calibration: the file does not read",
		  false,
		  false },
		/* capture: the options it needs, what each case decides, what is out of its scope, its own values */
		{ { "capture", "--capture_golden=golden", "--capture_out=out",
		    "tables", "pack.anemone" },
		  "refused: capture: needs --capture_golden, --capture_cases and --capture_out\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--temperature=0.5", "tables",
		    "pack.anemone" },
		  "refused: --temperature: " CAPTURE_GREEDY,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--top_p=0.9", "tables", "pack.anemone" },
		  "refused: --top_p: " CAPTURE_GREEDY,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--generated_tokens=8", "tables",
		    "pack.anemone" },
		  "refused: --generated_tokens: each case's continuation decides\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--ignore_eos=1", "tables", "pack.anemone" },
		  "refused: --ignore_eos: a forced continuation ends at its limit alone\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--prompt_token_ids=ids.json", "tables",
		    "pack.anemone" },
		  "refused: --prompt_token_ids: each case's prompt_ids.json decides\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--reasoning_effort=1", "tables",
		    "pack.anemone" },
		  "refused: --reasoning_effort: each case's prompt decides\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--batch_rows=8", "tables", "pack.anemone" },
		  "refused: --batch_rows: BATCH heads are out of capture's scope\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--original_batch=1", "tables",
		    "pack.anemone" },
		  "refused: --original_batch=1: not an option of this command\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--original_head_offload=1", "tables",
		    "pack.anemone" },
		  "refused: --original_head_offload: the offloaded head is out of capture's scope\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--dump_draft_ring=ring.bin", "tables",
		    "pack.anemone" },
		  "refused: --dump_draft_ring: a diagnostic, out of capture's scope\n",
		  false,
		  false },
		{ { CAPTURE_WORDS, "--capture_rows=9", "tables",
		    "pack.anemone" },
		  "refused: --capture_rows=9" CAPTURE_VALUE,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--capture_rows=0", "tables",
		    "pack.anemone" },
		  "refused: --capture_rows=0" CAPTURE_VALUE,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--capture_rows=2", "--capture_rows=3",
		    "tables", "pack.anemone" },
		  "refused: --capture_rows=3" CAPTURE_VALUE,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--capture_positions=0", "tables",
		    "pack.anemone" },
		  "refused: --capture_positions=0" CAPTURE_VALUE,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--capture_free=2", "tables",
		    "pack.anemone" },
		  "refused: --capture_free=2" CAPTURE_VALUE,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--capture_free=1", "--capture_free=1",
		    "tables", "pack.anemone" },
		  "refused: --capture_free=1" CAPTURE_VALUE,
		  false,
		  false },
		{ { CAPTURE_WORDS, "--capture_out=other", "tables",
		    "pack.anemone" },
		  "refused: --capture_out=other" CAPTURE_VALUE,
		  false,
		  false },
		{ { "capture", "--capture_golden=", "--capture_cases=keys.txt",
		    "--capture_out=out", "tables", "pack.anemone" },
		  "refused: --capture_golden=" CAPTURE_VALUE,
		  false,
		  false },
		{ { "generate", "--capture_rows=2", "messages", "tables",
		    "pack.anemone" },
		  "refused: --capture_rows=2: a value, a command or a repeat this option refuses",
		  false,
		  false },
		/* the explicit BATCH on the TP4_EXL3 base: the inserted depth yields, an explicit one is refused */
		{ { EXL3_BATCH_SERVE, "tables", "pack.exl3map" },
		  YIELD_LINE,
		  true,
		  true },
		{ { EXL3_BATCH_SERVE, "--pipeline_steps=2", "tables",
		    "pack.exl3map" },
		  "refused: --batch_rows: BATCH runs on TP4_EXL3 (or with --original_batch=1) at "
		  "--pipeline_steps=1\n",
		  false,
		  true }
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(tests) / sizeof(tests[0]);
	     index++) {
		cases_run++;
		const pid_t child = fork();
		if (child == 0)
			_exit(child_case(&tests[index]) ? 0 : 1);
		int status = 0;
		const bool held = child > 0 &&
				  waitpid(child, &status, 0) == child &&
				  WIFEXITED(status) && WEXITSTATUS(status) == 0;
		if (!held)
			(void)fprintf(stderr,
				      "FAILED: parse refusal case %zu: %s\n",
				      index, tests[index].message);
		passed = held && passed;
	}
	(void)printf("parse refusals: %s (%u cases)\n",
		     passed ? "PASS" : "FAIL", cases_run);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
