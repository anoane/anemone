/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU test of the engine patch's bb_exl3_pass_prepare (exl3_v3_b): exact output-shard passes with product_tiles
 * 0 (retained), 4 and 8 (thin), at 6 and 64 / 1024 rows (wide): modules, names, grids, same records and axes. */
#include <stdio.h>
#include <string.h>

#include "anemone/exl3_tp.h"

#define MOST_KERNELS 64
#define FILE_BYTES 64
#define NAME_BYTES 96
/* the cases: rows 6 (thin) and 64, 1024 (wide); product tiles 0 (retained), 4 and 8 */
#define THIN_ROWS_CASE 6
#define WIDE_ROWS_CASE 64
#define WIDEST_ROWS_CASE 1024
#define ROW_CASES 3
#define FOUR_TILES 4
#define EIGHT_TILES 8
#define TILE_CASES 3
/* the shard plan: a 5120 x 2304 product of 3 bits a weight over 4 ranks */
#define OUTER_SIZE 5120
#define INNER_SIZE 2304
#define BITS 3
#define RANKS 4
/* the record: 6 experts a row; up to THIN_ROWS_MOST rows a job a pair, else jobs of JOB_PAIRS pairs, each of the 384
 * experts' last one partial (JOB_PARTIAL more pairs at most) */
#define EXPERTS_A_ROW 6u
#define THIN_ROWS_MOST 8
#define EXPERTS 384u
#define JOB_PAIRS 8u
#define JOB_PARTIAL 7u
#define BITS_PER_WEIGHT 3u
#define ACTIVATION_LIMIT 10.0f
/* the bytes a pair's products take for each of the 5120 columns */
#define PAIR_COLUMN_BYTES 16u
#define PAIR_COLUMNS 5120u
#define JOBS_OFFSET (1ull << 20)
#define INPUT_OFFSET (1ull << 24)
#define OPERANDS_OFFSET ((1ull << 24) + (1ull << 26))
#define ARENA_BYTES (1ull << 36)
#define WEIGHT_OFFSET (1ull << 32)
#define WEIGHT_BYTES (1ull << 30)
#define GRID_Z 2 /* a launch grid's third dimension */
/* the expected grid: a product pass's blocks of PRODUCT_COLUMNS columns, another pass's of PASS_COLUMNS, divided by
 * the tiles (WIDE_TILES when wide); the input and finish passes in groups of INPUT_GROUP, rounded up */
#define PRODUCT_COLUMNS 16u
#define PASS_COLUMNS 128u
#define WIDE_TILES 4u
#define INPUT_GROUP 4u
#define INPUT_ROUNDING 3u
static char files[MOST_KERNELS][FILE_BYTES], names[MOST_KERNELS][NAME_BYTES];
static uint32_t kernel_count;
static int failures;

MUST_CHECK static result_status stub_prepare(struct device_card *card,
					     const char *file, const char *name,
					     uint32_t argument_bytes,
					     uint32_t *kernel_out)
{
    (void)card;
    (void)argument_bytes;
    for (uint32_t index = 0; index < kernel_count; index++)
        if (strcmp(files[index], file) == 0 && strcmp(names[index], name) == 0) {
            *kernel_out = index + 1u;
	    return STATUS_OK;
	}
    (void)snprintf(files[kernel_count], sizeof(files[0]), "%s", file);
    (void)snprintf(names[kernel_count], sizeof(names[0]), "%s", name);
    *kernel_out = ++kernel_count;
    return STATUS_OK;
}

int main(void)
{
	struct backend_operations backend;
	memset(&backend, 0, sizeof(backend));
	backend.kernel_prepare = stub_prepare;
	struct device_card *card = (struct device_card *)&backend;
	const uint32_t rows_cases[] = { THIN_ROWS_CASE, WIDE_ROWS_CASE,
					WIDEST_ROWS_CASE },
		       tiles_cases[] = { 0, FOUR_TILES, EIGHT_TILES };
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		struct bb_exl3_output_shard_plan plan;
		if (bb_exl3_output_shard_prepare(
			    OUTER_SIZE, INNER_SIZE, BITS, RANKS, rank,
			    BB_EXL3_REFERENCE_ORDER, &plan) != STATUS_OK) {
			printf("FAIL plan rank %u\n", rank);
			return 1;
		}
		for (uint32_t r = 0; r < ROW_CASES; r++)
			for (uint32_t t = 0; t < TILE_CASES; t++) {
				const uint32_t rows = rows_cases[r],
					       tiles = tiles_cases[t];
				printf("--- rank %u rows %u tiles %u (gate/up out %u, down out %u)\n",
				       rank, rows, tiles,
				       plan.gate_up.output_count,
				       plan.down.output_count);
				for (uint32_t pass = 0; pass < BB_EXL3_PASSES;
				     pass++) {
					const bool down = pass >=
							  BB_EXL3_DOWN_PASS;
					struct exl3_expert_record record;
					memset(&record, 0, sizeof(record));
					record.pair_count =
						rows * EXPERTS_A_ROW;
					record.pairs_per_row = EXPERTS_A_ROW;
					record.job_count =
						rows <= THIN_ROWS_MOST ?
							rows * EXPERTS_A_ROW :
							(rows * EXPERTS_A_ROW +
							 JOB_PARTIAL *
								 EXPERTS) /
								JOB_PAIRS;
					record.bits_per_weight =
						BITS_PER_WEIGHT;
					record.activation_limit =
						ACTIVATION_LIMIT;
					record.input_size =
						down ? plan.down.output_count :
						       plan.gate_up.input_count;
					record.inner_size =
						down ? plan.down.input_count :
						       plan.gate_up.output_count;
					const uint64_t big =
						(uint64_t)record.pair_count *
						PAIR_COLUMNS *
						PAIR_COLUMN_BYTES;
					record.jobs_offset = JOBS_OFFSET;
					record.input_offset = INPUT_OFFSET;
					record.operands_offset =
						OPERANDS_OFFSET;
					record.products_offset =
						record.operands_offset + big;
					record.output_offset =
						pass == BB_EXL3_FINISH_PASS ?
							record.products_offset :
							0;
					record.hidden_offset =
						record.products_offset + big;
					const struct bb_exl3_pass_description
						description = {
							.plan = &plan,
							.record = record,
							.arena_bytes =
								ARENA_BYTES,
							.weight_offset =
								WEIGHT_OFFSET,
							.weight_bytes =
								WEIGHT_BYTES,
							.pass = (enum bb_exl3_pass)
								pass,
							.product_tiles = tiles
						};
					struct kernel_launch launch;
					const result_status status =
						bb_exl3_pass_prepare(
							&description, &backend,
							card, &launch);
					if (status != STATUS_OK) {
						printf("FAIL pass %u status %d\n",
						       pass, (int)status);
						failures++;
						continue;
					}
					const uint32_t kernel =
						launch.kernel - 1u;
					printf("pass %u: %s/%s grid %u x %u x %u block %u\n",
					       pass, files[kernel],
					       names[kernel], launch.grid[0],
					       launch.grid[1],
					       launch.grid[GRID_Z],
					       launch.block[0]);
					const bool product =
						pass == BB_EXL3_GATE_UP_PASS ||
						pass == BB_EXL3_DOWN_PASS;
					const bool wide =
						rows > THIN_ROWS_MOST &&
						tiles != 0 && product;
					const uint32_t width =
						pass == BB_EXL3_GATE_UP_PASS ||
								pass == BB_EXL3_ACTIVATE_PASS ?
							record.inner_size :
							record.input_size;
					uint32_t expected_x =
						product ?
							width / PRODUCT_COLUMNS :
							width / PASS_COLUMNS;
					if (product && tiles != 0)
						expected_x /=
							wide ? WIDE_TILES :
							       tiles;
					if (tiles != 0 &&
					    (pass == BB_EXL3_INPUT_PASS ||
					     pass == BB_EXL3_FINISH_PASS))
						expected_x = (expected_x +
							      INPUT_ROUNDING) /
							     INPUT_GROUP;
					if (launch.grid[0] != expected_x ||
					    launch.grid[1] !=
						    record.job_count) {
						printf("FAIL grid %u expected %u\n",
						       launch.grid[0],
						       expected_x);
						failures++;
					}
					if (wide && strstr(names[kernel],
							   "_t4_g4p") == NULL) {
						printf("FAIL wide name\n");
						failures++;
					}
				}
			}
	}
    printf("result=%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
