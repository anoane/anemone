/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU test of the engine patch's bb_exl3_pass_prepare (exl3_v3_claude): exact output-shard passes with product_tiles
 * 0 (retained), 4 and 8 (thin), at 6 and 64 / 1024 rows (wide): modules, names, grids, same records and axes. */
#include <stdio.h>
#include <string.h>

#include "anemone/exl3_tp.h"

#define MOST_KERNELS 64
static char files[MOST_KERNELS][64], names[MOST_KERNELS][96];
static uint32_t kernel_count;
static int failures;

static result_status stub_prepare(struct device_card *card, const char *file,
				  const char *name, uint32_t argument_bytes,
				  uint32_t *kernel_out)
{
    (void)card;
    (void)argument_bytes;
    for (uint32_t index = 0; index < kernel_count; index++)
        if (strcmp(files[index], file) == 0 && strcmp(names[index], name) == 0) {
            *kernel_out = index + 1u;
	    return STATUS_OK;
	}
    snprintf(files[kernel_count], sizeof(files[0]), "%s", file);
    snprintf(names[kernel_count], sizeof(names[0]), "%s", name);
    *kernel_out = ++kernel_count;
    return STATUS_OK;
}

int main(void)
{
	struct backend_operations backend;
	memset(&backend, 0, sizeof(backend));
	backend.kernel_prepare = stub_prepare;
	struct device_card *card = (struct device_card *)&backend;
	const uint32_t rows_cases[] = { 6, 64, 1024 },
		       tiles_cases[] = { 0, 4, 8 };
	for (uint32_t rank = 0; rank < 4; rank++) {
		struct bb_exl3_output_shard_plan plan;
		if (bb_exl3_output_shard_prepare(5120, 2304, 3, 4, rank,
						 BB_EXL3_REFERENCE_ORDER,
						 &plan) != STATUS_OK) {
			printf("FAIL plan rank %u\n", rank);
			return 1;
		}
		for (uint32_t r = 0; r < 3; r++)
			for (uint32_t t = 0; t < 3; t++) {
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
					record.pair_count = rows * 6u;
					record.pairs_per_row = 6u;
					record.job_count = rows <= 8 ?
								   rows * 6u :
								   (rows * 6u +
								    7u * 384u) /
									   8u;
					record.bits_per_weight = 3u;
					record.activation_limit = 10.0f;
					record.input_size =
						down ? plan.down.output_count :
						       plan.gate_up.input_count;
					record.inner_size =
						down ? plan.down.input_count :
						       plan.gate_up.output_count;
					const uint64_t big =
						(uint64_t)record.pair_count *
						5120u * 16u;
					record.jobs_offset = 1ull << 20;
					record.input_offset = 1ull << 24;
					record.operands_offset =
						(1ull << 24) + (1ull << 26);
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
							.arena_bytes = 1ull
								       << 36,
							.weight_offset = 1ull
									 << 32,
							.weight_bytes = 1ull
									<< 30,
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
					       launch.grid[1], launch.grid[2],
					       launch.block[0]);
					const bool product =
						pass == BB_EXL3_GATE_UP_PASS ||
						pass == BB_EXL3_DOWN_PASS;
					const bool wide = rows > 8 &&
							  tiles != 0 && product;
					const uint32_t width =
						pass == BB_EXL3_GATE_UP_PASS ||
								pass == BB_EXL3_ACTIVATE_PASS ?
							record.inner_size :
							record.input_size;
					uint32_t expected_x =
						product ? width / 16u :
							  width / 128u;
					if (product && tiles != 0)
						expected_x /= wide ? 4u : tiles;
					if (tiles != 0 &&
					    (pass == BB_EXL3_INPUT_PASS ||
					     pass == BB_EXL3_FINISH_PASS))
						expected_x =
							(expected_x + 3u) / 4u;
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
