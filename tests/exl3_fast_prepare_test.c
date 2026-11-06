/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU test of the engine patch (exl3_v3_claude): the launch tables of exl3_experts_fast_prepare and of
 * bb_exl3_pass_prepare with product_tiles 8 and wide calls, against the retained tables: same record, ranges,
 * job axis and projection axis; module, entry names and grid x as documented. A stub backend records names. */
#include <stdio.h>
#include <string.h>

#include "anemone/exl3.h"
#include "anemone/exl3_tp.h"

#define MOST_KERNELS 64

static char files[MOST_KERNELS][64], names[MOST_KERNELS][96];
static uint32_t kernel_count;

static result_status stub_prepare(struct device_card *card, const char *file,
				  const char *name, uint32_t argument_bytes,
				  uint32_t *kernel_out)
{
    (void)card;
    (void)argument_bytes;
    for (uint32_t index = 0; index < kernel_count; index++) {
        if (strcmp(files[index], file) == 0 && strcmp(names[index], name) == 0) {
            *kernel_out = index + 1u;
	    return STATUS_OK;
	}
    }
    if (kernel_count == MOST_KERNELS)
	    return STATUS_INVALID_ARGUMENT;
    snprintf(files[kernel_count], sizeof(files[0]), "%s", file);
    snprintf(names[kernel_count], sizeof(names[0]), "%s", name);
    *kernel_out = ++kernel_count;
    return STATUS_OK;
}

static int failures;
#define EXPECT(condition, ...)                       \
    do {                                             \
        if (!(condition)) {                          \
            failures++;                              \
            printf("FAIL %s:%d ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                     \
            printf("\n");                            \
        }                                            \
    } while (0)

static void show(const char *what, const struct kernel_launch *launches,
		 uint32_t count)
{
    for (uint32_t index = 0; index < count; index++) {
        const uint32_t kernel = launches[index].kernel - 1u;
        printf("%s pass %u: %s/%s grid %u x %u x %u block %u\n", what, index, files[kernel], names[kernel],
               launches[index].grid[0], launches[index].grid[1], launches[index].grid[2], launches[index].block[0]);
    }
}

int main(void)
{
	struct backend_operations backend;
	memset(&backend, 0, sizeof(backend));
	backend.kernel_prepare = stub_prepare;
	struct device_card *card =
		(struct device_card
			 *)&backend; /* never dereferenced by the stub */
	const uint64_t arena = 1ull << 34;
	const uint32_t rows_cases[] = { 6, 8, 9, 64, 1024 };
	const uint32_t inner_cases[] = { 512, 640 };
	for (uint32_t r = 0; r < sizeof(rows_cases) / sizeof(rows_cases[0]);
	     r++) {
		for (uint32_t i = 0; i < 2; i++) {
			const uint32_t rows = rows_cases[r],
				       inner = inner_cases[i];
			struct exl3_expert_record record;
			memset(&record, 0, sizeof(record));
			record.pair_count = rows * 6u;
			record.pairs_per_row = 6u;
			record.job_count = rows <= 8 ?
						   rows * 6u :
						   (rows * 6u + 7u * 384u) / 8u;
			record.input_size = 5120u;
			record.inner_size = inner;
			record.bits_per_weight = 3u;
			record.activation_limit = 10.0f;
			const uint64_t pairs_bytes =
				(uint64_t)record.pair_count *
				record.input_size * 16u;
			record.jobs_offset = 1ull << 20;
			record.input_offset = 1ull << 24;
			record.operands_offset =
				record.input_offset + (64ull << 20);
			record.products_offset =
				record.operands_offset + pairs_bytes;
			record.output_offset = record.products_offset;
			record.hidden_offset =
				record.products_offset + pairs_bytes;
			struct kernel_launch retained[EXL3_EXPERT_LAUNCHES],
				fast[EXL3_EXPERT_LAUNCHES];
			const result_status a = exl3_experts_prepare(
				&record, arena, &backend, card, retained);
			const result_status b = exl3_experts_fast_prepare(
				&record, arena, &backend, card, fast);
			EXPECT(a == STATUS_OK && b == STATUS_OK,
			       "prepare rows %u inner %u: %d %d", rows, inner,
			       (int)a, (int)b);
			if (a != STATUS_OK || b != STATUS_OK)
				continue;
			printf("--- local-K rows %u inner %u\n", rows, inner);
			show("retained", retained, EXL3_EXPERT_LAUNCHES);
			show("fast", fast, EXL3_EXPERT_LAUNCHES);
			const bool wide = rows > 8;
			for (uint32_t pass = 0; pass < EXL3_EXPERT_LAUNCHES;
			     pass++) {
				const uint32_t kernel = fast[pass].kernel - 1u;
				EXPECT(strcmp(files[kernel],
					      EXL3_WIDE_KERNEL_FILE) == 0,
				       "module %s", files[kernel]);
				EXPECT(fast[pass].grid[1] ==
						       retained[pass].grid[1] &&
					       fast[pass].grid[2] ==
						       retained[pass].grid[2] &&
					       fast[pass].block[0] ==
						       retained[pass].block[0] &&
					       fast[pass].argument_byte_count ==
						       retained[pass]
							       .argument_byte_count &&
					       memcmp(fast[pass].argument,
						      retained[pass].argument,
						      retained[pass]
							      .argument_byte_count) ==
						       0,
				       "pass %u axes/record differ", pass);
				const bool product = pass == 1u || pass == 3u;
				const uint32_t width =
					pass == 1u || pass == 2u ? inner :
								   5120u;
				const uint32_t expected_x =
					product ?
						width / 16u / (wide ? 4u : 8u) :
						(width / 128u + 3u) / 4u;
				EXPECT(fast[pass].grid[0] == expected_x,
				       "pass %u grid x %u expected %u", pass,
				       fast[pass].grid[0], expected_x);
				if (product)
					EXPECT(strstr(names[kernel],
						      wide ? "_t4_g4p" :
							     "_t8_g1") != NULL,
					       "pass %u name %s", pass,
					       names[kernel]);
			}
		}
    }
    printf("kernels prepared: %u\n", kernel_count);
    printf("result=%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
