/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU test of the engine patch (exl3_v3_b): the launch tables of exl3_experts_fast_prepare and of
 * bb_exl3_pass_prepare with product_tiles 8 and wide calls, against the retained tables: same record, ranges,
 * job axis and projection axis; module, entry names and grid x as documented. A stub backend records names.
 * Per arithmetic class: class 120 native takes the *_native modules with class 80's launches otherwise; class 120's
 * emulation of class 80 takes class 80's retained launches for both forms (kernel_exl3_wide has no build of it); the
 * relaxed order only on class 80; any other class refused with the launches untouched. */
#include <stdio.h>
#include <string.h>

#include "engine/library/exl3/exl3.h"
#include "engine/library/exl3/exl3_tp.h"

#define MOST_KERNELS 64
#define FILE_BYTES 64
#define NAME_BYTES 96
#define CLASS_CASE_ROWS 6u
#define CLASS_CASE_INNER 640u
#define NOT_A_CLASS 90u /* the classes are 80, 120 and 1200 (backend_seam.h) */
#define LAUNCH_FILL 0xA5
/* the products of gate and up: the relaxed form's _s8 */
#define GATE_UP_PASS 1u
#define ACTIVATE_PASS 2u
#define DOWN_PASS 3u
#define PAIRS_PER_ROW 6u
/* LINEAR_MAXIMUM_ROWS: up to here a call is thin, one job a pair */
#define THIN_ROWS 8u
/* a wide call's jobs: its pairs in 8-row jobs, at most one more an expert */
#define ROUTED_EXPERTS 384u
#define HIDDEN 5120u
#define BITS 3u
#define ACTIVATION_LIMIT 10.0f
/* more than the operands' and the products' bytes of a pair's input */
#define SCRATCH_BYTES_PER_INPUT 16u
/* the cases: rows 6, THIN_ROWS (thin), 9, 64 and 1024 (wide), each with inner 512 and 640 */
#define CASE_THIN_ROWS 6
#define CASE_WIDE_LEAST_ROWS 9
#define CASE_WIDE_ROWS 64
#define CASE_WIDEST_ROWS 1024
#define CASE_INNER 512
#define CASE_WIDER_INNER 640
#define INNER_CASES 2
#define GRID_Z 2 /* a launch grid's third dimension */
/* the expected grid: a product pass's blocks of PRODUCT_COLUMNS columns over the tiles (WIDE_TILES wide, else
 * THIN_TILES), another pass's of PASS_COLUMNS in groups of INPUT_GROUP, rounded up */
#define PRODUCT_COLUMNS 16u
#define WIDE_TILES 4u
#define THIN_TILES 8u
#define PASS_COLUMNS 128u
#define INPUT_GROUP 4u
#define INPUT_ROUNDING 3u
#define JOBS_OFFSET (1ull << 20)
#define INPUT_OFFSET (1ull << 24)
#define INPUT_BYTES (64ull << 20)
#define ARENA_BYTES (1ull << 34)

static char files[MOST_KERNELS][FILE_BYTES], names[MOST_KERNELS][NAME_BYTES];
static uint32_t kernel_count;

MUST_CHECK static result_status stub_prepare(struct device_card *card,
					     const char *file, const char *name,
					     uint32_t argument_bytes,
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
    (void)snprintf(files[kernel_count], sizeof(files[0]), "%s", file);
    (void)snprintf(names[kernel_count], sizeof(names[0]), "%s", name);
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

/* A routed-experts record of rows rows (PAIRS_PER_ROW pairs a row) within ARENA_BYTES, its ranges disjoint. */
static struct exl3_expert_record record_of(uint32_t rows, uint32_t inner)
{
	struct exl3_expert_record record;
	memset(&record, 0, sizeof(record));
	record.pair_count = rows * PAIRS_PER_ROW;
	record.pairs_per_row = PAIRS_PER_ROW;
	record.job_count = rows <= THIN_ROWS ?
				   rows * PAIRS_PER_ROW :
				   (rows * PAIRS_PER_ROW +
				    (THIN_ROWS - 1u) * ROUTED_EXPERTS) /
					   THIN_ROWS;
	record.input_size = HIDDEN;
	record.inner_size = inner;
	record.bits_per_weight = BITS;
	record.activation_limit = ACTIVATION_LIMIT;
	const uint64_t pairs_bytes = (uint64_t)record.pair_count *
				     record.input_size *
				     SCRATCH_BYTES_PER_INPUT;
	record.jobs_offset = JOBS_OFFSET;
	record.input_offset = INPUT_OFFSET;
	record.operands_offset = record.input_offset + INPUT_BYTES;
	record.products_offset = record.operands_offset + pairs_bytes;
	record.output_offset = record.products_offset;
	record.hidden_offset = record.products_offset + pairs_bytes;
	return record;
}

typedef MUST_CHECK result_status (*experts_prepare_function)(
	const struct exl3_expert_record *, uint64_t,
	const struct backend_operations *, struct device_card *, uint32_t,
	struct kernel_launch *);

/* Every launch's module is file; the launches equal reference's but for the kernel. */
static void expect_class_launches(const char *what, uint32_t arithmetic_class,
				  const struct kernel_launch *launches,
				  const struct kernel_launch *reference,
				  const char *file)
{
	for (uint32_t pass = 0; pass < EXL3_EXPERT_LAUNCHES; pass++) {
		const uint32_t kernel = launches[pass].kernel - 1u;
		const uint32_t reference_kernel = reference[pass].kernel - 1u;
		if (kernel >= kernel_count ||
		    reference_kernel >= kernel_count) {
			EXPECT(false, "%s class %u pass %u: no kernel recorded",
			       what, arithmetic_class, pass);
			continue;
		}
		struct kernel_launch same = launches[pass];
		same.kernel = reference[pass].kernel;
		EXPECT(strcmp(files[kernel], file) == 0 &&
			       strcmp(names[kernel], names[reference_kernel]) ==
				       0,
		       "%s class %u pass %u: %s/%s, expected %s/%s", what,
		       arithmetic_class, pass, files[kernel], names[kernel],
		       file, names[reference_kernel]);
		EXPECT(memcmp(&same, &reference[pass], sizeof(same)) == 0,
		       "%s class %u pass %u: launch differs from class 80's",
		       what, arithmetic_class, pass);
	}
}

static void class_cases(const struct backend_operations *backend,
			struct device_card *card)
{
	const uint64_t arena = ARENA_BYTES;
	const struct exl3_expert_record record =
		record_of(CLASS_CASE_ROWS, CLASS_CASE_INNER);
	/* class 80's retained launches: class 120's emulation gets them for both forms (no emulated wide build) */
	struct kernel_launch retained[EXL3_EXPERT_LAUNCHES];
	const result_status retained_status =
		exl3_experts_prepare(&record, arena, backend, card,
				     ARITHMETIC_CLASS_SM_80, retained);
	EXPECT(retained_status == STATUS_OK, "retained class 80 prepare: %d",
	       (int)retained_status);
	const struct {
		const char *what;
		experts_prepare_function prepare;
		const char *native_file;
	} forms[] = { { "retained", exl3_experts_prepare,
			EXL3_KERNEL_FILE_NATIVE },
		      { "fast", exl3_experts_fast_prepare,
			EXL3_WIDE_KERNEL_FILE_NATIVE } };
	for (uint32_t form = 0; retained_status == STATUS_OK &&
				form < sizeof(forms) / sizeof(forms[0]);
	     form++) {
		struct kernel_launch reference[EXL3_EXPERT_LAUNCHES],
			launches[EXL3_EXPERT_LAUNCHES];
		const result_status reference_status =
			forms[form].prepare(&record, arena, backend, card,
					    ARITHMETIC_CLASS_SM_80, reference);
		EXPECT(reference_status == STATUS_OK, "%s class 80 prepare: %d",
		       forms[form].what, (int)reference_status);
		const result_status emulated_status =
			forms[form].prepare(&record, arena, backend, card,
					    ARITHMETIC_CLASS_SM_120, launches);
		EXPECT(emulated_status == STATUS_OK, "%s class 120 prepare: %d",
		       forms[form].what, (int)emulated_status);
		if (emulated_status == STATUS_OK)
			expect_class_launches(forms[form].what,
					      ARITHMETIC_CLASS_SM_120, launches,
					      retained, EXL3_KERNEL_FILE);
		const result_status native_status = forms[form].prepare(
			&record, arena, backend, card,
			ARITHMETIC_CLASS_SM_120_NATIVE, launches);
		EXPECT(native_status == STATUS_OK,
		       "%s class 120 native prepare: %d", forms[form].what,
		       (int)native_status);
		if (reference_status == STATUS_OK && native_status == STATUS_OK)
			expect_class_launches(forms[form].what,
					      ARITHMETIC_CLASS_SM_120_NATIVE,
					      launches, reference,
					      forms[form].native_file);
	}
	struct kernel_launch launches[EXL3_EXPERT_LAUNCHES], untouched;
	EXPECT(exl3_experts_relaxed_prepare(&record, arena, backend, card,
					    ARITHMETIC_CLASS_SM_80,
					    launches) == STATUS_OK &&
		       launches[GATE_UP_PASS].kernel - 1u < kernel_count &&
		       strstr(names[launches[GATE_UP_PASS].kernel - 1u],
			      "_s8") != NULL &&
		       strcmp(files[launches[GATE_UP_PASS].kernel - 1u],
			      EXL3_KERNEL_FILE) == 0,
	       "relaxed class 80 prepare");
	const uint32_t refused[] = { ARITHMETIC_CLASS_SM_120,
				     ARITHMETIC_CLASS_SM_120_NATIVE, 0u,
				     NOT_A_CLASS };
	for (uint32_t index = 0; index < sizeof(refused) / sizeof(refused[0]);
	     index++) {
		const bool is_pro = refused[index] == ARITHMETIC_CLASS_SM_120 ||
				    refused[index] ==
					    ARITHMETIC_CLASS_SM_120_NATIVE;
		memset(launches, LAUNCH_FILL, sizeof(launches));
		memset(&untouched, LAUNCH_FILL, sizeof(untouched));
		EXPECT(exl3_experts_relaxed_prepare(&record, arena, backend,
						    card, refused[index],
						    launches) ==
				       (is_pro ? STATUS_UNSUPPORTED :
						 STATUS_INVALID_ARGUMENT) &&
			       memcmp(&launches[0], &untouched,
				      sizeof(untouched)) == 0,
		       "relaxed class %u refused untouched", refused[index]);
		if (is_pro)
			continue;
		EXPECT(exl3_experts_prepare(&record, arena, backend, card,
					    refused[index], launches) ==
				       STATUS_INVALID_ARGUMENT &&
			       memcmp(&launches[0], &untouched,
				      sizeof(untouched)) == 0,
		       "retained class %u refused untouched", refused[index]);
		EXPECT(exl3_experts_fast_prepare(&record, arena, backend, card,
						 refused[index], launches) ==
				       STATUS_INVALID_ARGUMENT &&
			       memcmp(&launches[0], &untouched,
				      sizeof(untouched)) == 0,
		       "fast class %u refused untouched", refused[index]);
	}
}

static void show(const char *what, const struct kernel_launch *launches,
		 uint32_t count)
{
    for (uint32_t index = 0; index < count; index++) {
        const uint32_t kernel = launches[index].kernel - 1u;
        printf("%s pass %u: %s/%s grid %u x %u x %u block %u\n", what, index, files[kernel], names[kernel],
               launches[index].grid[0], launches[index].grid[1], launches[index].grid[GRID_Z],
               launches[index].block[0]);
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
	const uint64_t arena = ARENA_BYTES;
	const uint32_t rows_cases[] = { CASE_THIN_ROWS, THIN_ROWS,
					CASE_WIDE_LEAST_ROWS, CASE_WIDE_ROWS,
					CASE_WIDEST_ROWS };
	const uint32_t inner_cases[] = { CASE_INNER, CASE_WIDER_INNER };
	for (uint32_t r = 0; r < sizeof(rows_cases) / sizeof(rows_cases[0]);
	     r++) {
		for (uint32_t i = 0; i < INNER_CASES; i++) {
			const uint32_t rows = rows_cases[r],
				       inner = inner_cases[i];
			const struct exl3_expert_record record =
				record_of(rows, inner);
			struct kernel_launch retained[EXL3_EXPERT_LAUNCHES],
				fast[EXL3_EXPERT_LAUNCHES];
			const result_status a = exl3_experts_prepare(
				&record, arena, &backend, card,
				ARITHMETIC_CLASS_SM_80, retained);
			const result_status b = exl3_experts_fast_prepare(
				&record, arena, &backend, card,
				ARITHMETIC_CLASS_SM_80, fast);
			EXPECT(a == STATUS_OK && b == STATUS_OK,
			       "prepare rows %u inner %u: %d %d", rows, inner,
			       (int)a, (int)b);
			if (a != STATUS_OK || b != STATUS_OK)
				continue;
			printf("--- local-K rows %u inner %u\n", rows, inner);
			show("retained", retained, EXL3_EXPERT_LAUNCHES);
			show("fast", fast, EXL3_EXPERT_LAUNCHES);
			const bool wide = rows > THIN_ROWS;
			for (uint32_t pass = 0; pass < EXL3_EXPERT_LAUNCHES;
			     pass++) {
				const uint32_t kernel = fast[pass].kernel - 1u;
				EXPECT(strcmp(files[kernel],
					      EXL3_WIDE_KERNEL_FILE) == 0,
				       "module %s", files[kernel]);
				EXPECT(fast[pass].grid[1] ==
						       retained[pass].grid[1] &&
					       fast[pass].grid[GRID_Z] ==
						       retained[pass]
							       .grid[GRID_Z] &&
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
				const bool product = pass == GATE_UP_PASS ||
						     pass == DOWN_PASS;
				const uint32_t width =
					pass == GATE_UP_PASS ||
							pass == ACTIVATE_PASS ?
						inner :
						HIDDEN;
				const uint32_t expected_x =
					product ? width / PRODUCT_COLUMNS /
							  (wide ? WIDE_TILES :
								  THIN_TILES) :
						  (width / PASS_COLUMNS +
						   INPUT_ROUNDING) /
							  INPUT_GROUP;
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
    class_cases(&backend, card);
    printf("kernels prepared: %u\n", kernel_count);
    printf("result=%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
