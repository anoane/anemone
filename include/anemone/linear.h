/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear.h: the linear on the cards: the launches (struct kernel_launch of backend_seam.h) of the kernels of
 * linear_records.h, each prepared once at load from a description that names every arena offset and count.
 *
 * Contract:
 *   - Nothing here runs per token. First users: the model's load, which fills its launch tables, and the card tests
 *     of the kernels. linear_kernels_prepare asks a card for every kernel but the wide trunk instances, once;
 *     linear_wide_kernels_prepare for those the wide launches of its class take. A prepare function is
 *     arithmetic on its description: it touches
 *     no card, allocates nothing and reads nothing of an arena. A launch it gives is run as it is, by launch_run, as
 *     often as the model likes; the launches of a chain run in the order of their enumeration. Any thread may call
 *     any function here.
 *   - A range (struct linear_range) is what the planner set aside in the arena for one array of
 *     linear_records.h. The comment of every range says which array it holds and whether the launches read or write
 *     it; the byte count functions at the end say how many bytes an array has.
 *   - A prepare function checks its description against every rule of linear_records.h that can be checked without
 *     the contents of the arena, and against the size of the arena. It refuses, with
 *     STATUS_INVALID_ARGUMENT, one error line that names the rule and the launches left untouched:
 *       arguments  a NULL argument; a handle of a kernel it launches that is BACKEND_NO_KERNEL; a reserved
 *                  field, of the description, of a matrix or of a shape, that is not zero;
 *       matrices   a weight format other than the one its kernel reads (WEIGHT_FORMAT_E2M1_SCALED for the
 *                  routed experts, WEIGHT_FORMAT_E4M3_SCALED everywhere else), or one this build does not
 *                  carry; a shape that format does not store (stored_byte_count of weight_format_seam.h); an
 *                  input_size that is no multiple of LINEAR_VALUES_PER_LOAD, which E4M3_SCALED stores and no
 *                  kernel reads; a slice_count for which no kernel is stamped (not a count of
 *                  LINEAR_EVERY_SLICE_COUNT) or above input_size / LINEAR_VALUES_PER_LOAD; a
 *                  segment_count of 0 or one that does not divide output_size / LINEAR_OUTPUTS_PER_TILE; in
 *                  a chain, a down matrix whose input_size is not the output_size of gate and up, and an up matrix
 *                  that has not the sizes of gate; a matrix_count of a trunk launch other than 1 or 2, and two
 *                  matrices of one launch that differ in input_size, slice_count, weight format or wide form (so gate
 *                  and up of the shared expert) or have more outputs together than 32 bits count; a wide form not of
 *                  enum linear_wide_form, and in a launch of more than LINEAR_MAXIMUM_ROWS rows an
 *                  interleaved form of a matrix of more than one slice or of bfloat16 rows;
 *       counts     a width (of quantized rows, and of a row sum: the output_size of the routed experts' down) that
 *                  is 0 or no multiple of LINEAR_VALUES_PER_LOAD; a row_count of 0 or above its bound
 *                  (trunk launches and the shared expert LINEAR_WIDE_MOST_ROWS, quantized rows and the
 *                  routed experts MODEL_WIDE_MOST_ROWS); a pairs_per_row of 0, or so many that row_count *
 *                  pairs_per_row passes 32 bits; a job_count of 0; an activation_limit that is not above zero (a
 *                  not-a-number is not; a model without a limit gives +infinity);
 *       ranges     a byte_offset that is no multiple of LINEAR_OFFSET_ALIGNMENT_BYTES; a range that leaves
 *                  the arena (held in 64 bits, with no wrap); a byte_count below the bytes the launches touch there
 *                  (counted in 64 bits: a count beyond them fits no range); a matrix whose byte_count is not exactly
 *                  what its format stores for its shape, which is what a pack's index gives; two ranges of one
 *                  description that share a byte.
 *     The last rule asks more than one kernel does (a kernel forbids only what it writes to lie on something else
 *     it touches): every range a description names is an array of its own, so that no launch, of this chain or of
 *     the one that fills what is read here, destroys what another reads.
 *   - Not checked here, because it is not known at load: what the arena holds. Whoever writes a job table (struct
 *     expert_job_record) checks, for every one of its job_count entries: row_count is at most
 *     LINEAR_MAXIMUM_ROWS (0 for an entry not in use: every entry is read); each of its three offsets is a
 *     multiple of LINEAR_OFFSET_ALIGNMENT_BYTES and its matrix, of the shape and format of the description,
 *     lies wholly inside the range expert_weights; every pair_index in use is below row_count * pairs_per_row of the
 *     description; no pair stands in two columns of the table. Nor are the card's own limits of a launch known
 *     here (blocks of a grid, threads and shared memory of a block): launch_run refuses a launch beyond them.
 *   - row_count is the number of rows the launches cover and what the ranges are checked for, for the routed experts
 *     too: a job has as many columns, so one job holds every row of its expert. Launches prepared for different row
 *     counts may name the same ranges: each uses the first rows, and the pairs of the first rows, of every array.
 *     A trunk launch (and each of the shared expert's) of more than LINEAR_MAXIMUM_ROWS rows takes its
 *     matrix's wide form (enum linear_wide_form): the wide instance of its slice count, or the interleaved
 *     instance of one slice, with the record, block and shared memory of the thin one and the same bits for every row
 *     (linear_records.h), and a block for each group of 8 rows (interleaved: each chunk of 32) of each tile, or one a
 *     tile (fewer only where grid[0] would pass 32 bits); its handle, BACKEND_NO_KERNEL until
 *     linear_wide_kernels_prepare, is refused as a kernel not prepared. Up to LINEAR_MAXIMUM_ROWS the
 *     launch is the thin one, unchanged, whatever the form.
 *   - slice_count is S of linear_records.h, and the bits of a product depend on it: one S is chosen per matrix
 *     before a golden is recorded, and it is the same on every card that must give the same bits.
 *     linear_default_slice_count gives it from a fixed table of measured counts: one per kernel family,
 *     shape, class of card and profile, never chosen by row count, chunk or timing, so that greedy tokens repeat and
 *     drafted tokens are the undrafted ones on one layout; the two matrices of one trunk launch (attn.wq_a with
 *     attn.wkv) get one count. The reference's order (one slice, one chain in the order the reference adds in) is
 *     S = 1 in every profile. One exception, which keeps every bit: on class 80 and in class 120's emulation the
 *     slices are ordered, one chain in the reference order whatever S (linear_ordered_chains of
 *     kernel_linear_common.cuh), so a wide launch there takes the count linear_default_wide_form measured
 *     fastest; class 120 native keeps its default count. On the cards, linear_exact_card_test holds 1 to 8 slices to
 *     that order on sums made to depend on it (its cases held in order have 12 loads, too few for 16), and
 *     trunk_wide_card_test every count of attn.wq_a with attn.wkv and of attn.wo_a, 16 among them, to the default
 *     count's bits on the model's shapes; a 170HX at 16 slices also gave the linear_trunk_card_test digest of the
 *     PRO's emulation at 8.
 *   - Every launch has grid[2], block[1] and block[2] of 1, and the grid, block and shared memory of the table in
 *     linear_records.h. Where that table leaves block[0] open (the quantizer, the row sum, the hidden step) it is
 *     LINEAR_ELEMENT_THREADS.
 */
#ifndef LINEAR_H
#define LINEAR_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear_records.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"

/* The handles of the kernels of the linear on one card; of a product kernel, one per instance, in the order of
 * LINEAR_EVERY_SLICE_COUNT. The wide trunk instances (more than LINEAR_MAXIMUM_ROWS rows) come last,
 * the interleaved instance of one slice the very last. */
struct linear_kernels_record {
	uint32_t activation_quantize;
	uint32_t expert_gate_up[LINEAR_SLICE_INSTANCES];
	uint32_t expert_down[LINEAR_SLICE_INSTANCES];
	uint32_t expert_row_sum;
	uint32_t trunk_linear_quantized[LINEAR_SLICE_INSTANCES];
	uint32_t trunk_linear_bfloat16[LINEAR_SLICE_INSTANCES];
	uint32_t trunk_hidden;
	uint32_t trunk_linear_quantized_wide[LINEAR_SLICE_INSTANCES];
	uint32_t trunk_linear_bfloat16_wide[LINEAR_SLICE_INSTANCES];
	uint32_t trunk_linear_quantized_interleaved; /* of one slice: in place of a wide launch's (linear_records.h) */
};

/* First user: the model's load, once for each open card. Prepares every kernel but the wide trunk instances (the
 * interleaved one among them) in the order of the members (at most BACKEND_MAXIMUM_KERNELS), each with the
 * file name, the function name and the record size linear_records.h gives it; on a card of arithmetic_class
 * ARITHMETIC_CLASS_SM_120_NATIVE the product files' *_NATIVE builds. The wide ones it sets to
 * BACKEND_NO_KERNEL. The first kernel_prepare
 * that fails ends it with that status, behind one error line of its own that names the kernel; a NULL argument or a
 * table without kernel_prepare is STATUS_INVALID_ARGUMENT. On failure kernels_out is untouched. */
MUST_CHECK result_status linear_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct linear_kernels_record *kernels_out);

/* The two precision profiles (most precise; fastest without losing quality), the order of the additions of a
 * product (the profile's fixed order, or the reference's), and the kernel family of a product. */
enum linear_profile_code {
	LINEAR_PROFILE_PRECISION = 0,
	LINEAR_PROFILE_SPEED = 1
};
enum linear_order_code { LINEAR_ORDER_FIXED = 0, LINEAR_ORDER_REFERENCE = 1 };

/* The CMP trunk order is pinned at startup. Reference keeps thin launches through 8 rows and takes the ordered
 * wide instance above them; relaxed takes the measured decode/wide forms where the load proved the scale fold. */
enum cmp_order_code { CMP_ORDER_REFERENCE = 0, CMP_ORDER_RELAXED = 1 };
struct relaxed_linear_kernels;

/* First user: the model's load, on a card that runs a wide step's trunk launches. Prepares the wide trunk instances
 * that linear_default_wide_form gives the matrices of the card's class in that profile and order (of the
 * wide instances, and the interleaved one of one slice: those alone, so that a card holds no kernel it never
 * launches) into a table linear_kernels_prepare filled for the same card and class, in the order of the
 * members and as that function does, the other members left as they are; failures as there, kernels untouched. */
MUST_CHECK result_status linear_wide_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	enum linear_profile_code profile, enum linear_order_code order,
	uint32_t arithmetic_class, struct linear_kernels_record *kernels);
enum linear_family_code {
	LINEAR_FAMILY_EXPERT_GATE_UP =
		0, /* gate and up of a routed expert: of either */
	LINEAR_FAMILY_EXPERT_DOWN = 1,
	LINEAR_FAMILY_TRUNK_QUANTIZED =
		2, /* of each matrix of a trunk launch, alone or of two */
	LINEAR_FAMILY_TRUNK_BFLOAT16 =
		3 /* the shape of the whole matrix, its segments included */
};
/* First user: the model's load, which fills every slice_count. The slice count of a product of a matrix of shape
 * (its sizes alone are read) on a card of arithmetic_class (backend_seam.h). A profile, order or family outside its
 * enumeration, a NULL argument, and a shape or class the table has no row for are STATUS_INVALID_ARGUMENT
 * with one error line; slice_count_out is then untouched. */
MUST_CHECK result_status linear_default_slice_count(
	enum linear_profile_code profile, enum linear_order_code order,
	enum linear_family_code family, const struct matrix_shape_record *shape,
	uint32_t arithmetic_class, uint32_t *slice_count_out);

/* The form of a trunk launch of more than LINEAR_MAXIMUM_ROWS rows (linear_records.h): the wide instance of
 * the matrix's slice count with a block for each group of 8 rows of each tile, or a block a tile; the interleaved
 * instance (quantized rows in one slice) with a block for each chunk of LINEAR_INTERLEAVED_ROWS rows of each
 * tile, or a block a tile. Every form gives the thin instance's bits. */
enum linear_wide_form {
	LINEAR_WIDE_GROUP_BLOCKS = 0,
	LINEAR_WIDE_TILE_BLOCKS = 1,
	LINEAR_INTERLEAVED_CHUNK_BLOCKS = 2,
	LINEAR_INTERLEAVED_TILE_BLOCKS = 3,
	LINEAR_WIDE_FORM_COUNT = 4 /* number of forms; not a form */
};

/* Cold explicit reference-wide lane for legal new matrix shapes, including TP shards. Prepares every ordinary
 * wide slice instance, preserving all other handles. It assigns no measured default policy or speed credit;
 * the caller supplies the admitted slice count and GROUP/TILE form in each matrix and qualifies the new shapes. */
MUST_CHECK result_status linear_wide_reference_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class, struct linear_kernels_record *kernels);
/* First user: the model's load, for a wide step's trunk launches. The slice count and the form of a launch of more
 * than LINEAR_MAXIMUM_ROWS rows of a matrix of the trunk families, from the table of
 * linear_default_slice_count: the fastest the wide trunk card test measured at 1,024 rows (26 October 2026)
 * among those with the bits of the matrix's thin launches. Class 120 native keeps the default count (its bits depend
 * on it); class 80 and class 120's emulation take the fastest count measured, any count adding in the reference
 * order there (above). Refused as linear_default_slice_count is, and an expert family or a NULL form_out;
 * the outputs are then untouched. */
MUST_CHECK result_status linear_default_wide_form(
	enum linear_profile_code profile, enum linear_order_code order,
	enum linear_family_code family, const struct matrix_shape_record *shape,
	uint32_t arithmetic_class, uint32_t *slice_count_out,
	enum linear_wide_form *form_out);

/* Bytes of the arena set aside for one array. */
struct linear_range {
	uint64_t byte_offset;
	uint64_t byte_count;
};

/* A matrix where it lies in the arena, as the index of a pack gives it, the slices its products are cut into and the
 * form of its launches of more than LINEAR_MAXIMUM_ROWS rows (enum linear_wide_form). */
struct linear_matrix {
	struct linear_range
		stored; /* read: exactly the bytes its format stores for its shape */
	struct matrix_shape_record shape;
	enum weight_format_code weight_format;
	uint32_t slice_count;
	enum linear_wide_form wide_form;
	uint32_t reserved_zero;
};

/* Rows into the codes, exponents and operands every quantized matrix is fed with: one launch of
 * activation_quantize_entry, of 1 to MODEL_WIDE_MOST_ROWS rows (a wide step's chunk too). The rows of a layer
 * are quantized once: its routed experts and its shared expert read the same arrays. Operands have the bytes of values
 * (linear_values_byte_count). */
struct linear_quantize_description {
	struct linear_range values; /* read: values [row_count][width] */
	struct linear_range codes; /* written: codes [row_count][width] */
	struct linear_range
		exponents; /* written: exponents [row_count][width / 32] */
	struct linear_range operands; /* written: operands [row_count][width] */
	uint32_t width;
	uint32_t row_count;
};
MUST_CHECK result_status linear_quantize_prepare(
	const struct linear_quantize_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launch_out);

/* y = W x for quantized rows, of one matrix or of two that read the same rows (attn.wq_a with attn.wkv; gate and
 * up of the shared expert): one launch of trunk_linear_quantized_value. Of matrices and outputs the first
 * matrix_count are read. */
struct linear_trunk_quantized_description {
	struct linear_matrix
		matrices[LINEAR_MERGED_MATRICES]; /* WEIGHT_FORMAT_E4M3_SCALED */
	struct linear_range codes; /* read: codes [row_count][input_size] */
	struct linear_range
		exponents; /* read: exponents [row_count][input_size / 32] */
	struct linear_range operands; /* read: operands [row_count][input_size] */
	struct linear_range outputs
		[LINEAR_MERGED_MATRICES]; /* written: [row_count][output_size] */
	uint32_t matrix_count; /* 1 or 2 */
	uint32_t row_count;
};
MUST_CHECK result_status linear_trunk_quantized_prepare(
	const struct linear_trunk_quantized_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launch_out);

/* y = W x for rows that are not quantized: one launch of trunk_linear_bfloat16_value. segment_count is 1 for an
 * ordinary matrix and the number of groups for one that is block-diagonal over groups (linear_records.h). */
struct linear_trunk_bfloat16_description {
	struct linear_matrix matrix; /* WEIGHT_FORMAT_E4M3_SCALED */
	struct linear_range
		values; /* read: values [row_count][segment_count * input_size] */
	struct linear_range output; /* written: values [row_count][output_size] */
	uint32_t row_count;
	uint32_t segment_count;
};
MUST_CHECK result_status linear_trunk_bfloat16_prepare(
	const struct linear_trunk_bfloat16_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launch_out);

/* The routed experts of a layer for a job table that lies in the arena, from the rows' codes to the float32 sum of
 * each row's experts: 1 to MODEL_WIDE_MOST_ROWS rows (a wide step's chunk: its jobs, of 8 columns at most as
 * any, come from the wide select), not the thin products' 8. A pair is one row at one of its pairs_per_row routed
 * experts. The first three launches leave
 * the pair outputs of the pairs that stand in a job (the second quantizes the hidden vector of every pair, a stale
 * one where the pair stands in no job); the outputs of a pair computed elsewhere are put into pair_output before the
 * last launch, which sums whatever lies there. */
enum linear_routed_experts_launch {
	LINEAR_ROUTED_EXPERTS_GATE_UP =
		0, /* expert_gate_up_value: jobs and the rows' arrays to hidden */
	LINEAR_ROUTED_EXPERTS_QUANTIZE =
		1, /* activation_quantize_entry: hidden of every pair to its arrays */
	LINEAR_ROUTED_EXPERTS_DOWN =
		2, /* expert_down_value: jobs and the hidden arrays to pair_output */
	LINEAR_ROUTED_EXPERTS_ROW_SUM =
		3, /* expert_row_sum_entry: pair_output to row_output */
	LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT = 4
};
struct linear_routed_experts_description {
	struct linear_range jobs; /* read: struct expert_job_record [job_count] */
	struct linear_range
		expert_weights; /* read: holds every matrix a job names: at least the larger shape */
	struct linear_range
		codes; /* read: codes [row_count][gate_up_shape.input_size] */
	struct linear_range
		exponents; /* read: exponents [row_count][gate_up_shape.input_size / 32] */
	struct linear_range
		operands; /* read: operands [row_count][gate_up_shape.input_size] */
	struct linear_range
		hidden; /* scratch: values [pairs][gate_up_shape.output_size] */
	struct linear_range
		hidden_codes; /* scratch: codes [pairs][gate_up_shape.output_size] */
	struct linear_range
		hidden_exponents; /* scratch: exponents [pairs][gate_up_shape.output_size / 32] */
	struct linear_range
		hidden_operands; /* scratch: operands [pairs][gate_up_shape.output_size] */
	struct linear_range
		pair_output; /* scratch: values [pairs][down_shape.output_size] */
	struct linear_range
		row_output; /* written: sums [row_count][down_shape.output_size] */
	struct matrix_shape_record
		gate_up_shape; /* of the gate and of the up matrix of every expert */
	struct matrix_shape_record down_shape;
	enum weight_format_code
		weight_format; /* native linear: E2M1; router-only EXL3 dispatches separate passes */
	uint32_t job_count; /* entries of the table: every one is read at every step */
	uint32_t row_count;
	uint32_t pairs_per_row; /* pairs = row_count * pairs_per_row */
	uint32_t gate_up_slice_count;
	uint32_t down_slice_count;
	float activation_limit; /* of the hidden step */
	uint32_t reserved_zero;
};
/* launches_out: LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT launches. */
MUST_CHECK result_status linear_routed_experts_prepare(
	const struct linear_routed_experts_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launches_out);

/* The shared expert of a layer, an expert of three trunk matrices, from the rows' codes to its bfloat16 output. */
enum linear_shared_expert_launch {
	LINEAR_SHARED_EXPERT_GATE_UP =
		0, /* trunk_linear_quantized_value, two matrices: to gate_output and
                                                   up_output */
	LINEAR_SHARED_EXPERT_HIDDEN =
		1, /* trunk_hidden_entry: gate_output and up_output to hidden */
	LINEAR_SHARED_EXPERT_QUANTIZE =
		2, /* activation_quantize_entry: hidden to its arrays */
	LINEAR_SHARED_EXPERT_DOWN =
		3, /* trunk_linear_quantized_value: the hidden arrays to output */
	LINEAR_SHARED_EXPERT_LAUNCH_COUNT = 4
};
struct linear_shared_expert_description {
	struct linear_matrix
		gate; /* the three: WEIGHT_FORMAT_E4M3_SCALED; gate and up of one
                                                   slice_count */
	struct linear_matrix up;
	struct linear_matrix down;
	struct linear_range
		codes; /* read: codes [row_count][gate.shape.input_size] */
	struct linear_range
		exponents; /* read: exponents [row_count][gate.shape.input_size / 32] */
	struct linear_range
		operands; /* read: operands [row_count][gate.shape.input_size] */
	struct linear_range
		gate_output; /* scratch: values [row_count][gate.shape.output_size] */
	struct linear_range up_output; /* scratch: the same array for up */
	struct linear_range
		hidden; /* scratch: the same array for the hidden step */
	struct linear_range
		hidden_codes; /* scratch: codes [row_count][gate.shape.output_size] */
	struct linear_range
		hidden_exponents; /* scratch: exponents [row_count][gate.shape.output_size / 32] */
	struct linear_range
		hidden_operands; /* scratch: operands [row_count][gate.shape.output_size] */
	struct linear_range
		output; /* written: values [row_count][down.shape.output_size] */
	uint32_t row_count;
	float activation_limit; /* of the hidden step */
};
/* launches_out: LINEAR_SHARED_EXPERT_LAUNCH_COUNT launches. */
MUST_CHECK result_status linear_shared_expert_prepare(
	const struct linear_shared_expert_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	struct kernel_launch *launches_out);

/* Class-80 trunk selection, prepared at load. The descriptions and refusals of the corresponding prepare above
 * still apply, as do relaxed_linear_records.h's additional shape/segment bounds when its kernel is selected.
 * Other classes and reference launches of at most 8 rows retain every byte of those launches. Class 80 above
 * 8 rows takes the ordered instance by default. Relaxed selection requires every matrix in that product to have
 * a nonzero fold reference from relaxed_fold_check; zero (or a NULL reference array) keeps that product in
 * reference order. Quantized references must be RELAXED_FOLD_REFERENCE; bfloat16 references must be valid
 * float32 power fields. The shared array is gate, up, down; the quantized array has matrix_count entries.
 * Relaxed decode slice counts are fixed per measured shape, independent of row_count. One split is used by every
 * wide product. An unknown class or order is INVALID_ARGUMENT. Output launches are untouched on any failure. */
MUST_CHECK result_status linear_trunk_quantized_select_prepare(
	const struct linear_trunk_quantized_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	const struct relaxed_linear_kernels *relaxed_kernels,
	uint32_t arithmetic_class, enum cmp_order_code cmp_order,
	const int32_t *fold_references, struct kernel_launch *launch_out);
MUST_CHECK result_status linear_trunk_bfloat16_select_prepare(
	const struct linear_trunk_bfloat16_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	const struct relaxed_linear_kernels *relaxed_kernels,
	uint32_t arithmetic_class, enum cmp_order_code cmp_order,
	int32_t fold_reference, struct kernel_launch *launch_out);
MUST_CHECK result_status linear_shared_expert_select_prepare(
	const struct linear_shared_expert_description *description,
	uint64_t arena_byte_count, const struct linear_kernels_record *kernels,
	const struct relaxed_linear_kernels *relaxed_kernels,
	uint32_t arithmetic_class, enum cmp_order_code cmp_order,
	const int32_t *fold_references, struct kernel_launch *launches_out);

/* First user: the planner, which lays out an arena at load. The bytes of each array of linear_records.h for a width
 * the kernels take, or UINT64_MAX where the count is beyond 64 bits: no range holds that. A matrix has the bytes its
 * format's stored_byte_count gives; every range begins on a multiple of LINEAR_OFFSET_ALIGNMENT_BYTES. */
uint64_t
linear_values_byte_count(uint64_t vector_count,
			 uint64_t width); /* bfloat16 [vector][width] */
uint64_t linear_codes_byte_count(uint64_t vector_count,
				 uint64_t width); /* E4M3 [vector][width] */
uint64_t
linear_exponents_byte_count(uint64_t vector_count,
			    uint64_t width); /* E8M0 [vector][width / 32] */
uint64_t linear_sums_byte_count(uint64_t row_count,
				uint64_t width); /* float32 [row][width] */
uint64_t
linear_jobs_byte_count(uint64_t job_count); /* struct expert_job_record */

#endif
