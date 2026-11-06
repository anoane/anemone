/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Cold text TP/PP composition using original packed shards; ordinary PP entry points and program ABI stay intact.
 * Canonical metadata owns global source/index identities. Views retain local tensors and expert tables while
 * borrowing global references. All preparation inputs remain immutable.
 *
 * Layouts: pairs14/15+PRO11, quad29+PRO11, or pair14/single7/single8+PRO11. TP2 uses1152 expert channels;
 * TP4 uses512/640/512/640. Attention heads split evenly and KV/router/mHC replicate. Native PRO retains the
 * original head/DSpark. State is16-bit, CMP local order reference; each layout/wire/shard recipe has its own salt.
 * Exact quad replicates original attention/shared products, uses576/1280 output-axis routed shards and opaque
 * hidden/FP32-sum gathers. Its WIDE/FULL FFN executes ordered slabs of at most8 rows. This is row slicing,
 * not overlapping row streams. Quad exact/BF16 groups coalesce at the prefill-cone boundary; PRO stays separate.
 * Default-off exact attention_owner executes attention only on its closed source/index-group owner, copies the
 * complete STREAMS/CARRIED_PRE residuals to the other FFN ranks, then runs the four-rank FFN companion group.
 * It retains an explicit source-20 front and PRO pool/key/candidate mirrors. Only live owner rings are checkpointed;
 * each attention/FFN boundary stays explicit, so this capacity control does not claim whole-prefix graph speed.
 * Context>=16384 is required by candidate-reader descriptions. THIN rows<=8; WIDE/FULL64..1024 retain ordered
 * state slabs and chunk products. Capacities>8 use shared site mailboxes with active-row, layer/step phases.
 * THIN alone publishes the result-card draft round record. WIDE/FULL publish 8-row views from the canonical
 * VIEWS area on every working card (WIDE only through layer20's front); their round offsets are NOWHERE.
 * Internal QUAD_ALL descriptions contain four actual CMPs, all40 EXL3 target layers on rank groups0..3,
 * and the full native head/draft on rank0. Each target executes in a four-member group; rank0 captures taps37..39.
 * Its target graphs coalesce at the same source-front boundary, with head/draft outside the companion range.
 * Public descriptions borrow the immutable four-card metadata. Slot4 stays NULL and is never a working card.
 * Caller must qualify direct peers before loading and pass its component/runtime gates before serving.
 * This factory establishes no numerical quality, speed, memory peak or concurrent-flight admission.
 *
 * Caller attaches the returned group map before graph preparation or steps. The driver must enqueue every group
 * member before peer waits and retire all cards before another flight uses its STEP/scratch/mailboxes.
 * One allocation owns the normal program, launches, replicated checkpoints and group ranges. Destroy through
 * model program_destroy; the companion descriptor then becomes invalid. Failure leaves both outputs untouched.
 * Preparation queues no model/GPU work; backend kernel_prepare only loads immutable kernel metadata.
 */
/* QUAD_ALL's default retains full-K EXL3 output shards and lossless gathers. An explicit
 * exact_output_shards=false + wire_format=BF16 instead selects512/640 inner shards, local FP32 expert sums,
 * sharded dense/shared work and BF16 partial sums. This changes arithmetic and is held for separate quality
 * gates. Its Stage1 expert products retain four parts. Rank0 owns the unsharded head and native draft.
 * Load and program description retain ordinary validation; main owns component and runtime gates. */
#ifndef MODEL_V41_TP2_PROGRAM_H
#define MODEL_V41_TP2_PROGRAM_H

#include "anemone/step.h"
#include "model_v41_tp2_stage.h"

#define V41_DATA_TP2_PROGRAM_CARDS 5u
#define V41_DATA_TP2_PAIR_A_LAYERS 14u
#define V41_DATA_TP2_PREFIX_LAYERS 29u
#define V41_DATA_TP2_PRO_CARD 4u
#define V41_DATA_TP2_PAIR_B_CARD 2u
#define V41_DATA_TP4_EXACT_RECIPE UINT64_C(0x4558414354545034)
#define V41_DATA_TP4_RING_IDENTITY (1ull << 50u)
#define V41_DATA_TP2_PROGRAM_RECIPE \
	0x5450325F46503330ull /* TP2 reference products, ordered FP32 rank sums */
#define V41_DATA_TP2_WIRE_IDENTITY_SHIFT 32u
#define V41_DATA_TP_LAYOUT_PAIRS 0u
#define V41_DATA_TP_LAYOUT_QUAD 1u
#define V41_DATA_TP_LAYOUT_HYBRID 2u
#define V41_DATA_TP_LAYOUT_QUAD_ALL \
	3u /* four-CMP EXL3 all40 + native head/draft on CMP0 */
#define V41_DATA_TP_LAYOUT_COUNT 3u /* native layouts; QUAD_ALL requires EXL3 */
#define V41_DATA_TP_LAYOUT_COLD_COUNT 4u
#define V41_DATA_TP_LAYOUT_IDENTITY_SHIFT 40u
#define V41_DATA_TP_WIDE_IDENTITY (1ull << 48u)
#define V41_DATA_TP4_ALIGNED_IDENTITY (1ull << 49u)
#define V41_DATA_TP4_ATTENTION_OWNER_IDENTITY (1ull << 51u)
#define V41_DATA_TP4_EXL3_IDENTITY (1ull << 52u)
#define V41_DATA_TP4_EXL3_ORDER_SHIFT 53u

struct v41_tp2_program_description {
	const struct model_plan_record *canonical;
	const struct model_plan_record *views[V41_DATA_TP2_PROGRAM_CARDS];
	const struct v41_tp2_stage_plan *stages[V41_DATA_TP2_PROGRAM_CARDS];
	uint32_t slice_count;
	uint32_t batch_verifiers; /* the next BATCH program's verifiers (draft walks); 0: rows / (draft positions + 1) */
};

MUST_CHECK result_status v41_tp2_program_prepare(
	const struct v41_tp2_program_description *description,
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct card_description_record *card_descriptions,
	uint32_t row_count, struct execution_program **program_out,
	struct step_companion_program *companions_out);

/* Wide/full variants retain the ordinary ordered state slabs and the same group/segment prefix. */
MUST_CHECK result_status
v41_tp_program_prepare(const struct v41_tp2_program_description *description,
		       const struct backend_operations *backend,
		       struct device_card *const *cards,
		       const struct card_description_record *card_descriptions,
		       enum program_kind kind, uint32_t row_count,
		       struct execution_program **program_out,
		       struct step_companion_program *companions_out);

#endif
