/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * mma_unit_probe: unit 3a of round 1 of the speed campaign (research/speed/ROUND_1.md, "mma_unit"; ledger M3): what
 * each tensor-core form of a card computes, one instruction at a time, against answers the host holds exactly (a
 * sum is a whole number of units of 2^e in 64 bits; the forms' codes keep it under 2^62 units). It times nothing.
 * Per form of the build (mma_unit_kernels.cuh lists them):
 *   layout    every element of A, then of B, alone against the other side's elements of its input: every sum must be
 *             the documented layout's (PTX ISA 9.7.16.5: with w elements to a register, the lowest bits first, lane
 *             4 q + t holds of A outputs q and q + 8, the second in the odd registers, at inputs w t .. w t + w - 1
 *             and, in registers 2 and 3, those plus K / 2; of B column q at the same inputs; of D outputs q, q + 8
 *             by columns 2 t, 2 t + 1).
 *   products  every pair of the form's codes alone in a sum. Codes: every pattern of an 8- or 4-bit type; f16: A an
 *             E2M1 nibble or an E4M3 byte as today's kernels place them (times 2^-14, 2^-8) or a whole number in a
 *             subnormal's mantissa (K-quant codes), B a placed E4M3 byte; bf16: A an E2M1 nibble times 2^-126 or a
 *             whole number times 2^-133, B a normal number of four significant bits in 16 binades (a carried
 *             activation). Counted apart: pairs with a subnormal code, with a byte whose padding bits are set, with
 *             a code that is no number (there: sums that are no NaN).
 *   sums      128 bases, each in eight orders of its inputs (term k at input ((2 o + 1) k + o) mod K; the last:
 *             K - 1 - k). Float forms: a term is the largest product, or one whose lowest bit lies 19 to 36 bits
 *             under that one's top (the widest the codes have, where they have less; of normal codes where those
 *             reach it), or zero, with signs of its own in every output and column; column 0 adds to zero, column c
 *             to an odd number of up to 3 c + 1 bits. Whole numbers: any pattern; accumulators at both ends of s32.
 *             A sum is `exact` (the magnitudes of its terms add up to at most 24 bits of their common unit, as unit
 *             10 found every group sum of the experts; whole numbers: to the largest s32: any order and rounding give
 *             one word), `inexact` (25 to 27 bits; beyond s32) or `wide` (28 and more). Per class: the first of
 *             unit_rules that gives the most of the card's words, and in how many (base, output, column) the eight
 *             orders gave different words: where that is 0 these inputs do not show the order of additions inside.
 *   columns   16 bases with each column alone (the others zero) against the same column among eight.
 *   scales    block-scaled forms, per selector kernel: every lane's scale bytes are powers of two of their own and a
 *             sum's exponent says whose it took; the codes 0, 1 by the largest codes, 254, 255 by the smallest normal.
 * And the two conversions: all 2^32 fp32 patterns as either operand (--steps N: N of the 256 chunks, each a sample of
 * the whole range) against the definition written out in integers.
 * Lines beside the harness's, one per form (a tally is <differing>/<of>, na of nothing):
 *   FORM card= arch= form= layout= products= subnormal_codes= padded_codes= no_number_codes= sum_exact= order_exact=
 *        sum_inexact= order_inexact= sum_wide= order_wide= columns= selectors= scale_bytes=
 *        sum_*: <rule>:<words not the rule's>/<of>. selectors: a<byte><pair>:<t>.<y>/<t>.<y> (the lane place and
 *        byte that scaled outputs q and q + 8), b<byte><thread>:<t>.<y> (the columns), ? where quads disagree.
 *        scale_bytes: <side><code>:2^n, or the sum's word where it is no power of two times the product.
 *   FORM card= arch= form=convert_* first_operand=upper_byte|lower_byte (by the first pair, 1 and 2), then per class
 *        of pattern (in_range, saturated, not_a_number, zero_or_negative) <class>=<tally> <class>_codes=<lowest>..
 *        <highest> the card gave; the definition gives a code, and the tally counts, in the first two classes only.
 * CHECKs are the layout and the products of normal codes of f16 alone, the reference point every exact kernel
 * rests on: every other figure is what the card does, and a run that prints it has succeeded (a form whose layout
 * differs has no other figure worth reading). Shown failing: --break unit (the host expects every sum one last place
 * off), --break layout (the host reads the other half's output), both with BROKEN for FORM: their lines are no
 * facts; --break-route (three launches past their arrays or block), --break-guard (the harness's). No --corrupt:
 * nothing is stored. No diagnostics record in a kernel: its one index is the host's. Not shown: speed; pairs of
 * other codes; a rule that is none of unit_rules (the line says how far the nearest is).
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "mma_unit_kernels.cuh"

enum {
    UNIT_QUADS = 8, UNIT_LANES_PER_QUAD = 4,     /* lane 4 q + t: quad q holds outputs q and q + 8 and feeds column q */
    UNIT_OUTPUTS = 16, UNIT_HALVES = 2, UNIT_COLUMNS = 8, UNIT_COLUMNS_PER_LANE = 2,
    UNIT_LINES = UNIT_OUTPUTS + UNIT_COLUMNS, UNIT_SUMS = UNIT_OUTPUTS * UNIT_COLUMNS, UNIT_SIDES = 2,
    UNIT_WORD_BITS = 32, UNIT_WORD_BYTES = 4, UNIT_LONG_BITS = 64, UNIT_BYTE_BITS = 8, UNIT_BYTE_VALUES = 256,
    UNIT_MOST_INPUTS = 64, UNIT_MOST_CASES = 5120, UNIT_MOST_LIST = 4096, UNIT_PATTERN_SETS = 3,
    UNIT_SINGLE_BITS = 24, UNIT_MOST_INEXACT_BITS = 27, UNIT_MOST_SHIFT = 62, UNIT_MOST_SPREAD = 40,
    UNIT_ORDERS = 8, UNIT_SUM_BASES = 128, UNIT_SUM_CASES = UNIT_SUM_BASES * UNIT_ORDERS, UNIT_COLUMN_BASES = 16,
    UNIT_TERM_DRAWS = 4, UNIT_ZERO_DRAW = 0, UNIT_LOW_DRAW = 1, UNIT_BITS_PER_COLUMN = 3,
    UNIT_HIGH_COLUMNS = 2, UNIT_LOW_COLUMNS = 4, /* whole numbers: accumulators under the top of s32, over its bottom */
    UNIT_MARGIN_SHIFT = 4,                       /* by at most a sixteenth of what K products can add up to */
    UNIT_NORMAL = 0, UNIT_SUBNORMAL, UNIT_PADDED, UNIT_NO_NUMBER, UNIT_KINDS,       /* of a code, the worst last */
    UNIT_EXACT = 0, UNIT_INEXACT, UNIT_WIDE, UNIT_BUCKETS,                           /* of a sum */
    UNIT_F16 = 0, UNIT_BF16, UNIT_E4M3, UNIT_E2M1, UNIT_E2M3, UNIT_S8, UNIT_U8, UNIT_S4, UNIT_U4,     /* unit_types */
    UNIT_NO_PATTERNS = 0, UNIT_EVERY_PATTERN, UNIT_F16_E2M1, UNIT_F16_E4M3, UNIT_F16_WHOLE, UNIT_BF16_E2M1,
    UNIT_BF16_WHOLE, UNIT_BF16_CARRIED,                                            /* unit_placements */
    UNIT_ONCE = 0, UNIT_PRODUCTS_FIRST, UNIT_ASCENDING, UNIT_DESCENDING,             /* the order of a rule */
    UNIT_NEAREST_EVEN = 0, UNIT_TOWARD_ZERO, UNIT_WRAPPING, UNIT_SATURATING,         /* its closing */
    UNIT_BROKEN_CASES = 0, UNIT_BROKEN_RESULTS, UNIT_BROKEN_BLOCK, UNIT_BROKEN_ROUTES, UNIT_WHOLE_RULES = 4,
    UNIT_SCALE_BIAS = 127, UNIT_OWN_SCALE_BASE = 63, UNIT_SOURCES_PER_QUAD = UNIT_LANES_PER_QUAD * UNIT_WORD_BYTES,
    UNIT_NO_SOURCE = UNIT_SOURCES_PER_QUAD, UNIT_MIXED_SOURCES, UNIT_SCALE_ENDS = 4, UNIT_STALE_BYTE = 0xFF,
    UNIT_CONVERT_BLOCK_THREADS = 256, UNIT_CONVERT_BLOCKS = 65536, UNIT_CONVERT_CHUNKS = 256,
    UNIT_CONVERT_CHUNK = UNIT_CONVERT_BLOCKS * UNIT_CONVERT_BLOCK_THREADS,
    UNIT_IN_RANGE = 0, UNIT_SATURATED, UNIT_NOT_A_NUMBER, UNIT_ZERO_OR_NEGATIVE, UNIT_GROUPS,    /* of a pattern */
    UNIT_SINGLE_FRACTION_BITS = 23, UNIT_E4M3_FIRST_NORMAL_FIELD = 121,      /* the fp32 exponent field of 2^-6 */
    UNIT_E4M3_DROPPED_BITS = 20, UNIT_E4M3_MOST_LIFT = 8,        /* of an fp32's 24 bits; a lift of 5 drops them all */
    UNIT_E4M3_CODES_PER_BINADE = 8, UNIT_E4M3_LARGEST = 126, UNIT_E4M3_SIGN = 0x80, UNIT_UE8M0_LARGEST = 254
};
#define UNIT_SCALE_ONE_WORD 0x7F7F7F7Fu          /* 2^0 in every byte: a side without scales */
#define UNIT_BYTE_REPEAT 0x01010101u
#define UNIT_OWN_SCALES_WORD 0x4241403Fu         /* lane 0: the codes 63, 64, 65, 66; a lane more: four more each */
#define UNIT_OWN_SCALES_STEP 0x04040404u
#define UNIT_SINGLE_SIGN 0x80000000u
#define UNIT_SINGLE_INFINITY 0x7F800000u
#define UNIT_SINGLE_HALF_FRACTION 0x00400000u
#define UNIT_SINGLE_IMPLICIT_ONE 0x00800000u
#define UNIT_FREXP_OF_A_POWER 0.5                /* what frexp returns of a power of two */
#define UNIT_RANDOM_MULTIPLIER 6364136223846793005ull    /* a 64-bit congruential generator: its upper half is used */
#define UNIT_RANDOM_INCREMENT 1442695040888963407ull

typedef void (*unit_mma_kernel)(const struct unit_lane_record *, uint32_t *);
typedef void (*unit_convert_kernel)(uint16_t *, uint32_t);
typedef int (*unit_reference)(uint32_t single, uint32_t *group);
struct unit_type {                               /* how the bits of an element are read */
    uint8_t container_bits, code_shift, code_bits, exponent_bits, fraction_bits, bias, is_signed, has_no_number;
};
static const struct unit_type unit_types[] = {
    {16, 0, 16, 5, 10, 15, 0, 0}, {16, 0, 16, 8, 7, 127, 0, 0},      /* f16, bf16 */
    {8, 0, 8, 4, 3, 7, 0, 1},                    /* e4m3: 0x7F and 0xFF are no number */
    {8, 2, 4, 2, 1, 1, 0, 0}, {8, 0, 6, 2, 3, 1, 0, 0},      /* e2m1: the central four bits; e2m3: the lower six */
    {8, 0, 8, 0, 0, 0, 1, 0}, {8, 0, 8, 0, 0, 0, 0, 0}, {4, 0, 4, 0, 0, 0, 1, 0}, {4, 0, 4, 0, 0, 0, 0, 0}};
/* Pattern i of a set: constant | (i & magnitude_mask) << magnitude_shift | (i & sign_mask) << sign_shift. */
static const struct unit_placement { uint32_t count, constant, magnitude_mask, magnitude_shift, sign_mask, sign_shift; }
unit_placements[] = {
    {0, 0, 0, 0, 0, 0}, {0, 0, 0xFFFF, 0, 0, 0}, /* none; every pattern of the type: the count is the type's */
    {16, 0, 7, 9, 8, 12},                        /* an E2M1 nibble as fp16: its value times 2^-14 */
    {256, 0, 0x7F, 7, 0x80, 8},                  /* an E4M3 byte as fp16: its value times 2^-8 */
    {2048, 0, 0x3FF, 0, 0x400, 5},               /* a whole number in an fp16 subnormal: times 2^-24 */
    {16, 0, 7, 6, 8, 12},                        /* an E2M1 nibble as bf16: its value times 2^-126 */
    {256, 0, 0x7F, 0, 0x80, 8},                  /* a whole number in a bf16 subnormal: times 2^-133 */
    {256, 0x6800, 0x7F, 4, 0x80, 8}};            /* a normal bf16 of four significant bits, 2^81 to under 2^97 */
static const struct unit_rule { const char *name; uint8_t order, closing; } unit_rules[] = {
    {"exact_sum_to_nearest_even", UNIT_ONCE, UNIT_NEAREST_EVEN}, {"exact_sum_toward_zero", UNIT_ONCE, UNIT_TOWARD_ZERO},
    {"products_then_accumulator_to_nearest_even", UNIT_PRODUCTS_FIRST, UNIT_NEAREST_EVEN},
    {"products_then_accumulator_toward_zero", UNIT_PRODUCTS_FIRST, UNIT_TOWARD_ZERO},
    {"chain_ascending_to_nearest_even", UNIT_ASCENDING, UNIT_NEAREST_EVEN},
    {"chain_ascending_toward_zero", UNIT_ASCENDING, UNIT_TOWARD_ZERO},
    {"chain_descending_to_nearest_even", UNIT_DESCENDING, UNIT_NEAREST_EVEN},
    {"chain_descending_toward_zero", UNIT_DESCENDING, UNIT_TOWARD_ZERO},
    {"wrapping", UNIT_ONCE, UNIT_WRAPPING}, {"saturating_once", UNIT_ONCE, UNIT_SATURATING},     /* whole numbers: */
    {"saturating_chain_ascending", UNIT_ASCENDING, UNIT_SATURATING},                             /* the last four */
    {"saturating_chain_descending", UNIT_DESCENDING, UNIT_SATURATING}};
enum { UNIT_RULES = PROBE_COUNT_OF(unit_rules), UNIT_FLOAT_RULES = UNIT_RULES - UNIT_WHOLE_RULES };
static const uint8_t unit_spreads[] = {19, 20, 21, 22, 23, 24, 30, 36};  /* bits from a low term to the largest's top */
static const char *const unit_bucket_names[UNIT_BUCKETS] = {"exact", "inexact", "wide"};
struct unit_selector { uint8_t a_byte, a_pair, b_byte, b_thread; unit_mma_kernel kernel; };
struct unit_form {
    const char *name;
    unit_mma_kernel kernel;
    uint8_t a_type, b_type, inputs;              /* unit_types; the K of the shape */
    uint8_t a_sets[UNIT_PATTERN_SETS], b_sets[UNIT_PATTERN_SETS];     /* unit_placements: the codes of each side */
    const struct unit_selector *selectors;       /* a block-scaled form: its UNIT_SELECTOR_COUNT kernels */
};
struct unit_tally { uint64_t differing, of; };
#define UNIT_TALLY(tally) probe_pairs((uint32_t)(tally).differing, (uint32_t)(tally).of)
#define UNIT_LINE (probe_flag_given("--break") ? "BROKEN" : "FORM")      /* a line of --break is no fact of the card */
struct unit_value { int64_t mantissa; int32_t exponent; uint32_t kind; };      /* mantissa times 2^exponent */

static struct {
    struct unit_lane_record *card_records;
    uint32_t *card_results;
    uint16_t *card_converted;
    uint32_t list[UNIT_SIDES][UNIT_MOST_LIST], count[UNIT_SIDES];       /* the form's patterns, A's then B's */
    uint32_t extreme[UNIT_SIDES][UNIT_SIDES];    /* [0]: each list's largest number; [1]: its smallest normal one */
    uint32_t low[UNIT_MOST_SPREAD][UNIT_SIDES];  /* per spread a pair whose product's lowest bit lies there; 0: none */
    int32_t exponent, largest_top;               /* the sums' unit is 2^exponent; the top bit of the largest product */
    uint32_t wrong_last_place;                   /* --break unit */
    uint64_t random_state;
} unit;
static struct unit_lane_record unit_records[UNIT_MOST_CASES * UNIT_LANES];
static uint32_t unit_results[UNIT_MOST_CASES * UNIT_CASE_SUMS];

/* ---- what the bits of an operand stand for ---- */
static struct unit_value unit_value_of(uint32_t type_index, uint32_t pattern)
{
    const struct unit_type *type = &unit_types[type_index];
    uint32_t code_mask = (1u << type->code_bits) - 1u, code = pattern >> type->code_shift & code_mask;
    uint32_t sign = code >> (type->code_bits - 1u), fraction_mask = (1u << type->fraction_bits) - 1u;
    uint32_t field = code >> type->fraction_bits & ((1u << type->exponent_bits) - 1u), fraction = code & fraction_mask;
    struct unit_value value = {code, 0, UNIT_NORMAL};
    if (type->exponent_bits == 0) {              /* a whole number: two's complement where the type has a sign */
        value.mantissa -= type->is_signed && sign ? (int64_t)code_mask + 1 : 0;
        return value;
    }
    value.mantissa = field == 0 ? fraction : fraction + fraction_mask + 1u;     /* a subnormal has no implicit one */
    value.mantissa = sign ? -value.mantissa : value.mantissa;
    value.exponent = (int32_t)(field == 0 ? 1u : field) - type->bias - type->fraction_bits;
    value.kind = (pattern & ~(code_mask << type->code_shift)) != 0 ? UNIT_PADDED
                 : type->has_no_number && (code & code_mask >> 1) == code_mask >> 1 ? UNIT_NO_NUMBER
                 : field == 0 && fraction != 0 ? UNIT_SUBNORMAL : UNIT_NORMAL;
    return value;
}
static uint32_t unit_is_whole(const struct unit_form *form) { return unit_types[form->a_type].exponent_bits == 0; }
static int unit_top_bit(uint64_t magnitude) { return UNIT_LONG_BITS - 1 - __builtin_clzll(magnitude); }
static uint32_t unit_random(void)                /* a fixed sequence: a base is the same in every order and run */
{
    unit.random_state = unit.random_state * UNIT_RANDOM_MULTIPLIER + UNIT_RANDOM_INCREMENT;
    return (uint32_t)(unit.random_state >> UNIT_WORD_BITS);
}

/* The form's lists of patterns, each padded with zeros to whole cases of pairs; per side the largest number and the
 * smallest positive normal one; the unit of every sum; per spread a pair of patterns for the low terms of the sums. */
static void unit_list_patterns(const struct unit_form *form)
{
    struct unit_value a, b;
    memset(unit.low, 0, sizeof(unit.low));
    unit.exponent = 0;
    for (uint32_t side = 0; side < UNIT_SIDES; side++) {
        uint32_t type = side ? form->b_type : form->a_type, *list = unit.list[side], count = 0;
        double most = 0.0, least = INFINITY;
        int32_t least_exponent = INT32_MAX;
        for (uint32_t set = 0; set < UNIT_PATTERN_SETS; set++) {
            uint32_t set_index = side ? form->b_sets[set] : form->a_sets[set];
            const struct unit_placement *placed = &unit_placements[set_index];
            uint32_t every = set_index == UNIT_EVERY_PATTERN ? 1u << unit_types[type].container_bits : 0;
            PROBE_REQUIRE(count + placed->count + every <= UNIT_MOST_LIST, "a form's codes are too many for its list");
            for (uint32_t index = 0; index < placed->count + every; index++, count++) {
                list[count] = placed->constant | (index & placed->magnitude_mask) << placed->magnitude_shift
                              | (index & placed->sign_mask) << placed->sign_shift;
                a = unit_value_of(type, list[count]);
                double amount = ldexp((double)a.mantissa, a.exponent), normal = a.kind == UNIT_NORMAL ? amount : 0.0;
                if (a.kind > UNIT_SUBNORMAL) { continue; }               /* padded, or no number: in no sum */
                least_exponent = a.exponent < least_exponent ? a.exponent : least_exponent;
                if (amount > most) { most = amount; unit.extreme[0][side] = list[count]; }
                if (normal > 0.0 && normal < least) { least = normal; unit.extreme[1][side] = list[count]; }
            }
        }
        while (count % (side ? UNIT_COLUMNS : UNIT_OUTPUTS) != 0) { list[count++] = 0; }
        unit.count[side] = count;
        unit.exponent += least_exponent;
    }
    PROBE_REQUIRE(unit.count[0] / UNIT_OUTPUTS * (unit.count[1] / UNIT_COLUMNS) <= UNIT_MOST_CASES, "too many pairs");
    a = unit_value_of(form->a_type, unit.extreme[0][0]);
    b = unit_value_of(form->b_type, unit.extreme[0][1]);
    unit.largest_top = a.exponent + b.exponent + unit_top_bit((uint64_t)(a.mantissa * b.mantissa));
    /* Low terms: pass 0 takes normal codes, pass 1 subnormal ones too: no sum rests on a subnormal code needlessly. */
    for (uint32_t pair = 0; !unit_is_whole(form) && pair < UNIT_PADDED * unit.count[0] * unit.count[1]; pair++) {
        uint32_t worst = pair / unit.count[1] / unit.count[0], a_index = pair / unit.count[1] % unit.count[0];
        const uint32_t patterns[UNIT_SIDES] = {unit.list[0][a_index], unit.list[1][pair % unit.count[1]]};
        a = unit_value_of(form->a_type, patterns[0]);
        b = unit_value_of(form->b_type, patterns[1]);
        uint64_t product = (uint64_t)llabs(a.mantissa * b.mantissa);     /* never above the largest product */
        int32_t spread = product != 0 ? unit.largest_top - (a.exponent + b.exponent + __builtin_ctzll(product)) + 1 : 0;
        if (a.kind <= worst && b.kind <= worst && spread > 0 && spread < UNIT_MOST_SPREAD
            && unit.low[spread][0] == 0) {
            memcpy(unit.low[spread], patterns, sizeof(patterns));
        }
    }
}

/* ---- the documented layout: where an element lies in a case's records and results ---- */
/* The record word and the shift of element (line, input) of A (side 0: line is an output) or of B (a column). */
static uint32_t *unit_operand(const struct unit_form *form, uint32_t side, uint64_t unit_case, uint32_t line,
                              uint32_t input, uint32_t *shift)
{
    uint32_t bits = unit_types[form->a_type].container_bits, per_register = UNIT_WORD_BITS / bits;
    uint32_t span = UNIT_LANES_PER_QUAD * per_register;      /* the inputs a register holds, over the lanes of a quad */
    struct unit_lane_record *lane = unit_records;
    PROBE_INDEX_CHECK(unit_case, UNIT_MOST_CASES);
    lane += unit_case * UNIT_LANES + line % UNIT_QUADS * UNIT_LANES_PER_QUAD + input % span / per_register;
    *shift = input % per_register * bits;
    return side ? &lane->b[input / span] : &lane->a[input / span * UNIT_HALVES + line / UNIT_QUADS];
}
static void unit_put(const struct unit_form *form, uint32_t side, uint64_t unit_case, uint32_t line, uint32_t input,
                     uint32_t pattern)
{
    uint32_t shift, *word = unit_operand(form, side, unit_case, line, input, &shift);
    *word |= pattern << shift;
}
static struct unit_value unit_get(const struct unit_form *form, uint32_t side, uint64_t unit_case, uint32_t line,
                                  uint32_t input)
{
    uint32_t shift, word = *unit_operand(form, side, unit_case, line, input, &shift);
    uint32_t type = side ? form->b_type : form->a_type;
    return unit_value_of(type, word >> shift & ((1u << unit_types[type].container_bits) - 1u));
}
static uint32_t unit_sum_at(uint32_t output, uint32_t column)        /* among the UNIT_CASE_SUMS words of a case */
{
    return (output % UNIT_QUADS * UNIT_LANES_PER_QUAD + column / UNIT_COLUMNS_PER_LANE) * UNIT_SUMS_PER_LANE
           + output / UNIT_QUADS * UNIT_COLUMNS_PER_LANE + column % UNIT_COLUMNS_PER_LANE;
}
static uint32_t *unit_accumulator(uint64_t unit_case, uint32_t output, uint32_t column)
{
    uint32_t at = unit_sum_at(output, column);
    return &unit_records[unit_case * UNIT_LANES + at / UNIT_SUMS_PER_LANE].c[at % UNIT_SUMS_PER_LANE];
}

/* The terms of sum (output, column) of a case as its records hold them: the product at every input and the
 * accumulator, in units of 2^unit.exponent; returns the worst kind among its codes. */
static uint32_t unit_terms(const struct unit_form *form, uint64_t unit_case, uint32_t output, uint32_t column,
                           int64_t *products, int64_t *accumulator)
{
    uint32_t kind = UNIT_NORMAL;
    float single;
    for (uint32_t input = 0; input < form->inputs; input++) {
        struct unit_value a = unit_get(form, 0, unit_case, output, input);
        struct unit_value b = unit_get(form, 1, unit_case, column, input);
        int64_t product = a.mantissa * b.mantissa;
        int32_t raised = a.exponent + b.exponent - unit.exponent;
        PROBE_REQUIRE(product == 0 || (raised >= 0 && raised + unit_top_bit(llabs(product)) < UNIT_MOST_SHIFT),
                      "an operand outside the form's lists");
        products[input] = product == 0 ? 0 : product * ((int64_t)1 << raised);
        kind = a.kind > kind ? a.kind : kind;
        kind = b.kind > kind ? b.kind : kind;
    }
    memcpy(&single, unit_accumulator(unit_case, output, column), sizeof(single));
    *accumulator = unit_is_whole(form) ? (int32_t)*unit_accumulator(unit_case, output, column)
                                       : (int64_t)ldexp((double)single, -unit.exponent);
    return kind;
}

/* ---- the rules a sum may follow ---- */
/* A sum closed as a rule closes it: wrapped or limited to s32, or rounded to the 24 bits of an fp32. */
static int64_t unit_closed(int64_t sum, uint32_t closing)
{
    uint64_t magnitude = sum < 0 ? 0u - (uint64_t)sum : (uint64_t)sum;
    int dropped = magnitude != 0 ? unit_top_bit(magnitude) - (UNIT_SINGLE_BITS - 1) : 0;
    if (closing == UNIT_WRAPPING) { return (int32_t)(uint32_t)(uint64_t)sum; }
    if (closing == UNIT_SATURATING) { return sum > INT32_MAX ? INT32_MAX : sum < INT32_MIN ? INT32_MIN : sum; }
    if (dropped <= 0) { return sum; }
    uint64_t kept = magnitude >> dropped, rest = magnitude & ((1ull << dropped) - 1u), half = 1ull << (dropped - 1);
    kept += closing == UNIT_NEAREST_EVEN && (rest > half || (rest == half && (kept & 1u) != 0));
    return sum < 0 ? -(int64_t)(kept << dropped) : (int64_t)(kept << dropped);
}

/* The word a rule makes of the products, in the order of the instruction's inputs, and the accumulator. */
static uint32_t unit_ruled(const struct unit_form *form, uint32_t rule_index, const int64_t *products,
                           int64_t accumulator)
{
    const struct unit_rule *rule = &unit_rules[rule_index];
    int64_t sum = rule->order == UNIT_PRODUCTS_FIRST ? 0 : accumulator;
    float single;
    for (uint32_t step = 0; step < form->inputs; step++) {
        sum += products[rule->order == UNIT_DESCENDING ? form->inputs - 1u - step : step];
        sum = rule->order >= UNIT_ASCENDING ? unit_closed(sum, rule->closing) : sum;
    }
    sum = rule->order == UNIT_PRODUCTS_FIRST ? unit_closed(sum, rule->closing) + accumulator : sum;
    sum = unit_closed(sum, rule->closing);
    uint32_t word = (uint32_t)(uint64_t)sum;
    single = (float)ldexp((double)sum, unit.exponent);
    if (!unit_is_whole(form)) { memcpy(&word, &single, sizeof(word)); }
    return word ^ unit.wrong_last_place;         /* --break unit: one last place off */
}

/* ---- launches ---- */
/* Check 10: case_count blocks of block_threads threads stay inside the two card arrays and the card's limits. */
static int unit_launch_is_valid(const char *kernel, const void *results, uint64_t case_count, uint64_t block_threads)
{
    int valid = 1;
    probe_launch_geometry(&valid, kernel, NULL, case_count, 1, 1, block_threads, 0);
    probe_launch_value(&valid, kernel, "threads of a block: one warp", block_threads, UNIT_LANES, UNIT_LANES);
    probe_launch_range(&valid, kernel, "records, in cases", 0, case_count, UNIT_LANES * sizeof(struct unit_lane_record),
                       probe_cuda_array_bytes(unit.card_records));
    probe_launch_range(&valid, kernel, "results, in cases", 0, case_count, UNIT_CASE_SUMS * sizeof(uint32_t),
                       probe_cuda_array_bytes(results));
    return valid;
}
static void unit_clear(uint64_t case_count)      /* records of nothing: zeros, and 2^0 in every scale byte */
{
    memset(unit_records, 0, case_count * UNIT_LANES * sizeof(struct unit_lane_record));
    for (uint64_t lane = 0; lane < case_count * UNIT_LANES; lane++) {
        unit_records[lane].scale_a = unit_records[lane].scale_b = UNIT_SCALE_ONE_WORD;
    }
}

/* The first case_count cases of unit_records through a kernel: unit_results gets what the card returned. */
static void unit_run(const struct unit_form *form, unit_mma_kernel kernel, uint64_t case_count)
{
    size_t record_bytes = case_count * UNIT_LANES * sizeof(struct unit_lane_record);
    size_t result_bytes = case_count * UNIT_CASE_SUMS * sizeof(uint32_t);
    PROBE_REQUIRE(unit_launch_is_valid(form->name, unit.card_results, case_count, UNIT_LANES), "a launch was refused");
    PROBE_CHECK_CUDA(cudaMemcpy(unit.card_records, unit_records, record_bytes, cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemset(unit.card_results, UNIT_STALE_BYTE, result_bytes));
    kernel<<<(unsigned int)case_count, UNIT_LANES>>>(unit.card_records, unit.card_results);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_AFTER_LAUNCH();
    PROBE_CHECK_CUDA(cudaMemcpy(unit_results, unit.card_results, result_bytes, cudaMemcpyDeviceToHost));
    probe_digest_add(unit_results, result_bytes);
}

/* ---- layout and products: every sum of the cases against its exact value ---- */
/* The cases were run. half_flip (--break layout): the card's word is read from the other half's output. */
static void unit_judge(const struct unit_form *form, uint64_t cases, uint32_t half_flip, struct unit_tally *tallies)
{
    for (uint64_t sum = 0; sum < cases * UNIT_SUMS; sum++) {
        uint32_t output = sum / UNIT_COLUMNS % UNIT_OUTPUTS, column = sum % UNIT_COLUMNS;
        int64_t products[UNIT_MOST_INPUTS], accumulator;
        uint32_t kind = unit_terms(form, sum / UNIT_SUMS, output, column, products, &accumulator);
        uint32_t word = unit_results[sum / UNIT_SUMS * UNIT_CASE_SUMS + unit_sum_at(output ^ half_flip, column)];
        uint32_t wrong = word != unit_ruled(form, unit_is_whole(form) * UNIT_FLOAT_RULES, products, accumulator);
        tallies[kind].of++;                      /* a code that is no number: the sum must be a NaN */
        tallies[kind].differing += kind == UNIT_NO_NUMBER ? (word & ~UNIT_SINGLE_SIGN) <= UNIT_SINGLE_INFINITY : wrong;
    }
}

/* Every element of A, then of B, alone against the other side's elements of its input: one product a sum. */
static uint64_t unit_build_layout(const struct unit_form *form)
{
    uint64_t unit_case = 0;
    unit_clear(UNIT_MOST_CASES);
    for (; unit_case < (uint64_t)UNIT_LINES * form->inputs; unit_case++) {
        uint32_t line = (uint32_t)unit_case / form->inputs, input = (uint32_t)unit_case % form->inputs;
        uint32_t side = line >= UNIT_OUTPUTS;
        unit_put(form, side, unit_case, line % UNIT_OUTPUTS, input, unit.extreme[0][side]);
        for (uint32_t other = 0; other < (side ? UNIT_OUTPUTS : UNIT_COLUMNS); other++) {
            unit_put(form, !side, unit_case, other, input, unit.extreme[0][!side]);
        }
    }
    return unit_case;
}

/* Sixteen codes of A (a code an output) by eight of B (a code a column) a case, at an input that moves on. */
static uint64_t unit_build_pairs(const struct unit_form *form)
{
    uint64_t unit_case = 0;
    unit_clear(UNIT_MOST_CASES);
    for (uint32_t a_first = 0; a_first < unit.count[0]; a_first += UNIT_OUTPUTS) {
        for (uint32_t b_first = 0; b_first < unit.count[1]; b_first += UNIT_COLUMNS, unit_case++) {
            for (uint32_t line = 0; line < UNIT_LINES; line++) {
                uint32_t side = line >= UNIT_OUTPUTS, at = line % UNIT_OUTPUTS;
                unit_put(form, side, unit_case, at, (uint32_t)unit_case % form->inputs,
                         unit.list[side][(side ? b_first : a_first) + at]);
            }
        }
    }
    return unit_case;
}

/* ---- sums ---- */
/* Base `base` of the form's sums into a case, its terms at the inputs of order `order`. only_column under
 * UNIT_COLUMNS: the other columns of B and their accumulators stay zero. Every draw is made whatever is stored. */
static void unit_build_sum(const struct unit_form *form, uint64_t unit_case, uint32_t base, uint32_t order,
                           uint32_t only_column)
{
    const struct unit_type *a_type = &unit_types[form->a_type], *b_type = &unit_types[form->b_type];
    uint32_t signs[UNIT_SIDES] = {1u << (a_type->code_shift + a_type->code_bits - 1u),
                                  1u << (b_type->code_shift + b_type->code_bits - 1u)};
    uint32_t reversed = order == UNIT_ORDERS - 1u, multiplier = reversed ? form->inputs - 1u : order + order + 1u;
    uint32_t spread = unit_spreads[base % PROBE_COUNT_OF(unit_spreads)], mask = (1u << a_type->container_bits) - 1u;
    uint32_t whole = unit_is_whole(form), all_bits = a_type->container_bits + b_type->container_bits;
    uint64_t margin = ((uint64_t)form->inputs << all_bits) >> UNIT_MARGIN_SHIFT;
    while (!whole && spread > 1 && unit.low[spread][0] == 0) { spread--; }       /* the widest the codes have */
    unit.random_state = ((uint64_t)form->a_type * UNIT_MOST_INPUTS + form->inputs) * UNIT_SUM_BASES + base;
    for (uint32_t term = 0; term < form->inputs; term++) {       /* term 0: the largest product; term 1: a low one */
        uint32_t input = ((term + reversed) * multiplier + order * !reversed) % form->inputs;     /* or K - 1 - term */
        uint32_t draw = unit_random() % UNIT_TERM_DRAWS, is_low = term == 1 || (term > 1 && draw == UNIT_LOW_DRAW);
        for (uint32_t line = 0; line < UNIT_LINES; line++) {
            uint32_t side = line >= UNIT_OUTPUTS, flip = unit_random();
            uint32_t pattern = is_low ? unit.low[spread][side] : unit.extreme[0][side];
            pattern = term > 1 && draw == UNIT_ZERO_DRAW ? 0 : pattern ^ (flip & 1u ? signs[side] : 0);
            if (!side || only_column >= UNIT_COLUMNS || line - UNIT_OUTPUTS == only_column) {
                unit_put(form, side, unit_case, line % UNIT_OUTPUTS, input, whole ? flip & mask : pattern);
            }
        }
    }
    for (uint32_t sum = 0; sum < UNIT_SUMS; sum++) {
        uint32_t column = sum % UNIT_COLUMNS, bits = column * UNIT_BITS_PER_COLUMN, first = unit_random();
        uint32_t second = unit_random(), room = spread + UNIT_BITS_PER_COLUMN - (bits < spread ? bits : spread);
        int32_t lowest = unit.largest_top - (int32_t)spread + 1 + (int32_t)(second % room);
        float single = (float)ldexp((double)(first >> (UNIT_WORD_BITS - 1u - bits) | 1u), lowest);
        uint32_t word = column <= UNIT_HIGH_COLUMNS ? (uint32_t)INT32_MAX - (uint32_t)(first % margin)
                        : column <= UNIT_LOW_COLUMNS ? (uint32_t)INT32_MIN + (uint32_t)(first % margin) : first;
        single = second >> (UNIT_WORD_BITS - 1u) != 0 ? -single : single;
        if (!whole) { memcpy(&word, &single, sizeof(word)); }
        if (column != 0 && (only_column >= UNIT_COLUMNS || column == only_column)) {
            *unit_accumulator(unit_case, sum / UNIT_COLUMNS, column) = word;
        }
    }
}

/* The cases of the sums: every base in every order, then UNIT_COLUMN_BASES bases with each column alone. */
static uint64_t unit_build_sums(const struct unit_form *form)
{
    uint32_t unit_case = 0;
    unit_clear(UNIT_MOST_CASES);
    for (; unit_case < UNIT_SUM_CASES; unit_case++) {
        unit_build_sum(form, unit_case, unit_case / UNIT_ORDERS, unit_case % UNIT_ORDERS, UNIT_COLUMNS);
    }
    for (uint32_t alone = 0; alone < UNIT_COLUMN_BASES * UNIT_COLUMNS; alone++, unit_case++) {
        unit_build_sum(form, unit_case, alone / UNIT_COLUMNS, 0, alone % UNIT_COLUMNS);
    }
    return unit_case;
}

/* Prints the sums' fields of the FORM line: per class of (base, output, column) the first rule that gives the most
 * of the card's words over the eight orders, and the sums an order moved; then the columns alone. */
static void unit_judge_sums(const struct unit_form *form)
{
    uint64_t of[UNIT_BUCKETS] = {0}, moved[UNIT_BUCKETS] = {0}, matches[UNIT_BUCKETS][UNIT_RULES] = {{0}};
    uint32_t whole = unit_is_whole(form), first = whole * UNIT_FLOAT_RULES, end = whole ? UNIT_RULES : UNIT_FLOAT_RULES;
    struct unit_tally columns = {0, 0};
    for (uint32_t sum = 0; sum < UNIT_SUM_BASES * UNIT_SUMS; sum++) {
        uint32_t base = sum / UNIT_SUMS, column = sum % UNIT_COLUMNS, output = sum / UNIT_COLUMNS % UNIT_OUTPUTS;
        uint32_t at = unit_sum_at(output, column), alone = UNIT_SUM_CASES + base * UNIT_COLUMNS + column;
        uint32_t among = unit_results[base * UNIT_ORDERS * UNIT_CASE_SUMS + at], bucket, bits, differs = 0;
        uint64_t magnitudes = 0, every = 0;
        int64_t products[UNIT_MOST_INPUTS], accumulator;
        unit_terms(form, base * UNIT_ORDERS, output, column, products, &accumulator);
        for (uint32_t term = 0; term <= form->inputs; term++) {      /* the last: the accumulator */
            uint64_t magnitude = (uint64_t)llabs(term < form->inputs ? products[term] : accumulator);
            magnitudes += magnitude;
            every |= magnitude;
        }
        bits = every != 0 ? (uint32_t)unit_top_bit(magnitudes >> __builtin_ctzll(every)) + 1u : 0;   /* of their unit */
        bucket = (whole ? magnitudes <= (uint64_t)INT32_MAX : bits <= UNIT_SINGLE_BITS) ? UNIT_EXACT
                 : whole || bits <= UNIT_MOST_INEXACT_BITS ? UNIT_INEXACT : UNIT_WIDE;
        for (uint32_t order = 0; order < UNIT_ORDERS; order++) {
            uint32_t word = unit_results[(base * UNIT_ORDERS + order) * UNIT_CASE_SUMS + at];
            unit_terms(form, base * UNIT_ORDERS + order, output, column, products, &accumulator);
            differs |= word != among;
            for (uint32_t rule = first; rule < end; rule++) {            /* the rules of the form's kind */
                matches[bucket][rule] += word == unit_ruled(form, rule, products, accumulator);
            }
        }
        of[bucket]++;
        moved[bucket] += differs;
        columns.of += base < UNIT_COLUMN_BASES;
        columns.differing += base < UNIT_COLUMN_BASES && among != unit_results[alone * UNIT_CASE_SUMS + at];
    }
    for (uint32_t bucket = 0; bucket < UNIT_BUCKETS; bucket++) {
        uint32_t best = first, words = (uint32_t)of[bucket] * UNIT_ORDERS;
        for (uint32_t rule = first; rule < end; rule++) {
            best = matches[bucket][rule] > matches[bucket][best] ? rule : best;
        }
        printf(" sum_%s=%s:%s order_%s=%s", unit_bucket_names[bucket], words != 0 ? unit_rules[best].name : "none",
               probe_pairs(words - (uint32_t)matches[bucket][best], words), unit_bucket_names[bucket],
               probe_pairs((uint32_t)moved[bucket], (uint32_t)of[bucket]));
    }
    printf(" columns=%s", UNIT_TALLY(columns));
}

/* ---- block-scaled forms and conversions: sm_120a ---- */
#if PROBE_SM == UNIT_SM_120A
#define UNIT_SELECTOR_ROW(form, spelling, a_byte, a_pair, b_byte, b_thread)                                           \
    {a_byte, a_pair, b_byte, b_thread, unit_kernel_##form##_a##a_byte##a_pair##_b##b_byte##b_thread},
static const struct unit_selector unit_e2m1_scales[UNIT_SELECTOR_COUNT] = {
    UNIT_EVERY_SELECTOR(UNIT_SELECTOR_ROW, e2m1_e4m3_scaled, "")};
static const struct unit_selector unit_e4m3_scales[UNIT_SELECTOR_COUNT] = {
    UNIT_EVERY_SELECTOR(UNIT_SELECTOR_ROW, e4m3_scaled, "")};

/* 1 and the power: the fp32 of a word is the product of two patterns times a power of two. */
static int unit_power_of(const struct unit_form *form, const uint32_t *patterns, uint32_t word, int *power)
{
    struct unit_value a = unit_value_of(form->a_type, patterns[0]), b = unit_value_of(form->b_type, patterns[1]);
    float single;
    memcpy(&single, &word, sizeof(single));
    double fraction = frexp((double)single / ldexp((double)(a.mantissa * b.mantissa), a.exponent + b.exponent), power);
    *power -= 1;
    return fraction == UNIT_FREXP_OF_A_POWER;
}
/* A case of one product a sum, at input 0; the scale word of every lane on `side` is word + lane * step. */
static void unit_scale_case(const struct unit_form *form, uint32_t unit_case, uint32_t side, uint32_t word,
                            uint32_t step, const uint32_t *patterns)
{
    for (uint32_t lane = 0; lane < UNIT_LANES; lane++) {
        struct unit_lane_record *record = &unit_records[unit_case * UNIT_LANES + lane];
        *(side ? &record->scale_b : &record->scale_a) = word + lane * step;
        if (lane < UNIT_OUTPUTS) { unit_put(form, 0, unit_case, lane, 0, patterns[0]); }
        if (lane < UNIT_COLUMNS) { unit_put(form, 1, unit_case, lane, 0, patterns[1]); }
    }
}

/* Whose scale byte the sums of case `side` show for an output half (side 0) or for the columns (side 1), every
 * lane's bytes being powers of their own: "<lane place>.<byte>" where each quad took that byte of its own lanes. */
static void unit_print_source(const struct unit_form *form, uint32_t side, uint32_t half, const char *before)
{
    uint32_t found = UNIT_NO_SOURCE;
    for (uint32_t sum = 0; sum < UNIT_QUADS * UNIT_COLUMNS; sum++) {
        uint32_t quad = sum / UNIT_COLUMNS, other = sum % UNIT_COLUMNS, place = UNIT_MIXED_SOURCES;
        uint32_t output = (side ? other : quad) + half * UNIT_QUADS, at = unit_sum_at(output, side ? quad : other);
        int power = 0, fits = unit_power_of(form, unit.extreme[0], unit_results[side * UNIT_CASE_SUMS + at], &power);
        uint32_t source = (uint32_t)(power + UNIT_SCALE_BIAS - UNIT_OWN_SCALE_BASE);         /* 4 lane + byte */
        if (fits && source / UNIT_SOURCES_PER_QUAD == quad) { place = source % UNIT_SOURCES_PER_QUAD; }
        if (found != UNIT_NO_SOURCE && found != place) { place = UNIT_MIXED_SOURCES; }
        found = place;
    }
    if (found < UNIT_NO_SOURCE) { printf("%s%u.%u", before, found / UNIT_WORD_BYTES, found % UNIT_WORD_BYTES); }
    if (found >= UNIT_NO_SOURCE) { printf("%s?", before); }
}

static void unit_check_scales(const struct unit_form *form)          /* prints the last two fields of the FORM line */
{
    static const uint8_t ends[UNIT_SCALE_ENDS] = {0, 1, 254, 255};
    unit_clear(UNIT_SIDES);                      /* every selector's input: case 0: A's scale bytes their own; 1: B's */
    unit_scale_case(form, 0, 0, UNIT_OWN_SCALES_WORD, UNIT_OWN_SCALES_STEP, unit.extreme[0]);
    unit_scale_case(form, 1, 1, UNIT_OWN_SCALES_WORD, UNIT_OWN_SCALES_STEP, unit.extreme[0]);
    printf(" selectors=");
    for (uint32_t entry = 0; entry < UNIT_SELECTOR_COUNT; entry++) {
        const struct unit_selector *selector = &form->selectors[entry];
        unit_run(form, selector->kernel, UNIT_SIDES);
        if (selector->b_byte + selector->b_thread == 0) {
            printf("%sa%u%u:", entry != 0 ? "," : "", selector->a_byte, selector->a_pair);
            unit_print_source(form, 0, 0, "");
            unit_print_source(form, 0, 1, "/");
        }
        if (selector->a_byte + selector->a_pair == 0) {
            printf(",b%u%u:", selector->b_byte, selector->b_thread);
            unit_print_source(form, 1, 0, "");
        }
    }
    unit_clear(UNIT_SIDES * UNIT_SCALE_ENDS);
    for (uint32_t unit_case = 0; unit_case < UNIT_SIDES * UNIT_SCALE_ENDS; unit_case++) {
        uint32_t code = ends[unit_case / UNIT_SIDES];            /* a small scale by large operands, a large by small */
        unit_scale_case(form, unit_case, unit_case % UNIT_SIDES, code * UNIT_BYTE_REPEAT, 0,
                        unit.extreme[code > UNIT_SCALE_BIAS]);
    }
    unit_run(form, form->kernel, UNIT_SIDES * UNIT_SCALE_ENDS);
    printf(" scale_bytes=");
    for (uint32_t unit_case = 0; unit_case < UNIT_SIDES * UNIT_SCALE_ENDS; unit_case++) {
        uint32_t code = ends[unit_case / UNIT_SIDES], word = unit_results[unit_case * UNIT_CASE_SUMS];
        int power = 0, fits = unit_power_of(form, unit.extreme[code > UNIT_SCALE_BIAS], word, &power);
        printf("%s%c%u:", unit_case != 0 ? "," : "", unit_case % UNIT_SIDES ? 'b' : 'a', code);
        if (fits) { printf("2^%d", power); }
        if (!fits) { printf("0x%08x", word); }
    }
}

/* cvt.rn.satfinite.e4m3x2.f32 by its definition: the nearest E4M3 number, a tie to the even code, beyond 448 the
 * code of 448, the sign kept. -1: the definition gives no code (not a number; a negative value that becomes zero). */
static int unit_e4m3_of_single(uint32_t single, uint32_t *group)
{
    uint32_t field = (single >> UNIT_SINGLE_FRACTION_BITS) % UNIT_BYTE_VALUES, magnitude = single & ~UNIT_SINGLE_SIGN;
    uint32_t significand = (single & (UNIT_SINGLE_IMPLICIT_ONE - 1u)) | (field != 0 ? UNIT_SINGLE_IMPLICIT_ONE : 0);
    uint32_t lifted = field < UNIT_E4M3_FIRST_NORMAL_FIELD ? UNIT_E4M3_FIRST_NORMAL_FIELD - field : 0;   /* subnormal */
    uint32_t dropped = UNIT_E4M3_DROPPED_BITS + (lifted < UNIT_E4M3_MOST_LIFT ? lifted : (uint32_t)UNIT_E4M3_MOST_LIFT);
    uint32_t kept = significand >> dropped, rest = significand & ((1u << dropped) - 1u), half = 1u << (dropped - 1u);
    uint32_t code = (field + lifted - UNIT_E4M3_FIRST_NORMAL_FIELD) * UNIT_E4M3_CODES_PER_BINADE + kept
                    + (rest > half || (rest == half && (kept & 1u) != 0));
    *group = magnitude > UNIT_SINGLE_INFINITY ? UNIT_NOT_A_NUMBER : code > UNIT_E4M3_LARGEST ? UNIT_SATURATED
             : code == 0 && magnitude != single ? UNIT_ZERO_OR_NEGATIVE : UNIT_IN_RANGE;
    if (code > UNIT_E4M3_LARGEST) { code = UNIT_E4M3_LARGEST; }
    code |= single >> (UNIT_WORD_BITS - UNIT_BYTE_BITS) & UNIT_E4M3_SIGN;
    return *group >= UNIT_NOT_A_NUMBER ? -1 : (int)code;
}
/* cvt.rp.satfinite.ue8m0x2.f32 by its definition: the code of the smallest power of two that is not under the
 * value, at most 254. -1: the definition gives no code (not a number; zero and the negative values). */
static int unit_ue8m0_of_single(uint32_t single, uint32_t *group)
{
    uint32_t field = (single >> UNIT_SINGLE_FRACTION_BITS) % UNIT_BYTE_VALUES, magnitude = single & ~UNIT_SINGLE_SIGN;
    uint32_t fraction = single & (UNIT_SINGLE_IMPLICIT_ONE - 1u);
    uint32_t code = field == 0 ? fraction > UNIT_SINGLE_HALF_FRACTION : field + (fraction != 0);
    *group = magnitude > UNIT_SINGLE_INFINITY ? UNIT_NOT_A_NUMBER
             : magnitude != single || single == 0 ? UNIT_ZERO_OR_NEGATIVE
             : code > UNIT_UE8M0_LARGEST ? UNIT_SATURATED : UNIT_IN_RANGE;
    if (code > UNIT_UE8M0_LARGEST) { code = UNIT_UE8M0_LARGEST; }
    return *group >= UNIT_NOT_A_NUMBER ? -1 : (int)code;
}

static void unit_check_conversion(const char *name, unit_convert_kernel convert, unit_reference reference)
{
    static const char *const names[UNIT_GROUPS] = {"in_range", "saturated", "not_a_number", "zero_or_negative"};
    static uint16_t converted[UNIT_CONVERT_CHUNK];
    uint64_t of[UNIT_GROUPS] = {0}, differing[UNIT_GROUPS] = {0};
    uint32_t lowest[UNIT_GROUPS], highest[UNIT_GROUPS] = {0}, chunks = UNIT_CONVERT_CHUNKS, group = 0;
    uint32_t one = (uint32_t)reference(UNIT_FP32_PATTERN(0), &group), first_is_upper = 0;
    int valid = 1;
    memset(lowest, UNIT_STALE_BYTE, sizeof(lowest));
    if (probe_flag_given(PROBE_FLAG_STEPS) && probe_flag_number(PROBE_FLAG_STEPS, 0) < chunks) {
        chunks = (uint32_t)probe_flag_number(PROBE_FLAG_STEPS, 0);       /* a short run: each chunk samples the range */
    }
    probe_launch_geometry(&valid, name, NULL, UNIT_CONVERT_BLOCKS, 1, 1, UNIT_CONVERT_BLOCK_THREADS, 0);
    probe_launch_range(&valid, name, "converted pairs", 0, UNIT_CONVERT_CHUNK, sizeof(uint16_t),
                       probe_cuda_array_bytes(unit.card_converted));
    PROBE_REQUIRE(valid, "a conversion's launch was refused");           /* every chunk is this launch */
    for (uint32_t chunk = 0; chunk < chunks; chunk++) {
        convert<<<UNIT_CONVERT_BLOCKS, UNIT_CONVERT_BLOCK_THREADS>>>(unit.card_converted, chunk * UNIT_CONVERT_CHUNK);
        PROBE_CHECK_CUDA(cudaGetLastError());
        PROBE_AFTER_LAUNCH();
        PROBE_CHECK_CUDA(cudaMemcpy(converted, unit.card_converted, sizeof(converted), cudaMemcpyDeviceToHost));
        first_is_upper = chunk == 0 ? (uint32_t)converted[0] >> UNIT_BYTE_BITS == one : first_is_upper;  /* 1 and 2 */
        for (uint32_t index = 0; index < UNIT_SIDES * UNIT_CONVERT_CHUNK; index++) {
            uint32_t pattern = UNIT_FP32_PATTERN(chunk * UNIT_CONVERT_CHUNK + index / UNIT_SIDES);
            uint32_t side = index % UNIT_SIDES, both = converted[index / UNIT_SIDES];
            uint32_t byte = side == first_is_upper ? both % UNIT_BYTE_VALUES : both >> UNIT_BYTE_BITS;
            int code = reference(pattern + side * UNIT_PARTNER_STEP, &group);
            of[group]++;
            differing[group] += code >= 0 && byte != ((uint32_t)code ^ unit.wrong_last_place);
            lowest[group] = byte < lowest[group] ? byte : lowest[group];
            highest[group] = byte > highest[group] ? byte : highest[group];
        }
    }
    printf("%s card=%" PRIu64 " arch=%s form=%s first_operand=%s", UNIT_LINE, probe_flag_number("--card-index", 0),
           PROBE_ARCHITECTURE, name, first_is_upper ? "upper_byte" : "lower_byte");
    for (group = 0; group < UNIT_GROUPS; group++) {
        printf(" %s=%" PRIu64 "/%" PRIu64 " %s_codes=0x%02x..0x%02x", names[group], differing[group], of[group],
               names[group], lowest[group] % UNIT_BYTE_VALUES, highest[group]);
    }
    printf("\n");
}
#endif

/* ---- the forms, and a run ---- */
#define UNIT_EVERY {UNIT_EVERY_PATTERN}, {UNIT_EVERY_PATTERN}
static const struct unit_form unit_forms[] = {
    {"f16", unit_kernel_f16, UNIT_F16, UNIT_F16, 16, {UNIT_F16_E2M1, UNIT_F16_E4M3, UNIT_F16_WHOLE}, {UNIT_F16_E4M3},
     NULL},
    {"bf16", unit_kernel_bf16, UNIT_BF16, UNIT_BF16, 16, {UNIT_BF16_E2M1, UNIT_BF16_WHOLE}, {UNIT_BF16_CARRIED}, NULL},
    {"s8", unit_kernel_s8, UNIT_S8, UNIT_S8, 32, UNIT_EVERY, NULL},
    {"u8_s8", unit_kernel_u8_s8, UNIT_U8, UNIT_S8, 32, UNIT_EVERY, NULL},
    {"s8_saturating", unit_kernel_s8_saturating, UNIT_S8, UNIT_S8, 32, UNIT_EVERY, NULL},
    {"u4_s4", unit_kernel_u4_s4, UNIT_U4, UNIT_S4, 64, UNIT_EVERY, NULL},
    {"u4", unit_kernel_u4, UNIT_U4, UNIT_U4, 64, UNIT_EVERY, NULL},
    {"s4", unit_kernel_s4, UNIT_S4, UNIT_S4, 64, UNIT_EVERY, NULL},
#if PROBE_SM == UNIT_SM_120A
    {"e2m1_e4m3", unit_kernel_e2m1_e4m3, UNIT_E2M1, UNIT_E4M3, 32, UNIT_EVERY, NULL},
    {"e2m1_e4m3_scaled", unit_kernel_e2m1_e4m3_scaled_a00_b00, UNIT_E2M1, UNIT_E4M3, 32, UNIT_EVERY, unit_e2m1_scales},
    {"e2m3_e4m3", unit_kernel_e2m3_e4m3, UNIT_E2M3, UNIT_E4M3, 32, UNIT_EVERY, NULL},
    {"e4m3", unit_kernel_e4m3, UNIT_E4M3, UNIT_E4M3, 32, UNIT_EVERY, NULL},
    {"e4m3_k16", unit_kernel_e4m3_k16, UNIT_E4M3, UNIT_E4M3, 16, UNIT_EVERY, NULL},
    {"e4m3_scaled", unit_kernel_e4m3_scaled_a00_b00, UNIT_E4M3, UNIT_E4M3, 32, UNIT_EVERY, unit_e4m3_scales},
#endif
};

static void unit_check_case(void *, const struct probe_case *probe_case)
{
    static const char *const refused[UNIT_BROKEN_ROUTES] = {"refused_cases", "refused_results", "refused_block"};
    uint32_t wrong_unit = probe_flag_has_word("--break", "unit") != 0;
    uint32_t wrong_layout = probe_flag_has_word("--break", "layout") * UNIT_QUADS;   /* the other half's output */
    /* --break-route: three launches past what a kernel indexes by or is launched with; check 10 must refuse each */
    for (uint32_t broken = 0; probe_flag_given(PROBE_FLAG_BREAK_ROUTE) && broken < UNIT_BROKEN_ROUTES; broken++) {
        const void *results = broken == UNIT_BROKEN_RESULTS ? (const void *)unit_results : unit.card_results;
        int valid = unit_launch_is_valid(refused[broken], results, UNIT_MOST_CASES + (broken == UNIT_BROKEN_CASES),
                                         UNIT_LANES << (broken == UNIT_BROKEN_BLOCK));
        probe_check_count(probe_case->name, "all", refused[broken], valid ? 0 : 1, 1, 1);
    }
    for (uint32_t index = 0; !probe_flag_given(PROBE_FLAG_BREAK_ROUTE) && index < PROBE_COUNT_OF(unit_forms); index++) {
        const struct unit_form *form = &unit_forms[index];
        struct unit_tally layout[UNIT_KINDS] = {}, pairs[UNIT_KINDS] = {};
        uint64_t cases;
        unit_list_patterns(form);
        unit.wrong_last_place = 0;               /* the layout's check is another than --break unit's */
        cases = unit_build_layout(form);
        unit_run(form, form->kernel, cases);
        unit_judge(form, cases, wrong_layout, layout);           /* its codes are normal ones: one tally */
        unit.wrong_last_place = wrong_unit;
        cases = unit_build_pairs(form);
        unit_run(form, form->kernel, cases);
        unit_judge(form, cases, 0, pairs);
        if (index == 0) {                        /* f16, normal codes: what every exact kernel rests on today */
            probe_check_count(probe_case->name, form->name, "reference_layout", layout[UNIT_NORMAL].differing,
                              layout[UNIT_NORMAL].of, wrong_layout != 0);
            probe_check_count(probe_case->name, form->name, "reference_products", pairs[UNIT_NORMAL].differing,
                              pairs[UNIT_NORMAL].of, wrong_unit != 0);
        }
        printf("%s card=%" PRIu64 " arch=%s form=%s layout=%s products=%s subnormal_codes=%s padded_codes=%s"
               " no_number_codes=%s", UNIT_LINE, probe_flag_number("--card-index", 0), PROBE_ARCHITECTURE, form->name,
               UNIT_TALLY(layout[UNIT_NORMAL]), UNIT_TALLY(pairs[UNIT_NORMAL]), UNIT_TALLY(pairs[UNIT_SUBNORMAL]),
               UNIT_TALLY(pairs[UNIT_PADDED]), UNIT_TALLY(pairs[UNIT_NO_NUMBER]));
        unit_run(form, form->kernel, unit_build_sums(form));
        unit_judge_sums(form);
#if PROBE_SM == UNIT_SM_120A
        if (form->selectors != NULL) { unit_check_scales(form); }
#endif
        printf("%s\n", form->selectors != NULL ? "" : " selectors=na scale_bytes=na");
    }
#if PROBE_SM == UNIT_SM_120A
    if (!probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        unit_check_conversion("convert_e4m3", unit_convert_e4m3, unit_e4m3_of_single);
        unit_check_conversion("convert_ue8m0", unit_convert_ue8m0, unit_ue8m0_of_single);
    }
#endif
    probe_verdict("a_line_per_form", PROBE_NAN, PROBE_NAN, PROBE_NAN, PROBE_NAN, 0, 0);  /* the unit only measures */
    probe_cuda_check_guards(probe_case->name);
}

static int unit_open_card(void *, struct probe_card_facts *facts)
{
    if (!probe_cuda_open(facts)) { return 0; }
    unit.card_records = (struct unit_lane_record *)probe_cuda_allocate(sizeof(unit_records));
    unit.card_results = (uint32_t *)probe_cuda_allocate(sizeof(unit_results));
    unit.card_converted = (uint16_t *)probe_cuda_allocate(UNIT_CONVERT_CHUNK * sizeof(uint16_t));
    return 1;
}
static void unit_describe_case(void *, const struct probe_case *, struct probe_case_facts *) {}  /* no RESULT line */
static void unit_close_card(void *) { probe_cuda_close(); }

static struct probe_flag unit_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_WORD_LIST("--break", "unit,layout", "", "what the host gets wrong on purpose: its check must report"),
};
static struct probe_program unit_program = {
    "mma_unit", unit_flags, PROBE_COUNT_OF(unit_flags), NULL, 0, NULL, NULL, unit_open_card, unit_describe_case,
    unit_check_case, NULL, NULL, NULL, NULL, NULL, unit_close_card, NULL};       /* no instance: nothing is timed */

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &unit_program); }
