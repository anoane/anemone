/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Opt-in original-weight ViT check. Exactly one approved card, only inside the owner's gpu_run.sh.
 * The patch file is two little-endian uint32 grid dimensions followed by contiguous BF16 patches.
 * Writes final projected BF16 cells; no language-model state or weights are loaded. VISION_TRACE_DIRECTORY,
 * when set to an existing directory, also replays the tower after timing and writes pass_NNN.bf16 captures.
 * Trace downloads are chunked to the fixed staging capacity and include all three QKV components after RoPE.
 * VISION_TRACE_REFERENCE_DIRECTORY optionally restores every previously written scratch/output range from
 * its last reference writer before each pass, isolating arithmetic. Reference/trace directories must differ.
 * The normal final output is saved before replay and remains the result of a complete candidate forward.
 * VISION_TEST_MLP_K_PARTITION and VISION_TEST_ALIGNER_K_PARTITION optionally set positive, tile-aligned
 * K partition widths within their respective input widths. Unset means the product's original launch flags.
 * These experimental launch-record overrides apply equally to timed forwards and diagnostic replay. */
#ifndef TEST_VISION_ORIGINAL_H
#define TEST_VISION_ORIGINAL_H
#include <stdbool.h>
bool test_vision_original_card(const char *pack_path, const char *patch_path,
			       const char *output_path);
#endif
