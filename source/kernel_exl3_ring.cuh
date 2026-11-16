/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* A lane's ring of one EXL3 tile for the m16n8k16 A fragment: lane l decodes ring positions 8 l .. 8 l + 7, whose
 * windows end at S(8 l) + offset(symbol) (exl3_records.h). Two cyclic words from the first window's start hold all
 * eight windows at every width this build decodes (checked here per width), so a lane loads them once a tile. */
#ifndef KERNEL_EXL3_RING_CUH
#define KERNEL_EXL3_RING_CUH

#include "anemone/exl3_records.h"

#define EXL3_LANE_SYMBOLS 8u
#define EXL3_LANE_LOAD_BITS 64u /* the two words a lane loads */

template <uint32_t bits_x2> struct exl3_lane_ring {
	static constexpr uint32_t ring_bits =
		EXL3_TILE_VALUES * bits_x2 / EXL3_HALF_BITS_PER_BIT;
	static constexpr uint32_t ring_words = ring_bits / EXL3_WORD_BITS;
	/* ring bit where the window of the lane's first symbol starts */
	__host__ __device__ static constexpr uint32_t begin(uint32_t lane)
	{
		return (exl3_window_end(lane * EXL3_LANE_SYMBOLS, bits_x2) +
			ring_bits - EXL3_STATE_BITS) %
		       ring_bits;
	}
	/* bits from the end of the lane's first window to the end of window symbol's */
	__host__ __device__ static constexpr uint32_t offset(uint32_t symbol)
	{
		return exl3_window_end(symbol, bits_x2) -
		       exl3_window_end(0u, bits_x2);
	}
	__host__ __device__ static constexpr bool two_words()
	{
		for (uint32_t lane = 0; lane < EXL3_WARP; lane++)
			if (begin(lane) % EXL3_WORD_BITS + EXL3_STATE_BITS +
				    offset(EXL3_LANE_SYMBOLS - 1u) >
			    EXL3_LANE_LOAD_BITS)
				return false;
		return true;
	}
	static_assert(((EXL3_SUPPORTED_BITS_X2 >> bits_x2) & 1u) != 0 &&
			      ring_bits % EXL3_WORD_BITS == 0 && two_words(),
		      "a supported width whose lane windows fit two words");
};

#endif
