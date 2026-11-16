/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Load-time byte/count arithmetic. SIZE_MAX is a sticky invalid-size sentinel. */
#ifndef CHECKED_SIZE_H
#define CHECKED_SIZE_H

#include <stddef.h>
#include <stdint.h>

static inline size_t size_add(size_t first, size_t second)
{
	return second > SIZE_MAX - first ? SIZE_MAX : first + second;
}

static inline size_t size_multiply(size_t first, size_t second)
{
	return first == SIZE_MAX || second == SIZE_MAX ||
			       (second != 0 && first > SIZE_MAX / second) ?
		       SIZE_MAX :
		       first * second;
}

static inline size_t size_round_up(size_t bytes, size_t alignment)
{
	if (alignment == 0) {
		return SIZE_MAX;
	}
	const size_t padded = size_add(bytes, alignment - 1u);
	return padded == SIZE_MAX ? SIZE_MAX : padded / alignment * alignment;
}

#endif
