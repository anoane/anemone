/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * record_layout.h: the layout checks every records header uses, in C11 and in CUDA (C++): RECORD_HOLDS states a
 * property of a layout at compile time, RECORD_HAS a record's size in bytes, RECORD_HALVES and RECORD_FLOATS the
 * bytes of so many 16-bit and 32-bit values.
 */

#ifndef RECORD_LAYOUT_H
#define RECORD_LAYOUT_H

#include <stdint.h>

#ifdef __cplusplus
#define RECORD_HOLDS(condition) static_assert(condition, #condition)
#define RECORD_HAS(record, byte_count) \
	static_assert(sizeof(struct record) == (byte_count), #record)
#else
#define RECORD_HOLDS(condition) _Static_assert(condition, #condition)
#define RECORD_HAS(record, byte_count) \
	_Static_assert(sizeof(struct record) == (byte_count), #record)
#endif
#define RECORD_HALVES(count) (sizeof(uint16_t) * (count))
#define RECORD_FLOATS(count) (sizeof(float) * (count))

#endif
