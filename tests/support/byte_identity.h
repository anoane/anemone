/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_BYTE_IDENTITY_H
#define TEST_BYTE_IDENTITY_H
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* Whether two objects hold the same bytes, padding and float bits included. The tests compare bytes on purpose (a
 * refusal leaves every byte as it was, a copy is exact): through void pointers, the comparison is of bytes, not of
 * the objects' values. */
static inline bool bytes_identical(const void *first, const void *second,
				   size_t bytes)
{
	return memcmp(first, second, bytes) == 0;
}

#endif
