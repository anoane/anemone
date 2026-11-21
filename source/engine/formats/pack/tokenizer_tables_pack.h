/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Startup binding only; private to main and its source tests. No new runtime seam. */
#ifndef TOKENIZER_TABLES_PACK_H
#define TOKENIZER_TABLES_PACK_H

#include <stdint.h>

#include "engine/storage/load/loader.h"
#include "pack.h"
#include "common/status.h"

/* Call only after load_run succeeded for this immutable pack/layout/result. The externally opened tables stay
 * immutable through engine_destroy. Exact bytes, rather than vocabulary size, bind the tokenizer used for prompt
 * encoding and response decoding to the checked loaded checkpoint. No allocation, card work or token-path call.
 * Missing tables: UNSUPPORTED. Bad metadata/placement: CORRUPT_PACK; differing external bytes: INVALID_ARGUMENT.
 * The pack digest detects damage, not malicious substitution or checkpoint provenance. */
MUST_CHECK result_status tokenizer_tables_match(
	const struct pack_record *pack, const struct load_layout *layout,
	const struct load_result *loaded, const void *external_tables,
	uint64_t external_byte_count);

#endif
