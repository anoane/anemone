/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef STORAGE_PREPARE_H
#define STORAGE_PREPARE_H
#include "engine/storage/profile/storage_profile.h"
#define PREPARE_CHUNKS_MOST 16384u
#define PREPARE_CHUNK_BYTES (64u << 20u)
struct prepare_manifest {
	char source[PATH_MAX], pack_identity[STORAGE_PROFILE_KEY_BYTES];
	char hashes[PREPARE_CHUNKS_MOST][STORAGE_PROFILE_KEY_BYTES];
	uint64_t bytes, chunk_bytes;
	uint32_t chunks, known;
};
struct prepare_file {
	FILE *stream;
	char temporary[PATH_MAX], final[PATH_MAX];
};
MUST_CHECK result_status prepare_hash(const void *bytes, size_t count,
				      char output[STORAGE_PROFILE_KEY_BYTES]);
/* The digests of two strings of one length in one interleaved pass (sha256_of_two); each equals prepare_hash's. */
MUST_CHECK result_status
prepare_hash_two(const void *first, const void *second, size_t count,
		 char first_output[STORAGE_PROFILE_KEY_BYTES],
		 char second_output[STORAGE_PROFILE_KEY_BYTES]);
MUST_CHECK result_status prepare_manifest_read(const char *path,
					       struct prepare_manifest *output);
MUST_CHECK result_status prepare_manifest_write(
	const char *path, const struct prepare_manifest *manifest);
MUST_CHECK result_status prepare_file_open(const char *path,
					   struct prepare_file *file);
MUST_CHECK result_status prepare_file_commit(struct prepare_file *file);
void prepare_file_abort(struct prepare_file *file);
/* A JSON string of text; a byte sequence that is no UTF-8 is written as U+FFFD. */
void prepare_json_text(FILE *output, const char *text);
/* Whether text is UTF-8 throughout, so that prepare_json_text records it exactly. */
bool prepare_utf8_text(const char *text);
#endif
