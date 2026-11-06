/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU-only paired-logit measurement. With two capture folders, reports distances without inventing quality gates. */
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "capture_reader.h"
#include "capture_writer.h"

#define CAPTURE_ARGUMENTS 3
#define CANDIDATE_ARGUMENT 2
#define LOGIT_VOCABULARY_DIMENSION 2u
#define LOGIT_DIMENSIONS 3u
#define TOKEN_DIMENSIONS 2u
#define LOGIT_VOCABULARY_LIMIT 1048576u
#define FIXTURE_COLUMNS 2u
#define FIXTURE_LOG_THREE 1.0986122886681098f
#define FIXTURE_FORWARD_KL 0.130812035941137
#define FIXTURE_REVERSE_KL 0.143841036225890
#define FIXTURE_TOLERANCE 1e-7
#define TOKEN_BATCH 256u
#define FIXTURE_LOG_TWO 0.6931471805599453
#define FIXTURE_PATH_BYTES 256u
#define FIXTURE_ROWS (TOKEN_BATCH + 1u)
#define FIXTURE_VOCABULARY 3u
#define FIXTURE_LOGITS (FIXTURE_ROWS * FIXTURE_VOCABULARY)
#define FIXTURE_RECORDS 2u

struct distances {
	double forward_kl, reverse_kl, maximum_absolute, squared_error,
		mean_shift, shifted_squared_error;
	uint32_t same_top;
};

static bool same_distances(const struct distances *first,
			   const struct distances *second)
{
	return first->forward_kl == second->forward_kl &&
	       first->reverse_kl == second->reverse_kl &&
	       first->maximum_absolute == second->maximum_absolute &&
	       first->squared_error == second->squared_error &&
	       first->mean_shift == second->mean_shift &&
	       first->shifted_squared_error == second->shifted_squared_error &&
	       first->same_top == second->same_top;
}

/* Max subtraction keeps exponentials finite; all accumulation and log-probabilities use FP64. */
static bool row_measure(const float *reference, const float *candidate,
			uint32_t columns, struct distances *out)
{
	double reference_maximum = -INFINITY, candidate_maximum = -INFINITY;
	double reference_normalizer = 0, candidate_normalizer = 0;
	uint32_t reference_top = 0, candidate_top = 0;
	struct distances result = { 0 };
	if (reference == NULL || candidate == NULL || out == NULL ||
	    columns == 0)
		return false;
	for (uint32_t column = 0; column < columns; column++) {
		if (!isfinite(reference[column]) ||
		    !isfinite(candidate[column]))
			return false;
		if ((double)reference[column] > reference_maximum) {
			reference_maximum = reference[column];
			reference_top = column;
		}
		if ((double)candidate[column] > candidate_maximum) {
			candidate_maximum = candidate[column];
			candidate_top = column;
		}
		const double difference =
			(double)candidate[column] - (double)reference[column];
		result.maximum_absolute =
			fmax(result.maximum_absolute, fabs(difference));
		result.squared_error += difference * difference;
		result.mean_shift += difference;
	}
	result.mean_shift /= columns;
	for (uint32_t column = 0; column < columns; column++) {
		reference_normalizer +=
			exp((double)reference[column] - reference_maximum);
		candidate_normalizer +=
			exp((double)candidate[column] - candidate_maximum);
	}
	reference_normalizer = log(reference_normalizer);
	candidate_normalizer = log(candidate_normalizer);
	for (uint32_t column = 0; column < columns; column++) {
		const double reference_log = (double)reference[column] -
					     reference_maximum -
					     reference_normalizer;
		const double candidate_log = (double)candidate[column] -
					     candidate_maximum -
					     candidate_normalizer;
		const double centered = (double)candidate[column] -
					(double)reference[column] -
					result.mean_shift;
		result.forward_kl +=
			exp(reference_log) * (reference_log - candidate_log);
		result.reverse_kl +=
			exp(candidate_log) * (candidate_log - reference_log);
		result.shifted_squared_error += centered * centered;
	}
	result.forward_kl = fmax(0, result.forward_kl);
	result.reverse_kl = fmax(0, result.reverse_kl);
	result.same_top = reference_top == candidate_top ? 1u : 0;
	*out = result;
	return true;
}

static bool fixture(void)
{
	const float reference[FIXTURE_COLUMNS] = { FIXTURE_LOG_THREE, 0 };
	const float candidate[FIXTURE_COLUMNS] = { 0, 0 };
	const float shifted[FIXTURE_COLUMNS] = { 1, 1 };
	const float invalid[FIXTURE_COLUMNS] = { NAN, 0 };
	struct distances result = { 0 };
	bool passed =
		row_measure(reference, candidate, FIXTURE_COLUMNS, &result) &&
		fabs(result.forward_kl - FIXTURE_FORWARD_KL) <
			FIXTURE_TOLERANCE &&
		fabs(result.reverse_kl - FIXTURE_REVERSE_KL) <
			FIXTURE_TOLERANCE;
	passed = passed &&
		 row_measure(candidate, shifted, FIXTURE_COLUMNS, &result) &&
		 result.forward_kl == 0 && result.reverse_kl == 0 &&
		 result.maximum_absolute == 1 &&
		 result.shifted_squared_error == 0 && result.same_top == 1;
	passed = passed &&
		 row_measure(reference, reference, FIXTURE_COLUMNS, &result) &&
		 result.maximum_absolute == 0 && result.forward_kl == 0 &&
		 !row_measure(reference, invalid, FIXTURE_COLUMNS, &result);
	const float opposite[FIXTURE_COLUMNS] = { 0, FIXTURE_LOG_THREE };
	passed = passed &&
		 row_measure(reference, opposite, FIXTURE_COLUMNS, &result) &&
		 result.same_top == 0;
	const float extremes[FIXTURE_COLUMNS] = { FLT_MAX, -FLT_MAX };
	passed = passed &&
		 row_measure(extremes, candidate, FIXTURE_COLUMNS, &result) &&
		 fabs(result.forward_kl - FIXTURE_LOG_TWO) <
			 FIXTURE_TOLERANCE &&
		 isfinite(result.reverse_kl) && isfinite(result.squared_error);
	struct distances untouched;
	memset(&result, 0, sizeof(result));
	result.mean_shift = FIXTURE_LOG_TWO;
	memcpy(&untouched, &result, sizeof(untouched));
	const float nonfinite[][FIXTURE_COLUMNS] = { { INFINITY, 0 },
						     { -INFINITY, 0 },
						     { 0, NAN } };
	for (uint32_t index = 0;
	     index < sizeof(nonfinite) / sizeof(nonfinite[0]); index++) {
		passed = passed &&
			 !row_measure(nonfinite[index], candidate,
				      FIXTURE_COLUMNS, &result) &&
			 same_distances(&result, &untouched);
		passed = passed &&
			 !row_measure(candidate, nonfinite[index],
				      FIXTURE_COLUMNS, &result) &&
			 same_distances(&result, &untouched);
	}
	passed = passed &&
		 !row_measure(NULL, candidate, FIXTURE_COLUMNS, &result) &&
		 !row_measure(reference, NULL, FIXTURE_COLUMNS, &result) &&
		 !row_measure(reference, candidate, 0, &result) &&
		 !row_measure(reference, candidate, FIXTURE_COLUMNS, NULL) &&
		 same_distances(&result, &untouched);
	return passed;
}

static bool tokens_match(const struct test_capture *reference,
			 const struct test_capture *candidate)
{
	const struct test_capture_tensor *const tokens =
		test_capture_find(reference, "input.token_ids");
	const struct test_capture_tensor *const other =
		test_capture_find(candidate, "input.token_ids");
	if (tokens == NULL || other == NULL ||
	    tokens->type != TEST_CAPTURE_INT64 ||
	    other->type != TEST_CAPTURE_INT64 ||
	    tokens->dimension_count != TOKEN_DIMENSIONS ||
	    other->dimension_count != TOKEN_DIMENSIONS ||
	    tokens->dimensions[0] != 1 ||
	    memcmp(tokens->dimensions, other->dimensions,
		   sizeof(tokens->dimensions)) != 0)
		return false;
	int64_t left[TOKEN_BATCH], right[TOKEN_BATCH];
	for (uint64_t first = 0; first < tokens->element_count;
	     first += TOKEN_BATCH) {
		const uint64_t count = tokens->element_count - first <
						       TOKEN_BATCH ?
					       tokens->element_count - first :
					       TOKEN_BATCH;
		const result_status left_status = test_capture_read(
			reference, "input.token_ids", TEST_CAPTURE_INT64,
			tokens->dimensions, TOKEN_DIMENSIONS, first, count,
			left, count * sizeof(left[0]));
		const result_status right_status = test_capture_read(
			candidate, "input.token_ids", TEST_CAPTURE_INT64,
			other->dimensions, TOKEN_DIMENSIONS, first, count,
			right, count * sizeof(right[0]));
		if (left_status != STATUS_OK || right_status != STATUS_OK ||
		    memcmp(left, right, (size_t)count * sizeof(left[0])) != 0)
			return false;
	}
	return true;
}

static bool captures_measure(const struct test_capture *reference,
			     const struct test_capture *candidate, bool report)
{
	if (reference == NULL || candidate == NULL)
		return false;
	const struct test_capture_tensor *const logits =
		test_capture_find(reference, "logits.all_rows");
	const struct test_capture_tensor *const other =
		test_capture_find(candidate, "logits.all_rows");
	const struct test_capture_tensor *const tokens =
		test_capture_find(reference, "input.token_ids");
	if (!tokens_match(reference, candidate) || logits == NULL ||
	    other == NULL || tokens == NULL ||
	    logits->type != TEST_CAPTURE_FLOAT32 ||
	    other->type != TEST_CAPTURE_FLOAT32 ||
	    logits->dimension_count != LOGIT_DIMENSIONS ||
	    other->dimension_count != LOGIT_DIMENSIONS ||
	    memcmp(logits->dimensions, other->dimensions,
		   sizeof(logits->dimensions)) != 0 ||
	    logits->dimensions[0] != 1 || logits->dimensions[1] == 0 ||
	    logits->dimensions[1] != tokens->element_count ||
	    logits->dimensions[LOGIT_VOCABULARY_DIMENSION] == 0 ||
	    logits->dimensions[LOGIT_VOCABULARY_DIMENSION] >
		    LOGIT_VOCABULARY_LIMIT)
		return false;
	const uint32_t columns =
		(uint32_t)logits->dimensions[LOGIT_VOCABULARY_DIMENSION];
	const uint64_t rows = logits->dimensions[1];
	float *const left = calloc(columns, sizeof(float));
	float *const right = calloc(columns, sizeof(float));
	bool passed = left != NULL && right != NULL;
	struct distances total = { 0 };
	uint64_t top_matches = 0;
	double maximum_kl = 0;
	for (uint64_t row = 0; passed && row < rows; row++) {
		const result_status left_status = test_capture_read(
			reference, "logits.all_rows", TEST_CAPTURE_FLOAT32,
			logits->dimensions, LOGIT_DIMENSIONS, row * columns,
			columns, left, (uint64_t)columns * sizeof(float));
		const result_status right_status = test_capture_read(
			candidate, "logits.all_rows", TEST_CAPTURE_FLOAT32,
			other->dimensions, LOGIT_DIMENSIONS, row * columns,
			columns, right, (uint64_t)columns * sizeof(float));
		struct distances result = { 0 };
		passed = left_status == STATUS_OK &&
			 right_status == STATUS_OK &&
			 row_measure(left, right, columns, &result);
		total.forward_kl += result.forward_kl;
		total.reverse_kl += result.reverse_kl;
		maximum_kl = fmax(maximum_kl, result.forward_kl);
		total.maximum_absolute =
			fmax(total.maximum_absolute, result.maximum_absolute);
		total.squared_error += result.squared_error;
		total.shifted_squared_error += result.shifted_squared_error;
		top_matches += result.same_top;
	}
	if (passed && report) {
		(void)printf(
			"{\"rows\":%llu,\"columns\":%u,\"mean_forward_kl_nats\":%.17g,"
			"\"mean_reverse_kl_nats\":%.17g,\"maximum_forward_kl_nats\":%.17g,"
			"\"maximum_absolute_logit_error\":%.17g,\"rms_logit_error\":%.17g,"
			"\"shifted_rms_logit_error\":%.17g,\"same_top_rows\":%llu}\n",
			(unsigned long long)rows, columns,
			total.forward_kl / (double)rows,
			total.reverse_kl / (double)rows, maximum_kl,
			total.maximum_absolute,
			sqrt(total.squared_error / ((double)rows * columns)),
			sqrt(total.shifted_squared_error /
			     ((double)rows * columns)),
			(unsigned long long)top_matches);
	}
	free(right);
	free(left);
	return passed;
}

enum fixture_change {
	FIXTURE_IDENTITY,
	FIXTURE_TOKEN_VALUE,
	FIXTURE_TOKEN_SHAPE,
	FIXTURE_TOKEN_TYPE,
	FIXTURE_LOGIT_ROWS,
	FIXTURE_LOGIT_COLUMNS,
	FIXTURE_LOGIT_SHAPE,
	FIXTURE_LOGIT_TYPE,
	FIXTURE_NONFINITE,
	FIXTURE_TRUNCATION,
	FIXTURE_MISSING_TOKENS,
	FIXTURE_MISSING_LOGITS,
	FIXTURE_CHANGE_COUNT
};

/* Both token batches and the final logit row must be read; a comparison cannot silently accept a partial capture. */
static bool fixture_capture_write(const char *folder,
				  enum fixture_change change)
{
	int64_t tokens[FIXTURE_ROWS];
	float logits[FIXTURE_LOGITS];
	for (uint32_t row = 0; row < FIXTURE_ROWS; row++)
		tokens[row] = (int64_t)(row % FIXTURE_VOCABULARY);
	for (uint32_t index = 0; index < FIXTURE_LOGITS; index++)
		logits[index] = (float)(index % FIXTURE_VOCABULARY);
	if (change == FIXTURE_TOKEN_VALUE)
		tokens[FIXTURE_ROWS - 1u] = tokens[FIXTURE_ROWS - 1u] == 0 ? 1 :
									     0;
	if (change == FIXTURE_NONFINITE)
		logits[FIXTURE_LOGITS - 1u] = NAN;
	const uint64_t token_shape[] = { 1, FIXTURE_ROWS, 1 };
	uint64_t logit_shape[] = { 1, FIXTURE_ROWS, FIXTURE_VOCABULARY };
	uint32_t logit_dimensions = LOGIT_DIMENSIONS;
	if (change == FIXTURE_LOGIT_ROWS)
		logit_shape[1]--;
	if (change == FIXTURE_LOGIT_COLUMNS)
		logit_shape[LOGIT_VOCABULARY_DIMENSION]--;
	if (change == FIXTURE_LOGIT_SHAPE) {
		logit_shape[0] = FIXTURE_ROWS;
		logit_shape[1] = FIXTURE_VOCABULARY;
		logit_dimensions = TOKEN_DIMENSIONS;
	}
	struct test_capture_writer *writer = NULL;
	uint32_t token_record = 0, logit_record = 0;
	bool passed = test_capture_writer_create(FIXTURE_RECORDS, &writer) ==
		      STATUS_OK;
	if (passed && change != FIXTURE_MISSING_TOKENS)
		passed = test_capture_writer_add(
				 writer, "input.token_ids",
				 change == FIXTURE_TOKEN_TYPE ?
					 TEST_CAPTURE_FLOAT32 :
					 TEST_CAPTURE_INT64,
				 token_shape,
				 change == FIXTURE_TOKEN_SHAPE ?
					 LOGIT_DIMENSIONS :
					 TOKEN_DIMENSIONS,
				 1, TEST_CAPTURE_NO_LAYER,
				 TEST_CAPTURE_WRITER_NO_POSITION, "synthetic",
				 &token_record) == STATUS_OK;
	if (passed && change != FIXTURE_MISSING_LOGITS)
		passed = test_capture_writer_add(
				 writer, "logits.all_rows",
				 change == FIXTURE_LOGIT_TYPE ?
					 TEST_CAPTURE_BFLOAT16 :
					 TEST_CAPTURE_FLOAT32,
				 logit_shape, logit_dimensions, 1,
				 TEST_CAPTURE_NO_LAYER,
				 TEST_CAPTURE_WRITER_NO_POSITION, "synthetic",
				 &logit_record) == STATUS_OK;
	passed = passed &&
		 test_capture_writer_open(writer, folder) == STATUS_OK;
	if (passed && change != FIXTURE_MISSING_TOKENS)
		passed = test_capture_writer_put(writer, token_record, 0,
						 tokens) == STATUS_OK;
	if (passed && change != FIXTURE_MISSING_LOGITS)
		passed = test_capture_writer_put(writer, logit_record, 0,
						 logits) == STATUS_OK;
	const result_status closed =
		test_capture_writer_close(writer, passed, "{}", "{}");
	return closed == STATUS_OK && passed;
}

static bool fixture_capture_remove(const char *folder)
{
	char path[FIXTURE_PATH_BYTES];
	const char *const files[] = { "manifest.json", "tensors.bin" };
	bool passed = true;
	for (uint32_t index = 0; index < sizeof(files) / sizeof(files[0]);
	     index++) {
		const int written = snprintf(path, sizeof(path), "%s/%s",
					     folder, files[index]);
		if (written <= 0 || (size_t)written >= sizeof(path) ||
		    unlink(path) != 0)
			passed = false;
	}
	return rmdir(folder) == 0 && passed;
}

static bool capture_fixture(void)
{
	char directory[] = "/tmp/anemone_numerics_logits_XXXXXX";
	if (mkdtemp(directory) == NULL)
		return false;
	char reference_path[FIXTURE_PATH_BYTES];
	const int written = snprintf(reference_path, sizeof(reference_path),
				     "%s/reference", directory);
	bool passed = written > 0 && (size_t)written < sizeof(reference_path) &&
		      fixture_capture_write(reference_path, FIXTURE_IDENTITY);
	struct test_capture *reference = NULL;
	passed = passed &&
		 test_capture_open(reference_path, &reference) == STATUS_OK;
	for (uint32_t index = 0;
	     reference != NULL && index < FIXTURE_CHANGE_COUNT; index++) {
		const enum fixture_change change = (enum fixture_change)index;
		char candidate_path[FIXTURE_PATH_BYTES];
		const int candidate_written =
			snprintf(candidate_path, sizeof(candidate_path),
				 "%s/case_%u", directory, index);
		bool ready = candidate_written > 0 &&
			     (size_t)candidate_written <
				     sizeof(candidate_path) &&
			     fixture_capture_write(candidate_path, change);
		struct test_capture *candidate = NULL;
		ready = ready && test_capture_open(candidate_path,
						   &candidate) == STATUS_OK;
		if (ready && change == FIXTURE_TRUNCATION) {
			char binary_path[FIXTURE_PATH_BYTES];
			const int binary_written =
				snprintf(binary_path, sizeof(binary_path),
					 "%s/tensors.bin", candidate_path);
			ready = binary_written > 0 &&
				(size_t)binary_written < sizeof(binary_path) &&
				truncate(
					binary_path,
					(off_t)(sizeof(int64_t) * FIXTURE_ROWS +
						sizeof(float) *
							(FIXTURE_LOGITS -
							 1u))) == 0;
		}
		const bool measured =
			ready && captures_measure(reference, candidate, false);
		bool case_passed = ready &&
				   measured == (change == FIXTURE_IDENTITY);
		if (ready && change == FIXTURE_LOGIT_ROWS)
			case_passed =
				case_passed &&
				!captures_measure(candidate, candidate, false);
		test_capture_close(candidate);
		case_passed = fixture_capture_remove(candidate_path) &&
			      case_passed;
		if (!case_passed)
			(void)fprintf(stderr,
				      "FAILED: numerical capture fixture %u\n",
				      index);
		passed = case_passed && passed;
	}
	test_capture_close(reference);
	passed = fixture_capture_remove(reference_path) && passed;
	return rmdir(directory) == 0 && passed &&
	       !captures_measure(NULL, NULL, false);
}

int main(int argument_count, char **arguments)
{
	if (!fixture() || !capture_fixture())
		return EXIT_FAILURE;
	if (argument_count == 1) {
		(void)puts(
			"numerics_logits_test: probability, shift, extreme, nonfinite and 12 capture fixtures passed");
		return EXIT_SUCCESS;
	}
	if (argument_count != CAPTURE_ARGUMENTS)
		return EXIT_FAILURE;
	struct test_capture *reference = NULL, *candidate = NULL;
	const result_status reference_status =
		test_capture_open(arguments[1], &reference);
	const result_status candidate_status =
		test_capture_open(arguments[CANDIDATE_ARGUMENT], &candidate);
	const bool passed = reference_status == STATUS_OK &&
			    candidate_status == STATUS_OK &&
			    captures_measure(reference, candidate, true);
	test_capture_close(candidate);
	test_capture_close(reference);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
