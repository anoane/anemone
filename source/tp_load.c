/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/tp_load.h"

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>

#include "anemone/hooks.h"
#include "anemone/pack_digest.h"

#define MATRIX_DIMENSIONS 2u
#define NANOSECONDS_PER_SECOND 1000000000ull
#define STAGED_JOBS_MOST \
	64u /* a worker stages at most this many jobs before their uploads begin */
#define STAGED_ALIGNMENT_BYTES \
	BACKEND_STAGING_ALIGNMENT_BYTES /* each staged job begins on a page of its slot */
#define WORKER_TIMER_SLACK_NANOSECONDS 1000ul
#define BUSY_PAUSE_NANOSECONDS \
	20000l /* between two upload_begin calls a full card refused */
#define EXTENT_SLOTTED 0u /* an extent a worker's own source slot holds */
#define EXTENT_LARGE 1u /* an extent above it: through the large lane */
#define EXTENT_KINDS 2u
#define COPY_FAILED_PENALTY \
	1e18 /* seconds added to a copy whose read failed: it is chosen last */

static uint64_t padded_bytes(uint64_t bytes)
{
	const uint64_t alignment = PACK_DIGEST_WORD_BYTES;
	return bytes > UINT64_MAX - (alignment - 1u) ?
		       UINT64_MAX :
		       (bytes + alignment - 1u) / alignment * alignment;
}

static bool same_shape(const struct matrix_shape_record *first,
		       const struct matrix_shape_record *second)
{
	return first->output_size == second->output_size &&
	       first->input_size == second->input_size &&
	       first->bits_per_weight == second->bits_per_weight &&
	       first->reserved_must_be_zero == second->reserved_must_be_zero;
}

MUST_CHECK static result_status job_bytes(const struct tp_load_job *job,
					  const struct pack_tensor *tensor,
					  uint64_t *bytes_out)
{
	if (job->reserved_zero != 0 || tensor->byte_count == 0 ||
	    (job->mode != TP_LOAD_FULL && job->mode != TP_LOAD_SHARD))
		return STATUS_INVALID_ARGUMENT;
	uint64_t bytes = tensor->byte_count;
	if (job->mode == TP_LOAD_SHARD) {
		/* a plain matrix: row shards of 16-bit elements (tp_weights.h) */
		const bool plain = tensor->weight_format == WEIGHT_FORMAT_NONE;
		const struct matrix_shape_record source = {
			tensor->dimensions[0], tensor->dimensions[1],
			plain ? TENSOR_SHARD_PLAIN_BITS :
				tensor->bits_per_weight,
			0
		};
		struct tensor_shard_plan checked;
		if (tensor->dimension_count != MATRIX_DIMENSIONS ||
		    (plain && tensor->element_type != PACK_ELEMENT_BFLOAT16 &&
		     tensor->element_type != PACK_ELEMENT_FLOAT16) ||
		    job->shard.description.format !=
			    (enum weight_format_code)tensor->weight_format ||
		    !same_shape(&job->shard.description.source_shape, &source))
			return STATUS_INVALID_ARGUMENT;
		const result_status status =
			tensor_shard_prepare(&job->shard.description, &checked);
		if (status != STATUS_OK)
			return status;
		if (checked.source_byte_count != tensor->byte_count ||
		    job->shard.source_byte_count != checked.source_byte_count ||
		    job->shard.shard_byte_count != checked.shard_byte_count ||
		    !same_shape(&job->shard.shard_shape, &checked.shard_shape))
			return STATUS_INVALID_ARGUMENT;
		bytes = checked.shard_byte_count;
	}
	*bytes_out = bytes;
	return STATUS_OK;
}

result_status tp_load_prepare(const struct tp_load_description *description,
			      struct tp_load_plan *plan_out)
{
	if (description == NULL || plan_out == NULL ||
	    description->pack == NULL || description->jobs == NULL ||
	    description->job_count == 0 || description->card_count == 0 ||
	    description->card_count > BACKEND_MAXIMUM_CARDS ||
	    description->maximum_extent_byte_count == 0 ||
	    description->maximum_extent_byte_count >
		    TP_LOAD_MAXIMUM_EXTENT_BYTES)
		return STATUS_INVALID_ARGUMENT;
	const struct pack_header *const header =
		pack_header_of(description->pack);
	const struct pack_tensor *const tensors =
		pack_tensors_of(description->pack);
	const struct pack_extent *const extents =
		pack_extents_of(description->pack);
	const struct pack_unit_record_2 *const units =
		pack_units_of(description->pack);
	if ((uint64_t)description->job_count >
	    (uint64_t)header->tensor_count * description->card_count)
		return STATUS_INVALID_ARGUMENT;
	struct tp_load_plan prepared = { .description = *description };
	uint64_t destination_end[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint32_t previous_extent = 0;
	for (uint32_t index = 0; index < description->job_count; index++) {
		const struct tp_load_job *const job = &description->jobs[index];
		if (job->tensor_index >= header->tensor_count ||
		    job->card >= description->card_count ||
		    job->destination_offset %
				    WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES !=
			    0)
			return STATUS_INVALID_ARGUMENT;
		const struct pack_tensor *const tensor =
			&tensors[job->tensor_index];
		const struct pack_extent *const extent =
			&extents[tensor->extent_index];
		if (index != 0 && tensor->extent_index < previous_extent)
			return STATUS_INVALID_ARGUMENT;
		previous_extent = tensor->extent_index;
		if (units[extent->unit_index].unit_kind ==
			    PLACEMENT_UNIT_HOST_ONLY ||
		    extent->byte_count > description->maximum_extent_byte_count)
			return STATUS_UNSUPPORTED;
		uint64_t bytes = 0;
		const result_status status = job_bytes(job, tensor, &bytes);
		if (status != STATUS_OK)
			return status;
		const uint64_t padded = padded_bytes(bytes);
		const uint64_t arena =
			description->card_arena_byte_count[job->card];
		if (padded == UINT64_MAX || job->destination_offset > arena ||
		    padded > arena - job->destination_offset ||
		    job->destination_offset < destination_end[job->card])
			return STATUS_INVALID_ARGUMENT;
		destination_end[job->card] = job->destination_offset + padded;
		prepared.source_workspace_byte_count =
			extent->byte_count >
					prepared.source_workspace_byte_count ?
				extent->byte_count :
				prepared.source_workspace_byte_count;
		prepared.output_workspace_byte_count =
			padded > prepared.output_workspace_byte_count ?
				padded :
				prepared.output_workspace_byte_count;
	}
	*plan_out = prepared;
	return STATUS_OK;
}

static bool disjoint(const void *first, uint64_t first_bytes,
		     const void *second, uint64_t second_bytes)
{
	const uintptr_t first_address = (uintptr_t)first;
	const uintptr_t second_address = (uintptr_t)second;
	return first_bytes <= UINTPTR_MAX - first_address &&
	       second_bytes <= UINTPTR_MAX - second_address &&
	       (first_address + first_bytes <= second_address ||
		second_address + second_bytes <= first_address);
}

/* The digest of a piece of an extent that begins first_word_index words into it (pack_digest.h: a sum of terms). */
static uint64_t piece_digest(const unsigned char *bytes, uint64_t byte_count,
			     uint64_t first_word_index)
{
	return pack_digest_of_piece(bytes, byte_count, first_word_index);
}

static uint64_t monotonic_nanoseconds(void)
{
	struct timespec now = { 0 };
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

result_status tp_load_copies_init(struct tp_load_copies *copies,
				  uint32_t reads_most)
{
	if (copies == NULL)
		return STATUS_INVALID_ARGUMENT;
	(void)memset(copies, 0, sizeof(*copies));
	copies->reads_most = reads_most;
	if (pthread_mutex_init(&copies->mutex, NULL) != 0)
		return STATUS_OUT_OF_MEMORY;
	if (pthread_cond_init(&copies->read_ended, NULL) != 0) {
		(void)pthread_mutex_destroy(&copies->mutex);
		return STATUS_OUT_OF_MEMORY;
	}
	return STATUS_OK;
}

void tp_load_copies_destroy(struct tp_load_copies *copies)
{
	if (copies != NULL) {
		(void)pthread_cond_destroy(&copies->read_ended);
		(void)pthread_mutex_destroy(&copies->mutex);
	}
}

/* Under the mutex: whether copy has room for one more read. */
static bool copy_has_room(const struct tp_load_copies *copies, uint32_t copy)
{
	return copies->reads_most == 0 ||
	       copies->reads_in_flight[copy] < copies->reads_most;
}

/* Under the mutex: a read of bytes from copy is in flight from now. */
static void copy_begin_locked(struct tp_load_copies *copies, uint32_t copy,
			      uint64_t bytes, uint64_t now)
{
	copies->busy_since[copy] = copies->reads_in_flight[copy] == 0 ?
					   now :
					   copies->busy_since[copy];
	copies->reads_in_flight[copy]++;
	copies->in_flight_bytes[copy] += bytes;
}

/* The copy a read of bytes begins from (copies NULL: copy 0), now counted in flight. */
static uint32_t copy_take(struct tp_load_copies *copies, uint32_t copy_count,
			  uint64_t bytes, uint32_t preferred)
{
	if (copies == NULL)
		return 0;
	if (copies->split != 0 && copy_count > 1 && preferred < copy_count) {
		(void)pthread_mutex_lock(&copies->mutex);
		const bool sound = copies->failed_reads[preferred] == 0;
		while (sound && !copy_has_room(copies, preferred))
			(void)pthread_cond_wait(&copies->read_ended,
						&copies->mutex);
		if (sound)
			copy_begin_locked(copies, preferred, bytes,
					  monotonic_nanoseconds());
		(void)pthread_mutex_unlock(&copies->mutex);
		if (sound)
			return preferred;
	}
	double rates[STORAGE_MAXIMUM_COPIES] = { 0 };
	double fastest = 0.0, soonest = 0.0;
	uint32_t chosen = copy_count;
	(void)pthread_mutex_lock(&copies->mutex);
	bool room = false;
	for (uint32_t copy = 0; copy < copy_count; copy++)
		room = room || copy_has_room(copies, copy);
	while (!room) {
		(void)pthread_cond_wait(&copies->read_ended, &copies->mutex);
		for (uint32_t copy = 0; copy < copy_count; copy++)
			room = room || copy_has_room(copies, copy);
	}
	const uint64_t now = monotonic_nanoseconds();
	for (uint32_t copy = 0; copy < copy_count; copy++) {
		const uint64_t busy = copies->busy_nanoseconds[copy] +
				      (copies->reads_in_flight[copy] != 0 ?
					       now - copies->busy_since[copy] :
					       0);
		rates[copy] = busy != 0 ?
				      (double)copies->delivered_bytes[copy] /
					      (double)busy :
				      0.0;
		fastest = rates[copy] > fastest ? rates[copy] : fastest;
	}
	for (uint32_t copy = 0; copy < copy_count; copy++) {
		const double rate = rates[copy] > 0.0 ? rates[copy] :
				    fastest > 0.0     ? fastest :
							1.0;
		const double finish =
			(double)(copies->in_flight_bytes[copy] + bytes) / rate +
			(copies->failed_reads[copy] != 0 ? COPY_FAILED_PENALTY :
							   0.0);
		if (copy_has_room(copies, copy) &&
		    (chosen == copy_count || finish < soonest)) {
			chosen = copy;
			soonest = finish;
		}
	}
	copy_begin_locked(copies, chosen, bytes, now);
	(void)pthread_mutex_unlock(&copies->mutex);
	return chosen;
}

/* A read of bytes from copy begins (another copy after the chosen one), or has ended: delivered or failed. */
static void copy_reading(struct tp_load_copies *copies, uint32_t copy,
			 uint64_t bytes)
{
	if (copies == NULL)
		return;
	(void)pthread_mutex_lock(&copies->mutex);
	while (!copy_has_room(copies, copy))
		(void)pthread_cond_wait(&copies->read_ended, &copies->mutex);
	copy_begin_locked(copies, copy, bytes, monotonic_nanoseconds());
	(void)pthread_mutex_unlock(&copies->mutex);
}

static void copy_done(struct tp_load_copies *copies, uint32_t copy,
		      uint64_t bytes, bool delivered)
{
	if (copies == NULL)
		return;
	const uint64_t now = monotonic_nanoseconds();
	(void)pthread_mutex_lock(&copies->mutex);
	copies->reads_in_flight[copy]--;
	copies->in_flight_bytes[copy] -= bytes;
	copies->delivered_bytes[copy] += delivered ? bytes : 0;
	copies->failed_reads[copy] += delivered ? 0u : 1u;
	copies->busy_nanoseconds[copy] +=
		copies->reads_in_flight[copy] == 0 ?
			now - copies->busy_since[copy] :
			0;
	(void)pthread_cond_broadcast(&copies->read_ended);
	(void)pthread_mutex_unlock(&copies->mutex);
}

/* A copy delivered bytes that are not the pack's: it is chosen last from now on. */
static void copy_fault(struct tp_load_copies *copies, uint32_t copy)
{
	if (copies == NULL)
		return;
	(void)pthread_mutex_lock(&copies->mutex);
	copies->failed_reads[copy]++;
	(void)pthread_mutex_unlock(&copies->mutex);
}

/* The copy of the attempt-th read of an extent: the chosen first, then the others in order. */
static uint32_t copy_of_attempt(uint32_t attempt, uint32_t first)
{
	return attempt == 0	    ? first :
	       attempt - 1u < first ? attempt - 1u :
				      attempt;
}

MUST_CHECK static result_status
extent_read(const struct tp_load_request *request,
	    const struct storage_description *storage,
	    const struct pack_extent *extent, void *workspace,
	    struct tp_load_copies *copies, struct tp_load_result *result)
{
	result_status last_status = STATUS_INPUT_OUTPUT_ERROR;
	bool wrong_digest = false;
	const uint32_t first = copy_take(copies, storage->copy_count,
					 extent->byte_count,
					 storage->copy_count - 1u);
	for (uint32_t attempt = 0; attempt < storage->copy_count; attempt++) {
		const uint32_t copy = copy_of_attempt(attempt, first);
		if (attempt != 0)
			copy_reading(copies, copy, extent->byte_count);
		TIMER_BEGIN(tp_load_extent_read);
		last_status = request->storage->reader_read(
			request->reader, copy, extent->byte_offset, workspace,
			extent->byte_count);
		TIMER_END(tp_load_extent_read);
		copy_done(copies, copy, extent->byte_count,
			  last_status == STATUS_OK);
		if (last_status != STATUS_OK) {
			if (last_status != STATUS_INPUT_OUTPUT_ERROR)
				return last_status;
			continue;
		}
		result->source_bytes_read += extent->byte_count;
		uint64_t digest = 0;
		TIMER_BEGIN(tp_load_extent_digest);
		last_status = pack_digest_of_host_bytes(
			workspace, extent->byte_count, &digest);
		TIMER_END(tp_load_extent_digest);
		if (last_status != STATUS_OK)
			return last_status;
		if (digest == extent->digest) {
			result->extents_verified++;
			return STATUS_OK;
		}
		wrong_digest = true;
		copy_fault(copies, copy);
	}
	return wrong_digest ? STATUS_CORRUPT_PACK : last_status;
}

/* Copies a job's tensor or shard out of its verified extent into output, zero-padded to a digest word, and digests
 * the staged bytes: what the card must hold afterwards. */
MUST_CHECK static result_status
job_stage(const struct tp_load_job *job, const struct pack_tensor *tensor,
	  const unsigned char *source, unsigned char *output,
	  uint64_t *padded_out, uint64_t *expected_out)
{
	uint64_t bytes = 0;
	result_status status = job_bytes(job, tensor, &bytes);
	if (status != STATUS_OK)
		return status;
	const uint64_t padded = padded_bytes(bytes);
	TIMER_BEGIN(tp_load_job_stage);
	(void)memset(output, 0, (size_t)padded);
	if (job->mode == TP_LOAD_SHARD)
		status = tensor_shard_extract(
			&job->shard, source + tensor->byte_offset_in_extent,
			tensor->byte_count, output, padded);
	else
		(void)memcpy(output, source + tensor->byte_offset_in_extent,
			     (size_t)bytes);
	TIMER_END(tp_load_job_stage);
	TIMER_BEGIN(tp_load_job_digest);
	if (status == STATUS_OK)
		status =
			pack_digest_of_host_bytes(output, padded, expected_out);
	TIMER_END(tp_load_job_digest);
	*padded_out = padded;
	return status;
}

/* A job whose upload has begun: waits for it, digests its range on the card and compares with the staged bytes. */
MUST_CHECK static result_status job_check(const struct tp_load_request *request,
					  const struct tp_load_job *job,
					  uint64_t ticket, uint64_t padded,
					  uint64_t expected,
					  struct tp_load_result *result)
{
	struct device_card *const card = request->cards[job->card];
	TIMER_BEGIN(tp_load_upload_wait);
	result_status status = request->backend->upload_wait(
		card, ticket, request->wait_limit_nanoseconds);
	TIMER_END(tp_load_upload_wait);
	uint64_t delivered = 0;
	TIMER_BEGIN(tp_load_card_digest);
	if (status == STATUS_OK)
		status = request->backend->digest(
			card, job->destination_offset, padded,
			request->wait_limit_nanoseconds, &delivered);
	TIMER_END(tp_load_card_digest);
	if (status == STATUS_OK && delivered != expected)
		status = STATUS_CORRUPT_PACK;
	if (status == STATUS_OK) {
		result->uploaded_bytes += padded;
		result->tensors_uploaded++;
		result->card_receipts[job->card] += pack_digest_term(
			delivered,
			job->destination_offset / PACK_DIGEST_WORD_BYTES);
	}
	return status;
}

MUST_CHECK static result_status
tensor_upload(const struct tp_load_request *request,
	      const struct tp_load_job *job, const struct pack_tensor *tensor,
	      struct tp_load_result *result)
{
	uint64_t padded = 0, expected = 0, ticket = 0;
	result_status status = job_stage(job, tensor, request->source_workspace,
					 request->output_workspace, &padded,
					 &expected);
	if (status == STATUS_OK)
		status = request->backend->upload_begin(
			request->cards[job->card], job->destination_offset,
			request->output_workspace, padded, &ticket);
	if (status == STATUS_OK)
		status = job_check(request, job, ticket, padded, expected,
				   result);
	return status;
}

/* Runs work on the calling thread and on up to thread_count - 1 more, then joins them. The work is taken by whoever
 * runs, so a thread that cannot be created only makes one fewer. */
static void threads_run(uint32_t thread_count, void *(*work)(void *),
			void *context)
{
	pthread_t helpers[TP_LOAD_MAXIMUM_WORKERS];
	uint32_t started = 0;
	while (started + 1u < thread_count &&
	       pthread_create(&helpers[started], NULL, work, context) == 0)
		started++;
	(void)work(context);
	for (uint32_t helper = 0; helper < started; helper++)
		(void)pthread_join(helpers[helper], NULL);
}

static uint64_t smaller_of(uint64_t first, uint64_t second)
{
	return first < second ? first : second;
}

/* What the workers of one pipelined run share. */
struct load_shared {
	const struct tp_load_request *request;
	const struct tp_load_description *description;
	const struct storage_description *storage;
	uint64_t source_slot_bytes, output_slot_bytes;
	pthread_mutex_t mutex; /* guards the three below */
	/* per kind of extent (EXTENT_SLOTTED, EXTENT_LARGE: above a worker's source slot), where the next one may begin;
	 * one worker walks the large extents from the start, so that the large lane does not trail the others */
	uint32_t next_job[EXTENT_KINDS];
	bool large_walker_taken;
	result_status status; /* the first failure */
	struct tp_load_result result; /* the sum of the finished workers' */
	pthread_mutex_t
		large_lane; /* its holder uses the request's two workspaces */
};

/* A source and an output buffer: a worker's own slots, or the request's workspaces. */
struct load_lane {
	unsigned char *source, *output;
	uint64_t source_bytes, output_bytes;
};

struct staged_job {
	const struct tp_load_job *job;
	uint64_t output_offset, padded, expected, ticket;
};

/* Takes the jobs of the next extent in plan order; false once every extent is taken or a worker failed. */
/* Under the mutex: the next extent of kind at or after its cursor, its jobs [first, end); false when none is left. */
static bool extent_of_kind(struct load_shared *shared, uint32_t kind,
			   uint32_t *first_out, uint32_t *end_out)
{
	const struct tp_load_job *const jobs = shared->description->jobs;
	const struct pack_tensor *const tensors =
		pack_tensors_of(shared->description->pack);
	const struct pack_extent *const extents =
		pack_extents_of(shared->description->pack);
	const uint32_t job_count = shared->description->job_count;
	for (uint32_t first = shared->next_job[kind]; first < job_count;) {
		const uint32_t extent =
			tensors[jobs[first].tensor_index].extent_index;
		uint32_t end = first;
		while (end < job_count &&
		       tensors[jobs[end].tensor_index].extent_index == extent)
			end++;
		const uint32_t found =
			extents[extent].byte_count > shared->source_slot_bytes ?
				EXTENT_LARGE :
				EXTENT_SLOTTED;
		shared->next_job[kind] = end;
		if (found == kind) {
			*first_out = first;
			*end_out = end;
			return true;
		}
		first = end;
	}
	return false;
}

/* Takes the jobs of the next extent in plan order, of the worker's kind first, then of the other; false once every
 * extent is taken or a worker failed. */
static bool extent_claim(struct load_shared *shared, uint32_t kind,
			 uint32_t *first_out, uint32_t *count_out)
{
	uint32_t first = 0, end = 0;
	(void)pthread_mutex_lock(&shared->mutex);
	const bool claimed = shared->status == STATUS_OK &&
			     (extent_of_kind(shared, kind, &first, &end) ||
			      extent_of_kind(shared, EXTENT_KINDS - 1u - kind,
					     &first, &end));
	(void)pthread_mutex_unlock(&shared->mutex);
	*first_out = first;
	*count_out = end - first;
	return claimed;
}

/* When upload_begin says BUSY: false once BUSY has lasted the wait limit, else true after a short pause. */
static bool busy_pause(uint64_t *busy_since, uint64_t wait_limit_nanoseconds)
{
	const uint64_t nanoseconds = monotonic_nanoseconds();
	if (nanoseconds == 0)
		return false;
	*busy_since = *busy_since == 0 ? nanoseconds : *busy_since;
	if (nanoseconds - *busy_since >= wait_limit_nanoseconds)
		return false;
	const struct timespec pause = { .tv_nsec = BUSY_PAUSE_NANOSECONDS };
	(void)clock_nanosleep(CLOCK_MONOTONIC, 0, &pause, NULL);
	return true;
}

/* Begins the uploads of the staged jobs, then checks each. A card with its most uploads unfinished answers BUSY:
 * the worker then checks the uploads it has begun first, as every worker does before it pauses, so the oldest
 * unfinished uploads are always being waited for. After a failure the rest are still waited for, so that no copy
 * reads the output slot once this returns (unless its card failed: the caller then drains or closes it). */
MUST_CHECK static result_status
batch_flush(const struct tp_load_request *request, unsigned char *output,
	    struct staged_job *staged, uint32_t count,
	    struct tp_load_result *result)
{
	result_status status = STATUS_OK;
	uint32_t begun = 0, checked = 0;
	uint64_t busy_since = 0;
	while (begun < count && status == STATUS_OK) {
		struct staged_job *const item = &staged[begun];
		TIMER_BEGIN(tp_load_upload_begin);
		status = request->backend->upload_begin(
			request->cards[item->job->card],
			item->job->destination_offset,
			output + item->output_offset, item->padded,
			&item->ticket);
		TIMER_END(tp_load_upload_begin);
		if (status == STATUS_OK) {
			begun++;
			busy_since = 0;
			continue;
		}
		if (status != STATUS_BUSY_VALUE)
			break;
		status = STATUS_OK;
		for (; checked < begun && status == STATUS_OK; checked++)
			status = job_check(request, staged[checked].job,
					   staged[checked].ticket,
					   staged[checked].padded,
					   staged[checked].expected, result);
		if (status == STATUS_OK &&
		    !busy_pause(&busy_since, request->wait_limit_nanoseconds))
			status = STATUS_BUSY_VALUE;
	}
	for (; checked < begun && status == STATUS_OK; checked++)
		status = job_check(request, staged[checked].job,
				   staged[checked].ticket,
				   staged[checked].padded,
				   staged[checked].expected, result);
	for (; checked < begun; checked++) {
		const result_status drained = request->backend->upload_wait(
			request->cards[staged[checked].job->card],
			staged[checked].ticket,
			request->wait_limit_nanoseconds);
		(void)drained; /* the failure above is what the run returns */
	}
	return status;
}

/* One extent and its jobs: read and verified into the lane's source, the jobs staged in its output, which is
 * flushed whenever the next job would not fit. An extent or job larger than the worker's own slots waits for the
 * large lane. */
MUST_CHECK static result_status extent_load(struct load_shared *shared,
					    const struct load_lane *own,
					    uint32_t first, uint32_t count,
					    struct tp_load_result *result)
{
	const struct tp_load_request *const request = shared->request;
	const struct tp_load_job *const jobs = shared->description->jobs;
	const struct pack_tensor *const tensors =
		pack_tensors_of(shared->description->pack);
	const struct pack_extent *const extent = &pack_extents_of(
		shared->description
			->pack)[tensors[jobs[first].tensor_index].extent_index];
	result_status status = STATUS_OK;
	bool large = extent->byte_count > own->source_bytes;
	for (uint32_t index = first;
	     index < first + count && !large && status == STATUS_OK; index++) {
		uint64_t bytes = 0;
		status = job_bytes(&jobs[index],
				   &tensors[jobs[index].tensor_index], &bytes);
		large = padded_bytes(bytes) > own->output_bytes;
	}
	if (status != STATUS_OK)
		return status;
	struct load_lane lane = *own;
	if (large) {
		(void)pthread_mutex_lock(&shared->large_lane);
		lane = (struct load_lane){
			request->source_workspace, request->output_workspace,
			request->source_workspace_byte_count,
			request->output_workspace_byte_count
		};
	}
	status = extent_read(request, shared->storage, extent, lane.source,
			     request->copies, result);
	struct staged_job staged[STAGED_JOBS_MOST];
	uint32_t staged_count = 0;
	uint64_t used = 0;
	for (uint32_t index = first;
	     index < first + count && status == STATUS_OK; index++) {
		const struct tp_load_job *const job = &jobs[index];
		const struct pack_tensor *const tensor =
			&tensors[job->tensor_index];
		uint64_t bytes = 0;
		status = job_bytes(job, tensor, &bytes);
		if (status == STATUS_OK &&
		    (staged_count == STAGED_JOBS_MOST ||
		     padded_bytes(bytes) > lane.output_bytes - used)) {
			status = batch_flush(request, lane.output, staged,
					     staged_count, result);
			staged_count = 0;
			used = 0;
		}
		struct staged_job *const item = &staged[staged_count];
		*item = (struct staged_job){ .job = job,
					     .output_offset = used };
		if (status == STATUS_OK)
			status = job_stage(job, tensor, lane.source,
					   lane.output + used, &item->padded,
					   &item->expected);
		if (status == STATUS_OK) {
			staged_count++;
			used = smaller_of(lane.output_bytes,
					  (used + item->padded +
					   STAGED_ALIGNMENT_BYTES - 1u) /
						  STAGED_ALIGNMENT_BYTES *
						  STAGED_ALIGNMENT_BYTES);
		}
	}
	if (status == STATUS_OK && staged_count != 0)
		status = batch_flush(request, lane.output, staged, staged_count,
				     result);
	if (large)
		(void)pthread_mutex_unlock(&shared->large_lane);
	return status;
}

static void *load_worker_run(void *context)
{
	struct load_shared *const shared = context;
	const struct backend_operations *const backend =
		shared->request->backend;
	/* the backend sleeps between looks at a card: with the default 50 us of slack a 20 us sleep lasts ~70 */
	(void)prctl(PR_SET_TIMERSLACK, WORKER_TIMER_SLACK_NANOSECONDS, 0, 0, 0);
	void *source = NULL, *output = NULL;
	/* Without its own slots a worker takes every extent through the large lane; slots need staging_retire, which
	 * takes them back with copies still in flight after a failure. */
	const bool slotted =
		backend->staging_retire != NULL &&
		backend->staging_allocate(shared->source_slot_bytes, &source) ==
			STATUS_OK &&
		backend->staging_allocate(shared->output_slot_bytes, &output) ==
			STATUS_OK;
	const struct load_lane own = { source, output,
				       slotted ? shared->source_slot_bytes : 0,
				       slotted ? shared->output_slot_bytes :
						 0 };
	struct tp_load_result result = { 0 };
	result_status status = STATUS_OK;
	uint32_t first = 0, count = 0;
	(void)pthread_mutex_lock(&shared->mutex);
	const uint32_t kind = shared->large_walker_taken ? EXTENT_SLOTTED :
							   EXTENT_LARGE;
	shared->large_walker_taken = true;
	(void)pthread_mutex_unlock(&shared->mutex);
	while (status == STATUS_OK &&
	       extent_claim(shared, kind, &first, &count))
		status = extent_load(shared, &own, first, count, &result);
	if (output != NULL)
		backend->staging_retire(output);
	if (source != NULL)
		backend->staging_retire(source);
	(void)pthread_mutex_lock(&shared->mutex);
	shared->status = shared->status == STATUS_OK ? status : shared->status;
	shared->result.source_bytes_read += result.source_bytes_read;
	shared->result.uploaded_bytes += result.uploaded_bytes;
	shared->result.extents_verified += result.extents_verified;
	shared->result.tensors_uploaded += result.tensors_uploaded;
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++)
		shared->result.card_receipts[card] +=
			result.card_receipts[card];
	(void)pthread_mutex_unlock(&shared->mutex);
	return NULL;
}

MUST_CHECK static result_status
load_pipelined(const struct tp_load_request *request,
	       const struct tp_load_plan *checked,
	       const struct storage_description *storage,
	       struct tp_load_result *result_out)
{
	struct load_shared shared = {
		.request = request,
		.description = &checked->description,
		.storage = storage,
		.source_slot_bytes =
			smaller_of(TP_LOAD_WORKER_SLOT_BYTES,
				   checked->source_workspace_byte_count),
		/* room for several jobs of an extent: their uploads overlap */
		.output_slot_bytes = smaller_of(
			TP_LOAD_WORKER_SLOT_BYTES,
			checked->source_workspace_byte_count >
					checked->output_workspace_byte_count ?
				checked->source_workspace_byte_count :
				checked->output_workspace_byte_count),
		.status = STATUS_OK
	};
	if (pthread_mutex_init(&shared.mutex, NULL) != 0)
		return STATUS_OUT_OF_MEMORY;
	if (pthread_mutex_init(&shared.large_lane, NULL) != 0) {
		(void)pthread_mutex_destroy(&shared.mutex);
		return STATUS_OUT_OF_MEMORY;
	}
	const int timer_slack = prctl(PR_GET_TIMERSLACK, 0, 0, 0, 0);
	threads_run(request->worker_count, load_worker_run, &shared);
	if (timer_slack > 0)
		(void)prctl(PR_SET_TIMERSLACK, (unsigned long)timer_slack, 0, 0,
			    0);
	(void)pthread_mutex_destroy(&shared.large_lane);
	(void)pthread_mutex_destroy(&shared.mutex);
	if (shared.status == STATUS_OK)
		*result_out = shared.result;
	return shared.status;
}

result_status tp_load_run(const struct tp_load_request *request,
			  struct tp_load_result *result_out)
{
	if (request == NULL || result_out == NULL || request->plan == NULL ||
	    request->storage == NULL ||
	    request->storage->reader_describe == NULL ||
	    request->storage->reader_read == NULL || request->reader == NULL ||
	    request->backend == NULL ||
	    request->backend->upload_begin == NULL ||
	    request->backend->upload_wait == NULL ||
	    request->backend->digest == NULL || request->cards == NULL ||
	    request->source_workspace == NULL ||
	    request->output_workspace == NULL ||
	    request->wait_limit_nanoseconds == 0 ||
	    request->worker_count > TP_LOAD_MAXIMUM_WORKERS)
		return STATUS_INVALID_ARGUMENT;
	struct tp_load_plan checked;
	result_status status =
		tp_load_prepare(&request->plan->description, &checked);
	if (status != STATUS_OK)
		return status;
	if (checked.source_workspace_byte_count !=
		    request->plan->source_workspace_byte_count ||
	    checked.output_workspace_byte_count !=
		    request->plan->output_workspace_byte_count ||
	    request->source_workspace_byte_count <
		    checked.source_workspace_byte_count ||
	    request->output_workspace_byte_count <
		    checked.output_workspace_byte_count ||
	    (uintptr_t)request->source_workspace %
			    BACKEND_STAGING_ALIGNMENT_BYTES !=
		    0 ||
	    !disjoint(request->source_workspace,
		      request->source_workspace_byte_count,
		      request->output_workspace,
		      request->output_workspace_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const struct tp_load_description *const description =
		&checked.description;
	const void *const protected_ranges[] = { request, request->plan,
						 description->jobs,
						 (const void *)request->cards,
						 result_out };
	const uint64_t protected_bytes[] = {
		sizeof(*request), sizeof(*request->plan),
		(uint64_t)description->job_count * sizeof(description->jobs[0]),
		(uint64_t)description->card_count * sizeof(request->cards[0]),
		sizeof(*result_out)
	};
	for (uint32_t range = 0;
	     range < sizeof(protected_ranges) / sizeof(protected_ranges[0]);
	     range++) {
		if (!disjoint(request->source_workspace,
			      request->source_workspace_byte_count,
			      protected_ranges[range],
			      protected_bytes[range]) ||
		    !disjoint(request->output_workspace,
			      request->output_workspace_byte_count,
			      protected_ranges[range], protected_bytes[range]))
			return STATUS_INVALID_ARGUMENT;
	}
	for (uint32_t index = 0; index < description->job_count; index++) {
		if (request->cards[description->jobs[index].card] == NULL)
			return STATUS_INVALID_ARGUMENT;
	}
	struct storage_description storage;
	status = request->storage->reader_describe(request->reader, &storage);
	if (status != STATUS_OK)
		return status;
	if (storage.byte_count !=
		    pack_header_of(description->pack)->file_byte_count ||
	    storage.copy_count == 0 ||
	    storage.copy_count > STORAGE_MAXIMUM_COPIES ||
	    storage.block_alignment_bytes == 0 ||
	    (storage.block_alignment_bytes &
	     (storage.block_alignment_bytes - 1u)) != 0 ||
	    PACK_BLOCK_BYTES % storage.block_alignment_bytes != 0)
		return STATUS_INVALID_ARGUMENT;
	if (request->worker_count != 0)
		return load_pipelined(request, &checked, &storage, result_out);
	const struct pack_tensor *const tensors =
		pack_tensors_of(description->pack);
	const struct pack_extent *const extents =
		pack_extents_of(description->pack);
	struct tp_load_result result = { 0 };
	uint32_t last_extent = UINT32_MAX;
	for (uint32_t index = 0;
	     index < description->job_count && status == STATUS_OK; index++) {
		const struct tp_load_job *const job = &description->jobs[index];
		const struct pack_tensor *const tensor =
			&tensors[job->tensor_index];
		if (tensor->extent_index != last_extent) {
			status = extent_read(request, &storage,
					     &extents[tensor->extent_index],
					     request->source_workspace, NULL,
					     &result);
			last_extent = tensor->extent_index;
		}
		if (status == STATUS_OK)
			status = tensor_upload(request, job, tensor, &result);
	}
	if (status == STATUS_OK)
		*result_out = result;
	return status;
}

/* The host selection shares only the pack/storage seam with the selected GPU loader above. */
static bool host_pack_apart(const struct pack_record *pack, const void *memory,
			    uint64_t bytes)
{
	const struct pack_header *const header = pack_header_of(pack);
	return disjoint(memory, bytes, header, sizeof(*header)) &&
	       disjoint(memory, bytes, pack_extents_of(pack),
			(uint64_t)header->extent_count *
				sizeof(struct pack_extent)) &&
	       disjoint(memory, bytes, pack_units_of(pack),
			(uint64_t)header->unit_count *
				sizeof(struct pack_unit_record_2)) &&
	       disjoint(memory, bytes, pack_tensors_of(pack),
			(uint64_t)header->tensor_count *
				sizeof(struct pack_tensor)) &&
	       disjoint(memory, bytes, pack_name_bytes_of(pack),
			header->name_byte_count);
}

MUST_CHECK static result_status
host_layout_check(const struct pack_record *pack, const uint64_t *offsets,
		  uint32_t extent_count, bool check_offsets,
		  struct tp_host_load_plan *plan)
{
	const struct pack_header *const header = pack_header_of(pack);
	if (extent_count != header->extent_count)
		return STATUS_INVALID_ARGUMENT;
	const struct pack_extent *const extents = pack_extents_of(pack);
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	*plan = (struct tp_host_load_plan){ .pack = pack,
					    .extent_offsets = offsets,
					    .extent_count = extent_count };
	for (uint32_t index = 0; index < extent_count; index++) {
		const struct pack_extent *const extent = &extents[index];
		if (extent->unit_index >= header->unit_count)
			return STATUS_INVALID_ARGUMENT;
		const bool selected = units[extent->unit_index].unit_kind ==
				      PLACEMENT_UNIT_HOST_ONLY;
		const uint64_t offset = selected ? plan->host_byte_count :
						   TP_HOST_LOAD_NOWHERE;
		if (check_offsets && offsets[index] != offset)
			return STATUS_INVALID_ARGUMENT;
		if (!selected)
			continue;
		if (extent->byte_count == 0 ||
		    extent->byte_count % PACK_BLOCK_BYTES != 0 ||
		    extent->byte_offset % PACK_BLOCK_BYTES != 0 ||
		    extent->byte_offset > header->file_byte_count ||
		    extent->byte_count >
			    header->file_byte_count - extent->byte_offset ||
		    extent->byte_count >
			    (uint64_t)PTRDIFF_MAX - plan->host_byte_count)
			return STATUS_INVALID_ARGUMENT;
		plan->host_byte_count += extent->byte_count;
		plan->host_extent_count++;
	}
	return STATUS_OK;
}

result_status tp_host_load_prepare(const struct pack_record *pack,
				   uint64_t *extent_offsets,
				   uint32_t extent_count,
				   struct tp_host_load_plan *plan_out)
{
	if (pack == NULL || extent_offsets == NULL || plan_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t map_bytes =
		(uint64_t)extent_count * sizeof(extent_offsets[0]);
	if (!disjoint(extent_offsets, map_bytes, plan_out, sizeof(*plan_out)) ||
	    !host_pack_apart(pack, extent_offsets, map_bytes) ||
	    !host_pack_apart(pack, plan_out, sizeof(*plan_out)))
		return STATUS_INVALID_ARGUMENT;
	struct tp_host_load_plan prepared;
	const result_status status = host_layout_check(
		pack, extent_offsets, extent_count, false, &prepared);
	if (status != STATUS_OK)
		return status;
	const struct pack_extent *const extents = pack_extents_of(pack);
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	uint64_t offset = 0;
	for (uint32_t index = 0; index < extent_count; index++) {
		const bool selected =
			units[extents[index].unit_index].unit_kind ==
			PLACEMENT_UNIT_HOST_ONLY;
		extent_offsets[index] = selected ? offset :
						   TP_HOST_LOAD_NOWHERE;
		offset += selected ? extents[index].byte_count : 0;
	}
	*plan_out = prepared;
	return STATUS_OK;
}

static bool host_request_apart(const struct tp_host_load_request *request,
			       const struct tp_host_load_result *result)
{
	const struct tp_host_load_plan *const plan = request->plan;
	const void *const metadata[] = { request, plan, request->storage,
					 plan->extent_offsets };
	const uint64_t bytes[] = { sizeof(*request), sizeof(*plan),
				   sizeof(*request->storage),
				   (uint64_t)plan->extent_count *
					   sizeof(plan->extent_offsets[0]) };
	for (uint32_t index = 0; index < sizeof(metadata) / sizeof(metadata[0]);
	     index++) {
		if (!disjoint(request->host_arena,
			      request->host_arena_byte_count, metadata[index],
			      bytes[index]) ||
		    !disjoint(result, sizeof(*result), metadata[index],
			      bytes[index]))
			return false;
	}
	return disjoint(request->host_arena, request->host_arena_byte_count,
			result, sizeof(*result)) &&
	       host_pack_apart(plan->pack, request->host_arena,
			       request->host_arena_byte_count) &&
	       host_pack_apart(plan->pack, result, sizeof(*result));
}

MUST_CHECK static result_status
host_extent_read(const struct tp_host_load_request *request,
		 const struct storage_description *storage,
		 const struct pack_extent *extent, unsigned char *destination,
		 struct tp_host_load_result *result)
{
	result_status last_status = STATUS_INPUT_OUTPUT_ERROR;
	bool wrong_digest = false;
	for (uint32_t copy = 0; copy < storage->copy_count; copy++) {
		uint64_t digest = 0;
		result->copy_retries += copy != 0 ? 1u : 0u;
		for (uint64_t done = 0; done < extent->byte_count;) {
			const uint64_t left = extent->byte_count - done;
			const uint64_t bytes = left < TP_HOST_LOAD_CHUNK_BYTES ?
						       left :
						       TP_HOST_LOAD_CHUNK_BYTES;
			result->requested_bytes += bytes;
			result->read_calls++;
			TIMER_BEGIN(tp_host_read);
			last_status = request->storage->reader_read(
				request->reader, copy,
				extent->byte_offset + done, destination + done,
				bytes);
			TIMER_END(tp_host_read);
			if (last_status != STATUS_OK)
				break;
			TIMER_BEGIN(tp_host_digest);
			digest += piece_digest(destination + done, bytes,
					       done / PACK_DIGEST_WORD_BYTES);
			TIMER_END(tp_host_digest);
			done += bytes;
		}
		if (last_status == STATUS_OK && digest == extent->digest) {
			result->extents_verified++;
			return STATUS_OK;
		}
		wrong_digest = wrong_digest || last_status == STATUS_OK;
	}
	return wrong_digest ? STATUS_CORRUPT_PACK : last_status;
}

/* What the threads reading one extent from one copy share. */
struct host_shared {
	const struct tp_host_load_request *request;
	const struct pack_extent *extent;
	unsigned char *destination;
	uint32_t copy; /* unless striped: then each chunk's is copy_take's */
	uint32_t copy_count;
	bool striped;
	pthread_mutex_t mutex; /* guards the four below */
	uint64_t next_byte; /* where the next chunk to take begins, in the extent */
	result_status read_status; /* the first failed read */
	uint64_t digest; /* the sum of the read chunks' */
	struct tp_host_load_result counts;
};

static void *host_chunks_read(void *context)
{
	struct host_shared *const shared = context;
	const uint64_t extent_bytes = shared->extent->byte_count;
	for (;;) {
		(void)pthread_mutex_lock(&shared->mutex);
		const uint64_t done = shared->next_byte;
		const bool taken = shared->read_status == STATUS_OK &&
				   done < extent_bytes;
		const uint64_t bytes =
			taken ? smaller_of(extent_bytes - done,
					   TP_HOST_LOAD_CHUNK_BYTES) :
				0;
		shared->next_byte += bytes;
		shared->counts.requested_bytes += bytes;
		shared->counts.read_calls += taken ? 1u : 0u;
		(void)pthread_mutex_unlock(&shared->mutex);
		if (!taken)
			return NULL;
		struct tp_load_copies *const copies = shared->request->copies;
		const uint32_t copy =
			shared->striped ? copy_take(copies, shared->copy_count,
						    bytes, 0) :
					  shared->copy;
		if (!shared->striped)
			copy_reading(copies, copy, bytes);
		TIMER_BEGIN(tp_host_read);
		const result_status status =
			shared->request->storage->reader_read(
				shared->request->reader, copy,
				shared->extent->byte_offset + done,
				shared->destination + done, bytes);
		TIMER_END(tp_host_read);
		copy_done(copies, copy, bytes, status == STATUS_OK);
		uint64_t digest = 0;
		TIMER_BEGIN(tp_host_digest);
		if (status == STATUS_OK)
			digest = piece_digest(shared->destination + done, bytes,
					      done / PACK_DIGEST_WORD_BYTES);
		TIMER_END(tp_host_digest);
		(void)pthread_mutex_lock(&shared->mutex);
		shared->digest += digest;
		shared->read_status = shared->read_status == STATUS_OK ?
					      status :
					      shared->read_status;
		(void)pthread_mutex_unlock(&shared->mutex);
	}
}

/* host_extent_read on worker_count threads: the copies in order, every chunk of one copy, then its digest. */
MUST_CHECK static result_status
host_extent_read_parallel(const struct tp_host_load_request *request,
			  const struct storage_description *storage,
			  const struct pack_extent *extent,
			  unsigned char *destination,
			  struct tp_host_load_result *result)
{
	result_status last_status = STATUS_INPUT_OUTPUT_ERROR;
	bool wrong_digest = false;
	/* turbo load: first every chunk from the copy soonest to deliver it; then, should that fail, copy by copy */
	const bool striped = request->copies != NULL && storage->copy_count > 1;
	for (uint32_t pass = striped ? 0u : 1u; pass <= storage->copy_count;
	     pass++) {
		const uint32_t copy = pass == 0 ? 0 : pass - 1u;
		result->copy_retries += pass > 1u ? 1u : 0u;
		struct host_shared shared = { .request = request,
					      .extent = extent,
					      .destination = destination,
					      .copy = copy,
					      .copy_count = storage->copy_count,
					      .striped = pass == 0,
					      .read_status = STATUS_OK };
		if (pthread_mutex_init(&shared.mutex, NULL) != 0)
			return STATUS_OUT_OF_MEMORY;
		threads_run(request->worker_count, host_chunks_read, &shared);
		(void)pthread_mutex_destroy(&shared.mutex);
		result->requested_bytes += shared.counts.requested_bytes;
		result->read_calls += shared.counts.read_calls;
		last_status = shared.read_status;
		if (last_status == STATUS_OK &&
		    shared.digest == extent->digest) {
			result->extents_verified++;
			return STATUS_OK;
		}
		/* a striped pass that differs names no copy: the passes copy by copy decide */
		wrong_digest = wrong_digest ||
			       (pass != 0 && last_status == STATUS_OK);
	}
	return wrong_digest ? STATUS_CORRUPT_PACK : last_status;
}

result_status tp_host_load_run(const struct tp_host_load_request *request,
			       struct tp_host_load_result *result_out)
{
	if (request == NULL || result_out == NULL || request->plan == NULL ||
	    request->plan->pack == NULL ||
	    request->plan->extent_offsets == NULL || request->storage == NULL ||
	    request->reader == NULL ||
	    request->storage->reader_describe == NULL ||
	    request->storage->reader_read == NULL ||
	    (request->host_arena == NULL &&
	     request->host_arena_byte_count != 0) ||
	    request->worker_count > TP_LOAD_MAXIMUM_WORKERS ||
	    !host_request_apart(request, result_out))
		return STATUS_INVALID_ARGUMENT;
	struct tp_host_load_plan checked;
	result_status status = host_layout_check(request->plan->pack,
						 request->plan->extent_offsets,
						 request->plan->extent_count,
						 true, &checked);
	if (status != STATUS_OK)
		return status;
	if (checked.host_byte_count != request->plan->host_byte_count ||
	    checked.host_extent_count != request->plan->host_extent_count ||
	    checked.host_byte_count > request->host_arena_byte_count ||
	    (checked.host_byte_count != 0 && request->host_arena == NULL))
		return STATUS_INVALID_ARGUMENT;
	struct storage_description storage;
	status = request->storage->reader_describe(request->reader, &storage);
	if (status != STATUS_OK)
		return status;
	if (storage.byte_count !=
		    pack_header_of(checked.pack)->file_byte_count ||
	    storage.copy_count == 0 ||
	    storage.copy_count > STORAGE_MAXIMUM_COPIES ||
	    storage.block_alignment_bytes == 0 ||
	    (storage.block_alignment_bytes &
	     (storage.block_alignment_bytes - 1u)) != 0 ||
	    PACK_BLOCK_BYTES % storage.block_alignment_bytes != 0 ||
	    (uintptr_t)request->host_arena % storage.block_alignment_bytes !=
		    0 ||
	    checked.host_byte_count > UINT64_MAX / storage.copy_count)
		return STATUS_INVALID_ARGUMENT;
	const struct pack_extent *const extents = pack_extents_of(checked.pack);
	struct tp_host_load_result result = { 0 };
	for (uint32_t index = 0;
	     index < checked.extent_count && status == STATUS_OK; index++) {
		if (checked.extent_offsets[index] == TP_HOST_LOAD_NOWHERE)
			continue;
		unsigned char *const destination =
			(unsigned char *)request->host_arena +
			checked.extent_offsets[index];
		status = request->worker_count != 0 ?
				 host_extent_read_parallel(request, &storage,
							   &extents[index],
							   destination,
							   &result) :
				 host_extent_read(request, &storage,
						  &extents[index], destination,
						  &result);
	}
	if (status == STATUS_OK)
		*result_out = result;
	return status;
}
