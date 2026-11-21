/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "engram_fit.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "engram_disk.h"
#include "common/hooks.h"
#include "engine/storage/load/loader.h"
#include "engine/program/model_seam.h"
#include "engine/runtime/step/step.h"
#include "engine/storage/load/tp_load.h"

#define MEMINFO_PATH "/proc/meminfo"
#define MEMINFO_LINE_BYTES 256u
#define KIB_BYTES 1024u
#define GIB_BYTES 1073741824.0
#define DECIMAL_BASE 10
#define TP_WORKSPACES \
	2u /* a TP load's source and output workspaces, and each worker's slot of each */

static uint64_t sum(uint64_t left, uint64_t right)
{
	return left > UINT64_MAX - right ? UINT64_MAX : left + right;
}

static uint64_t product(uint64_t left, uint64_t right)
{
	return right != 0 && left > UINT64_MAX / right ? UINT64_MAX :
							 left * right;
}

result_status engram_fit_needs_of_pack(const struct pack_record *pack,
				       const struct engram_description *tables,
				       bool tensor, uint32_t staging_cards,
				       uint64_t piece_bytes,
				       uint32_t load_threads,
				       struct engram_fit_needs *needs_out)
{
	if (pack == NULL || tables == NULL || needs_out == NULL ||
	    tables->table_count == 0 ||
	    tables->table_count > ENGRAM_MAXIMUM_TABLES)
		return STATUS_INVALID_ARGUMENT;
	const struct pack_header *const header = pack_header_of(pack);
	const struct pack_extent *const extents = pack_extents_of(pack);
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	struct engram_fit_needs needs = {
		.table_count = tables->table_count,
		.staging_cards_most =
			!tensor && staging_cards == 0 ?
				ENGRAM_RAM_FIT_STAGING_CARDS_MOST :
				0
	};
	uint64_t table_bytes = 0, host_bytes = 0, largest_card_extent = 0;
	for (uint32_t table = 0; table < tables->table_count; table++) {
		const uint32_t unit = tables->table_unit_index[table];
		if (unit >= header->unit_count ||
		    units[unit].unit_kind != PLACEMENT_UNIT_HOST_ONLY ||
		    units[unit].extent_count != 1)
			return STATUS_INVALID_ARGUMENT;
		const uint64_t bytes =
			extents[units[unit].first_extent_index].byte_count;
		needs.table_blocks[table] = bytes / PACK_BLOCK_BYTES;
		table_bytes = sum(table_bytes, bytes);
	}
	for (uint32_t extent = 0; extent < header->extent_count; extent++) {
		const bool host_only =
			units[extents[extent].unit_index].unit_kind ==
			PLACEMENT_UNIT_HOST_ONLY;
		const uint64_t bytes = extents[extent].byte_count;
		host_bytes = host_only ? sum(host_bytes, bytes) : host_bytes;
		largest_card_extent =
			!host_only && bytes > largest_card_extent ?
				bytes :
				largest_card_extent;
	}
	needs.other_host_bytes = host_bytes - table_bytes;
	/* a TP program pins copies of the host experts it keeps, which lie in those units (step_experts.c) */
	needs.pinned_expert_bytes =
		!tensor ? 0 :
		needs.other_host_bytes < STEP_MOST_PINNED_EXPERT_BYTES ?
			  needs.other_host_bytes :
			  STEP_MOST_PINNED_EXPERT_BYTES;
	/* PP: the loader's staging slots (loader.c); TP: its workspaces of the largest card extent and each
	 * worker's slots (tp_load.c) */
	needs.staging_bytes =
		tensor ? sum(product(TP_WORKSPACES, largest_card_extent),
			     product((uint64_t)load_threads * TP_WORKSPACES,
				     TP_LOAD_WORKER_SLOT_BYTES)) :
			 product((uint64_t)(staging_cards != 0 ?
						    staging_cards :
						    needs.staging_cards_most) *
					 LOADER_STAGING_SLOTS_PER_CARD,
				 piece_bytes != 0 ? piece_bytes :
						    LOADER_CHUNK_BYTES);
	*needs_out = needs;
	return STATUS_OK;
}

bool engram_fit_memory_of(FILE *meminfo, uint64_t *total_out,
			  uint64_t *available_out)
{
	char line[MEMINFO_LINE_BYTES];
	unsigned long long total = 0, available = 0, kibibytes = 0;
	bool has_total = false, has_available = false;
	while ((!has_total || !has_available) &&
	       fgets(line, sizeof(line), meminfo) != NULL) {
		if (!has_total &&
		    sscanf(line, "MemTotal: %llu kB", &kibibytes) == 1) {
			has_total = true;
			total = kibibytes;
		} else if (!has_available &&
			   sscanf(line, "MemAvailable: %llu kB", &kibibytes) ==
				   1) {
			has_available = true;
			available = kibibytes;
		}
	}
	const bool valid = has_total && has_available &&
			   total <= UINT64_MAX / KIB_BYTES &&
			   available <= UINT64_MAX / KIB_BYTES;
	*total_out = valid ? (uint64_t)total * KIB_BYTES : 0;
	*available_out = valid ? (uint64_t)available * KIB_BYTES : 0;
	return valid;
}

void engram_fit_memory_read(struct engram_fit_needs *needs)
{
	FILE *const meminfo = fopen(MEMINFO_PATH, "re");
	needs->memory_total = needs->memory_available = 0;
	if (meminfo != NULL) {
		(void)engram_fit_memory_of(meminfo, &needs->memory_total,
					   &needs->memory_available);
		(void)fclose(meminfo);
	}
#if CHECKING_HOOKS_ENABLED
	const char *const seam = getenv(ENGRAM_RAM_FIT_MEMORY_SEAM);
	if (seam != NULL && seam[0] != '\0')
		needs->memory_available = strtoull(seam, NULL, DECIMAL_BASE);
#endif
}

uint64_t engram_fit_headroom_bytes(uint64_t memory_total)
{
	const uint64_t share = memory_total / ENGRAM_RAM_FIT_HEADROOM_SHARE;
	return share > ENGRAM_RAM_FIT_HEADROOM_LEAST_BYTES ?
		       share :
		       ENGRAM_RAM_FIT_HEADROOM_LEAST_BYTES;
}

static uint64_t tables_in_ram_bytes(const struct engram_fit_needs *needs,
				    uint32_t percent)
{
	uint64_t bytes = 0;
	for (uint32_t table = 0; table < needs->table_count; table++)
		bytes = sum(bytes,
			    product(percent >= ENGRAM_RAM_PERCENT ?
					    needs->table_blocks[table] :
					    engram_disk_ram_blocks(
						    needs->table_blocks[table],
						    percent),
				    PACK_BLOCK_BYTES));
	return bytes;
}

static uint64_t store_bytes(const struct engram_fit_needs *needs,
			    uint32_t percent)
{
	if (percent >= ENGRAM_RAM_PERCENT)
		return 0;
	/* the store's own sum less its prefix, which tables_in_ram_bytes counts */
	const uint64_t store =
		engram_disk_host_bytes(needs->table_blocks, needs->table_count,
				       percent, needs->cache_bytes);
	return sum(store - tables_in_ram_bytes(needs, percent),
		   needs->hot_bytes);
}

static uint64_t model_bytes(const struct engram_fit_needs *needs)
{
	return sum(sum(needs->other_host_bytes, needs->pinned_expert_bytes),
		   needs->staging_bytes);
}

uint64_t engram_fit_need_bytes(const struct engram_fit_needs *needs,
			       uint32_t percent)
{
	return sum(sum(tables_in_ram_bytes(needs, percent),
		       store_bytes(needs, percent)),
		   model_bytes(needs));
}

void engram_fit_decide(const struct engram_fit_needs *needs,
		       enum engram_ram_source source, uint32_t given_percent,
		       struct engram_fit_decision *decision_out)
{
	const uint64_t headroom =
		engram_fit_headroom_bytes(needs->memory_total);
	const uint64_t available = needs->memory_available;
	struct engram_fit_decision decision = { .headroom_bytes = headroom };
	const bool unknown = available == 0;
	if (source != ENGRAM_RAM_AUTO) {
		decision.percent = given_percent;
		decision.fit_case =
			unknown || sum(engram_fit_need_bytes(needs,
							     given_percent),
				       headroom) <= available ?
				ENGRAM_FIT_GIVEN :
				ENGRAM_FIT_GIVEN_OVER;
	} else if (unknown) {
		decision.percent = ENGRAM_RAM_PERCENT;
		decision.fit_case = ENGRAM_FIT_MEMORY_UNKNOWN;
	} else {
		/* the first p from 100 down whose need fits is the largest that fits */
		uint32_t percent = ENGRAM_RAM_PERCENT;
		while (percent > 0 && sum(engram_fit_need_bytes(needs, percent),
					  headroom) > available)
			percent--;
		const uint64_t disk_need = engram_fit_need_bytes(needs, 0);
		decision.percent = percent;
		decision.fit_case =
			percent == ENGRAM_RAM_PERCENT ? ENGRAM_FIT_EVERY_TABLE :
			percent > 0		      ? ENGRAM_FIT_SHARE :
			sum(disk_need, headroom) <= available ?
				      ENGRAM_FIT_DISK_ONLY :
			disk_need <= available ?
				      ENGRAM_FIT_DISK_ONLY_IN_HEADROOM :
				      ENGRAM_FIT_MODEL_OVER;
	}
	decision.need_bytes =
		sum(engram_fit_need_bytes(needs, decision.percent), headroom);
	*decision_out = decision;
}

static double gib(uint64_t bytes)
{
	return (double)bytes / GIB_BYTES;
}

static const char *case_text(enum engram_fit_case fit_case,
			     enum engram_ram_source source)
{
	switch (fit_case) {
	case ENGRAM_FIT_EVERY_TABLE:
		return "auto: every table fits";
	case ENGRAM_FIT_SHARE:
		return "auto: the largest whole share that fits";
	case ENGRAM_FIT_DISK_ONLY:
		return "auto: the disk store only";
	case ENGRAM_FIT_DISK_ONLY_IN_HEADROOM:
		return "auto: the disk store only, inside the headroom";
	case ENGRAM_FIT_MODEL_OVER:
		return "auto: refused, the model's own host memory does not fit";
	case ENGRAM_FIT_MEMORY_UNKNOWN:
		return "auto: MemAvailable unreadable, every table in RAM";
	case ENGRAM_FIT_GIVEN:
	case ENGRAM_FIT_GIVEN_OVER:
	default:
		return source == ENGRAM_RAM_PROFILE ?
			       "storage profile turbo_engram" :
			       "--engram_ram_percent";
	}
}

void engram_fit_print(FILE *output, const struct engram_fit_needs *needs,
		      const struct engram_fit_decision *decision,
		      enum engram_ram_source source)
{
	const uint32_t percent = decision->percent;
	const uint64_t available = needs->memory_available;
	char bound[ENGRAM_RAM_FIT_BOUND_TEXT_BYTES] = "";
	if (needs->staging_cards_most != 0)
		(void)snprintf(bound, sizeof(bound),
			       " (%" PRIu32
			       " cards at most: counted after the fit)",
			       needs->staging_cards_most);
	(void)fprintf(output, "engram ram percent: %" PRIu32 " (%s%s)\n",
		      percent, case_text(decision->fit_case, source),
		      decision->fit_case == ENGRAM_FIT_GIVEN_OVER ?
			      "; forced, it does not fit" :
			      "");
	(void)fprintf(
		output,
		"engram ram fit: percent=%" PRIu32
		" MemAvailable=%.2f GiB need=%.2f GiB = tables %.2f + disk store %.2f + other host %.2f + pinned "
		"experts %.2f + load staging %.2f%s + headroom %.2f GiB (max(%.2f GiB, MemTotal %.2f GiB / %u))\n",
		percent, gib(available), gib(decision->need_bytes),
		gib(tables_in_ram_bytes(needs, percent)),
		gib(store_bytes(needs, percent)), gib(needs->other_host_bytes),
		gib(needs->pinned_expert_bytes), gib(needs->staging_bytes),
		bound, gib(decision->headroom_bytes),
		gib(ENGRAM_RAM_FIT_HEADROOM_LEAST_BYTES),
		gib(needs->memory_total), ENGRAM_RAM_FIT_HEADROOM_SHARE);
	if (decision->fit_case == ENGRAM_FIT_DISK_ONLY_IN_HEADROOM ||
	    decision->fit_case == ENGRAM_FIT_GIVEN_OVER)
		(void)fprintf(
			output,
			"engram ram fit: warning: need %.2f GiB is over MemAvailable %.2f GiB%s\n",
			gib(decision->need_bytes), gib(available),
			decision->fit_case == ENGRAM_FIT_GIVEN_OVER ?
				"; the given percent is kept" :
				" with the headroom; the load fits without it");
	if (decision->fit_case == ENGRAM_FIT_MODEL_OVER) {
		const uint64_t store = store_bytes(needs, 0);
		const uint64_t items[] = { needs->other_host_bytes,
					   needs->pinned_expert_bytes,
					   needs->staging_bytes, store };
		const char *const names[] = { "other host-only units",
					      "pinned host experts",
					      "load staging",
					      "the disk store's digests" };
		size_t largest = 0;
		for (size_t item = 1; item < sizeof(items) / sizeof(items[0]);
		     item++)
			largest = items[item] > items[largest] ? item : largest;
		(void)fprintf(
			output,
			"engram ram fit: refused: the model's own host memory, %.2f GiB with no Engram table in RAM "
			"and no headroom, is %.2f GiB over MemAvailable %.2f GiB; its largest item is %s (%.2f GiB). "
			"This is the model's host memory, not Engram's: free host memory first\n",
			gib(engram_fit_need_bytes(needs, 0)),
			gib(engram_fit_need_bytes(needs, 0) - available),
			gib(available), names[largest], gib(items[largest]));
	}
}

void engram_fit_disk_refusal_print(FILE *output, uint32_t percent,
				   bool direct_storage)
{
	(void)fprintf(
		output,
		"engram ram fit: refused: auto chose %" PRIu32
		" (the tables do not fit in RAM) and the disk tables could not be prepared (above):"
		" free host memory, %s, or give --engram_ram_percent=100\n",
		percent,
		direct_storage ?
			"add a local pack copy (turbo load's replicas, or --engram_sources=PATH with "
			"--engram_ram_percent=0..99)" :
			"give a flat pack on local media as --engram_sources=PATH with "
			"--engram_ram_percent=0..99 (an EXL3 map's replicas are maps)");
}
