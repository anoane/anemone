/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef ENGRAM_FIT_H
#define ENGRAM_FIT_H
/*
 * engram_fit.h: where the immutable n-gram tables live when no --engram_ram_percent is given (owner, 14 Nov 2026: in
 * RAM by default; on NVMe by itself, with the reason printed, when they do not fit). One decision before the load
 * (commands.c), from plan numbers only, against MemAvailable:
 *   need(p) = tables(p) + store(p) + model + headroom, where
 *   - tables(p) is the tables' RAM share: at 100 every table's extent (the loader's host arena), below it each
 *     table's engram_disk_ram_blocks (the disk store's own prefix);
 *   - store(p), below 100, is the disk store's block digests and cache (engram_disk_host_bytes);
 *   - model is what the load holds in host memory beside the tables: the other host-only units, the page-locked
 *     copies of host experts (TP only, at most those units; a PP placement that leaves experts in host memory is
 *     refused before the load) and the load's staging (PP: each card's slots of the load's piece, for at most
 *     ENGRAM_RAM_FIT_STAGING_CARDS_MOST cards when none are counted yet);
 *   - headroom is max(ENGRAM_RAM_FIT_HEADROOM_LEAST_BYTES, MemTotal / ENGRAM_RAM_FIT_HEADROOM_SHARE): what is planned
 *     only after the load (the step driver's staging, the session buffers, the backend's host allocations) and a
 *     margin, not a measured need.
 * Decision (14 Nov):
 *   - need(100) fits: 100, every table in RAM (the path before this decision);
 *   - else the largest whole p in 1..99 whose need fits: a RAM prefix of each table, the rest read from NVMe;
 *   - else 0 when need(0) fits, and 0 with a warning when need(0) fits only without the headroom;
 *   - else refused before the load: the model's own host memory (with the store's digests) is over MemAvailable
 *     with no Engram RAM and no headroom, a load that could only end out of memory;
 *   - MemAvailable unreadable: 100.
 * A given percent (the knob, or a storage profile's) is kept: its fit is printed, with a warning when it does not fit.
 * Outputs are the same at every percent (the disk store is checked identical to all-RAM), so this is a capacity
 * decision of no profile axis.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "engine/backend/backend_seam.h"
#include "engram.h"
#include "engine/formats/pack/pack.h"
#include "common/status.h"

#define ENGRAM_RAM_FIT_HEADROOM_LEAST_BYTES (8ull << 30u)
#define ENGRAM_RAM_FIT_HEADROOM_SHARE \
	32u /* of MemTotal: 15.7 GiB of the reference workstation's 503 GiB */
/* The PP staging's cards when no --card_order counts them: the fit runs before the cards are counted, so it charges the
 * most a load stages to, an upper bound the budget line names (5 cards stage 0.625 GiB of the 2 GiB it charges). */
#define ENGRAM_RAM_FIT_STAGING_CARDS_MOST BACKEND_MAXIMUM_CARDS
#define ENGRAM_RAM_FIT_BOUND_TEXT_BYTES 64u
/* Checking builds only: a MemAvailable in bytes that replaces /proc/meminfo's (the fit's test seam). */
#define ENGRAM_RAM_FIT_MEMORY_SEAM "ANEMONE_CHECKING_MEMORY_AVAILABLE_BYTES"

enum engram_ram_source { ENGRAM_RAM_AUTO, ENGRAM_RAM_KNOB, ENGRAM_RAM_PROFILE };
enum engram_fit_case {
	ENGRAM_FIT_EVERY_TABLE, /* 100 */
	ENGRAM_FIT_SHARE, /* 1..99 */
	ENGRAM_FIT_DISK_ONLY, /* 0 */
	ENGRAM_FIT_DISK_ONLY_IN_HEADROOM, /* 0, fitting only without the headroom: a warning */
	ENGRAM_FIT_MODEL_OVER, /* refused */
	ENGRAM_FIT_MEMORY_UNKNOWN, /* 100 */
	ENGRAM_FIT_GIVEN, /* the knob's or the profile's percent, which fits */
	ENGRAM_FIT_GIVEN_OVER /* the same, not fitting: a warning */
};

struct engram_fit_needs {
	uint64_t table_blocks[ENGRAM_MAXIMUM_TABLES];
	uint32_t table_count;
	uint32_t staging_cards_most; /* PP with cards not yet counted: ENGRAM_RAM_FIT_STAGING_CARDS_MOST, else 0 */
	uint64_t cache_bytes,
		hot_bytes; /* the disk store's cache and hot set (a given percent's) */
	uint64_t other_host_bytes; /* the host-only units beside the tables */
	uint64_t pinned_expert_bytes;
	uint64_t staging_bytes;
	uint64_t memory_total,
		memory_available; /* bytes; memory_available 0: unreadable */
};
struct engram_fit_decision {
	uint32_t percent;
	enum engram_fit_case fit_case;
	uint64_t need_bytes; /* need(percent), the headroom included */
	uint64_t headroom_bytes;
};

/* The needs of a pack's tables (tables from engram_open) before its load: tensor is a TP load (its host experts and
 * workspaces), staging_cards the cards a PP load stages to (0: not counted yet, ENGRAM_RAM_FIT_STAGING_CARDS_MOST),
 * piece_bytes a PP load's piece, its staging slots' size (0: LOADER_CHUNK_BYTES), and load_threads a TP load's
 * workers. The memory numbers stay 0 (engram_fit_memory_read). STATUS_INVALID_ARGUMENT for a table unit that is no
 * host-only unit of one extent, or no table. */
MUST_CHECK result_status engram_fit_needs_of_pack(
	const struct pack_record *pack, const struct engram_description *tables,
	bool tensor, uint32_t staging_cards, uint64_t piece_bytes,
	uint32_t load_threads, struct engram_fit_needs *needs_out);
/* MemTotal and MemAvailable of a /proc/meminfo text in bytes; false (outputs 0) when either is missing or wraps. */
bool engram_fit_memory_of(FILE *meminfo, uint64_t *total_out,
			  uint64_t *available_out);
/* Sets needs' memory numbers from /proc/meminfo (0 when unreadable); in a checking build ENGRAM_RAM_FIT_MEMORY_SEAM,
 * when set, replaces MemAvailable. */
void engram_fit_memory_read(struct engram_fit_needs *needs);
uint64_t engram_fit_headroom_bytes(uint64_t memory_total);
/* need(percent) as above, without the headroom (UINT64_MAX when the sum wraps). */
uint64_t engram_fit_need_bytes(const struct engram_fit_needs *needs,
			       uint32_t percent);
/* The decision (the header comment); a given source keeps given_percent. */
void engram_fit_decide(const struct engram_fit_needs *needs,
		       enum engram_ram_source source, uint32_t given_percent,
		       struct engram_fit_decision *decision_out);
/* Two lines (the percent with its source, then the whole budget), a third for a warning, and for a refusal the
 * shortfall and its largest item. */
void engram_fit_print(FILE *output, const struct engram_fit_needs *needs,
		      const struct engram_fit_decision *decision,
		      enum engram_ram_source source);
/* The refusal when auto chose percent (below 100) and no disk table could be prepared, with the remedies the storage
 * takes: a direct pack turbo load's replicas or --engram_sources with a percent (auto refuses the sources alone), an
 * EXL3 map (whose replicas are maps) a flat pack as --engram_sources with a percent; both can force 100. */
void engram_fit_disk_refusal_print(FILE *output, uint32_t percent,
				   bool direct_storage);
#endif
