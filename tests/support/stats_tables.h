/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* stats_tables.h: test storage for a statistics record's slot tables (scheduler_stats.h: a copy fills the record's own
 * tables). Test support: no part of the engine. A test keeps a struct test_stats_tables beside its record (static or
 * heap: it is large) and names it in the record's initializer with TEST_STATS_TABLES. */
#ifndef TEST_STATS_TABLES_H
#define TEST_STATS_TABLES_H

#include "engine/runtime/scheduler/scheduler_stats.h"
#include "engine/server/live_metrics.h"

#define TEST_STATS_SLOTS \
	8u /* the slots of the engines and snapshots these tests build */

struct test_stats_tables {
	struct engine_slot_stats slots[TEST_STATS_SLOTS];
	struct engine_session_stats sessions[TEST_STATS_SLOTS];
};

/* designated initializers of a struct engine_stats whose tables are tables' */
#define TEST_STATS_TABLES(tables)                               \
	.slot_room = TEST_STATS_SLOTS, .slots = (tables).slots, \
	.sessions = (tables).sessions

/* a sampler's slot tables (live_metrics.h): every sample's slots */
struct test_sampler_tables {
	struct live_counter_slot slots[LIVE_SAMPLE_CAPACITY * TEST_STATS_SLOTS];
};
#define TEST_SAMPLER_TABLES(tables) \
	.slot_room = TEST_STATS_SLOTS, .slots = (tables).slots

/* the windows' slot tables: the initializer of a struct live_window_record [LIVE_WINDOW_COUNT] */
struct test_window_tables {
	struct live_slot_window slots[LIVE_WINDOW_COUNT][TEST_STATS_SLOTS];
};
#define TEST_WINDOWS_NAMED 2u /* the windows TEST_WINDOW_TABLES names */
_Static_assert(LIVE_WINDOW_COUNT == TEST_WINDOWS_NAMED,
	       "TEST_WINDOW_TABLES names every window");
#define TEST_WINDOW_TABLES(tables)                                             \
	{                                                                      \
		{ .slot_room = TEST_STATS_SLOTS, .slots = (tables).slots[0] }, \
		{                                                              \
			.slot_room = TEST_STATS_SLOTS,                         \
			.slots = (tables).slots[1]                             \
		}                                                              \
	}

#endif
