/* A side effect in the argument of a hook is lost where the hook is compiled out. The last six lines are clean. */
#include "anemone/hooks.h"

void count_down(int remaining_rows, int row_count, int value, int other);

void count_down(int remaining_rows, int row_count, int value, int other)
{
	ASSERT(remaining_rows-- > 0);
	COUNTER_ADD(rows_seen, row_count++);
	LOG_DEBUG_ENTRY("%d", value = other);
	LOG_WARNING_ENTRY("%d", value += other);
	TIMER_ADD(rows_timer, value <<= 1);
	if (FAULT_POINT(--value)) {
		return;
	}
    LOG_TRACE("unbalanced %d", (value;
}

void clean_hooks(int value, int other);

void clean_hooks(int value, int other)
{
	ASSERT(value == other || value <= other || value >= other ||
	       value != other);
	LOG_ERROR("value = %d, other++ %d", value, other);
	COUNTER_ADD(rows_seen, value << 1 >> 1);
	TIMER_BEGIN(rows_timer);
	TIMER_END(rows_timer);
	value = FAULT_POINT(rows_fault) ? other : value;
}
