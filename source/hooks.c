/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/hooks.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NANOSECONDS_PER_SECOND 1000000000ull
#define LOWEST_PRINTABLE_BYTE 0x20
#define DELETE_BYTE 0x7f
#define REPLACEMENT_BYTE '?'

_Static_assert(sizeof(LOG_CUT_MARK) <= LOG_MESSAGE_CAPACITY,
	       "the cut mark fits in a message");

#if LOGGING_HOOKS_ENABLED

static const char *const log_level_names[LOG_LEVEL_COUNT] = {
	[LOG_LEVEL_ERROR] = "error",
	[LOG_LEVEL_WARNING] = "warning",
	[LOG_LEVEL_INFORMATION] = "information",
	[LOG_LEVEL_DEBUG] = "debug",
	[LOG_LEVEL_TRACE] = "trace",
};

#if CHECKING_HOOKS_ENABLED
#define DEFAULT_LOG_THRESHOLD LOG_LEVEL_INFORMATION
#else
#define DEFAULT_LOG_THRESHOLD LOG_LEVEL_WARNING
#endif

static _Atomic int log_threshold = DEFAULT_LOG_THRESHOLD;      /* always a level: set_threshold sees to it */

/* Zero when the clock cannot be read. Leaves errno as it was. */
static uint64_t monotonic_nanoseconds(void)
{
    struct timespec now;
    const int saved_errno = errno;
    const bool clock_was_read = clock_gettime(CLOCK_MONOTONIC, &now) == 0;
    errno = saved_errno;
    if (!clock_was_read || now.tv_sec < 0 || now.tv_nsec < 0) {
        return 0;
    }
    return (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec;
}

void log_set_threshold(enum log_level_code level)
{
    int threshold = (int)level;
    if (threshold < LOG_LEVEL_ERROR) {
	    threshold = LOG_LEVEL_ERROR;
    }
    if (threshold >= LOG_LEVEL_COUNT) {
	    threshold = LOG_LEVEL_COUNT - 1;
    }
    atomic_store(&log_threshold, threshold);
}

bool log_level_is_written(enum log_level_code level)
{
    const int level_number = (int)level;
    return level_number >= LOG_LEVEL_ERROR && level_number < LOG_LEVEL_COUNT &&
	   level_number <= atomic_load(&log_threshold);
}

void log_write(enum log_level_code level, const char *source_file,
	       int source_line, const char *format, ...)
{
    const int saved_errno = errno;
    char message[LOG_MESSAGE_CAPACITY];
    const char *text = message;
    const char *level_name = NULL;
    va_list values;
    int message_length = 0;
    uint64_t now = 0;
    if (format == NULL || !log_level_is_written(level)) {
	    return;
    }
    level_name = log_level_names[level];
    va_start(values, format);
    message_length = vsnprintf(message, sizeof(message), format, values);
    va_end(values);
    if (message_length < 0) {
        text = "(this message could not be formatted)";     /* the buffer holds nothing that may be read */
    } else {
        if ((size_t)message_length >= sizeof(message)) {
		memcpy(&message[sizeof(message) - sizeof(LOG_CUT_MARK)],
		       LOG_CUT_MARK, sizeof(LOG_CUT_MARK));
	}
        /* a log is one line: a byte from outside can neither end it nor drive a terminal */
        for (size_t byte_index = 0; message[byte_index] != '\0'; byte_index++) {
            const unsigned char byte = (unsigned char)message[byte_index];
            if (byte < LOWEST_PRINTABLE_BYTE || byte == DELETE_BYTE) {
                message[byte_index] = REPLACEMENT_BYTE;
            }
        }
    }
    now = monotonic_nanoseconds();
    /* one fprintf per line, so lines of different threads do not interleave */
    (void)fprintf(stderr, "%llu.%09llu anemone %s %s:%d: %s\n", (unsigned long long)(now / NANOSECONDS_PER_SECOND),
                  (unsigned long long)(now % NANOSECONDS_PER_SECOND), level_name != NULL ? level_name : "log",
                  source_file != NULL ? source_file : "?", source_line, text);
    errno = saved_errno;
}

#endif

#if TIMING_HOOKS_ENABLED

static _Atomic(struct hook_total_record *) registered_totals = NULL;

uint64_t hook_clock_nanoseconds(void)
{
    return monotonic_nanoseconds();
}

void hook_total_add(struct hook_total_record *hook_total, uint64_t amount)
{
    bool was_registered = false;
    if (hook_total == NULL || hook_total->name == NULL) {
        return;
    }
    was_registered = atomic_load(&hook_total->is_registered);
    /* the first event puts the total on the list: the compare-exchange lets exactly one thread do it, and next
     * is written before the list shows the total and never after */
    if (!was_registered && atomic_compare_exchange_strong(&hook_total->is_registered, &was_registered, true)) {
	    struct hook_total_record *head = atomic_load(&registered_totals);
	    do {
		    hook_total->next = head;
	    } while (!atomic_compare_exchange_weak(&registered_totals, &head,
						   hook_total));
    }
    atomic_fetch_add(&hook_total->total, amount);
    atomic_fetch_add(&hook_total->event_count, 1);
}

void hook_timer_end(struct hook_total_record *timer_total,
		    uint64_t started_nanoseconds)
{
    const uint64_t now = monotonic_nanoseconds();
    /* zero is a clock that could not be read: at either end it counts the event with no time */
    const bool both_ends_were_read = started_nanoseconds != 0 && now >= started_nanoseconds;
    hook_total_add(timer_total,
		   both_ends_were_read ? now - started_nanoseconds : 0);
}

/* The order of the report: counters before timers, then by name. Zero: the two are one line of the report. */
static int hook_total_order(const struct hook_total_record *left,
			    const struct hook_total_record *right)
{
    if (left->is_timer != right->is_timer) {
        return left->is_timer ? 1 : -1;
    }
    return strcmp(left->name, right->name);
}

void hooks_report(FILE *stream)
{
	struct hook_total_record *const first_total =
		atomic_load(&registered_totals);
	struct hook_total_record *printed_last = NULL;
	if (stream == NULL) {
		return;
	}
    /* each round prints the smallest name after the one printed last; no memory, and the places are few */
    for (;;) {
	    struct hook_total_record *smallest = NULL;
	    uint64_t total = 0;
	    uint64_t event_count = 0;
	    for (struct hook_total_record *candidate = first_total;
		 candidate != NULL; candidate = candidate->next) {
		    if ((printed_last == NULL ||
			 hook_total_order(candidate, printed_last) > 0) &&
			(smallest == NULL ||
			 hook_total_order(candidate, smallest) < 0)) {
			    smallest = candidate;
		    }
	    }
	if (smallest == NULL) {
            return;
        }
	for (struct hook_total_record *candidate = first_total;
	     candidate != NULL; candidate = candidate->next) {
		if (hook_total_order(candidate, smallest) == 0) {
			total += atomic_load(&candidate->total);
			event_count += atomic_load(&candidate->event_count);
		}
	}
	if (smallest->is_timer) {
            (void)fprintf(stream, "timer %s: %llu.%09llu s in %llu calls\n", smallest->name,
                          (unsigned long long)(total / NANOSECONDS_PER_SECOND),
                          (unsigned long long)(total % NANOSECONDS_PER_SECOND), (unsigned long long)event_count);
        } else {
            (void)fprintf(stream, "counter %s: %llu in %llu events\n", smallest->name, (unsigned long long)total,
                          (unsigned long long)event_count);
        }
        printed_last = smallest;
    }
}

#endif

#if CHECKING_HOOKS_ENABLED

static pthread_mutex_t armed_fault_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic bool armed_fault_exists = false;                 /* so a fault point costs one load when none is armed */
static char armed_fault_name
	[FAULT_NAME_CAPACITY]; /* this and the two below: guarded by the lock */
static uint64_t armed_fault_call_number = 0;
static uint64_t armed_fault_calls_seen = 0;

_Noreturn void assertion_failed(const char *condition_text,
				const char *source_file, int source_line)
{
	log_write(LOG_LEVEL_ERROR, source_file, source_line,
		  "assertion failed: %s",
		  condition_text != NULL ? condition_text : "?");
	abort();
}

bool fault_arm(const char *fault_name, uint64_t call_number)
{
	const size_t name_length =
		fault_name != NULL ? strnlen(fault_name, FAULT_NAME_CAPACITY) :
				     0;
	if (name_length == 0 || name_length >= FAULT_NAME_CAPACITY ||
	    call_number == 0) {
		return false;
	}
    if (pthread_mutex_lock(&armed_fault_lock) != 0) {
        return false;
    }
    memcpy(armed_fault_name, fault_name, name_length + 1);
    armed_fault_call_number = call_number;
    armed_fault_calls_seen = 0;
    atomic_store(&armed_fault_exists, true);
    (void)pthread_mutex_unlock(&armed_fault_lock);
    return true;
}

uint64_t fault_disarm(void)
{
    uint64_t calls_seen = 0;
    if (pthread_mutex_lock(&armed_fault_lock) != 0) {
        return 0;
    }
    calls_seen = armed_fault_calls_seen;
    armed_fault_calls_seen = 0;
    atomic_store(&armed_fault_exists, false);
    (void)pthread_mutex_unlock(&armed_fault_lock);
    return calls_seen;
}

bool fault_point_fires(const char *fault_name)
{
    bool fires = false;
    if (fault_name == NULL || !atomic_load(&armed_fault_exists) || pthread_mutex_lock(&armed_fault_lock) != 0) {
        return false;
    }
    if (atomic_load(&armed_fault_exists) && strcmp(armed_fault_name, fault_name) == 0) {
        armed_fault_calls_seen++;
        fires = armed_fault_calls_seen == armed_fault_call_number;
    }
    (void)pthread_mutex_unlock(&armed_fault_lock);
    return fires;
}

#endif
