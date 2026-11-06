/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hooks.h: logging, assertions, fault points, timers and counters. Host C11 only: no kernel file includes it.
 *
 * Contract:
 *   - The build sets two switches, each to 0 or 1. CHECKING_HOOKS_ENABLED holds assertions, fault points
 *     and the information, debug and trace logs (debug and thread builds). TIMING_HOOKS_ENABLED holds
 *     timers, counters and hooks_report (debug, thread and profile builds). Logging is compiled only
 *     when either switch is enabled: release removes error/warning logging as well as optional diagnostics.
 *   - A hook whose switch is 0 leaves no call and no text; its arguments are read by the compiler and never
 *     evaluated. A log that the threshold drops does not evaluate its arguments either. So an argument of a hook
 *     never has a side effect.
 *   - Hooks may be used from any thread, and a hook preserves errno.
 *   - Every hook but TIMER_BEGIN is one expression or one statement and may be the whole body of an `if`
 *     without braces.
 *   - Logs in diagnostic builds. Errors are always written; the threshold may drop everything else. A log is
 *     one line on the standard error stream, `<seconds>.<nanoseconds> anemone <level> <file>:<line>: <message>`.
 *     In the message every byte
 *     below 0x20 and the byte 0x7f become '?', and a message that does not fit in LOG_MESSAGE_CAPACITY
 *     bytes is cut and ends in LOG_CUT_MARK. The process must make descriptors 0 to 2 valid before its
 *     first open and must ignore SIGPIPE; a log can block while the reader of the stream is stalled.
 *   - Timers and counters are named by an identifier. hooks_report() prints one line per name, counters
 *     by name and then timers by name: the same name at several places is one total. A counter's amount is
 *     unsigned: a signed or floating amount does not compile. TIMER_BEGIN declares two variables, so it
 *     stands where a declaration may (not as the body of an `if` or a loop, not right after a label), once per
 *     scope and not again in a scope inside it; TIMER_END follows in the same scope or one inside it. A
 *     return between the two is neither timed nor counted. A timer costs two clock reads and two atomic
 *     additions, and a total shared by threads a cache line transfer per event: hooks go around a launch, a read
 *     or a phase, never inside a loop over rows. A report taken while hooks run may show a total and its event
 *     count one event apart.
 *   - A fault point is a named place where a test makes the outside world fail, so that the engine's own handling
 *     runs: `bytes_read = FAULT_POINT(drive_read) ? -1 : pread(descriptor, destination, byte_count, 0);`
 *     STATUS_INJECTED_FAULT is only for places with no real failure to imitate. The name is an
 *     identifier, not a string. With its switch at 0 the macro is the constant false. An injected failure sets
 *     no errno: code that reads errno behind a fault point clears it first.
 */
#ifndef HOOKS_H
#define HOOKS_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "anemone/status.h"

#if !defined(CHECKING_HOOKS_ENABLED) || !defined(TIMING_HOOKS_ENABLED)
#error "the build must define ANEMONE_CHECKING_HOOKS_ENABLED and ANEMONE_TIMING_HOOKS_ENABLED, each as 0 or 1"
#endif

#define LOGGING_HOOKS_ENABLED (CHECKING_HOOKS_ENABLED || TIMING_HOOKS_ENABLED)

enum log_level_code {
	LOG_LEVEL_ERROR,
	LOG_LEVEL_WARNING,
	LOG_LEVEL_INFORMATION,
	LOG_LEVEL_DEBUG,
	LOG_LEVEL_TRACE,
	LOG_LEVEL_COUNT
};

#define LOG_MESSAGE_CAPACITY \
	1024 /* bytes of one message, its terminating zero included */
#define LOG_CUT_MARK " [cut]" /* the last bytes of a message that did not fit */
#define PRINTF_ARGUMENT_INDEX_OF_FORMAT 4
#define PRINTF_ARGUMENT_INDEX_OF_VALUES 5

/* Logging functions exist only in diagnostic builds. Release retains type-checked, unevaluated macros.
 * True when a message would be written now: always for an error, never for a value that is no level. */
bool log_level_is_written(enum log_level_code level);

/* Writes one line if the level is written. Use the macros: they skip the arguments of a log that is dropped. */
void log_write(enum log_level_code level, const char *source_file,
	       int source_line, const char *format, ...)
	__attribute__((format(printf, PRINTF_ARGUMENT_INDEX_OF_FORMAT,
			      PRINTF_ARGUMENT_INDEX_OF_VALUES)));

/* Messages above this level are dropped, errors never: a value that is no level becomes the nearest level.
 * Default: information with the checking hooks, warning without them. */
void log_set_threshold(enum log_level_code level);

/* The two macros below are the inside of the log macros, not hooks: the level is an enumerator. */
#define LOG_WHEN(is_written, level, ...)                                    \
	((is_written) ? log_write(level, __FILE__, __LINE__, __VA_ARGS__) : \
			(void)0)
#if LOGGING_HOOKS_ENABLED
#define LOG_AT(level, ...) \
	LOG_WHEN(log_level_is_written(level), level, __VA_ARGS__)
#else
#define LOG_AT(level, ...) LOG_WHEN(false, level, __VA_ARGS__)
#endif

#define LOG_ERROR(...) LOG_AT(LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_WARNING_ENTRY(...) LOG_AT(LOG_LEVEL_WARNING, __VA_ARGS__)

#if CHECKING_HOOKS_ENABLED

#define FAULT_NAME_CAPACITY \
	64 /* bytes of a fault name, its terminating zero included */

_Noreturn void assertion_failed(const char *condition_text,
				const char *source_file, int source_line);
bool fault_point_fires(const char *fault_name);

/* Arms one fault: the fault point of this name fires once, the call_number-th time it is reached from now on,
 * counted from one over all threads. The name is copied. Only one fault is armed at a time: arming again replaces
 * it. Returns false and changes nothing for a NULL, empty or too long name and for call number zero. */
MUST_CHECK bool fault_arm(const char *fault_name, uint64_t call_number);

/* Disarms, and returns how often the armed fault point was reached (0 when none was armed): a test expects a
 * number above zero, which catches a name that no fault point has. */
uint64_t fault_disarm(void);

#define LOG_INFORMATION(...) LOG_AT(LOG_LEVEL_INFORMATION, __VA_ARGS__)
#define LOG_DEBUG_ENTRY(...) LOG_AT(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_TRACE(...) LOG_AT(LOG_LEVEL_TRACE, __VA_ARGS__)
#define ASSERT(condition)        \
	((condition) ? (void)0 : \
		       assertion_failed(#condition, __FILE__, __LINE__))
#define ASSERT_TEXT(condition, expression_text) \
	((condition) ? (void)0 :                \
		       assertion_failed(expression_text, __FILE__, __LINE__))
#define FAULT_POINT(fault_name) fault_point_fires(#fault_name)

#else

/* Compiled out: the arguments stay in a branch that is never taken, or in a sizeof, which evaluates nothing. */
#define LOG_INFORMATION(...) LOG_WHEN(false, LOG_LEVEL_INFORMATION, __VA_ARGS__)
#define LOG_DEBUG_ENTRY(...) LOG_WHEN(false, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_TRACE(...) LOG_WHEN(false, LOG_LEVEL_TRACE, __VA_ARGS__)
#define ASSERT(condition) ((void)sizeof((condition) ? 1 : 0))
#define ASSERT_TEXT(condition, expression_text) \
	((void)sizeof((condition) ? 1 : 0))
#define FAULT_POINT(fault_name) false

#endif

#if TIMING_HOOKS_ENABLED

/* One timer or counter at one place. Only the macros below make one. */
struct hook_total_record {
	const char *name;
	_Atomic uint64_t
		total; /* nanoseconds for a timer, the sum of the amounts for a counter */
	_Atomic uint64_t event_count;
	_Atomic bool is_registered;
	bool is_timer;
	struct hook_total_record *next;
};

uint64_t hook_clock_nanoseconds(void);
void hook_total_add(struct hook_total_record *hook_total, uint64_t amount);
void hook_timer_end(struct hook_total_record *timer_total,
		    uint64_t started_nanoseconds);

/* Prints one line per name that was used: counters by name, then timers by name. */
void hooks_report(FILE *stream);

#define TIMER_BEGIN(timer_name)                                \
	static struct hook_total_record timer_##timer_name = { \
		.name = #timer_name, .is_timer = true          \
	};                                                     \
	const uint64_t timer_started_##timer_name = hook_clock_nanoseconds()
#define TIMER_END(timer_name) \
	hook_timer_end(&timer_##timer_name, timer_started_##timer_name)
#define COUNTER_ADD(counter_name, amount)                                  \
	do {                                                               \
		static struct hook_total_record counter_##counter_name = { \
			.name = #counter_name                              \
		};                                                         \
		hook_total_add(&counter_##counter_name, (amount));         \
	} while (false)

#else

#define TIMER_BEGIN(timer_name) ((void)0)
#define TIMER_END(timer_name) ((void)0)
#define COUNTER_ADD(counter_name, amount) ((void)sizeof((uint64_t)(amount)))

#endif

#endif
