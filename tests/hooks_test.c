/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hooks_test: the status texts and every hook, in whichever build it is compiled for. A section that needs a switch
 * is compiled only with it; where a hook is compiled out, the test shows that its argument was not evaluated.
 *
 * While the log is read back, descriptor 2 is a temporary file whose name the test prints first. The test removes
 * it at its end, so the file is still there, with any sanitizer report, only when the test died in such a section.
 */
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "anemone/status.h"

#define TEST_TIME_LIMIT_SECONDS 300             /* the alarm ends the test here: the deadline of every wait in it */
#define COUNTED_AMOUNT 7
#define COUNTER_PLACES 2                        /* places where the same timer and the same counter stand */
#define CHECKING_HOOKS_WITH_AN_ARGUMENT 4       /* the assertion and the three logs below a warning */
#define FAULT_FIRES_ON_CALL 3
#define FAULT_POINT_CALLS 5
#define WORKER_THREAD_COUNT 8
#define START_LINE_PARTIES (WORKER_THREAD_COUNT + 1)    /* the workers and the main thread */
#define EVENTS_PER_WORKER 20000
#define SHARED_FAULT_FIRES_ON_CALL EVENTS_PER_WORKER    /* a call that comes while every worker runs */
#define LONG_MESSAGE_EXCESS_BYTES 100           /* how far the long message goes past the capacity of a message */
#define CAPTURE_CAPACITY 8192                   /* more than the test logs between two readings */
#define TEXT_CAPACITY 4096                      /* more than the longest line the test compares */
#define FORMATTED_ARGUMENT_INDEX_OF_FORMAT 1
#define FORMATTED_ARGUMENT_INDEX_OF_VALUES 2

_Static_assert(
	sizeof(result_status) == sizeof(int) &&
		sizeof(enum log_level_code) == sizeof(int),
	"the test builds values outside both enumerations from the bytes of an int");

static int failures = 0;
static volatile unsigned evaluation_count = 0;
static char capture_path[] = "/tmp/anemone_hooks_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */
static pthread_barrier_t start_line;
static _Atomic int start_line_failures = 0;
static _Atomic int worker_fault_fires = 0;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s\n", what);
        failures++;
    }
}

/* Like expect, and shows the text that was not the expected one. */
static void expect_text(bool condition, const char *what, const char *text)
{
    expect(condition, what);
    if (!condition) {
        (void)printf("    the text was: \"%s\"\n", text);
    }
}

/* For what the test cannot go on without. */
static _Noreturn void give_up(const char *what)
{
    (void)printf("FAILED: %s\n", what);
    exit(EXIT_FAILURE);
}

/* Stands inside hook arguments and counts how often it ran: a hook that is compiled out, and a log that is dropped,
 * must not run it. */
static unsigned count_evaluation(void)
{
    evaluation_count = evaluation_count + 1;
    return 1;
}

#if LOGGING_HOOKS_ENABLED

static const char *formatted(const char *format, ...)
    __attribute__((format(printf, FORMATTED_ARGUMENT_INDEX_OF_FORMAT, FORMATTED_ARGUMENT_INDEX_OF_VALUES)));

/* Formats into one shared buffer: the result is good until the next call. Main thread only. */
static const char *formatted(const char *format, ...)
{
    static char text[TEXT_CAPACITY];
    va_list values;
    int length = 0;
    va_start(values, format);
    length = vsnprintf(text, sizeof(text), format, values);
    va_end(values);
    if (length < 0 || (size_t)length >= sizeof(text)) {
        expect(false, "a text of the test fits its buffer");
        text[0] = '\0';
    }
    return text;
}

/* Moves the cursor past the text when the cursor stands on it. */
static bool skip_text(const char **cursor, const char *text)
{
    const size_t length = strlen(text);
    if (length == 0 || strncmp(*cursor, text, length) != 0) {
        return false;
    }
    *cursor += length;
    return true;
}

/* Moves the cursor past a run of decimal digits; false when it stands on none. */
static bool skip_digits(const char **cursor)
{
    const size_t digit_count = strspn(*cursor, "0123456789");
    *cursor += digit_count;
    return digit_count > 0;
}

/* Moves the cursor past one log line of this file with this level and message, the line end included. The message
 * must not be a result of formatted(). */
static bool skip_log_line(const char **cursor, const char *level_name, const char *message)
{
    return skip_digits(cursor) && skip_text(cursor, ".") && skip_digits(cursor) &&
           skip_text(cursor, formatted(" anemone %s %s:", level_name, __FILE__)) && skip_digits(cursor) &&
           skip_text(cursor, formatted(": %s\n", message));
}

#endif

/* From here on, what is logged goes to the temporary file, from its start. */
static void capture_begin(void)
{
    if (capture_descriptor < 0 || fflush(stderr) != 0 || ftruncate(capture_descriptor, 0) != 0 ||
        lseek(capture_descriptor, 0, SEEK_SET) != 0 || dup2(capture_descriptor, STDERR_FILENO) != STDERR_FILENO) {
        give_up("the error stream could not be pointed at the temporary file");
    }
}

/* Copies what was logged since capture_begin into text, as a string, and puts descriptor 2 back. */
static void capture_end(char *text, size_t text_capacity)
{
    ssize_t byte_count = -1;
    if (fflush(stderr) == 0) {
        byte_count = pread(capture_descriptor, text, text_capacity - 1, 0);
    }
    if (dup2(error_stream_descriptor, STDERR_FILENO) != STDERR_FILENO || byte_count < 0) {
        give_up("the temporary file could not be read back");
    }
    text[(size_t)byte_count] = '\0';
}

static void test_status_texts(void)
{
    const int below_the_statuses = -1;
    result_status outside_status = STATUS_OK;
    for (int status = 0; status < STATUS_COUNT; status++) {
	    const char *text = status_text((result_status)status);
	    expect(text != NULL && text[0] != '\0' &&
			   strcmp(text, "unknown status") != 0,
		   "every status has a text of its own");
    }
    expect(strcmp(status_text(STATUS_OK), "ok") == 0, "the text of OK");
    expect(strcmp(status_text(STATUS_BUSY_VALUE), "busy") == 0,
	   "the text of BUSY");
    expect(strcmp(status_text(STATUS_COUNT), "unknown status") == 0,
	   "a value past the statuses has a text");
    /* built from its bytes: a cast of a constant that is no status is what the analyser refuses */
    memcpy(&outside_status, &below_the_statuses, sizeof(outside_status));
    expect(strcmp(status_text(outside_status), "unknown status") == 0,
	   "a value below the statuses has a text");
}

/* Diagnostic builds exercise formatting; release proves error/warning text and arguments disappear. */
static void test_log(void)
{
    char captured[CAPTURE_CAPACITY];
#if LOGGING_HOOKS_ENABLED
    char long_message[LOG_MESSAGE_CAPACITY + LONG_MESSAGE_EXCESS_BYTES];
    char expected_message[LOG_MESSAGE_CAPACITY];
    const int below_the_levels = -1;
    enum log_level_code outside_level = LOG_LEVEL_TRACE;
    const char *cursor = captured;
    int errno_after_log = 0;

    evaluation_count = 0;
    log_set_threshold(LOG_LEVEL_ERROR);
    capture_begin();
    LOG_WARNING_ENTRY("a warning under the error threshold %u",
		      count_evaluation());
    LOG_ERROR("an error is always written");
    capture_end(captured, sizeof(captured));
    expect_text(skip_log_line(&cursor, "error", "an error is always written") && *cursor == '\0',
                "the threshold drops a warning and never an error", captured);
    expect(evaluation_count == 0, "a log that is dropped does not evaluate its arguments");

    log_set_threshold(LOG_LEVEL_WARNING);
    capture_begin();
    LOG_WARNING_ENTRY("a warning that is written %u", count_evaluation());
    capture_end(captured, sizeof(captured));
    cursor = captured;
    expect_text(skip_log_line(&cursor, "warning", "a warning that is written 1") && *cursor == '\0',
                "a warning at the threshold is one whole line", captured);
    expect(evaluation_count == 1, "a log that is written evaluates its arguments once");

    memset(long_message, 'x', sizeof(long_message) - 1);
    long_message[sizeof(long_message) - 1] = '\0';
    memset(expected_message, 'x', sizeof(expected_message) - 1);
    expected_message[sizeof(expected_message) - 1] = '\0';
    capture_begin();
    LOG_ERROR("%.*s", LOG_MESSAGE_CAPACITY - 1, long_message);
    LOG_ERROR("%s", long_message);
    capture_end(captured, sizeof(captured));
    cursor = captured;
    expect_text(skip_log_line(&cursor, "error", expected_message), "the longest message that fits is whole", captured);
    memcpy(&expected_message[sizeof(expected_message) - sizeof(LOG_CUT_MARK)],
	   LOG_CUT_MARK, sizeof(LOG_CUT_MARK));
    expect_text(skip_log_line(&cursor, "error", expected_message) && *cursor == '\0',
                "a longer message is cut to the capacity and ends in the mark", captured);

    capture_begin();
    LOG_ERROR("%s",
	      "new\nline, escape\x1b, delete\x7f, tab\t, high byte \xc3\xa9.");
    capture_end(captured, sizeof(captured));
    cursor = captured;
    expect_text(skip_log_line(&cursor, "error", "new?line, escape?, delete?, tab?, high byte \xc3\xa9.") &&
                    *cursor == '\0',
                "control bytes are replaced, every other byte is kept, and the log stays one line", captured);

    /* with descriptor 2 closed the write fails and the C library sets errno: the writer must put it back */
    if (close(STDERR_FILENO) != 0) {
        give_up("descriptor 2 could not be closed");
    }
    errno = EDOM;
    LOG_ERROR("this line has nowhere to go");
    errno_after_log = errno;
    if (dup2(error_stream_descriptor, STDERR_FILENO) != STDERR_FILENO) {
        give_up("descriptor 2 could not be put back");
    }
    clearerr(stderr);
    expect(errno_after_log == EDOM, "the log writer preserves errno");

    memcpy(&outside_level, &below_the_levels, sizeof(outside_level));
    log_set_threshold(outside_level);
    expect(log_level_is_written(LOG_LEVEL_ERROR) &&
		   !log_level_is_written(LOG_LEVEL_WARNING),
	   "a threshold below the levels writes errors, and only errors");
    log_set_threshold(LOG_LEVEL_COUNT);
    expect(log_level_is_written(LOG_LEVEL_TRACE),
	   "a threshold past the levels writes every level");
    expect(!log_level_is_written(LOG_LEVEL_COUNT) &&
		   !log_level_is_written(outside_level),
	   "a value that is no level is never written");
    log_set_threshold(LOG_LEVEL_WARNING);
#else
    evaluation_count = 0;
    capture_begin();
    LOG_ERROR("hooks_test_release_error %u", count_evaluation());
    LOG_WARNING_ENTRY("hooks_test_release_warning %u", count_evaluation());
    capture_end(captured, sizeof(captured));
    expect_text(captured[0] == '\0', "release writes no error or warning log", captured);
    expect(evaluation_count == 0, "release logging evaluates no argument");
#endif
}

/* The assertion that holds and the logs below a warning: written with the checking hooks, absent without them. */
static void test_hooks_below_a_warning(void)
{
    char captured[CAPTURE_CAPACITY];

    evaluation_count = 0;
    capture_begin();
    LOG_INFORMATION("information under the warning threshold %u",
		    count_evaluation());
    capture_end(captured, sizeof(captured));
    expect_text(captured[0] == '\0', "a log that is dropped or compiled out writes nothing", captured);
    expect(evaluation_count == 0, "a log that is dropped or compiled out does not evaluate its arguments");

#if LOGGING_HOOKS_ENABLED
    log_set_threshold(LOG_LEVEL_TRACE);
#endif
    capture_begin();
    ASSERT(count_evaluation() > 0);
    LOG_INFORMATION("neither must this be in a release build %u",
		    count_evaluation());
    LOG_DEBUG_ENTRY("debug %u", count_evaluation());
    LOG_TRACE("trace %u", count_evaluation());
    capture_end(captured, sizeof(captured));
#if LOGGING_HOOKS_ENABLED
    log_set_threshold(LOG_LEVEL_WARNING);
#endif
#if CHECKING_HOOKS_ENABLED
    {
        const char *cursor = captured;
        expect_text(skip_log_line(&cursor, "information", "neither must this be in a release build 1") &&
                        skip_log_line(&cursor, "debug", "debug 1") && skip_log_line(&cursor, "trace", "trace 1") &&
                        *cursor == '\0',
                    "with the checking hooks, every log below a warning is one whole line", captured);
    }
    expect(evaluation_count == CHECKING_HOOKS_WITH_AN_ARGUMENT,
           "with the checking hooks, each hook evaluates its argument once");
#else
    expect_text(captured[0] == '\0', "without the checking hooks, nothing below a warning is written", captured);
    expect(evaluation_count == 0, "without the checking hooks, no hook evaluates its argument");
#endif
}

#if CHECKING_HOOKS_ENABLED

/* A failing assertion, in a child process: it must log its condition and end the child through abort(). */
static void test_failing_assertion(void)
{
    char captured[CAPTURE_CAPACITY];
    const char *cursor = captured;
    int wait_status = 0;
    pid_t child = 0;

    capture_begin();
    child = fork();
    if (child == 0) {
	    ASSERT(failures < 0);
	    _exit(EXIT_SUCCESS);
    }
    if (child < 0 || waitpid(child, &wait_status, 0) != child) {
        give_up("the child process of the failing assertion could not be started or awaited");
    }
    capture_end(captured, sizeof(captured));
    expect(WIFSIGNALED(wait_status) && WTERMSIG(wait_status) == SIGABRT, "a failing assertion ends in abort()");
    expect_text(skip_log_line(&cursor, "error", "assertion failed: failures < 0") && *cursor == '\0',
                "a failing assertion logs its condition as an error", captured);
}

static void test_fault_points(void)
{
	char longest_name[FAULT_NAME_CAPACITY + 1];
	int fired_calls = 0;

	memset(longest_name, 'n', sizeof(longest_name) - 1);
	longest_name[sizeof(longest_name) - 1] = '\0';
	expect(!fault_arm(NULL, 1) && !fault_arm("", 1) &&
		       !fault_arm(longest_name, 1) &&
		       !fault_arm("hooks_test_fault", 0),
	       "no name, an empty name, a too long name and call number zero are refused");
	expect(!FAULT_POINT(hooks_test_fault) && fault_disarm() == 0,
	       "a refused arming arms nothing");
	longest_name[FAULT_NAME_CAPACITY - 1] = '\0';
	expect(fault_arm(longest_name, 1) && fault_disarm() == 0,
	       "the longest name is accepted");

	expect(fault_arm("hooks_test_fault", FAULT_FIRES_ON_CALL),
	       "a fault is armed");
	for (int call = 1; call <= FAULT_POINT_CALLS; call++) {
		expect(!FAULT_POINT(hooks_test_other_fault),
		       "a fault point of another name does not fire");
		if (FAULT_POINT(hooks_test_fault)) {
			fired_calls++;
			expect(call == FAULT_FIRES_ON_CALL,
			       "the fault fires on the armed call");
		}
    }
    expect(fired_calls == 1, "an armed fault fires exactly once");
    expect(fault_disarm() == FAULT_POINT_CALLS,
	   "only the calls of the armed name are counted");

    expect(fault_arm("hooks_test_fault", 1) && fault_disarm() == 0,
	   "a fault disarmed before it is reached has seen no call");
    expect(!FAULT_POINT(hooks_test_fault), "a disarmed fault does not fire");

    expect(fault_arm("hooks_test_fault", 1) &&
		   fault_arm("hooks_test_other_fault", 1),
	   "a fault is armed while another one is");
    expect(!FAULT_POINT(hooks_test_fault) &&
		   FAULT_POINT(hooks_test_other_fault),
	   "arming again replaces the armed fault");
    expect(fault_disarm() == 1, "arming again starts the count again");
}

#endif

#if TIMING_HOOKS_ENABLED

static bool skip_counter_line(const char **cursor, const char *name, uint64_t total, uint64_t event_count)
{
    return skip_text(cursor, formatted("counter %s: %llu in %llu events\n", name, (unsigned long long)total,
                                       (unsigned long long)event_count));
}

/* The time of a timer is whatever the clock gave: only its form is compared. */
static bool skip_timer_line(const char **cursor, const char *name, uint64_t call_count)
{
    return skip_text(cursor, formatted("timer %s: ", name)) && skip_digits(cursor) && skip_text(cursor, ".") &&
           skip_digits(cursor) && skip_text(cursor, formatted(" s in %llu calls\n", (unsigned long long)call_count));
}

/* The report as one string, which the caller frees; NULL when it could not be written into memory. */
static char *report_text(void)
{
    char *text = NULL;
    size_t byte_count = 0;
    FILE *stream = open_memstream(&text, &byte_count);
    if (stream == NULL) {
        return NULL;
    }
    hooks_report(stream);
    if (fclose(stream) != 0) {
        free(text);
        return NULL;
    }
    return text;
}

/* Compares the whole report with everything the test has used so far: one line per name, counters by name and
 * then timers by name. */
static void expect_report(bool workers_have_run, const char *what)
{
    const uint64_t place_count = COUNTER_PLACES;
    const uint64_t worker_count = WORKER_THREAD_COUNT;
    const uint64_t worker_event_count = worker_count * EVENTS_PER_WORKER;
    const uint64_t worker_amount = worker_count * COUNTED_AMOUNT;
    char *report = report_text();
    const char *cursor = report;
    bool is_as_expected = false;

    if (report == NULL) {
        expect(false, "the report is written into memory");
        return;
    }
    is_as_expected = skip_counter_line(&cursor, "hooks_test_counter", place_count * COUNTED_AMOUNT, place_count) &&
                     skip_counter_line(&cursor, "hooks_test_evaluated", 1, 1);
    if (workers_have_run) {
        is_as_expected = is_as_expected &&
                         skip_counter_line(&cursor, "hooks_test_reach_first", worker_amount, worker_count) &&
                         skip_counter_line(&cursor, "hooks_test_reach_second", worker_amount, worker_count) &&
                         skip_counter_line(&cursor, "hooks_test_reach_third", worker_amount, worker_count) &&
                         skip_counter_line(&cursor, "hooks_test_shared_counter", worker_event_count * COUNTED_AMOUNT,
                                           worker_event_count) &&
                         skip_timer_line(&cursor, "hooks_test_shared_timer", worker_event_count);
    }
    is_as_expected = is_as_expected && skip_timer_line(&cursor, "hooks_test_timer", place_count) && *cursor == '\0';
    expect_text(is_as_expected, what, report);
    free(report);
}

#endif

/* One timer and one counter, each at two places, and a hook as the whole body of an `if`. */
static void test_timers_and_counters(void)
{
    evaluation_count = 0;
    {
	    TIMER_BEGIN(hooks_test_timer);
	    COUNTER_ADD(hooks_test_counter, COUNTED_AMOUNT);
	    TIMER_END(hooks_test_timer);
    }
    {
	    TIMER_BEGIN(hooks_test_timer);
	    COUNTER_ADD(hooks_test_counter, COUNTED_AMOUNT);
	    TIMER_END(hooks_test_timer);
    }
    /* no braces, on purpose: a hook is one statement, and the else must still belong to this if */
    if (failures >= 0)
	    COUNTER_ADD(hooks_test_evaluated, count_evaluation());
    else
	    ASSERT(failures < 0);
#if TIMING_HOOKS_ENABLED
    expect(evaluation_count == 1, "with the timing hooks, a counter evaluates its amount once");
    expect_report(false, "the report adds up the places of a name and orders its lines");
#else
    expect(evaluation_count == 0, "without the timing hooks, a counter does not evaluate its amount");
#endif
}

/* Waits until every thread stands at the same line, so that all of them reach the hook after it together. */
static void wait_at_start_line(void)
{
    const int result = pthread_barrier_wait(&start_line);
    if (result != 0 && result != PTHREAD_BARRIER_SERIAL_THREAD) {
        atomic_fetch_add(&start_line_failures, 1);
    }
}

/* Each counter and the timer are reached for the first time by all workers at once: every start line is one
 * more shot at the race between the threads that put a total on the list. */
static void *worker(void *unused)
{
    (void)unused;
    wait_at_start_line();
    COUNTER_ADD(hooks_test_reach_first, COUNTED_AMOUNT);
    wait_at_start_line();
    COUNTER_ADD(hooks_test_reach_second, COUNTED_AMOUNT);
    wait_at_start_line();
    COUNTER_ADD(hooks_test_reach_third, COUNTED_AMOUNT);
    wait_at_start_line();
    LOG_WARNING_ENTRY("a worker logs while the others do");
    wait_at_start_line();
    wait_at_start_line();
    for (int event = 0; event < EVENTS_PER_WORKER; event++) {
	    TIMER_BEGIN(hooks_test_shared_timer);
	    COUNTER_ADD(hooks_test_shared_counter, COUNTED_AMOUNT);
	    if (FAULT_POINT(hooks_test_shared_fault)) {
		    atomic_fetch_add(&worker_fault_fires, 1);
	    }
	    TIMER_END(hooks_test_shared_timer);
    }
    return NULL;
}

/* The hooks from several threads at once. The main thread stands at every start line too: it holds the error
 * stream only while the workers log, so a report of the thread sanitizer about the rest is seen. */
static void test_threads(void)
{
    pthread_t workers[WORKER_THREAD_COUNT];
    char captured[CAPTURE_CAPACITY];
#if LOGGING_HOOKS_ENABLED
    const char *cursor = captured;
    bool every_line_is_whole = true;
#endif

    if (pthread_barrier_init(&start_line, NULL, START_LINE_PARTIES) != 0) {
        give_up("the start line could not be made");
    }
#if CHECKING_HOOKS_ENABLED
    expect(fault_arm("hooks_test_shared_fault", SHARED_FAULT_FIRES_ON_CALL),
	   "the workers' fault is armed");
#endif
    for (int worker_index = 0; worker_index < WORKER_THREAD_COUNT; worker_index++) {
        if (pthread_create(&workers[worker_index], NULL, worker, NULL) != 0) {
            give_up("a worker thread could not be started");
        }
    }
    wait_at_start_line();                       /* the first counter */
    wait_at_start_line();                       /* the second counter */
    wait_at_start_line();                       /* the third counter */
    capture_begin();
    wait_at_start_line();                       /* the workers log */
    wait_at_start_line();                       /* every worker has logged */
    capture_end(captured, sizeof(captured));
    wait_at_start_line();                       /* the loop over the shared timer, counter and fault point */
#if TIMING_HOOKS_ENABLED
    {
        char *report_while_hooks_run = report_text();
        expect(report_while_hooks_run != NULL, "a report can be taken while hooks run");
        free(report_while_hooks_run);
    }
#endif
    for (int worker_index = 0; worker_index < WORKER_THREAD_COUNT; worker_index++) {
        if (pthread_join(workers[worker_index], NULL) != 0) {
            give_up("a worker thread could not be joined");
        }
    }
    expect(pthread_barrier_destroy(&start_line) == 0, "the start line is taken down");
    expect(atomic_load(&start_line_failures) == 0, "every thread met the others at every start line");

#if LOGGING_HOOKS_ENABLED
    for (int line_index = 0; line_index < WORKER_THREAD_COUNT; line_index++) {
        every_line_is_whole = skip_log_line(&cursor, "warning", "a worker logs while the others do") &&
                              every_line_is_whole;
    }
    expect_text(every_line_is_whole && *cursor == '\0', "each worker's warning is one whole line", captured);
#else
    expect_text(captured[0] == '\0', "release worker log hooks leave no output", captured);
#endif
#if CHECKING_HOOKS_ENABLED
    expect(atomic_load(&worker_fault_fires) == 1, "a fault armed for all threads fires in exactly one of them");
    expect(fault_disarm() == (uint64_t)WORKER_THREAD_COUNT * EVENTS_PER_WORKER,
	   "the armed fault point counts the calls of every thread");
#else
    expect(atomic_load(&worker_fault_fires) == 0, "without the checking hooks, a fault point never fires");
#endif
#if TIMING_HOOKS_ENABLED
    expect_report(true, "the report holds every event of every thread, each name once");
#endif
}

int main(void)
{
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    if (setvbuf(stdout, NULL, _IONBF, 0) != 0) {
        give_up("the output stream could not be made unbuffered");
    }
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        give_up("the temporary file that stands in for the error stream could not be made");
    }
    (void)printf("hooks_test: while the log is read back, the error stream is %s\n", capture_path);

    test_status_texts();
    test_log();
    test_hooks_below_a_warning();
#if CHECKING_HOOKS_ENABLED
    test_failing_assertion();
    test_fault_points();
#endif
    test_timers_and_counters();
    test_threads();

    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0,
           "the temporary file is removed");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("hooks_test passed (checking hooks %s, timing hooks %s)\n",
		 CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out",
		 TIMING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
