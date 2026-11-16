/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Linux process collector. Scrape-only fixed buffers, bounded reads; no allocator or device calls. */
#include "server_metrics.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#define PROCESS_BUFFER_BYTES 4096u
#define PROCESS_READ_LIMIT 1024u
#define PROCESS_DECIMAL_BASE 10
#define PROCESS_STAT_LAST_FIELD 24u
#define PROCESS_DIRECTORY_NAME_OFFSET 19u
#define PROCESS_DIRECTORY_LENGTH_OFFSET 16u
#define PROCESS_LINE_BYTES 2048u
enum process_stat_field {
	PROCESS_STATE_FIELD = 3,
	PROCESS_USER_TICKS_FIELD = 14,
	PROCESS_SYSTEM_TICKS_FIELD = 15,
	PROCESS_START_FIELD = 22,
	PROCESS_VIRTUAL_FIELD = 23,
	PROCESS_RESIDENT_FIELD = 24
};
#define PROCESS_CPU_SAMPLE 3u

static bool proc_read(const char *path, char buffer[PROCESS_BUFFER_BYTES])
{
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	if (descriptor < 0)
		return false;
	size_t used = 0;
	for (uint32_t attempt = 0;
	     attempt < PROCESS_READ_LIMIT && used + 1u < PROCESS_BUFFER_BYTES;
	     attempt++) {
		const ssize_t got = read(descriptor, buffer + used,
					 PROCESS_BUFFER_BYTES - 1u - used);
		if (got > 0)
			used += (size_t)got;
		else if (got == 0)
			break;
		else if (errno != EINTR) {
			(void)close(descriptor);
			return false;
		}
	}
	const int closed = close(descriptor);
	buffer[used] = '\0';
	return closed == 0 && used + 1u < PROCESS_BUFFER_BYTES;
}

#define PROCESS_BOOT_LINE_BYTES 64u
static bool boot_seconds(uint64_t *seconds)
{
	const int descriptor = open("/proc/stat", O_RDONLY | O_CLOEXEC);
	if (descriptor < 0)
		return false;
	char buffer[PROCESS_BUFFER_BYTES], line[PROCESS_BOOT_LINE_BYTES];
	size_t used = 0;
	bool found = false, truncated = false;
	for (uint32_t attempt = 0; attempt < PROCESS_READ_LIMIT && !found;
	     attempt++) {
		const ssize_t got = read(descriptor, buffer, sizeof(buffer));
		if (got < 0 && errno == EINTR)
			continue;
		if (got <= 0)
			break;
		for (size_t index = 0; index < (size_t)got && !found; index++) {
			if (buffer[index] == '\n') {
				line[used] = '\0';
				if (!truncated &&
				    used > sizeof("btime ") - 1u &&
				    memcmp(line, "btime ",
					   sizeof("btime ") - 1u) == 0) {
					char *tail = NULL;
					errno = 0;
					const unsigned long long value =
						strtoull(
							line + sizeof("btime ") -
								1u,
							&tail,
							PROCESS_DECIMAL_BASE);
					if (errno == 0 && tail == line + used) {
						*seconds = (uint64_t)value;
						found = true;
					}
				}
				used = 0;
				truncated = false;
			} else if (used + 1u < sizeof(line))
				line[used++] = buffer[index];
			else
				truncated = true;
		}
	}
	const int closed = close(descriptor);
	return found && closed == 0;
}

static bool open_descriptors(uint64_t *count)
{
	const int descriptor =
		open("/proc/self/fd", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if (descriptor < 0)
		return false;
	unsigned char buffer[PROCESS_BUFFER_BYTES];
	uint64_t entries = 0;
	bool complete = false, valid = true;
	for (uint32_t attempt = 0; attempt < PROCESS_READ_LIMIT && valid;
	     attempt++) {
		const long got = syscall(SYS_getdents64, descriptor, buffer,
					 sizeof(buffer));
		if (got < 0 && errno == EINTR)
			continue;
		if (got <= 0) {
			complete = got == 0;
			break;
		}
		size_t position = 0;
		while (position < (size_t)got) {
			if ((size_t)got - position <=
			    PROCESS_DIRECTORY_NAME_OFFSET) {
				valid = false;
				break;
			}
			uint16_t length = 0;
			memcpy(&length,
			       buffer + position +
				       PROCESS_DIRECTORY_LENGTH_OFFSET,
			       sizeof(length));
			if (length <= PROCESS_DIRECTORY_NAME_OFFSET ||
			    length > (size_t)got - position) {
				valid = false;
				break;
			}
			const unsigned char *const name =
				buffer + position +
				PROCESS_DIRECTORY_NAME_OFFSET;
			const size_t bytes =
				(size_t)length - PROCESS_DIRECTORY_NAME_OFFSET;
			if (memchr(name, '\0', bytes) == NULL) {
				valid = false;
				break;
			}
			if (name[0] != '.')
				entries++;
			position += length;
		}
	}
	const int closed = close(descriptor);
	if (!complete || !valid || closed != 0)
		return false;
	*count = entries;
	return true;
}

static pthread_once_t process_once = PTHREAD_ONCE_INIT;
static uint64_t process_boot;
static bool process_boot_valid;
static long process_ticks, process_pages;

static void process_initialize(void)
{
	process_boot_valid = boot_seconds(&process_boot);
	process_ticks = sysconf(_SC_CLK_TCK);
	process_pages = sysconf(_SC_PAGESIZE);
}

void server_metrics_process_init(void)
{
	(void)pthread_once(&process_once, process_initialize);
}

static bool process_stat(uint64_t values[PROCESS_STAT_LAST_FIELD + 1u])
{
	char stat[PROCESS_BUFFER_BYTES];
	if (!proc_read("/proc/self/stat", stat))
		return false;
	char *token = strrchr(stat, ')');
	if (token == NULL)
		return false;
	token++;
	for (uint32_t field = PROCESS_STATE_FIELD;
	     field <= PROCESS_STAT_LAST_FIELD; field++) {
		while (*token == ' ')
			token++;
		char *const end = strchr(token, ' ');
		if (end == NULL)
			return false;
		static const uint32_t wanted[] = { 14, 15, 22, 23, 24 };
		for (uint32_t index = 0;
		     index < sizeof(wanted) / sizeof(wanted[0]); index++) {
			if (field != wanted[index])
				continue;
			char *tail = NULL;
			errno = 0;
			if (*token < '0' || *token > '9')
				return false;
			const unsigned long long value =
				strtoull(token, &tail, PROCESS_DECIMAL_BASE);
			if (errno != 0 || tail != end)
				return false;
			values[field] = (uint64_t)value;
		}
		token = end + 1;
	}
	return true;
}

result_status server_metrics_process(struct chat_buffer *output)
{
	if (output == NULL)
		return STATUS_INVALID_ARGUMENT;
	server_metrics_process_init();
	uint64_t values[PROCESS_STAT_LAST_FIELD + 1u] = { 0 }, fds = 0;
	const bool stat_valid = process_stat(values);
	const bool fds_valid = open_descriptors(&fds);
	struct rlimit limit = { 0 };
	const bool limit_valid = getrlimit(RLIMIT_NOFILE, &limit) == 0 &&
				 limit.rlim_cur != RLIM_INFINITY;
	static const char *const names[] = { "virtual_memory_bytes",
					     "resident_memory_bytes",
					     "start_time_seconds",
					     "cpu_seconds_total",
					     "open_fds",
					     "max_fds" };
	static const char *const helps[] = {
		"Virtual memory size in bytes.",
		"Resident memory size in bytes.",
		"Start time of the process since unix epoch in seconds.",
		"Total user and system CPU time spent in seconds.",
		"Number of open file descriptors.",
		"Maximum number of open file descriptors."
	};
	const bool valid[] = { stat_valid,
			       stat_valid && process_pages > 0,
			       stat_valid && process_boot_valid &&
				       process_ticks > 0,
			       stat_valid && process_ticks > 0,
			       fds_valid,
			       limit_valid };
	const double ticks = process_ticks > 0 ? (double)process_ticks : 1;
	const double samples[] = {
		(double)values[PROCESS_VIRTUAL_FIELD],
		(double)values[PROCESS_RESIDENT_FIELD] * (double)process_pages,
		(double)process_boot +
			(double)values[PROCESS_START_FIELD] / ticks,
		((double)values[PROCESS_USER_TICKS_FIELD] +
		 (double)values[PROCESS_SYSTEM_TICKS_FIELD]) /
			ticks,
		(double)fds,
		(double)limit.rlim_cur
	};
	for (uint32_t index = 0; index < sizeof(names) / sizeof(names[0]);
	     index++) {
		/* procfs is optional. A restricted or transiently unreadable source omits only its dependent
		 * families. */
		if (!valid[index] || !isfinite(samples[index]))
			continue;
		char line[PROCESS_LINE_BYTES];
		const int count = snprintf(
			line, sizeof(line),
			"# HELP process_%s %s\n# TYPE process_%s %s\nprocess_%s %.17g\n",
			names[index], helps[index], names[index],
			index == PROCESS_CPU_SAMPLE ? "counter" : "gauge",
			names[index], samples[index]);
		if (count < 0 || (size_t)count >= sizeof(line))
			return STATUS_OUT_OF_MEMORY;
		if (output->capacity != 0 &&
		    (size_t)count > output->capacity - output->count)
			return STATUS_OUT_OF_MEMORY;
		chat_buffer_write(output, line, (size_t)count);
	}
	return output->is_short ? STATUS_OUT_OF_MEMORY : STATUS_OK;
}
