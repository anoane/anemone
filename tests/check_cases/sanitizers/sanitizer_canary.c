/*
 * sanitizer_canary: one deliberate fault for each argument, so that `make self_test` shows every sanitizer of the
 * debug and thread builds stopping a program once. It is not a test: when no sanitizer stops the fault it says so
 * and succeeds, and the self-test then fails.
 */
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_BYTE_COUNT 16
#define RACING_THREAD_COUNT 2
#define ADDITIONS_PER_THREAD 100000
#define BEYOND_EVERY_INTEGER 1.0e30f

static int racing_total = 0;

/* Address sanitizer. The pointer is read back through a volatile, so the compiler does not know the size of what
 * it points at and only the address sanitizer can see the write. */
static void heap_overflow(void)
{
    volatile size_t first_byte_past_the_block = BLOCK_BYTE_COUNT;
    char *volatile block = malloc(BLOCK_BYTE_COUNT);
    char *first_byte = block;
    if (first_byte != NULL) {
        first_byte[first_byte_past_the_block] = '\0';
        free(first_byte);
    }
}

/* Undefined-behaviour sanitizer, in the debug and in the thread build. */
static void signed_overflow(void)
{
    volatile int largest_integer = INT_MAX;
    (void)printf("%d\n", largest_integer + 1);
}

/* The float-cast-overflow part of the undefined-behaviour set, which -fsanitize=undefined alone leaves out. */
static void float_cast_overflow(void)
{
    volatile float beyond_every_integer = BEYOND_EVERY_INTEGER;
    (void)printf("%d\n", (int)beyond_every_integer);
}

static void *add_without_a_lock(void *unused)
{
    (void)unused;
    for (int addition = 0; addition < ADDITIONS_PER_THREAD; addition++) {
        racing_total++;
    }
    return NULL;
}

/* Thread sanitizer. */
static void data_race(void)
{
    pthread_t threads[RACING_THREAD_COUNT];
    int started_count = 0;
    while (started_count < RACING_THREAD_COUNT
           && pthread_create(&threads[started_count], NULL, add_without_a_lock, NULL) == 0) {
        started_count++;
    }
    for (int thread_index = 0; thread_index < started_count; thread_index++) {
        (void)pthread_join(threads[thread_index], NULL);
    }
    (void)printf("%d\n", racing_total);
}

int main(int argument_count, char **arguments)
{
    const char *fault_name = argument_count > 1 ? arguments[1] : "none";
    if (strcmp(fault_name, "heap_overflow") == 0) {
        heap_overflow();
    } else if (strcmp(fault_name, "signed_overflow") == 0) {
        signed_overflow();
    } else if (strcmp(fault_name, "float_cast_overflow") == 0) {
        float_cast_overflow();
    } else if (strcmp(fault_name, "data_race") == 0) {
        data_race();
    } else {
        (void)fprintf(stderr, "sanitizer_canary: no fault is named %s\n", fault_name);
        return EXIT_FAILURE;
    }
    (void)printf("sanitizer_canary: no sanitizer stopped the fault %s\n", fault_name);
    return EXIT_SUCCESS;
}
