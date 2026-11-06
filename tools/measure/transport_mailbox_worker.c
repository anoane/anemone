/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* C11 driver-API worker: one visible GPU per process, fixed mapped pages and no host work per message. */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "transport_mailbox_records.h"
#include "transport_vmm_mailbox.h"

struct transport_control {
    _Atomic uint32_t stage[TRANSPORT_PARTICIPANTS];
    _Atomic uint32_t failed;
    _Atomic uint64_t began[TRANSPORT_PARTICIPANTS];
    _Atomic uint64_t ended[TRANSPORT_PARTICIPANTS];
};
_Static_assert(sizeof(struct transport_control) <= TRANSPORT_CONTROL_BYTES, "control area");

struct transport_case { uint32_t byte_count; uint32_t fault; };
static const struct transport_case cases[] = {
    {TRANSPORT_TEN_KIB, TRANSPORT_FAULT_NONE}, {TRANSPORT_FORTY_KIB, TRANSPORT_FAULT_NONE},
    {TRANSPORT_EIGHTY_KIB, TRANSPORT_FAULT_NONE}, {TRANSPORT_ONE_MIB, TRANSPORT_FAULT_NONE},
    {TRANSPORT_MAXIMUM_PAYLOAD, TRANSPORT_FAULT_NONE}, {TRANSPORT_TEN_KIB, TRANSPORT_FAULT_PAYLOAD},
    {TRANSPORT_TEN_KIB, TRANSPORT_FAULT_TRAILER_SEQUENCE}, {TRANSPORT_TEN_KIB, TRANSPORT_FAULT_TRAILER_SIZE},
    {TRANSPORT_TEN_KIB, TRANSPORT_FAULT_AHEAD}, {TRANSPORT_TEN_KIB, TRANSPORT_FAULT_MISSING_SENDER}
};
static const uint32_t block_sizes[] = {32u, 64u, 128u, TRANSPORT_THREADS};
_Static_assert(sizeof(cases) / sizeof(cases[0]) == TRANSPORT_TEST_COUNT, "case count");

static int checked(CUresult result, const char *phase)
{
    if (result == CUDA_SUCCESS) {
        return 1;
    }
    const char *name = "unknown";
    if (cuGetErrorName(result, &name) != CUDA_SUCCESS) {
        name = "unknown";
    }
    fprintf(stderr, "phase=%s result=%s\n", phase, name);
    return 0;
}

static uint64_t now_nanoseconds(void)
{
    struct timespec time_value;
    if (clock_gettime(CLOCK_MONOTONIC, &time_value) != 0) {
        return 0;
    }
    return (uint64_t)time_value.tv_sec * TRANSPORT_NANOSECONDS_PER_SECOND + (uint64_t)time_value.tv_nsec;
}

static int meet(struct transport_control *control, uint32_t role, uint32_t stage)
{
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = TRANSPORT_HOST_POLL_NANOSECONDS};
    const uint64_t began = now_nanoseconds();
    atomic_store_explicit(&control->stage[role], stage, memory_order_release);
    while (atomic_load_explicit(&control->stage[1u - role], memory_order_acquire) < stage) {
        if (atomic_load_explicit(&control->failed, memory_order_acquire) || began == 0 ||
            now_nanoseconds() - began > TRANSPORT_HOST_WAIT_NANOSECONDS) {
            return 0;
        }
        if (nanosleep(&pause, NULL) != 0 && errno != EINTR) {
            return 0;
        }
    }
    return 1;
}

static int expected_identity(CUdevice device)
{
    static const CUdevice_attribute attributes[] = {CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                                   CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID};
    static const char *const variables[] = {
        "ANEMONE_EXPECTED_DOMAIN", "ANEMONE_EXPECTED_BUS", "ANEMONE_EXPECTED_DEVICE"};
    for (size_t index = 0; index < sizeof(attributes) / sizeof(attributes[0]); index++) {
        const char *text = getenv(variables[index]);
        char *end = NULL;
        int actual = 0;
        if (text == NULL || *text == '\0') {
            return 0;
        }
        errno = 0;
        unsigned long wanted = strtoul(text, &end, TRANSPORT_DECIMAL_BASE);
        if (errno || end == text || *end != '\0' || wanted > INT32_MAX ||
            !checked(cuDeviceGetAttribute(&actual, attributes[index], device), "identity") || actual < 0 ||
            (unsigned long)actual != wanted) {
            return 0;
        }
    }
    return 1;
}

static int bounded_stream_wait(CUstream stream)
{
    const uint64_t began = now_nanoseconds();
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = TRANSPORT_HOST_POLL_NANOSECONDS};
    CUresult result;
    do {
        result = cuStreamQuery(stream);
        if (result == CUDA_SUCCESS) {
            return 1;
        }
        if (result != CUDA_ERROR_NOT_READY || began == 0 ||
            now_nanoseconds() - began > TRANSPORT_HOST_WAIT_NANOSECONDS) {
            fprintf(stderr, "stream failed or exceeded host deadline\n");
            return 0;
        }
        if (nanosleep(&pause, NULL) != 0 && errno != EINTR) {
            return 0;
        }
    } while (1);
}

int main(int argument_count, char **arguments)
{
    if ((argument_count != TRANSPORT_ARGUMENT_COUNT && argument_count != TRANSPORT_VMM_ARGUMENT_COUNT) ||
        (argument_count == TRANSPORT_VMM_ARGUMENT_COUNT && strcmp(arguments[TRANSPORT_ARGUMENT_ROUTE], "vmm") != 0) ||
        (strcmp(arguments[1], "0") != 0 && strcmp(arguments[1], "1") != 0)) {
        fprintf(stderr, "usage: worker role(0|1) mailbox_file module_file sender_label [vmm]\n");
        return EXIT_FAILURE;
    }
    const int use_vmm = argument_count == TRANSPORT_VMM_ARGUMENT_COUNT;
    const int check_only = getenv("DEVICE_TRANSPORT_CHECK") != NULL;
    const uint32_t role = (uint32_t)(arguments[1][0] - '0');
    char *end = NULL;
    errno = 0;
    const unsigned long sender = strtoul(arguments[TRANSPORT_ARGUMENT_SENDER], &end, TRANSPORT_DECIMAL_BASE);
    if (errno || end == arguments[TRANSPORT_ARGUMENT_SENDER] || *end ||
	sender >= EXCHANGE_MAXIMUM_SITES) {
	    return EXIT_FAILURE;
    }
    int descriptor = open(arguments[TRANSPORT_ARGUMENT_MAILBOX], O_RDWR | O_NOFOLLOW);
    struct stat metadata;
    if (descriptor < 0) {
        return EXIT_FAILURE;
    }
    if (fstat(descriptor, &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_uid != geteuid() ||
        metadata.st_size != TRANSPORT_MAILBOX_BYTES) {
        if (close(descriptor) != 0) {
            fprintf(stderr, "close failed\n");
        }
        return EXIT_FAILURE;
    }
    void *mailbox = mmap(NULL, TRANSPORT_MAILBOX_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    const int closed = close(descriptor) == 0;
    if (mailbox == MAP_FAILED) {
        return EXIT_FAILURE;
    }
    struct transport_control *control = mailbox;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction functions[TRANSPORT_KERNEL_COUNT] = {0};
    CUstream stream = NULL;
    CUdeviceptr mapped = 0;
    CUdeviceptr input = 0;
    CUdeviceptr output = 0;
    CUdeviceptr card_status = 0;
    struct transport_mapping owned = {0};
    struct transport_mapping peer = {0};
    CUdeviceptr slot_address = 0;
    CUdeviceptr credit_address = 0;
    uint64_t credit_stride = TRANSPORT_SLOT_STRIDE;
    uint32_t *pattern = NULL;
    int registered = 0;
    int success = 0;
    int count = 0;
    uint32_t stage = 0;
    uint64_t sequence_base = 0;
    if (!closed || !atomic_is_lock_free(&control->stage[0]) || !atomic_is_lock_free(&control->began[0])) {
        goto cleanup;
    }
    fprintf(stderr, "role=%u phase=initialize\n", role);
    if (!checked(cuInit(0), "initialize") || !checked(cuDeviceGetCount(&count), "count") || count != 1 ||
        !checked(cuDeviceGet(&device, 0), "device") || !expected_identity(device) ||
        !checked(cuDevicePrimaryCtxSetFlags(device, CU_CTX_SCHED_AUTO), "context_flags") ||
        !checked(cuDevicePrimaryCtxRetain(&context, device), "context") ||
        !checked(cuCtxSetCurrent(context), "current") ||
        !checked(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "stream") ||
        !checked(cuModuleLoad(&module, arguments[TRANSPORT_ARGUMENT_MODULE]), "module") ||
        !checked(cuMemAlloc(&input, TRANSPORT_MAXIMUM_PAYLOAD), "input") ||
        !checked(cuMemAlloc(&output, TRANSPORT_MAXIMUM_PAYLOAD), "output") ||
        !checked(cuMemAlloc(&card_status, sizeof(struct transport_status)), "status")) {
        goto cleanup;
    }
    static const char *const kernel_names[] = {"transport_mailbox", "transport_mailbox_cached",
        "transport_mailbox_validate", "transport_mailbox_cached_validate", "transport_mailbox_fault"};
    for (size_t index = 0; index < sizeof(kernel_names) / sizeof(kernel_names[0]); index++) {
        if (!checked(cuModuleGetFunction(&functions[index], module, kernel_names[index]), "function")) {
            goto cleanup;
        }
    }
    fprintf(stderr, "role=%u phase=register\n", role);
    if (!checked(cuMemHostRegister(mailbox, TRANSPORT_MAILBOX_BYTES,
                                   CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP), "register")) {
        goto cleanup;
    }
    registered = 1;
    if (!checked(cuMemHostGetDevicePointer(&mapped, mailbox, 0), "mapped_pointer")) {
        goto cleanup;
    }
    slot_address = mapped + TRANSPORT_CONTROL_BYTES;
    credit_address = slot_address + TRANSPORT_CREDIT_OFFSET;
    if (use_vmm) {
	    const char *socket_text = getenv("INFERENCE_SOCKET_FD");
	    if (socket_text == NULL || *socket_text == '\0') {
		    goto cleanup;
	    }
        errno = 0;
        const unsigned long socket_value = strtoul(socket_text, &end, TRANSPORT_DECIMAL_BASE);
        if (errno || end == socket_text || *end || socket_value > INT32_MAX) {
            goto cleanup;
        }
	const size_t owned_bytes =
		role == 0 ? TRANSPORT_PARTICIPANTS * EXCHANGE_LINE_BYTES :
			    TRANSPORT_PARTICIPANTS * TRANSPORT_SLOT_STRIDE;
	fprintf(stderr, "role=%u phase=vmm_create\n", role);
        if (!transport_mapping_create(&owned, device, owned_bytes) ||
            !checked(cuMemsetD8(owned.address, 0, owned.byte_count), "vmm_clear") ||
            !transport_mapping_exchange(&peer, &owned, device, (int)socket_value)) {
            if (close((int)socket_value) != 0) {
                fprintf(stderr, "socket close failed\n");
            }
            goto cleanup;
        }
        if (close((int)socket_value) != 0) {
            goto cleanup;
        }
        slot_address = role == 0 ? peer.address : owned.address;
        credit_address = role == 0 ? owned.address : peer.address;
	credit_stride = EXCHANGE_LINE_BYTES;
    }
    pattern = malloc(TRANSPORT_MAXIMUM_PAYLOAD);
    if (pattern == NULL) {
        goto cleanup;
    }
    for (uint32_t word = 0; word < TRANSPORT_MAXIMUM_PAYLOAD / TRANSPORT_WORD_BYTES; word++) {
        pattern[word] = ((word / TRANSPORT_VECTOR_WORDS) * TRANSPORT_PATTERN_MULTIPLIER) ^ TRANSPORT_PATTERN_SEED;
    }
    if (!checked(cuMemcpyHtoD(input, pattern, TRANSPORT_MAXIMUM_PAYLOAD), "upload") ||
        !checked(cuMemsetD8(output, 0, TRANSPORT_MAXIMUM_PAYLOAD), "clear_output") || !meet(control, role, ++stage)) {
        goto cleanup;
    }
    fprintf(stderr, "role=%u phase=measure\n", role);
    for (uint32_t test = 0; test < TRANSPORT_TEST_COUNT; test++) {
        const uint32_t bytes = cases[test].byte_count;
        const uint32_t fault = cases[test].fault;
        const uint32_t rounds = fault == TRANSPORT_FAULT_NONE ? (check_only ? TRANSPORT_CHECK_ROUNDS :
            (bytes < TRANSPORT_LARGE_BOUNDARY ? TRANSPORT_SMALL_ROUNDS : TRANSPORT_LARGE_ROUNDS)) :
            TRANSPORT_CORRUPT_SEQUENCE + 1u;
        const uint32_t samples = fault == TRANSPORT_FAULT_NONE && !check_only ?
            TRANSPORT_SAMPLE_COUNT + TRANSPORT_WARMUP_SAMPLES : 1u;
        for (uint32_t sample = 0; sample < samples; sample++) {
          const uint32_t variants = fault == TRANSPORT_FAULT_NONE ? TRANSPORT_VARIANT_COUNT : 1u;
          for (uint32_t position = 0; position < variants; position++) {
            const uint32_t variant = (position + sample) % variants;
            const uint32_t cached = fault == TRANSPORT_FAULT_NONE ? variant % TRANSPORT_PARTICIPANTS : 1u;
            const uint32_t threads = fault == TRANSPORT_FAULT_NONE ?
                block_sizes[variant / TRANSPORT_PARTICIPANTS] : TRANSPORT_THREADS;
            const uint32_t kernel = fault != TRANSPORT_FAULT_NONE ? TRANSPORT_KERNEL_FAULT :
                cached + (sample == 0 ? TRANSPORT_PARTICIPANTS : 0);
            struct transport_status result;
            struct transport_arguments record = {
                .mailbox = slot_address, .control = mapped, .credit = credit_address,
                .input = input, .output = output,
                .status = card_status, .sequence_base = sequence_base,
                .deadline_nanoseconds = TRANSPORT_WAIT_NANOSECONDS,
                .parity_stride = TRANSPORT_SLOT_STRIDE, .byte_count = bytes, .round_count = rounds,
                .credit_parity_stride = credit_stride,
                .role = role, .fault = fault, .sender = (uint32_t)sender, .reserved_zero = 0
            };
            /* Negative tests get a fresh lifetime only after both cards completed the previous lifetime. */
            if (fault != TRANSPORT_FAULT_NONE && role == 0) {
		    memset((unsigned char *)mailbox + TRANSPORT_READY_OFFSET, 0,
			   TRANSPORT_PARTICIPANTS * EXCHANGE_LINE_BYTES);
		    memset((unsigned char *)mailbox + TRANSPORT_CONTROL_BYTES,
			   0,
			   TRANSPORT_MAILBOX_BYTES - TRANSPORT_CONTROL_BYTES);
		    sequence_base = 0;
		    record.sequence_base = 0;
            } else if (fault != TRANSPORT_FAULT_NONE) {
                sequence_base = 0;
                record.sequence_base = 0;
            }
            if (fault != TRANSPORT_FAULT_NONE && use_vmm &&
                !checked(cuMemsetD8(owned.address, 0, owned.byte_count), "vmm_reset")) {
                goto cleanup;
            }
            if (!checked(cuMemsetD8(card_status, 0, sizeof(result)), "clear_status") || !meet(control, role, ++stage)) {
                goto cleanup;
            }
            void *parameters[] = {&record};
            atomic_store_explicit(&control->began[role], now_nanoseconds(), memory_order_release);
            if (!checked(cuLaunchKernel(functions[kernel], 1, 1, 1, threads, 1, 1, 0, stream, parameters, NULL),
                         "launch") || !bounded_stream_wait(stream)) {
                goto cleanup;
            }
            atomic_store_explicit(&control->ended[role], now_nanoseconds(), memory_order_release);
            if (!checked(cuMemcpyDtoH(&result, card_status, sizeof(result)), "status_download") ||
                !meet(control, role, ++stage)) {
                goto cleanup;
            }
            int passed;
            if (fault == TRANSPORT_FAULT_NONE) {
                passed = result.failure == TRANSPORT_FAILURE_NONE && result.completed == rounds + 1u &&
                         result.mismatched_vectors == 0 && result.sequence == sequence_base + rounds + 1u;
                if (role != 0 && passed) {
                    if (!checked(cuMemcpyDtoH(pattern, output, bytes), "payload_download")) {
                        goto cleanup;
                    }
                    for (uint32_t word = 0; word < bytes / TRANSPORT_WORD_BYTES; word++) {
                        const uint32_t expected = ((word / TRANSPORT_VECTOR_WORDS) * TRANSPORT_PATTERN_MULTIPLIER) ^
                            TRANSPORT_PATTERN_SEED ^ (uint32_t)(sequence_base + rounds);
                        passed = passed && pattern[word] == expected;
                    }
                }
            } else if (role == 0) {
                passed = result.failure == TRANSPORT_FAILURE_LATE || fault == TRANSPORT_FAULT_MISSING_SENDER;
            } else {
                const uint32_t expected = fault == TRANSPORT_FAULT_PAYLOAD ? TRANSPORT_FAILURE_PAYLOAD :
                    fault == TRANSPORT_FAULT_AHEAD ? TRANSPORT_FAILURE_AHEAD :
                    fault == TRANSPORT_FAULT_MISSING_SENDER ? TRANSPORT_FAILURE_LATE : TRANSPORT_FAILURE_TRAILER;
                passed = result.failure == expected;
            }
            const uint64_t first = atomic_load_explicit(&control->began[0], memory_order_acquire);
            const uint64_t second = atomic_load_explicit(&control->began[1], memory_order_acquire);
            const uint64_t first_end = atomic_load_explicit(&control->ended[0], memory_order_acquire);
            const uint64_t second_end = atomic_load_explicit(&control->ended[1], memory_order_acquire);
            const uint64_t start = first < second ? first : second;
            const uint64_t finish = first_end > second_end ? first_end : second_end;
            {
                printf("{\"bytes\":%u,\"sample\":%u,\"warmup\":%s,\"rounds\":%u,\"fault\":%u,"
                       "\"threads\":%u,\"cached_publication\":%s,"
                       "\"all_rank_wall_us_per_message\":%.6f,\"participant_device_us_per_message\":%.6f,"
                       "\"failure\":%u,\"completed\":%u,\"passed\":%s}\n",
                       bytes, sample, sample < TRANSPORT_WARMUP_SAMPLES && fault == 0 ? "true" : "false", rounds,
                       fault, threads, cached ? "true" : "false",
                       (double)(finish - start) * TRANSPORT_MICROSECONDS_PER_NANOSECOND / rounds,
                       (double)result.elapsed_nanoseconds * TRANSPORT_MICROSECONDS_PER_NANOSECOND / rounds,
                       result.failure, result.completed, passed ? "true" : "false");
                if (fflush(stdout) != 0) {
                    goto cleanup;
                }
            }
            if (!passed || !meet(control, role, ++stage)) {
                goto cleanup;
            }
            sequence_base += rounds + 1u;
          }
        }
    }
    success = 1;
cleanup:
    if (!success) {
        atomic_store_explicit(&control->failed, 1, memory_order_release);
    }
    free(pattern);
    if (!transport_mapping_release(&peer)) {
        success = 0;
    }
    if (!transport_mapping_release(&owned)) {
        success = 0;
    }
    if (registered && !checked(cuMemHostUnregister(mailbox), "unregister")) {
        success = 0;
    }
    if (card_status && !checked(cuMemFree(card_status), "free_status")) {
        success = 0;
    }
    if (output && !checked(cuMemFree(output), "free_output")) {
        success = 0;
    }
    if (input && !checked(cuMemFree(input), "free_input")) {
        success = 0;
    }
    if (module && !checked(cuModuleUnload(module), "unload")) {
        success = 0;
    }
    if (stream && !checked(cuStreamDestroy(stream), "destroy_stream")) {
        success = 0;
    }
    if (context && !checked(cuDevicePrimaryCtxRelease(device), "release_context")) {
        success = 0;
    }
    if (munmap(mailbox, TRANSPORT_MAILBOX_BYTES) != 0) {
        success = 0;
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
