/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * launch_path_probe: what the host adds to a decode step by the way it issues a card's kernels and waits for them.
 *
 * The owner's present engine launches about 28 kernels per layer (1,099 per token on three cards) and spends 22.6
 * ms per token of which 16.3 ms is kernel time. Engines in the clones differ on the cure: fewer and fused kernels,
 * one captured graph per card per step (ds4, FreeToken, vLLM), and a host that spins on a word in mapped memory
 * instead of waiting in the driver (TensorFold, b12x). This probe prices each choice on this machine's cards for
 * a step of one card: LAYERS layers of K small kernels each, every kernel reading its input from the one before.
 *
 *   issue    individual launches on one stream  |  one captured graph, replayed
 *   wait     the driver's stream synchronize     |  the host spinning on a word the last kernel writes into
 *                                                   page-locked host memory mapped on the card
 *
 * Reported per card and per K (4, 12 and 28 kernels per layer): microseconds per step until the host knows the
 * step is done, the part of it spent inside the issuing calls, and the kernels' own time (the same step timed by
 * events on the card), so that the difference is what the host and the driver add. Every step's result is checked
 * (each kernel adds one to a counter chain: the last value must be the number of kernels), and --corrupt drops one
 * kernel from the step, which the check must report.
 *
 *   launch_path_probe [--layers N] [--steps N] [--corrupt]
 *
 * Measurement tool: not part of the engine.
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <vector>

#define DEFAULT_LAYER_COUNT 13                   /* a third of the model: one card's share */
#define DEFAULT_STEP_COUNT 400
#define WARM_UP_STEPS 40
#define WORK_BLOCKS 64                           /* each kernel: a small grid, as the engine's small kernels are */
#define WORK_THREADS_PER_BLOCK 128
#define WORK_ELEMENTS (WORK_BLOCKS * WORK_THREADS_PER_BLOCK)
#define SPIN_LIMIT_SECONDS 2.0
#define MICROSECONDS_PER_SECOND 1.0e6
#define MAX_CARDS 16
#define GRAPH_STEP_BASE 1000000u                 /* a graph's steps are numbered from here: no individual step has such a number */
static const int kernels_per_layer_cases[] = {4, 12, 28};
#define KERNEL_COUNT_CASES (sizeof(kernels_per_layer_cases) / sizeof(kernels_per_layer_cases[0]))

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        cudaError_t check_cuda_status = (call);                                                       \
        if (check_cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                      \
                    cudaGetErrorString(check_cuda_status));                                           \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

/* One small kernel of the chain: every element becomes its predecessor's value plus one. */
__global__ void chain_kernel(const unsigned int *input, unsigned int *output)
{
    const unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
    output[element] = input[element] + 1u;
}

/* The last kernel of a step: tells the host, through memory mapped on the card, which step is done. */
__global__ void done_kernel(const unsigned int *last_output, volatile unsigned int *host_word, unsigned int step_number)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        host_word[1] = last_output[0];
        host_word[0] = step_number;
    }
}

static double seconds_now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct step_state {
    unsigned int *buffers[2];                    /* the chain ping-pongs between two buffers on the card */
    volatile unsigned int *host_word;            /* [0] the step number, [1] the chain's last value */
    unsigned int *step_number_on_card;           /* the captured graph's own count of the steps it has run */
    cudaStream_t stream;
    int kernel_count;
    int dropped_kernel;                          /* --corrupt: this kernel of the step is not issued; -1 for none */
};

/* The same for a captured graph, which cannot take a new argument per replay: it counts the steps on the card. */
__global__ void done_from_card_kernel(const unsigned int *last_output, volatile unsigned int *host_word, unsigned int *step_counter)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned int step_number = step_counter[0] + 1u;
        step_counter[0] = step_number;
        host_word[1] = last_output[0];
        host_word[0] = step_number;
    }
}

static void issue_chain(const step_state &state)
{
    CHECK_CUDA(cudaMemsetAsync(state.buffers[0], 0, WORK_ELEMENTS * sizeof(unsigned int), state.stream));
    for (int kernel = 0; kernel < state.kernel_count; kernel++) {
        if (kernel == state.dropped_kernel) {
            continue;
        }
        chain_kernel<<<WORK_BLOCKS, WORK_THREADS_PER_BLOCK, 0, state.stream>>>(state.buffers[kernel % 2], state.buffers[(kernel + 1) % 2]);
    }
}

static const unsigned int *last_buffer(const step_state &state)
{
    const int issued = state.kernel_count - (state.dropped_kernel >= 0 ? 1 : 0);
    return state.buffers[issued % 2];
}

/* Spins until the card has written this step's number, or SPIN_LIMIT_SECONDS pass. */
static bool spin_for_step(const step_state &state, unsigned int step_number)
{
    const double deadline = seconds_now() + SPIN_LIMIT_SECONDS;
    while (state.host_word[0] != step_number) {
        _mm_pause();
        if (seconds_now() > deadline) {
            return false;
        }
    }
    return true;
}

struct timings {
    std::vector<double> until_done, issuing;
    bool results_right = true;
};

static void report(const char *label, timings &measured, double kernel_microseconds)
{
    std::sort(measured.until_done.begin(), measured.until_done.end());
    std::sort(measured.issuing.begin(), measured.issuing.end());
    const size_t middle = measured.until_done.size() / 2;
    printf("      %-46s %8.1f us per step (%7.1f us inside the issuing calls); kernels alone %8.1f us: the host adds %6.1f us%s\n",
           label, measured.until_done[middle], measured.issuing[middle], kernel_microseconds,
           measured.until_done[middle] - kernel_microseconds, measured.results_right ? "" : "   RESULT WRONG");
}

static bool measure_card(int card, int layer_count, int step_count, bool corrupt)
{
    cudaDeviceProp properties;
    bool all_right = true;
    CHECK_CUDA(cudaSetDevice(card));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, card));
    printf("  card %d %s\n", card, properties.name);
    for (size_t count_case = 0; count_case < KERNEL_COUNT_CASES; count_case++) {
        step_state state;
        void *host_memory = NULL;
        cudaEvent_t started_on_card, finished_on_card;
        cudaGraph_t graph;
        cudaGraphExec_t graph_instance;
        state.kernel_count = layer_count * kernels_per_layer_cases[count_case];
        state.dropped_kernel = corrupt ? state.kernel_count / 2 : -1;
        CHECK_CUDA(cudaMalloc(&state.buffers[0], WORK_ELEMENTS * sizeof(unsigned int)));
        CHECK_CUDA(cudaMalloc(&state.buffers[1], WORK_ELEMENTS * sizeof(unsigned int)));
        CHECK_CUDA(cudaMalloc(&state.step_number_on_card, sizeof(unsigned int)));
        CHECK_CUDA(cudaHostAlloc(&host_memory, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
        state.host_word = (volatile unsigned int *)host_memory;
        state.host_word[0] = 0u;
        CHECK_CUDA(cudaStreamCreate(&state.stream));
        CHECK_CUDA(cudaEventCreate(&started_on_card));
        CHECK_CUDA(cudaEventCreate(&finished_on_card));

        /* the kernels' own time: the step between two events on the card, no host in between */
        float kernel_milliseconds = 0.0f;
        for (int repeat = 0; repeat < WARM_UP_STEPS; repeat++) {
            CHECK_CUDA(cudaEventRecord(started_on_card, state.stream));
            issue_chain(state);
            CHECK_CUDA(cudaEventRecord(finished_on_card, state.stream));
            CHECK_CUDA(cudaEventSynchronize(finished_on_card));
            float once = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&once, started_on_card, finished_on_card));
            kernel_milliseconds = (repeat == 0 || once < kernel_milliseconds) ? once : kernel_milliseconds;
        }
        const double kernel_microseconds = (double)kernel_milliseconds * 1000.0;

        /* the same step captured once as a graph; its last node reads the step number from card memory */
        CHECK_CUDA(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeThreadLocal));
        issue_chain(state);
        done_from_card_kernel<<<1, 1, 0, state.stream>>>(last_buffer(state), state.host_word, state.step_number_on_card);
        CHECK_CUDA(cudaStreamEndCapture(state.stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&graph_instance, graph, 0));

        const unsigned int expected_last = (unsigned int)state.kernel_count;   /* what a whole chain leaves: with --corrupt it is one less */
        unsigned int graph_step_number = GRAPH_STEP_BASE;
        CHECK_CUDA(cudaMemcpy(state.step_number_on_card, &graph_step_number, sizeof(graph_step_number), cudaMemcpyHostToDevice));
        printf("    %d kernels per layer, %d layers = %d kernels per step:\n", kernels_per_layer_cases[count_case], layer_count, state.kernel_count);
        for (int use_graph = 0; use_graph <= 1; use_graph++) {
            for (int spin = 0; spin <= 1; spin++) {
                timings measured;
                for (int step = 0; step < WARM_UP_STEPS + step_count; step++) {
                    const unsigned int step_number = use_graph ? ++graph_step_number : (unsigned int)(1 + step + 100000 * (spin + 1));
                    const double started = seconds_now();
                    if (use_graph) {
                        CHECK_CUDA(cudaGraphLaunch(graph_instance, state.stream));
                    } else {
                        issue_chain(state);
                        done_kernel<<<1, 1, 0, state.stream>>>(last_buffer(state), state.host_word, step_number);
                    }
                    const double issued = seconds_now();
                    if (spin) {
                        if (!spin_for_step(state, step_number)) {
                            fprintf(stderr, "the card did not report step %u within %.0f s\n", step_number, SPIN_LIMIT_SECONDS);
                            exit(EXIT_FAILURE);
                        }
                    } else {
                        CHECK_CUDA(cudaStreamSynchronize(state.stream));
                    }
                    const double done = seconds_now();
                    measured.results_right = measured.results_right && state.host_word[0] == step_number && state.host_word[1] == expected_last;
                    if (step >= WARM_UP_STEPS) {
                        measured.until_done.push_back((done - started) * MICROSECONDS_PER_SECOND);
                        measured.issuing.push_back((issued - started) * MICROSECONDS_PER_SECOND);
                    }
                    CHECK_CUDA(cudaStreamSynchronize(state.stream));   /* the next step starts on an empty queue */
                }
                char label[96];
                snprintf(label, sizeof(label), "%s, %s:", use_graph ? "one captured graph" : "individual launches",
                         spin ? "host spins on a mapped word" : "driver's synchronize");
                report(label, measured, kernel_microseconds);
                all_right = all_right && measured.results_right;
            }
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaGraphExecDestroy(graph_instance));
        CHECK_CUDA(cudaGraphDestroy(graph));
        CHECK_CUDA(cudaEventDestroy(started_on_card));
        CHECK_CUDA(cudaEventDestroy(finished_on_card));
        CHECK_CUDA(cudaStreamDestroy(state.stream));
        CHECK_CUDA(cudaFreeHost(host_memory));
        CHECK_CUDA(cudaFree(state.step_number_on_card));
        CHECK_CUDA(cudaFree(state.buffers[0]));
        CHECK_CUDA(cudaFree(state.buffers[1]));
    }
    return all_right;
}

int main(int argument_count, char **arguments)
{
    int layer_count = DEFAULT_LAYER_COUNT, step_count = DEFAULT_STEP_COUNT;
    bool corrupt = false;
    for (int argument_index = 1; argument_index < argument_count; argument_index++) {
        const bool has_value = argument_index + 1 < argument_count;
        if (strcmp(arguments[argument_index], "--corrupt") == 0) {
            corrupt = true;
        } else if (strcmp(arguments[argument_index], "--layers") == 0 && has_value) {
            layer_count = atoi(arguments[++argument_index]);
        } else if (strcmp(arguments[argument_index], "--steps") == 0 && has_value) {
            step_count = atoi(arguments[++argument_index]);
        } else {
            fprintf(stderr, "usage: %s [--layers N] [--steps N] [--corrupt]\n", arguments[0]);
            return EXIT_FAILURE;
        }
    }
    int card_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&card_count));
    if (layer_count < 1 || step_count < 10 || card_count < 1 || card_count > MAX_CARDS) {
        fprintf(stderr, "--layers at least 1, --steps at least 10, and 1 to %d cards\n", MAX_CARDS);
        return EXIT_FAILURE;
    }
    printf("a step of one card, %d steps each%s:\n", step_count, corrupt ? ", with one kernel left out of every step" : "");
    bool all_right = true;
    for (int card = 0; card < card_count; card++) {
        all_right = measure_card(card, layer_count, step_count, corrupt) && all_right;
    }
    if (corrupt) {
        printf("%s\n", all_right ? "FAILED: the missing kernel was not noticed" : "the missing kernel was noticed");
        return all_right ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    printf("%s\n", all_right ? "every step's result was right" : "FAILED: a step's result was wrong");
    return all_right ? EXIT_SUCCESS : EXIT_FAILURE;
}
