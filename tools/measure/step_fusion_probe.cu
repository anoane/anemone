/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step_fusion_probe: round 1, package 12 of the speed campaign (research/speed/ROUND_1_attention.md; the premise of
 * ledger S1): do two kernels of ONE card overlap when the graph lets them, for the pairs a layer really has. It
 * times nothing of the model: a pair is two launches of the `work` kernel of card_work_kernels.cuh, unchanged (blocks
 * of 8 warps, 87,040 B a block: a launch's size is its blocks), or `reader` (below) beside one.
 *
 * Cases, --pair: first + second launch, in blocks of work (MB a launch reads):
 *   query_keyvalue  80 (7.0: query A) + 32 (2.8: key/value)       router_shared  32 + 272 (23.7: the shared expert's
 *   routed_shared   1,296 (112.8: a layer's routed experts) + 272                gate with up)
 *   scan_wqb:low, scan_wqb:full   reader (134.2: the keys of a rate-2 scan at 1M tokens; with 4 warps an SM, or with
 *                   32, the read probe's occupancy) + 480 (41.8: wq_b)
 * Instances, interleaved in one run by probe_harness.h, every kernel an explicit kernel node: baseline (one graph,
 * the second node depending on the first) and baseline_again; siblings:graph (the same two nodes, no edge);
 * first_alone and second_alone (a graph of one node each); siblings:streams (those two graphs, each on a stream of
 * its own). --mode step only: a stream would need 64 instantiated copies of every graph. A timed step: its launches
 * take the next bytes of a cold ring and become the node parameters of its graphs, which are uploaded (none of that
 * timed); then the graph launches between the harness's two marks. The marks are events of the default stream, which
 * follow all work of the probe's two streams: the stop mark is the join. RESULT has mb=na: an instance reads its own
 * bytes.
 *
 * VERDICT, with a = first_alone, b = second_alone, c = the chain, s = siblings:graph (medians of this run):
 * overlap_share (c - s) / min(a, b), 1 full overlap and 0 none; chain_over_alone_sum c / (a + b), the gaps between
 * nodes; streams_over_graph: all measured. Bars on s / c, with the pairs of siblings:graph: for query_keyvalue and
 * router_shared a share of 0.70 or more confirms and 0.25 or less refutes; routed_shared under 0.90 of the chain
 * reopens rank 9, 0.97 to 1.03 is the expected nothing. A bar "at least" is given to probe_verdict inverted: its
 * line shows 1 / bar and c / s.
 *
 * Checked per case before anything is timed, every instance on the same bytes (the launches of the checked step):
 *   graph_edges         a graph's nodes less its root nodes, read back: 1 in the chain, 0 elsewhere
 *   host_reference      every output word against what it must be: the host's value where the host forms one (every
 *                       checksum; of a work launch the sums of every 37th block, exact by construction), the chain's
 *                       elsewhere
 *   bits_against_chain  every output word of the instance's launches against the chain's: 0 may differ
 *   guard_bytes; DIGEST nothing wrote behind a card array; all outputs, for the two CMP 170HX to agree on
 * Every launch, checked or timed, is validated on the host first (fusion_step_is_valid), and two launches with no
 * edge between them are refused if they write one array. Shown failing: --corrupt (the megabyte of weights each
 * launch of the checked step begins with is overwritten on the card: host_reference reports for every instance),
 * --break bits (a word of the host's copy of each of the chain's outputs is changed: bits_against_chain reports
 * for every other instance), --break edge (siblings:graph is built with the edge: its graph_edges reports),
 * --break-route (four launches of siblings:graph past one bound each, the last with the first launch's checksum
 * array as its own: all refused, none launched), --break-guard. The debug build's kernels take a diagnostics
 * record, each launch its own, and check their bounds themselves: a status word one of them stores ends the run.
 *
 *   step_fusion_probe --mode step [--pair NAME,...] [--break bits,edge] and the flags of probe_harness.h
 *
 * Measurement tool: not part of the engine.
 */
#include "probe_harness.h"
#include "card_work_kernels.cuh"

enum {
    READER_BLOCK_THREADS = 128,                  /* 4 warps: a block an SM is `low`, eight are `full` */
    READER_LOW_BLOCKS_PER_SM = 1, READER_FULL_BLOCKS_PER_SM = 8,
    READER_WARP_LOADS = 262144,                  /* of the ring's packed words: 524,288 keys of 256 B, 134.2 MB */
    READER_LOADS = READER_WARP_LOADS * WORK_LANES_PER_WARP,     /* 16-byte loads */
    FUSION_FIRST = 0, FUSION_SECOND = 1, FUSION_LAUNCHES = 2,
    FUSION_SIBLINGS_GRAPH = 2, FUSION_SIBLINGS_STREAMS, FUSION_FIRST_ALONE, FUSION_SECOND_ALONE, FUSION_INSTANCES,
    FUSION_NO_GRAPH = FUSION_INSTANCES,          /* of a launch an instance does not make */
    FUSION_ROUTED_SHARED = 2,                    /* --pair: its third word; the two before it are the narrow pairs */
    FUSION_NARROW_BLOCKS = 32, FUSION_QUERY_BLOCKS = 80, FUSION_SHARED_BLOCKS = 272, FUSION_WQB_BLOCKS = 480,
    FUSION_WARP_LOADS_PER_BLOCK = WORK_WARPS_PER_BLOCK * WORK_MOST_LOADS_PER_WARP,
    FUSION_MOST_STEP_WARP_LOADS = READER_WARP_LOADS + WORK_STEP_WARP_LOADS,   /* no pair takes more of the ring */
    FUSION_COLD_L2_SIZES = 4,                    /* CONVENTIONS 3.5: bytes read between two reads of the same bytes */
    FUSION_CHECKED_PLACE = 0,                    /* the ring's warp load at which the checked step begins */
    FUSION_BUILT_PLACE = FUSION_MOST_STEP_WARP_LOADS,    /* a graph is built for another place than it is checked at:
                                                            a parameter update that did not arrive cannot pass */
    FUSION_CHECKED_BLOCK_STRIDE = 37,            /* the blocks whose sums the host forms, block 0 first */
    FUSION_MOST_READER_BLOCKS = 2048,            /* 256 SMs of `full`; no launch of work has as many blocks */
    FUSION_SUM_WORDS = WORK_STEP_BLOCKS * WORK_SUMS_PER_BLOCK,              /* a launch's outputs, as allocated */
    FUSION_CHECKSUM_WORDS = FUSION_MOST_READER_BLOCKS * READER_BLOCK_THREADS,
    FUSION_OUTPUT_WORDS = FUSION_SUM_WORDS + FUSION_CHECKSUM_WORDS,
    FUSION_KERNEL_PARAMETERS = 1 + PROBE_DIAGNOSTICS,
    FUSION_CORRUPTED_BYTES = 1024 * 1024, FUSION_CORRUPTED_BYTE = 0x5A,
    FUSION_STALE_BYTE = 0xFF,                    /* the outputs before a checked launch: a NaN, and no checksum */
    FUSION_FILL_THREADS = 256, FUSION_FIGURE_BYTES = 256,
    /* --break-route, in turn: what is put past its bound */
    FUSION_BROKEN_RING_END = 0, FUSION_BROKEN_BLOCKS, FUSION_BROKEN_BLOCK_THREADS, FUSION_BROKEN_SHARED_OUTPUT,
    FUSION_BROKEN_END
};
#define FUSION_FLAG_PAIR "--pair"
#define FUSION_FLAG_BREAK "--break"
#define FUSION_PAIRS "query_keyvalue,router_shared,routed_shared,scan_wqb:low,scan_wqb:full"
#define FUSION_CASE_PREFIX "pair:"               /* a case's name before the pair's: RESULT part= is the rest */
#define FUSION_RUNS(instance, launch) (fusion_forms[instance].graph_of[launch] != FUSION_NO_GRAPH)
#define FUSION_BUILDS_A_GRAPH(instance) (fusion_forms[instance].graph_of[FUSION_FIRST] == (instance) \
                                         || fusion_forms[instance].graph_of[FUSION_SECOND] == (instance))
#define FUSION_INSTANCE(name) {name, "work_kernel+reader_kernel", 'C', 1, 0, 0, 0, 0, 0}

static struct probe_flag fusion_flags[] = {
    PROBE_COMMON_FLAGS,
    PROBE_CASE_WORDS(FUSION_FLAG_PAIR, FUSION_PAIRS, FUSION_PAIRS, "the two kernels of a step"),
    PROBE_WORD_LIST(FUSION_FLAG_BREAK, "bits,edge", "", "show that check failing"),
};
static struct probe_instance fusion_instances[FUSION_INSTANCES] = {
    FUSION_INSTANCE("baseline"),         FUSION_INSTANCE("baseline_again"), FUSION_INSTANCE("siblings:graph"),
    FUSION_INSTANCE("siblings:streams"), FUSION_INSTANCE("first_alone"),    FUSION_INSTANCE("second_alone"),
};
/* What the instance of the same index issues: per launch the graph that carries it, named by the instance that
 * builds it (FUSION_NO_GRAPH: the launch is not made), and the edges of the graph it builds. */
static const struct fusion_form {
    uint32_t graph_of[FUSION_LAUNCHES], edges;
} fusion_forms[FUSION_INSTANCES] = {
    {{PROBE_BASELINE, PROBE_BASELINE}, 1},               {{PROBE_BASELINE, PROBE_BASELINE}, 1},
    {{FUSION_SIBLINGS_GRAPH, FUSION_SIBLINGS_GRAPH}, 0}, {{FUSION_FIRST_ALONE, FUSION_SECOND_ALONE}, 0},
    {{FUSION_FIRST_ALONE, FUSION_NO_GRAPH}, 0},          {{FUSION_NO_GRAPH, FUSION_SECOND_ALONE}, 0},
};
static const struct fusion_pair {
    uint32_t blocks[FUSION_LAUNCHES];            /* of work; the first unused where the reader is the first launch */
    uint32_t reader_blocks_per_sm;               /* 0: the first launch is work too */
} fusion_pairs[] = {
    {{FUSION_QUERY_BLOCKS, FUSION_NARROW_BLOCKS}, 0}, {{FUSION_NARROW_BLOCKS, FUSION_SHARED_BLOCKS}, 0},
    {{WORK_STEP_BLOCKS, FUSION_SHARED_BLOCKS}, 0},    {{0, FUSION_WQB_BLOCKS}, READER_LOW_BLOCKS_PER_SM},
    {{0, FUSION_WQB_BLOCKS}, READER_FULL_BLOCKS_PER_SM},
};
/* Acceptance: bars "at most" on siblings:graph over the chain. as_share (the two narrow pairs): the ratio at which
 * that share of the shorter kernel is gone; else (routed_shared) the ratio itself. at_least: the bar is the other
 * way round, which probe_verdict gets as the inverse ratio. */
static const struct fusion_bar {
    uint32_t as_share, at_least;
    double level;
    const char *figure;
} fusion_bars[] = {{1, 0, 0.70, "share_0.70_or_more"},      {1, 1, 0.25, "share_0.25_or_less"},
                   {0, 0, 0.90, "under_0.90_of_the_chain"}, {0, 1, 0.97, "0.97_of_the_chain_or_more"},
                   {0, 0, 1.03, "1.03_of_the_chain_or_less"}};
static const char *const fusion_broken_names[FUSION_BROKEN_END] = {
    "refused_ring_past_its_end", "refused_block_too_many", "refused_block_of_other_threads",
    "refused_siblings_with_one_output_array"};

/* The reader: 16-byte loads of the ring folded to a checksum a thread, bound by reading and by nothing else. Thread t
 * of the grid's T reads loads t, t + T, ... of READER_LOADS from the launch's place on: a warp 512 contiguous bytes
 * at a time, the launch's bytes once whatever the grid. It takes the work kernel's record, so that a step's launches
 * are one kind of record, and uses of it ring_packed, first_warp_load and checksums ([thread of the grid]). What a
 * launch must satisfy is in fusion_step_is_valid; the debug kernel checks its bounds itself, as the work kernel. */
__global__ void reader_kernel(struct work_arguments arguments WORK_DIAGNOSTICS_PARAMETER)
{
    uint64_t thread_count = (uint64_t)gridDim.x * blockDim.x, thread = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint4 *loads = arguments.ring_packed + arguments.first_warp_load * WORK_LANES_PER_WARP;
    uint32_t checksum = 0;
#if PROBE_DIAGNOSTICS
    if (arguments.first_warp_load + READER_WARP_LOADS > diagnostics.ring_warp_loads
        || gridDim.x > diagnostics.block_count || blockDim.x != READER_BLOCK_THREADS) {
        *diagnostics.status_word = ((uint64_t)WORK_FAILED_RING << WORK_WORD_BITS) | blockIdx.x;    /* one plain store */
        return;
    }
#endif
    for (uint64_t load = thread; load < READER_LOADS; load += thread_count) {
        uint4 loaded = loads[load];
        uint32_t words[WORK_WORDS_PER_LOAD] = {loaded.x, loaded.y, loaded.z, loaded.w};
        checksum = work_fold(checksum, work_load_word(words, 0));
    }
    arguments.checksums[thread] = checksum;
}

struct fusion_launch {                           /* one kernel of a step: validated, then a node's parameters */
    uint32_t is_reader, block_count, block_threads;
    uint64_t warp_loads;                         /* what it takes of the ring */
    struct work_arguments work;
#if PROBE_DIAGNOSTICS
    struct work_diagnostics diagnostics;         /* its two arrays and the ring's size: set once, with the card */
#endif
};
struct fusion_graph {
    cudaGraph_t graph;                           /* NULL: none is built */
    cudaGraphExec_t executable;
    cudaGraphNode_t nodes[FUSION_LAUNCHES];
    uint64_t edges;                              /* read back when it was built */
};
static struct {
    uint4 *ring_packed;                          /* the card's arrays: struct work_arguments says what each holds */
    uint32_t *ring_scales;
    uint4 *input_words;
    uint16_t *input_exponents;
    float *sums[FUSION_LAUNCHES];                /* a launch's own outputs */
    uint32_t *checksums[FUSION_LAUNCHES];
    uint64_t ring_warp_loads, cursor;            /* cursor: where the next timed step begins in the ring */
    uint32_t sm_count;
    struct fusion_launch step[FUSION_LAUNCHES];  /* the one step that is prepared */
    struct fusion_graph graphs[FUSION_INSTANCES];        /* of the case at hand, by the instance that builds each */
    cudaStream_t streams[FUSION_LAUNCHES];       /* a launch's graph goes to the stream of the launch's index */
    uint32_t host_input_words[WORK_INPUT_WORDS];         /* the host's copies of what was uploaded */
    uint16_t host_input_exponents[WORK_INPUT_EXPONENTS];
} fusion;

/* The launches of one step of an instance, one behind the other in the ring from warp load `place` on; returns where
 * they end. Nothing is checked here: fusion_step_is_valid does that, before anything is launched. */
static uint64_t fusion_make_step(const struct probe_case *probe_case, uint32_t instance, uint64_t place)
{
    const struct fusion_pair *pair = &fusion_pairs[probe_case_value(probe_case, FUSION_FLAG_PAIR)];
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        struct fusion_launch *launch = &fusion.step[index];
        const struct work_arguments work = {fusion.ring_packed, fusion.ring_scales, fusion.input_words,
                                            fusion.input_exponents, fusion.sums[index], fusion.checksums[index],
                                            place, WORK_MOST_LOADS_PER_WARP, 1};
        if (!FUSION_RUNS(instance, index)) { continue; }
        launch->is_reader = index == FUSION_FIRST && pair->reader_blocks_per_sm != 0u;
        launch->block_count = launch->is_reader ? fusion.sm_count * pair->reader_blocks_per_sm : pair->blocks[index];
        launch->block_threads = launch->is_reader ? (uint32_t)READER_BLOCK_THREADS : (uint32_t)WORK_BLOCK_THREADS;
        launch->warp_loads = launch->is_reader ? (uint64_t)READER_WARP_LOADS
                                               : (uint64_t)launch->block_count * FUSION_WARP_LOADS_PER_BLOCK;
        launch->work = work;
#if PROBE_DIAGNOSTICS
        launch->diagnostics.block_count = launch->block_count;
#endif
        place += launch->warp_loads;
    }
    return place;
}

/* Check 10 for the step in fusion.step as `instance` issues it: every launch inside its arrays as allocated and
 * inside the card's limits; and two launches with no edge between them may run at once, so they must not write one
 * array. 1: it holds; 0: a REFUSED line said why not. */
static int fusion_step_is_valid(uint32_t instance)
{
    const struct fusion_launch *first = &fusion.step[FUSION_FIRST], *second = &fusion.step[FUSION_SECOND];
    const char *reader = "reader_kernel";
    int valid = 1;
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        const struct fusion_launch *launch = &fusion.step[index];
        const struct work_arguments *work = &launch->work;
        const struct work_array_bytes bytes = {
            probe_cuda_array_bytes(work->ring_packed), probe_cuda_array_bytes(work->ring_scales),
            probe_cuda_array_bytes(work->input_words), probe_cuda_array_bytes(work->input_exponents),
            probe_cuda_array_bytes(work->sums),        probe_cuda_array_bytes(work->checksums)};
        if (!FUSION_RUNS(instance, index)) { continue; }
        if (launch->is_reader) {
            probe_launch_geometry(&valid, reader, &fusion_instances[instance], launch->block_count, 1, 1,
                                  launch->block_threads, 0);
            probe_launch_value(&valid, reader, "threads of a block of this kernel", launch->block_threads,
                               READER_BLOCK_THREADS, READER_BLOCK_THREADS);
            probe_launch_range(&valid, reader, "ring_packed, in warp loads", work->first_warp_load, READER_WARP_LOADS,
                               WORK_PACKED_BYTES_PER_WARP_LOAD, bytes.ring_packed);
            probe_launch_range(&valid, reader, "checksums, in blocks", 0, launch->block_count,
                               READER_BLOCK_THREADS * sizeof(uint32_t), bytes.checksums);
        } else {
            valid &= work_launch_is_valid(&fusion_instances[instance], work, &bytes, launch->block_count,
                                          launch->block_threads);
        }
#if PROBE_DIAGNOSTICS
        valid &= work_diagnostics_are_valid(&launch->diagnostics,
                                            probe_cuda_array_bytes(launch->diagnostics.status_word),
                                            probe_cuda_array_bytes(launch->diagnostics.thread_loads));
#endif
    }
    if (FUSION_RUNS(instance, FUSION_FIRST) && FUSION_RUNS(instance, FUSION_SECOND)
        && fusion_forms[instance].edges == 0u) {         /* the second is a work launch; the reader has no sums */
        const int shared = second->work.checksums == first->work.checksums
                           || (!first->is_reader && second->work.sums == first->work.sums);
        probe_launch_value(&valid, "siblings", "output arrays the two launches share", shared ? 1u : 0u, 0, 0);
    }
    return valid;
}

/* A launch as a kernel node's parameters: its kernel, the validated grid and block, and its arguments, which the
 * runtime copies while it takes them. */
static cudaKernelNodeParams fusion_node(struct fusion_launch *launch, void **kernel_arguments)
{
    cudaKernelNodeParams node = {};
    kernel_arguments[0] = &launch->work;
#if PROBE_DIAGNOSTICS
    kernel_arguments[1] = &launch->diagnostics;
#endif
    node.func = launch->is_reader ? (void *)reader_kernel : (void *)work_kernel;
    node.gridDim = dim3(launch->block_count);
    node.blockDim = dim3(launch->block_threads);
    node.kernelParams = kernel_arguments;
    return node;
}

/* The graphs an instance launches, each once, on the stream of the first launch it carries: uploaded, or launched. */
static void fusion_issue(uint32_t instance, int upload)
{
    const uint32_t *graph_of = fusion_forms[instance].graph_of;
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        if (FUSION_RUNS(instance, index) && (index == FUSION_FIRST || graph_of[index] != graph_of[FUSION_FIRST])) {
            cudaGraphExec_t executable = fusion.graphs[graph_of[index]].executable;
            PROBE_CHECK_CUDA(upload ? cudaGraphUpload(executable, fusion.streams[index])
                                    : cudaGraphLaunch(executable, fusion.streams[index]));
        }
    }
}

/* Not timed: the validated step becomes the node parameters of the instance's graphs, and they are uploaded, so
 * that the timed launch carries neither. */
static void fusion_set_graphs(uint32_t instance)
{
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        void *kernel_arguments[FUSION_KERNEL_PARAMETERS];
        if (FUSION_RUNS(instance, index)) {
            const struct fusion_graph *graph = &fusion.graphs[fusion_forms[instance].graph_of[index]];
            const cudaKernelNodeParams node = fusion_node(&fusion.step[index], kernel_arguments);
            PROBE_CHECK_CUDA(cudaGraphExecKernelNodeSetParams(graph->executable, graph->nodes[index], &node));
        }
    }
    fusion_issue(instance, 1);
}

/* Timed: the graph launches of the prepared step, and nothing else. */
static uint32_t fusion_launch_step(void *, const struct probe_case *, uint32_t instance, uint32_t)
{
    fusion_issue(instance, 0);
    PROBE_AFTER_LAUNCH();
    return (FUSION_RUNS(instance, FUSION_FIRST) ? 1u : 0u) + (FUSION_RUNS(instance, FUSION_SECOND) ? 1u : 0u);
}

static void fusion_destroy_graphs(void)
{
    for (uint32_t instance = 0; instance < FUSION_INSTANCES; instance++) {
        if (fusion.graphs[instance].graph != NULL) {
            PROBE_CHECK_CUDA(cudaGraphExecDestroy(fusion.graphs[instance].executable));
            PROBE_CHECK_CUDA(cudaGraphDestroy(fusion.graphs[instance].graph));
            fusion.graphs[instance].graph = NULL;
        }
    }
}

/* The graphs of a case, from explicit kernel nodes, so that an edge exists exactly where the instance says: the
 * second node is given the first as its dependency, or nothing. Built from a validated step at FUSION_BUILT_PLACE.
 * The edges are read back as nodes less root nodes, which is their number in a graph of at most two nodes. */
static void fusion_build_graphs(const struct probe_case *probe_case)
{
    fusion_destroy_graphs();                     /* of the case before */
    for (uint32_t instance = 0; instance < FUSION_INSTANCES; instance++) {
        struct fusion_graph *graph = &fusion.graphs[instance];
        const int edged = fusion_forms[instance].edges != 0u
                          || (instance == FUSION_SIBLINGS_GRAPH && probe_flag_has_word(FUSION_FLAG_BREAK, "edge"));
        size_t node_count = 0, root_count = 0;
        if (!FUSION_BUILDS_A_GRAPH(instance)) { continue; }
        fusion_make_step(probe_case, instance, FUSION_BUILT_PLACE);
        PROBE_REQUIRE(fusion_step_is_valid(instance), "the step a graph is built from was refused");
        PROBE_CHECK_CUDA(cudaGraphCreate(&graph->graph, 0));
        for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
            void *kernel_arguments[FUSION_KERNEL_PARAMETERS];
            const int after_first = edged && index == FUSION_SECOND;
            if (fusion_forms[instance].graph_of[index] == instance) {
                const cudaKernelNodeParams node = fusion_node(&fusion.step[index], kernel_arguments);
                PROBE_CHECK_CUDA(cudaGraphAddKernelNode(&graph->nodes[index], graph->graph,
                                                        after_first ? &graph->nodes[FUSION_FIRST] : NULL,
                                                        after_first ? 1u : 0u, &node));
            }
        }
        PROBE_CHECK_CUDA(cudaGraphGetNodes(graph->graph, NULL, &node_count));
        PROBE_CHECK_CUDA(cudaGraphGetRootNodes(graph->graph, NULL, &root_count));
        graph->edges = node_count - root_count;
        PROBE_CHECK_CUDA(cudaGraphInstantiate(&graph->executable, graph->graph, 0));
    }
}

/* The ring is walked in order, whatever the instance: a step reads the bytes behind those of the step before it,
 * and the same bytes again only after the rest of the ring, four L2 sizes or more. */
static int fusion_prepare_step(void *, const struct probe_case *probe_case, uint32_t instance, uint64_t, uint32_t)
{
    uint64_t end = fusion_make_step(probe_case, instance, fusion.cursor);
    if (end > fusion.ring_warp_loads) { end = fusion_make_step(probe_case, instance, 0); }
    fusion.cursor = end;
    if (!fusion_step_is_valid(instance)) { return 0; }
    fusion_set_graphs(instance);
    return 1;
}

/* A ring array gets its stored words from a kernel, a block an SM: over a CMP 170HX's link it would take a second. */
static void fusion_fill(void *words, uint64_t word_count, uint32_t kind)
{
    int valid = 1;
    probe_launch_geometry(&valid, "work_fill_kernel", NULL, fusion.sm_count, 1, 1, FUSION_FILL_THREADS, 0);
    probe_launch_range(&valid, "work_fill_kernel", "ring words", 0, word_count, sizeof(uint32_t),
                       probe_cuda_array_bytes(words));
    PROBE_REQUIRE(valid, "the launch that fills the ring was refused");
    work_fill_kernel<<<fusion.sm_count, FUSION_FILL_THREADS>>>((uint32_t *)words, word_count, kind);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
}

static int fusion_open_card(void *, struct probe_card_facts *facts)
{
    if (probe_flag_has_word(PROBE_FLAG_MODE, PROBE_MODE_STREAM) || probe_flag_given(PROBE_FLAG_SOAK_SECONDS)) {
        fprintf(stderr, "give --mode step, and no soak: a stream is %u steps prepared at once, which would need as "
                        "many instantiated copies of every graph\n", PROBE_STREAM_STEPS);
        return 0;
    }
    if (!probe_cuda_open(facts)) { return 0; }
    probe_run.breaking |= (uint32_t)probe_flag_given(FUSION_FLAG_BREAK);    /* a list is no break flag to the harness */
    fusion.sm_count = (uint32_t)facts->sm_count;
    fusion.ring_warp_loads = FUSION_COLD_L2_SIZES * facts->l2_bytes / WORK_PACKED_BYTES_PER_WARP_LOAD
                             + (uint64_t)FUSION_MOST_STEP_WARP_LOADS + FUSION_MOST_STEP_WARP_LOADS;
    const uint64_t packed_bytes = fusion.ring_warp_loads * WORK_PACKED_BYTES_PER_WARP_LOAD;
    const uint64_t scale_bytes = fusion.ring_warp_loads * WORK_SCALE_BYTES_PER_WARP_LOAD;
    fusion.ring_packed = (uint4 *)probe_cuda_allocate(packed_bytes);
    fusion.ring_scales = (uint32_t *)probe_cuda_allocate(scale_bytes);
    fusion.input_words = (uint4 *)probe_cuda_allocate(sizeof(fusion.host_input_words));
    fusion.input_exponents = (uint16_t *)probe_cuda_allocate(sizeof(fusion.host_input_exponents));
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        fusion.sums[index] = (float *)probe_cuda_allocate(FUSION_SUM_WORDS * sizeof(float));
        fusion.checksums[index] = (uint32_t *)probe_cuda_allocate(FUSION_CHECKSUM_WORDS * sizeof(uint32_t));
        PROBE_CHECK_CUDA(cudaStreamCreate(&fusion.streams[index]));
#if PROBE_DIAGNOSTICS
        fusion.step[index].diagnostics.status_word = (uint64_t *)probe_cuda_allocate(sizeof(uint64_t));
        fusion.step[index].diagnostics.thread_loads = (uint32_t *)probe_cuda_allocate(
            sizeof(uint32_t) * FUSION_MOST_READER_BLOCKS * WORK_BLOCK_THREADS);
        fusion.step[index].diagnostics.ring_warp_loads = fusion.ring_warp_loads;
        PROBE_CHECK_CUDA(cudaMemset(fusion.step[index].diagnostics.status_word, 0, sizeof(uint64_t)));
#endif
    }
    fusion_fill(fusion.ring_packed, packed_bytes / sizeof(uint32_t), WORK_PACKED);
    fusion_fill(fusion.ring_scales, scale_bytes / sizeof(uint32_t), WORK_SCALES);
    for (uint32_t word = 0; word < WORK_INPUT_WORDS; word++) {
        fusion.host_input_words[word] = work_stored_word(word, WORK_INPUTS);
    }
    for (uint32_t pair = 0; pair < WORK_INPUT_EXPONENTS; pair++) {
        fusion.host_input_exponents[pair] = (uint16_t)work_stored_word(pair, WORK_EXPONENTS);
    }
    PROBE_CHECK_CUDA(cudaMemcpy(fusion.input_words, fusion.host_input_words, sizeof(fusion.host_input_words),
                                cudaMemcpyHostToDevice));
    PROBE_CHECK_CUDA(cudaMemcpy(fusion.input_exponents, fusion.host_input_exponents,
                                sizeof(fusion.host_input_exponents), cudaMemcpyHostToDevice));
    facts->ring_bytes = packed_bytes + scale_bytes;
    return 1;
}

static void fusion_close_card(void *)
{
    fusion_destroy_graphs();
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        PROBE_CHECK_CUDA(cudaStreamDestroy(fusion.streams[index]));
    }
    probe_cuda_close();
}

/* Every case begins here, the profiler's too: its graphs are built before anything of it is checked or timed. */
static void fusion_describe_case(void *, const struct probe_case *probe_case, struct probe_case_facts *facts)
{
    facts->format = "synthetic";
    facts->part = probe_case->name + strlen(FUSION_CASE_PREFIX);
    facts->rows = 1;
    facts->experts = facts->overlap = -1;
    fusion_build_graphs(probe_case);
}

/* What the host can form of a launch's outputs, written over `expected` (sums, then checksums): every checksum, and
 * of a work launch the sums of every FUSION_CHECKED_BLOCK_STRIDE-th block; the ring's words from their definition. */
static void fusion_expect(const struct fusion_launch *launch, uint32_t *expected)
{
    static float sums[WORK_SUMS_PER_BLOCK];
    uint32_t *checksums = expected + FUSION_SUM_WORDS;
    const uint64_t thread_count = (uint64_t)launch->block_count * launch->block_threads;
    const uint64_t first_word = launch->work.first_warp_load * WORK_LANES_PER_WARP * WORK_WORDS_PER_LOAD;
    for (uint64_t block = 0; !launch->is_reader && block < launch->block_count; block++) {
        float *block_sums = block % FUSION_CHECKED_BLOCK_STRIDE == 0 ? sums : NULL;
        work_host_block(fusion.host_input_words, fusion.host_input_exponents, &launch->work, block, block_sums,
                        checksums + block * WORK_LANES_PER_WARP);
        if (block_sums != NULL) { memcpy(expected + block * WORK_SUMS_PER_BLOCK, sums, sizeof(sums)); }
    }
    if (launch->is_reader) { memset(checksums, 0, thread_count * sizeof(uint32_t)); }
    for (uint64_t load = 0; launch->is_reader && load < READER_LOADS; load++) {
        uint32_t words[WORK_WORDS_PER_LOAD];
        for (uint32_t word = 0; word < WORK_WORDS_PER_LOAD; word++) {
            words[word] = work_stored_word(first_word + load * WORK_WORDS_PER_LOAD + word, WORK_PACKED);
        }
        checksums[load % thread_count] = work_fold(checksums[load % thread_count], work_load_word(words, 0));
    }
}

/* The step in fusion.step, issued as the instance's timed step is, on an idle card. The outputs are made stale
 * first (a word a launch does not write cannot pass for a result) and read back whole: sums, then checksums. */
static void fusion_run_checked(uint32_t instance, uint32_t result[FUSION_LAUNCHES][FUSION_OUTPUT_WORDS])
{
    const size_t sum_bytes = FUSION_SUM_WORDS * sizeof(float);
    const size_t checksum_bytes = FUSION_CHECKSUM_WORDS * sizeof(uint32_t);
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        PROBE_CHECK_CUDA(cudaMemset(fusion.sums[index], FUSION_STALE_BYTE, sum_bytes));
        PROBE_CHECK_CUDA(cudaMemset(fusion.checksums[index], FUSION_STALE_BYTE, checksum_bytes));
    }
    fusion_set_graphs(instance);
    (void)fusion_launch_step(NULL, NULL, instance, 0);
    PROBE_CHECK_CUDA(cudaGetLastError());
    PROBE_CHECK_CUDA(cudaDeviceSynchronize());
    for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
        PROBE_CHECK_CUDA(cudaMemcpy(result[index], fusion.sums[index], sum_bytes, cudaMemcpyDeviceToHost));
        PROBE_CHECK_CUDA(cudaMemcpy(result[index] + FUSION_SUM_WORDS, fusion.checksums[index], checksum_bytes,
                                    cudaMemcpyDeviceToHost));
#if PROBE_DIAGNOSTICS
        uint64_t status_word = 0;                /* what a kernel's own bound check stored: nothing, ever */
        PROBE_CHECK_CUDA(cudaMemcpy(&status_word, fusion.step[index].diagnostics.status_word, sizeof(status_word),
                                    cudaMemcpyDeviceToHost));
        PROBE_REQUIRE(status_word == 0, "a kernel refused a launch that the host had validated");
#endif
    }
}

/* --break-route: the checked step of siblings:graph past one bound each, whatever kernel its first launch is. The
 * validation every real launch passes must refuse each (1 of 1: the failure the flag requires); none is launched. */
static void fusion_break_route(const struct probe_case *probe_case)
{
    struct fusion_launch *first = &fusion.step[FUSION_FIRST], *second = &fusion.step[FUSION_SECOND];
    for (uint32_t broken = 0; broken < FUSION_BROKEN_END; broken++) {
        fusion_make_step(probe_case, FUSION_SIBLINGS_GRAPH, FUSION_CHECKED_PLACE);
        if (broken == FUSION_BROKEN_RING_END) {              /* one warp load past it */
            first->work.first_warp_load = fusion.ring_warp_loads - first->warp_loads + 1u;
        } else if (broken == FUSION_BROKEN_BLOCKS) {         /* more than either kernel's output arrays hold */
            first->block_count = FUSION_MOST_READER_BLOCKS + 1u;
        } else if (broken == FUSION_BROKEN_BLOCK_THREADS) {
            first->block_threads += first->block_threads;
        } else {                                             /* valid alone; not beside the first launch */
            second->work.checksums = first->work.checksums;
        }
        probe_check_count(probe_case->name, fusion_instances[FUSION_SIBLINGS_GRAPH].name, fusion_broken_names[broken],
                          fusion_step_is_valid(FUSION_SIBLINGS_GRAPH) ? 0u : 1u, 1, 1);
    }
}

static void fusion_check_case(void *, const struct probe_case *probe_case)
{
    static uint32_t result[FUSION_LAUNCHES][FUSION_OUTPUT_WORDS], chain[FUSION_LAUNCHES][FUSION_OUTPUT_WORDS];
    static uint32_t expected[FUSION_LAUNCHES][FUSION_OUTPUT_WORDS];
    const int corrupt = probe_flag_given(PROBE_FLAG_CORRUPT);
    const int break_bits = probe_flag_has_word(FUSION_FLAG_BREAK, "bits");
    uint64_t second_place = 0;
    if (probe_flag_given(PROBE_FLAG_BREAK_ROUTE)) {
        fusion_break_route(probe_case);
        return;
    }
    for (uint32_t instance = 0; instance < FUSION_INSTANCES; instance++) {
        const char *variant = fusion_instances[instance].name;
        uint64_t off_host = 0, off_chain = 0, compared = 0;
        if (instance == PROBE_BASELINE_AGAIN) { continue; }      /* the baseline's graph a second time */
        if (FUSION_BUILDS_A_GRAPH(instance)) {
            probe_check_count(probe_case->name, variant, "graph_edges",
                              fusion.graphs[instance].edges != fusion_forms[instance].edges ? 1u : 0u, 1,
                              instance == FUSION_SIBLINGS_GRAPH && probe_flag_has_word(FUSION_FLAG_BREAK, "edge"));
        }
        /* every instance's launches lie where the chain's do: the second alone behind the room of the first */
        fusion_make_step(probe_case, instance,
                         FUSION_RUNS(instance, FUSION_FIRST) ? (uint64_t)FUSION_CHECKED_PLACE : second_place);
        PROBE_REQUIRE(fusion_step_is_valid(instance), "the checked step was refused");
        if (instance == PROBE_BASELINE) { second_place = fusion.step[FUSION_SECOND].work.first_warp_load; }
        /* --corrupt: the megabyte each launch begins with: inside what the validated launch reads of the ring */
        for (uint32_t index = 0; corrupt && instance == PROBE_BASELINE && index < FUSION_LAUNCHES; index++) {
            const uint64_t first_byte = fusion.step[index].work.first_warp_load * WORK_PACKED_BYTES_PER_WARP_LOAD;
            PROBE_REQUIRE(FUSION_CORRUPTED_BYTES <= fusion.step[index].warp_loads * WORK_PACKED_BYTES_PER_WARP_LOAD,
                          "a launch that reads less than --corrupt overwrites");
            PROBE_CHECK_CUDA(cudaMemset((uint8_t *)fusion.ring_packed + first_byte, FUSION_CORRUPTED_BYTE,
                                        FUSION_CORRUPTED_BYTES));
        }
        fusion_run_checked(instance, result);
        for (uint32_t index = 0; index < FUSION_LAUNCHES; index++) {
            if (!FUSION_RUNS(instance, index)) { continue; }
            if (instance == PROBE_BASELINE) {
                memcpy(chain[index], result[index], sizeof(chain[index]));
                memcpy(expected[index], result[index], sizeof(expected[index]));
                fusion_expect(&fusion.step[index], expected[index]);
                chain[index][FUSION_SUM_WORDS] ^= break_bits ? 1u : 0u;      /* the launch's first checksum */
            }
            for (uint32_t word = 0; word < FUSION_OUTPUT_WORDS; word++) {
                off_host += result[index][word] != expected[index][word] ? 1u : 0u;
                off_chain += result[index][word] != chain[index][word] ? 1u : 0u;
            }
            compared += FUSION_OUTPUT_WORDS;
            probe_digest_add(result[index], sizeof(result[index]));
        }
        probe_check_count(probe_case->name, variant, "host_reference", off_host, compared, corrupt);
        if (instance != PROBE_BASELINE) {
            probe_check_count(probe_case->name, variant, "bits_against_chain", off_chain, compared, break_bits);
        }
    }
    probe_cuda_check_guards(probe_case->name);
}

static void fusion_verdicts(void *, const struct probe_case *probe_case, const struct probe_result *results,
                            double noise_ratio)
{
    static const char *const measured_names[] = {"overlap_share", "chain_over_alone_sum", "streams_over_graph"};
    const struct probe_result *siblings = &results[FUSION_SIBLINGS_GRAPH];
    const double chain = results[PROBE_BASELINE].step.median, joined = siblings->step.median;
    const double first = results[FUSION_FIRST_ALONE].step.median, second = results[FUSION_SECOND_ALONE].step.median;
    const double shorter = isnan(first) || first < second ? first : second;      /* no number unless both were timed */
    const double measured[] = {(chain - joined) / shorter, chain / (first + second),
                               results[FUSION_SIBLINGS_STREAMS].step.median / joined};
    const uint64_t pair = probe_case_value(probe_case, FUSION_FLAG_PAIR);
    char figure[FUSION_FIGURE_BYTES];
    for (uint32_t index = 0; index < PROBE_COUNT_OF(measured_names); index++) {
        PROBE_REQUIRE(snprintf(figure, sizeof(figure), "%s_at_%s", measured_names[index], probe_case->name) > 0,
                      "a figure's name");
        probe_verdict(figure, PROBE_NAN, PROBE_NAN, measured[index], noise_ratio, index == 0 ? siblings->same_sign : 0u,
                      index == 0 ? siblings->rounds : 0u);
    }
    for (uint32_t index = 0; index < PROBE_COUNT_OF(fusion_bars); index++) {
        const struct fusion_bar *bar = &fusion_bars[index];
        const double at_most = bar->as_share ? 1.0 - bar->level * shorter / chain : bar->level;
        if (pair > FUSION_ROUTED_SHARED || bar->as_share != (pair < FUSION_ROUTED_SHARED ? 1u : 0u)) { continue; }
        PROBE_REQUIRE(snprintf(figure, sizeof(figure), "%s_at_%s", bar->figure, probe_case->name) > 0,
                      "a figure's name");
        probe_verdict(figure, bar->at_least ? 1.0 / at_most : at_most, (at_most - 1.0) * chain,
                      bar->at_least ? 1.0 / siblings->ratio : siblings->ratio, noise_ratio, siblings->same_sign,
                      siblings->rounds);
    }
}

static struct probe_program fusion_program = {
    "step_fusion", fusion_flags, PROBE_COUNT_OF(fusion_flags), fusion_instances, FUSION_INSTANCES,
    NULL,                                        /* no context: one card, one state */
    NULL,                                        /* nothing to configure: the instances are a fixed table */
    fusion_open_card, fusion_describe_case, fusion_check_case, fusion_prepare_step, fusion_launch_step,
    probe_cuda_mark_start, probe_cuda_elapsed_microseconds, fusion_verdicts, fusion_close_card,
    NULL};                                       /* the harness's rounds are the protocol */

int main(int argument_count, char **arguments) { return probe_main(argument_count, arguments, &fusion_program); }
