# SPDX-License-Identifier: AGPL-3.0-or-later
# The Engram I/O bench (tools/measure/engram_io_bench.c): a measurement probe built from the engine's pack, Engram,
# digest and storage sources with the release flags, and nothing else. make -f tools/measure/engram_io_bench.mk
include Makefile
.DEFAULT_GOAL := engram_io_bench
ENGRAM_IO_BENCH_SOURCES := tools/measure/engram_io_bench.c tools/measure/engram_io_engines.c \
                           tools/measure/engram_io_characterise.c
ENGRAM_IO_ENGINE_SOURCES := source/engram.c source/engram_disk.c source/storage_direct_linux.c source/pack.c \
                            source/pack_digest.c source/status.c source/hooks.c \
                            source/weight_format.c source/exl3.c source/exl3_layout.c source/prepare_common.c \
                            source/engram_hot.c source/sha256.c
ENGRAM_IO_BENCH_OBJECTS := $(patsubst %.c,build/release/%.o,$(ENGRAM_IO_BENCH_SOURCES) $(ENGRAM_IO_ENGINE_SOURCES))
build/release/tools/measure/%.o: tools/measure/%.c tools/measure/engram_io_bench.h Makefile \
    tools/measure/engram_io_bench.mk
	@mkdir -p $(dir $@)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(DEPENDENCY_FLAGS) -Wno-missing-prototypes -c $< -o $@
.PHONY: engram_io_bench
engram_io_bench: build/release/engram_io_bench
build/release/engram_io_bench: $(ENGRAM_IO_BENCH_OBJECTS)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ -lm -o $@
# The step I/O stress (tools/measure/engram_step_stress.c): serving's engram_steps over real pack copies at peak load.
ENGRAM_STEP_STRESS_SOURCES := tools/measure/engram_step_stress.c source/engram_step.c source/storage_flight_linux.c \
                              $(ENGRAM_IO_ENGINE_SOURCES)
ENGRAM_STEP_STRESS_OBJECTS := $(patsubst %.c,build/release/%.o,$(ENGRAM_STEP_STRESS_SOURCES))
.PHONY: engram_step_stress
engram_step_stress: build/release/engram_step_stress
build/release/engram_step_stress: $(ENGRAM_STEP_STRESS_OBJECTS)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ -lm -lpthread -o $@
# The SHA-256 rate probe (tools/measure/sha256_rate.c): the copier's chunk hash, one stream a call against two side by
# side, on 1 to N threads.
SHA256_RATE_SOURCES := tools/measure/sha256_rate.c source/sha256.c source/hooks.c source/status.c
SHA256_RATE_OBJECTS := $(patsubst %.c,build/release/%.o,$(SHA256_RATE_SOURCES))
.PHONY: sha256_rate
sha256_rate: build/release/sha256_rate
build/release/sha256_rate: $(SHA256_RATE_OBJECTS)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ -lpthread -o $@
# The load arena probe (tools/measure/load_arena_probe.c, 17 Nov 2026): the load stage's host read into fresh or
# touched arenas, with each point's CPU split and page counters; linked with the engine's objects, not part of anemone.
LOAD_ARENA_PROBE_OBJECTS := build/release/tools/measure/load_arena_probe.o $(ENGINE_OBJECTS_release)
.PHONY: load_arena_probe
load_arena_probe: build/release/load_arena_probe
build/release/load_arena_probe: $(LOAD_ARENA_PROBE_OBJECTS)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ $(LIBRARIES) -o $@
