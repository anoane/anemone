# SPDX-License-Identifier: AGPL-3.0-or-later
include Makefile
.DEFAULT_GOAL := engram_trace
.PHONY: engram_trace
engram_trace: build/release/engram_trace
build/release/engram_trace: build/release/tools/measure/engram_trace.o $(ENGINE_OBJECTS_release)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ $(LIBRARIES) -o $@
.PHONY: engram_replay
engram_replay: build/release/engram_replay
build/release/engram_replay: build/release/tools/measure/engram_replay.o $(ENGINE_OBJECTS_release)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ $(LIBRARIES) -o $@
