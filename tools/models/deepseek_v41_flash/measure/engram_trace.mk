# SPDX-License-Identifier: AGPL-3.0-or-later
include Makefile
.DEFAULT_GOAL := engram_trace
.PHONY: engram_trace
engram_trace: build/release/engram_trace
build/release/engram_trace: build/release/tools/models/deepseek_v41_flash/measure/engram_trace.o \
    $(ENGINE_OBJECTS_release)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ $(LIBRARIES) -o $@
