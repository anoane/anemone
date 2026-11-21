# SPDX-License-Identifier: AGPL-3.0-or-later
# The Engram replay tool (tools/measure/engram_replay.c), built with the release flags:
# make -f tools/measure/engram_replay.mk
include Makefile
.DEFAULT_GOAL := engram_replay
.PHONY: engram_replay
engram_replay: build/release/engram_replay
build/release/engram_replay: build/release/tools/measure/engram_replay.o $(ENGINE_OBJECTS_release)
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $^ $(LIBRARIES) -o $@
