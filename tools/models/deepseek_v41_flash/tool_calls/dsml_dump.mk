# SPDX-License-Identifier: AGPL-3.0-or-later
# Run from the engine directory: make -j48 -f tools/models/deepseek_v41_flash/tool_calls/dsml_dump.mk dsml_dump
# The answer layer's sources with the engine's exact release flags, no CUDA code: build/release/dsml_dump. In a tree of
# v0.0.5-alpha's layout (with this file and dsml_dump.c copied in), its sources, for dsml_dump.c's comparison.
DSML_DUMP_FOLDER := $(dir $(lastword $(MAKEFILE_LIST)))
include Makefile

.DEFAULT_GOAL := dsml_dump
.PHONY: dsml_dump
ifneq ($(wildcard source/engine/server/api/chat_completions.c),)
DSML_DUMP_SOURCES := source/engine/server/api/chat_completions.c source/engine/input/image/request_images.c \
                     source/common/json.c source/engine/server/api/server_messages.c \
                     source/engine/server/api/server_responses.c source/common/sha256.c \
                     source/engine/input/tokenizer/tokenizer.c \
                     source/models/deepseek_v41_flash/prompt/ds41f_chat_template.c source/common/status.c \
                     source/common/hooks.c
else
DSML_DUMP_SOURCES := source/chat_completions.c source/chat_images.c source/json.c source/server_messages.c \
                     source/server_responses.c source/sha256.c source/tokenizer.c source/chat_template.c \
                     source/status.c source/hooks.c
endif
DSML_DUMP_OBJECTS := $(patsubst %.c,build/release/%.o,$(DSML_DUMP_FOLDER)dsml_dump.c $(DSML_DUMP_SOURCES))
-include $(DSML_DUMP_OBJECTS:.o=.d)

dsml_dump: build/release/dsml_dump
build/release/dsml_dump: $(DSML_DUMP_OBJECTS) $(DSML_DUMP_FOLDER)dsml_dump.mk
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $(DSML_DUMP_OBJECTS) -lm -o $@
