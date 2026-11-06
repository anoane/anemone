# SPDX-License-Identifier: AGPL-3.0-or-later
# Run from engine/: make -j48 -f tools/measure/request_front_end.mk request_front_end > build.log 2>&1
# This inherits the engine's exact release flags, without building or linking any CUDA code.
include Makefile

.DEFAULT_GOAL := request_front_end
.PHONY: request_front_end
REQUEST_FRONT_END_SOURCES := tools/measure/request_front_end.c source/tokenizer.c source/json.c \
                            source/chat_template.c source/chat_completions.c source/status.c source/hooks.c
REQUEST_FRONT_END_OBJECTS := $(patsubst %.c,build/release/%.o,$(REQUEST_FRONT_END_SOURCES))
-include $(REQUEST_FRONT_END_OBJECTS:.o=.d)

request_front_end: build/release/request_front_end
build/release/request_front_end: $(REQUEST_FRONT_END_OBJECTS) tools/measure/request_front_end.mk
	$(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $(REQUEST_FRONT_END_OBJECTS) -lm -o $@
