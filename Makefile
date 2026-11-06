# SPDX-License-Identifier: AGPL-3.0-or-later
# Anemone: build and checks. Everything is built and run on the server, never on a development laptop.
#
#   make debug      -O1, address and undefined-behaviour sanitizers, every hook
#   make thread     the same code under the thread sanitizer, every hook
#   make profile    the flags of release, with the timing hooks (timers, counters, their report) left in
#   make release    optimised and hardened, every hook compiled out: what ships
#   make check      lint + header_check + analysis + tests + self_test; each target below says what it proves
#   make card_tests the tests that need the cards (tests/*_card_test.c): run from a batch, with nothing else on them
#   make clean

SHELL := /bin/sh
.DELETE_ON_ERROR:

CC := gcc
CUDA_DIRECTORY := /usr/local/cuda
NVCC := $(CUDA_DIRECTORY)/bin/nvcc
COMPUTE_SANITIZER := $(CUDA_DIRECTORY)/bin/compute-sanitizer
CUOBJDUMP := $(CUDA_DIRECTORY)/bin/cuobjdump
CPPCHECK := cppcheck
CLANG_TIDY := clang-tidy
STRIP := strip

# ---- What is built ---------------------------------------------------------------------------------------------
# Found, never listed: a file in these directories is compiled, analysed and tested, and layout_lint.py refuses an
# engine or test file anywhere else. source/main.c is the program, build/<build>/anemone; every other source/*.c is
# linked into it and into every test program. tests/<name>_test.c holds a main and becomes the test program
# <name>_test; any other tests/*.c is linked into every test program. A test named <name>_card_test needs the
# cards: `make tests` builds it and `make card_tests` runs it.
HEADERS := $(sort $(wildcard include/anemone/*.h source/*.h tests/*.h))
PROGRAM_SOURCE := source/main.c
HOST_SOURCES := $(filter-out $(PROGRAM_SOURCE),$(sort $(wildcard source/*.c)))
TEST_PROGRAM_SOURCES := $(sort $(wildcard tests/*_test.c))
TEST_SUPPORT_SOURCES := $(filter-out $(TEST_PROGRAM_SOURCES),$(sort $(wildcard tests/*.c)))
KERNEL_SOURCES := $(sort $(wildcard source/*.cu))
# The kernel modules as one C table (tools/build/embed_kernel_images.c), linked into every program: the release set
# into profile and release, the checked set into debug and thread (see Kernels).
KERNEL_SET_debug := kernels_checked
KERNEL_SET_thread := kernels_checked
KERNEL_SET_profile := kernels
KERNEL_SET_release := kernels
CASES := tests/check_cases
CANARY_SOURCE := $(CASES)/sanitizers/sanitizer_canary.c
ifeq ($(and $(HEADERS),$(HOST_SOURCES),$(TEST_PROGRAM_SOURCES)),)
$(error no header, source or test program found: make runs in the engine directory)
endif

# ---- Flags -----------------------------------------------------------------------------------------------------
# The host this build is tuned for. (The check of the processor at start-up comes with the file that holds main.)
HOST_ARCHITECTURE := znver4
WARNINGS := -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion -Wshadow -Wstrict-prototypes \
            -Wmissing-prototypes -Wpointer-arith -Wcast-qual -Wcast-align=strict -Wwrite-strings -Wformat=2 \
            -Wformat-signedness -Wswitch-enum -Wvla -Walloca -Wnull-dereference -Wundef -Werror
# Linux is the only platform: direct reads and what the kernel tells about a drive are behind this level.
SYSTEM_LEVEL := -D_GNU_SOURCE
# -iquote source: a header private to the engine's sources. -isystem: the CUDA driver interface, a C header that
# only source/backend_cuda.c includes.
INCLUDE_FLAGS := -Iinclude -iquote source -isystem $(CUDA_DIRECTORY)/include
# -ffp-contract=off: a result never depends on what the compiler chose to fuse. A fused multiply-add is written
# fma() where it is wanted.
COMMON_FLAGS := -std=c11 $(SYSTEM_LEVEL) $(INCLUDE_FLAGS) -march=$(HOST_ARCHITECTURE) -ffp-contract=off -pthread \
                $(WARNINGS)
DEPENDENCY_FLAGS := -MMD -MP
# The CUDA driver library: part of the driver, not of the toolkit.
LIBRARIES := -lcuda -lm

# The two hook switches of include/anemone/hooks.h. Checking hooks: assertions, fault points, and the information,
# debug and trace logs. Timing hooks: timers, counters and their report. Either switch enables logging;
# both zero compile out error/warning logs too. thread has the switches of debug, so
# debug, profile and release are every combination a build uses.
HOOKS_debug := -DCHECKING_HOOKS_ENABLED=1 -DTIMING_HOOKS_ENABLED=1
HOOKS_thread := $(HOOKS_debug)
HOOKS_profile := -DCHECKING_HOOKS_ENABLED=0 -DTIMING_HOOKS_ENABLED=1
HOOKS_release := -DCHECKING_HOOKS_ENABLED=0 -DTIMING_HOOKS_ENABLED=0
SWITCH_COMBINATIONS := debug profile release

# Sanitizer builds. gcc's "undefined" leaves out float-cast-overflow and bounds-strict; a numeric engine wants both.
SANITIZED := -O1 -g -fno-omit-frame-pointer -U_FORTIFY_SOURCE -fno-sanitize-recover=all
UNDEFINED_BEHAVIOUR := undefined,float-cast-overflow,bounds-strict
# Optimised builds: one flag set, so that profile times what release ships. The hardening is stated here, not
# inherited from however the compiler was configured.
HARDENING := -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 -fstack-protector-strong -fstack-clash-protection \
             -fcf-protection=full -fPIE
HARDENED_LINK := -pie -Wl,-z,relro,-z,now,-z,noexecstack
# -fno-tree-vectorize: gcc 15.2's vectorizer (-O3 -flto -march=znver4) miscompiled host loops of the program builder
# on 25 October 2026: a release build dropped QUERY_NORMED's chunk offset in rows_select (every slab of a full program
# read slab 0's normed queries; seen under gdb), where -O1, the profile build and this flag computed it right
# (PLAN_0.2_ALPHA.md section 9). The host code is control, not arithmetic: no loop of it needs the vectorizer, and a
# silent wrong answer costs more than any of them saves. The kernels are nvcc's.
OPTIMISED := -O3 -fno-tree-vectorize -flto=auto -fno-plt -DNDEBUG $(HARDENING)

COMPILE_debug := $(SANITIZED) -fsanitize=address,$(UNDEFINED_BEHAVIOUR) $(HOOKS_debug)
COMPILE_thread := $(SANITIZED) -fsanitize=thread,$(UNDEFINED_BEHAVIOUR) $(HOOKS_thread)
COMPILE_profile := $(OPTIMISED) -g -fno-omit-frame-pointer $(HOOKS_profile)
COMPILE_release := $(OPTIMISED) $(HOOKS_release)
LINK_debug :=
LINK_thread :=
LINK_profile := $(HARDENED_LINK)
LINK_release := $(HARDENED_LINK)

# Whether a sanitizer report stops the program is decided here, for every recipe: neither the environment nor the
# command line of make can switch a report off.
ADDRESS_SANITIZER_OPTIONS := abort_on_error=1:detect_leaks=1:strict_string_checks=1:detect_stack_use_after_return=1
override export ASAN_OPTIONS := $(ADDRESS_SANITIZER_OPTIONS)
override export UBSAN_OPTIONS := halt_on_error=1:print_stacktrace=1
override export TSAN_OPTIONS := halt_on_error=1:second_deadlock_stack=1

# ---- The four builds -------------------------------------------------------------------------------------------
BUILD_TYPES := debug thread profile release
.PHONY: all $(BUILD_TYPES) kernels lint header_check analysis tests card_tests self_test check clean
all: release

# CLI help/axis tables are pinned source/cli_*_data.h; no generators are needed.

# An object depends on its source, on every header it included (the .d file the compiler writes beside it) and on
# this Makefile: a changed flag rebuilds everything, and no check passes on objects of the flags before.
define BUILD_RULES
ENGINE_OBJECTS_$(1) := $$(patsubst %.c,build/$(1)/%.o,$$(HOST_SOURCES) build/$$(KERNEL_SET_$(1))/kernel_images.c)
TEST_SUPPORT_OBJECTS_$(1) := $$(patsubst %.c,build/$(1)/%.o,$$(TEST_SUPPORT_SOURCES))
TEST_PROGRAMS_$(1) := $$(patsubst tests/%.c,build/$(1)/%,$$(TEST_PROGRAM_SOURCES))
PROGRAM_$(1) := build/$(1)/anemone
$(1): $$(TEST_PROGRAMS_$(1)) $$(PROGRAM_$(1))
build/$(1)/source/startup_axes.o: source/cli_axes_data.h
build/$(1)/%.o: %.c Makefile
	@mkdir -p $$(dir $$@)
	$$(CC) $$(COMMON_FLAGS) $$(COMPILE_$(1)) $$(DEPENDENCY_FLAGS) -c $$< -o $$@
$$(TEST_PROGRAMS_$(1)): build/$(1)/%: build/$(1)/tests/%.o $$(ENGINE_OBJECTS_$(1)) $$(TEST_SUPPORT_OBJECTS_$(1)) Makefile
	$$(CC) $$(COMMON_FLAGS) $$(COMPILE_$(1)) $$(LINK_$(1)) $$(filter %.o,$$^) $$(LIBRARIES) -o $$@
$$(PROGRAM_$(1)): build/$(1)/$$(PROGRAM_SOURCE:.c=.o) $$(ENGINE_OBJECTS_$(1)) Makefile
	$$(CC) $$(COMMON_FLAGS) $$(COMPILE_$(1)) $$(LINK_$(1)) $$(filter %.o,$$^) $$(LIBRARIES) -o $$@
build/$(1)/check_cases/sanitizer_canary: build/$(1)/$$(CANARY_SOURCE:.c=.o) Makefile
	@mkdir -p $$(dir $$@)
	$$(CC) $$(COMMON_FLAGS) $$(COMPILE_$(1)) $$(LINK_$(1)) $$< -o $$@
endef
$(foreach build_type,$(BUILD_TYPES),$(eval $(call BUILD_RULES,$(build_type))))
TEST_PROGRAMS := $(foreach build_type,$(BUILD_TYPES),$(TEST_PROGRAMS_$(build_type)))
-include $(foreach build_type,$(BUILD_TYPES),$(patsubst %.c,build/$(build_type)/%.d,\
    $(HOST_SOURCES) $(PROGRAM_SOURCE) build/$(KERNEL_SET_$(build_type))/kernel_images.c $(TEST_PROGRAM_SOURCES) \
    $(TEST_SUPPORT_SOURCES) $(CANARY_SOURCE)))

# ---- Kernels ---------------------------------------------------------------------------------------------------
# Device code only. Each source/*.cu becomes one module per architecture, which the backend loads when it opens a
# card. --fmad=false: as on the host, nothing is fused unless the source says so. Two sets: build/kernels/ with the
# device checks compiled out (CHECKING_HOOKS_ENABLED=0), for profile and release, and build/kernels_checked/
# with them, for debug and thread. Each set is written into one C table of the same name, so that a program carries
# its kernels and reads none from a path. Beside every module lies the assembler's report (.report); a spill, any
# use of local memory, or an instance of a product kernel whose registers do not fit its block fails the build.
KERNEL_ARCHITECTURES := sm_80 sm_120a
KERNEL_SETS := kernels kernels_checked
KERNEL_HOOKS_kernels := -DCHECKING_HOOKS_ENABLED=0 -DNDEBUG
KERNEL_HOOKS_kernels_checked := -DCHECKING_HOOKS_ENABLED=1
NVCC_FLAGS := -O3 -std=c++17 -Iinclude --fmad=false -Werror all-warnings \
              -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow,-Werror \
              -Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage,--warning-as-error
# The PRO's own arithmetic (class 120 native, backend_seam.h; main.c's default): the kernel files whose sums
# emulate class 80 on class 120 are built a second time for sm_120a with LINEAR_CLASS_FORMS, as <file>_native,
# the module a card of that class takes (linear.c, attention_core.c). The plain sm_120a builds stay class 120's
# emulation (--pro_emulation=1), the sm_80 builds class 80's own.
NATIVE_KERNEL_SOURCES := source/kernel_trunk_linear.cu source/kernel_expert_linear.cu source/kernel_attention_core.cu \
                         source/kernel_index_scan.cu source/kernel_attention_output.cu
# The trunk and expert ordered/relaxed modules use class 80's operands and are never selected on the PRO.
SM_80_ONLY_KERNEL_SOURCES := source/kernel_trunk_relaxed.cu source/kernel_expert_relaxed.cu source/kernel_exl3_tc.cu source/kernel_exl3_tcp.cu \
                            source/kernel_head_mma.cu source/kernel_trunk_mma.cu
KERNEL_SOURCES_OF = $(if $(filter sm_80,$(1)),$(KERNEL_SOURCES),$(filter-out $(SM_80_ONLY_KERNEL_SOURCES),$(KERNEL_SOURCES)))
KERNEL_MODULES_OF = $(foreach architecture,$(KERNEL_ARCHITECTURES),\
    $(patsubst source/%.cu,build/$(1)/$(architecture)/%.cubin,$(call KERNEL_SOURCES_OF,$(architecture)))) \
    $(patsubst source/%.cu,build/$(1)/sm_120a/%_native.cubin,$(NATIVE_KERNEL_SOURCES))
KERNEL_MODULES := $(foreach kernel_set,$(KERNEL_SETS),$(call KERNEL_MODULES_OF,$(kernel_set)))
# An instance of S slices of a product kernel has a block of 32 S threads (gate with up, the producer and the
# attention's output groups: 64 S, linear_records.h, expert_fused_records.h, attention_output_records.h; the wide and
# interleaved trunk instances as the thin ones, the shared join as the trunk's): the registers the assembler gave it,
# in the register file's units of 8, must fit the file for that block, or every launch of it would fail at run time.
# registers_fit reads a report and fails naming each instance that does not.
REGISTER_FILE := 65536
REGISTER_UNIT := 8
THREADS_PER_SLICE := 32
GATE_UP_THREADS_PER_SLICE := 64
PRODUCT_NAMES := expert_(gate_up(_quantize)?|down|shared_join)|trunk_linear_(quantized|bfloat16)(_wide|_interleaved)?
INSTANCE_NAME := ^($(PRODUCT_NAMES)|attention_(groups|projection))_[0-9]+$$
registers_fit = awk '/Function properties for/ { name = $$NF } /Used [0-9]+ registers/ && name ~ /$(INSTANCE_NAME)/ { \
    for (field = 1; field < NF; field++) if ($$field == "Used") registers = $$(field + 1); \
    registers = int((registers + $(REGISTER_UNIT) - 1) / $(REGISTER_UNIT)) * $(REGISTER_UNIT); \
    threads = substr(name, match(name, /[0-9]+$$/)) * \
              (name ~ /gate_up|attention_groups/ ? $(GATE_UP_THREADS_PER_SLICE) : $(THREADS_PER_SLICE)); \
    if (registers * threads > $(REGISTER_FILE)) { print name " takes " registers " registers in a block of " threads; \
                                                 wrong = 1 } } END { exit wrong }' $(1)
define KERNEL_RULES
build/$(1)/$(2)/%.cubin: source/%.cu Makefile
	@mkdir -p $$(dir $$@)
	$$(NVCC) $$(NVCC_FLAGS) $$(KERNEL_HOOKS_$(1)) -arch=$(2) -cubin -MMD -MF $$(@:.cubin=.d) $$< -o $$@ \
	    > $$(@:.cubin=.report) 2>&1 || { cat $$(@:.cubin=.report); exit 1; }
	@$$(call registers_fit,$$(@:.cubin=.report))
endef
$(foreach kernel_set,$(KERNEL_SETS),$(foreach architecture,$(KERNEL_ARCHITECTURES),\
    $(eval $(call KERNEL_RULES,$(kernel_set),$(architecture)))))
# Only the vision CUDA module consumes this generated numeric initializer.
VISION_ROPE_CORRECTION_DATA := generated/vision_rope_corrections.h
VISION_ROPE_CORRECTION_INCLUDE := build/vision_rope_corrections.inc
VISION_ROPE_CORRECTION_KERNELS := $(foreach kernel_set,$(KERNEL_SETS),$(foreach architecture,$(KERNEL_ARCHITECTURES),\
    build/$(kernel_set)/$(architecture)/kernel_vision.cubin))
$(VISION_ROPE_CORRECTION_INCLUDE): $(VISION_ROPE_CORRECTION_DATA) Makefile
	@mkdir -p $(dir $@)
	cp $(VISION_ROPE_CORRECTION_DATA) $@
$(VISION_ROPE_CORRECTION_KERNELS): $(VISION_ROPE_CORRECTION_INCLUDE)
$(VISION_ROPE_CORRECTION_KERNELS): private NVCC_FLAGS += -Ibuild

# The hand-written GEMM policy consumes declarations built from measured numeric tables.
VISION_GEMM_POLICY_DATA := generated/vision_gemm_policy.h
VISION_GEMM_POLICY_INCLUDE := build/vision_gemm_policy.inc
VISION_GEMM_POLICY_OBJECTS := $(foreach build_type,$(BUILD_TYPES),build/$(build_type)/source/vision_gemm_policy.o)
$(VISION_GEMM_POLICY_INCLUDE): $(VISION_GEMM_POLICY_DATA) Makefile
	@mkdir -p $(dir $@)
	cp $(VISION_GEMM_POLICY_DATA) $@
$(VISION_GEMM_POLICY_OBJECTS): $(VISION_GEMM_POLICY_INCLUDE)

define NATIVE_KERNEL_RULES
build/$(1)/sm_120a/%_native.cubin: source/%.cu Makefile
	@mkdir -p $$(dir $$@)
	$$(NVCC) $$(NVCC_FLAGS) $$(KERNEL_HOOKS_$(1)) -DLINEAR_CLASS_FORMS -arch=sm_120a -cubin \
	    -MMD -MF $$(@:.cubin=.d) $$< -o $$@ > $$(@:.cubin=.report) 2>&1 || { cat $$(@:.cubin=.report); exit 1; }
	@$$(call registers_fit,$$(@:.cubin=.report))
endef
$(foreach kernel_set,$(KERNEL_SETS),$(eval $(call NATIVE_KERNEL_RULES,$(kernel_set))))
EMBED_KERNEL_IMAGES := build/tools/embed_kernel_images
$(EMBED_KERNEL_IMAGES): tools/build/embed_kernel_images.c Makefile
	@mkdir -p $(dir $@)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $< -o $@
define KERNEL_IMAGES_RULE
build/$(1)/kernel_images.c: $(call KERNEL_MODULES_OF,$(1)) $(EMBED_KERNEL_IMAGES) Makefile
	$(EMBED_KERNEL_IMAGES) $$@ $(call KERNEL_MODULES_OF,$(1))
endef
$(foreach kernel_set,$(KERNEL_SETS),$(eval $(call KERNEL_IMAGES_RULE,$(kernel_set))))
kernels: $(KERNEL_MODULES)
-include $(KERNEL_MODULES:.cubin=.d)

# ---- Snapshot checks ---------------------------------------------------------
TEST_BUILD ?= release
HEADER_ALONE := $(CC) $(COMMON_FLAGS) -fsyntax-only -x c
HOST_FIXTURE_WRITER := build/tools/write_host_fixtures
.PHONY: host_tests host_test_fixtures lint analysis self_test check
$(HOST_FIXTURE_WRITER): tools/build/write_host_fixtures.c Makefile
	@mkdir -p $(dir $@)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $< -o $@
host_test_fixtures: $(HOST_FIXTURE_WRITER)
	@mkdir -p tools/vision/fixtures
	$(HOST_FIXTURE_WRITER)
host_tests: $(TEST_BUILD) host_test_fixtures
	@set -e; for test_program in $(filter-out %_card_test,$(TEST_PROGRAMS_$(TEST_BUILD))); do \
	    echo "HOST_TEST $$test_program"; $$test_program; done

# header_check proves that every header compiles alone, under each combination of the hook switches.
HEADER_CHECK_TARGETS := $(addprefix header_check_,$(SWITCH_COMBINATIONS))
.PHONY: $(HEADER_CHECK_TARGETS)
header_check: $(HEADER_CHECK_TARGETS)
$(HEADER_CHECK_TARGETS): header_check_%:
	@echo "header_check ($*): $(words $(HEADERS)) headers, each compiled alone"
	@set -e; for header in $(HEADERS); do $(HEADER_ALONE) $(HOOKS_$*) $$header; done

tests: host_tests

# card_tests proves that every test of the cards passes on every card of this machine: under the address and
# undefined-behaviour sanitizers, with the timing hooks only, with no hook, and then with every access of the
# kernels to card memory checked by compute-sanitizer (which is told not to count a driver call that answers with
# an error: the tests ask for a kernel that does not exist and for other refusals on purpose, and hold every such
# answer to its status themselves). The cards must be free: a batch that stops whatever else
# uses them runs this target, `make check` does not. (Not under the thread sanitizer: the driver library is not
# built for it. What the loader does with threads is proved by `make tests` against a backend in host memory.)
CARD_TEST_BUILD_TYPES := debug profile release
# The driver maps memory where the address sanitizer keeps a protected gap: a program that opens a card needs that
# gap unprotected, or the driver's start-up fails for lack of memory. Only this target lifts it.
card_tests: override export ASAN_OPTIONS := $(ADDRESS_SANITIZER_OPTIONS):protect_shadow_gap=0
CARD_TEST_PROGRAMS := $(filter %_card_test,$(foreach build_type,$(CARD_TEST_BUILD_TYPES),$(TEST_PROGRAMS_$(build_type))))
# The gate of the linear compares with what the reference captured from a real pack, which a machine may not hold:
# LINEAR_GATE="<gate file> <pack copy>..." gives it its arguments in every build; without them it says that it
# compared nothing. It is not run under compute-sanitizer, which takes a program that never calls the driver for a
# failure: its kernels, and the model's shapes, are those of the tests that are.
LINEAR_GATE ?=
GATE_TEST := %linear_gate_card_test
card_tests: $(CARD_TEST_BUILD_TYPES)
	@set -e; for test_program in $(filter-out $(GATE_TEST),$(CARD_TEST_PROGRAMS)); do \
	    echo "$$test_program"; $$test_program; done
	@set -e; for test_program in $(filter $(GATE_TEST),$(CARD_TEST_PROGRAMS)); do \
	    echo "$$test_program $(LINEAR_GATE)"; $$test_program $(LINEAR_GATE); done
	@set -e; for test_program in $(filter-out $(GATE_TEST),$(filter %_card_test,$(TEST_PROGRAMS_profile))); do \
	    echo "compute-sanitizer $$test_program"; \
	    $(COMPUTE_SANITIZER) --error-exitcode 1 --report-api-errors no $$test_program; done

clean:
	rm -rf build

# Explicit development checks and regeneration; no implicit Python prerequisites.
ifneq ($(wildcard tools/development.mk),)
include tools/development.mk
else
lint analysis self_test check:
	@echo "Repository-only audit target; development tooling is excluded from this snapshot." >&2
	@exit 2
endif
