# SPDX-License-Identifier: AGPL-3.0-or-later
# Development-only targets. Release uses only the pinned snapshot inputs.
PYTHON := python3
.PHONY: regenerate regenerate_axes regenerate_help regenerate_vision check_generated \
        python_tests host_tests_all release_audit

# Pinned files remain ordinary build inputs. Regeneration is an explicit operation.
regenerate_axes:
	$(PYTHON) tools/cli/generate_axes.py
regenerate_help: regenerate_axes
	$(PYTHON) tools/cli/generate_help.py
regenerate_vision:
	$(PYTHON) tools/build/embed_vision_gemm_policy.py tools/vision/fixtures/gemm_policy.json generated/vision_gemm_policy.h
	$(PYTHON) tools/build/embed_vision_rope_corrections.py tools/vision/fixtures/rope_correction.hex generated/vision_rope_corrections.h
regenerate: regenerate_help regenerate_vision

check_generated:
	$(PYTHON) tools/cli/generate_axes.py --check
	$(PYTHON) tools/cli/generate_help.py --check
	@mkdir -p build/regeneration
	$(PYTHON) tools/build/embed_vision_gemm_policy.py tools/vision/fixtures/gemm_policy.json build/regeneration/vision_gemm_policy.h
	$(PYTHON) tools/build/embed_vision_rope_corrections.py tools/vision/fixtures/rope_correction.hex build/regeneration/vision_rope_corrections.h
	cmp generated/vision_gemm_policy.h build/regeneration/vision_gemm_policy.h
	cmp generated/vision_rope_corrections.h build/regeneration/vision_rope_corrections.h

python_tests:
	$(PYTHON) tools/cli/axes_registry_test.py
	$(PYTHON) tools/golden_score_test.py
	$(PYTHON) -m unittest discover -s tools/oracle -p 'test_*.py'

host_tests_all: $(BUILD_TYPES) host_test_fixtures
	@set -e; for test_program in $(filter-out %_card_test,$(TEST_PROGRAMS)); do \
	    echo "HOST_TEST $$test_program"; $$test_program; done

release_audit: release
	$(RELEASE_HAS_NO_HOOKS) build/release
	$(RELEASE_KERNELS_HAVE_NO_HOOKS) $(call KERNEL_MODULES_OF,kernels)

LAYOUT_LINT := $(PYTHON) tools/check/layout_lint.py
LITERAL_LINT := $(PYTHON) tools/check/literal_lint.py
RELEASE_HAS_NO_HOOKS := $(PYTHON) tools/check/release_has_no_hooks.py
RELEASE_KERNELS_HAVE_NO_HOOKS := $(PYTHON) tools/check/release_kernels_have_no_hooks.py
ANALYSED_FILES := $(HEADERS) $(HOST_SOURCES) $(PROGRAM_SOURCE) $(TEST_PROGRAM_SOURCES) $(TEST_SUPPORT_SOURCES)
HEADER_ALONE := $(CC) $(COMMON_FLAGS) -fsyntax-only -x c
# cppcheck is not given the CUDA header: it analyses the engine's code, and what it cannot resolve it leaves alone.
CPPCHECK_C := $(CPPCHECK) --quiet --language=c --std=c11 --enable=warning,performance,portability \
              --error-exitcode=1 $(SYSTEM_LEVEL) -Iinclude -Isource
CLANG_TIDY_FLAGS := -x c -std=c11 $(SYSTEM_LEVEL) $(INCLUDE_FLAGS)

# lint proves, for the engine and the tests: no line over 120 columns, no file or directory the build would not
# compile; no bare number, no escape hatch, no side effect in the argument of a hook, no status without
# MUST_CHECK. No count of lines: the code is as long as what it needs (STYLE.md).
lint:
	$(PYTHON) tools/cli/generate_help.py --check
	$(PYTHON) tools/cli/generate_axes.py --check
	$(LAYOUT_LINT) .
	$(LITERAL_LINT) include source tests

# analysis proves that cppcheck (which takes no suppression from a source file) and clang-tidy (.clang-tidy: names,
# magic numbers, bug patterns, the analyzer; diagnostics inside headers are kept) find nothing in any header,
# source or test, under each combination of the hook switches, so both sides of every #if are analysed.
ANALYSIS_TARGETS := $(addprefix analysis_,$(SWITCH_COMBINATIONS))
.PHONY: $(ANALYSIS_TARGETS)
analysis: $(ANALYSIS_TARGETS)
$(ANALYSIS_TARGETS): $(VISION_GEMM_POLICY_INCLUDE)
$(ANALYSIS_TARGETS): analysis_%:
	$(CPPCHECK_C) $(HOOKS_$*) $(ANALYSED_FILES)
	$(CLANG_TIDY) --quiet $(ANALYSED_FILES) -- $(CLANG_TIDY_FLAGS) $(HOOKS_$*)

TRAP_INSTRUCTION := BPT.TRAP
no_trap_in = { status=0; for module in $(1); do \
    if ! $(CUOBJDUMP) -sass $$module > $$module.sass || ! grep -q -F 'Function :' $$module.sass; then \
        echo $$module is not disassembled; status=1; \
    elif grep -q -F '$(TRAP_INSTRUCTION)' $$module.sass; then echo $$module holds a trap; status=1; fi; \
    done; [ $$status = 0 ]; }
# self_test proves that the checks above can fail: each is run once on a case under tests/check_cases/ that was made
# to break it. A case passes only if the check fails AND says what the case was made for, so a missing tool, a
# crash or an unrelated error is not taken for the failure.
#   $(call reports_exactly,<file>,<command>)   the command fails, and its list of problems (what it writes to
#                                              standard error) is exactly the file: nothing missed, nothing more
#   $(call shown_failing,<text>,<command>)     the command fails, and its output holds the text
SELF_TEST_DIRECTORY := build/self_test
WHAT_IT_SAID := $(SELF_TEST_DIRECTORY)/output
define reports_exactly
@if $(2) > /dev/null 2> $(SELF_TEST_DIRECTORY)/problems; then echo "self_test: this had to fail: $(2)"; exit 1; fi
@diff -u $(1) $(SELF_TEST_DIRECTORY)/problems
@echo "shown failing: every line of $(1)"
endef
define shown_failing
@if $(2) > $(WHAT_IT_SAID) 2>&1; then echo "self_test: this had to fail: $(2)"; exit 1; fi
@grep -q -F -e '$(1)' $(WHAT_IT_SAID) || { cat $(WHAT_IT_SAID); echo "self_test: no '$(1)' from: $(2)"; exit 1; }
@echo "shown failing: $(1)"
endef
layout_case = $(call reports_exactly,$(CASES)/layout/$(1)/expected_problems,$(LAYOUT_LINT) $(CASES)/layout/$(1))
compile_case = $(CC) $(COMMON_FLAGS) $(COMPILE_debug) -c $(CASES)/compiler/$(1) -o $(SELF_TEST_DIRECTORY)/case.o
tidy_case = $(CLANG_TIDY) --quiet $(CASES)/clang_tidy/$(1) -- $(CLANG_TIDY_FLAGS) $(HOOKS_debug)
HOOKS_HEADER_WITHOUT_TIMING_SWITCH := $(HEADER_ALONE) -DCHECKING_HOOKS_ENABLED=1 include/anemone/hooks.h
HOOKS_HEADER_WITHOUT_CHECKING_SWITCH := $(HEADER_ALONE) -DTIMING_HOOKS_ENABLED=1 include/anemone/hooks.h
FUSING_CASE := $(CC) $(COMMON_FLAGS) $(COMPILE_release) $(LINK_release) $(CASES)/contraction/fused_multiply_add.c
SPILLING_CASE := $(NVCC) $(NVCC_FLAGS) $(KERNEL_HOOKS_kernels) -arch=sm_80 -cubin $(CASES)/kernels/spilling_kernel.cu \
                 -o $(SELF_TEST_DIRECTORY)/spilling_kernel.cubin
SPILL_MESSAGE := Registers are spilled
REGISTERS_REPORT := $(SELF_TEST_DIRECTORY)/register_hungry_kernel.report
REGISTERS_CASE := $(NVCC) $(NVCC_FLAGS) $(KERNEL_HOOKS_kernels) -arch=sm_80 -cubin \
                  $(CASES)/kernels/register_hungry_kernel.cu -o $(SELF_TEST_DIRECTORY)/register_hungry_kernel.cubin
CHECKED_PRODUCT_MODULES := $(foreach architecture,$(KERNEL_ARCHITECTURES),$(foreach file,expert trunk,\
    build/kernels_checked/$(architecture)/kernel_$(file)_linear.cubin))
CANARY_debug := build/debug/check_cases/sanitizer_canary
CANARY_thread := build/thread/check_cases/sanitizer_canary
self_test: $(BUILD_TYPES) $(CANARY_debug) $(CANARY_thread)
	@mkdir -p $(SELF_TEST_DIRECTORY)/stripped
# the lint: every case of tests/check_cases/literal_lint/ in one run, then nothing it can read
	$(call reports_exactly,$(CASES)/literal_lint.expected,$(LITERAL_LINT) $(CASES)/literal_lint)
	$(call shown_failing,neither a file nor a directory,$(LITERAL_LINT) $(CASES)/no_such_directory)
	$(call shown_failing,no file to lint,$(LITERAL_LINT) $(CASES)/layout/empty)
# the layout lint: a tree that breaks every rule once, and a tree with no engine file
	$(call layout_case,every_rule)
	$(call layout_case,empty)
# header_check: a header that needs another one, and hooks.h with either switch left undefined
	$(call shown_failing,unknown type name,$(HEADER_ALONE) $(HOOKS_debug) $(CASES)/header_check/header_not_alone.h)
	$(call shown_failing,TIMING_HOOKS_ENABLED,$(HOOKS_HEADER_WITHOUT_TIMING_SWITCH))
	$(call shown_failing,CHECKING_HOOKS_ENABLED,$(HOOKS_HEADER_WITHOUT_CHECKING_SWITCH))
# the compiler: a dropped result_status, from a direct call, from a call through a table, and cast to void
	$(call shown_failing,unused-result,$(call compile_case,ignored_status_direct.c))
	$(call shown_failing,unused-result,$(call compile_case,ignored_status_through_a_table.c))
	$(call shown_failing,unused-result,$(call compile_case,ignored_status_cast_to_void.c))
# -ffp-contract=off: with the flags of release nothing is fused; with the flag turned round the case sees it
# (-fno-lto there, so that no merging of options at link time decides which of the two flags holds)
	$(FUSING_CASE) -o $(SELF_TEST_DIRECTORY)/not_fused
	$(SELF_TEST_DIRECTORY)/not_fused
	$(FUSING_CASE) -ffp-contract=fast -fno-lto -o $(SELF_TEST_DIRECTORY)/fused
	$(call shown_failing,were fused,$(SELF_TEST_DIRECTORY)/fused)
# cppcheck: an error it must report although a comment in the file asks it not to
	$(call shown_failing,arrayIndexOutOfBounds,$(CPPCHECK_C) $(CASES)/cppcheck/out_of_bounds_with_a_suppression.c)
# clang-tidy: one case for each option of .clang-tidy, because a misspelt option is ignored without a word
	$(call shown_failing,2 is a magic number,$(call tidy_case,magic_integer.c))
	$(call shown_failing,100.0 is a magic number,$(call tidy_case,magic_float.c))
	$(call shown_failing,3 is a magic number,$(call tidy_case,bit_field_width.c))
	$(call shown_failing,invalid case style for union,$(call tidy_case,union_name.c))
	$(call shown_failing,side effect in ASSERT,$(call tidy_case,assert_side_effect.c))
	$(call shown_failing,invalid case style for function,$(call tidy_case,name_in_a_header.c))
# the sanitizers: one deliberate fault each, in the builds whose test runs rely on them
	$(call shown_failing,heap-buffer-overflow,$(CANARY_debug) heap_overflow)
	$(call shown_failing,signed integer overflow,$(CANARY_debug) signed_overflow)
	$(call shown_failing,outside the range of representable values,$(CANARY_debug) float_cast_overflow)
	$(call shown_failing,signed integer overflow,$(CANARY_thread) signed_overflow)
	$(call shown_failing,data race,$(CANARY_thread) data_race)
# release_has_no_hooks: the builds that do hold hooks, and a release executable stripped of its symbols
	$(call shown_failing,hooks_test_timer,$(RELEASE_HAS_NO_HOOKS) build/profile)
	$(call shown_failing,neither must this,$(RELEASE_HAS_NO_HOOKS) build/debug)
	$(STRIP) -o $(SELF_TEST_DIRECTORY)/stripped/hooks_test build/release/hooks_test
	$(call shown_failing,no symbol main,$(RELEASE_HAS_NO_HOOKS) $(SELF_TEST_DIRECTORY)/stripped)
	$(NVCC) -O3 -std=c++17 --fmad=false -Werror all-warnings -arch=sm_80 -cubin \
	    $(CASES)/kernels/logging_canary.cu \
	    -o $(SELF_TEST_DIRECTORY)/logging_canary.cubin
	$(call shown_failing,optional device instrumentation,$(RELEASE_KERNELS_HAVE_NO_HOOKS) \
	    $(SELF_TEST_DIRECTORY)/logging_canary.cubin)
# the kernels: each checked module of the products, on each architecture, holds the traps the release set must not;
# a kernel made to spill is not built; an instance that takes more registers than its block may have is named
	@for module in $(CHECKED_PRODUCT_MODULES); do \
	    if $(call no_trap_in,$$module) > $(WHAT_IT_SAID) 2>&1; then echo "self_test: no trap in $$module"; exit 1; fi; \
	    grep -q -F "$$module holds a trap" $(WHAT_IT_SAID) || { cat $(WHAT_IT_SAID); exit 1; }; done
	@echo "shown failing: holds a trap, in each checked module of the products"
	$(call shown_failing,$(SPILL_MESSAGE),$(SPILLING_CASE))
	$(REGISTERS_CASE) > $(REGISTERS_REPORT) 2>&1
	$(call shown_failing,expert_gate_up_16 takes,$(call registers_fit,$(REGISTERS_REPORT)))

check: lint header_check analysis host_tests_all python_tests release_audit self_test
	@echo "check passed"
