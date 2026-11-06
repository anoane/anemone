/* Every escape hatch, once. The text is searched raw, so each line is reported although most are comments. */
int silenced = 4096; // NOLINT
// NOLINTNEXTLINE(readability-magic-numbers)
// cppcheck-suppress arrayIndexOutOfBounds
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC system_header
#pragma GCC optimize("O3")
__attribute__((optimize("O3"))) static void other_flags(void) {}
__attribute__((__optimize__("O3"))) static void other_flags_again(void) {}
__attribute__((no_sanitize("undefined"))) static void unsanitized(void) {}
__attribute__((no_sanitize_address)) static void unsanitized_address(void) {}
__attribute__((no_address_safety_analysis)) static void unsanitized_old_spelling(void) {}
#include <assert.h>
#include <cassert>
/* built with -ffast-math */
/* nvcc --use_fast_math */
#pragma STDC FP_CONTRACT ON
_Pragma("GCC \
diagnostic ignored \"-Wshadow\"")
static void plan_optimize(void) {}
