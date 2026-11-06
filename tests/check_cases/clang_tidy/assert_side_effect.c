/* A side effect in an assertion is lost where assertions are compiled out: clang-tidy knows ASSERT. */
#include "anemone/hooks.h"

void count_down(int remaining_rows);

void count_down(int remaining_rows)
{
	ASSERT(remaining_rows-- > 0);
}
