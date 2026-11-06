/* Must not compile: a cast to void is not a look at the status. */
#include "anemone/status.h"

MUST_CHECK result_status canary_operation(void);
void cast_the_status_away(void);

void cast_the_status_away(void)
{
    (void)canary_operation();
}
