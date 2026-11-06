/* Must not compile: the status of a direct call is dropped. */
#include "anemone/status.h"

MUST_CHECK result_status canary_operation(void);
void drop_the_status_of_a_direct_call(void);

void drop_the_status_of_a_direct_call(void)
{
    canary_operation();
}
