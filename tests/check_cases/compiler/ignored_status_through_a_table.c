/* Must not compile: the status of a call through a table member is dropped. This is how every seam is called. */
#include "anemone/status.h"

struct canary_operations {
	MUST_CHECK result_status (*operation)(void);
};
void drop_the_status_of_a_call_through_a_table(const struct canary_operations *operations);

void drop_the_status_of_a_call_through_a_table(const struct canary_operations *operations)
{
    operations->operation();
}
