/* Whatever returns a status carries MUST_CHECK: the first five do not, the rest are clean. */
#include "anemone/status.h"

result_status prototype_without_it(void);
extern result_status extern_prototype_without_it(void);
struct canary_operations {
	result_status (*member_without_it)(void);
	MUST_CHECK result_status (*member_with_it)(void);
};
typedef result_status (*callback_without_it)(void);
static inline result_status static_function_without_it(void)
{
	return STATUS_OK;
}

MUST_CHECK result_status prototype_with_it(void);
MUST_CHECK
result_status prototype_with_it_on_the_line_before(void);
MUST_CHECK static inline result_status static_function_with_it(void)
{
	return STATUS_OK;
}
static inline MUST_CHECK result_status with_it_after_static(void)
{
	return STATUS_OK;
}
result_status prototype_with_it(void)
{
	result_status status = (result_status)0;
	return status;
}
const char *status_text(result_status status);
