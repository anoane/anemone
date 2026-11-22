/* Only a `static const` table, or a `const` array at file scope, gives names: a scalar, a table that is not
   constant, a constant array inside a function and an initializer are code. */
static const int scalar_constant = 256;
struct pair { int first; int second; };
struct pair changeable_pair = {.first = 256, .second = 1};
const int shared_at_file_scope[] = { 4, 8 };
int local_table(int index);
int local_table(int index)
{
	const int not_at_file_scope[] = { 4, 8 };
	return not_at_file_scope[index];
}
