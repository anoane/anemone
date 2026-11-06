/* Only a `static const` table gives names: a scalar, a table that is not constant and an initializer are code. */
static const int scalar_constant = 256;
struct pair { int first; int second; };
struct pair changeable_pair = {.first = 256, .second = 1};
const int not_static[] = {4, 8};
