/* A table or enumeration ends at its closing brace: the rest of that line is code. */
static const int shape_table[] = {
    4, 8,
}; int bare_after_a_table = 42;
static const int one_line[] = {4, 8}; int bare_after_one_line = 43;
enum { FIRST_VALUE = 44 }; int bare_after_an_enumeration = 45;
static const int sized_by_a_bare_number[2] = {4, 8};
