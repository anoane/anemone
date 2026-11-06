/* Two openings under #if and #else with one closing brace: the lint cannot follow it and says so three ways. */
#define WIDE_TABLE 1
#if WIDE_TABLE
static const int shape_table[] = {
#else
static const int shape_table[] = {
#endif
    4, 8,
};
int rest_of_the_file(int value)
{
    return value * 4096 + 77;
}
