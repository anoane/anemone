/* A table closed inside #if: what follows the first closing brace is code. */
static const int shape_table[] = {
    4,
#if WIDE_TABLE
};
#else
    8 };
#endif
