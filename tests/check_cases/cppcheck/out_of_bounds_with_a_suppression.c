/* cppcheck must report the write past the table. The comment that asks it not to must change nothing: the build
 * runs cppcheck without inline suppressions. */
#define TABLE_ENTRY_COUNT 2

void write_past_the_table(void);

void write_past_the_table(void)
{
    int table[TABLE_ENTRY_COUNT] = {0, 0};
    /* cppcheck-suppress arrayIndexOutOfBounds */
    table[TABLE_ENTRY_COUNT] = 1;
}
