/* Reports nothing: every spelling and every place the lint allows. If this file is reported, the lint is wrong. */
#include <stdint.h>
#include <x86intrin.h>

#define ROW_CAPACITY 4096
#define SCALE_OF_ONE_STEP 0x1.8p4
#define LONG_CONSTANT (ROW_CAPACITY * \
                       8)
#define TERMINATOR '\0'

enum table_kind { TABLE_KIND_SMALL = 4, TABLE_KIND_LARGE = 64 };
typedef enum {
    SHIFT_OF_A_PAGE = 12,
} shift;
static const int shape_table[] = {4, 8, 16};
struct shape { int rows; int columns; };
static const struct shape nested_table[] = {
    {.rows = 4, .columns = 8},
    {.rows = 16, .columns = 32},
};

int allowed_numbers(const char *text, int64_t value);

int allowed_numbers(const char *text, int64_t value)
{
    uint64_t unsigned_forms = 0u + 1U + 0ul + 1UL + 0ull + 1ULL + 1lu + 1llu;
    int64_t signed_forms = 0l + 1L + 0ll + 1LL - 1;
    double floating_forms = 0.0 + 1.0 + 0. + 1. + 0.0f + 1.0F + 1.0L;
    int compared = value == 1 || value <= 0 || value >= 1 || value != 0;
    return text[0] != '\0' && text[1] == '1' && unsigned_forms > 0 && signed_forms > 0 && floating_forms > 0.0
           && compared && shape_table[0] && nested_table[1].rows && "4096 in a string"[0] == '4';   // 4096
}
