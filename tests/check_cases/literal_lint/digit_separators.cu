/* A digit separator is part of its number: it opens no character constant and hides nothing after it. */
static int row_capacity(void) { return 4'096; }
static int not_hidden_by_the_separator(int value)
{
    return value * 4096 + 77;
}
static int second(void) { return 1'000'000; }
