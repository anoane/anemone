/* The body of a macro with parameters is code, on every line of it. */
#define ROUND_UP_TO_PAGE(byte_count) (((byte_count) + 4095) & ~4095ull)
#define STAMP_KERNEL(name)                    \
    static int name(int value)                \
    {                                         \
        return (value * 37 + 11) >> 5;        \
    }
STAMP_KERNEL(mix)
