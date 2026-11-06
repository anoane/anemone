/* 2 is a power of two and is on the list clang-tidy ignores by default: with the engine's options it is reported. */
int doubled(int value);

int doubled(int value)
{
    return value * 2;
}
