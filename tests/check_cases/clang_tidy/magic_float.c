/* 100.0 is on the list clang-tidy ignores by default: with the engine's options it is reported. */
double as_percentage(double fraction);

double as_percentage(double fraction)
{
    return fraction * 100.0;
}
