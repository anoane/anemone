/* A quote that does not close on its line is reported and hides nothing. */
const char *unclosed_string = "no closing quote; int bare_after_the_quote = 42;
char unclosed_character = 'x; int bare_after_the_character = 43;
const char *spliced_string = "a string continued \
on the next line";
