/* A string or a comment neither hides the code beside it nor counts as code. */
const char *text = "#define X 5"; int bare_beside_a_string = 42;
int bare_before_comments = 43; /* 99 */ // 98
