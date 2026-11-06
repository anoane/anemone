/* A function that returns or takes an enum, with its brace on the same line, opens no enumeration. */
enum card_kind { CARD_KIND_SMALL, CARD_KIND_LARGE };
static enum card_kind pick(int value) {
    return value > 37 ? CARD_KIND_LARGE : CARD_KIND_SMALL;
}
static int rows_of(enum card_kind kind) { return kind == CARD_KIND_LARGE ? 512 : 1; }
