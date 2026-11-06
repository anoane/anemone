/* A cast to an enum type in front of a brace opens no enumeration: the switch body is linted. */
enum card_kind { CARD_KIND_SMALL, CARD_KIND_LARGE };
int backoff_microseconds(int requested_kind);
int backoff_microseconds(int requested_kind)
{
    switch ((enum card_kind)requested_kind) {
    case CARD_KIND_SMALL:
        return 250000;
    default:
        return 0;
    }
}
