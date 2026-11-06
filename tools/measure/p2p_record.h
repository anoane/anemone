/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * p2p_record.h: the peer-to-peer gate's record, the key it is kept under, and the AER counters of the cards and their
 * ports: host code only, which p2p_gate_probe.cu and link_hop_probe.cu share through p2p_common.cuh
 * (design/peer_to_peer.md 6).
 *
 * THE RECORD: a text file of four lines that the gate writes and every peer user reads.
 *   p2p_gate_record 1
 *   key <16 hex digits>
 *   verdict passed | refused | none_offered | incomplete
 *   pairs[ <accessor's PCI location>><owner's PCI location>]...     the ordered pairs that passed; none unless passed
 * The key hashes the gate's version, the driver's version (/sys/module/nvidia/version), the loaded module's
 * srcversion and build id (the linker's hash of the whole module; srcversion leaves out the prebuilt core object, where
 * the peer-to-peer patches live), the whole of /proc/driver/nvidia/params, every card the driver holds with its PCI
 * location, UUID and VBIOS (/proc/driver/nvidia/gpus, sorted by location), the boot id and the kernel's command line.
 * Any change gives another key, and a record under another key is stale: no pair. A reader trusts `passed` under this
 * machine's key only, and of it only the listed pairs. UUIDs, the boot id and the command line enter hashes only.
 *
 * Measurement tool: not part of the engine.
 */
#ifndef P2P_RECORD_H
#define P2P_RECORD_H
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define P2P_GATE_VERSION 2u                      /* the gate's own version: part of the key */
#define P2P_RECORD_FORMAT 1u
#define P2P_EXIT_CRASH 4                         /* a failed call or an impossible state: no verdict was reached */
#define P2P_MAX_CARDS 8
#define P2P_BUS_TEXT_BYTES 16                    /* "0000:01:00.0" and its end */
#define P2P_PAIR_TEXT_BYTES (2 * P2P_BUS_TEXT_BYTES)
#define P2P_NAME_BYTES 256
#define P2P_VALUE_BYTES 512                      /* one value read from a file of the driver */
#define P2P_FILE_BYTES 16384                     /* a file of the driver or a record, whole */
#define P2P_PATH_BYTES 4096
#define P2P_VERDICT_BYTES 32
#define P2P_HEX_DIGITS_OF_KEY 16
#define P2P_HEX_BASE 16
#define P2P_DECIMAL_BASE 10
#define P2P_NOTE_HEADER_WORDS 3                  /* an ELF note: name size, descriptor size, type; then both, padded */
#define P2P_NOTE_ALIGNMENT 4u
#define P2P_MOST_BUILD_ID_BYTES 64u
#define P2P_HEX_DIGITS_PER_BYTE 2u
#define P2P_BUILD_ID_TEXT_BYTES (P2P_MOST_BUILD_ID_BYTES * P2P_HEX_DIGITS_PER_BYTE + 1u)

#define P2P_FNV_OFFSET 0xcbf29ce484222325ull     /* 64-bit FNV-1a: a change detector, not a secret */
#define P2P_FNV_PRIME 0x100000001b3ull

/* ---- the key and the record ---- */

/* Reads a small file whole into text (ended): its length, or -1 if it cannot be read or does not fit. */
static inline long p2p_read_file(const char *path, char *text, size_t capacity)
{
    int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        return -1;
    }
    size_t length = 0;
    for (;;) {
        if (length + 1 >= capacity) {            /* full: the file is longer than any this tool expects */
            close(descriptor);
            return -1;
        }
        ssize_t got = read(descriptor, text + length, capacity - 1 - length);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got < 0) {
            close(descriptor);
            return -1;
        }
        if (got == 0) {
            break;
        }
        length += (size_t)got;
    }
    close(descriptor);
    text[length] = '\0';
    return (long)length;
}

/* The value of the first line of text that begins with prefix, blanks around it removed: 1, or 0 if no such line. */
static inline int p2p_line_value(const char *text, const char *prefix, char *value, size_t capacity)
{
    size_t prefix_length = strlen(prefix);
    for (const char *line = text; *line != '\0';) {
        const char *line_end = strchr(line, '\n');
        size_t line_length = line_end != NULL ? (size_t)(line_end - line) : strlen(line);
        if (line_length >= prefix_length && strncmp(line, prefix, prefix_length) == 0) {
            const char *begin = line + prefix_length, *end = line + line_length;
            while (begin < end && (*begin == ' ' || *begin == '\t')) {
                begin++;
            }
            while (end > begin && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r')) {
                end--;
            }
            if ((size_t)(end - begin) >= capacity) {
                return 0;
            }
            memcpy(value, begin, (size_t)(end - begin));
            value[end - begin] = '\0';
            return 1;
        }
        line += line_length + (line_end != NULL);
    }
    return 0;
}

static inline uint64_t p2p_hash_bytes(uint64_t hash, const void *bytes, size_t count)
{
    const unsigned char *at = (const unsigned char *)bytes;
    for (size_t index = 0; index < count; index++) {
        hash = (hash ^ at[index]) * P2P_FNV_PRIME;
    }
    return hash;
}

/* A named part: its name, its length, its bytes, so that no two different lists of parts hash one stream. */
static inline uint64_t p2p_hash_part(uint64_t hash, const char *name, const void *bytes, size_t count)
{
    uint64_t length = count;
    hash = p2p_hash_bytes(hash, name, strlen(name) + 1);
    hash = p2p_hash_bytes(hash, &length, sizeof(length));
    return p2p_hash_bytes(hash, bytes, count);
}

struct p2p_key {
    uint64_t value;                              /* the hash of every part */
    int complete;                                /* 0: a part could not be read: no record is written or trusted */
    char missing[P2P_NAME_BYTES];                /* the first part that could not be read */
    char driver[P2P_VALUE_BYTES];                /* printed as they are */
    char srcversion[P2P_VALUE_BYTES];
    char build_id[P2P_BUILD_ID_TEXT_BYTES];
    char registry[P2P_VALUE_BYTES];
    int card_count;
    char buses[P2P_MAX_CARDS][P2P_BUS_TEXT_BYTES];
    char vbios[P2P_MAX_CARDS][P2P_VALUE_BYTES];
    uint64_t cards_hash, boot_hash, command_line_hash;      /* printed as hashes only */
};

static inline void p2p_key_missing(struct p2p_key *key, const char *part)
{
    if (key->complete) {
        snprintf(key->missing, sizeof(key->missing), "%s", part);
    }
    key->complete = 0;
}

/* Reads a file and hashes it as a part; 0 if it cannot be read. */
static inline int p2p_key_file(struct p2p_key *key, uint64_t *hash, const char *name, const char *path, char *text,
                               uint64_t *part_hash)
{
    long length = p2p_read_file(path, text, P2P_FILE_BYTES);
    if (length < 0) {
        p2p_key_missing(key, name);
        return 0;
    }
    *hash = p2p_hash_part(*hash, name, text, (size_t)length);
    if (part_hash != NULL) {
        *part_hash = p2p_hash_part(P2P_FNV_OFFSET, name, text, (size_t)length);
    }
    return 1;
}

static inline int p2p_bus_name_is_plain(const char *name)
{
    size_t length = strlen(name);
    if (length == 0 || length >= P2P_BUS_TEXT_BYTES) {
        return 0;
    }
    return strspn(name, "0123456789abcdef:.") == length;
}

static inline int p2p_compare_names(const void *left, const void *right)
{
    return strcmp((const char *)left, (const char *)right);
}

/* The cards the driver holds, sorted by PCI location, with their UUIDs and VBIOS versions. */
static inline void p2p_key_cards(struct p2p_key *key, uint64_t *hash, char *text)
{
    static const char *const directory_path = "/proc/driver/nvidia/gpus";
    char uuid[P2P_VALUE_BYTES], path[P2P_PATH_BYTES];
    DIR *directory = opendir(directory_path);
    if (directory == NULL) {
        p2p_key_missing(key, "cards");
        return;
    }
    for (struct dirent *entry = readdir(directory); entry != NULL; entry = readdir(directory)) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (!p2p_bus_name_is_plain(entry->d_name) || key->card_count == P2P_MAX_CARDS) {
            p2p_key_missing(key, "cards (an unexpected entry, or too many)");
            break;
        }
        memcpy(key->buses[key->card_count++], entry->d_name, strlen(entry->d_name) + 1);   /* its length was checked */
    }
    closedir(directory);
    qsort(key->buses, (size_t)key->card_count, P2P_BUS_TEXT_BYTES, p2p_compare_names);
    key->cards_hash = P2P_FNV_OFFSET;
    for (int card = 0; card < key->card_count; card++) {
        snprintf(path, sizeof(path), "%s/%s/information", directory_path, key->buses[card]);
        if (p2p_read_file(path, text, P2P_FILE_BYTES) < 0 || !p2p_line_value(text, "GPU UUID:", uuid, sizeof(uuid)) ||
            !p2p_line_value(text, "Video BIOS:", key->vbios[card], sizeof(key->vbios[card]))) {
            p2p_key_missing(key, "a card's UUID or VBIOS");
            return;
        }
        key->cards_hash = p2p_hash_part(key->cards_hash, "bus", key->buses[card], strlen(key->buses[card]));
        key->cards_hash = p2p_hash_part(key->cards_hash, "uuid", uuid, strlen(uuid));
        key->cards_hash = p2p_hash_part(key->cards_hash, "vbios", key->vbios[card], strlen(key->vbios[card]));
    }
    if (key->card_count == 0) {
        p2p_key_missing(key, "cards (none)");
    }
    *hash = p2p_hash_part(*hash, "cards", &key->cards_hash, sizeof(key->cards_hash));
}

/* The loaded module's build id: the descriptor of its ELF note, in hex. The note is hashed whole. */
static inline void p2p_key_build_id(struct p2p_key *key, uint64_t *hash, char *text)
{
    uint32_t header[P2P_NOTE_HEADER_WORDS];      /* name size, descriptor size, type */
    long length = p2p_read_file("/sys/module/nvidia/notes/.note.gnu.build-id", text, P2P_FILE_BYTES);
    if (length < (long)sizeof(header)) {
        p2p_key_missing(key, "build id");
        return;
    }
    memcpy(header, text, sizeof(header));
    uint64_t name_end = sizeof(header) + ((uint64_t)header[0] + P2P_NOTE_ALIGNMENT - 1u) / P2P_NOTE_ALIGNMENT *
                                             P2P_NOTE_ALIGNMENT;
    if (header[1] == 0 || header[1] > P2P_MOST_BUILD_ID_BYTES || name_end + header[1] > (uint64_t)length) {
        p2p_key_missing(key, "build id (a note of another form)");
        return;
    }
    for (uint32_t at = 0; at < header[1]; at++) {
        snprintf(key->build_id + P2P_HEX_DIGITS_PER_BYTE * at, P2P_HEX_DIGITS_PER_BYTE + 1u, "%02x",
                 (unsigned int)(unsigned char)text[name_end + at]);
    }
    *hash = p2p_hash_part(*hash, "build id", text, (size_t)length);
}

/* This machine's key as it is now. */
static inline void p2p_machine_key(struct p2p_key *key)
{
    static char text[P2P_FILE_BYTES];
    uint32_t version = P2P_GATE_VERSION;
    memset(key, 0, sizeof(*key));
    key->complete = 1;
    uint64_t hash = p2p_hash_part(P2P_FNV_OFFSET, "gate", &version, sizeof(version));
    if (p2p_key_file(key, &hash, "driver", "/sys/module/nvidia/version", text, NULL)) {
        p2p_line_value(text, "", key->driver, sizeof(key->driver));
    }
    if (p2p_key_file(key, &hash, "srcversion", "/sys/module/nvidia/srcversion", text, NULL)) {
        p2p_line_value(text, "", key->srcversion, sizeof(key->srcversion));
    }
    p2p_key_build_id(key, &hash, text);
    if (p2p_key_file(key, &hash, "parameters", "/proc/driver/nvidia/params", text, NULL)) {
        if (!p2p_line_value(text, "RegistryDwords:", key->registry, sizeof(key->registry))) {
            p2p_key_missing(key, "registry");
        }
        size_t length = strlen(key->registry);
        if (length >= 2 && key->registry[0] == '"' && key->registry[length - 1] == '"') {   /* printed in quotes */
            memmove(key->registry, key->registry + 1, length - 2);
            key->registry[length - 2] = '\0';
        }
    }
    p2p_key_cards(key, &hash, text);
    p2p_key_file(key, &hash, "boot", "/proc/sys/kernel/random/boot_id", text, &key->boot_hash);
    p2p_key_file(key, &hash, "command line", "/proc/cmdline", text, &key->command_line_hash);
    key->value = hash;
}

enum p2p_record_state { P2P_RECORD_TRUSTED, P2P_RECORD_ABSENT, P2P_RECORD_STALE, P2P_RECORD_MALFORMED };
static const char *const p2p_record_state_names[] = {"trusted", "absent", "stale", "malformed"};
static const char *const p2p_verdicts[] = {"passed", "refused", "none_offered", "incomplete"};

struct p2p_record {
    uint64_t key;
    char verdict[P2P_VERDICT_BYTES];
    char pairs[P2P_FILE_BYTES];                  /* " a>b c>d ...", or empty */
};

/* One pair of a record: "<bus>><bus>", each a plain PCI location. */
static inline int p2p_pair_token_is_plain(const char *token, size_t length)
{
    char bus[P2P_BUS_TEXT_BYTES];
    const char *separator = (const char *)memchr(token, '>', length);
    if (separator == NULL) {
        return 0;
    }
    size_t first = (size_t)(separator - token), second = length - first - 1;
    if (first == 0 || first >= sizeof(bus) || second == 0 || second >= sizeof(bus)) {
        return 0;
    }
    memcpy(bus, token, first);
    bus[first] = '\0';
    if (!p2p_bus_name_is_plain(bus)) {
        return 0;
    }
    memcpy(bus, separator + 1, second);
    bus[second] = '\0';
    return p2p_bus_name_is_plain(bus);
}

/* Reads a record and judges it against this machine's key: trusted only if it is exactly the four lines of the
 * form above and its key is this one. A record is never trusted under an incomplete key. */
static inline enum p2p_record_state p2p_record_read(const char *path, const struct p2p_key *key,
                                                    struct p2p_record *record)
{
    enum { FORMAT_LINE, KEY_LINE, VERDICT_LINE, PAIRS_LINE, RECORD_LINES };
    static const char key_prefix[] = "key ", verdict_prefix[] = "verdict ", pairs_prefix[] = "pairs";
    static char text[P2P_FILE_BYTES];
    char first_line[P2P_NAME_BYTES];
    char *lines[RECORD_LINES];
    memset(record, 0, sizeof(*record));
    if (p2p_read_file(path, text, sizeof(text)) < 0) {
        return errno == ENOENT ? P2P_RECORD_ABSENT : P2P_RECORD_MALFORMED;
    }
    char *cursor = text;
    for (int line = 0; line < RECORD_LINES; line++) {
        char *line_end = strchr(cursor, '\n');
        if (line_end == NULL) {
            return P2P_RECORD_MALFORMED;
        }
        *line_end = '\0';
        lines[line] = cursor;
        cursor = line_end + 1;
    }
    snprintf(first_line, sizeof(first_line), "p2p_gate_record %u", P2P_RECORD_FORMAT);
    const char *key_text = lines[KEY_LINE] + strlen(key_prefix);
    const char *verdict = lines[VERDICT_LINE] + strlen(verdict_prefix);
    if (*cursor != '\0' || strcmp(lines[FORMAT_LINE], first_line) != 0 ||
        strncmp(lines[KEY_LINE], key_prefix, strlen(key_prefix)) != 0 ||
        strlen(key_text) != P2P_HEX_DIGITS_OF_KEY || strspn(key_text, "0123456789abcdef") != P2P_HEX_DIGITS_OF_KEY ||
        strncmp(lines[VERDICT_LINE], verdict_prefix, strlen(verdict_prefix)) != 0 ||
        strncmp(lines[PAIRS_LINE], pairs_prefix, strlen(pairs_prefix)) != 0) {
        return P2P_RECORD_MALFORMED;
    }
    int known = 0;
    for (size_t at = 0; at < sizeof(p2p_verdicts) / sizeof(p2p_verdicts[0]); at++) {
        known |= strcmp(verdict, p2p_verdicts[at]) == 0;
    }
    const char *pairs = lines[PAIRS_LINE] + strlen(pairs_prefix);
    for (const char *at = pairs; *at != '\0';) {
        if (*at != ' ') {
            return P2P_RECORD_MALFORMED;
        }
        at++;
        size_t token_length = strcspn(at, " ");
        if (!p2p_pair_token_is_plain(at, token_length)) {
            return P2P_RECORD_MALFORMED;
        }
        at += token_length;
    }
    if (!known || (strcmp(verdict, "passed") != 0 && *pairs != '\0')) {
        return P2P_RECORD_MALFORMED;
    }
    record->key = (uint64_t)strtoull(key_text, NULL, P2P_HEX_BASE);
    snprintf(record->verdict, sizeof(record->verdict), "%s", verdict);
    snprintf(record->pairs, sizeof(record->pairs), "%s", pairs);
    return key->complete && record->key == key->value ? P2P_RECORD_TRUSTED : P2P_RECORD_STALE;
}

/* 1 only if a trusted record passed this ordered pair (the accessor reads and writes the owner's memory). */
static inline int p2p_record_allows(const struct p2p_record *record, const char *accessor_bus, const char *owner_bus)
{
    char token[P2P_PAIR_TEXT_BYTES + 2];
    if (strcmp(record->verdict, "passed") != 0) {
        return 0;
    }
    snprintf(token, sizeof(token), " %s>%s", accessor_bus, owner_bus);
    for (const char *found = strstr(record->pairs, token); found != NULL; found = strstr(found + 1, token)) {
        char after = found[strlen(token)];
        if (after == '\0' || after == ' ') {
            return 1;
        }
    }
    return 0;
}

/* Writes a record whole or not at all: a file beside it, flushed to the disk, then renamed over it. 1 on success. */
static inline int p2p_record_write(const char *path, const struct p2p_key *key, const char *verdict, const char *pairs)
{
    char partial[P2P_PATH_BYTES];
    static char text[P2P_FILE_BYTES];
    if (!key->complete || snprintf(partial, sizeof(partial), "%s.partial", path) >= (int)sizeof(partial)) {
        return 0;
    }
    int length = snprintf(text, sizeof(text), "p2p_gate_record %u\nkey %016" PRIx64 "\nverdict %s\npairs%s\n",
                          P2P_RECORD_FORMAT, key->value, verdict, pairs);
    if (length < 0 || length >= (int)sizeof(text)) {
        return 0;
    }
    int descriptor = open(partial, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0644);
    if (descriptor < 0) {
        return 0;
    }
    size_t written = 0;
    while (written < (size_t)length) {
        ssize_t put = write(descriptor, text + written, (size_t)length - written);
        if (put < 0 && errno == EINTR) {
            continue;
        }
        if (put <= 0) {
            close(descriptor);
            unlink(partial);
            return 0;
        }
        written += (size_t)put;
    }
    if (fsync(descriptor) != 0 || close(descriptor) != 0 || rename(partial, path) != 0) {
        unlink(partial);
        return 0;
    }
    return 1;
}

/* Prints the key's parts that may be shown; the UUIDs, the boot id and the command line only as hashes. */
static inline void p2p_key_print(const struct p2p_key *key)
{
    printf("KEY driver=%s srcversion=%s build_id=%s registry=\"%s\" force_p2p=%s cards=%d cards_hash=%016" PRIx64
           " boot_hash=%016" PRIx64 " command_line_hash=%016" PRIx64 " key=%016" PRIx64 " complete=%s%s%s\n",
           key->driver, key->srcversion, key->build_id, key->registry,
           strstr(key->registry, "ForceP2P") != NULL ? "yes" : "no", key->card_count, key->cards_hash,
           key->boot_hash, key->command_line_hash, key->value, key->complete ? "yes" : "no",
           key->complete ? "" : " missing=", key->complete ? "" : key->missing);
    for (int card = 0; card < key->card_count; card++) {
        printf("KEY_CARD bus=%s vbios=%s\n", key->buses[card], key->vbios[card]);
    }
}

/* ---- the AER counters of a card and of the PCI port above it ---- */

enum p2p_aer_kind { P2P_AER_CORRECTABLE, P2P_AER_NONFATAL, P2P_AER_FATAL, P2P_AER_KINDS };
enum p2p_aer_place { P2P_AER_CARD, P2P_AER_PORT, P2P_AER_PLACES };
static const char *const p2p_aer_files[P2P_AER_KINDS] = {"aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"};
static const char *const p2p_aer_lines[P2P_AER_KINDS] = {"TOTAL_ERR_COR", "TOTAL_ERR_NONFATAL", "TOTAL_ERR_FATAL"};

struct p2p_aer {
    int has_port;                                /* 0: the card sits on a root bus, with no port above it */
    char port[P2P_BUS_TEXT_BYTES];
    unsigned long long totals[P2P_AER_PLACES][P2P_AER_KINDS];
};

/* The three AER totals of one PCI function: 1, or 0 if one cannot be read. */
static inline int p2p_aer_function(const char *function, unsigned long long totals[P2P_AER_KINDS])
{
    static char text[P2P_FILE_BYTES];
    char path[P2P_PATH_BYTES], value[P2P_VALUE_BYTES];
    for (int kind = 0; kind < P2P_AER_KINDS; kind++) {
        char *end = NULL;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", function, p2p_aer_files[kind]);
        if (p2p_read_file(path, text, sizeof(text)) < 0 ||
            !p2p_line_value(text, p2p_aer_lines[kind], value, sizeof(value)) || value[0] < '0' || value[0] > '9') {
            return 0;
        }
        errno = 0;
        totals[kind] = strtoull(value, &end, P2P_DECIMAL_BASE);
        if (errno != 0 || *end != '\0') {
            return 0;
        }
    }
    return 1;
}

/* The AER totals of the card at bus and of the port above it (the parent of its device in sysfs): 1, or 0 if one of
 * them cannot be read. */
static inline int p2p_aer_read(const char *bus, struct p2p_aer *aer)
{
    char path[P2P_PATH_BYTES], resolved[PATH_MAX];
    memset(aer, 0, sizeof(*aer));
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s", bus);
    if (!p2p_bus_name_is_plain(bus) || !p2p_aer_function(bus, aer->totals[P2P_AER_CARD]) ||
        realpath(path, resolved) == NULL) {
        return 0;
    }
    char *last = strrchr(resolved, '/');
    if (last == NULL) {
        return 0;
    }
    *last = '\0';
    const char *parent = strrchr(resolved, '/') != NULL ? strrchr(resolved, '/') + 1 : resolved;
    aer->has_port = p2p_bus_name_is_plain(parent) && strchr(parent, '.') != NULL;
    if (!aer->has_port) {
        return 1;
    }
    snprintf(aer->port, sizeof(aer->port), "%s", parent);
    return p2p_aer_function(aer->port, aer->totals[P2P_AER_PORT]);
}

#endif
