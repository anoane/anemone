/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* --weights, --model_dir and --exl3_map through the real parser, without a model or a card: the folder's fixed names (a
 * link followed to a regular file, a missing name, a directory, a dangling link, a too long folder, a swapped file),
 * the EXL3 map by its name in the folder and each refusal of one, the family against the placement and the pack
 * argument, the folder against positional files, and serve's copies before its load (turbo_load_copies, over a
 * stand-in search and storage): the search beside two given copies (on by default) with the one line naming each copy
 * given or found, --turbo_load=0 keeping exactly the given copies and naming them, one given copy printing as before,
 * and a tuned load's copies (the profile's) neither searched beside nor printed. Each parse runs in a fresh child, so
 * every case starts from commands.c's defaults. */
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include "anemone/storage_exl3.h"
#include "commands.h"

#define TEST_PATH_BYTES 512u
#define TEST_WORDS 8u
#define TEST_LOG_BYTES 4096u
#define TEST_WIDER_COPIES 2u /* the wider storage's copies */
#define TEST_TUNED_COPIES 3u /* a tuned load's copies: the profile's list */
#define TEST_REPLICA "replica.anemone" /* the stand-in search's one replica */
#define FIFO_OPEN_SECONDS \
	5u /* a blocking open of the FIFO would wait for a writer past this */

static char root[TEST_PATH_BYTES], folder[TEST_PATH_BYTES],
	elsewhere[TEST_PATH_BYTES], broken[TEST_PATH_BYTES],
	linked[TEST_PATH_BYTES]; /* model.exl3map linked to a target without the suffix */
static char tables_real[PATH_MAX], pack_real[PATH_MAX], map_real[PATH_MAX],
	linked_real[PATH_MAX]; /* folder/other.exl3map; linked/model.exl3map's target */
static unsigned cases_run;

enum test_check {
	CHECK_NONE,
	CHECK_FOLDER_GENERATE,
	CHECK_FOLDER_SERVE,
	CHECK_FOLDER_LOAD,
	CHECK_EXL3_DEFAULT,
	CHECK_RAW,
	CHECK_TP4_PRO,
	CHECK_LOAD_EXL3,
	CHECK_FOLDER_MAP, /* --exl3_map=other: folder/other.exl3map */
	CHECK_LINKED_MAP, /* the default name: linked/model.exl3map */
	CHECK_TURBO_BESIDE, /* two given copies: the search runs beside them */
	CHECK_TURBO_OFF, /* two given copies with --turbo_load=0: no search */
	CHECK_TURBO_ALONE, /* one given copy: the search, no copies line */
	CHECK_TURBO_TUNED /* a tuned load's copies: no search, no line */
};

struct source_case {
	const char *words[TEST_WORDS];
	const char *message; /* in the child's stderr, or NULL */
	enum test_check check;
	bool parse_expected;
};

static bool joined(char *out, const char *directory, const char *name)
{
	const int written =
		snprintf(out, TEST_PATH_BYTES, "%s/%s", directory, name);
	return written > 0 && (size_t)written < TEST_PATH_BYTES;
}

static bool file_write(const char *directory, const char *name)
{
	char path[TEST_PATH_BYTES];
	FILE *const file = joined(path, directory, name) ? fopen(path, "wb") :
							   NULL;
	return file != NULL && fputs("x", file) != EOF && fclose(file) == 0;
}

/* root/model: tokenizer.bin and a link original.anemone -> root/elsewhere/original.anemone; root/broken: a folder
 * named tokenizer.bin and a dangling link original.anemone. */
static bool fixture(void)
{
	const char *const base = getenv("TMPDIR") != NULL ? getenv("TMPDIR") :
							    "/tmp";
	char target[TEST_PATH_BYTES], link[TEST_PATH_BYTES],
		missing[TEST_PATH_BYTES];
	const int written = snprintf(root, sizeof(root),
				     "%s/model_source_test.XXXXXX", base);
	return written > 0 && (size_t)written < sizeof(root) &&
	       mkdtemp(root) != NULL && joined(folder, root, "model") &&
	       joined(elsewhere, root, "elsewhere") &&
	       joined(broken, root, "broken") && mkdir(folder, S_IRWXU) == 0 &&
	       mkdir(elsewhere, S_IRWXU) == 0 && mkdir(broken, S_IRWXU) == 0 &&
	       file_write(folder, "tokenizer.bin") &&
	       file_write(elsewhere, "original.anemone") &&
	       joined(target, elsewhere, "original.anemone") &&
	       joined(link, folder, "original.anemone") &&
	       symlink(target, link) == 0 &&
	       joined(link, broken, "tokenizer.bin") &&
	       mkdir(link, S_IRWXU) == 0 && joined(missing, root, "none") &&
	       joined(link, broken, "original.anemone") &&
	       symlink(missing, link) == 0 &&
	       joined(link, folder, "tokenizer.bin") &&
	       realpath(link, tables_real) != NULL &&
	       realpath(target, pack_real) != NULL &&
	       joined(linked, root, "linked") && mkdir(linked, S_IRWXU) == 0 &&
	       file_write(elsewhere, "libertai.map") &&
	       joined(target, elsewhere, "libertai.map") &&
	       joined(link, linked, "model.exl3map") &&
	       symlink(target, link) == 0 &&
	       realpath(target, linked_real) != NULL &&
	       file_write(folder, "other.exl3map") &&
	       joined(target, folder, "other.exl3map") &&
	       realpath(target, map_real) != NULL;
}

static void fixture_remove(void)
{
	static const char *const files[] = {
		"model/tokenizer.bin",	      "model/original.anemone",
		"model/other.exl3map",	      "linked/model.exl3map",
		"elsewhere/original.anemone", "elsewhere/libertai.map",
		"broken/original.anemone"
	};
	char path[TEST_PATH_BYTES];
	for (size_t index = 0; index < sizeof(files) / sizeof(files[0]);
	     index++)
		if (joined(path, root, files[index]))
			(void)unlink(path);
	if (joined(path, broken, "tokenizer.bin"))
		(void)rmdir(path);
	(void)rmdir(folder);
	(void)rmdir(linked);
	(void)rmdir(elsewhere);
	(void)rmdir(broken);
	(void)rmdir(root);
}

static bool resolver_cases(void)
{
	struct model_file_record record = { 0 };
	struct stat metadata;
	char too_long[MODEL_FOLDER_PATH_BYTES + 1u];
	memset(too_long, 'a', MODEL_FOLDER_PATH_BYTES);
	too_long[MODEL_FOLDER_PATH_BYTES] = '\0';
	char not_folder[TEST_PATH_BYTES];
	bool passed =
		model_folder_resolve(folder,
				     model_file_names[MODEL_FILE_TABLES],
				     &record) == STATUS_OK &&
		strcmp(record.path, tables_real) == 0 &&
		stat(tables_real, &metadata) == 0 &&
		model_file_is(&record, metadata.st_dev, metadata.st_ino) &&
		!model_file_is(&record, metadata.st_dev, metadata.st_ino + 1u);
	/* a link is followed to its real path on another folder */
	passed = passed &&
		 model_folder_resolve(folder,
				      model_file_names[MODEL_FILE_ORIGINAL],
				      &record) == STATUS_OK &&
		 strcmp(record.path, pack_real) == 0;
	passed = passed &&
		 model_folder_resolve(folder, model_file_names[MODEL_FILE_EXL3],
				      &record) == STATUS_INPUT_OUTPUT_ERROR &&
		 model_folder_resolve(broken,
				      model_file_names[MODEL_FILE_TABLES],
				      &record) == STATUS_INPUT_OUTPUT_ERROR &&
		 model_folder_resolve(broken,
				      model_file_names[MODEL_FILE_ORIGINAL],
				      &record) == STATUS_INPUT_OUTPUT_ERROR &&
		 joined(not_folder, folder, "tokenizer.bin") &&
		 model_folder_resolve(not_folder,
				      model_file_names[MODEL_FILE_TABLES],
				      &record) == STATUS_INVALID_ARGUMENT &&
		 model_folder_resolve(too_long,
				      model_file_names[MODEL_FILE_TABLES],
				      &record) == STATUS_INVALID_ARGUMENT &&
		 model_folder_resolve("", model_file_names[MODEL_FILE_TABLES],
				      &record) == STATUS_INVALID_ARGUMENT;
	/* a name that is no file name in the folder: empty, or a path (here one to the folder's own tables) */
	passed = passed &&
		 model_folder_resolve(folder, "", &record) ==
			 STATUS_INVALID_ARGUMENT &&
		 model_folder_resolve(folder, "../model/tokenizer.bin",
				      &record) == STATUS_INVALID_ARGUMENT;
	return passed;
}

/* The folder's files as the run sees them: the reads check device and inode, and a swapped file refuses. */
static bool open_cases(void)
{
	struct model_file_record record = { 0 };
	uint64_t byte_count = 0;
	bool passed = model_folder_resolve(folder,
					   model_file_names[MODEL_FILE_TABLES],
					   &record) == STATUS_OK;
	char *bytes = passed ? file_read(record.path, &record, &byte_count) :
			       NULL;
	passed = passed && bytes != NULL && byte_count == 1u;
	free(bytes);
	record.inode++;
	bytes = passed ? file_read(record.path, &record, &byte_count) : NULL;
	const bool refused = bytes == NULL;
	free(bytes);
	return passed && refused && byte_count == 0;
}

static bool turbo_copies_checked(enum test_check check);

static bool checked(enum test_check check)
{
	switch (check) {
	case CHECK_NONE:
		return true;
	case CHECK_FOLDER_GENERATE:
	case CHECK_FOLDER_SERVE:
	case CHECK_FOLDER_LOAD:
		if (model_folder_files(is_generating) != STATUS_OK ||
		    copy_count != 1u || strcmp(copy_paths[0], pack_real) != 0 ||
		    model_pack != &model_files[MODEL_FILE_ORIGINAL])
			return false;
		if (check == CHECK_FOLDER_LOAD)
			return front_end_paths[TABLES_FILE] == NULL;
		return strcmp(front_end_paths[TABLES_FILE], tables_real) == 0 &&
		       (check == CHECK_FOLDER_SERVE ?
				front_end_paths[MESSAGES_FILE] == NULL :
				strcmp(front_end_paths[MESSAGES_FILE],
				       "messages") == 0);
	case CHECK_EXL3_DEFAULT:
		return execution_placement == EXECUTION_TP4_EXL3 &&
		       weights_family == WEIGHTS_EXL3;
	case CHECK_RAW:
		return execution_placement == EXECUTION_PP5 &&
		       weights_family == WEIGHTS_UNSET && model_dir == NULL &&
		       copy_count == 1u &&
		       strcmp(copy_paths[0], "pack.anemone") == 0;
	case CHECK_TP4_PRO:
		return execution_placement == EXECUTION_TP4_PRO &&
		       weights_family == WEIGHTS_ORIGINAL;
	case CHECK_TURBO_BESIDE:
	case CHECK_TURBO_OFF:
	case CHECK_TURBO_ALONE:
	case CHECK_TURBO_TUNED:
		return turbo_copies_checked(check);
	case CHECK_LOAD_EXL3: /* load keeps its own placement and only picks the family's pack */
		return execution_placement == EXECUTION_PP5 &&
		       weights_family == WEIGHTS_EXL3;
	case CHECK_FOLDER_MAP:
	case CHECK_LINKED_MAP:
		return model_folder_files(is_generating) == STATUS_OK &&
		       copy_count == 1u &&
		       model_pack == &model_files[MODEL_FILE_EXL3] &&
		       strcmp(copy_paths[0], check == CHECK_FOLDER_MAP ?
						     map_real :
						     linked_real) == 0;
	}
	return false;
}

static bool child_case(const struct source_case *test)
{
	char program[] = "anemone";
	char *arguments[TEST_WORDS + 1u] = { program };
	char *copies[TEST_WORDS] = { 0 };
	int count = 1;
	bool passed = true;
	for (size_t index = 0; index < TEST_WORDS && test->words[index] != NULL;
	     index++) {
		copies[index] = strdup(test->words[index]);
		passed = passed && copies[index] != NULL;
		arguments[count++] = copies[index];
	}
	FILE *const diagnostic = tmpfile();
	const int saved = dup(STDERR_FILENO);
	passed = passed && diagnostic != NULL && saved >= 0 &&
		 dup2(fileno(diagnostic), STDERR_FILENO) == STDERR_FILENO;
	const bool parsed = passed && parse_arguments(count, arguments);
	passed = passed && parsed == test->parse_expected &&
		 (!parsed || checked(test->check));
	(void)fflush(stderr);
	passed = saved >= 0 && dup2(saved, STDERR_FILENO) == STDERR_FILENO &&
		 passed;
	char text[TEST_LOG_BYTES] = "";
	if (diagnostic != NULL) {
		passed = fseek(diagnostic, 0L, SEEK_SET) == 0 && passed;
		clearerr(diagnostic);
		text[fread(text, 1u, sizeof(text) - 1u, diagnostic)] = '\0';
		(void)fclose(diagnostic);
	}
	passed = passed &&
		 (test->message == NULL || strstr(text, test->message) != NULL);
	for (size_t index = 0; index < TEST_WORDS; index++)
		free(copies[index]);
	return passed;
}

static bool parse_cases(void)
{
	char option[TEST_PATH_BYTES + sizeof("--model_dir=")],
		linked_option[TEST_PATH_BYTES + sizeof("--model_dir=")],
		long_map[sizeof("--exl3_map=") + MODEL_FOLDER_NAME_BYTES];
	const int written =
		snprintf(option, sizeof(option), "--model_dir=%s", folder);
	const int linked_written = snprintf(
		linked_option, sizeof(linked_option), "--model_dir=%s", linked);
	if (written < 0 || (size_t)written >= sizeof(option) ||
	    linked_written < 0 ||
	    (size_t)linked_written >= sizeof(linked_option))
		return false;
	/* a name whose file name, with the suffix, would be longer than NAME_MAX */
	memset(long_map, 'a', sizeof(long_map) - 1u);
	memcpy(long_map, "--exl3_map=", strlen("--exl3_map="));
	long_map[sizeof(long_map) - 1u] = '\0';
	const struct source_case tests[] = {
		{ { "generate", option, "messages" },
		  NULL,
		  CHECK_FOLDER_GENERATE,
		  true },
		{ { "serve", option }, NULL, CHECK_FOLDER_SERVE, true },
		{ { "load", option }, NULL, CHECK_FOLDER_LOAD, true },
		{ { "serve", option, "extra" },
		  "--model_dir replaces the tables and pack arguments",
		  CHECK_NONE,
		  false },
		/* an empty folder: with the guard gone, serve would parse with no positional argument */
		{ { "serve", "--model_dir=" }, NULL, CHECK_NONE, false },
		/* generate's messages stay positional under --model_dir */
		{ { "generate", option }, NULL, CHECK_NONE, false },
		{ { "load", "--weights=exl3", "pack.exl3map" },
		  NULL,
		  CHECK_LOAD_EXL3,
		  true },
		/* an explicit PP5 is load's own placement for either family */
		{ { "load", "--weights=exl3", "--placement=pp5",
		    "pack.exl3map" },
		  NULL,
		  CHECK_LOAD_EXL3,
		  true },
		{ { "serve", "--weights=exl3", "tables", "pack.exl3map" },
		  "weights exl3 -> GPU split TP4_EXL3 (default for the family)",
		  CHECK_EXL3_DEFAULT,
		  true },
		{ { "serve", "--weights=exl3", "--placement=pp5", "tables",
		    "pack.exl3map" },
		  "--weights=exl3 conflicts with --placement=pp5",
		  CHECK_NONE,
		  false },
		{ { "serve", "--weights=original", "--placement=tp4_exl3",
		    "tables", "pack.anemone" },
		  "--weights=original conflicts with --placement=tp4_exl3",
		  CHECK_NONE,
		  false },
		{ { "serve", "--weights=exl3", "--placement=tp4_exl3", "tables",
		    "pack.anemone" },
		  "--weights=exl3 needs an .exl3map pack",
		  CHECK_NONE,
		  false },
		{ { "serve", "--weights=original", "tables", "pack.exl3map" },
		  "--weights=original needs an original pack",
		  CHECK_NONE,
		  false },
		/* exact case: a case-insensitive match would parse this one */
		{ { "serve", "--weights=EXL3", "tables", "pack.exl3map" },
		  NULL,
		  CHECK_NONE,
		  false },
		{ { "serve", "--weights=exl3", "--weights=exl3", "tables",
		    "pack.exl3map" },
		  NULL,
		  CHECK_NONE,
		  false },
		{ { "generate", "--weights=original", "--placement=tp4_pro",
		    "messages", "tables", "pack.anemone" },
		  NULL,
		  CHECK_TP4_PRO,
		  true },
		/* without either flag the parse is the old one */
		{ { "serve", "tables", "pack.anemone" }, NULL, CHECK_RAW, true },
		{ { "serve", "tables", "pack.anemone", "copy.anemone" },
		  NULL,
		  CHECK_TURBO_BESIDE,
		  true },
		{ { "serve", "--turbo_load=0", "tables", "pack.anemone",
		    "copy.anemone" },
		  NULL,
		  CHECK_TURBO_OFF,
		  true },
		{ { "serve", "tables", "pack.anemone" },
		  NULL,
		  CHECK_TURBO_ALONE,
		  true },
		{ { "serve", "tables", "pack.anemone" },
		  NULL,
		  CHECK_TURBO_TUNED,
		  true },
		/* --exl3_map: the folder's EXL3 map by its name, model.exl3map by default (here a link) */
		{ { "serve", option, "--weights=exl3", "--exl3_map=other" },
		  NULL,
		  CHECK_FOLDER_MAP,
		  true },
		{ { "load", option, "--weights=exl3", "--exl3_map=other" },
		  NULL,
		  CHECK_FOLDER_MAP,
		  true },
		/* linked holds the map alone: load reads no tables */
		{ { "load", linked_option, "--weights=exl3" },
		  NULL,
		  CHECK_LINKED_MAP,
		  true },
		{ { "serve", option, "--weights=exl3", "--exl3_map=" },
		  NULL,
		  CHECK_NONE,
		  false },
		{ { "serve", option, "--weights=exl3", "--exl3_map=../other" },
		  "--exl3_map=../other: the name of a map in the folder, without '/'",
		  CHECK_NONE,
		  false },
		{ { "serve", option, "--weights=exl3", long_map },
		  "the name of a map in the folder, without '/', at most 247 bytes",
		  CHECK_NONE,
		  false },
		{ { "serve", option, "--weights=exl3", "--exl3_map=other",
		    "--exl3_map=other" },
		  NULL,
		  CHECK_NONE,
		  false },
		{ { "serve", "--weights=exl3", "--exl3_map=other", "tables",
		    "pack.exl3map" },
		  "--exl3_map=other names the EXL3 map in --model_dir's folder",
		  CHECK_NONE,
		  false },
		{ { "serve", option, "--exl3_map=other" },
		  "--exl3_map=other names the EXL3 map in --model_dir's folder",
		  CHECK_NONE,
		  false },
		{ { "serve", option, "--weights=original", "--exl3_map=other" },
		  "--exl3_map=other names the EXL3 map in --model_dir's folder",
		  CHECK_NONE,
		  false }
	};
	bool passed = true;
	for (size_t index = 0; index < sizeof(tests) / sizeof(tests[0]);
	     index++) {
		cases_run++;
		const pid_t child = fork();
		if (child == 0)
			_exit(child_case(&tests[index]) ? 0 : 1);
		int status = 0;
		const bool accepted =
			child > 0 && waitpid(child, &status, 0) == child &&
			WIFEXITED(status) && WEXITSTATUS(status) == 0;
		if (!accepted)
			(void)fprintf(stderr,
				      "failed model source case %zu: %s %s\n",
				      index, tests[index].words[0],
				      tests[index].words[1] != NULL ?
					      tests[index].words[1] :
					      "");
		passed = accepted && passed;
	}
	return passed;
}

/* The folder's pack as a reader opened it: another device or inode than the resolved one refuses. */
MUST_CHECK static result_status
other_describe(const struct storage_reader *opened,
	       struct storage_description *description_out)
{
	(void)opened;
	*description_out =
		(struct storage_description){ .byte_count = 1u,
					      .copy_count = 1u,
					      .block_alignment_bytes = 1u,
					      .first_device = 1u,
					      .first_inode = 1u };
	return STATUS_OK;
}

static bool swapped_pack_case(void)
{
	static const struct storage_operations_record other = {
		.name = "other", .reader_describe = other_describe
	};
	static const struct pack_header header = { 0 };
	cases_run++;
	const pid_t child = fork();
	if (child == 0) {
		model_dir = folder;
		_exit(model_folder_files(false) == STATUS_OK &&
				      model_pack_opened(&other, NULL,
							&header) ==
					      STATUS_INPUT_OUTPUT_ERROR ?
			      0 :
			      1);
	}
	int status = 0;
	return child > 0 && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* The folder's fixed name sets the pack's reader: model.exl3map linked to a target without the .exl3map suffix is
 * still read as an EXL3 map, and original.anemone as an original pack. */
static bool reader_choice_case(void)
{
	cases_run++;
	const pid_t child = fork();
	if (child == 0) {
		model_dir = linked;
		weights_family = WEIGHTS_EXL3;
		const bool exl3 = model_folder_files(false) == STATUS_OK &&
				  pack_storage() == &storage_exl3 &&
				  storage_for_path(copy_paths[0]) !=
					  &storage_exl3;
		model_dir = folder;
		weights_family = WEIGHTS_ORIGINAL;
		_exit(exl3 && model_folder_files(false) == STATUS_OK &&
				      pack_storage() == &storage_direct ?
			      0 :
			      1);
	}
	int status = 0;
	return child > 0 && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* The turbo reader over the pack and its replicas reads the resolved pack first: its device and inode are checked. The
 * fixture names its one replica (replicas_add's search reads the mount points; the fixture never opens a pack). */
static uint64_t test_opened_device, test_opened_inode;
static char test_replica_paths[1][REPLICA_PATH_CAPACITY];
MUST_CHECK static result_status wider_open(const char *const *paths,
					   uint32_t count,
					   struct storage_reader **reader_out)
{
	(void)paths;
	(void)count;
	*reader_out = (struct storage_reader *)&test_opened_inode;
	return STATUS_OK;
}
static void wider_close(struct storage_reader *opened)
{
	(void)opened;
}
MUST_CHECK static result_status
wider_describe(const struct storage_reader *opened,
	       struct storage_description *description_out)
{
	(void)opened;
	*description_out = (struct storage_description){
		.byte_count = 1u,
		.copy_count = TEST_WIDER_COPIES,
		.block_alignment_bytes = 1u,
		.first_device = test_opened_device,
		.first_inode = test_opened_inode
	};
	return STATUS_OK;
}
/* The stand-in search: the copies it was given (none before it runs) and one replica found. */
static uint32_t searched_count;
static const char *searched_copies[TEST_WIDER_COPIES];
static uint32_t stand_in_search(const char *const *copies, uint32_t count,
				const struct pack_header *header,
				char (*paths)[REPLICA_PATH_CAPACITY],
				uint32_t capacity, bool external)
{
	(void)header;
	(void)external;
	searched_count = count;
	for (uint32_t copy = 0; copy < count && copy < TEST_WIDER_COPIES;
	     copy++)
		searched_copies[copy] = copies[copy];
	if (capacity == 0)
		return 0;
	(void)snprintf(paths[0], REPLICA_PATH_CAPACITY, "%s", TEST_REPLICA);
	return 1u;
}

/* Serve's copies before its load, as run_program calls turbo_load_copies after the parse, over the wider storage and
 * the stand-in search, its stdout read back. */
static bool turbo_copies_checked(enum test_check check)
{
	static const struct storage_operations_record wider = {
		.name = "wider",
		.reader_open = wider_open,
		.reader_close = wider_close,
		.reader_describe = wider_describe
	};
	static const char *const profile_copies[TEST_TUNED_COPIES] = {
		"pack.anemone", "tuned-1.anemone", "tuned-2.anemone"
	};
	static const struct pack_header header = { 0 };
	const bool alone = check == CHECK_TURBO_ALONE ||
			   check == CHECK_TURBO_TUNED;
	const uint32_t given = alone ? 1u : TEST_WIDER_COPIES;
	if (copy_count != given ||
	    turbo_load_searches() != (check != CHECK_TURBO_OFF))
		return false;
	if (check ==
	    CHECK_TURBO_TUNED) { /* as storage_profile_apply's tuned load */
		tuned_load_active = true;
		copy_paths = profile_copies;
		copy_count = TEST_TUNED_COPIES;
	}
	storage_operations = &wider;
	reader = NULL;
	turbo_load_search = stand_in_search;
	char text[TEST_LOG_BYTES] = "";
	FILE *const output = tmpfile();
	(void)fflush(stdout);
	const int saved = dup(STDOUT_FILENO);
	bool passed = output != NULL && saved >= 0 &&
		      dup2(fileno(output), STDOUT_FILENO) == STDOUT_FILENO;
	passed = turbo_load_copies(&header) && passed;
	(void)fflush(stdout);
	passed = saved >= 0 && dup2(saved, STDOUT_FILENO) == STDOUT_FILENO &&
		 passed;
	if (saved >= 0)
		(void)close(saved);
	if (output != NULL) {
		passed = fseek(output, 0L, SEEK_SET) == 0 && passed;
		text[fread(text, 1u, sizeof(text) - 1u, output)] = '\0';
		(void)fclose(output);
	}
	const bool searches = check != CHECK_TURBO_OFF &&
			      check != CHECK_TURBO_TUNED;
	passed = passed && searched_count == (searches ? given : 0u) &&
		 (!searches ||
		  (strcmp(searched_copies[0], "pack.anemone") == 0 &&
		   (alone || strcmp(searched_copies[1], "copy.anemone") == 0)));
	if (check == CHECK_TURBO_BESIDE)
		return passed && copy_count == TEST_WIDER_COPIES + 1u &&
		       strstr(text, "find the replicas") != NULL &&
		       strstr(text,
			      "turbo load: the load reads 3 copies: 0 pack.anemone (given), 1 copy.anemone "
			      "(given), 2 " TEST_REPLICA " (search)\n") != NULL;
	if (check == CHECK_TURBO_OFF)
		return passed && copy_count == TEST_WIDER_COPIES &&
		       strcmp(text,
			      "turbo load: the load reads 2 copies: 0 pack.anemone (given), 1 copy.anemone "
			      "(given)\n") == 0;
	if (check == CHECK_TURBO_ALONE)
		return passed && copy_count == TEST_WIDER_COPIES &&
		       strstr(text, "turbo load: copy 1 " TEST_REPLICA "\n") !=
			       NULL &&
		       strstr(text, "the load reads") == NULL;
	return passed && copy_count == TEST_TUNED_COPIES && text[0] == '\0';
}

static bool turbo_identity_case(void)
{
	cases_run++;
	const pid_t child = fork();
	if (child == 0) {
		static const struct storage_operations_record wider = {
			.name = "wider",
			.reader_open = wider_open,
			.reader_close = wider_close,
			.reader_describe = wider_describe
		};
		model_dir = folder;
		bool passed = model_folder_files(false) == STATUS_OK;
		storage_operations = &wider;
		reader = NULL;
		(void)snprintf(test_replica_paths[0], REPLICA_PATH_CAPACITY,
			       "%s", pack_real);
		const char (*const found_paths)[REPLICA_PATH_CAPACITY] =
			(const char (*)[REPLICA_PATH_CAPACITY])
				test_replica_paths;
		test_opened_device = passed ? model_pack->device : 0;
		test_opened_inode = passed ? model_pack->inode + 1u : 0;
		passed = passed &&
			 replicas_found_add(found_paths, 1u) ==
				 STATUS_INPUT_OUTPUT_ERROR &&
			 copy_count == 1u;
		test_opened_inode = passed ? model_pack->inode : 0;
		passed = passed &&
			 replicas_found_add(found_paths, 1u) == STATUS_OK &&
			 copy_count == TEST_WIDER_COPIES;
		_exit(passed ? 0 : 1);
	}
	int status = 0;
	return child > 0 && waitpid(child, &status, 0) == child &&
	       WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* The tables' open never blocks: a FIFO put at the resolved name after its check refuses at once (the open is
 * non-blocking, then its device and inode differ), where a blocking open would wait for a writer. */
static bool fifo_tables_case(void)
{
	cases_run++;
	struct model_file_record record = { 0 };
	struct stat file;
	int descriptor = -1;
	const bool made =
		joined(record.path, folder, "fifo_tables") &&
		(descriptor = open(record.path, O_WRONLY | O_CREAT | O_EXCL,
				   S_IRUSR | S_IWUSR)) >= 0 &&
		fstat(descriptor, &file) == 0;
	if (descriptor >= 0)
		(void)close(descriptor);
	record.device = made ? (uint64_t)file.st_dev : 0;
	record.inode = made ? (uint64_t)file.st_ino : 0;
	const bool swapped = made && unlink(record.path) == 0 &&
			     mkfifo(record.path, S_IRUSR | S_IWUSR) == 0;
	const pid_t child = swapped ? fork() : -1;
	if (child == 0) {
		(void)alarm(FIFO_OPEN_SECONDS);
		uint64_t byte_count = 1u;
		char *const bytes =
			file_read(record.path, &record, &byte_count);
		_exit(bytes == NULL && byte_count == 0 ? 0 : 1);
	}
	int status = 0;
	const bool passed = child > 0 && waitpid(child, &status, 0) == child &&
			    WIFEXITED(status) && WEXITSTATUS(status) == 0;
	(void)unlink(record.path);
	return passed;
}

int main(void)
{
	bool passed = fixture();
	if (!passed)
		(void)fputs("failed model source fixture\n", stderr);
	passed = passed && resolver_cases();
	passed = passed && open_cases();
	passed = parse_cases() && passed;
	passed = swapped_pack_case() && passed;
	passed = turbo_identity_case() && passed;
	passed = fifo_tables_case() && passed;
	passed = reader_choice_case() && passed;
	fixture_remove();
	(void)printf(
		"model source: %s (%u parser cases, folder resolution, swapped files)\n",
		passed ? "PASS" : "FAIL", cases_run);
	return passed ? 0 : 1;
}
