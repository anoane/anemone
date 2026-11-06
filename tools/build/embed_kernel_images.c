/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* embed_kernel_images.c: writes the C table that puts the kernel modules inside the program (the C form of
 * tools/build/embed_kernel_images.py, for builds without Python).
 *
 *     embed_kernel_images <output .c> <build/kernels/<architecture>/<kernel name>.cubin> ...
 *
 * Every module becomes one array of bytes and one entry of cuda_kernel_images, the table declared in
 * source/backend_cuda_kernel_images.h: architecture is the name of the module's directory, kernel_name its file name
 * without the suffix. Modules are taken in byte order of their paths, as the Python tool sorts them, so both write
 * the same table. The output goes to a temporary file first, so that a failed run leaves no half table.
 *
 * Build tool: not part of the engine. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODULE_SUFFIX ".cubin"
#define BYTES_PER_LINE 24
#define MINIMUM_ARGUMENT_COUNT 2
#define EXIT_STATUS_USAGE 2
#define PARTIAL_SUFFIX ".partial"

static int path_order(const void *first, const void *second)
{
	return strcmp(*(char *const *)first, *(char *const *)second);
}

/* The module's directory name (its architecture) and its file name without the suffix, into fixed buffers. */
static int module_names(const char *path, char *architecture,
			size_t architecture_size, char *kernel,
			size_t kernel_size)
{
	const char *file = strrchr(path, '/');
	if (file == NULL || file == path)
		return -1;
	const char *directory_end = file++;
	const char *directory = directory_end;
	while (directory > path && directory[-1] != '/')
		directory--;
	const size_t directory_length = (size_t)(directory_end - directory);
	const size_t file_length = strlen(file),
		     suffix_length = strlen(MODULE_SUFFIX);
	if (directory_length == 0 || directory_length >= architecture_size ||
	    file_length <= suffix_length ||
	    strcmp(file + file_length - suffix_length, MODULE_SUFFIX) != 0 ||
	    file_length - suffix_length >= kernel_size)
		return -1;
	memcpy(architecture, directory, directory_length);
	architecture[directory_length] = '\0';
	memcpy(kernel, file, file_length - suffix_length);
	kernel[file_length - suffix_length] = '\0';
	return 0;
}

static int module_write(FILE *output, const char *path, size_t index)
{
	FILE *module = fopen(path, "rb");
	if (module == NULL) {
		fprintf(stderr, "%s: cannot be read\n", path);
		return -1;
	}
	unsigned char bytes[BYTES_PER_LINE];
	size_t total = 0, count;
	fprintf(output, "static const unsigned char kernel_image_%zu[] = {\n",
		index);
	while ((count = fread(bytes, 1, sizeof(bytes), module)) > 0) {
		fputs("    ", output);
		for (size_t byte = 0; byte < count; byte++)
			fprintf(output, byte + 1 < count ? "%u, " : "%u,\n",
				(unsigned)bytes[byte]);
		total += count;
	}
	const int failed = ferror(module);
	fclose(module);
	if (failed || total == 0) {
		fprintf(stderr, "%s: %s\n", path,
			failed ? "a read error" : "an empty module");
		return -1;
	}
	fputs("};\n", output);
	return 0;
}

int main(int argument_count, char **arguments)
{
	if (argument_count < MINIMUM_ARGUMENT_COUNT) {
		fprintf(stderr,
			"embed_kernel_images <output .c> <architecture>/<kernel name>" MODULE_SUFFIX
			" ...\n");
		return EXIT_STATUS_USAGE;
	}
	const char *const output_path = arguments[1];
	char **const modules = arguments + 2;
	const size_t module_count = (size_t)(argument_count - 2);
	qsort(modules, module_count, sizeof(modules[0]), path_order);
	const size_t partial_size =
		strlen(output_path) + sizeof(PARTIAL_SUFFIX);
	char *const partial_path = malloc(partial_size);
	char architecture[256], kernel[256];
	if (partial_path == NULL)
		return 1;
	snprintf(partial_path, partial_size, "%s%s", output_path,
		 PARTIAL_SUFFIX);
	FILE *output = fopen(partial_path, "w");
	if (output == NULL) {
		fprintf(stderr, "%s: cannot be written\n", partial_path);
		free(partial_path);
		return 1;
	}
	int status = 0;
	fputs("/* SPDX-License-Identifier: AGPL-3.0-or-later */\n"
	      "/* Written by tools/build/embed_kernel_images.c: a build product, never edited. */\n"
	      "#include \"backend_cuda_kernel_images.h\"\n\n",
	      output);
	for (size_t index = 0; index < module_count && status == 0; index++) {
		if (module_names(modules[index], architecture,
				 sizeof(architecture), kernel,
				 sizeof(kernel)) != 0) {
			fprintf(stderr,
				"%s: not <architecture>/<kernel name>" MODULE_SUFFIX
				"\n",
				modules[index]);
			status = 1;
		} else if (module_write(output, modules[index], index) != 0) {
			status = 1;
		}
	}
	/* A table needs one entry to be valid C: with no kernel the count is 0 and the entry is never read. */
	fputs("\nconst struct cuda_kernel_image cuda_kernel_images[] = {\n",
	      output);
	for (size_t index = 0; index < module_count && status == 0; index++) {
		if (module_names(modules[index], architecture,
				 sizeof(architecture), kernel,
				 sizeof(kernel)) == 0)
			fprintf(output,
				"    {\"%s\", \"%s\", kernel_image_%zu, sizeof(kernel_image_%zu)},\n",
				architecture, kernel, index, index);
	}
	if (module_count == 0)
		fputs("    {\"\", \"\", 0, 0},\n", output);
	fprintf(output, "};\nconst uint32_t cuda_kernel_image_count = %zu;\n",
		module_count);
	if (fclose(output) != 0)
		status = 1;
	if (status == 0 && rename(partial_path, output_path) != 0) {
		fprintf(stderr, "%s: cannot be renamed\n", partial_path);
		status = 1;
	}
	if (status != 0)
		remove(partial_path);
	free(partial_path);
	return status;
}
