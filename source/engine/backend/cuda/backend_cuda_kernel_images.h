/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_cuda_kernel_images.h: the kernel modules the build put inside the program.
 *
 * Contract:
 *   - The build compiles every source/kernel_*.cu once per architecture and writes the modules into one table,
 *     build/kernels/kernel_images.c (tools/build/embed_kernel_images.py), which is linked into every program: a
 *     program carries its kernels and reads no kernel from a path.
 *   - An entry is found by its two names. architecture is the build's name for it ("sm_80", "sm_120a");
 *     kernel_name is the kernel file without directory and suffix ("kernel_pack_digest"). bytes is the module as
 *     the driver loads it and stays valid for the life of the program.
 */
#ifndef BACKEND_CUDA_KERNEL_IMAGES_H
#define BACKEND_CUDA_KERNEL_IMAGES_H

#include <stdint.h>

struct cuda_kernel_image {
	const char *architecture;
	const char *kernel_name;
	const unsigned char *bytes;
	uint64_t byte_count;
};

extern const struct cuda_kernel_image cuda_kernel_images[];
extern const uint32_t cuda_kernel_image_count;

#endif
