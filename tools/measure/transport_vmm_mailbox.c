/* SPDX-License-Identifier: AGPL-3.0-or-later */
#define _POSIX_C_SOURCE 200809L
#include "transport_vmm_mailbox.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define TRANSPORT_SOCKET_TIMEOUT_SECONDS 20

static int mapping_checked(CUresult result, const char *phase)
{
    if (result == CUDA_SUCCESS) {
        return 1;
    }
    const char *name = "unknown";
    if (cuGetErrorName(result, &name) != CUDA_SUCCESS) {
        name = "unknown";
    }
    fprintf(stderr, "phase=%s result=%s\n", phase, name);
    return 0;
}

static int mapping_map(struct transport_mapping *mapping, CUdevice device)
{
    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!mapping_checked(cuMemAddressReserve(&mapping->address, mapping->byte_count, 0, 0, 0), "vmm_reserve") ||
        !mapping_checked(cuMemMap(mapping->address, mapping->byte_count, 0, mapping->handle, 0), "vmm_map")) {
        return 0;
    }
    mapping->mapped = 1;
    return mapping_checked(cuMemSetAccess(mapping->address, mapping->byte_count, &access, 1), "vmm_grant");
}

int transport_mapping_create(struct transport_mapping *mapping, CUdevice device, size_t byte_count)
{
    CUmemAllocationProp properties;
    size_t granularity = 0;
    memset(&properties, 0, sizeof(properties));
    if (mapping == NULL || byte_count == 0) {
        return 0;
    }
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    if (!mapping_checked(cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                          "vmm_granularity") || granularity == 0 || byte_count > SIZE_MAX - (granularity - 1)) {
        return 0;
    }
    mapping->byte_count = (byte_count + granularity - 1) / granularity * granularity;
    return mapping_checked(cuMemCreate(&mapping->handle, mapping->byte_count, &properties, 0), "vmm_create") &&
           mapping_map(mapping, device);
}

static int send_descriptor(int socket_descriptor, int allocation_descriptor, size_t byte_count)
{
    union { struct cmsghdr alignment; unsigned char bytes[CMSG_SPACE(sizeof(int))]; } control;
    memset(&control, 0, sizeof(control));
    struct iovec vector = {.iov_base = &byte_count, .iov_len = sizeof(byte_count)};
    struct msghdr message;
    memset(&message, 0, sizeof(message));
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control.bytes;
    message.msg_controllen = sizeof(control.bytes);
    struct cmsghdr *header = CMSG_FIRSTHDR(&message);
    if (header == NULL) {
        return 0;
    }
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(header), &allocation_descriptor, sizeof(allocation_descriptor));
    return sendmsg(socket_descriptor, &message, MSG_NOSIGNAL) == (ssize_t)sizeof(byte_count);
}

static int receive_descriptor(int socket_descriptor, int *allocation_descriptor, size_t *byte_count)
{
    union { struct cmsghdr alignment; unsigned char bytes[CMSG_SPACE(sizeof(int))]; } control;
    memset(&control, 0, sizeof(control));
    struct iovec vector = {.iov_base = byte_count, .iov_len = sizeof(*byte_count)};
    struct msghdr message;
    memset(&message, 0, sizeof(message));
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control.bytes;
    message.msg_controllen = sizeof(control.bytes);
    if (recvmsg(socket_descriptor, &message, 0) != (ssize_t)sizeof(*byte_count) ||
        (message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) != 0) {
        return 0;
    }
    struct cmsghdr *header = CMSG_FIRSTHDR(&message);
    if (header == NULL || header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
        header->cmsg_len != CMSG_LEN(sizeof(int)) || CMSG_NXTHDR(&message, header) != NULL) {
        return 0;
    }
    memcpy(allocation_descriptor, CMSG_DATA(header), sizeof(*allocation_descriptor));
    return *allocation_descriptor >= 0;
}

int transport_mapping_exchange(struct transport_mapping *mapping, const struct transport_mapping *owned,
                               CUdevice device, int socket_descriptor)
{
    const struct timeval timeout = {.tv_sec = TRANSPORT_SOCKET_TIMEOUT_SECONDS, .tv_usec = 0};
    int allocation_descriptor = -1;
    int imported_descriptor = -1;
    int success = 0;
    if (mapping == NULL || owned == NULL || owned->handle == 0 || socket_descriptor < 0 ||
        setsockopt(socket_descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(socket_descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        return 0;
    }
    if (!mapping_checked(cuMemExportToShareableHandle(&allocation_descriptor, owned->handle,
                          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), "vmm_export")) {
        return 0;
    }
    if (!send_descriptor(socket_descriptor, allocation_descriptor, owned->byte_count) ||
        !receive_descriptor(socket_descriptor, &imported_descriptor, &mapping->byte_count) ||
        mapping->byte_count == 0) {
        goto cleanup;
    }
    fprintf(stderr, "phase=vmm_import\n");
    if (!mapping_checked(cuMemImportFromShareableHandle(&mapping->handle, (void *)(intptr_t)imported_descriptor,
                          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), "vmm_import") || !mapping_map(mapping, device)) {
        goto cleanup;
    }
    success = 1;
cleanup:
    if (imported_descriptor >= 0 && close(imported_descriptor) != 0) {
        success = 0;
    }
    if (close(allocation_descriptor) != 0) {
        success = 0;
    }
    return success;
}

int transport_mapping_release(struct transport_mapping *mapping)
{
    if (mapping == NULL) {
        return 0;
    }
    int success = 1;
    if (mapping->mapped && !mapping_checked(cuMemUnmap(mapping->address, mapping->byte_count), "vmm_unmap")) {
        success = 0;
    }
    if (mapping->handle && !mapping_checked(cuMemRelease(mapping->handle), "vmm_release")) {
        success = 0;
    }
    if (mapping->address && !mapping_checked(cuMemAddressFree(mapping->address, mapping->byte_count), "vmm_free")) {
        success = 0;
    }
    return success;
}
