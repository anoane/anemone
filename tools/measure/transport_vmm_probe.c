/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Allocation-scoped peer permissions only. No context-wide peer enable, kernels, or engine gate record. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARTICIPANTS 2
#define RESERVATION_BYTES (2u * 1024u * 1024u)

static int checked(CUresult result, const char *phase)
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

int main(void)
{
    CUdevice devices[PARTICIPANTS];
    CUcontext contexts[PARTICIPANTS] = {0};
    CUmemGenericAllocationHandle handle = 0;
    CUdeviceptr address = 0;
    size_t granularity = 0;
    size_t byte_count = 0;
    int mapped = 0;
    int count = 0;
    int success = 0;
    CUmemAllocationProp properties;
    CUmemAccessDesc access;
    memset(&properties, 0, sizeof(properties));
    memset(&access, 0, sizeof(access));
    fprintf(stderr, "phase=initialize\n");
    if (!checked(cuInit(0), "initialize") || !checked(cuDeviceGetCount(&count), "count") || count != PARTICIPANTS) {
        return EXIT_FAILURE;
    }
    for (int index = 0; index < PARTICIPANTS; index++) {
        int supported = 0;
        fprintf(stderr, "phase=retain participant=%d\n", index);
        if (!checked(cuDeviceGet(&devices[index], index), "device") ||
            !checked(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                         devices[index]), "vmm_attribute") || !supported) {
            goto cleanup;
        }
    }
    /* A permission grant needs only the allocation owner's context, not a second primary context. */
    fprintf(stderr, "phase=owner_context\n");
    if (!checked(cuDevicePrimaryCtxRetain(&contexts[0], devices[0]), "retain_owner")) {
        goto cleanup;
    }
    if (!checked(cuCtxSetCurrent(contexts[0]), "current")) {
        goto cleanup;
    }
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = devices[0];
    if (!checked(cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "granularity") || granularity == 0) {
        goto cleanup;
    }
    byte_count = ((RESERVATION_BYTES + granularity - 1) / granularity) * granularity;
    fprintf(stderr, "phase=reserve\n");
    if (!checked(cuMemAddressReserve(&address, byte_count, 0, 0, 0), "reserve") ||
        !checked(cuMemCreate(&handle, byte_count, &properties, 0), "allocate") ||
        !checked(cuMemMap(address, byte_count, 0, handle, 0), "map")) {
        goto cleanup;
    }
    mapped = 1;
    access.location = properties.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!checked(cuMemSetAccess(address, byte_count, &access, 1), "owner_access")) {
        goto cleanup;
    }
    fprintf(stderr, "phase=grant_peer\n");
    access.location.id = devices[1];
    CUresult grant_result = cuMemSetAccess(address, byte_count, &access, 1);
    printf("{\"vmm_peer_grant\":%s,\"result_code\":%d,\"payload_qualified\":false}\n",
           grant_result == CUDA_SUCCESS ? "true" : "false", (int)grant_result);
    fflush(stdout);
    success = grant_result == CUDA_SUCCESS || grant_result == CUDA_ERROR_NOT_SUPPORTED ||
              grant_result == CUDA_ERROR_INVALID_DEVICE;
cleanup:
    if (contexts[0] != NULL && !checked(cuCtxSetCurrent(contexts[0]), "cleanup_context")) {
        success = 0;
    }
    if (mapped && !checked(cuMemUnmap(address, byte_count), "unmap")) {
        success = 0;
    }
    if (handle != 0 && !checked(cuMemRelease(handle), "release")) {
        success = 0;
    }
    if (address != 0 && !checked(cuMemAddressFree(address, byte_count), "free_address")) {
        success = 0;
    }
    for (int index = 0; index < PARTICIPANTS; index++) {
        if (contexts[index] != NULL && !checked(cuDevicePrimaryCtxRelease(devices[index]), "release_context")) {
            success = 0;
        }
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
