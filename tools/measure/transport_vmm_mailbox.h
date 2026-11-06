/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Qualification only: allocation-scoped VMM sharing, one context per process. Not a production trust record. */
#ifndef TRANSPORT_VMM_MAILBOX_H
#define TRANSPORT_VMM_MAILBOX_H
#include <cuda.h>
#include <stddef.h>

struct transport_mapping {
    CUdeviceptr address;
    CUmemGenericAllocationHandle handle;
    size_t byte_count;
    int mapped;
};
int transport_mapping_create(struct transport_mapping *mapping, CUdevice device, size_t byte_count);
int transport_mapping_exchange(struct transport_mapping *mapping, const struct transport_mapping *owned,
                               CUdevice device, int socket_descriptor);
int transport_mapping_release(struct transport_mapping *mapping);
#endif
