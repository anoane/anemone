# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/backend/: the card seam, the copy record.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/backend_REQUIRES := common engine/formats/weights
