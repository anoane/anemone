# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/server/: the HTTP server.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/server_REQUIRES := engine/server/api engine/runtime/scheduler engine/runtime/speculation engine/input/tokenizer \
    engine/input/image models common engine/formats/pack
