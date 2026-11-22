/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* cli_autotune.h: the autotune stages the cli, the composition root, gives storage_autotune_cli (cli_run.c): the load
 * stage, then the Engram stage; the tests that run the command take the same table. */
#ifndef CLI_AUTOTUNE_H
#define CLI_AUTOTUNE_H

#include "engine/storage/autotune/storage_autotune.h"

#define CLI_AUTOTUNE_STAGES 2u
extern const struct storage_autotune_stage
	cli_autotune_stages[CLI_AUTOTUNE_STAGES];

#endif
