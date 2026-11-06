/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image work belongs to the engine worker's prefill. Attach is load-only and never opens a card.
 * Begin borrows immutable request buffers until final TAKEN, resets the slot's typed spans, and
 * validates the complete request before any upload. Advance executes at most one encoder-layer
 * quantum, then returns so other sessions may decode. One image owns shared scratch until it ends.
 * Call only with no pipeline flights; BUSY means another slot owns scratch and changes no state.
 * All driver waits retain its deadline and poison/quarantine rules. No allocation on advancement.
 * Attach receives logical card zero's selected arithmetic class, after the caller's native/emulation choice.
 * An absent tower is a no-op and performs no numerical-policy or device queries.
 */
#ifndef STEP_IMAGES_H
#define STEP_IMAGES_H
#include <stdbool.h>
#include <stdint.h>
#include "anemone/status.h"

struct step_driver;
struct execution_program;
struct model_plan_record;
struct chat_images;
MUST_CHECK result_status step_images_attach(struct step_driver *driver,
					    const struct model_plan_record *plan,
					    uint32_t arithmetic_class);
bool step_images_supported(const struct step_driver *driver);
MUST_CHECK result_status step_images_begin(struct step_driver *driver,
					   uint32_t slot,
					   const struct chat_images *images);
MUST_CHECK result_status step_images_advance(struct step_driver *driver,
					     uint32_t slot, bool *finished_out);
void step_images_cancel(struct step_driver *driver, uint32_t slot);
bool step_image_position(const struct step_driver *driver, uint32_t slot,
			 uint32_t position);
void step_images_free(struct step_driver *driver, bool staging_is_read);
/* Attach parallel image variants before any step or pipeline starts. The count/order, row kinds/capacities,
 * core card areas, segment topology, hand-offs, leases, views and snapshot state must match the base programs.
 * Only launches may differ. Programs are borrowed until destroy; the pointer table is copied. A step with any
 * IMAGE row selects its image variant; text-only steps retain the exact base program and graph table. Refusals
 * leave the driver unchanged. This never adds image arrays to session snapshots: request images are reencoded. */
MUST_CHECK result_status
step_image_programs_attach(struct step_driver *driver,
			   const struct execution_program *const *programs,
			   uint32_t program_count);

#endif
