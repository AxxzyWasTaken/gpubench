// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <stdbool.h>
#include "gb_common.h"

struct gl_ctx_info {
   bool es, compute, timer;
   int timer_bits;
   void (*emit_identity)(struct gb_json *j, const void *ctx);
   const void *ctx;
};

bool gb_gl_is_wave1(const char *test);
int gb_gl_run_wave1(const struct gl_ctx_info *ci, const struct gb_opts *o, bool do_verify);
