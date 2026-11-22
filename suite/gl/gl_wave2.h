// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <stdbool.h>
#include "gl_wave1.h"

bool gb_gl_is_wave2(const char *test);
int gb_gl_run_wave2(const struct gl_ctx_info *ci, const struct gb_opts *o, bool do_verify);
