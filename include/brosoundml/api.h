// Public entry to brosoundml's Bronze JavaScript API (src/api/api.h).
//
// A named guard rather than #pragma once: every sibling library ships this
// same two-line trampoline, and GCC's #pragma once identifies a header by
// content and mtime, not path — two siblings checked out in the same second
// make GCC silently skip the second include.
#ifndef BROSOUNDML_API_H
#define BROSOUNDML_API_H
#include "../../src/api/api.h"
#endif
