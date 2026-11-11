#pragma once

#include <string>

#include "h3api.h"

namespace spatial::h3 {

std::string h3_cell_to_string(H3Index cell);
void h3_check(H3Error error, const char* operation);

}  // namespace spatial::h3
