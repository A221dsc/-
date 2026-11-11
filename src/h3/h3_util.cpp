#include "h3/h3_util.hpp"

#include <sstream>
#include <stdexcept>

namespace spatial::h3 {

std::string h3_cell_to_string(H3Index cell) {
    char buffer[17] = {};
    if (H3_EXPORT(h3ToString)(cell, buffer, sizeof(buffer)) != E_SUCCESS) {
        return "<invalid-h3>";
    }
    return std::string(buffer);
}

void h3_check(H3Error error, const char* operation) {
    if (error == E_SUCCESS) return;
    std::ostringstream out;
    out << operation << " failed with H3 error " << static_cast<int>(error);
    throw std::runtime_error(out.str());
}

}  // namespace spatial::h3
