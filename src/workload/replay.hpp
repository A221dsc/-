#pragma once
#include "core/index.hpp"
#include <filesystem>

namespace spatial {
struct Event {
    char type;
    Id id;
    Point point;
    Query query;
};
struct Trace {
    Box bounds;
    std::vector<std::pair<Id,Point>> initial;
    std::vector<Event> events;
};
Trace read_trace(const std::filesystem::path& path);
}
