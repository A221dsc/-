#pragma once
#include "core/index.hpp"
#include <cstdint>
#include <filesystem>
#include <vector>

namespace spatial {
// One event of the M3 point-movement replay.
// 'U' = position update, 'R' = rectangle query, 'C' = circle query,
// 'P' = decision-window boundary emitted before the first event of that step.
struct M3Event {
    char type{};
    std::uint32_t step{};
    Id id{};
    Point point{};
    Query query{};
};

struct M3Trace {
    Box bounds{};
    std::vector<Point> points;   // initial positions, object id = index + 1
    std::vector<M3Event> events;
    std::uint64_t update_count{};
    std::uint64_t query_count{};
    std::size_t decision_points{};
    std::size_t bytes{};
};

// Reads the little-endian M3 binary replay. Layout:
//   header  "<8s4d2Q"  magic, world x0 y0 x1 y1, entities, event count
//   points  2 doubles  per entity
//   events  "<c7xq4d"  type, id/step, four payload doubles
// The reader rejects truncated, mis-sized or ill-typed files.
M3Trace read_m3_trace(const std::filesystem::path& path);
}
