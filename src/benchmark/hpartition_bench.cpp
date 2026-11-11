// Benchmark for the lightweight homogeneous partition index (one Quad per leaf).
#include "core/clock.hpp"
#include "indexes/homogeneous_partition.hpp"
#include "validation/oracle.hpp"
#include "workload/m3.hpp"
#include <windows.h>
#include <psapi.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spatial {
namespace {

double working_mib() {
    PROCESS_MEMORY_COUNTERS memory{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
        throw std::runtime_error("WorkingSetSize unavailable");
    return memory.WorkingSetSize / 1048576.0;
}

double peak_mib() {
    PROCESS_MEMORY_COUNTERS memory{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
        throw std::runtime_error("PeakWorkingSetSize unavailable");
    return memory.PeakWorkingSetSize / 1048576.0;
}

struct Plan {
    std::vector<std::array<int, 4>> rects;
    std::vector<unsigned> levels;
};

Plan read_leaf_plan(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open partition plan " + path);
    Plan plan;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        std::string kind;
        stream >> kind;
        if (kind != "L") continue;
        std::array<int, 4> rect{};
        unsigned level = 1;
        stream >> rect[0] >> rect[1] >> rect[2] >> rect[3] >> level;
        plan.rects.push_back(rect);
        plan.levels.push_back(level);
    }
    if (plan.rects.empty()) throw std::runtime_error("empty partition plan");
    return plan;
}

}  // namespace

int run_hpartition(const M3Trace& trace, int fine, int quad, const std::string& plan_path,
                   bool validate, bool profile) {
    (void)fine;
    const Plan plan = read_leaf_plan(plan_path);
    const double prebuild_mib = peak_mib();
    const double prebuild_ws = working_mib();
    HomogeneousPartitionIndex index(trace.bounds, quad, profile);
    std::unique_ptr<ArrayOracle> oracle;
    if (validate) oracle = std::make_unique<ArrayOracle>();
    const auto begin = Clock::now();
    index.set_partition(plan.rects, plan.levels);
    Id id = 0;
    for (const Point& point : trace.points) {
        index.set(++id, point);
        if (validate) oracle->set(id, point);
    }
    const double build_ms = (Clock::now() - begin).count() / 1e6;
    const double postbuild_ws = working_mib();
    double query_ms = 0.0, update_ms = 0.0;
    std::uint64_t queries = 0, mismatches = 0, candidates = 0;
    std::vector<std::int64_t> latency;
    latency.reserve(trace.query_count);
    const auto online_begin = Clock::now();
    for (const M3Event& event : trace.events) {
        if (event.type == 'P') continue;
        if (event.type == 'U') {
            const auto start = Clock::now();
            index.set(event.id, event.point);
            update_ms += (Clock::now() - start).count() / 1e6;
            if (validate) oracle->set(event.id, event.point);
        } else if (event.type == 'D') {
            const auto start = Clock::now();
            index.erase(event.id);
            update_ms += (Clock::now() - start).count() / 1e6;
            if (validate) oracle->erase(event.id);
        } else {
            const auto start = Clock::now();
            Answer answer = index.query(event.query);
            const auto elapsed = Clock::now() - start;
            query_ms += elapsed.count() / 1e6;
            latency.push_back(elapsed.count());
            candidates += answer.candidates;
            ++queries;
            if (validate) {
                std::sort(answer.ids.begin(), answer.ids.end());
                if (answer.ids != oracle->query(event.query)) ++mismatches;
            }
        }
    }
    const double online_ms = query_ms + update_ms;
    const double wall_ms = (Clock::now() - online_begin).count() / 1e6;
    std::sort(latency.begin(), latency.end());
    double p50 = 0.0, p95 = 0.0, p99 = 0.0;
    if (!latency.empty()) {
        auto pct = [&](int p) {
            const std::size_t at = (latency.size() * p + 99) / 100;
            return latency.at(at - 1) / 1e3;
        };
        p50 = pct(50);
        p95 = pct(95);
        p99 = pct(99);
    }
    std::cout << std::fixed << std::setprecision(6)
              << "online_ms=" << online_ms
              << " wall_ms=" << wall_ms
              << " query_ms=" << query_ms
              << " update_ms=" << update_ms
              << " switch_ms=0"
              << " build_ms=" << build_ms
              << " peak_mib=" << peak_mib()
              << " prebuild_mib=" << prebuild_mib
              << " index_delta_mib=" << (postbuild_ws - prebuild_ws)
              << " index_nodes=" << index.index_nodes()
              << " candidate_count=" << candidates
              << " p50_us=" << p50 << " p95_us=" << p95 << " p99_us=" << p99
              << " queries=" << queries
              << " leaves=" << plan.rects.size()
              << " route_cells=" << index.route_cells()
              << " touched_leaves=" << index.touched_leaves()
              << " local_queries=" << index.local_queries()
              << " route_ms=" << index.route_ms()
              << " local_ms=" << index.local_ms()
              << " mismatches=" << mismatches << "\n";
    return mismatches == 0 ? 0 : 1;
}
}  // namespace spatial
