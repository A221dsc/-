// Benchmark for the load-aware adaptive partition index.
//
// Plan file format (text):
//   L x0 y0 x1 y1 level      leaf rectangle in 64x64 sub-cell units
//   M window leaf mode benefit
// Modes are applied at the start of the given window; the shared scheduler
// enforces the 100-step cooldown, at most four rebuilds per window and the
// 6000-entity budget, ranking candidates by the supplied benefit.
//
// STATIC_MODE (1=FINE_GRID, 2=LOCAL_QUAD) forces every leaf to the same local
// index before the initial insert, which is the homogeneous all-partitions-same
// index baseline used by the H0 partition study.
#include "core/clock.hpp"
#include "indexes/partition.hpp"
#include "validation/oracle.hpp"
#include "workload/m3.hpp"
#include <windows.h>
#include <psapi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <string>
#include <vector>

namespace spatial {
namespace {

constexpr int cooldown_steps = 100;
constexpr unsigned max_rebuilds = 4;
constexpr double max_rebuild_entities = 6000.0;

double peak_mib() {
    PROCESS_MEMORY_COUNTERS memory{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
        throw std::runtime_error("PeakWorkingSetSize unavailable");
    return memory.PeakWorkingSetSize / 1048576.0;
}

struct Plan {
    std::vector<std::array<int, 4>> rects;
    std::vector<unsigned> levels;
    std::map<std::pair<int, unsigned>, std::pair<int, double>> modes;
};

Plan read_plan(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open partition plan " + path);
    Plan plan;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream stream(line);
        std::string kind;
        stream >> kind;
        if (kind == "L") {
            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            unsigned level = 1;
            stream >> x0 >> y0 >> x1 >> y1 >> level;
            plan.rects.push_back({x0, y0, x1, y1});
            plan.levels.push_back(level);
        } else if (kind == "M") {
            int window = 0, mode = 0;
            unsigned leaf = 0;
            double benefit = 0.0;
            stream >> window >> leaf >> mode >> benefit;
            plan.modes[{window, leaf}] = {mode, benefit};
        }
    }
    if (plan.rects.empty()) throw std::runtime_error("empty partition plan");
    return plan;
}

struct Outcome {
    double online_ms{}, query_ms{}, update_ms{}, switch_ms{}, build_ms{};
    double peak_mib{};
    std::size_t index_nodes{};
    std::uint64_t queries{}, promotions{}, switches{}, mismatches{}, candidate_count{};
    double p50_us{}, p95_us{}, p99_us{};
};

Mode mode_of(int value) {
    return value == 1 ? Mode::FineGrid : value == 2 ? Mode::LocalQuad : Mode::Scan;
}

Outcome replay(const M3Trace& trace, const Plan& plan, int fine, int quad, bool validate,
               int static_mode) {
    PartitionIndex index(trace.bounds, fine, quad, false, false);
    index.set_partition(plan.rects, plan.levels);
    if (static_mode != 0) {
        const Mode mode = mode_of(static_mode);
        for (unsigned leaf = 0; leaf < plan.rects.size(); ++leaf) index.set_mode(leaf, mode);
    }
    std::unique_ptr<ArrayOracle> oracle;
    if (validate) oracle = std::make_unique<ArrayOracle>();
    Id id = 0;
    Outcome outcome;
    std::vector<std::int64_t> latency;
    latency.reserve(trace.query_count);
    const auto build_begin = Clock::now();
    for (const Point& point : trace.points) {
        index.set(++id, point);
        if (validate) oracle->set(id, point);
    }
    outcome.build_ms = (Clock::now() - build_begin).count() / 1e6;
    index.reset_window();
    std::vector<std::int64_t> last_switch(plan.rects.size(),
                                          std::numeric_limits<std::int64_t>::min() / 2);
    int window = 0;
    auto apply = [&](int step) {
        std::vector<std::tuple<double, unsigned, int>> candidates;
        for (unsigned leaf = 0; leaf < plan.rects.size(); ++leaf) {
            const auto found = plan.modes.find({window, leaf});
            if (found == plan.modes.end()) continue;
            const int target = found->second.first;
            if (mode_of(target) == index.mode(leaf)) continue;
            candidates.emplace_back(found->second.second, leaf, target);
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        unsigned rebuilds = 0;
        double entities = 0.0;
        for (const auto& [benefit, leaf, target] : candidates) {
            const Mode mode = mode_of(target);
            const bool demotion = mode == Mode::Scan;
            const std::int64_t since = step - last_switch[leaf];
            if (since < cooldown_steps) continue;
            if (!demotion && benefit <= 0) continue;
            if (!demotion && rebuilds >= max_rebuilds) continue;
            if (!demotion && entities + index.entity_count(leaf) > max_rebuild_entities) continue;
            const auto begin = Clock::now();
            index.set_mode(leaf, mode);
            outcome.switch_ms += (Clock::now() - begin).count() / 1e6;
            last_switch[leaf] = step;
            ++outcome.switches;
            if (!demotion) {
                ++rebuilds;
                entities += static_cast<double>(index.entity_count(leaf));
                ++outcome.promotions;
            }
        }
    };
    apply(0);
    for (const M3Event& event : trace.events) {
        if (event.type == 'P') {
            index.reset_window();
            ++window;
            apply(event.step);
            continue;
        }
        if (event.type == 'U') {
            const auto begin = Clock::now();
            index.set(event.id, event.point);
            outcome.update_ms += (Clock::now() - begin).count() / 1e6;
            if (validate) oracle->set(event.id, event.point);
        } else if (event.type == 'D') {
            const auto begin = Clock::now();
            index.erase(event.id);
            outcome.update_ms += (Clock::now() - begin).count() / 1e6;
            if (validate) oracle->erase(event.id);
        } else {
            const auto begin = Clock::now();
            Answer answer = index.query(event.query);
            const auto elapsed = Clock::now() - begin;
            outcome.query_ms += elapsed.count() / 1e6;
            latency.push_back(elapsed.count());
            outcome.candidate_count += answer.candidates;
            ++outcome.queries;
            if (validate) {
                std::sort(answer.ids.begin(), answer.ids.end());
                if (answer.ids != oracle->query(event.query)) ++outcome.mismatches;
            }
        }
    }
    outcome.online_ms = outcome.query_ms + outcome.update_ms + outcome.switch_ms;
    outcome.index_nodes = index.index_nodes();
    outcome.peak_mib = peak_mib();
    std::sort(latency.begin(), latency.end());
    if (!latency.empty()) {
        auto pct = [&](int p) {
            const std::size_t at = (latency.size() * p + 99) / 100;
            return latency.at(at - 1) / 1e3;
        };
        outcome.p50_us = pct(50);
        outcome.p95_us = pct(95);
        outcome.p99_us = pct(99);
    }
    return outcome;
}

}  // namespace

int run_partition(const M3Trace& trace, int fine, int quad, const std::string& plan_path,
                  bool validate, int static_mode) {
    const Plan plan = read_plan(plan_path);
    const Outcome outcome = replay(trace, plan, fine, quad, validate, static_mode);
    std::cout << std::fixed << std::setprecision(6)
              << "online_ms=" << outcome.online_ms
              << " query_ms=" << outcome.query_ms
              << " update_ms=" << outcome.update_ms
              << " switch_ms=" << outcome.switch_ms
              << " build_ms=" << outcome.build_ms
              << " peak_mib=" << outcome.peak_mib
              << " index_nodes=" << outcome.index_nodes
              << " candidate_count=" << outcome.candidate_count
              << " p50_us=" << outcome.p50_us << " p95_us=" << outcome.p95_us
              << " p99_us=" << outcome.p99_us
              << " queries=" << outcome.queries
              << " promotions=" << outcome.promotions
              << " switches=" << outcome.switches
              << " leaves=" << plan.rects.size()
              << " mismatches=" << outcome.mismatches << "\n";
    return outcome.mismatches == 0 ? 0 : 1;
}
}  // namespace spatial
