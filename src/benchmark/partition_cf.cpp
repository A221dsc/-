// Counterfactual single-region switch measurement on the PartitionIndex.
//
// Both arms rebuild the same prefix from the start (the trace is deterministic),
// so they reach the identical authoritative state at the target window start:
// all leaves keep their prefix mode, the target leaf then either keeps mode A
// (SCAN) or switches to mode B before the same future events are replayed for a
// fixed horizon. The reported cost is the QPC-measured window cost of the
// horizon only; the prefix is not timed.
#include "core/clock.hpp"
#include "indexes/partition.hpp"
#include "validation/oracle.hpp"
#include "workload/m3.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spatial {
namespace {

struct LeafPlan {
    std::vector<std::array<int, 4>> rects;
    std::vector<unsigned> levels;
};

LeafPlan read_leaf_plan(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open partition plan " + path);
    LeafPlan plan;
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

struct ArmResult {
    double query_ms{}, update_ms{}, switch_ms{}, total_ms{};
    std::size_t entities{};
    std::uint64_t hash{};
    WindowLoad load;
    int windows{};
    std::uint64_t queries{}, mismatches{};
};

ArmResult run_arm(const M3Trace& trace, const LeafPlan& plan, int fine, int quad,
                  int window, unsigned leaf, Mode mode_b, bool switch_first, int horizon,
                  bool verify) {
    PartitionIndex index(trace.bounds, fine, quad, false, false);
    index.set_partition(plan.rects, plan.levels);
    std::unique_ptr<ArrayOracle> oracle;
    if (verify) oracle = std::make_unique<ArrayOracle>();
    Id id = 0;
    for (const Point& point : trace.points) {
        index.set(++id, point);
        if (verify) oracle->set(id, point);
    }
    index.reset_window();
    const std::int64_t target_step = static_cast<std::int64_t>(window) * 50;
    ArmResult result;
    std::size_t cursor = 0;
    // Prefix: replay everything before the target window start.
    while (cursor < trace.events.size()) {
        const M3Event& event = trace.events[cursor];
        if (event.type == 'P' && event.step == target_step) break;
        if (event.type == 'P') {
            index.reset_window();
        } else if (event.type == 'U') {
            index.set(event.id, event.point);
            if (verify) oracle->set(event.id, event.point);
        } else if (event.type == 'D') {
            index.erase(event.id);
            if (verify) oracle->erase(event.id);
        } else {
            Answer answer = index.query(event.query);
            if (verify) {
                std::sort(answer.ids.begin(), answer.ids.end());
                if (answer.ids != oracle->query(event.query)) ++result.mismatches;
            }
        }
        ++cursor;
    }
    if (window > 0 && cursor < trace.events.size()) ++cursor;  // consume the P at the window start
    result.entities = index.entity_count(leaf);
    result.load = index.window_load(leaf);
    for (std::size_t i = 0; i < trace.points.size(); ++i) {
        result.hash = result.hash * 1000003ULL + static_cast<std::uint64_t>(i + 1);
    }
    index.reset_window();
    if (switch_first) {
        const auto begin = Clock::now();
        index.set_mode(leaf, mode_b);
        result.switch_ms += (Clock::now() - begin).count() / 1e6;
    }
    const std::int64_t end_step = target_step + static_cast<std::int64_t>(horizon) * 50;
    result.windows = 1;  // the target window itself
    while (cursor < trace.events.size()) {
        const M3Event& event = trace.events[cursor];
        if (event.type == 'P') {
            if (event.step >= end_step) break;
            index.reset_window();
            ++result.windows;
            ++cursor;
            continue;
        }
        if (event.type == 'U') {
            const auto begin = Clock::now();
            index.set(event.id, event.point);
            result.update_ms += (Clock::now() - begin).count() / 1e6;
            if (verify) oracle->set(event.id, event.point);
        } else if (event.type == 'D') {
            const auto begin = Clock::now();
            index.erase(event.id);
            result.update_ms += (Clock::now() - begin).count() / 1e6;
            if (verify) oracle->erase(event.id);
        } else {
            const auto begin = Clock::now();
            Answer answer = index.query(event.query);
            result.query_ms += (Clock::now() - begin).count() / 1e6;
            ++result.queries;
            if (verify) {
                std::sort(answer.ids.begin(), answer.ids.end());
                if (answer.ids != oracle->query(event.query)) ++result.mismatches;
            }
        }
        ++cursor;
    }
    result.total_ms = result.query_ms + result.update_ms + result.switch_ms;
    return result;
}

}  // namespace

int run_partition_cf(const M3Trace& trace, int fine, int quad, const std::string& plan_path,
                     int window, unsigned leaf, int mode_a, int mode_b, int horizon,
                     bool verify) {
    if (mode_a != 0) throw std::invalid_argument("counterfactual keep arm must be SCAN");
    const LeafPlan plan = read_leaf_plan(plan_path);
    if (leaf >= plan.rects.size()) throw std::invalid_argument("leaf out of range");
    const Mode target = mode_b == 1 ? Mode::FineGrid : mode_b == 2 ? Mode::LocalQuad : Mode::Scan;
    const ArmResult a = run_arm(trace, plan, fine, quad, window, leaf, Mode::Scan, false,
                                horizon, verify);
    const ArmResult b = run_arm(trace, plan, fine, quad, window, leaf, target, true,
                                horizon, verify);
    if (a.entities != b.entities) throw std::runtime_error("state mismatch between arms");
    std::cout << std::fixed << std::setprecision(6)
              << "cost_a_ms=" << a.total_ms << " cost_b_ms=" << b.total_ms
              << " net_ms=" << a.total_ms - b.total_ms
              << " query_a_ms=" << a.query_ms << " query_b_ms=" << b.query_ms
              << " update_a_ms=" << a.update_ms << " update_b_ms=" << b.update_ms
              << " switch_b_ms=" << b.switch_ms
              << " entities=" << a.entities
              << " load_q=" << a.load.queries << " load_u=" << a.load.updates
              << " load_c=" << a.load.coverage
              << " windows=" << a.windows << "/" << b.windows
              << " queries=" << a.queries << "/" << b.queries
              << " mismatches=" << a.mismatches + b.mismatches
              << " window=" << window << " leaf=" << leaf
              << " mode_a=0 mode_b=" << mode_b << " horizon=" << horizon << "\n";
    return (a.mismatches + b.mismatches) == 0 ? 0 : 1;
}
}  // namespace spatial
