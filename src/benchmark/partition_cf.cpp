// Counterfactual single-region switch measurement on the PartitionIndex (v2).
//
// Both arms rebuild the same prefix from the deterministic trace, so they reach
// the identical authoritative state (verified by a full state digest over the
// entity ids, positions, leaves and count). The target leaf then keeps its mode
// or switches before the same future events are replayed for a fixed horizon.
//
// Two cost views are reported:
//   full  : QPC wall time of the horizon events (query + update + switch)
//   local : the target leaf's own query/base/local/switch components (profile)
// The profile mode is diagnostic only; the headline timing uses profile=0.
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
    double local_query_ms{}, local_base_ms{}, local_local_ms{}, local_switch_ms{};
    std::uint64_t digest_before{}, digest_after{};
    std::size_t entities{};
    WindowLoad load;
    int windows{};
    std::uint64_t queries{}, mismatches{};
};

ArmResult run_arm(const M3Trace& trace, const LeafPlan& plan, int fine, int quad,
                  int window, unsigned leaf, Mode mode_b, bool switch_first, int horizon,
                  bool verify, bool profile) {
    PartitionIndex index(trace.bounds, fine, quad, profile, true);
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
    if (window > 0 && cursor < trace.events.size()) ++cursor;  // the P at the target window start
    result.entities = index.entity_count(leaf);
    result.load = index.window_load(leaf);
    result.digest_before = index.state_digest();
    index.reset_window();
    if (switch_first) {
        const auto begin = Clock::now();
        index.set_mode(leaf, mode_b);
        result.switch_ms += (Clock::now() - begin).count() / 1e6;
    }
    const std::int64_t end_step = target_step + static_cast<std::int64_t>(horizon) * 50;
    result.windows = 1;
    auto accumulate_local = [&]() {
        if (!profile) return;
        const auto& t = index.leaf_times(leaf);
        result.local_query_ms += static_cast<double>(t.query_ns) / 1e6;
        result.local_base_ms += static_cast<double>(t.base_ns) / 1e6;
        result.local_local_ms += static_cast<double>(t.local_ns) / 1e6;
        result.local_switch_ms += static_cast<double>(t.switch_ns) / 1e6;
    };
    while (cursor < trace.events.size()) {
        const M3Event& event = trace.events[cursor];
        if (event.type == 'P') {
            accumulate_local();
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
    accumulate_local();
    result.digest_after = index.state_digest();
    result.total_ms = result.query_ms + result.update_ms + result.switch_ms;
    return result;
}

}  // namespace

int run_partition_cf(const M3Trace& trace, int fine, int quad, const std::string& plan_path,
                     int window, unsigned leaf, int mode_a, int mode_b, int horizon,
                     bool verify, bool profile, bool switch_first_order) {
    if (mode_a != 0) throw std::invalid_argument("counterfactual keep arm must be SCAN");
    const LeafPlan plan = read_leaf_plan(plan_path);
    if (leaf >= plan.rects.size()) throw std::invalid_argument("leaf out of range");
    const Mode target = mode_b == 1 ? Mode::FineGrid : mode_b == 2 ? Mode::LocalQuad : Mode::Scan;
    ArmResult a, b;
    if (switch_first_order) {
        b = run_arm(trace, plan, fine, quad, window, leaf, target, true, horizon, verify, profile);
        a = run_arm(trace, plan, fine, quad, window, leaf, Mode::Scan, false, horizon, verify,
                    profile);
    } else {
        a = run_arm(trace, plan, fine, quad, window, leaf, Mode::Scan, false, horizon, verify,
                    profile);
        b = run_arm(trace, plan, fine, quad, window, leaf, target, true, horizon, verify, profile);
    }
    const bool state_equal = a.digest_before == b.digest_before && a.digest_after == b.digest_after;
    std::cout << std::fixed << std::setprecision(6)
              << "cost_a_ms=" << a.total_ms << " cost_b_ms=" << b.total_ms
              << " net_ms=" << a.total_ms - b.total_ms
              << " query_a_ms=" << a.query_ms << " query_b_ms=" << b.query_ms
              << " update_a_ms=" << a.update_ms << " update_b_ms=" << b.update_ms
              << " switch_b_ms=" << b.switch_ms
              << " local_q_a=" << a.local_query_ms << " local_q_b=" << b.local_query_ms
              << " local_base_a=" << a.local_base_ms << " local_base_b=" << b.local_base_ms
              << " local_local_a=" << a.local_local_ms << " local_local_b=" << b.local_local_ms
              << " local_switch_a=" << a.local_switch_ms << " local_switch_b=" << b.local_switch_ms
              << " digest_before_a=" << a.digest_before << " digest_before_b=" << b.digest_before
              << " digest_after_a=" << a.digest_after << " digest_after_b=" << b.digest_after
              << " state_equal=" << (state_equal ? 1 : 0)
              << " entities=" << a.entities
              << " load_q=" << a.load.queries << " load_u=" << a.load.updates
              << " load_c=" << a.load.coverage
              << " windows=" << a.windows << "/" << b.windows
              << " queries=" << a.queries << "/" << b.queries
              << " mismatches=" << a.mismatches + b.mismatches
              << " window=" << window << " leaf=" << leaf
              << " mode_a=0 mode_b=" << mode_b << " horizon=" << horizon << "\n";
    return (a.mismatches + b.mismatches) == 0 && state_equal ? 0 : 1;
}
}  // namespace spatial
