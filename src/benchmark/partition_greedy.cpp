// Online cost-greedy selection on the adaptive partition index.
//
// The prior mode uses the per-leaf nominal loads from the plan P lines; the
// history mode uses the leaf loads observed in the window that just ended. Both
// share the same cost model from the selector config, the 100-step cooldown and
// the four-rebuild / 6000-entity window budget. Decisions use the actual
// executed modes, never an idealized requested state.
#include "core/clock.hpp"
#include "indexes/partition.hpp"
#include "indexes/selector.hpp"
#include "validation/oracle.hpp"
#include "workload/m3.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace spatial {
namespace {

constexpr int kCooldown = 100;
constexpr unsigned kMaxRebuilds = 4;
constexpr double kMaxEntities = 6000.0;

struct GreedyPlan {
    std::vector<std::array<int, 4>> rects;
    std::vector<unsigned> levels;
    std::map<std::pair<int, unsigned>, std::array<double, 4>> loads;  // q,n,u,c
};

GreedyPlan read_greedy_plan(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open partition plan " + path);
    GreedyPlan plan;
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
        } else if (kind == "P") {
            int window = 0;
            unsigned leaf = 0;
            std::array<double, 4> load{};
            stream >> window >> leaf >> load[0] >> load[1] >> load[2] >> load[3];
            plan.loads[{window, leaf}] = load;
        }
    }
    if (plan.rects.empty()) throw std::runtime_error("empty partition plan");
    return plan;
}

double cost_ns(const SelectorConfig& config, Mode mode, const std::array<double, 4>& load) {
    const auto& q = config.cost.query[static_cast<int>(mode)];
    const auto& u = config.cost.update[static_cast<int>(mode)];
    const double queries = load[0], entities = load[1], updates = load[2], coverage = load[3];
    const double root = std::sqrt(std::max(entities, 0.0));
    return queries * q[0] + queries * entities * q[1] + entities * coverage * q[2] +
           queries * root * q[3] + coverage * q[4] +
           updates * u[0] + updates * root * u[1];
}

double switch_ns(const SelectorConfig& config, Mode from, Mode to, double entities) {
    if (from == to) return 0.0;
    const auto& pair = config.cost.transition[static_cast<int>(from)][static_cast<int>(to)];
    return pair[0] + entities * pair[1];
}

struct GreedyOutcome {
    double online_ms{}, query_ms{}, update_ms{}, switch_ms{}, decide_ms{};
    std::uint64_t queries{}, promotions{}, switches{}, mismatches{}, decisions{};
};

GreedyOutcome replay_greedy(const M3Trace& trace, const GreedyPlan& plan,
                            const SelectorConfig& config, int fine, int quad,
                            bool use_prior, bool validate, int amortize) {
    PartitionIndex index(trace.bounds, fine, quad, false, true);
    index.set_partition(plan.rects, plan.levels);
    std::unique_ptr<ArrayOracle> oracle;
    if (validate) oracle = std::make_unique<ArrayOracle>();
    Id id = 0;
    for (const Point& point : trace.points) {
        index.set(++id, point);
        if (validate) oracle->set(id, point);
    }
    index.reset_window();
    std::vector<std::int64_t> last_switch(plan.rects.size(),
                                          std::numeric_limits<std::int64_t>::min() / 2);
    GreedyOutcome outcome;
    int window = 0;
    auto decide = [&](int next_window, std::int64_t step) {
        const auto begin = Clock::now();
        std::vector<std::tuple<double, unsigned, int>> candidates;
        for (unsigned leaf = 0; leaf < plan.rects.size(); ++leaf) {
            std::array<double, 4> now{};
            if (use_prior) {
                const auto found = plan.loads.find({next_window, leaf});
                if (found != plan.loads.end()) now = found->second;
            } else {
                const auto& observed = index.window_load(leaf);
                now = {static_cast<double>(observed.queries),
                       static_cast<double>(index.entity_count(leaf)),
                       static_cast<double>(observed.updates), observed.coverage};
            }
            const Mode current = index.mode(leaf);
            const double entities = static_cast<double>(index.entity_count(leaf));
            const int horizon = std::max(1, amortize);
            // Amortized cost over the declared future windows: the prior loads
            // are used when available (never actual future events); the history
            // mode repeats the observed load (persistence).
            auto sustained = [&](Mode mode) {
                double total = 0.0;
                for (int h = 0; h < horizon; ++h) {
                    std::array<double, 4> load = now;
                    const auto found = plan.loads.find({next_window + h, leaf});
                    if (use_prior && found != plan.loads.end()) load = found->second;
                    total += cost_ns(config, mode, load);
                }
                return total;
            };
            double best = sustained(current) + switch_ns(config, current, current, entities);
            Mode target = current;
            for (Mode mode : {Mode::Scan, Mode::FineGrid, Mode::LocalQuad}) {
                const double value = sustained(mode) + switch_ns(config, current, mode, entities);
                if (value < best) {
                    best = value;
                    target = mode;
                }
            }
            if (target == current) continue;
            const double benefit = sustained(current) - best;
            if (benefit <= 0) continue;
            candidates.emplace_back(benefit, leaf, static_cast<int>(target));
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        unsigned rebuilds = 0;
        double entities = 0.0;
        std::int64_t switch_inside = 0;
        for (const auto& [benefit, leaf, target] : candidates) {
            const Mode mode = static_cast<Mode>(target);
            const bool demotion = mode == Mode::Scan;
            if (step - last_switch[leaf] < kCooldown) continue;
            if (!demotion && rebuilds >= kMaxRebuilds) continue;
            if (!demotion && entities + index.entity_count(leaf) > kMaxEntities) continue;
            const auto switch_begin = Clock::now();
            index.set_mode(leaf, mode);
            const auto switch_elapsed = (Clock::now() - switch_begin).count();
            switch_inside += switch_elapsed;
            outcome.switch_ms += switch_elapsed / 1e6;
            last_switch[leaf] = step;
            ++outcome.switches;
            if (!demotion) {
                ++rebuilds;
                entities += static_cast<double>(index.entity_count(leaf));
                ++outcome.promotions;
            }
            ++outcome.decisions;
        }
        outcome.decide_ms += ((Clock::now() - begin).count() - switch_inside) / 1e6;
    };
    decide(0, 0);
    for (const M3Event& event : trace.events) {
        if (event.type == 'P') {
            decide(window + 1, event.step);
            index.reset_window();
            ++window;
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
            outcome.query_ms += (Clock::now() - begin).count() / 1e6;
            ++outcome.queries;
            if (validate) {
                std::sort(answer.ids.begin(), answer.ids.end());
                if (answer.ids != oracle->query(event.query)) ++outcome.mismatches;
            }
        }
    }
    outcome.online_ms = outcome.query_ms + outcome.update_ms + outcome.switch_ms +
                        outcome.decide_ms;
    return outcome;
}

}  // namespace

int run_partition_greedy(const M3Trace& trace, int fine, int quad, const std::string& plan_path,
                         const std::string& config_path, bool use_prior, bool validate,
                         int amortize) {
    const GreedyPlan plan = read_greedy_plan(plan_path);
    const SelectorConfig config = read_selector_config(config_path);
    const GreedyOutcome outcome = replay_greedy(trace, plan, config, fine, quad, use_prior,
                                                validate, amortize);
    std::cout << std::fixed << std::setprecision(6)
              << "online_ms=" << outcome.online_ms
              << " query_ms=" << outcome.query_ms
              << " update_ms=" << outcome.update_ms
              << " switch_ms=" << outcome.switch_ms
              << " decide_ms=" << outcome.decide_ms
              << " queries=" << outcome.queries
              << " promotions=" << outcome.promotions
              << " switches=" << outcome.switches
              << " decisions=" << outcome.decisions
              << " leaves=" << plan.rects.size()
              << " mismatches=" << outcome.mismatches << "\n";
    return outcome.mismatches == 0 ? 0 : 1;
}
}  // namespace spatial
