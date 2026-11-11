// Persistent interactive replay environment for reinforcement learning.
//
// One AdaptiveRegionalIndex stays alive and advances one 50-step window per
// `step` command. Every reported cost is measured on the real index: per-region
// query/local-update/build/switch nanoseconds come from the index profile
// stats, base update time is measured around set/erase, and query correctness
// can be checked against an independent array oracle.
#include "core/clock.hpp"
#include "indexes/regional.hpp"
#include "validation/oracle.hpp"
#include "workload/m3.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace spatial {
namespace {

constexpr int cooldown_steps = 100;
constexpr unsigned region_limit = AdaptiveRegionalIndex::region_count;
constexpr unsigned max_rebuilds = 4;
constexpr double max_rebuild_entities = 6000.0;

class Server {
public:
    Server(const M3Trace& trace, int fine, int quad, bool verify)
        : trace_(trace), fine_(fine), quad_(quad), verify_(verify),
          index_(std::make_unique<AdaptiveRegionalIndex>(trace.bounds, Mode::Scan, fine, quad,
                                                         true, true)) {}

    void reset() {
        index_ = std::make_unique<AdaptiveRegionalIndex>(trace_.bounds, Mode::Scan, fine_, quad_,
                                                         true, true);
        index_->reset_window();
        Id id = 0;
        for (const Point& point : trace_.points) index_->set(++id, point);
        index_->reset_window();
        oracle_.reset();
        if (verify_) {
            oracle_ = std::make_unique<ArrayOracle>();
            Id oid = 0;
            for (const Point& point : trace_.points) oracle_->set(++oid, point);
        }
        cursor_ = 0;
        window_ = 0;
        step_ = 0;
        done_ = false;
        last_switch_.fill(std::numeric_limits<std::int64_t>::min() / 2);
        base_ns_.fill(0.0);
        mismatch_ = 0;
        query_events_ = 0;
    }

    std::string state_json() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3);
        out << "{\"window\":" << window_ << ",\"step\":" << step_
            << ",\"done\":" << (done_ ? "true" : "false")
            << ",\"rebuilds_left\":" << max_rebuilds
            << ",\"entities_left\":" << max_rebuild_entities
            << ",\"mismatches\":" << mismatch_ << ",\"entities\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            out << index_->entity_count(r);
        }
        out << "],\"queries\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            out << index_->window_load(r).queries;
        }
        out << "],\"updates\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            out << index_->window_load(r).updates;
        }
        out << "],\"coverage\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            out << index_->window_load(r).coverage;
        }
        out << "],\"modes\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            out << static_cast<int>(index_->mode(r));
        }
        out << "],\"cooldown\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            const std::int64_t since = step_ - last_switch_[r];
            out << (since >= cooldown_steps ? 0 : cooldown_steps - since);
        }
        out << "]}";
        return out.str();
    }

    std::string step_json(const std::vector<int>& actions) {
        unsigned rebuilds = 0;
        double entities = 0.0;
        std::vector<int> applied(region_limit, 0);
        for (unsigned r = 0; r < region_limit; ++r) applied[r] = static_cast<int>(index_->mode(r));
        for (unsigned region = 0; region < region_limit && region < actions.size(); ++region) {
            const int requested = actions[region];
            const Mode target = requested == 1 ? Mode::FineGrid
                               : requested == 2 ? Mode::LocalQuad : Mode::Scan;
            if (target == index_->mode(region)) continue;
            const std::int64_t since = step_ - last_switch_[region];
            const bool cooling = since < cooldown_steps;
            const bool demotion = target == Mode::Scan;
            const double count = static_cast<double>(index_->entity_count(region));
            const bool budget = demotion || (rebuilds < max_rebuilds &&
                                             entities + count <= max_rebuild_entities);
            if (cooling || !budget) continue;
            index_->switch_mode(region, target);
            last_switch_[region] = step_;
            if (!demotion) {
                ++rebuilds;
                entities += count;
            }
            applied[region] = requested;
        }
        std::fill(base_ns_.begin(), base_ns_.end(), 0.0);
        std::int64_t window_update_ns = 0;
        bool finished = false;
        while (cursor_ < trace_.events.size()) {
            const M3Event& event = trace_.events[cursor_++];
            if (event.type == 'P') {
                step_ = event.step;
                finished = true;
                break;
            }
            if (event.type == 'U') {
                const auto begin = Clock::now();
                index_->set(event.id, event.point);
                const auto elapsed = Clock::now() - begin;
                window_update_ns += elapsed.count();
                base_ns_[index_->region_of(event.point)] += static_cast<double>(elapsed.count());
                if (verify_) oracle_->set(event.id, event.point);
            } else if (event.type == 'D') {
                const auto begin = Clock::now();
                index_->erase(event.id);
                const auto elapsed = Clock::now() - begin;
                window_update_ns += elapsed.count();
                if (verify_) oracle_->erase(event.id);
            } else {
                Answer answer = index_->query(event.query);
                ++query_events_;
                if (verify_) {
                    std::sort(answer.ids.begin(), answer.ids.end());
                    const auto truth = oracle_->query(event.query);
                    if (answer.ids != truth) ++mismatch_;
                }
            }
        }
        if (cursor_ >= trace_.events.size()) done_ = true;
        double query_ns = 0, update_ns = 0, build_ns = 0, switch_ns = 0;
        std::vector<double> region_us(region_limit, 0.0);
        for (unsigned r = 0; r < region_limit; ++r) {
            const auto& stats = index_->region_stats(r);
            const double ns = static_cast<double>(stats.query_ns + stats.local_update_ns +
                                                  stats.build_ns + stats.switch_ns) +
                              base_ns_[r];
            region_us[r] = ns / 1000.0;
            query_ns += static_cast<double>(stats.query_ns);
            update_ns += static_cast<double>(stats.local_update_ns) + base_ns_[r];
            build_ns += static_cast<double>(stats.build_ns);
            switch_ns += static_cast<double>(stats.switch_ns);
        }
        const double total_ns = query_ns + update_ns + build_ns + switch_ns;
        std::ostringstream out;
        out << std::fixed << std::setprecision(4);
        out << "{\"ok\":true,\"window\":" << window_ << ",\"step\":" << step_
            << ",\"done\":" << (done_ ? "true" : "false") << ",\"applied\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            out << applied[r];
        }
        out << "],\"cost_ms\":" << total_ns / 1e6
            << ",\"query_ms\":" << query_ns / 1e6
            << ",\"base_update_ms\":" << window_update_ns / 1e6
            << ",\"maintenance_ms\":" << (update_ns - window_update_ns) / 1e6
            << ",\"build_ms\":" << build_ns / 1e6
            << ",\"switch_ms\":" << switch_ns / 1e6
            << ",\"policy_ms\":0,\"queries\":" << query_events_
            << ",\"mismatches\":" << mismatch_ << ",\"region_us\":[";
        for (unsigned r = 0; r < region_limit; ++r) {
            if (r) out << ',';
            out << region_us[r];
        }
        out << "],\"state\":" << state_json() << "}";
        index_->reset_window();
        ++window_;
        (void)finished;
        return out.str();
    }

private:
    const M3Trace& trace_;
    int fine_, quad_;
    bool verify_;
    std::size_t cursor_{0};
    int window_{0};
    std::int64_t step_{0};
    bool done_{false};
    std::array<std::int64_t, region_limit> last_switch_{};
    std::array<double, region_limit> base_ns_{};
    std::uint64_t mismatch_{0};
    std::uint64_t query_events_{0};
    std::unique_ptr<AdaptiveRegionalIndex> index_;
    std::unique_ptr<ArrayOracle> oracle_;
};

}  // namespace


// Replay a fixed per-window action plan through the real index on the same
// QPC timing path as `measure`. Used to time a learned policy fairly against
// the built-in selectors without the profile timers.
int run_replayplan(const M3Trace& trace, int fine, int quad,
                   const std::vector<std::vector<int>>& plan) {
    AdaptiveRegionalIndex index(trace.bounds, Mode::Scan, fine, quad, false, false);
    Id id = 0;
    for (const Point& point : trace.points) index.set(++id, point);
    index.reset_window();
    std::vector<std::int64_t> last_switch(region_limit,
                                          std::numeric_limits<std::int64_t>::min() / 2);
    std::int64_t query_ns = 0, update_ns = 0, switch_ns = 0;
    std::uint64_t switches = 0, promotions = 0, queries = 0;
    auto apply = [&](const std::vector<int>& actions, std::int64_t step) {
        unsigned rebuilds = 0;
        double entities = 0.0;
        for (unsigned region = 0; region < region_limit; ++region) {
            if (region >= actions.size()) break;
            const Mode target = actions[region] == 1 ? Mode::FineGrid
                               : actions[region] == 2 ? Mode::LocalQuad : Mode::Scan;
            if (target == index.mode(region)) continue;
            const std::int64_t since = step - last_switch[region];
            const bool demotion = target == Mode::Scan;
            const double count = static_cast<double>(index.entity_count(region));
            if (since < cooldown_steps) continue;
            if (!demotion && (rebuilds >= max_rebuilds ||
                              entities + count > max_rebuild_entities)) continue;
            const auto begin = Clock::now();
            index.switch_mode(region, target);
            switch_ns += (Clock::now() - begin).count();
            last_switch[region] = step;
            ++switches;
            if (!demotion) { ++rebuilds; entities += count; ++promotions; }
        }
    };
    std::size_t window = 0;
    apply(plan.empty() ? std::vector<int>(region_limit, 0) : plan[0], 0);
    for (const M3Event& event : trace.events) {
        if (event.type == 'P') {
            index.reset_window();
            ++window;
            if (window < plan.size()) apply(plan[window], event.step);
            continue;
        }
        if (event.type == 'U') {
            const auto begin = Clock::now();
            index.set(event.id, event.point);
            update_ns += (Clock::now() - begin).count();
        } else if (event.type == 'D') {
            const auto begin = Clock::now();
            index.erase(event.id);
            update_ns += (Clock::now() - begin).count();
        } else {
            const auto begin = Clock::now();
            const Answer answer = index.query(event.query);
            query_ns += (Clock::now() - begin).count();
            ++queries;
            (void)answer;
        }
    }
    const double online = static_cast<double>(query_ns + update_ns + switch_ns) / 1e6;
    std::cout << std::fixed << std::setprecision(6)
              << "online_ms=" << online
              << " query_ms=" << static_cast<double>(query_ns) / 1e6
              << " update_ms=" << static_cast<double>(update_ns) / 1e6
              << " switch_ms=" << static_cast<double>(switch_ns) / 1e6
              << " promotions=" << promotions << " switches=" << switches
              << " queries=" << queries << "\n";
    return 0;
}

int run_rlserve(const M3Trace& trace, int fine, int quad, bool verify) {
    Server server(trace, fine, quad, verify);
    server.reset();
    std::cout << server.state_json() << '\n' << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "close") break;
        if (line == "reset") {
            server.reset();
            std::cout << server.state_json() << '\n' << std::flush;
            continue;
        }
        if (line == "state") {
            std::cout << server.state_json() << '\n' << std::flush;
            continue;
        }
        if (line.rfind("step", 0) == 0) {
            std::vector<int> actions;
            std::stringstream stream(line.substr(4));
            std::string token;
            while (std::getline(stream, token, ',')) {
                if (token.empty()) continue;
                actions.push_back(std::stoi(token));
            }
            std::cout << server.step_json(actions) << '\n' << std::flush;
            continue;
        }
        std::cout << "{\"ok\":false,\"error\":\"unknown command\"}" << '\n' << std::flush;
    }
    return 0;
}

}  // namespace spatial
