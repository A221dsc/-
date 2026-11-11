// Interactive RL replay on the PartitionIndex (fixed leaf set).
//
// Same line protocol as rlserve: reset/state/step/close. The leaf count is
// variable (the plan fixes it), so the state arrays and the action vector have
// exactly leaf_count entries. Query/update/switch costs are the index's own
// per-leaf decomposition; the build inside maintain/switch is counted only once.
#include "core/clock.hpp"
#include "indexes/partition.hpp"
#include "validation/oracle.hpp"
#include "workload/m3.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace spatial {
namespace {

constexpr int kCooldown = 100;
constexpr unsigned kMaxRebuilds = 4;
constexpr double kMaxEntities = 6000.0;

struct Plan {
    std::vector<std::array<int, 4>> rects;
    std::vector<unsigned> levels;
};

Plan read_plan(const std::string& path) {
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

class PartitionServer {
public:
    PartitionServer(const M3Trace& trace, int fine, int quad, const Plan& plan, bool verify)
        : trace_(trace), fine_(fine), quad_(quad), plan_(plan), verify_(verify),
          index_(std::make_unique<PartitionIndex>(trace.bounds, fine, quad, true, true)) {}

    void reset() {
        index_ = std::make_unique<PartitionIndex>(trace_.bounds, fine_, quad_, true, true);
        index_->set_partition(plan_.rects, plan_.levels);
        index_->reset_window();
        Id id = 0;
        for (const Point& point : trace_.points) index_->set(++id, point);
        index_->reset_window();
        if (verify_) {
            oracle_ = std::make_unique<ArrayOracle>();
            Id oid = 0;
            for (const Point& point : trace_.points) oracle_->set(++oid, point);
        } else {
            oracle_.reset();
        }
        cursor_ = 0;
        window_ = 0;
        step_ = 0;
        done_ = false;
        last_switch_.assign(plan_.rects.size(),
                            std::numeric_limits<std::int64_t>::min() / 2);
        mismatch_ = 0;
        queries_ = 0;
    }

    std::string state_json() const {
        const unsigned leaves = static_cast<unsigned>(plan_.rects.size());
        std::ostringstream out;
        out << std::fixed << std::setprecision(3);
        out << "{\"window\":" << window_ << ",\"step\":" << step_
            << ",\"done\":" << (done_ ? "true" : "false")
            << ",\"leaves\":" << leaves
            << ",\"rebuilds_left\":" << kMaxRebuilds
            << ",\"entities_left\":" << kMaxEntities
            << ",\"mismatches\":" << mismatch_ << ",\"entities\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << index_->entity_count(r);
        }
        out << "],\"queries\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << index_->window_load(r).queries;
        }
        out << "],\"updates\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << index_->window_load(r).updates;
        }
        out << "],\"coverage\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << index_->window_load(r).coverage;
        }
        out << "],\"modes\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << static_cast<int>(index_->mode(r));
        }
        out << "],\"cooldown\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            const std::int64_t since = step_ - last_switch_[r];
            out << (since >= kCooldown ? 0 : kCooldown - since);
        }
        out << "]}";
        return out.str();
    }

    std::string step_json(const std::vector<int>& actions, const std::vector<double>& benefits) {
        const unsigned leaves = static_cast<unsigned>(plan_.rects.size());
        std::vector<int> applied(leaves, 0), rejected(leaves, 1);
        for (unsigned r = 0; r < leaves; ++r) applied[r] = static_cast<int>(index_->mode(r));
        struct Candidate { unsigned leaf; Mode target; double benefit; double entities; };
        std::vector<Candidate> candidates;
        for (unsigned leaf = 0; leaf < leaves && leaf < actions.size(); ++leaf) {
            const Mode target = actions[leaf] == 1 ? Mode::FineGrid
                               : actions[leaf] == 2 ? Mode::LocalQuad : Mode::Scan;
            if (target == index_->mode(leaf)) continue;
            const double benefit = leaf < benefits.size() ? benefits[leaf] : 1.0;
            candidates.push_back({leaf, target, benefit,
                                  static_cast<double>(index_->entity_count(leaf))});
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const auto& a, const auto& b) { return a.benefit > b.benefit; });
        unsigned rebuilds = 0;
        double entities = 0.0;
        for (const auto& candidate : candidates) {
            const bool demotion = candidate.target == Mode::Scan;
            if (step_ - last_switch_[candidate.leaf] < kCooldown) { rejected[candidate.leaf] = 2; continue; }
            if (!demotion && candidate.benefit <= 0) { rejected[candidate.leaf] = 5; continue; }
            if (!demotion && rebuilds >= kMaxRebuilds) { rejected[candidate.leaf] = 3; continue; }
            if (!demotion && entities + candidate.entities > kMaxEntities) {
                rejected[candidate.leaf] = 4;
                continue;
            }
            index_->set_mode(candidate.leaf, candidate.target);
            last_switch_[candidate.leaf] = step_;
            if (!demotion) {
                ++rebuilds;
                entities += candidate.entities;
            }
            applied[candidate.leaf] = static_cast<int>(candidate.target);
            rejected[candidate.leaf] = 0;
        }
        while (cursor_ < trace_.events.size()) {
            const M3Event& event = trace_.events[cursor_++];
            if (event.type == 'P') {
                step_ = event.step;
                break;
            }
            if (event.type == 'U') {
                index_->set(event.id, event.point);
                if (verify_) oracle_->set(event.id, event.point);
            } else if (event.type == 'D') {
                index_->erase(event.id);
                if (verify_) oracle_->erase(event.id);
            } else {
                Answer answer = index_->query(event.query);
                ++queries_;
                if (verify_) {
                    std::sort(answer.ids.begin(), answer.ids.end());
                    if (answer.ids != oracle_->query(event.query)) ++mismatch_;
                }
            }
        }
        if (cursor_ >= trace_.events.size()) done_ = true;
        double query_ms = 0, update_ms = 0, switch_ms = 0;
        std::vector<double> region_us(leaves, 0.0);
        for (unsigned r = 0; r < leaves; ++r) {
            const auto& t = index_->leaf_times(r);
            query_ms += static_cast<double>(t.query_ns) / 1e6;
            update_ms += static_cast<double>(t.base_ns + t.local_ns) / 1e6;
            switch_ms += static_cast<double>(t.switch_ns) / 1e6;
            region_us[r] = static_cast<double>(t.query_ns + t.base_ns + t.local_ns +
                                               t.switch_ns) / 1000.0;
        }
        const double cost_ms = query_ms + update_ms + switch_ms;
        std::ostringstream out;
        out << std::fixed << std::setprecision(4);
        out << "{\"ok\":true,\"window\":" << window_ << ",\"step\":" << step_
            << ",\"done\":" << (done_ ? "true" : "false") << ",\"applied\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << applied[r];
        }
        out << "],\"rejected\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << rejected[r];
        }
        out << "],\"cost_ms\":" << cost_ms
            << ",\"query_ms\":" << query_ms
            << ",\"base_update_ms\":" << update_ms
            << ",\"maintenance_ms\":0"
            << ",\"switch_ms\":" << switch_ms
            << ",\"build_ms\":0,\"region_sum_ms\":" << cost_ms
            << ",\"policy_ms\":0,\"queries\":" << queries_
            << ",\"mismatches\":" << mismatch_ << ",\"region_us\":[";
        for (unsigned r = 0; r < leaves; ++r) {
            if (r) out << ',';
            out << region_us[r];
        }
        out << "],\"state\":" << state_json() << "}";
        index_->reset_window();
        ++window_;
        return out.str();
    }

private:
    const M3Trace& trace_;
    int fine_, quad_;
    Plan plan_;
    bool verify_;
    std::size_t cursor_{0};
    int window_{0};
    std::int64_t step_{0};
    bool done_{false};
    std::vector<std::int64_t> last_switch_;
    std::uint64_t mismatch_{0}, queries_{0};
    std::unique_ptr<PartitionIndex> index_;
    std::unique_ptr<ArrayOracle> oracle_;
};

}  // namespace

int run_partition_rlserve(const M3Trace& trace, int fine, int quad, const std::string& plan_path,
                          bool verify) {
    const Plan plan = read_plan(plan_path);
    PartitionServer server(trace, fine, quad, plan, verify);
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
            std::vector<double> benefits;
            std::string body = line.substr(4);
            const auto split = body.find(';');
            std::string actions_text = split == std::string::npos ? body : body.substr(0, split);
            std::stringstream stream(actions_text);
            std::string token;
            while (std::getline(stream, token, ',')) {
                if (!token.empty()) actions.push_back(std::stoi(token));
            }
            if (split != std::string::npos) {
                std::stringstream extra(body.substr(split + 1));
                while (std::getline(extra, token, ',')) {
                    if (!token.empty()) benefits.push_back(std::stod(token));
                }
            }
            std::cout << server.step_json(actions, benefits) << '\n' << std::flush;
            continue;
        }
        std::cout << "{\"ok\":false,\"error\":\"unknown command\"}" << '\n' << std::flush;
    }
    return 0;
}
}  // namespace spatial
