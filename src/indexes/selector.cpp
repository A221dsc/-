#include "selector.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace spatial {
namespace {
template <std::size_t N>
double dot(const std::array<double, N>& a, const std::array<double, N>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::optional<Mode> mode_from_name(std::string_view name) {
    if (name == "SCAN") return Mode::Scan;
    if (name == "FINE_GRID") return Mode::FineGrid;
    if (name == "LOCAL_QUAD") return Mode::LocalQuad;
    return std::nullopt;
}

double require_non_negative(double value, std::string_view key) {
    if (!std::isfinite(value) || value < 0)
        throw std::runtime_error("invalid selector " + std::string(key));
    return value;
}

bool read_tokens(std::ifstream& in, std::string& line, std::istringstream& stream, std::string& key) {
    while (std::getline(in, line)) {
        const auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        stream.clear();
        stream.str(line);
        stream >> key;
        return true;
    }
    return false;
}
}

const char* policy_name(Policy policy) {
    return policy == Policy::Threshold ? "threshold" : "greedy";
}

std::optional<Policy> policy_from_name(std::string_view name) {
    if (name == "threshold") return Policy::Threshold;
    if (name == "greedy") return Policy::Greedy;
    return std::nullopt;
}

std::string_view outcome_name(Outcome outcome) {
    switch (outcome) {
        case Outcome::Applied: return "applied";
        case Outcome::Keep: return "keep";
        case Outcome::Empty: return "empty_region";
        case Outcome::Cooldown: return "cooldown";
        case Outcome::SwitchBudget: return "switch_budget";
        case Outcome::RebuildBudget: return "rebuild_budget";
        case Outcome::DenseFloor: return "dense_floor";
        case Outcome::Margin: return "below_margin";
    }
    return "unknown";
}

SelectorConfig read_selector_config(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open selector config " + path.string());
    SelectorConfig config;
    std::array<int, 3> query_seen{}, update_seen{};
    std::array<std::array<int, 3>, 3> transition_seen{};
    bool threshold_seen = false, margin_seen = false, alpha_seen = false, version_seen = false;
    std::string line, key;
    std::istringstream tokens;
    while (read_tokens(in, line, tokens, key)) {
        if (key == "version") {
            int version = 0;
            if (!(tokens >> version) || version != 1)
                throw std::runtime_error("unsupported selector config version");
            version_seen = true;
        } else if (key == "margin") {
            if (!(tokens >> config.margin)) throw std::runtime_error("invalid selector margin");
            if (!(config.margin >= 0 && config.margin < 1))
                throw std::runtime_error("selector margin out of range");
            margin_seen = true;
        } else if (key == "fusion") {
            int flag = 0;
            if (!(tokens >> flag) || (flag != 0 && flag != 1))
                throw std::runtime_error("invalid fusion flag");
            config.fusion = flag == 1;
        } else if (key == "alpha") {
            if (!(tokens >> config.alpha)) throw std::runtime_error("invalid selector alpha");
            if (!(config.alpha > 0 && config.alpha <= 1))
                throw std::runtime_error("alpha must be in (0, 1]");
            alpha_seen = true;
        } else if (key == "threshold") {
            auto& t = config.threshold;
            if (!(tokens >> t.min_entities >> t.min_queries >> t.min_ratio >> t.quad_entities
                          >> t.fine_coverage >> t.hysteresis >> t.dense_entities))
                throw std::runtime_error("invalid threshold row");
            require_non_negative(t.min_entities, "min_entities");
            require_non_negative(t.min_queries, "min_queries");
            require_non_negative(t.min_ratio, "min_ratio");
            require_non_negative(t.quad_entities, "quad_entities");
            require_non_negative(t.fine_coverage, "fine_coverage");
            require_non_negative(t.hysteresis, "hysteresis");
            require_non_negative(t.dense_entities, "dense_entities");
            if (!(t.hysteresis > 0 && t.hysteresis <= 1))
                throw std::runtime_error("hysteresis must be in (0, 1]");
            threshold_seen = true;
        } else if (key == "query" || key == "update") {
            std::string name;
            if (!(tokens >> name)) throw std::runtime_error("missing mode in " + key + " row");
            const auto mode = mode_from_name(name);
            if (!mode) throw std::runtime_error("unknown mode " + name);
            const auto index = static_cast<unsigned>(*mode);
            if (key == "query") {
                auto& row = config.cost.query[index];
                for (double& value : row)
                    if (!(tokens >> value)) throw std::runtime_error("short query row");
                for (double value : row) require_non_negative(value, "query coefficient");
                if (query_seen[index]++) throw std::runtime_error("duplicate query row");
            } else {
                auto& row = config.cost.update[index];
                for (double& value : row)
                    if (!(tokens >> value)) throw std::runtime_error("short update row");
                for (double value : row) require_non_negative(value, "update coefficient");
                if (update_seen[index]++) throw std::runtime_error("duplicate update row");
            }
        } else if (key == "switch") {
            std::string from_name, to_name;
            if (!(tokens >> from_name >> to_name)) throw std::runtime_error("missing switch pair");
            const auto from = mode_from_name(from_name), to = mode_from_name(to_name);
            if (!from || !to) throw std::runtime_error("unknown switch mode");
            auto& row = config.cost.transition[static_cast<unsigned>(*from)][static_cast<unsigned>(*to)];
            if (!(tokens >> row[0] >> row[1])) throw std::runtime_error("short switch row");
            require_non_negative(row[0], "switch intercept");
            require_non_negative(row[1], "switch slope");
            ++transition_seen[static_cast<unsigned>(*from)][static_cast<unsigned>(*to)];
        } else {
            throw std::runtime_error("unknown selector config key " + key);
        }
    }
    if (!version_seen || !threshold_seen || !margin_seen || !alpha_seen)
        throw std::runtime_error("selector config lacks version, threshold, margin or alpha");
    for (int seen : query_seen)
        if (seen != 1) throw std::runtime_error("selector config needs one query row per mode");
    for (int seen : update_seen)
        if (seen != 1) throw std::runtime_error("selector config needs one update row per mode");
    for (unsigned from = 0; from < 3; ++from)
        for (unsigned to = 0; to < 3; ++to)
            if (transition_seen[from][to] != 1)
                throw std::runtime_error("selector config needs every ordered switch pair");
    for (unsigned mode = 0; mode < 3; ++mode)
        if (config.cost.transition[mode][mode][0] != 0 || config.cost.transition[mode][mode][1] != 0)
            throw std::runtime_error("diagonal switch coefficients must be zero");
    return config;
}

Selector::Selector(Policy policy, SelectorConfig config, std::shared_ptr<const ScenarioPrior> prior)
    : policy_(policy), config_(std::move(config)), prior_(std::move(prior)) {
    last_switch_.fill(-1000000);
}

double Selector::running_cost_ns(Mode mode, std::size_t entities, const WindowLoad& load) const {
    const auto index = static_cast<unsigned>(mode);
    const double n = static_cast<double>(entities);
    const double q = static_cast<double>(load.queries);
    const double u = static_cast<double>(load.updates);
    const double c = std::max(0.0, load.coverage);
    const std::array<double, 5> query{q, q * n, n * c, q * std::sqrt(n), c};
    const std::array<double, 2> update{u, u * std::sqrt(n)};
    return dot(config_.cost.query[index], query) + dot(config_.cost.update[index], update);
}

double Selector::transition_cost_ns(Mode from, Mode to, std::size_t entities) const {
    if (from == to || entities == 0) return 0;
    const auto& row = config_.cost.transition[static_cast<unsigned>(from)][static_cast<unsigned>(to)];
    return row[0] + row[1] * static_cast<double>(entities);
}

Mode Selector::threshold_target(Mode current, std::size_t entities, const WindowLoad& load,
                                bool floor_active) const {
    const auto& t = config_.threshold;
    const double n = static_cast<double>(entities);
    const double q = static_cast<double>(load.queries);
    const double u = static_cast<double>(load.updates);
    const double hysteresis = std::clamp(t.hysteresis, 1e-3, 1.0);
    // An indexed region is demoted only once its load falls below hysteresis * threshold,
    // which keeps a region from oscillating around a single boundary.
    const double scale = current == Mode::Scan ? 1.0 : hysteresis;
    const bool crowded = n >= scale * t.min_entities && q >= scale * t.min_queries &&
                         q >= scale * t.min_ratio * u;
    if (!crowded && !floor_active) return Mode::Scan;
    const double coverage = q > 0 ? load.coverage / q : 0.0;
    const double quad_boundary = t.quad_entities *
        (current == Mode::LocalQuad ? hysteresis : current == Mode::FineGrid ? 1.0 / hysteresis : 1.0);
    const double fine_boundary = t.fine_coverage *
        (current == Mode::FineGrid ? 1.0 / hysteresis : current == Mode::LocalQuad ? hysteresis : 1.0);
    if (n >= quad_boundary || coverage > fine_boundary) return Mode::LocalQuad;
    return Mode::FineGrid;
}

std::vector<Decision> Selector::decide(AdaptiveRegionalIndex& index, std::int64_t step) {
    ++counters_.windows;
    std::vector<Decision> proposals;
    proposals.reserve(AdaptiveRegionalIndex::region_count);
    const double alpha = std::clamp(config_.alpha, 1e-3, 1.0);
    const auto prior_window = static_cast<std::size_t>(step / window_steps);
    const bool prior_on = prior_ && prior_->valid() && prior_->has(prior_window) &&
                          prior_->regions() > 0;
    for (unsigned region = 0; region < AdaptiveRegionalIndex::region_count; ++region) {
        Decision decision;
        decision.region = region;
        decision.entities = index.entity_count(region);
        decision.load = index.window_load(region);
        decision.old_mode = index.mode(region);
        decision.target = decision.old_mode;
        auto& smooth = smoothed_[region];
        const double raw_q = static_cast<double>(decision.load.queries);
        const double raw_u = static_cast<double>(decision.load.updates);
        const double raw_c = decision.load.coverage;
        if (!smooth.initialised) {
            smooth = {raw_q, raw_u, raw_c, true};
        } else {
            smooth.queries = alpha * raw_q + (1.0 - alpha) * smooth.queries;
            smooth.updates = alpha * raw_u + (1.0 - alpha) * smooth.updates;
            smooth.coverage = alpha * raw_c + (1.0 - alpha) * smooth.coverage;
        }
        std::size_t forecast_entities = decision.entities;
        if (prior_on && region < prior_->regions()) {
            const auto& cell = prior_->at(prior_window, region);
            decision.forecast = {static_cast<std::uint64_t>(std::llround(cell.queries)),
                                 static_cast<std::uint64_t>(std::llround(cell.updates)),
                                 cell.coverage};
            forecast_entities = static_cast<std::size_t>(std::llround(cell.entities));
            if (config_.fusion) {
                decision.forecast.queries = std::max(
                    decision.forecast.queries,
                    static_cast<std::uint64_t>(std::llround(smooth.queries)));
                decision.forecast.updates = std::max(
                    decision.forecast.updates,
                    static_cast<std::uint64_t>(std::llround(smooth.updates)));
                decision.forecast.coverage = std::max(decision.forecast.coverage, smooth.coverage);
                forecast_entities = std::max(forecast_entities, decision.entities);
            }
        } else {
            decision.forecast = {static_cast<std::uint64_t>(std::llround(smooth.queries)),
                                 static_cast<std::uint64_t>(std::llround(smooth.updates)),
                                 smooth.coverage};
        }
        if (decision.entities == 0) {
            if (decision.old_mode != Mode::Scan) {
                decision.target = Mode::Scan;
                decision.outcome = Outcome::Empty;
                proposals.push_back(decision);
            }
            continue;
        }
        const bool quiet = decision.load.queries == 0 && decision.load.updates == 0 &&
                           smooth.queries < 1 && smooth.updates < 1 &&
                           (!prior_on || (decision.forecast.queries == 0 &&
                                          decision.forecast.updates == 0));
        if (quiet) continue;
        const bool floor_active = config_.threshold.dense_entities > 0 &&
                                  static_cast<double>(decision.entities) >= config_.threshold.dense_entities;
        Mode target = decision.old_mode;
        if (policy_ == Policy::Threshold) {
            target = threshold_target(decision.old_mode, forecast_entities, decision.forecast,
                                      floor_active);
        } else {
            const double stay = running_cost_ns(decision.old_mode, forecast_entities, decision.forecast);
            double best = stay;
            for (Mode candidate : {Mode::Scan, Mode::FineGrid, Mode::LocalQuad}) {
                const double cost = running_cost_ns(candidate, forecast_entities, decision.forecast) +
                                    transition_cost_ns(decision.old_mode, candidate, decision.entities);
                if (cost < best) {
                    best = cost;
                    target = candidate;
                }
            }
            decision.has_estimate = true;
            decision.baseline_ns = stay;
            decision.estimate_ns = best;
            if (target != decision.old_mode && best >= stay * (1.0 - config_.margin)) {
                decision.target = decision.old_mode;
                decision.outcome = Outcome::Margin;
                proposals.push_back(decision);
                continue;
            }
        }
        if (floor_active && target == Mode::Scan && decision.old_mode != Mode::Scan)
            target = decision.old_mode;   // the structural floor keeps dense regions indexed
        if (floor_active && target == Mode::Scan && decision.old_mode == Mode::Scan)
            target = Mode::LocalQuad;     // and pre-indexes a dense region that is still cold
        if (target == decision.old_mode) {
            if (decision.has_estimate) decision.outcome = Outcome::Keep;
            continue;
        }
        decision.target = target;
        if (decision.has_estimate)
            decision.estimate_ns = running_cost_ns(target, forecast_entities, decision.forecast) +
                                   transition_cost_ns(decision.old_mode, target, decision.entities);
        proposals.push_back(decision);
    }

    std::stable_sort(proposals.begin(), proposals.end(), [&](const Decision& a, const Decision& b) {
        const bool demote_a = a.target == Mode::Scan, demote_b = b.target == Mode::Scan;
        if (demote_a != demote_b) return demote_a;   // demotions are free and free a slot first
        if (policy_ == Policy::Greedy) {
            const double gain_a = a.baseline_ns - a.estimate_ns;
            const double gain_b = b.baseline_ns - b.estimate_ns;
            if (gain_a != gain_b) return gain_a > gain_b;
        } else {
            const double gain_a = a.old_mode == Mode::Scan ? static_cast<double>(a.load.queries) : 0.0;
            const double gain_b = b.old_mode == Mode::Scan ? static_cast<double>(b.load.queries) : 0.0;
            if (gain_a != gain_b) return gain_a > gain_b;
        }
        return a.region < b.region;
    });

    unsigned rebuilds = 0;
    double rebuilt = 0;
    for (Decision& decision : proposals) {
        if (decision.outcome == Outcome::Margin) { ++counters_.blocked; continue; }
        if (decision.target != Mode::Scan) {
            if (step - last_switch_[decision.region] < cooldown_steps) {
                decision.outcome = Outcome::Cooldown;
                ++counters_.blocked;
                continue;
            }
            if (rebuilds >= max_rebuilds_per_window) {
                decision.outcome = Outcome::SwitchBudget;
                ++counters_.blocked;
                continue;
            }
            if (rebuilt + static_cast<double>(decision.entities) > max_rebuild_entities) {
                decision.outcome = Outcome::RebuildBudget;
                ++counters_.blocked;
                continue;
            }
        } else if (step - last_switch_[decision.region] < cooldown_steps) {
            decision.outcome = Outcome::Cooldown;
            ++counters_.blocked;
            continue;
        }
        const auto before = index.times().switch_ns;
        index.switch_mode(decision.region, decision.target);
        decision.switch_ns = index.times().switch_ns - before;
        decision.applied = true;
        decision.outcome = Outcome::Applied;
        last_switch_[decision.region] = step;
        if (decision.target == Mode::Scan) {
            ++counters_.demotions;
        } else {
            ++rebuilds;
            rebuilt += static_cast<double>(decision.entities);
            if (decision.old_mode == Mode::Scan) ++counters_.promotions;
            else ++counters_.retunes;
        }
    }
    return proposals;
}
}
