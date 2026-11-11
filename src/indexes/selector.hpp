#pragma once
#include "prior.hpp"
#include "regional.hpp"
#include <array>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace spatial {
enum class Policy { Threshold, Greedy };

// Rule parameters of the threshold policy S3. Every value is chosen on the
// independent validation traces; none of them is fitted to the test traces.
struct ThresholdParams {
    double min_entities{1};    // below this region size scanning is always cheaper
    double min_queries{1};     // below this window query count an index cannot amortise
    double min_ratio{0};       // queries per update below which maintenance dominates
    double quad_entities{1e9}; // region size from which the local quadtree tier is preferred
    double fine_coverage{0};   // mean covered fraction above which the quadtree is preferred
    double hysteresis{0.7};    // demotion needs load < hysteresis * threshold
    double dense_entities{0};  // structural floor: dense regions are never left to SCAN
};

// Linear predictors in nanoseconds, fitted on the training replays only.
// Query basis: Q, Q*N, N*C, Q*sqrt(N), C   (Q touches, N entities, C coverage sum)
// Update basis: U, U*sqrt(N)
// Switch basis: 1, N   for each ordered (from, to) mode pair
struct CostModel {
    std::array<std::array<double, 5>, 3> query{};
    std::array<std::array<double, 2>, 3> update{};
    std::array<std::array<std::array<double, 2>, 3>, 3> transition{};
};

struct SelectorConfig {
    ThresholdParams threshold;
    double margin{0.1};
    double alpha{1.0};   // EWMA weight of the newest window; 1 = raw last window
    bool fusion{false};  // take the per-region maximum of prior and history forecasts
    int uncertainty{0};  // 0 point prior, 1 expected cost, 2 quantile penalty, 3 CVaR penalty
    double risk_lambda{0.0};
    double cvar_alpha{0.8};   // confidence level of the CVaR risk measure
    int horizon_windows{1};   // number of 50-step windows summed by the cost model
    int fusion_mode{0};       // 0 prior only, 1 reliability-weighted prior/history blend
    double fusion_decay{0.7}; // EWMA factor of the reliability estimate
    CostModel cost;
};

enum class Outcome { Applied, Keep, Empty, Cooldown, SwitchBudget, RebuildBudget, DenseFloor, Margin };
std::string_view outcome_name(Outcome outcome);
const char* policy_name(Policy policy);
std::optional<Policy> policy_from_name(std::string_view name);

struct Decision {
    unsigned region{};
    std::size_t entities{};
    WindowLoad load;      // observed window that drove the decision
    WindowLoad forecast;  // smoothed load used by the rule
    Mode old_mode{Mode::Scan};
    Mode target{Mode::Scan};
    Outcome outcome{Outcome::Keep};
    bool applied{};
    bool has_estimate{};
    double baseline_ns{};   // predicted cost of keeping the current mode
    double estimate_ns{};   // predicted cost of the target, switch included
    std::int64_t switch_ns{};
};

struct SelectorCounters {
    std::uint64_t promotions{}, demotions{}, retunes{}, blocked{}, windows{};
};

// Online regional selection. Decisions only read the load accumulated since the
// previous call plus current entity counts and modes: no future event is
// visible, and the switch cost is produced by the index during the call.
class Selector {
public:
    static constexpr int window_steps = 50;
    static constexpr int cooldown_steps = 2 * window_steps;
    static constexpr unsigned max_rebuilds_per_window = 4;
    static constexpr double max_rebuild_entities = 6000;

    Selector(Policy policy, SelectorConfig config,
             std::shared_ptr<const ScenarioPrior> prior = nullptr);

    std::vector<Decision> decide(AdaptiveRegionalIndex& index, std::int64_t step);

    Policy policy() const { return policy_; }
    bool feedforward() const { return static_cast<bool>(prior_); }
    const SelectorConfig& config() const { return config_; }
    const SelectorCounters& counters() const { return counters_; }

private:
    struct Smoothed {
        double queries{}, updates{}, coverage{};
        bool initialised{};
    };

    Policy policy_;
    SelectorConfig config_;
    std::shared_ptr<const ScenarioPrior> prior_;
    std::array<std::int64_t, AdaptiveRegionalIndex::region_count> last_switch_{};
    std::array<Smoothed, AdaptiveRegionalIndex::region_count> smoothed_{};
    double reliability_{1.0};
    SelectorCounters counters_{};

    double running_cost_ns(Mode mode, std::size_t entities, const WindowLoad& load) const;
    double transition_cost_ns(Mode from, Mode to, std::size_t entities) const;
    Mode threshold_target(Mode current, std::size_t entities, const WindowLoad& load,
                          bool floor_active) const;
};

// Reads the tagged selector configuration produced by scripts/calibrate_m3.py.
SelectorConfig read_selector_config(const std::filesystem::path& path);
}
