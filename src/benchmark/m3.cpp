// M3 benchmark driver: long-horizon regional mode selection.
//
// Subcommands
//   measure    METHOD FINE QUAD PARAM CPU TRACE CONFIG|NONE
//   validate   METHOD FINE QUAD PARAM CPU TRACE CONFIG|NONE
//   profile    METHOD FINE QUAD PARAM CPU TRACE CONFIG|NONE PREFIX
//   switchprobe FINE QUAD CPU TRACE
//
// The primary timing path never validates and never records per-region clocks.
// The profile path replays the same events with per-region clocks and the
// decision log; it is used for counterfactual cost analysis, not for the
// headline timings. The O(1) decision hash of both paths must agree.
#include "core/clock.hpp"
#include "indexes/prior.hpp"
#include "indexes/regional.hpp"
#include "indexes/selector.hpp"
#include "validation/oracle.hpp"
#include "workload/m3.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#include <psapi.h>
#else
#include <sched.h>
#include <sys/resource.h>
#endif

using namespace spatial;

namespace spatial {
int run_rlserve(const M3Trace& trace, int fine, int quad, bool verify);
int run_partition(const M3Trace& trace, int fine, int quad,
                  const std::string& plan_path, bool validate);
int run_replayplan(const M3Trace& trace, int fine, int quad,
                   const std::vector<std::vector<int>>& plan);
}

namespace {
void pin(int cpu) {
#ifdef _WIN32
    if (cpu < 0 || cpu >= int(sizeof(DWORD_PTR) * 8) ||
        !SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << cpu))
        throw std::runtime_error("CPU affinity failed");
#else
    if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::runtime_error("invalid CPU");
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        throw std::runtime_error("CPU affinity failed");
#endif
}

double peak_mib() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS memory{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
        throw std::runtime_error("PeakWorkingSetSize unavailable");
    return memory.PeakWorkingSetSize / 1048576.0;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("RSS unavailable");
    return usage.ru_maxrss / 1024.0;
#endif
}

double ms_of(Ns duration) { return duration.count() / 1e6; }

struct Method {
    std::string name;
    std::optional<Mode> fixed;
    std::optional<Policy> policy;
    bool classic{};
    bool feedforward{};
    bool fusion{};
    int uncertainty{};
};

Method parse_method(std::string_view name) {
    Method method;
    method.name = std::string(name);
    // Full method names used by the M4 reports; the short identifiers remain as
    // aliases so the M1-M3 reproduction commands keep working.
    if (name == "Fixed Regional Scan") name = "S0_ALL_SCAN";
    else if (name == "Fixed Local Fine Grid") name = "S1_ALL_FINE";
    else if (name == "Fixed Local Quadtree") name = "S2_ALL_QUAD";
    else if (name == "Uniform Grid") name = "UNIFORM_GRID";
    else if (name == "PR Quadtree") name = "PR_QUADTREE";
    else if (name == "Boost R*-Tree") name = "RSTAR_BOOST";
    else if (name == "libspatialindex R*-Tree") name = "RSTAR_LIBSPATIALINDEX";
    else if (name == "Threshold-Based Adaptive Regional Index Selection") name = "S3_THRESHOLD";
    else if (name == "Threshold-Based Adaptive Regional Index Selection (Scenario)")
        { name = "S3_THRESHOLD"; method.feedforward = true; }
    else if (name == "Cost-Aware Greedy Regional Index Selection") name = "S4_GREEDY";
    else if (name == "Cost-Aware Greedy Regional Index Selection (Scenario)")
        { name = "S4_GREEDY"; method.feedforward = true; }
    else if (name == "Cost-Aware Greedy Regional Index Selection (Scenario + History)")
        { name = "S4_GREEDY"; method.feedforward = true; method.fusion = true; }
    else if (name == "History-Load Cost-Aware Greedy Regional Index Selection")
        { name = "S4_GREEDY"; }
    else if (name == "Nominal Point Feed-Forward Regional Index Selection")
        { name = "S4_GREEDY"; method.feedforward = true; }
    else if (name == "Footprint-Calibrated Feed-Forward Regional Index Selection")
        { name = "S4_GREEDY"; method.feedforward = true; }
    else if (name == "Uncertainty-Aware Feed-Forward Regional Index Selection")
        { name = "S4_GREEDY"; method.feedforward = true; method.uncertainty = 1; }
    else if (name == "Risk-Constrained Uncertainty-Aware Feed-Forward Regional Index Selection" ||
             name == "Quantile-Cost-Penalized Feed-Forward Regional Index Selection")
        { name = "S4_GREEDY"; method.feedforward = true; method.uncertainty = 2; }
    else if (name == "CVaR-Risk-Constrained Feed-Forward Regional Index Selection")
        { name = "S4_GREEDY"; method.feedforward = true; method.uncertainty = 3; }
    else if (name == "Historical-Error-Feedback Robust Feed-Forward Regional Index Selection")
        { name = "S4_GREEDY"; method.feedforward = true; method.uncertainty = 1; }
    if (const auto mode = fixed_mode(name)) {
        method.fixed = mode;
        return method;
    }
    if (name == "S3_THRESHOLD") {
        method.policy = Policy::Threshold;
        return method;
    }
    if (name == "S4_GREEDY") {
        method.policy = Policy::Greedy;
        return method;
    }
    if (name == "UNIFORM_GRID" || name == "PR_QUADTREE" || name == "RSTAR_BOOST" ||
        name == "RSTAR_LIBSPATIALINDEX") {
        method.classic = true;
        method.name = std::string(name);
        return method;
    }
    throw std::invalid_argument("unknown method " + method.name);
}

struct Context {
    std::unique_ptr<Index> index;
    AdaptiveRegionalIndex* regional{};
    std::unique_ptr<Selector> selector;
};

Context create(const Method& method, const M3Trace& trace, int fine, int quad, int param,
               const SelectorConfig* config, bool profile,
               const std::shared_ptr<const ScenarioPrior>& prior = nullptr) {
    Context context;
    if (method.fixed) {
        context.index = std::make_unique<AdaptiveRegionalIndex>(
            trace.bounds, *method.fixed, fine, quad, profile, profile);
    } else if (method.policy) {
        if (!config) throw std::invalid_argument("policy method requires a selector config");
        if (method.feedforward && !prior)
            throw std::invalid_argument("feed-forward method requires a scenario prior table");
        context.index = std::make_unique<AdaptiveRegionalIndex>(
            trace.bounds, Mode::LocalQuad, fine, quad, profile, true);
        SelectorConfig local = *config;
        local.fusion = method.fusion;
        local.uncertainty = method.uncertainty;
        context.selector = std::make_unique<Selector>(*method.policy, local,
                                                      method.feedforward ? prior : nullptr);
    } else {
        context.index = make_index(method.name, param, trace.bounds);
    }
    context.regional = dynamic_cast<AdaptiveRegionalIndex*>(context.index.get());
    return context;
}

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

struct WindowMeta {
    std::uint64_t switches{};
    double policy_ms{};
};

struct Outcome {
    double build_ms{}, online_ms{}, query_ms{}, update_ms{}, switch_ms{}, policy_ms{};
    std::uint64_t query_count{}, update_count{}, delete_count{}, result_count{}, candidate_count{};
    std::uint64_t switches{}, decision_calls{};
    double p50_us{}, p95_us{}, p99_us{};
    double peak_rss_mb{}, prebuild_rss_mb{}, trace_load_ms{};
    unsigned final_scan{}, final_fine{}, final_quad{};
    std::size_t final_indexed_entities{};
    std::uint64_t decision_hash{1469598103934665603ULL};
    SelectorCounters counters{};
};

struct ProfileSink {
    std::ostream* regions{};
    std::ostream* decisions{};
    std::ostream* timeline{};
    std::vector<Decision> pending;
    std::vector<std::uint32_t> pending_step;
    std::vector<int> pending_window;
    WindowMeta meta{};
};

void write_region_rows(std::ostream& out, AdaptiveRegionalIndex& index, int window,
                       std::uint32_t step_end) {
    for (unsigned region = 0; region < AdaptiveRegionalIndex::region_count; ++region) {
        const auto& stats = index.region_stats(region);
        const auto& load = index.window_load(region);
        out << window << ',' << step_end << ',' << region << ',' << mode_name(index.mode(region))
            << ',' << index.entity_count(region) << ',' << load.queries << ',' << load.updates
            << ',' << std::fixed << std::setprecision(6) << load.coverage << ','
            << ms_of(Ns(stats.query_ns)) << ',' << ms_of(Ns(stats.local_update_ns)) << ','
            << ms_of(Ns(stats.build_ns)) << ',' << ms_of(Ns(stats.switch_ns)) << ','
            << stats.local_updates << ',' << stats.builds << ',' << stats.switches << '\n';
    }
}

void write_timeline_row(std::ostream& out, AdaptiveRegionalIndex& index, int window,
                        std::uint32_t step_end, const WindowMeta& meta) {
    unsigned scan = 0, fine = 0, quad = 0;
    std::size_t scan_entities = 0, fine_entities = 0, quad_entities = 0;
    for (unsigned region = 0; region < AdaptiveRegionalIndex::region_count; ++region) {
        const auto entities = index.entity_count(region);
        switch (index.mode(region)) {
            case Mode::Scan: ++scan; scan_entities += entities; break;
            case Mode::FineGrid: ++fine; fine_entities += entities; break;
            case Mode::LocalQuad: ++quad; quad_entities += entities; break;
        }
    }
    out << window << ',' << step_end << ',' << scan << ',' << fine << ',' << quad << ','
        << scan_entities << ',' << fine_entities << ',' << quad_entities << ','
        << meta.switches << ',' << std::fixed << std::setprecision(6) << meta.policy_ms << '\n';
}

Outcome replay(Context& context, const M3Trace& trace, ProfileSink* sink) {
    Outcome outcome;
    std::vector<Ns> latency;
    latency.reserve(trace.query_count);

    const double prebuild = peak_mib();
    const auto build_start = Clock::now();
    Id id = 0;
    for (const Point& point : trace.points) context.index->set(++id, point);
    const auto build_end = Clock::now();
    outcome.build_ms = ms_of(build_end - build_start);
    outcome.prebuild_rss_mb = prebuild;
    if (context.regional) context.regional->reset_window();

    Ns query_time{}, update_time{};
    const auto online_start = Clock::now();
    int window = 0;
    std::uint32_t last_boundary = 0;

    auto close_window = [&](std::uint32_t step_end) {
        if (!context.regional) return;
        if (sink && sink->regions) write_region_rows(*sink->regions, *context.regional, window, step_end);
        if (sink && sink->timeline)
            write_timeline_row(*sink->timeline, *context.regional, window, step_end, sink->meta);
        if (sink && sink->decisions) {
            for (std::size_t i = 0; i < sink->pending.size(); ++i) {
                const Decision& decision = sink->pending[i];
                const auto& stats = context.regional->region_stats(decision.region);
                const double observed = ms_of(Ns(stats.query_ns)) + ms_of(Ns(stats.local_update_ns));
                std::ostream& out = *sink->decisions;
                out << sink->pending_step[i] << ',' << decision.region << ',' << decision.entities
                    << ',' << decision.load.queries << ',' << decision.load.updates << ','
                    << mode_name(decision.old_mode) << ',' << mode_name(decision.target) << ','
                    << std::fixed << std::setprecision(6) << ms_of(Ns(decision.switch_ns)) << ',';
                if (decision.has_estimate) out << decision.estimate_ns * 1e-6;
                out << ',' << observed << ',' << outcome_name(decision.outcome) << ','
                    << (decision.load.queries ? decision.load.coverage / decision.load.queries : 0.0)
                    << ',' << decision.forecast.queries << ',' << decision.forecast.updates
                    << ',' << sink->pending_window[i] << '\n';
            }
            sink->pending.clear();
            sink->pending_step.clear();
            sink->pending_window.clear();
        }
    };

    for (const M3Event& event : trace.events) {
        if (event.type == 'P') {
            close_window(event.step);
            last_boundary = event.step;
            if (context.regional) {
                WindowMeta meta;
                if (context.selector) {
                    const auto decision_start = Clock::now();
                    auto decisions = context.selector->decide(*context.regional, event.step);
                    meta.policy_ms = ms_of(Clock::now() - decision_start);
                    outcome.policy_ms += meta.policy_ms;
                    ++outcome.decision_calls;
                    for (const Decision& decision : decisions) {
                        if (!decision.applied) continue;
                        ++meta.switches;
                        outcome.decision_hash = mix(outcome.decision_hash, event.step);
                        outcome.decision_hash = mix(outcome.decision_hash, decision.region);
                        outcome.decision_hash = mix(outcome.decision_hash, unsigned(decision.old_mode));
                        outcome.decision_hash = mix(outcome.decision_hash, unsigned(decision.target));
                        if (sink && sink->decisions) {
                            sink->pending.push_back(decision);
                            sink->pending_step.push_back(event.step);
                            sink->pending_window.push_back(window + 1);
                        }
                    }
                }
                if (sink) sink->meta = meta;
                context.regional->reset_window();
            }
            ++window;
            continue;
        }
        if (event.type == 'U') {
            const auto begin = Clock::now();
            context.index->set(event.id, event.point);
            const auto elapsed = Clock::now() - begin;
            update_time += elapsed;
            ++outcome.update_count;
            continue;
        }
        if (event.type == 'D') {
            const auto begin = Clock::now();
            context.index->erase(event.id);
            const auto elapsed = Clock::now() - begin;
            update_time += elapsed;
            ++outcome.delete_count;
            continue;
        }
        const auto begin = Clock::now();
        Answer answer = context.index->query(event.query);
        const auto elapsed = Clock::now() - begin;
        query_time += elapsed;
        latency.push_back(elapsed);
        ++outcome.query_count;
        outcome.result_count += answer.ids.size();
        outcome.candidate_count += answer.candidates;
    }
    const auto online = Clock::now() - online_start;
    close_window(trace.decision_points ? last_boundary + 50 : 0);
    outcome.online_ms = ms_of(online);
    outcome.query_ms = ms_of(query_time);
    outcome.update_ms = ms_of(update_time);
    outcome.peak_rss_mb = peak_mib();
    if (context.regional) {
        const auto times = context.regional->times();
        outcome.switch_ms = ms_of(Ns(times.switch_ns));
        outcome.switches = times.switches;
        for (unsigned region = 0; region < AdaptiveRegionalIndex::region_count; ++region) {
            switch (context.regional->mode(region)) {
                case Mode::Scan: ++outcome.final_scan; break;
                case Mode::FineGrid: ++outcome.final_fine; break;
                case Mode::LocalQuad: ++outcome.final_quad; break;
            }
            if (context.regional->mode(region) != Mode::Scan)
                outcome.final_indexed_entities += context.regional->entity_count(region);
        }
    }
    if (context.selector) outcome.counters = context.selector->counters();
    std::sort(latency.begin(), latency.end());
    auto percentile = [&](int pct) {
        const std::size_t index = (latency.size() * pct + 99) / 100;
        return latency.at(index - 1).count() / 1e3;
    };
    outcome.p50_us = percentile(50);
    outcome.p95_us = percentile(95);
    outcome.p99_us = percentile(99);
    return outcome;
}

int validate(const M3Trace& trace, const Method& method, int fine, int quad, int param,
             const SelectorConfig* config, const std::shared_ptr<const ScenarioPrior>& prior) {
    auto context = create(method, trace, fine, quad, param, config, false, prior);
    ArrayOracle oracle;
    Id id = 0;
    for (const Point& point : trace.points) {
        context.index->set(++id, point);
        oracle.set(id, point);
    }
    std::uint64_t queries = 0, updates = 0;
    for (std::size_t i = 0; i < trace.events.size(); ++i) {
        const M3Event& event = trace.events[i];
        if (event.type == 'P') {
            if (context.selector) context.selector->decide(*context.regional, event.step);
            if (context.regional) context.regional->reset_window();
        } else if (event.type == 'U') {
            context.index->set(event.id, event.point);
            oracle.set(event.id, event.point);
            ++updates;
        } else if (event.type == 'D') {
            context.index->erase(event.id);
            oracle.erase(event.id);
        } else {
            Answer answer = context.index->query(event.query);
            std::sort(answer.ids.begin(), answer.ids.end());
            auto expected = oracle.query(event.query);
            if (answer.ids != expected) {
                std::size_t missing = 0, extra = 0;
                std::size_t a = 0, b = 0;
                while (a < answer.ids.size() || b < expected.size()) {
                    if (b == expected.size() || (a < answer.ids.size() && answer.ids[a] < expected[b])) {
                        ++extra; ++a;
                    } else if (a == answer.ids.size() || expected[b] < answer.ids[a]) {
                        ++missing; ++b;
                    } else { ++a; ++b; }
                }
                throw std::runtime_error("oracle mismatch at event " + std::to_string(i) +
                                         ": missing " + std::to_string(missing) + " extra " +
                                         std::to_string(extra) + " index " + std::to_string(answer.ids.size()) +
                                         " expected " + std::to_string(expected.size()));
            }
            ++queries;
        }
    }
    std::cout << "queries=" << queries << " updates=" << updates
              << " entities=" << oracle.size() << " mismatches=0\n";
    return 0;
}

void print_outcome(const Outcome& outcome, const M3Trace& trace) {
    std::cout << std::fixed << std::setprecision(6)
              << "build_ms=" << outcome.build_ms
              << " online_ms=" << outcome.online_ms
              << " query_ms=" << outcome.query_ms
              << " update_ms=" << outcome.update_ms
              << " switch_ms=" << outcome.switch_ms
              << " policy_ms=" << outcome.policy_ms
              << " query_count=" << outcome.query_count
              << " update_count=" << outcome.update_count
              << " delete_count=" << outcome.delete_count
              << " result_count=" << outcome.result_count
              << " candidate_count=" << outcome.candidate_count
              << " p50_us=" << outcome.p50_us
              << " p95_us=" << outcome.p95_us
              << " p99_us=" << outcome.p99_us
              << " peak_rss_mb=" << outcome.peak_rss_mb
              << " prebuild_rss_mb=" << outcome.prebuild_rss_mb
              << " trace_load_ms=" << outcome.trace_load_ms
              << " trace_bytes=" << trace.bytes
              << " switches=" << outcome.switches
              << " decision_calls=" << outcome.decision_calls
              << " decision_hash=" << outcome.decision_hash
              << " final_scan=" << outcome.final_scan
              << " final_fine=" << outcome.final_fine
              << " final_quad=" << outcome.final_quad
              << " final_indexed_entities=" << outcome.final_indexed_entities
              << " promotions=" << outcome.counters.promotions
              << " demotions=" << outcome.counters.demotions
              << " retunes=" << outcome.counters.retunes
              << " blocked=" << outcome.counters.blocked
              << '\n';
}

std::optional<SelectorConfig> load_config(const char* path) {
    if (!path || std::string_view(path) == "NONE") return std::nullopt;
    return read_selector_config(path);
}

void probe_switches(const M3Trace& trace, int fine, int quad) {
    AdaptiveRegionalIndex index(trace.bounds, Mode::Scan, fine, quad, false, false);
    Id id = 0;
    for (const Point& point : trace.points) index.set(++id, point);
    std::vector<unsigned> regions;
    for (unsigned region = 0; region < AdaptiveRegionalIndex::region_count; ++region)
        if (index.entity_count(region) > 0) regions.push_back(region);
    std::stable_sort(regions.begin(), regions.end(), [&](unsigned a, unsigned b) {
        return index.entity_count(a) < index.entity_count(b);
    });
    const std::size_t wanted = std::min<std::size_t>(regions.size(), 36);
    std::vector<unsigned> sample;
    for (std::size_t i = 0; i < wanted; ++i)
        sample.push_back(wanted == 1 ? regions.back()
                                     : regions[(regions.size() - 1) * i / (wanted - 1)]);
    std::cout << "from,to,entities,seq,switch_ns\n";
    std::uint64_t seq = 0;
    const std::array<std::pair<Mode, Mode>, 6> pairs{{
        {Mode::Scan, Mode::FineGrid}, {Mode::FineGrid, Mode::LocalQuad},
        {Mode::LocalQuad, Mode::Scan}, {Mode::Scan, Mode::LocalQuad},
        {Mode::LocalQuad, Mode::FineGrid}, {Mode::FineGrid, Mode::Scan}}};
    for (unsigned region : sample) {
        for (const auto& [from, to] : pairs) {
            if (index.mode(region) != from) index.switch_mode(region, from);
            const auto before = index.times().switch_ns;
            index.switch_mode(region, to);
            const auto delta = index.times().switch_ns - before;
            std::cout << mode_name(from) << ',' << mode_name(to) << ',' << index.entity_count(region)
                      << ',' << ++seq << ',' << delta << '\n';
        }
        if (index.mode(region) != Mode::Scan) index.switch_mode(region, Mode::Scan);
    }
}

// Dump the load the index itself observes per region and window. This is the
// semantic reference for the offline load labels and the nominal derivation.
void probe_loads(const M3Trace& trace, int fine, int quad, const std::string& prefix) {
    AdaptiveRegionalIndex index(trace.bounds, Mode::Scan, fine, quad, false, true);
    Id id = 0;
    for (const Point& point : trace.points) index.set(++id, point);
    index.reset_window();
    std::ofstream out(prefix + "_loads_window.csv");
    out << "window,region,query_count,update_count,coverage,entity_count\n";
    int window = 0;
    auto dump = [&]() {
        for (unsigned region = 0; region < AdaptiveRegionalIndex::region_count; ++region) {
            const auto& load = index.window_load(region);
            out << window << ',' << region << ',' << load.queries << ',' << load.updates << ','
                << std::fixed << std::setprecision(6) << load.coverage << ','
                << index.entity_count(region) << '\n';
        }
        ++window;
    };
    for (const M3Event& event : trace.events) {
        if (event.type == 'P') {
            dump();
            index.reset_window();
        } else if (event.type == 'U') {
            index.set(event.id, event.point);
        } else if (event.type == 'D') {
            index.erase(event.id);
        } else {
            index.query(event.query);
        }
    }
    dump();
}

void run_profile(Context& context, const M3Trace& trace, const std::string& prefix,
                 double trace_load_ms) {
    std::ofstream regions(prefix + "_regions.csv");
    std::ofstream decisions(prefix + "_decisions.csv");
    std::ofstream timeline(prefix + "_timeline.csv");
    if (!regions || !decisions || !timeline)
        throw std::runtime_error("cannot open profile outputs for " + prefix);
    regions << "window_index,step_end,region_id,mode,entities,query_count,update_count,coverage,"
               "query_ms,local_update_ms,local_build_ms,switch_ms,local_updates,local_builds,local_switches\n";
    decisions << "step,region_id,entity_count,query_count,update_count,old_mode,selected_mode,"
                 "switch_cost_ms,estimated_cost,observed_cost,status,coverage,forecast_query_count,"
                 "forecast_update_count,window_index\n";
    timeline << "window_index,step_end,scan_regions,fine_regions,quad_regions,scan_entities,"
                "fine_entities,quad_entities,switches,policy_ms\n";
    ProfileSink sink;
    sink.regions = &regions;
    sink.decisions = &decisions;
    sink.timeline = &timeline;
    Outcome outcome = replay(context, trace, &sink);
    outcome.trace_load_ms = trace_load_ms;
    print_outcome(outcome, trace);
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("usage: spatial_m3 measure|validate|profile|switchprobe ...");
        const std::string_view mode = argv[1];
        if (mode == "measure" || mode == "validate") {
            if (argc != 9 && argc != 10)
                throw std::invalid_argument("usage: spatial_m3 " + std::string(mode) +
                                            " METHOD FINE QUAD PARAM CPU TRACE CONFIG|NONE [PRIOR|NONE]");
            const auto method = parse_method(argv[2]);
            const int fine = std::stoi(argv[3]), quad = std::stoi(argv[4]);
            const int param = std::stoi(argv[5]);
            pin(std::stoi(argv[6]));
            const auto load_start = Clock::now();
            const auto trace = read_m3_trace(argv[7]);
            const auto load_ms = ms_of(Clock::now() - load_start);
            const auto config = load_config(argv[8]);
            std::shared_ptr<const ScenarioPrior> prior;
            if (argc == 10 && std::string_view(argv[9]) != "NONE")
                prior = std::make_shared<const ScenarioPrior>(ScenarioPrior::read(argv[9]));
            if (mode == "validate") return validate(trace, method, fine, quad, param,
                                                    config ? &*config : nullptr, prior);
            auto context = create(method, trace, fine, quad, param,
                                  config ? &*config : nullptr, false, prior);
            auto outcome = replay(context, trace, nullptr);
            outcome.trace_load_ms = load_ms;
            print_outcome(outcome, trace);
            return 0;
        }
        if (mode == "profile") {
            if (argc != 10 && argc != 11)
                throw std::invalid_argument("usage: spatial_m3 profile METHOD FINE QUAD PARAM CPU TRACE CONFIG|NONE PREFIX [PRIOR|NONE]");
            const auto method = parse_method(argv[2]);
            const int fine = std::stoi(argv[3]), quad = std::stoi(argv[4]);
            const int param = std::stoi(argv[5]);
            pin(std::stoi(argv[6]));
            const auto load_start = Clock::now();
            const auto trace = read_m3_trace(argv[7]);
            const auto load_ms = ms_of(Clock::now() - load_start);
            const auto config = load_config(argv[8]);
            if (method.policy && !config) throw std::invalid_argument("policy profile needs a config");
            std::shared_ptr<const ScenarioPrior> prior;
            if (argc == 11 && std::string_view(argv[10]) != "NONE")
                prior = std::make_shared<const ScenarioPrior>(ScenarioPrior::read(argv[10]));
            auto context = create(method, trace, fine, quad, param,
                                  config ? &*config : nullptr, true, prior);
            run_profile(context, trace, argv[9], load_ms);
            return 0;
        }
        if (mode == "partition") {
            if (argc != 8) throw std::invalid_argument("usage: spatial_m3 partition FINE QUAD CPU TRACE PLAN VALIDATE");
            pin(std::stoi(argv[4]));
            const auto trace = read_m3_trace(argv[5]);
            return run_partition(trace, std::stoi(argv[2]), std::stoi(argv[3]), argv[6],
                                 std::stoi(argv[7]) != 0);
        }
        if (mode == "replayplan") {
            if (argc != 7) throw std::invalid_argument("usage: spatial_m3 replayplan FINE QUAD CPU TRACE PLAN");
            pin(std::stoi(argv[4]));
            const auto trace = read_m3_trace(argv[5]);
            std::ifstream plan_in(argv[6]);
            if (!plan_in) throw std::runtime_error("cannot open action plan");
            std::vector<std::vector<int>> plan;
            std::string line;
            while (std::getline(plan_in, line)) {
                std::vector<int> row;
                std::stringstream stream(line);
                std::string token;
                while (std::getline(stream, token, ',')) {
                    if (!token.empty()) row.push_back(std::stoi(token));
                }
                if (!row.empty()) plan.push_back(std::move(row));
            }
            return run_replayplan(trace, std::stoi(argv[2]), std::stoi(argv[3]), plan);
        }
        if (mode == "rlserve") {
            if (argc != 7) throw std::invalid_argument("usage: spatial_m3 rlserve FINE QUAD CPU TRACE VERIFY");
            pin(std::stoi(argv[4]));
            const auto trace = read_m3_trace(argv[5]);
            return run_rlserve(trace, std::stoi(argv[2]), std::stoi(argv[3]),
                               std::stoi(argv[6]) != 0);
        }
        if (mode == "loadprobe") {
            if (argc != 7) throw std::invalid_argument("usage: spatial_m3 loadprobe FINE QUAD CPU TRACE PREFIX");
            pin(std::stoi(argv[4]));
            const auto trace = read_m3_trace(argv[5]);
            probe_loads(trace, std::stoi(argv[2]), std::stoi(argv[3]), argv[6]);
            return 0;
        }
        if (mode == "switchprobe") {
            if (argc != 6) throw std::invalid_argument("usage: spatial_m3 switchprobe FINE QUAD CPU TRACE");
            pin(std::stoi(argv[4]));
            const auto trace = read_m3_trace(argv[5]);
            probe_switches(trace, std::stoi(argv[2]), std::stoi(argv[3]));
            return 0;
        }
        throw std::invalid_argument("unknown mode " + std::string(mode));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
