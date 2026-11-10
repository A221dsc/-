#include "core/clock.hpp"
#include "indexes/regional.hpp"
#include "validation/oracle.hpp"
#include "workload/replay.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#ifdef _WIN32
#include <psapi.h>
#else
#include <sched.h>
#include <sys/resource.h>
#endif

using namespace spatial;
void pin(int cpu) {
#ifdef _WIN32
    if (cpu < 0 || cpu >= int(sizeof(DWORD_PTR)*8) ||
        !SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << cpu))
        throw std::runtime_error("CPU affinity failed");
#else
    if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::runtime_error("invalid CPU");
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask) != 0)
        throw std::runtime_error("CPU affinity failed");
#endif
}
double peak_mib() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS memory{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory)))
        throw std::runtime_error("PeakWorkingSetSize unavailable");
    return memory.PeakWorkingSetSize/1048576.0;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF,&usage) != 0) throw std::runtime_error("RSS unavailable");
    return usage.ru_maxrss/1024.0;
#endif
}
double ms(Ns duration) { return duration.count()/1e6; }

std::unique_ptr<Index> create(std::string_view name, int param, Box box, bool profile = false) {
    if (const auto mode = fixed_mode(name))
        return std::make_unique<AdaptiveRegionalIndex>(
            box,*mode,*mode == Mode::FineGrid ? param : 250,
            *mode == Mode::LocalQuad ? param : 16,profile);
    return make_index(name,param,box);
}

void apply_switch(AdaptiveRegionalIndex* regional, const Event& e) {
    if (regional) regional->switch_mode(static_cast<unsigned>(e.id),Mode(int(e.point.x)));
}

struct RegionRow {
    int phase;
    unsigned region;
    std::size_t entities;
    Mode mode;
    RegionalStats stats;
};

void snapshot(AdaptiveRegionalIndex& index, int phase, std::vector<RegionRow>& rows) {
    for (unsigned r = 0; r < AdaptiveRegionalIndex::region_count; ++r)
        rows.push_back({phase,r,index.entity_count(r),index.mode(r),index.region_stats(r)});
    index.reset_window();
}

void validate(const Trace& trace, std::string_view name, int param) {
    auto index = create(name,param,trace.bounds);
    auto* regional = dynamic_cast<AdaptiveRegionalIndex*>(index.get());
    Oracle oracle;
    std::unordered_set<Id> active;
    for (auto [id,p] : trace.initial) {
        if (!active.insert(id).second) throw std::runtime_error("duplicate initial ID");
        index->set(id,p);
        oracle.set(id,p);
    }
    std::size_t queries = 0, peak = active.size();
    for (std::size_t i = 0; i < trace.events.size(); ++i) {
        const auto& e = trace.events[i];
        if (e.type == 'P') continue;
        if (e.type == 'S') {
            apply_switch(regional,e);
        } else if (e.type == 'R' || e.type == 'C') {
            auto a = index->query(e.query);
            std::sort(a.ids.begin(),a.ids.end());
            if (a.ids != oracle.query(e.query))
                throw std::runtime_error("oracle mismatch at event " + std::to_string(i));
            ++queries;
        } else if (e.type == 'D') {
            if (!active.erase(e.id)) throw std::runtime_error("missing delete ID");
            index->erase(e.id);
            oracle.erase(e.id);
        } else {
            const bool inserted = active.insert(e.id).second;
            if (inserted != (e.type == 'I')) throw std::runtime_error("invalid insert/update");
            index->set(e.id,e.point);
            oracle.set(e.id,e.point);
            peak = std::max(peak,active.size());
        }
    }
    if (queries == 0) throw std::runtime_error("trace has no queries");
    std::cout << queries << ',' << peak << '\n';
}

void measure(const Trace& trace, std::string_view name, int param, Ns load,
             bool extended, bool profile) {
    std::vector<Ns> latency;
    latency.reserve(trace.events.size());
    std::uint64_t updates = 0, mutations = 0, candidates = 0, visits = 0, hits = 0;
    bool has_visits = true;
    Ns update_time{}, mutation_time{}, query_time{};
    const double prebuild_peak = peak_mib();
    const auto build_start = Clock::now();
    auto index = create(name,param,trace.bounds,profile);
    auto* regional = dynamic_cast<AdaptiveRegionalIndex*>(index.get());
    if (profile && !regional) throw std::invalid_argument("regional profile requires S0/S1/S2");
    for (auto [id,p] : trace.initial) index->set(id,p);
    const auto build_end = Clock::now();
    const auto initial_times = regional ? regional->times() : RegionalTimes{};
    std::vector<RegionRow> regions;
    if (profile) snapshot(*regional,-1,regions);
    int phases = 0;
    bool pending_window = false;
    const auto online_start = Clock::now();
    for (const auto& e : trace.events) {
        if (e.type == 'P') {
            if (profile) snapshot(*regional,phases++,regions);
            pending_window = false;
            continue;
        }
        pending_window = true;
        if (e.type == 'S') {
            apply_switch(regional,e);
        } else if (e.type == 'R' || e.type == 'C') {
            const auto begin = Clock::now();
            {
                auto a = index->query(e.query);
                candidates += a.candidates;
                visits += a.visits.value_or(0);
                has_visits &= a.visits.has_value();
                hits += a.ids.size();
            }
            const auto elapsed = Clock::now() - begin;
            query_time += elapsed;
            latency.push_back(elapsed);
        } else {
            const auto begin = Clock::now();
            if (e.type == 'D') index->erase(e.id);
            else index->set(e.id,e.point);
            const auto elapsed = Clock::now()-begin;
            if (e.type == 'U') {
                update_time += elapsed;
                ++updates;
            } else {
                mutation_time += elapsed;
                ++mutations;
            }
        }
    }
    const auto online = Clock::now()-online_start;
    const double peak = peak_mib();
    if (profile && pending_window) snapshot(*regional,phases,regions);
    std::sort(latency.begin(),latency.end());
    auto us = [&](std::size_t pct) {
        return latency.at((latency.size()*pct+99)/100-1).count()/1e3;
    };
    std::cout << std::fixed << std::setprecision(8)
              << ms(build_end-build_start) << ',' << ms(online) << ','
              << ms(query_time) << ',' << ms(update_time+mutation_time) << ','
              << ms(update_time) << ',' << ms(mutation_time) << ','
              << us(50) << ',' << us(95) << ',' << us(99) << ','
              << (update_time.count() ? 1e9*updates/update_time.count() : 0.0) << ','
              << updates << ',' << mutations << ',' << latency.size() << ','
              << candidates << ',';
    if (has_visits) std::cout << visits;
    std::cout << ',' << hits << ',' << peak << ',' << prebuild_peak << ',' << ms(load);
    if (extended) {
        const auto times = regional ? regional->times() : RegionalTimes{};
        std::cout << ',';
        if (profile) std::cout << ms(Ns(times.base_ns-initial_times.base_ns));
        std::cout << ',';
        if (profile || !regional) std::cout << ms(Ns(times.local_ns-initial_times.local_ns));
        std::cout << ',' << ms(Ns(times.switch_ns-initial_times.switch_ns)) << ',';
        if (profile || !regional) std::cout << times.local_updates-initial_times.local_updates;
        std::cout << ',' << times.switches-initial_times.switches;
    }
    std::cout << '\n';
    for (const auto& row : regions) {
        const auto& s = row.stats;
        std::cout << row.phase << ',' << row.region << ',' << row.entities << ',' << mode_name(row.mode)
                  << ',' << s.queries << ',' << s.object_updates << ',' << s.local_updates
                  << ',' << s.builds << ',' << s.switches << ',' << ms(Ns(s.query_ns))
                  << ',' << ms(Ns(s.local_update_ns)) << ',' << ms(Ns(s.build_ns))
                  << ',' << ms(Ns(s.switch_ns)) << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::invalid_argument("usage: spatial_bench validate|measure|measure_m2|profile_m2 NAME PARAM CPU TRACE");
        const std::string_view mode = argv[1], name = argv[2];
        const int param = std::stoi(argv[3]);
        pin(std::stoi(argv[4]));
        auto begin = Clock::now();
        auto trace = read_trace(argv[5]);
        const auto load = Clock::now()-begin;
        if (mode == "validate") validate(trace,name,param);
        else if (mode == "measure" || mode == "measure_m2" || mode == "profile_m2")
            measure(trace,name,param,load,mode != "measure",mode == "profile_m2");
        else throw std::invalid_argument("unknown mode");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
