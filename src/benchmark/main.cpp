#include "core/index.hpp"
#include "validation/oracle.hpp"
#include "workload/replay.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sched.h>
#include <sys/resource.h>
#endif

using namespace spatial;
using Ns = std::chrono::nanoseconds;
#ifdef _WIN32
struct Clock {
    static Ns now() {
        LARGE_INTEGER counter{};
        QueryPerformanceCounter(&counter);
        static const auto epoch = counter.QuadPart;
        static const double ns_per_tick = [] {
            LARGE_INTEGER frequency{};
            if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0)
                throw std::runtime_error("QueryPerformanceCounter frequency unavailable");
            return 1e9 / frequency.QuadPart;
        }();
        return Ns(static_cast<Ns::rep>((counter.QuadPart - epoch) * ns_per_tick));
    }
};
#else
using Clock = std::chrono::steady_clock;
#endif

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

void validate(const Trace& trace, std::string_view name, int param) {
    auto index = make_index(name,param,trace.bounds);
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
        if (e.type == 'R' || e.type == 'C') {
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

void measure(const Trace& trace, std::string_view name, int param, Ns load) {
    std::vector<Ns> latency;
    latency.reserve(trace.events.size());
    std::uint64_t updates = 0, mutations = 0, candidates = 0, visits = 0, hits = 0;
    bool has_visits = true;
    Ns update_time{}, mutation_time{}, query_time{};
    const double prebuild_peak = peak_mib();
    const auto build_start = Clock::now();
    auto index = make_index(name,param,trace.bounds);
    for (auto [id,p] : trace.initial) index->set(id,p);
    const auto online_start = Clock::now();
    for (const auto& e : trace.events) {
        if (e.type == 'R' || e.type == 'C') {
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
    std::sort(latency.begin(),latency.end());
    auto us = [&](std::size_t pct) {
        return latency.at((latency.size()*pct+99)/100-1).count()/1e3;
    };
    std::cout << std::fixed << std::setprecision(8)
              << ms(online_start-build_start) << ',' << ms(online) << ','
              << ms(query_time) << ',' << ms(update_time+mutation_time) << ','
              << ms(update_time) << ',' << ms(mutation_time) << ','
              << us(50) << ',' << us(95) << ',' << us(99) << ','
              << (update_time.count() ? 1e9*updates/update_time.count() : 0.0) << ','
              << updates << ',' << mutations << ',' << latency.size() << ','
              << candidates << ',';
    if (has_visits) std::cout << visits;
    std::cout << ',' << hits << ',' << peak << ',' << prebuild_peak << ',' << ms(load) << '\n';
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::invalid_argument("usage: spatial_bench validate|measure NAME PARAM CPU TRACE");
        const std::string_view mode = argv[1], name = argv[2];
        const int param = std::stoi(argv[3]);
        pin(std::stoi(argv[4]));
        auto begin = Clock::now();
        auto trace = read_trace(argv[5]);
        const auto load = Clock::now()-begin;
        if (mode == "validate") validate(trace,name,param);
        else if (mode == "measure") measure(trace,name,param,load);
        else throw std::invalid_argument("unknown mode");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
