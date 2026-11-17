#include "h3/h3_p0_v2.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using namespace spatial;
using namespace spatial::h3;
using namespace spatial::h3::p0v2;

namespace {

struct World {
    std::vector<std::optional<Point>> initial;
    std::vector<Frame> frames;
};

constexpr Point alpha{1100, 1250};
constexpr Point bravo{2950, 2750};
constexpr Point charlie{3050, 950};
constexpr double width = 4096;

std::uint64_t micros(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

std::uint64_t rss_peak() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return counters.PeakWorkingSetSize;
    return 0;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
        return usage.ru_maxrss;
#else
        return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
    }
    return 0;
#endif
}

class ArrivalTimer {
public:
    explicit ArrivalTimer(bool active) {
#if defined(_WIN32)
        if (active) {
            timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0x2, TIMER_ALL_ACCESS);
            if (!timer_) throw std::runtime_error("high-resolution arrival timer unavailable");
        }
#else
        (void)active;
#endif
    }

    ~ArrivalTimer() {
#if defined(_WIN32)
        if (timer_) CloseHandle(timer_);
#endif
    }

    void wait(Clock::time_point target) const {
#if defined(_WIN32)
        const auto remaining = (target - Clock::now()).count();
        if (remaining <= 0) return;
        LARGE_INTEGER due;
        due.QuadPart = -((remaining + 99) / 100);
        if (!SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE) ||
            WaitForSingleObject(timer_, INFINITE) != WAIT_OBJECT_0)
            throw std::runtime_error("high-resolution arrival wait failed");
        while (Clock::now() < target) std::this_thread::yield();
#else
        std::this_thread::sleep_until(target);
#endif
    }

    ArrivalTimer(const ArrivalTimer&) = delete;
    ArrivalTimer& operator=(const ArrivalTimer&) = delete;

private:
#if defined(_WIN32)
    HANDLE timer_{};
#endif
};

World generate(char scenario, unsigned seed, int count, int frames, int warmup) {
    std::mt19937 rng(seed);
    // Use the engine's integer output directly to make replay independent of
    // standard library distribution implementations.
    const auto unit = [&] { return static_cast<double>(rng()) / 4294967296.0; };
    const auto bounded = [&](int n) { return static_cast<int>(unit() * n); };
    const auto uniform = [&] { return Point{unit() * width, unit() * width}; };
    const auto clip = [](Point p) {
        return Point{std::clamp(p.x, 0.0, width), std::clamp(p.y, 0.0, width)};
    };
    const auto around = [&](Point p, double spread) {
        return clip({p.x + spread * (unit() + unit() - 1),
                     p.y + spread * (unit() + unit() - 1)});
    };

    World world;
    const auto base = [&](Slot slot) {
        switch (slot % 5) {
            case 0: case 1: return alpha;
            case 2: return bravo;
            case 3: return charlie;
            default: return uniform();
        }
    };
    for (int i = 0; i < count; ++i) world.initial.emplace_back(around(base(i), 750));

    auto reference = world.initial;
    std::vector<int> used(count, -1);
    for (int tick = 0; tick < frames + warmup; ++tick) {
        Frame frame;
        const double elapsed = static_cast<double>(std::max(0, tick - warmup)) / frames;
        const double shift = std::clamp(3 * elapsed - 1, 0.0, 1.0);
        const Point front{std::lerp(alpha.x, bravo.x, shift), std::lerp(alpha.y, bravo.y, shift)};
        for (int i = 0; i < 64; ++i) {
            const double pick = unit();
            Point center;
            if (scenario == 'C') center = pick < 0.80 ? around(front, 380) : uniform();
            else if (scenario == 'B') center = pick < 0.55 ? around(alpha, 600) : uniform();
            else if (pick < 0.65) center = around(alpha, 440);
            else if (pick < 0.82) center = around(bravo, 520);
            else center = uniform();
            const double radius = 110 + 240 * unit();
            frame.queries.push_back(i % 2 == 0 ? circle(center, radius)
                : Query{{center.x - radius, center.y - radius * .75,
                         center.x + radius, center.y + radius * .75}, false, {}, 0});
        }
        for (int i = 0; i < 128; ++i) {
            Slot slot;
            do {
                const double choice = unit();
                int group = bounded(5);
                if (scenario == 'A' && choice < .70) group = choice < .45 ? bounded(2) : 3;
                if (scenario == 'C' && choice < .80) group = unit() < shift ? 2 : bounded(2);
                slot = static_cast<Slot>(bounded(count / 5) * 5 + group);
            } while (used[slot] == tick);
            used[slot] = tick;
            const auto old = reference[slot];
            std::optional<Point> next;
            if (!old) next = around(base(slot), 500);
            else if (unit() < .015) next = std::nullopt;
            else if (scenario == 'B' && unit() < .84) {
                next = std::hypot(old->x - alpha.x, old->y - alpha.y) < 1000
                    ? around(bravo, 850) : around(alpha, 850);
            } else {
                next = around(*old, scenario == 'B' ? 220 : 140);
            }
            reference[slot] = next;
            frame.updates.push_back({slot, next});
        }
        world.frames.push_back(std::move(frame));
    }
    return world;
}

Mode parse_mode(std::string_view value) {
    for (const auto mode : {Mode::Serial, Mode::Lock, Mode::Static, Mode::Matched, Mode::Partition})
        if (value == name(mode)) return mode;
    throw std::invalid_argument("unknown method");
}

std::array<std::uint64_t, 3> quantiles(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    return {values[(values.size() * 50 - 1) / 100],
            values[(values.size() * 95 - 1) / 100],
            values[(values.size() * 99 - 1) / 100]};
}

struct Signature {
    std::uint64_t hash{14695981039346656037ULL};
    void add(std::uint64_t v) { hash = (hash ^ v) * 1099511628211ULL; }
};

struct Totals {
    std::uint64_t cells{}, lookups{}, candidates{}, hits{}, owner_fanout{}, region_fanout{}, routed{};
    std::uint64_t prepare{}, cover_cpu{}, route_cpu{}, dispatch{}, execute{}, merge{}, barrier{};
    std::uint64_t late_ticks{}, missed_ticks{}, deadline_misses{}, deadline_pending{};
    std::array<std::uint64_t, 3> stages{};
    std::vector<std::uint64_t> query_latencies, update_latencies;
    Signature signature;
};

void write_worker(std::ostream& out, const WorkerStats& s, std::uint64_t execute_ns) {
    out << ',' << s.query_tasks << ',' << s.update_tasks << ',' << s.cells << ',' << s.candidates
        << ',' << s.read_locks << ',' << s.write_locks << ',' << s.read_misses << ',' << s.write_misses
        << ',' << s.read_wait_ns << ',' << s.write_wait_ns << ',' << s.migration_wait_ns
        << ',' << s.queue_wait_ns << ',' << s.busy_ns << ',' << execute_ns - s.busy_ns
        << ',' << s.query_ns << ',' << s.update_ns << ',' << s.migration_ns
        << ',' << s.same_cell << ',' << s.cross_cell << ',' << s.intrapartition
        << ',' << s.cross_partition << ',' << s.cross_worker << ',' << s.insertions << ','
        << s.deletions << ',' << s.snapshot_cells << ',' << s.snapshot_entries;
}

void verify_oracle(std::span<const std::optional<Point>> ref, const Query& q,
                   const std::vector<Id>& hits) {
    std::size_t hit{};
    for (std::size_t slot = 0; slot < ref.size(); ++slot) {
        if (!ref[slot] || !q.matches(*ref[slot])) continue;
        if (hit >= hits.size() || static_cast<Id>(slot) != hits[hit++])
            throw std::logic_error("benchmark trace oracle mismatch");
    }
    if (hit != hits.size()) throw std::logic_error("benchmark trace extra hits");
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 11)
        throw std::invalid_argument(
            "usage: bench A|B|C METHOD WORKERS SEED ENTITIES TICKS WARMUP PERIOD_US PROFILE_FILE|- VERIFY");
    const std::string scenario = argv[1];
    const auto mode = parse_mode(argv[2]);
    const int workers = std::stoi(argv[3]);
    const auto seed = static_cast<unsigned>(std::stoul(argv[4]));
    const int entities = std::stoi(argv[5]);
    const int ticks = std::stoi(argv[6]);
    const int warmup = std::stoi(argv[7]);
    const auto period = std::chrono::microseconds{std::stol(argv[8])};
    const bool profile = std::string_view(argv[9]) != "-";
    const bool verify = std::string_view(argv[10]) == "1";
    if (scenario.size() != 1 || scenario.front() < 'A' || scenario.front() > 'C' ||
        entities < 640 || entities % 5 != 0 || ticks < 3 || warmup < 0 || period.count() < 0)
        throw std::invalid_argument("invalid benchmark dimensions");

    CellCover cover(PartitionConfig{});
    auto world = generate(scenario.front(), seed, entities, ticks, warmup);
    auto state = world.initial;
    const auto construction = Clock::now();
    Replay replay(cover, world.initial, mode, workers, profile);
    const auto initial_memory = replay.memory();
    const auto construction_us = micros(construction, Clock::now());
    for (int i = 0; i < warmup; ++i) {
        replay.run(world.frames[i], Clock::now());
        for (const auto& u : world.frames[i].updates) state[u.slot] = u.point;
    }
    std::vector<FrameResult> results;
    std::vector<std::uint64_t> admit_lags;
    results.reserve(ticks);
    admit_lags.reserve(ticks);
    const ArrivalTimer arrival_timer(period.count() > 0);
    const auto begin = Clock::now();
    for (int i = 0; i < ticks; ++i) {
        const auto ready = begin + i * period;
        if (period.count() > 0) arrival_timer.wait(ready);
        const auto admitted = Clock::now();
        admit_lags.push_back(period.count() > 0 ? micros(ready, admitted) : 0);
        results.push_back(replay.run(world.frames[warmup + i],
                                     period.count() > 0 ? ready : admitted));
    }
    const auto wall_us = micros(begin, Clock::now());
    const auto peak_rss = rss_peak();
    const auto final_memory = replay.memory();

    std::ofstream phases;
    if (profile) {
        phases.open(argv[9]);
        phases.exceptions(std::ios::badbit | std::ios::failbit);
        phases << "tick,worker,frame_ns,prepare_ns,cover_cpu_ns,route_cpu_ns,dispatch_ns,execute_ns,"
               "merge_ns,barrier_tail_ns,admit_lag_us,query_tasks,update_tasks,cells,candidates,"
               "read_locks,write_locks,read_misses,write_misses,read_wait_ns,write_wait_ns,"
               "migration_wait_ns,queue_wait_ns,busy_ns,idle_window_ns,query_ns,update_ns,migration_ns,"
               "same_cell,cross_cell,intrapartition,cross_partition,cross_worker,insertions,deletions,"
               "snapshot_cells,snapshot_entries\n";
    }
    Totals total;
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(period).count();
    for (int i = 0; i < ticks; ++i) {
        auto& answer = results[i];
        const auto& frame = world.frames[warmup + i];
        const auto& s = answer.stats;
        total.owner_fanout += s.owner_fanout;
        total.region_fanout += s.region_fanout;
        total.routed += s.routed_jobs;
        total.prepare += s.prepare_ns;
        total.cover_cpu += s.cover_cpu_ns;
        total.route_cpu += s.route_cpu_ns;
        total.dispatch += s.dispatch_ns;
        total.execute += s.execute_ns;
        total.merge += s.merge_ns;
        total.barrier += s.barrier_tail_ns;
        total.stages[i * 3 / ticks] += s.frame_ns;
        total.update_latencies.insert(total.update_latencies.end(), frame.updates.size(), s.update_latency_ns);
        if (period_ns > 0) {
            const auto deadline = static_cast<std::uint64_t>(period_ns);
            total.late_ticks += admit_lags[i] >= static_cast<std::uint64_t>(period.count());
            const auto last_response = std::max(s.update_latency_ns, answer.query_latency_ns.back());
            total.missed_ticks += last_response > deadline;
            total.deadline_misses += s.update_latency_ns > deadline ? frame.updates.size() : 0;
            const auto horizon = deadline * (ticks - i);
            total.deadline_pending += s.update_latency_ns > horizon ? frame.updates.size() : 0;
            for (const auto latency : answer.query_latency_ns) {
                total.deadline_misses += latency > deadline;
                total.deadline_pending += latency > horizon;
            }
        }

        for (std::size_t q = 0; q < answer.answers.size(); ++q) {
            auto& result = answer.answers[q];
            auto& ids = result.ids;
            std::sort(ids.begin(), ids.end());
            if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
                throw std::logic_error("duplicate answer in benchmark");
            if (verify) verify_oracle(state, frame.queries[q], ids);
            total.signature.add(ids.size());
            for (const auto id : ids) total.signature.add(id);
            total.cells += result.stats.candidate_cells;
            total.lookups += result.stats.bucket_lookups;
            total.candidates += result.stats.candidates;
            total.hits += ids.size();
        }
        for (const auto& u : frame.updates) state[u.slot] = u.point;
        total.query_latencies.insert(total.query_latencies.end(), answer.query_latency_ns.begin(),
                                     answer.query_latency_ns.end());
        if (profile) {
            for (std::size_t w = 0; w < s.workers.size(); ++w) {
                phases << i << ',' << w << ',' << s.frame_ns << ',' << s.prepare_ns << ','
                       << s.cover_cpu_ns << ',' << s.route_cpu_ns << ',' << s.dispatch_ns << ','
                       << s.execute_ns << ',' << s.merge_ns << ',' << s.barrier_tail_ns << ',' << admit_lags[i];
                write_worker(phases, s.workers[w], s.execute_ns);
                phases << '\n';
            }
        }
    }
    replay.validate(state);
    const auto q = quantiles(std::move(total.query_latencies));
    const auto u = quantiles(std::move(total.update_latencies));
    const auto n_queries = 64 * ticks;
    const auto n_updates = 128 * ticks;
    std::cout << "scenario,method,workers,seed,entities,ticks,warmup,period_us,profile,oracle,queries,updates,"
              << "wall_us,query_per_s,update_per_s,q50_ns,q95_ns,q99_ns,u50_ns,u95_ns,u99_ns,"
              << "candidate_cells,bucket_lookups,candidates,hits,signature,owner_fanout,region_fanout,routed_jobs,"
              << "prepare_ns,cover_cpu_ns,route_cpu_ns,dispatch_ns,execute_ns,merge_ns,barrier_tail_ns,"
              << "stage_a_ns,stage_transition_ns,stage_b_ns,late_ticks,missed_ticks,deadline_misses,pending_at_horizon,"
              << "cover_cells,regions,construct_us,mapping_ns,initial_index_bytes,index_bytes,snapshot_bytes,"
              << "mapping_bytes,peak_rss_bytes\n"
              << "W8-" << scenario << ',' << name(mode) << ',' << workers << ',' << seed << ','
              << entities << ',' << ticks << ',' << warmup << ',' << period.count() << ',' << profile << ','
              << verify << ',' << n_queries << ',' << n_updates << ',' << wall_us << ','
              << std::setprecision(10) << 1e6 * n_queries / wall_us << ',' << 1e6 * n_updates / wall_us << ','
              << q[0] << ',' << q[1] << ',' << q[2] << ',' << u[0] << ',' << u[1] << ',' << u[2] << ','
              << total.cells << ',' << total.lookups << ',' << total.candidates << ',' << total.hits << ','
              << total.signature.hash << ',' << total.owner_fanout << ',' << total.region_fanout << ','
              << total.routed << ',' << total.prepare << ',' << total.cover_cpu << ',' << total.route_cpu << ','
              << total.dispatch << ',' << total.execute << ',' << total.merge << ',' << total.barrier << ','
              << total.stages[0] << ',' << total.stages[1] << ',' << total.stages[2] << ','
              << total.late_ticks << ',' << total.missed_ticks << ',' << total.deadline_misses << ','
              << total.deadline_pending << ',' << cover.size() << ',' << replay.regions() << ','
              << construction_us << ',' << replay.mapping_ns() << ',' << initial_memory.index_bytes << ','
              << final_memory.index_bytes << ',' << final_memory.snapshot_bytes << ','
              << final_memory.mapping_bytes << ',' << peak_rss << '\n';
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return EXIT_FAILURE;
}
