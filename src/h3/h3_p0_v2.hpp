#pragma once

#include "h3/h3_management.hpp"

#include <chrono>
#include <span>
#include <string_view>

namespace spatial::h3::p0v2 {

#if defined(_WIN32)
struct Clock {
    using duration = std::chrono::nanoseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<Clock>;
    static constexpr bool is_steady = true;
    static time_point now() noexcept;
};
#else
using Clock = std::chrono::steady_clock;
#endif
using Slot = std::uint32_t;

enum class Mode { Serial, Lock, Static, Matched, Partition };
std::string_view name(Mode mode);
CellId locate(const CellCover& cover, Point point);

struct Change {
    Slot slot;
    std::optional<Point> point;
};

struct Frame {
    std::vector<Query> queries;
    std::vector<Change> updates;
};

struct WorkerStats {
    std::uint64_t query_tasks{}, update_tasks{}, cells{}, candidates{};
    std::uint64_t read_locks{}, write_locks{}, read_misses{}, write_misses{};
    std::uint64_t read_wait_ns{}, write_wait_ns{}, migration_wait_ns{};
    std::uint64_t queue_wait_ns{}, busy_ns{}, query_ns{}, update_ns{}, migration_ns{};
    std::uint64_t same_cell{}, cross_cell{}, intrapartition{}, cross_partition{}, cross_worker{};
    std::uint64_t insertions{}, deletions{}, snapshot_cells{}, snapshot_entries{};
};

struct FrameStats {
    std::uint64_t prepare_ns{}, cover_cpu_ns{}, route_cpu_ns{}, dispatch_ns{}, execute_ns{}, merge_ns{};
    std::uint64_t barrier_tail_ns{}, frame_ns{}, update_latency_ns{};
    std::uint64_t owner_fanout{}, region_fanout{}, routed_jobs{};
    std::vector<WorkerStats> workers;
};

struct FrameResult {
    std::vector<QueryResult> answers;
    std::vector<std::uint64_t> query_latency_ns;
    FrameStats stats;
};

struct Memory {
    std::size_t index_bytes{}, snapshot_bytes{}, mapping_bytes{};
};

class Replay {
public:
    // Single coordinator. A frame contains at most one change per slot, and
    // all returned answers are from the state at the start of that frame.
    Replay(const CellCover& cover, std::span<const std::optional<Point>> initial,
           Mode mode, int workers, bool profile = false);
    ~Replay();
    Replay(const Replay&) = delete;
    Replay& operator=(const Replay&) = delete;

    FrameResult run(const Frame& frame, Clock::time_point scheduled);
    void validate(std::span<const std::optional<Point>> expected) const;
    Memory memory() const;
    int regions() const;
    std::uint64_t mapping_ns() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spatial::h3::p0v2
