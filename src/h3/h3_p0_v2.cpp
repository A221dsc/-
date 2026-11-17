#include "h3/h3_p0_v2.hpp"

#include <algorithm>
#include <bit>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace spatial::h3::p0v2 {
namespace {

constexpr CellId absent = -1;

std::uint64_t nanos(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

class Workers {
public:
    explicit Workers(int count) {
        try {
            threads_.reserve(count);
            for (int worker = 0; worker < count; ++worker)
                threads_.emplace_back([this, worker] { loop(worker); });
        } catch (...) {
            stop();
            throw;
        }
    }

    ~Workers() { stop(); }

    void run(std::function<void(int)> task) {
        std::unique_lock lock(mutex_);
        task_ = std::move(task);
        error_ = {};
        remaining_ = threads_.size();
        ++generation_;
        ready_.notify_all();
        finished_.wait(lock, [this] { return remaining_ == 0; });
        task_ = {};
        if (error_) std::rethrow_exception(error_);
    }

private:
    void loop(int worker) {
        std::uint64_t seen{};
        std::unique_lock lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || seen != generation_; });
            if (stopping_) return;
            seen = generation_;
            lock.unlock();
            std::exception_ptr failure;
            try {
                task_(worker);
            } catch (...) {
                failure = std::current_exception();
            }
            lock.lock();
            if (failure && !error_) error_ = failure;
            if (--remaining_ == 0) finished_.notify_one();
        }
    }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::function<void(int)> task_;
    std::exception_ptr error_;
    std::uint64_t generation_{};
    std::size_t remaining_{};
    bool stopping_{};
    std::vector<std::thread> threads_;
};

struct OldEntry {
    Slot slot;
    Point point;
};

struct Bucket {
    std::vector<Slot> live;
    std::vector<OldEntry> prior;
    std::uint64_t saved_epoch{};
};

#if defined(_WIN32)
// The installed MinGW winpthreads implementation admitted two writers on
// the same lock during mixed replay. Use Windows' native reader/writer lock.
class SharedMutex {
public:
    void lock() noexcept { AcquireSRWLockExclusive(&lock_); }
    bool try_lock() noexcept { return TryAcquireSRWLockExclusive(&lock_); }
    void unlock() noexcept { ReleaseSRWLockExclusive(&lock_); }
    void lock_shared() noexcept { AcquireSRWLockShared(&lock_); }
    bool try_lock_shared() noexcept { return TryAcquireSRWLockShared(&lock_); }
    void unlock_shared() noexcept { ReleaseSRWLockShared(&lock_); }

private:
    SRWLOCK lock_ = SRWLOCK_INIT;
};
#else
using SharedMutex = std::shared_mutex;
#endif

struct alignas(64) CellLock {
    SharedMutex mutex;
};

struct Layout {
    std::vector<int> cell_to_worker;
    std::vector<int> cell_to_partition;
    std::vector<int> partition_to_worker;

    int worker(CellId cell) const { return cell_to_worker[cell]; }
};

std::uint64_t cell_order(const CellInfo& cell) {
    const auto x = static_cast<std::uint32_t>(cell.grid_x);
    const auto y = static_cast<std::uint32_t>(cell.grid_y);
    std::uint64_t key{};
    for (int bit = 0; bit < 32; ++bit) {
        key |= static_cast<std::uint64_t>((x >> bit) & 1U) << (2 * bit);
        key |= static_cast<std::uint64_t>((y >> bit) & 1U) << (2 * bit + 1);
    }
    return key;
}

void check_partition(const CellCover& cover, const std::vector<int>& part, int count) {
    std::vector<int> visited(cover.size(), 0);
    std::vector<int> component(count);
    std::vector<CellId> queue;
    for (CellId root = 0; root < static_cast<CellId>(cover.size()); ++root) {
        if (visited[root]) continue;
        const auto region = part[root];
        if (region < 0 || region >= count || ++component[region] != 1)
            throw std::logic_error("invalid or disconnected management region");
        queue.assign(1, root);
        visited[root] = 1;
        for (std::size_t i = 0; i < queue.size(); ++i) {
            for (const auto neighbor : cover.neighbors(queue[i])) {
                if (part[neighbor] != region || visited[neighbor]) continue;
                visited[neighbor] = 1;
                queue.push_back(neighbor);
            }
        }
    }
    if (std::find(component.begin(), component.end(), 0) != component.end())
        throw std::logic_error("empty management region");
}

Layout make_layout(const CellCover& cover, Mode mode, int workers,
                   const std::vector<CellId>& locations) {
    Layout layout;
    if (mode == Mode::Serial || mode == Mode::Lock) return layout;

    std::vector<double> weights(cover.size(), 1.0);
    for (const CellId cell : locations) {
        if (cell != absent) ++weights[cell];
    }
    layout.cell_to_worker.resize(cover.size());
    if (mode == Mode::Static) {
        std::vector<CellId> order(cover.size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<std::uint64_t> keys(cover.size());
        for (const auto cell : order) keys[cell] = cell_order(cover.cell(cell));
        std::stable_sort(order.begin(), order.end(),
                         [&](CellId a, CellId b) { return keys[a] < keys[b]; });
        const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        double accumulated{};
        for (const auto cell : order) {
            layout.cell_to_worker[cell] = std::min(workers - 1,
                static_cast<int>(workers * (accumulated + weights[cell] / 2) / total));
            accumulated += weights[cell];
        }
        return layout;
    }

    const int target = std::min(static_cast<int>(cover.size()), workers * 2);
    auto part = build_connected_assignment(cover, weights, target);
    const int count = *std::max_element(part.begin(), part.end()) + 1;
    if (count != target) throw std::logic_error("could not build the specified region count");
    check_partition(cover, part, count);
    std::vector<double> region_weights(count);
    for (std::size_t cell = 0; cell < part.size(); ++cell)
        region_weights[part[cell]] += weights[cell];
    std::vector<int> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return region_weights[a] > region_weights[b]; });
    std::vector<double> loads(workers);
    std::vector<int> owners(count);
    for (const int region : order) {
        const auto worker = std::min_element(loads.begin(), loads.end()) - loads.begin();
        owners[region] = static_cast<int>(worker);
        loads[worker] += region_weights[region];
    }
    for (std::size_t cell = 0; cell < part.size(); ++cell)
        layout.cell_to_worker[cell] = owners[part[cell]];
    if (mode == Mode::Partition) {
        layout.cell_to_partition = std::move(part);
        layout.partition_to_worker = std::move(owners);
    }
    return layout;
}

template<class Lock>
void acquire(Lock& lock, bool profile, std::uint64_t& count, std::uint64_t& misses,
             std::uint64_t& wait_ns) {
    ++count;
    if (!profile) {
        lock.lock();
        return;
    }
    if (lock.try_lock()) return;
    const auto begin = Clock::now();
    lock.lock();
    wait_ns += nanos(begin);
    ++misses;
}

struct Fragment {
    int owner{-1};
    std::vector<CellId> cells;
    QueryResult answer;
};

struct QueryPlan {
    std::vector<Fragment> fragments;
    std::uint64_t cover_ns{}, route_ns{};
    std::size_t owners{};
};

struct MovePlan {
    CellId source{absent}, destination{absent};
    int owner{-1};
    std::uint64_t route_ns{};
};

struct Job {
    bool query;
    std::size_t event;
    std::size_t piece{};
};

struct alignas(64) WorkerState {
    WorkerStats stats;
    Clock::time_point finished;
};

}  // namespace

#if defined(_WIN32)
Clock::time_point Clock::now() noexcept {
    static const auto frequency = [] {
        LARGE_INTEGER value;
        QueryPerformanceFrequency(&value);
        return value.QuadPart;
    }();
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    constexpr auto per_second = std::nano::den;
    return time_point(duration(count.QuadPart / frequency * per_second
                               + count.QuadPart % frequency * per_second / frequency));
}
#endif

std::string_view name(Mode mode) {
    switch (mode) {
        case Mode::Serial: return "A+";
        case Mode::Lock: return "A+Lock";
        case Mode::Static: return "A+Static";
        case Mode::Matched: return "A+StaticMatched";
        case Mode::Partition: return "B+";
    }
    throw std::invalid_argument("unknown mode");
}

CellId locate(const CellCover& cover, Point point) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !cover.config().map.contains(point)) throw std::invalid_argument("point outside map");
    const auto geo = cover.config().projection.toGeo(point);
    const LatLng ll{deg2rad(geo.lat_deg), deg2rad(geo.lng_deg)};
    H3Index index{};
    h3_check(H3_EXPORT(latLngToCell)(&ll, cover.config().resolution, &index), "latLngToCell");
    const auto cell = cover.id_of(index);
    if (cell == absent) throw std::logic_error("map point outside H3 universe");
    return cell;
}

class Replay::Impl {
public:
    Impl(const CellCover& cover, std::span<const std::optional<Point>> initial,
         Mode mode, int workers, bool profile)
        : cover_(cover), mode_(mode), workers_(checked_workers(mode, workers)),
          profile_(profile), buckets_(cover.size()),
          locks_(workers_ > 1 ? std::make_unique<CellLock[]>(cover.size()) : nullptr),
          points_(initial.size()), locations_(initial.size(), absent), offsets_(initial.size()),
          seen_(initial.size()), pool_(mode_ == Mode::Serial ? nullptr : std::make_unique<Workers>(workers_)) {
        if (initial.size() > std::numeric_limits<Slot>::max()) throw std::invalid_argument("too many slots");
        for (Slot slot = 0; slot < initial.size(); ++slot) {
            if (!initial[slot]) continue;
            points_[slot] = *initial[slot];
            insert(slot, locate(cover_, *initial[slot]));
        }
        const auto begin = Clock::now();
        layout_ = make_layout(cover_, mode_, workers_, locations_);
        mapping_ns_ = nanos(begin);
    }

    FrameResult run(const Frame& frame, Clock::time_point scheduled) {
        if (poisoned_) throw std::logic_error("replay unusable after failed mutation");
        const auto begin = Clock::now();
        ++epoch_;
        for (const auto& change : frame.updates) {
            if (change.slot >= seen_.size() || seen_[change.slot] == epoch_)
                throw std::invalid_argument("invalid or duplicate slot in frame");
            seen_[change.slot] = epoch_;
        }
        std::vector<QueryPlan> queries(frame.queries.size());
        std::vector<MovePlan> moves(frame.updates.size());
        std::atomic<std::size_t> next_plan{};
        const auto plan_count = queries.size() + moves.size();
        execute([&](int) {
            constexpr std::size_t batch = 4;
            for (auto first = next_plan.fetch_add(batch, std::memory_order_relaxed);
                 first < plan_count; first = next_plan.fetch_add(batch, std::memory_order_relaxed)) {
                for (auto index = first; index < std::min(first + batch, plan_count); ++index) {
                    if (index < queries.size()) {
                        queries[index] = plan_query(frame.queries[index]);
                    } else {
                        const auto n = index - queries.size();
                        moves[n] = plan_move(frame.updates[n]);
                    }
                }
            }
        });
        const auto prepared = Clock::now();
        FrameResult result;
        auto& stats = result.stats;
        stats.prepare_ns = nanos(begin, prepared);
        std::vector<std::vector<Job>> queues(fixed() ? workers_ : 1);
        const auto push = [&](int owner, Job job) {
            if (owner >= 0) ++stats.routed_jobs;
            queues[owner >= 0 ? owner : 0].push_back(job);
        };
        for (std::size_t index = 0; index < std::max(queries.size(), moves.size()); ++index) {
            // Update and query tasks share each queue and the same execution phase.
            if (index < moves.size()) {
                push(moves[index].owner, {false, index});
                stats.route_cpu_ns += moves[index].route_ns;
            }
            if (index >= queries.size()) continue;
            auto& plan = queries[index];
            stats.cover_cpu_ns += plan.cover_ns;
            stats.route_cpu_ns += plan.route_ns;
            stats.owner_fanout += plan.owners;
            if (mode_ == Mode::Partition) stats.region_fanout += plan.fragments.size();
            for (std::size_t piece = 0; piece < plan.fragments.size(); ++piece)
                push(plan.fragments[piece].owner, {true, index, piece});
        }
        std::vector<WorkerState> work(workers_);
        std::atomic<std::size_t> next_job{};
        const auto dispatched = Clock::now();
        stats.dispatch_ns = nanos(prepared, dispatched);
        poisoned_ = true;
        execute([&](int worker) {
            auto& local = work[worker].stats;
            const auto worker_start = Clock::now();
            const auto visit = [&](const Job& job) {
                const auto start = profile_ ? Clock::now() : Clock::time_point{};
                if (profile_) local.queue_wait_ns += nanos(dispatched, start);
                if (job.query) {
                    const auto& query = frame.queries[job.event];
                    auto& frag = queries[job.event].fragments[job.piece];
                    scan(query, frag, local);
                    ++local.query_tasks;
                    if (profile_) local.query_ns += nanos(start);
                } else {
                    const auto crossings = local.cross_worker;
                    apply(frame.updates[job.event], moves[job.event], local);
                    ++local.update_tasks;
                    if (profile_) {
                        const auto elapsed = nanos(start);
                        local.update_ns += elapsed;
                        if (local.cross_worker != crossings) local.migration_ns += elapsed;
                    }
                }
            };
            if (fixed()) {
                for (const auto& job : queues[worker]) visit(job);
            } else {
                constexpr std::size_t batch = 4;
                const auto& queue = queues.front();
                for (auto first = next_job.fetch_add(batch, std::memory_order_relaxed);
                     first < queue.size(); first = next_job.fetch_add(batch, std::memory_order_relaxed)) {
                    for (auto index = first; index < std::min(first + batch, queue.size()); ++index)
                        visit(queue[index]);
                }
            }
            work[worker].finished = Clock::now();
            local.busy_ns = nanos(worker_start, work[worker].finished);
        });
        // Completion of the worker phase publishes the next state.
        const auto published = Clock::now();
        stats.execute_ns = nanos(dispatched, published);
        stats.update_latency_ns = nanos(scheduled, published);
        stats.workers.reserve(workers_);
        const auto last = std::max_element(work.begin(), work.end(),
            [](const auto& a, const auto& b) { return a.finished < b.finished; });
        stats.barrier_tail_ns = nanos(last->finished, published);
        for (const auto& state : work) stats.workers.push_back(state.stats);

        result.answers.reserve(queries.size());
        result.query_latency_ns.reserve(queries.size());
        for (auto& plan : queries) {
            result.answers.push_back(gather(plan));
            result.query_latency_ns.push_back(nanos(scheduled));
        }
        const auto completed = Clock::now();
        stats.merge_ns = nanos(published, completed);
        stats.frame_ns = nanos(begin, completed);
        poisoned_ = false;
        return result;
    }

    void validate(std::span<const std::optional<Point>> expected) const {
        if (poisoned_ || expected.size() != points_.size()) throw std::logic_error("invalid state");
        std::vector<int> counts(points_.size());
        for (CellId cell = 0; cell < static_cast<CellId>(buckets_.size()); ++cell) {
            const auto& live = buckets_[cell].live;
            for (std::size_t offset = 0; offset < live.size(); ++offset) {
                const auto slot = live[offset];
                if (slot >= counts.size() || locations_[slot] != cell ||
                    offsets_[slot] != offset || ++counts[slot] != 1)
                    throw std::logic_error("bad bucket membership/offset");
            }
        }
        for (std::size_t slot = 0; slot < points_.size(); ++slot) {
            if (!expected[slot]) {
                if (locations_[slot] != absent || counts[slot] != 0)
                    throw std::logic_error("deleted entity in live state");
                continue;
            }
            if (counts[slot] != 1)
                throw std::logic_error("missing entity at slot " + std::to_string(slot));
            if (points_[slot] != *expected[slot])
                throw std::logic_error("wrong point at slot " + std::to_string(slot)
                    + ", actual=" + std::to_string(points_[slot].x) + "," + std::to_string(points_[slot].y)
                    + ", expected=" + std::to_string(expected[slot]->x) + "," + std::to_string(expected[slot]->y));
            if (locations_[slot] != locate(cover_, points_[slot]))
                throw std::logic_error("wrong H3 label at slot " + std::to_string(slot));
        }
    }

    Memory memory() const {
        Memory m{
            sizeof(Impl) + buckets_.capacity() * sizeof(Bucket)
                + (locks_ ? cover_.size() * sizeof(CellLock) : 0)
                + points_.capacity() * sizeof(Point) + locations_.capacity() * sizeof(CellId)
                + offsets_.capacity() * sizeof(std::size_t) + seen_.capacity() * sizeof(std::uint64_t),
            0,
            sizeof(int) * (layout_.cell_to_worker.capacity() + layout_.cell_to_partition.capacity()
                + layout_.partition_to_worker.capacity())};
        for (const auto& bucket : buckets_) {
            m.index_bytes += bucket.live.capacity() * sizeof(Slot);
            m.snapshot_bytes += bucket.prior.capacity() * sizeof(OldEntry);
        }
        return m;
    }

    int regions() const { return static_cast<int>(layout_.partition_to_worker.size()); }
    std::uint64_t mapping_ns() const { return mapping_ns_; }

private:
    static int checked_workers(Mode mode, int workers) {
        if (workers <= 0 || workers > 64 || (mode == Mode::Serial && workers != 1))
            throw std::invalid_argument("unsupported worker count");
        return workers;
    }

    bool fixed() const { return !layout_.cell_to_worker.empty(); }

    void execute(std::function<void(int)> work) {
        if (pool_) pool_->run(std::move(work));
        else work(0);
    }

    QueryPlan plan_query(const Query& query) const {
        const auto& box = query.box;
        if (!std::isfinite(box.x0) || !std::isfinite(box.x1) ||
            !std::isfinite(box.y0) || !std::isfinite(box.y1) || box.x0 > box.x1 || box.y0 > box.y1)
            throw std::invalid_argument("invalid query box");
        if (query.circular && (!std::isfinite(query.center.x) || !std::isfinite(query.center.y) ||
                               !std::isfinite(query.radius) || query.radius < 0))
            throw std::invalid_argument("invalid query circle");
        const auto begin = profile_ ? Clock::now() : Clock::time_point{};
        auto cells = cover_.candidate_cells(query);
        const auto routed = profile_ ? Clock::now() : Clock::time_point{};
        QueryPlan plan;
        if (profile_) plan.cover_ns = nanos(begin, routed);
        if (!fixed()) {
            plan.fragments.push_back({-1, std::move(cells), {}});
            return plan;
        }

        const bool management = mode_ == Mode::Partition;
        std::vector<int> piece(management ? regions() : workers_, -1);
        std::uint64_t owners{};
        for (const auto cell : cells) {
            const int group = management ? layout_.cell_to_partition[cell] : layout_.worker(cell);
            auto& position = piece[group];
            if (position < 0) {
                const int owner = management ? layout_.partition_to_worker[group] : group;
                position = static_cast<int>(plan.fragments.size());
                plan.fragments.push_back({owner, {}, {}});
                owners |= std::uint64_t{1} << owner;
            }
            plan.fragments[position].cells.push_back(cell);
        }
        plan.owners = std::popcount(owners);
        if (profile_) plan.route_ns = nanos(routed);
        return plan;
    }

    MovePlan plan_move(const Change& change) const {
        const auto source = locations_[change.slot];
        const auto destination = change.point ? locate(cover_, *change.point) : absent;
        const auto begin = profile_ ? Clock::now() : Clock::time_point{};
        const auto route = source != absent ? source : destination;
        const auto owner = fixed() && route != absent ? layout_.worker(route) : -1;
        return {source, destination, owner, profile_ ? nanos(begin) : 0};
    }

    void scan(const Query& query, Fragment& fragment, WorkerStats& local) {
        auto& answer = fragment.answer;
        const auto filter = [&](Slot slot, Point p) {
            ++answer.stats.candidates;
            if (query.matches(p)) answer.ids.push_back(static_cast<Id>(slot));
        };
        for (const auto cell : fragment.cells) {
            std::shared_lock<SharedMutex> guard;
            if (locks_) {
                guard = std::shared_lock(locks_[cell].mutex, std::defer_lock);
                acquire(guard, profile_, local.read_locks, local.read_misses, local.read_wait_ns);
            }
            const auto& bucket = buckets_[cell];
            if (bucket.saved_epoch == epoch_) {
                for (const auto& entry : bucket.prior) filter(entry.slot, entry.point);
            } else {
                for (const auto slot : bucket.live) filter(slot, points_[slot]);
            }
        }
        answer.stats.candidate_cells = answer.stats.bucket_lookups = fragment.cells.size();
        local.cells += fragment.cells.size();
        local.candidates += answer.stats.candidates;
    }

    void save(CellId cell, WorkerStats& worker) {
        auto& bucket = buckets_[cell];
        if (bucket.saved_epoch == epoch_) return;
        bucket.prior.clear();
        bucket.prior.reserve(bucket.live.size());
        for (const auto slot : bucket.live) bucket.prior.push_back({slot, points_[slot]});
        bucket.saved_epoch = epoch_;
        ++worker.snapshot_cells;
        worker.snapshot_entries += bucket.prior.size();
    }

    void insert(Slot slot, CellId cell) {
        auto& bucket = buckets_[cell].live;
        bucket.push_back(slot);
        offsets_[slot] = bucket.size() - 1;
        locations_[slot] = cell;
    }

    void erase(Slot slot, CellId cell) {
        auto& bucket = buckets_[cell].live;
        const auto offset = offsets_[slot];
        const auto tail = bucket.back();
        bucket[offset] = tail;
        offsets_[tail] = offset;
        bucket.pop_back();
        locations_[slot] = absent;
    }

    void apply(const Change& change, const MovePlan& plan, WorkerStats& worker) {
        const auto from = plan.source;
        const auto to = plan.destination;
        const auto slot = change.slot;
        if (from == absent && to == absent) return;
        const auto first = from == absent ? to : (to == absent ? from : std::min(from, to));
        const auto second = from != absent && to != absent && from != to ? std::max(from, to) : absent;
        const bool migration = second != absent && fixed() && plan.owner != layout_.worker(to);
        const auto before_wait = worker.write_wait_ns;
        std::unique_lock<SharedMutex> low, high;
        if (locks_) {
            low = std::unique_lock(locks_[first].mutex, std::defer_lock);
            acquire(low, profile_, worker.write_locks, worker.write_misses, worker.write_wait_ns);
            if (second != absent) {
                high = std::unique_lock(locks_[second].mutex, std::defer_lock);
                acquire(high, profile_, worker.write_locks, worker.write_misses, worker.write_wait_ns);
            }
        }
        save(first, worker);
        if (second != absent) save(second, worker);
        if (from == to) {
            ++worker.same_cell;
        } else {
            if (to != absent) {
                // Allocate first so failure cannot orphan a live entity.
                auto& live = buckets_[to].live;
                if (live.size() == live.capacity())
                    live.reserve(std::max<std::size_t>(4, live.size() * 2));
            }
            if (from != absent) erase(slot, from);
            if (to != absent) insert(slot, to);
            if (from == absent) ++worker.insertions;
            else if (to == absent) ++worker.deletions;
            else {
                ++worker.cross_cell;
                if (mode_ == Mode::Partition) {
                    if (layout_.cell_to_partition[from] == layout_.cell_to_partition[to]) ++worker.intrapartition;
                    else ++worker.cross_partition;
                }
                if (migration) {
                    ++worker.cross_worker;
                    worker.migration_wait_ns += worker.write_wait_ns - before_wait;
                }
            }
        }
        if (change.point) points_[slot] = *change.point;
    }

    static QueryResult gather(QueryPlan& plan) {
        if (plan.fragments.size() == 1) return std::move(plan.fragments.front().answer);
        QueryResult out;
        std::size_t hits{};
        for (const auto& fragment : plan.fragments) hits += fragment.answer.ids.size();
        out.ids.reserve(hits);
        for (const auto& fragment : plan.fragments) {
            const auto& answer = fragment.answer;
            out.ids.insert(out.ids.end(), answer.ids.begin(), answer.ids.end());
            out.stats.candidate_cells += answer.stats.candidate_cells;
            out.stats.bucket_lookups += answer.stats.bucket_lookups;
            out.stats.candidates += answer.stats.candidates;
        }
        return out;
    }

    const CellCover& cover_;
    Mode mode_;
    int workers_;
    bool profile_, poisoned_{};
    std::vector<Bucket> buckets_;
    std::unique_ptr<CellLock[]> locks_;
    std::vector<Point> points_;
    std::vector<CellId> locations_;
    std::vector<std::size_t> offsets_;
    std::vector<std::uint64_t> seen_;
    Layout layout_;
    std::uint64_t epoch_{}, mapping_ns_{};
    std::unique_ptr<Workers> pool_;
};

Replay::Replay(const CellCover& cover, std::span<const std::optional<Point>> initial,
               Mode mode, int workers, bool profile)
    : impl_(std::make_unique<Impl>(cover, initial, mode, workers, profile)) {}
Replay::~Replay() = default;
FrameResult Replay::run(const Frame& frame, Clock::time_point scheduled) {
    return impl_->run(frame, scheduled);
}
void Replay::validate(std::span<const std::optional<Point>> expected) const {
    impl_->validate(expected);
}
Memory Replay::memory() const { return impl_->memory(); }
int Replay::regions() const { return impl_->regions(); }
std::uint64_t Replay::mapping_ns() const { return impl_->mapping_ns(); }

}  // namespace spatial::h3::p0v2
