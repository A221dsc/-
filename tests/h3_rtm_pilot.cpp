#include "h3/h3_management.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace spatial;
using namespace spatial::h3;

namespace {

struct EntitySeed {
    Id id{};
    Point point;
};

struct MoveSeed {
    Id id{};
    Point from;
    Point to;
};

struct Workload {
    std::vector<EntitySeed> entities;
    std::vector<Query> training_queries;
    std::vector<Query> test_queries;
    std::vector<MoveSeed> training_moves;
    std::vector<MoveSeed> test_moves;
};

struct QueryPlan {
    int total_cells{};
    std::vector<std::pair<int, int>> region_cells;  // region, candidate cells
};

struct UpdatePlan {
    int old_region{-1};
    int new_region{-1};
};

struct PartitionData {
    std::vector<int> assignment;
    int regions{};
    std::vector<QueryPlan> training_queries;
    std::vector<QueryPlan> test_queries;
    std::vector<UpdatePlan> training_moves;
    std::vector<UpdatePlan> test_moves;
};

struct SimStats {
    double makespan{};
    double p50{};
    double p95{};
    double p99{};
    double avg_fanout{};
    double max_fanout{};
    double avg_regions{};
    double cross_messages{};
    double imbalance{};
};

PartitionConfig config_for_map(int resolution, double map_size) {
    PartitionConfig config;
    config.resolution = resolution;
    config.projection = LocalProjection(30.0, 120.0, 1.0);
    config.map = {0.0, 0.0, map_size, map_size};
    config.universe_margin_m = std::max(3000.0, map_size * 0.02);
    config.cell_margin_m = 2.0;
    return config;
}

H3Index h3_of(const CellCover& cover, Point point) {
    const GeoPoint geo = cover.config().projection.toGeo(point);
    LatLng ll{deg2rad(geo.lat_deg), deg2rad(geo.lng_deg)};
    H3Index cell = H3_NULL;
    h3_check(H3_EXPORT(latLngToCell)(&ll, cover.config().resolution, &cell),
             "latLngToCell");
    return cell;
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

Workload make_workload(const CellCover& cover, std::size_t entities,
                       std::size_t queries, std::size_t moves,
                       std::uint32_t seed, bool clustered) {
    Workload workload;
    std::mt19937 rng(seed);
    std::normal_distribution<double> step(0.0, 220.0);
    std::uniform_real_distribution<double> xy(cover.config().map.x0,
                                              cover.config().map.x1);
    std::uniform_real_distribution<double> size_dist(50.0, 1200.0);
    std::vector<Point> centers;
    for (int i = 0; i < 5; ++i) centers.push_back({xy(rng), xy(rng)});
    auto random_point = [&]() {
        if (clustered && (rng() % 10) < 7) {
            const Point& c = centers[static_cast<std::size_t>(rng() % 5)];
            return Point{std::clamp(c.x + step(rng), cover.config().map.x0,
                                    cover.config().map.x1),
                         std::clamp(c.y + step(rng), cover.config().map.y0,
                                    cover.config().map.y1)};
        }
        return Point{xy(rng), xy(rng)};
    };
    std::vector<Point> current;
    current.reserve(entities);
    for (std::size_t i = 0; i < entities; ++i) {
        const Point p = random_point();
        current.push_back(p);
        workload.entities.push_back({static_cast<Id>(i + 1), p});
    }
    auto make_queries = [&](std::uint32_t seed_value, std::size_t count) {
        std::mt19937 qrng(seed_value);
        std::vector<Query> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const Point c = random_point();
            const double half = size_dist(qrng);
            if (qrng() % 2 == 0) {
                out.push_back(circle(c, half));
            } else {
                out.push_back({{c.x - half, c.y - half, c.x + half, c.y + half},
                               false, {}, 0});
            }
        }
        return out;
    };
    workload.training_queries = make_queries(seed ^ 0x9e3779b9u, queries);
    workload.test_queries = make_queries(seed ^ 0x85ebca6bu, queries);

    auto make_moves = [&](std::uint32_t seed_value, std::size_t count) {
        std::mt19937 mrng(seed_value);
        std::vector<Point> positions;
        positions.reserve(entities);
        for (const auto& e : workload.entities) positions.push_back(e.point);
        std::vector<MoveSeed> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t index = static_cast<std::size_t>(mrng() % entities);
            Point next = positions[index];
            next.x = std::clamp(next.x + step(mrng), cover.config().map.x0,
                                cover.config().map.x1);
            next.y = std::clamp(next.y + step(mrng), cover.config().map.y0,
                                cover.config().map.y1);
            out.push_back({static_cast<Id>(index + 1), positions[index], next});
            positions[index] = next;
        }
        return out;
    };
    workload.training_moves = make_moves(seed ^ 0xc2b2ae35u, moves);
    workload.test_moves = make_moves(seed ^ 0x27d4eb2fu, moves);
    return workload;
}

int region_of_point(const CellCover& cover, const PartitionData& partition,
                    Point point) {
    const int cell = cover.id_of(h3_of(cover, point));
    if (cell < 0) return -1;
    if (static_cast<std::size_t>(cell) >= partition.assignment.size()) return -1;
    return partition.assignment[static_cast<std::size_t>(cell)];
}

QueryPlan make_query_plan(const CellCover& cover, const PartitionData& partition,
                          const Query& query) {
    QueryPlan plan;
    const auto cells = cover.candidate_cells(query);
    plan.total_cells = static_cast<int>(cells.size());
    std::vector<int> counts(static_cast<std::size_t>(partition.regions), 0);
    for (const CellId cell : cells) {
        const int region = partition.assignment[static_cast<std::size_t>(cell)];
        if (region < 0) continue;
        ++counts[static_cast<std::size_t>(region)];
    }
    for (int r = 0; r < partition.regions; ++r) {
        if (counts[static_cast<std::size_t>(r)] > 0) {
            plan.region_cells.push_back({r, counts[static_cast<std::size_t>(r)]});
        }
    }
    return plan;
}

PartitionData build_partition(const CellCover& cover, int target_regions,
                              const Workload& workload) {
    std::vector<double> weights(cover.size(), 1.0);
    for (const auto& e : workload.entities) {
        const int cell = cover.id_of(h3_of(cover, e.point));
        if (cell >= 0) weights[static_cast<std::size_t>(cell)] += 3.0;
    }
    for (const Query& q : workload.training_queries) {
        for (const CellId cell : cover.candidate_cells(q)) {
            weights[static_cast<std::size_t>(cell)] += 1.0;
        }
    }
    for (const MoveSeed& move : workload.training_moves) {
        const int a = cover.id_of(h3_of(cover, move.from));
        const int b = cover.id_of(h3_of(cover, move.to));
        if (a >= 0) weights[static_cast<std::size_t>(a)] += 1.0;
        if (b >= 0) weights[static_cast<std::size_t>(b)] += 1.0;
    }
    PartitionData data;
    data.assignment = build_connected_assignment(cover, weights, target_regions, 1);
    data.regions = *std::max_element(data.assignment.begin(), data.assignment.end()) + 1;
    data.training_queries.reserve(workload.training_queries.size());
    data.test_queries.reserve(workload.test_queries.size());
    for (const Query& q : workload.training_queries) {
        data.training_queries.push_back(make_query_plan(cover, data, q));
    }
    for (const Query& q : workload.test_queries) {
        data.test_queries.push_back(make_query_plan(cover, data, q));
    }
    data.training_moves.reserve(workload.training_moves.size());
    data.test_moves.reserve(workload.test_moves.size());
    for (const MoveSeed& move : workload.training_moves) {
        data.training_moves.push_back(
            {region_of_point(cover, data, move.from),
             region_of_point(cover, data, move.to)});
    }
    for (const MoveSeed& move : workload.test_moves) {
        data.test_moves.push_back(
            {region_of_point(cover, data, move.from),
             region_of_point(cover, data, move.to)});
    }
    return data;
}

std::vector<double> region_loads(const PartitionData& data, bool training) {
    std::vector<double> loads(static_cast<std::size_t>(data.regions), 0.0);
    const auto& queries = training ? data.training_queries : data.test_queries;
    for (const QueryPlan& plan : queries) {
        for (const auto& [region, cells] : plan.region_cells) {
            loads[static_cast<std::size_t>(region)] += cells;
        }
    }
    const auto& moves = training ? data.training_moves : data.test_moves;
    for (const UpdatePlan& move : moves) {
        if (move.old_region >= 0) loads[static_cast<std::size_t>(move.old_region)] += 2.0;
        if (move.new_region >= 0) loads[static_cast<std::size_t>(move.new_region)] += 2.0;
    }
    return loads;
}

[[maybe_unused]] std::vector<std::vector<double>> region_interactions(const PartitionData& data,
                                                     bool training) {
    std::vector<std::vector<double>> inter(
        static_cast<std::size_t>(data.regions),
        std::vector<double>(static_cast<std::size_t>(data.regions), 0.0));
    const auto& queries = training ? data.training_queries : data.test_queries;
    for (const QueryPlan& plan : queries) {
        for (std::size_t i = 0; i < plan.region_cells.size(); ++i) {
            for (std::size_t j = i + 1; j < plan.region_cells.size(); ++j) {
                const int a = plan.region_cells[i].first;
                const int b = plan.region_cells[j].first;
                inter[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] += 1.0;
                inter[static_cast<std::size_t>(b)][static_cast<std::size_t>(a)] += 1.0;
            }
        }
    }
    const auto& moves = training ? data.training_moves : data.test_moves;
    for (const UpdatePlan& move : moves) {
        if (move.old_region >= 0 && move.new_region >= 0 &&
            move.old_region != move.new_region) {
            inter[static_cast<std::size_t>(move.old_region)]
                 [static_cast<std::size_t>(move.new_region)] += 2.0;
            inter[static_cast<std::size_t>(move.new_region)]
                 [static_cast<std::size_t>(move.old_region)] += 2.0;
        }
    }
    return inter;
}

[[maybe_unused]] double mapping_score(const std::vector<int>& mapping,
                     const std::vector<double>& loads,
                     const std::vector<std::vector<double>>& inter, int workers,
                     double lambda) {
    std::vector<double> worker_load(static_cast<std::size_t>(workers), 0.0);
    for (std::size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] < 0) continue;
        worker_load[static_cast<std::size_t>(mapping[i])] += loads[i];
    }
    const double max_load = *std::max_element(worker_load.begin(), worker_load.end());
    double cross = 0.0;
    for (std::size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] < 0) continue;
        for (std::size_t j = i + 1; j < mapping.size(); ++j) {
            if (mapping[j] < 0) continue;
            if (mapping[i] != mapping[j]) cross += inter[i][j];
        }
    }
    return max_load + lambda * cross;
}

[[maybe_unused]] std::vector<int> fast_mapping(const std::vector<double>& loads,
                              const std::vector<std::vector<double>>& inter,
                              int workers, double lambda) {
    const int m = static_cast<int>(loads.size());
    std::vector<int> mapping(static_cast<std::size_t>(m), -1);
    std::vector<int> order(static_cast<std::size_t>(m));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return loads[static_cast<std::size_t>(a)] > loads[static_cast<std::size_t>(b)];
    });
    for (const int region : order) {
        double best = std::numeric_limits<double>::infinity();
        int best_worker = 0;
        for (int w = 0; w < workers; ++w) {
            mapping[static_cast<std::size_t>(region)] = w;
            const double score = mapping_score(mapping, loads, inter, workers, lambda);
            if (score < best) {
                best = score;
                best_worker = w;
            }
        }
        mapping[static_cast<std::size_t>(region)] = best_worker;
    }
    // Local search: move one region to another worker if it improves the objective.
    for (int pass = 0; pass < 4; ++pass) {
        bool improved = false;
        for (int region = 0; region < m; ++region) {
            const int old_worker = mapping[static_cast<std::size_t>(region)];
            int best_worker = old_worker;
            double best_score = mapping_score(mapping, loads, inter, workers, lambda);
            for (int w = 0; w < workers; ++w) {
                if (w == old_worker) continue;
                mapping[static_cast<std::size_t>(region)] = w;
                const double score = mapping_score(mapping, loads, inter, workers, lambda);
                if (score + 1e-9 < best_score) {
                    best_score = score;
                    best_worker = w;
                }
            }
            mapping[static_cast<std::size_t>(region)] = best_worker;
            improved = improved || (best_worker != old_worker);
        }
        if (!improved) break;
    }
    return mapping;
}

std::vector<int> fixed_mapping(int regions, int workers) {
    std::vector<int> mapping(static_cast<std::size_t>(regions), 0);
    for (int i = 0; i < regions; ++i) {
        mapping[static_cast<std::size_t>(i)] = i % workers;
    }
    return mapping;
}


double schedule_score(const PartitionData& data, bool training,
                      const std::vector<int>& mapping, int workers,
                      const std::vector<double>& loads, double lambda_cross,
                      double lambda_merge) {
    std::vector<double> worker_load(static_cast<std::size_t>(workers), 0.0);
    for (std::size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] < 0) continue;
        worker_load[static_cast<std::size_t>(mapping[i])] += loads[i];
    }
    const double max_load = *std::max_element(worker_load.begin(), worker_load.end());
    double cross = 0.0;
    double merge = 0.0;
    const auto& queries = training ? data.training_queries : data.test_queries;
    for (const QueryPlan& plan : queries) {
        std::vector<int> seen;
        for (const auto& [region, cells] : plan.region_cells) {
            (void)cells;
            if (region < 0 || mapping[static_cast<std::size_t>(region)] < 0) continue;
            const int w = mapping[static_cast<std::size_t>(region)];
            if (std::find(seen.begin(), seen.end(), w) == seen.end()) seen.push_back(w);
        }
        if (!seen.empty()) {
            cross += static_cast<double>(seen.size() - 1);
            merge += static_cast<double>(plan.region_cells.size() - 1);
        }
    }
    const auto& moves = training ? data.training_moves : data.test_moves;
    for (const UpdatePlan& move : moves) {
        if (move.old_region < 0 || move.new_region < 0) continue;
        if (mapping[static_cast<std::size_t>(move.old_region)] !=
            mapping[static_cast<std::size_t>(move.new_region)]) {
            cross += 1.0;
        }
    }
    return max_load + lambda_cross * cross + lambda_merge * merge;
}

std::vector<int> schedule_mapping(const PartitionData& data, bool training,
                                  int workers, double lambda_cross,
                                  double lambda_merge) {
    const auto loads = region_loads(data, training);
    const int m = static_cast<int>(loads.size());
    std::vector<int> mapping(static_cast<std::size_t>(m), -1);
    std::vector<int> order(static_cast<std::size_t>(m));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return loads[static_cast<std::size_t>(a)] > loads[static_cast<std::size_t>(b)];
    });
    for (const int region : order) {
        double best = std::numeric_limits<double>::infinity();
        int best_worker = 0;
        for (int w = 0; w < workers; ++w) {
            mapping[static_cast<std::size_t>(region)] = w;
            const double score = schedule_score(data, training, mapping, workers,
                                                loads, lambda_cross, lambda_merge);
            if (score < best) {
                best = score;
                best_worker = w;
            }
        }
        mapping[static_cast<std::size_t>(region)] = best_worker;
    }
    for (int pass = 0; pass < 3; ++pass) {
        bool improved = false;
        for (int region = 0; region < m; ++region) {
            const int old_worker = mapping[static_cast<std::size_t>(region)];
            int best_worker = old_worker;
            double best_score = schedule_score(data, training, mapping, workers,
                                               loads, lambda_cross, lambda_merge);
            for (int w = 0; w < workers; ++w) {
                if (w == old_worker) continue;
                mapping[static_cast<std::size_t>(region)] = w;
                const double score = schedule_score(data, training, mapping, workers,
                                                    loads, lambda_cross, lambda_merge);
                if (score + 1e-9 < best_score) {
                    best_score = score;
                    best_worker = w;
                }
            }
            mapping[static_cast<std::size_t>(region)] = best_worker;
            improved = improved || (best_worker != old_worker);
        }
        if (!improved) break;
    }
    return mapping;
}


SimStats simulate_partition(const Workload& workload, const PartitionData& partition,
                            const std::vector<int>& mapping, int workers,
                            double cell_cost, double update_cost,
                            double region_overhead, double msg_cost,
                            double merge_cost);

SimStats simulate_training(const PartitionData& part, const std::vector<int>& mapping,
                           int workers, double cell_cost, double update_cost,
                           double region_overhead, double msg_cost,
                           double merge_cost) {
    PartitionData eval = part;
    eval.test_queries = part.training_queries;
    eval.test_moves = part.training_moves;
    Workload dummy;
    dummy.test_queries.resize(eval.test_queries.size());
    dummy.test_moves.resize(eval.test_moves.size());
    return simulate_partition(dummy, eval, mapping, workers, cell_cost, update_cost,
                              region_overhead, msg_cost, merge_cost);
}

SimStats simulate_hash(const CellCover& cover, const Workload& workload,
                       int workers, double cell_cost, double update_cost,
                       double msg_cost, double merge_cost) {
    SimStats stats;
    std::vector<double> avail(static_cast<std::size_t>(workers), 0.0);
    std::vector<double> busy(static_cast<std::size_t>(workers), 0.0);
    std::vector<double> latency;
    latency.reserve(workload.test_queries.size() + workload.test_moves.size());
    double fanout_sum = 0.0;
    double max_fanout = 0.0;
    double total_cross = 0.0;
    auto owner = [&](H3Index cell) {
        return static_cast<int>(splitmix64(static_cast<std::uint64_t>(cell)) %
                                static_cast<std::uint64_t>(workers));
    };
    auto submit_query = [&](const Query& query) {
        const auto cells = cover.candidate_cells(query);
        std::vector<int> counts(static_cast<std::size_t>(workers), 0);
        for (const CellId cell : cells) {
            const int w = owner(cover.cell(cell).h3);
            ++counts[static_cast<std::size_t>(w)];
        }
        double max_finish = 0.0;
        int distinct = 0;
        int home = 0;
        double home_finish = -1.0;
        for (int w = 0; w < workers; ++w) {
            if (counts[static_cast<std::size_t>(w)] == 0) continue;
            const double service =
                cell_cost * static_cast<double>(counts[static_cast<std::size_t>(w)]);
            const double finish = avail[static_cast<std::size_t>(w)] + service;
            avail[static_cast<std::size_t>(w)] = finish;
            busy[static_cast<std::size_t>(w)] += service;
            max_finish = std::max(max_finish, finish);
            if (finish > home_finish) {
                home_finish = finish;
                home = w;
            }
            ++distinct;
        }
        const double completion =
            max_finish + msg_cost * std::max(0, distinct - 1) +
            merge_cost * std::max(0, distinct - 1);
        avail[static_cast<std::size_t>(home)] =
            std::max(avail[static_cast<std::size_t>(home)], completion);
        latency.push_back(completion);
        fanout_sum += distinct;
        max_fanout = std::max(max_fanout, static_cast<double>(distinct));
        total_cross += std::max(0, distinct - 1);
    };
    auto submit_move = [&](const MoveSeed& move) {
        const int a = owner(h3_of(cover, move.from));
        const int b = owner(h3_of(cover, move.to));
        const double service = update_cost;
        if (a == b) {
            avail[static_cast<std::size_t>(a)] += service;
            busy[static_cast<std::size_t>(a)] += service;
            latency.push_back(avail[static_cast<std::size_t>(a)]);
            return;
        }
        const double fa = avail[static_cast<std::size_t>(a)] + service;
        const double fb = avail[static_cast<std::size_t>(b)] + service;
        avail[static_cast<std::size_t>(a)] = fa;
        avail[static_cast<std::size_t>(b)] = fb;
        busy[static_cast<std::size_t>(a)] += service;
        busy[static_cast<std::size_t>(b)] += service;
        const double completion = std::max(fa, fb) + msg_cost;
        const int home = fa >= fb ? a : b;
        avail[static_cast<std::size_t>(home)] =
            std::max(avail[static_cast<std::size_t>(home)], completion);
        latency.push_back(completion);
        total_cross += 1.0;
    };
    for (const Query& q : workload.test_queries) submit_query(q);
    for (const MoveSeed& m : workload.test_moves) submit_move(m);
    stats.makespan = *std::max_element(avail.begin(), avail.end());
    std::sort(latency.begin(), latency.end());
    auto pct = [&](double p) {
        return latency[std::min(latency.size() - 1,
                                static_cast<std::size_t>(p * (latency.size() - 1)))];
    };
    stats.p50 = pct(0.50);
    stats.p95 = pct(0.95);
    stats.p99 = pct(0.99);
    stats.avg_fanout = fanout_sum / static_cast<double>(workload.test_queries.size());
    stats.max_fanout = max_fanout;
    stats.avg_regions = stats.avg_fanout;
    stats.cross_messages = total_cross;
    const double max_busy = *std::max_element(busy.begin(), busy.end());
    const double min_busy = *std::min_element(busy.begin(), busy.end());
    const double mean_busy = std::accumulate(busy.begin(), busy.end(), 0.0) / workers;
    stats.imbalance = mean_busy > 0 ? (max_busy - min_busy) / mean_busy : 0.0;
    return stats;
}

SimStats simulate_partition(const Workload& workload, const PartitionData& partition,
                            const std::vector<int>& mapping, int workers,
                            double cell_cost, double update_cost,
                            double region_overhead, double msg_cost,
                            double merge_cost) {
    SimStats stats;
    std::vector<double> avail(static_cast<std::size_t>(workers), 0.0);
    std::vector<double> busy(static_cast<std::size_t>(workers), 0.0);
    std::vector<double> latency;
    latency.reserve(workload.test_queries.size() + workload.test_moves.size());
    double fanout_sum = 0.0;
    double max_fanout = 0.0;
    double regions_sum = 0.0;
    double total_cross = 0.0;

    auto submit_counts = [&](const std::vector<std::pair<int, int>>& region_cells,
                             bool is_query) {
        double max_finish = 0.0;
        int distinct = 0;
        int regions = 0;
        int home = 0;
        double home_finish = -1.0;
        std::vector<int> workers_seen;
        for (const auto& [region, cells] : region_cells) {
            if (region < 0) continue;
            const int w = mapping[static_cast<std::size_t>(region)];
            if (std::find(workers_seen.begin(), workers_seen.end(), w) ==
                workers_seen.end()) {
                workers_seen.push_back(w);
                ++distinct;
            }
            ++regions;
            const double service = (is_query ? region_overhead : 0.0) +
                                   cell_cost * static_cast<double>(cells);
            const double finish = avail[static_cast<std::size_t>(w)] + service;
            avail[static_cast<std::size_t>(w)] = finish;
            busy[static_cast<std::size_t>(w)] += service;
            max_finish = std::max(max_finish, finish);
            if (finish > home_finish) {
                home_finish = finish;
                home = w;
            }
        }
        const double completion = max_finish +
                                  msg_cost * std::max(0, distinct - 1) +
                                  merge_cost * std::max(0, regions - 1);
        avail[static_cast<std::size_t>(home)] =
            std::max(avail[static_cast<std::size_t>(home)], completion);
        latency.push_back(completion);
        fanout_sum += distinct;
        max_fanout = std::max(max_fanout, static_cast<double>(distinct));
        regions_sum += regions;
        total_cross += std::max(0, distinct - 1);
    };

    for (std::size_t i = 0; i < workload.test_queries.size(); ++i) {
        submit_counts(partition.test_queries[i].region_cells, true);
    }
    for (const UpdatePlan& move : partition.test_moves) {
        std::vector<std::pair<int, int>> cells;
        if (move.old_region == move.new_region) {
            cells.push_back({move.old_region, 1});
        } else {
            if (move.old_region >= 0) cells.push_back({move.old_region, 1});
            if (move.new_region >= 0) cells.push_back({move.new_region, 1});
        }
        // Treat each update cell as one unit of update service.
        double max_finish = 0.0;
        int distinct = 0;
        int regions = 0;
        int home = 0;
        double home_finish = -1.0;
        std::vector<int> seen;
        for (const auto& [region, _] : cells) {
            const int w = mapping[static_cast<std::size_t>(region)];
            if (std::find(seen.begin(), seen.end(), w) == seen.end()) {
                seen.push_back(w);
                ++distinct;
            }
            ++regions;
            const double finish = avail[static_cast<std::size_t>(w)] + update_cost;
            avail[static_cast<std::size_t>(w)] = finish;
            busy[static_cast<std::size_t>(w)] += update_cost;
            max_finish = std::max(max_finish, finish);
            if (finish > home_finish) {
                home_finish = finish;
                home = w;
            }
        }
        const double completion = max_finish +
                                  msg_cost * std::max(0, distinct - 1) +
                                  merge_cost * std::max(0, regions - 1);
        avail[static_cast<std::size_t>(home)] =
            std::max(avail[static_cast<std::size_t>(home)], completion);
        latency.push_back(completion);
        fanout_sum += distinct;
        max_fanout = std::max(max_fanout, static_cast<double>(distinct));
        regions_sum += regions;
        total_cross += std::max(0, distinct - 1);
    }

    stats.makespan = *std::max_element(avail.begin(), avail.end());
    std::sort(latency.begin(), latency.end());
    auto pct = [&](double p) {
        return latency[std::min(latency.size() - 1,
                                static_cast<std::size_t>(p * (latency.size() - 1)))];
    };
    stats.p50 = pct(0.50);
    stats.p95 = pct(0.95);
    stats.p99 = pct(0.99);
    stats.avg_fanout = fanout_sum /
                       static_cast<double>(workload.test_queries.size() + workload.test_moves.size());
    stats.max_fanout = max_fanout;
    stats.avg_regions = regions_sum /
                        static_cast<double>(workload.test_queries.size() + workload.test_moves.size());
    stats.cross_messages = total_cross;
    const double max_busy = *std::max_element(busy.begin(), busy.end());
    const double min_busy = *std::min_element(busy.begin(), busy.end());
    const double mean_busy = std::accumulate(busy.begin(), busy.end(), 0.0) / workers;
    stats.imbalance = mean_busy > 0 ? (max_busy - min_busy) / mean_busy : 0.0;
    return stats;
}

void write_row(std::ostream& out, const std::string& map_name, int seed, int workers,
               const std::string& method, int regions, const SimStats& stats,
               double oracle_gap, double mapping_obj) {
    out << map_name << ',' << seed << ',' << workers << ',' << method << ','
        << regions << ',' << std::fixed << std::setprecision(3) << stats.makespan
        << ',' << stats.p50 << ',' << stats.p95 << ',' << stats.p99 << ','
        << stats.avg_fanout << ',' << stats.max_fanout << ','
        << stats.avg_regions << ',' << stats.cross_messages << ','
        << stats.imbalance << ',' << oracle_gap << ',' << mapping_obj << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string output = argc > 1 ? argv[1] : "results/h3_rtm/pilot.csv";
        const std::size_t entities = argc > 2 ? std::stoull(argv[2]) : 20000;
        const std::size_t queries = argc > 3 ? std::stoull(argv[3]) : 5000;
        const std::size_t moves = argc > 4 ? std::stoull(argv[4]) : 5000;
        std::ofstream out(output);
        if (!out) throw std::runtime_error("cannot open " + output);
        out << "map,seed,workers,method,regions,makespan_us,p50_us,p95_us,p99_us,"
               "avg_fanout,max_fanout,avg_regions,cross_messages,imbalance,"
               "oracle_gap,mapping_obj\n";

        struct MapSpec {
            std::string name;
            int resolution;
            double size;
            bool clustered;
            double cell_cost;
            double update_cost;
            double region_overhead;
            double msg_cost;
            double merge_cost;
        };
        const std::vector<MapSpec> maps{
            {"4km", 9, 4096.0, true, 3.48, 1.36, 0.5, 5.0, 1.0},
            {"20km", 10, 20000.0, false, 0.0936, 1.46, 0.5, 5.0, 1.0}};

        for (const MapSpec& map : maps) {
            auto cover = std::make_shared<CellCover>(config_for_map(map.resolution, map.size));
            for (const int seed : {101, 202, 303}) {
                const Workload workload = make_workload(*cover, entities, queries, moves,
                                                         static_cast<std::uint32_t>(seed),
                                                         map.clustered);
                for (const int workers : {4, 8}) {
                    const PartitionData part_b = build_partition(*cover, workers, workload);
                    const auto b_map = fixed_mapping(part_b.regions, workers);
                    const SimStats a = simulate_hash(*cover, workload, workers,
                                                     map.cell_cost, map.update_cost,
                                                     map.msg_cost, map.merge_cost);
                    const SimStats b = simulate_partition(workload, part_b, b_map, workers,
                                                          map.cell_cost, map.update_cost,
                                                          map.region_overhead, map.msg_cost,
                                                          map.merge_cost);

                    struct Candidate {
                        double factor{};
                        int m{};
                        PartitionData part;
                        std::vector<int> d_map;
                        SimStats train{};
                        SimStats test{};
                        SimStats oracle_test{};
                    };
                    std::vector<Candidate> candidates;
                    {
                        Candidate base;
                        base.factor = 1.0;
                        base.part = part_b;
                        base.m = part_b.regions;
                        base.d_map = b_map;
                        base.train = simulate_training(part_b, b_map, workers,
                                                       map.cell_cost, map.update_cost,
                                                       map.region_overhead, map.msg_cost,
                                                       map.merge_cost);
                        base.test = b;
                        const auto oracle_map = schedule_mapping(part_b, false, workers, 1.0, 0.5);
                        base.oracle_test = simulate_partition(workload, part_b, oracle_map, workers,
                                                              map.cell_cost, map.update_cost,
                                                              map.region_overhead, map.msg_cost,
                                                              map.merge_cost);
                        candidates.push_back(std::move(base));
                    }
                    for (const double factor : {1.25, 1.5, 2.0}) {
                        const int target_m = std::max(
                            workers, static_cast<int>(std::llround(workers * factor)));
                        Candidate cand;
                        cand.factor = factor;
                        cand.part = build_partition(*cover, target_m, workload);
                        cand.m = cand.part.regions;
                        cand.d_map = schedule_mapping(cand.part, true, workers, 1.0, 0.5);
                        cand.train = simulate_training(cand.part, cand.d_map, workers,
                                                       map.cell_cost, map.update_cost,
                                                       map.region_overhead, map.msg_cost,
                                                       map.merge_cost);
                        cand.test = simulate_partition(workload, cand.part, cand.d_map, workers,
                                                       map.cell_cost, map.update_cost,
                                                       map.region_overhead, map.msg_cost,
                                                       map.merge_cost);
                        const auto oracle_map = schedule_mapping(cand.part, false, workers, 1.0, 0.5);
                        cand.oracle_test = simulate_partition(workload, cand.part, oracle_map, workers,
                                                              map.cell_cost, map.update_cost,
                                                              map.region_overhead, map.msg_cost,
                                                              map.merge_cost);
                        candidates.push_back(std::move(cand));
                    }

                    std::size_t adaptive = 0;
                    std::size_t oracle = 0;
                    for (std::size_t i = 1; i < candidates.size(); ++i) {
                        if (candidates[i].train.makespan < candidates[adaptive].train.makespan) {
                            adaptive = i;
                        }
                        if (candidates[i].oracle_test.makespan < candidates[oracle].oracle_test.makespan) {
                            oracle = i;
                        }
                    }
                    const double oracle_gap =
                        candidates[oracle].oracle_test.makespan > 0
                            ? (candidates[adaptive].test.makespan -
                               candidates[oracle].oracle_test.makespan) /
                                  candidates[oracle].oracle_test.makespan
                            : 0.0;

                    write_row(out, map.name, seed, workers, "A_HASH", 0, a, 0.0, 0.0);
                    write_row(out, map.name, seed, workers, "B_M_EQ_T_FIXED",
                              part_b.regions, b, 0.0, 0.0);
                    for (const Candidate& cand : candidates) {
                        const SimStats c = simulate_partition(
                            workload, cand.part, fixed_mapping(cand.part.regions, workers),
                            workers, map.cell_cost, map.update_cost,
                            map.region_overhead, map.msg_cost, map.merge_cost);
                        const std::string tag = std::to_string(cand.factor);
                        write_row(out, map.name, seed, workers,
                                  "C_M" + tag + "_FIXED", cand.m, c, 0.0, 0.0);
                        write_row(out, map.name, seed, workers,
                                  "D_M" + tag + "_FAST", cand.m, cand.test,
                                  (cand.test.makespan - candidates[oracle].oracle_test.makespan) /
                                      std::max(1e-9, candidates[oracle].oracle_test.makespan),
                                  0.0);
                    }
                    write_row(out, map.name, seed, workers, "F_ADAPTIVE_TRAIN",
                              candidates[adaptive].m, candidates[adaptive].test,
                              oracle_gap, 0.0);
                    write_row(out, map.name, seed, workers, "G_ADAPTIVE_ORACLE",
                              candidates[oracle].m, candidates[oracle].oracle_test, 0.0, 0.0);

                    std::cout << map.name << " seed=" << seed << " workers=" << workers
                              << " m_B=" << part_b.regions
                              << " A=" << a.makespan << " B=" << b.makespan;
                    for (const Candidate& cand : candidates) {
                        std::cout << " C" << cand.factor << "=" << cand.test.makespan;
                    }
                    std::cout << " F=" << candidates[adaptive].test.makespan
                              << " G=" << candidates[oracle].oracle_test.makespan
                              << " m_F=" << candidates[adaptive].m
                              << " m_G=" << candidates[oracle].m
                              << " gap=" << oracle_gap << '\n';
                }
            }
        }
        std::cout << "wrote " << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "H3-RTM pilot failure: " << error.what() << '\n';
        return 1;
    }
}
