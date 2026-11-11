#include "h3/h3_management.hpp"
#include "validation/oracle.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace spatial;
using namespace spatial::h3;
using Clock = std::chrono::steady_clock;

namespace {

struct EntitySeed {
    Id id{};
    Point point;
};

struct MoveSeed {
    Id id{};
    Point to;
};

struct Workload {
    std::vector<EntitySeed> entities;
    std::vector<Query> queries;
    std::vector<Query> training_queries;
    std::vector<MoveSeed> moves;
    std::vector<double> entity_cell_count;
    std::vector<double> training_query_cell_count;
    std::vector<double> move_cell_count;
};

PartitionConfig config_for_bench(int resolution, double map_size) {
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

Point clamp_point(Point point, const Box& map) {
    return {std::clamp(point.x, map.x0, map.x1),
            std::clamp(point.y, map.y0, map.y1)};
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

    std::vector<Point> cluster_centers;
    for (int i = 0; i < 5; ++i) {
        cluster_centers.push_back({xy(rng), xy(rng)});
    }
    auto random_point = [&]() {
        if (clustered && (rng() % 10) < 7) {
            const Point& c = cluster_centers[static_cast<std::size_t>(rng() % 5)];
            return clamp_point({c.x + step(rng), c.y + step(rng)}, cover.config().map);
        }
        return Point{xy(rng), xy(rng)};
    };
    auto random_query = [&](std::mt19937& local_rng) {
        const Point c = random_point();
        const double half = size_dist(local_rng);
        if (local_rng() % 2 == 0) {
            return circle(c, half);
        }
        return Query{{c.x - half, c.y - half, c.x + half, c.y + half},
                     false, {}, 0};
    };

    std::vector<Point> current;
    current.reserve(entities);
    for (std::size_t i = 0; i < entities; ++i) {
        const Point point = random_point();
        current.push_back(point);
        workload.entities.push_back({static_cast<Id>(i + 1), point});
    }
    // Uniform-size distribution but a mix of 50-1200 m gives small/medium/large
    // queries. The caller can inspect the actual distribution in the CSV.
    for (std::size_t i = 0; i < queries; ++i) {
        workload.queries.push_back(random_query(rng));
    }
    std::mt19937 training_rng(seed ^ 0x9e3779b9u);
    for (std::size_t i = 0; i < queries; ++i) {
        workload.training_queries.push_back(random_query(training_rng));
    }
    for (std::size_t i = 0; i < moves; ++i) {
        const std::size_t index = static_cast<std::size_t>(rng() % entities);
        Point next = current[index];
        next.x += step(rng);
        next.y += step(rng);
        next = clamp_point(next, cover.config().map);
        current[index] = next;
        workload.moves.push_back({static_cast<Id>(index + 1), next});
    }

    workload.entity_cell_count.assign(cover.size(), 0.0);
    for (const EntitySeed& entity : workload.entities) {
        const int cell_id = cover.id_of(h3_of(cover, entity.point));
        if (cell_id >= 0) ++workload.entity_cell_count[static_cast<std::size_t>(cell_id)];
    }
    workload.training_query_cell_count.assign(cover.size(), 0.0);
    for (const Query& query : workload.training_queries) {
        for (const CellId cell : cover.candidate_cells(query)) {
            ++workload.training_query_cell_count[static_cast<std::size_t>(cell)];
        }
    }
    workload.move_cell_count.assign(cover.size(), 0.0);
    std::vector<Point> move_positions;
    move_positions.reserve(entities);
    for (const EntitySeed& entity : workload.entities) move_positions.push_back(entity.point);
    for (const MoveSeed& move : workload.moves) {
        const std::size_t index = static_cast<std::size_t>(move.id - 1);
        const int from = cover.id_of(h3_of(cover, move_positions[index]));
        const int to = cover.id_of(h3_of(cover, move.to));
        if (from >= 0) ++workload.move_cell_count[static_cast<std::size_t>(from)];
        if (to >= 0) ++workload.move_cell_count[static_cast<std::size_t>(to)];
        move_positions[index] = move.to;
    }
    return workload;
}

std::vector<double> add_one(std::vector<double> values) {
    for (double& value : values) value += 1.0;
    return values;
}

struct Aggregate {
    double total_ms{};
    std::vector<double> latency_ms;
    QueryStats stats;
};

Aggregate run_queries(ManagementIndex& index,
                      const std::vector<Query>& queries,
                      int threads = 1) {
    Aggregate aggregate;
    aggregate.latency_ms.resize(queries.size(), 0.0);
    aggregate.stats = {};
    if (threads <= 1) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const auto begin = Clock::now();
            const QueryResult result = index.query(queries[i]);
            const auto end = Clock::now();
            aggregate.latency_ms[i] =
                std::chrono::duration<double, std::milli>(end - begin).count();
            aggregate.stats.candidate_cells += result.stats.candidate_cells;
            aggregate.stats.bucket_lookups += result.stats.bucket_lookups;
            aggregate.stats.leaves_visited += result.stats.leaves_visited;
            aggregate.stats.candidates += result.stats.candidates;
        }
        aggregate.total_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return aggregate;
    }

    std::atomic<std::size_t> next{0};
    std::vector<QueryStats> thread_stats(static_cast<std::size_t>(threads));
    std::vector<std::thread> workers;
    const auto start = Clock::now();
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            QueryStats local;
            while (true) {
                const std::size_t i = next.fetch_add(1);
                if (i >= queries.size()) break;
                const auto begin = Clock::now();
                const QueryResult result = index.query(queries[i]);
                const auto end = Clock::now();
                aggregate.latency_ms[i] =
                    std::chrono::duration<double, std::milli>(end - begin).count();
                local.candidate_cells += result.stats.candidate_cells;
                local.bucket_lookups += result.stats.bucket_lookups;
                local.leaves_visited += result.stats.leaves_visited;
                local.candidates += result.stats.candidates;
            }
            thread_stats[static_cast<std::size_t>(t)] = local;
        });
    }
    for (auto& worker : workers) worker.join();
    aggregate.total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (const QueryStats& stat : thread_stats) {
        aggregate.stats.candidate_cells += stat.candidate_cells;
        aggregate.stats.bucket_lookups += stat.bucket_lookups;
        aggregate.stats.leaves_visited += stat.leaves_visited;
        aggregate.stats.candidates += stat.candidates;
    }
    return aggregate;
}

std::vector<double> percentiles(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto at = [&](double p) {
        if (values.empty()) return 0.0;
        const std::size_t index = std::min(
            values.size() - 1,
            static_cast<std::size_t>(p * static_cast<double>(values.size() - 1)));
        return values[index];
    };
    return {at(0.50), at(0.95), at(0.99)};
}

std::size_t validate_index(ManagementIndex& index, ArrayOracle& oracle,
                           const std::vector<Query>& queries,
                           std::size_t max_queries) {
    std::size_t mismatches = 0;
    const std::size_t count = std::min(max_queries, queries.size());
    for (std::size_t i = 0; i < count; ++i) {
        auto actual = index.query(queries[i]).ids;
        std::sort(actual.begin(), actual.end());
        const auto expected = oracle.query(queries[i]);
        if (actual != expected) {
            ++mismatches;
            if (mismatches <= 2) {
                std::cerr << "mismatch in " << index.name() << " q=" << i
                          << " actual=" << actual.size()
                          << " expected=" << expected.size() << " first actual=";
                for (std::size_t k = 0; k < std::min<std::size_t>(5, actual.size()); ++k) {
                    std::cerr << actual[k] << ' ';
                }
                std::cerr << " first expected=";
                for (std::size_t k = 0; k < std::min<std::size_t>(5, expected.size()); ++k) {
                    std::cerr << expected[k] << ' ';
                }
                std::cerr << '\n';
                std::cerr << " actual_all=";
                for (const Id id : actual) std::cerr << id << ' ';
                std::cerr << "\n expected_all=";
                for (const Id id : expected) std::cerr << id << ' ';
                std::cerr << '\n';
            }
        }
    }
    return mismatches;
}

void write_row(std::ostream& out, const std::string& method, int threads,
               std::size_t entities, std::size_t queries, std::size_t partitions,
               std::size_t nodes, std::size_t buckets, std::size_t bytes,
               double build_ms, double update_ms, const Aggregate& aggregate) {
    const auto pct = percentiles(aggregate.latency_ms);
    const double qps = aggregate.total_ms > 0.0
                           ? 1000.0 * static_cast<double>(queries) / aggregate.total_ms
                           : 0.0;
    out << method << ',' << threads << ',' << entities << ',' << queries << ','
        << partitions << ',' << nodes << ',' << buckets << ',' << bytes << ','
        << std::fixed << std::setprecision(3) << build_ms << ',' << update_ms << ','
        << aggregate.total_ms << ',' << qps << ',' << pct[0] << ',' << pct[1]
        << ',' << pct[2] << ',' << aggregate.stats.candidate_cells << ','
        << aggregate.stats.bucket_lookups << ',' << aggregate.stats.leaves_visited
        << ',' << aggregate.stats.candidates << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t entities = argc > 1 ? std::stoull(argv[1]) : 20000;
        const std::size_t queries = argc > 2 ? std::stoull(argv[2]) : 20000;
        const std::size_t updates = argc > 3 ? std::stoull(argv[3]) : 20000;
        const int resolution = argc > 4 ? std::stoi(argv[4]) : 9;
        const std::string output = argc > 5 ? argv[5] : "results/h3_partition/bench.csv";
        const bool clustered = argc > 6 ? std::string(argv[6]) == "clustered" : true;
        const double map_size = argc > 7 ? std::stod(argv[7]) : 4096.0;
        const std::uint32_t seed = argc > 8 ? static_cast<std::uint32_t>(std::stoul(argv[8]))
                                           : 20261112u;

        auto cover = std::make_shared<CellCover>(config_for_bench(resolution, map_size));
        const Workload workload = make_workload(*cover, entities, queries, updates,
                                                 seed, clustered);
        std::ofstream out(output);
        if (!out) throw std::runtime_error("cannot open " + output);
        out << "method,threads,entities,queries,partitions,nodes,buckets,bytes,"
               "build_ms,update_ms,total_ms,qps,p50_ms,p95_ms,p99_ms,"
               "candidate_cells,bucket_lookups,leaves_visited,candidates\n";
        std::cout << "cover=" << cover->size() << " cells, entities=" << entities
                  << ", queries=" << queries << ", updates=" << updates
                  << ", clustered=" << clustered << '\n';

        struct MethodSpec {
            std::string name;
            std::function<std::unique_ptr<ManagementIndex>()> factory;
        };
        std::vector<MethodSpec> methods;
        methods.push_back({"GLOBAL_H3", [&]() {
                               return std::make_unique<GlobalH3Index>(cover);
                           }});
        methods.push_back({"FIXED_K8_UNIFORM", [&]() {
                               std::vector<double> weights(cover->size(), 1.0);
                               auto assignment = build_connected_assignment(*cover, weights, 8);
                               const int k = *std::max_element(assignment.begin(), assignment.end()) + 1;
                               return std::make_unique<FixedPartitionIndex>(cover, assignment, k);
                           }});
        methods.push_back({"FIXED_K8_ENTITY", [&]() {
                               auto weights = add_one(workload.entity_cell_count);
                               auto assignment = build_connected_assignment(*cover, weights, 8);
                               const int k = *std::max_element(assignment.begin(), assignment.end()) + 1;
                               return std::make_unique<FixedPartitionIndex>(cover, assignment, k);
                           }});
        methods.push_back({"FIXED_K8_QUERY_TRAIN", [&]() {
                               auto weights = add_one(workload.training_query_cell_count);
                               auto assignment = build_connected_assignment(*cover, weights, 8);
                               const int k = *std::max_element(assignment.begin(), assignment.end()) + 1;
                               return std::make_unique<FixedPartitionIndex>(cover, assignment, k);
                           }});
        methods.push_back({"FIXED_K8_MOVEMENT_TRAIN", [&]() {
                               auto weights = add_one(workload.move_cell_count);
                               auto assignment = build_connected_assignment(*cover, weights, 8);
                               const int k = *std::max_element(assignment.begin(), assignment.end()) + 1;
                               return std::make_unique<FixedPartitionIndex>(cover, assignment, k);
                           }});
        methods.push_back({"CAPACITY_512_128", [&]() {
                               return std::make_unique<CapacityPartitionIndex>(cover, 512, 128);
                           }});

        for (const MethodSpec& spec : methods) {
            auto index = spec.factory();
            ArrayOracle oracle;
            for (const EntitySeed& entity : workload.entities) {
                oracle.set(entity.id, entity.point);
            }
            const auto build_start = Clock::now();
            for (const EntitySeed& entity : workload.entities) {
                index->set(entity.id, entity.point);
            }
            const double build_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - build_start).count();
            std::cerr << spec.name << " size=" << index->size()
                      << " buckets=" << index->bucket_count()
                      << " partitions=" << index->partition_count() << '\n';
            const std::size_t mismatches =
                validate_index(*index, oracle, workload.queries, 300);
            if (mismatches != 0) {
                throw std::runtime_error(spec.name + " has " +
                                         std::to_string(mismatches) + " query mismatches");
            }
            const std::size_t sentinel_n = std::min<std::size_t>(1000, workload.queries.size());
            std::vector<Query> sentinel;
            sentinel.reserve(sentinel_n);
            for (std::size_t i = 0; i < sentinel_n; ++i) {
                const std::size_t at = i * workload.queries.size() / sentinel_n;
                sentinel.push_back(workload.queries[at]);
            }
            auto sentinel_time = [&]() {
                std::vector<double> samples;
                for (int rep = 0; rep < 3; ++rep) {
                    samples.push_back(run_queries(*index, sentinel, 1).total_ms);
                }
                std::sort(samples.begin(), samples.end());
                return samples[1];
            };
            // Three independent passes; use the median pass. The deterministic
            // counters (candidate cells, bucket lookups, fan-out) are the primary
            // evidence; timing is corroboration under host noise.
            std::vector<Aggregate> passes;
            std::vector<double> drifts;
            for (int attempt = 0; attempt < 3; ++attempt) {
                const double sentinel_before = sentinel_time();
                passes.push_back(run_queries(*index, workload.queries, 1));
                const double sentinel_after = sentinel_time();
                const double drift = std::max(sentinel_before, sentinel_after) /
                        std::max(1e-9, std::min(sentinel_before, sentinel_after));
                drifts.push_back(drift);
                std::cout << "  sentinel " << spec.name << " pass=" << attempt
                          << " before=" << sentinel_before
                          << " after=" << sentinel_after << " drift=" << drift << '\n';
            }
            std::sort(passes.begin(), passes.end(),
                      [](const Aggregate& a, const Aggregate& b) {
                          return a.total_ms < b.total_ms;
                      });
            const Aggregate single = passes[1];
            const double max_drift = *std::max_element(drifts.begin(), drifts.end());
            if (max_drift > 1.15) {
                std::cout << "  WARNING " << spec.name
                          << " host timing drift exceeded 1.15 (max=" << max_drift
                          << "); deterministic counters remain valid\n";
            }
            write_row(out, spec.name, 1, entities, queries, index->partition_count(),
                      index->node_count(), index->bucket_count(),
                      index->estimated_bytes(), build_ms, 0.0, single);
            for (const int threads : {4, 8}) {
                const Aggregate parallel = run_queries(*index, workload.queries, threads);
                write_row(out, spec.name, threads, entities, queries,
                          index->partition_count(), index->node_count(),
                          index->bucket_count(), index->estimated_bytes(), build_ms,
                          0.0, parallel);
            }
            const auto update_start = Clock::now();
            for (const MoveSeed& move : workload.moves) {
                index->set(move.id, move.to);
                oracle.set(move.id, move.to);
            }
            const double update_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - update_start).count();
            const std::size_t post_mismatches =
                validate_index(*index, oracle, workload.queries, 100);
            if (post_mismatches != 0) {
                throw std::runtime_error(spec.name + " has post-update mismatches");
            }
            const Aggregate after = run_queries(*index, workload.queries, 1);
            write_row(out, spec.name + "_AFTER_UPDATES", 1, entities, queries,
                      index->partition_count(), index->node_count(),
                      index->bucket_count(), index->estimated_bytes(), build_ms,
                      update_ms, after);
            std::cout << spec.name << ": build=" << build_ms << " ms, q="
                      << single.total_ms << " ms, updates=" << update_ms
                      << " ms, partitions=" << index->partition_count()
                      << ", leaves_visited/q="
                      << static_cast<double>(single.stats.leaves_visited) / queries
                      << '\n';
        }
        std::cout << "wrote " << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark failure: " << error.what() << '\n';
        return 1;
    }
}
