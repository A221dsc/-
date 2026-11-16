#include "h3/h3_management.hpp"
#include "validation/oracle.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace spatial;
using namespace spatial::h3;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<Id> sorted(std::vector<Id> ids) {
    std::sort(ids.begin(), ids.end());
    return ids;
}

PartitionConfig config_for_test(int resolution, double size, bool clustered) {
    (void)clustered;
    PartitionConfig config;
    config.resolution = resolution;
    config.projection = LocalProjection(30.0, 120.0, 1.0);
    config.map = {0.0, 0.0, size, size};
    config.universe_margin_m = std::max(3000.0, size * 0.02);
    config.cell_margin_m = 2.0;
    return config;
}

struct WorkloadData {
    std::vector<std::pair<Id, Point>> entities;
    std::vector<Query> queries;
    std::vector<std::pair<Id, Point>> updates;
};

WorkloadData make_workload(const CellCover& cover, std::size_t entities,
                           std::size_t queries, std::size_t updates,
                           std::uint32_t seed, bool clustered) {
    WorkloadData data;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> xy(cover.config().map.x0,
                                              cover.config().map.x1);
    std::uniform_real_distribution<double> size_dist(50.0, 1200.0);
    std::normal_distribution<double> step(0.0, 220.0);
    std::vector<Point> centers;
    for (int i = 0; i < 5; ++i) centers.push_back({xy(rng), xy(rng)});
    auto point = [&]() {
        if (clustered && (rng() % 10) < 7) {
            const Point& c = centers[static_cast<std::size_t>(rng() % 5)];
            return Point{std::clamp(c.x + step(rng), cover.config().map.x0,
                                    cover.config().map.x1),
                         std::clamp(c.y + step(rng), cover.config().map.y0,
                                    cover.config().map.y1)};
        }
        return Point{xy(rng), xy(rng)};
    };
    std::vector<Point> positions;
    positions.reserve(entities);
    for (std::size_t i = 0; i < entities; ++i) {
        const Point p = point();
        positions.push_back(p);
        data.entities.push_back({static_cast<Id>(i + 1), p});
    }
    for (std::size_t i = 0; i < queries; ++i) {
        const Point c = point();
        const double half = size_dist(rng);
        if (rng() % 2 == 0) {
            data.queries.push_back(circle(c, half));
        } else {
            data.queries.push_back({{c.x - half, c.y - half, c.x + half, c.y + half},
                                    false, {}, 0});
        }
    }
    for (std::size_t i = 0; i < updates; ++i) {
        const std::size_t index = static_cast<std::size_t>(rng() % entities);
        Point next = positions[index];
        next.x = std::clamp(next.x + step(rng), cover.config().map.x0,
                            cover.config().map.x1);
        next.y = std::clamp(next.y + step(rng), cover.config().map.y0,
                            cover.config().map.y1);
        positions[index] = next;
        data.updates.push_back({static_cast<Id>(index + 1), next});
    }
    return data;
}

void check_query(WorkerShardedIndex& index, ArrayOracle& oracle, Query query,
                 const std::string& label) {
    const QueryResult result = index.query(query);
    auto actual = result.ids;
    std::sort(actual.begin(), actual.end());
    const auto expected = oracle.query(query);
    if (actual != expected) {
        throw std::runtime_error(label + ": query mismatch actual=" +
                                 std::to_string(actual.size()) + " expected=" +
                                 std::to_string(expected.size()));
    }
    require(result.stats.candidates >= result.ids.size(),
            label + ": bad candidate count");
}

int mapping_shift(int region, int workers) {
    return (region * 3 + 1) % workers;
}

void run_config(std::shared_ptr<CellCover> cover, int workers, double m_factor,
                bool parallel_check, const std::string& label) {
    const int target_m = std::max(workers, static_cast<int>(std::llround(workers * m_factor)));
    std::vector<double> weights(cover->size(), 1.0);
    const auto assignment = build_connected_assignment(*cover, weights, target_m, 1);
    const int regions = *std::max_element(assignment.begin(), assignment.end()) + 1;
    std::vector<int> mapping(static_cast<std::size_t>(regions), 0);
    for (int i = 0; i < regions; ++i) mapping[static_cast<std::size_t>(i)] = mapping_shift(i, workers);

    const bool clustered = cover->config().map.x1 <= 4096.0;
    const WorkloadData data = make_workload(*cover, 5000, 1000, 2000, 20261116u,
                                            clustered);
    WorkerShardedIndex index(cover, assignment, mapping);
    ArrayOracle oracle;
    for (const auto& [id, point] : data.entities) {
        index.set(id, point);
        oracle.set(id, point);
    }
    std::string why;
    require(index.validate(&why), label + ": initial validate: " + why);
    require(index.size() == data.entities.size(), label + ": size");
    std::size_t worker_sum = 0;
    for (int w = 0; w < static_cast<int>(index.worker_count()); ++w) {
        worker_sum += index.worker_entity_count(w);
    }
    require(worker_sum == index.size(), label + ": worker counts");

    for (const Query& query : data.queries) check_query(index, oracle, query, label);

    for (std::size_t i = 0; i < data.updates.size(); ++i) {
        index.set(data.updates[i].first, data.updates[i].second);
        oracle.set(data.updates[i].first, data.updates[i].second);
        if (i % 100 == 0) {
            check_query(index, oracle, circle({2048, 2048}, 800), label + " update");
            require(index.validate(&why), label + ": update validate: " + why);
        }
    }
    for (const Query& query : data.queries) check_query(index, oracle, query, label + " post-update");
    require(index.validate(&why), label + ": post-update validate: " + why);

    if (parallel_check) {
        std::vector<std::vector<Id>> results(data.queries.size());
        std::vector<std::thread> threads;
        const int thread_count = 8;
        for (int t = 0; t < thread_count; ++t) {
            threads.emplace_back([&, t] {
                for (std::size_t i = static_cast<std::size_t>(t); i < data.queries.size();
                     i += static_cast<std::size_t>(thread_count)) {
                    results[i] = sorted(index.query(data.queries[i]).ids);
                }
            });
        }
        for (auto& thread : threads) thread.join();
        for (std::size_t i = 0; i < data.queries.size(); ++i) {
            const auto expected = oracle.query(data.queries[i]);
            require(results[i] == expected, label + ": parallel query mismatch");
        }
    }

    std::cout << label << ": workers=" << workers << " regions=" << regions
              << " entities=" << index.size() << " queries=" << data.queries.size()
              << " updates=" << data.updates.size() << " buckets="
              << index.total_bucket_count() << " passed\n";
}

}  // namespace

int main() {
    try {
        {
            auto cover = std::make_shared<CellCover>(config_for_test(9, 4096.0, true));
            for (const int workers : {4, 8}) {
                for (const double factor : {1.0, 1.5, 2.0}) {
                    run_config(cover, workers, factor, workers == 8,
                               "4km/res9/t" + std::to_string(workers) + "/m" +
                                   std::to_string(factor));
                }
            }
        }
        {
            auto cover = std::make_shared<CellCover>(config_for_test(10, 20000.0, false));
            run_config(cover, 8, 2.0, true, "20km/res10/t8/m2");
        }
        std::cout << "H3-RTM worker-sharded correctness suite passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "H3-RTM correctness failure: " << error.what() << '\n';
        return 1;
    }
}
