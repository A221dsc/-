#include "h3/h3_management.hpp"
#include "validation/oracle.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_set>
#include <stdexcept>
#include <string>
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

PartitionConfig config_for_test() {
    PartitionConfig config;
    config.resolution = 9;
    config.projection = LocalProjection(30.0, 120.0, 1.0);
    config.map = {0.0, 0.0, 4096.0, 4096.0};
    config.universe_margin_m = 3000.0;
    config.cell_margin_m = 2.0;
    return config;
}

void check_partitions(const PartitionedBase& index) {
    const auto& assignment = index.cell_assignment();
    const auto cover = index.cell_cover();
    std::unordered_map<int, std::vector<CellId>> by_id;
    for (std::size_t cell = 0; cell < assignment.size(); ++cell) {
        if (assignment[cell] < 0) throw std::runtime_error("unassigned cell");
        by_id[assignment[cell]].push_back(static_cast<CellId>(cell));
    }
    for (const auto& [id, group] : by_id) {
        (void)id;
        if (group.empty()) throw std::runtime_error("empty partition");
        std::unordered_set<int> members(group.begin(), group.end());
        std::queue<int> queue;
        std::unordered_set<int> seen;
        queue.push(group.front());
        seen.insert(group.front());
        while (!queue.empty()) {
            const int cell = queue.front();
            queue.pop();
            for (const int neighbor : cover->neighbors(cell)) {
                if (members.count(neighbor) && seen.insert(neighbor).second) {
                    queue.push(neighbor);
                }
            }
        }
        if (seen.size() != group.size()) {
            throw std::runtime_error("partition is not connected");
        }
    }
}

void check_query(ManagementIndex& index, ArrayOracle& oracle, Query query) {
    const auto actual = sorted(index.query(query).ids);
    const auto expected = oracle.query(query);
    if (actual != expected) {
        std::cerr << index.name() << " mismatch: actual=" << actual.size()
                  << " expected=" << expected.size() << '\n';
        throw std::runtime_error("management query mismatch");
    }
}

void cover_tests() {
    auto cover = std::make_shared<CellCover>(config_for_test());
    require(cover->size() > 100, "cover too small");
    std::cout << "cover: " << cover->size() << " H3 cells, max radius "
              << cover->max_radius() << " m\n";
    const auto small = cover->candidate_cells(circle({2048, 2048}, 100));
    const auto large = cover->candidate_cells(
        {{0, 0, 4096, 4096}, false, {}, 0});
    require(!small.empty(), "small query has no candidate cells");
    require(large.size() >= small.size(), "large query has fewer candidates");
    require(large.size() <= cover->size(), "large query exceeds universe");
    std::cout << "candidates: small=" << small.size() << " large=" << large.size()
              << '\n';
}

void global_test(std::shared_ptr<CellCover> cover) {
    GlobalH3Index index(cover);
    ArrayOracle oracle;
    std::mt19937 rng(101);
    std::uniform_real_distribution<double> xy(0.0, 4096.0);
    for (Id id = 1; id <= 2000; ++id) {
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
    }
    for (int i = 0; i < 200; ++i) {
        const Point c{xy(rng), xy(rng)};
        check_query(index, oracle, circle(c, 100.0 + 900.0 * (rng() % 100) / 100.0));
        check_query(index, oracle, {{c.x, c.y, c.x + 500, c.y + 500}, false, {}, 0});
    }
    for (int i = 0; i < 1000; ++i) {
        const Id id = 1 + static_cast<Id>(rng() % 2000);
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
        if (i % 50 == 0) {
            check_query(index, oracle, circle({xy(rng), xy(rng)}, 200.0));
        }
    }
    check_query(index, oracle, {{0, 0, 4096, 4096}, false, {}, 0});
    std::cout << "global H3: 2000 entities, updates, 401+ queries passed\n";
}

void fixed_test(std::shared_ptr<CellCover> cover) {
    std::vector<double> weights(cover->size(), 1.0);
    const auto assignment = build_connected_assignment(*cover, weights, 8, 7);
    require(assignment.size() == cover->size(), "assignment size");
    FixedPartitionIndex index(cover, assignment, 8);
    ArrayOracle oracle;
    std::mt19937 rng(203);
    std::uniform_real_distribution<double> xy(0.0, 4096.0);
    for (Id id = 1; id <= 2000; ++id) {
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
    }
    for (int i = 0; i < 200; ++i) {
        const Point c{xy(rng), xy(rng)};
        check_query(index, oracle, circle(c, 100.0 + 900.0 * (rng() % 100) / 100.0));
        check_query(index, oracle, {{c.x, c.y, c.x + 500, c.y + 500}, false, {}, 0});
    }
    for (int i = 0; i < 1000; ++i) {
        const Id id = 1 + static_cast<Id>(rng() % 2000);
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
        if (i % 50 == 0) check_query(index, oracle, circle({xy(rng), xy(rng)}, 200.0));
    }
    check_query(index, oracle, {{0, 0, 4096, 4096}, false, {}, 0});
    require(index.partition_count() == 8, "fixed partition count");
    check_partitions(index);
    std::cout << "fixed H3 partition: 8 partitions, 2000 entities, updates, queries passed\n";
}

void capacity_stress_test(std::shared_ptr<CellCover> cover) {
    CapacityPartitionIndex index(cover, 8, 2);
    ArrayOracle oracle;
    std::mt19937 rng(509);
    std::uniform_real_distribution<double> xy(0.0, 4096.0);
    for (Id id = 1; id <= 2000; ++id) {
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
    }
    for (int i = 0; i < 5000; ++i) {
        const Id id = 1 + static_cast<Id>(rng() % 2000);
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
        if (i % 37 == 0) {
            check_query(index, oracle, circle({xy(rng), xy(rng)}, 300.0));
            check_partitions(index);
        }
    }
    check_query(index, oracle, {{0, 0, 4096, 4096}, false, {}, 0});
    check_partitions(index);
    std::cout << "capacity stress: partitions=" << index.partition_count()
              << " nodes=" << index.node_count() << " splits=" << index.split_count()
              << " merges=" << index.merge_count() << '\n';
}

void capacity_test(std::shared_ptr<CellCover> cover) {
    CapacityPartitionIndex index(cover, 128, 32);
    ArrayOracle oracle;
    std::mt19937 rng(307);
    std::uniform_real_distribution<double> xy(0.0, 4096.0);
    for (Id id = 1; id <= 3000; ++id) {
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
    }
    for (int i = 0; i < 100; ++i) {
        const Point c{xy(rng), xy(rng)};
        check_query(index, oracle, circle(c, 100.0 + 900.0 * (rng() % 100) / 100.0));
        check_query(index, oracle, {{c.x, c.y, c.x + 500, c.y + 500}, false, {}, 0});
    }
    for (int i = 0; i < 1000; ++i) {
        const Id id = 1 + static_cast<Id>(rng() % 3000);
        const Point p{xy(rng), xy(rng)};
        index.set(id, p);
        oracle.set(id, p);
        if (i % 50 == 0) check_query(index, oracle, circle({xy(rng), xy(rng)}, 200.0));
    }
    check_query(index, oracle, {{0, 0, 4096, 4096}, false, {}, 0});
    check_partitions(index);
    std::cout << "capacity tree: partitions=" << index.partition_count()
              << " nodes=" << index.node_count() << " splits=" << index.split_count()
              << " merges=" << index.merge_count() << '\n';
    require(index.split_count() > 0, "capacity tree did not split");
    for (Id id = 1; id <= 2500; ++id) {
        index.erase(id);
        oracle.erase(id);
    }
    check_query(index, oracle, {{0, 0, 4096, 4096}, false, {}, 0});
    check_partitions(index);
    std::cout << "capacity tree after deletes: partitions=" << index.partition_count()
              << " nodes=" << index.node_count() << " merges=" << index.merge_count()
              << '\n';
}

}  // namespace

int main() {
    try {
        auto cover = std::make_shared<CellCover>(config_for_test());
        cover_tests();
        global_test(cover);
        fixed_test(cover);
        capacity_test(cover);
        capacity_stress_test(cover);
        std::cout << "H3 management partition tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "H3 management test failure: " << error.what() << '\n';
        return 1;
    }
}
