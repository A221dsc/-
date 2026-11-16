#pragma once

#include "core/index.hpp"
#include "h3/geo.hpp"
#include "h3/h3_util.hpp"
#include "h3/local_projection.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "h3api.h"

namespace spatial::h3 {

using CellId = int;

struct PartitionConfig {
    int resolution = 9;
    LocalProjection projection{30.0, 120.0, 1.0};
    Box map{0.0, 0.0, 4096.0, 4096.0};
    double universe_margin_m = 5000.0;
    double cell_margin_m = 2.0;
};

struct CellInfo {
    H3Index h3{};
    CellId id{-1};
    Point center;
    BBox bbox;
    double radius{};  // conservative circumradius in local metres
    int grid_x{};
    int grid_y{};
};

// Fixed-resolution H3 universe plus a shared, conservative cell-cover used by
// every method in the A/B/C experiment. The cover is not part of the
// management partition; it is the common query front-end.
class CellCover {
public:
    explicit CellCover(PartitionConfig config);

    std::size_t size() const { return cells_.size(); }
    const PartitionConfig& config() const { return config_; }
    const CellInfo& cell(CellId id) const { return cells_.at(static_cast<std::size_t>(id)); }
    int id_of(H3Index h3) const;
    const std::vector<int>& neighbors(CellId id) const {
        return neighbors_.at(static_cast<std::size_t>(id));
    }
    double max_radius() const { return max_radius_; }

    // Conservative candidate cover: every cell that may contain a point inside
    // the query is returned. Exact entity filtering happens later.
    std::vector<CellId> candidate_cells(const Query& query) const;

private:
    PartitionConfig config_;
    std::vector<CellInfo> cells_;
    std::unordered_map<H3Index, CellId> h3_to_id_;
    std::vector<std::vector<int>> neighbors_;
    double max_radius_{};
    double grid_min_x_{};
    double grid_min_y_{};
    double grid_cell_{};
    int grid_w_{};
    int grid_h_{};
    std::vector<std::vector<CellId>> grid_;
};

struct QueryStats {
    std::uint64_t candidate_cells{};
    std::uint64_t bucket_lookups{};
    std::uint64_t leaves_visited{};
    std::uint64_t candidates{};
};

struct QueryResult {
    std::vector<Id> ids;
    QueryStats stats;
};

class ManagementIndex {
public:
    virtual ~ManagementIndex() = default;
    virtual void set(Id id, Point point) = 0;
    virtual void erase(Id id) = 0;
    virtual QueryResult query(Query query) const = 0;
    virtual std::size_t size() const = 0;
    virtual std::size_t partition_count() const = 0;
    virtual std::size_t node_count() const = 0;
    virtual std::size_t bucket_count() const = 0;
    virtual std::size_t estimated_bytes() const = 0;
    virtual std::optional<Point> point_of(Id id) const = 0;
    virtual std::string name() const = 0;
};

// One hash bucket per H3 cell. Every management partition uses this exact
// implementation; only the set of cells owned by a leaf differs.
class H3EntityIndex {
public:
    void add(H3Index cell, Id id);
    void remove(H3Index cell, Id id);
    const std::vector<Id>* bucket(H3Index cell) const;
    std::size_t bucket_count() const { return buckets_.size(); }
    std::size_t entity_count() const { return entity_count_; }
    std::size_t estimated_bytes() const;

private:
    std::unordered_map<H3Index, std::vector<Id>> buckets_;
    std::size_t entity_count_{};
};

class GlobalH3Index final : public ManagementIndex {
public:
    explicit GlobalH3Index(std::shared_ptr<CellCover> cover);

    void set(Id id, Point point) override;
    void erase(Id id) override;
    QueryResult query(Query query) const override;

    std::size_t size() const override { return points_.size(); }
    std::size_t partition_count() const override { return 1; }
    std::size_t node_count() const override { return 1; }
    std::size_t bucket_count() const override { return index_.bucket_count(); }
    std::size_t estimated_bytes() const override;
    std::optional<Point> point_of(Id id) const override;
    std::string name() const override { return "GLOBAL_H3"; }

private:
    std::shared_ptr<CellCover> cover_;
    std::unordered_map<Id, Point> points_;
    std::unordered_map<Id, H3Index> cell_of_;
    H3EntityIndex index_;
};

struct PartitionLeaf {
    int id{-1};
    std::vector<CellId> cells;
    H3EntityIndex index;
    std::uint64_t entity_count{};
};

class PartitionedBase : public ManagementIndex {
public:
    explicit PartitionedBase(std::shared_ptr<CellCover> cover);

    void set(Id id, Point point) override;
    void erase(Id id) override;
    QueryResult query(Query query) const override;

    std::size_t size() const override { return points_.size(); }
    std::size_t bucket_count() const override;
    std::size_t estimated_bytes() const override;
    std::optional<Point> point_of(Id id) const override;

    const std::vector<int>& cell_assignment() const { return cell_to_leaf_; }
    const std::shared_ptr<CellCover>& cell_cover() const { return cover_; }

protected:
    virtual void on_insert(int leaf_id, H3Index cell) { (void)leaf_id; (void)cell; }
    virtual void on_erase(int leaf_id, H3Index cell) { (void)leaf_id; (void)cell; }

    PartitionLeaf& leaf_of(H3Index cell) const;
    int leaf_id_of(H3Index cell) const;
    void register_leaf(std::unique_ptr<PartitionLeaf> leaf);
    PartitionLeaf* find_leaf(int leaf_id) const;

    std::shared_ptr<CellCover> cover_;
    std::unordered_map<Id, Point> points_;
    std::unordered_map<Id, H3Index> cell_of_;
    std::unordered_map<int, std::unique_ptr<PartitionLeaf>> leaves_;
    std::vector<int> cell_to_leaf_;
    std::size_t active_leaves_{};
};

class FixedPartitionIndex final : public PartitionedBase {
public:
    FixedPartitionIndex(std::shared_ptr<CellCover> cover,
                        std::vector<int> assignment, int partition_count);

    std::size_t partition_count() const override { return active_leaves_; }
    std::size_t node_count() const override { return active_leaves_; }
    std::string name() const override { return "FIXED_PARTITION"; }
};

// H3-RTM execution substrate: each connected management region is assigned to
// one worker, and each worker owns the H3-cell buckets of its regions. The
// authoritative point table is shared; worker indexes store IDs only.
class WorkerShardedIndex {
public:
    WorkerShardedIndex(std::shared_ptr<CellCover> cover,
                       std::vector<int> region_assignment,
                       std::vector<int> region_to_worker);

    void set(Id id, Point point);
    void erase(Id id);
    QueryResult query(Query query) const;

    std::size_t size() const { return points_.size(); }
    std::size_t worker_count() const { return worker_indexes_.size(); }
    std::size_t partition_count() const { return region_to_worker_.size(); }
    const std::vector<int>& cell_to_worker() const { return cell_to_worker_; }
    const std::vector<int>& region_to_worker() const { return region_to_worker_; }
    std::size_t worker_entity_count(int worker) const;
    std::size_t total_bucket_count() const;
    bool validate(std::string* why = nullptr) const;

private:
    std::shared_ptr<CellCover> cover_;
    std::vector<int> cell_to_worker_;
    std::vector<int> region_to_worker_;
    std::vector<H3EntityIndex> worker_indexes_;
    std::unordered_map<Id, Point> points_;
    std::unordered_map<Id, H3Index> cell_of_;
};

class CapacityPartitionIndex final : public PartitionedBase {
public:
    CapacityPartitionIndex(std::shared_ptr<CellCover> cover,
                           std::size_t split_threshold = 512,
                           std::size_t merge_threshold = 128);

    std::size_t partition_count() const override { return active_leaves_; }
    std::size_t node_count() const override { return node_count_; }
    std::string name() const override { return "CAPACITY_TREE"; }
    std::size_t split_count() const { return split_count_; }
    std::size_t merge_count() const { return merge_count_; }

private:
    struct Node {
        int id{-1};
        Node* parent{};
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        bool leaf{true};
        int leaf_id{-1};
        std::vector<CellId> cells;
        std::uint64_t entity_count{};
    };

    std::unique_ptr<Node> root_;
    std::unordered_map<int, Node*> node_of_leaf_;
    int next_node_id_{};
    int next_leaf_id_{};
    std::size_t split_threshold_{512};
    std::size_t merge_threshold_{128};
    std::size_t split_count_{};
    std::size_t merge_count_{};
    std::size_t node_count_{};

    void on_insert(int leaf_id, H3Index cell) override;
    void on_erase(int leaf_id, H3Index cell) override;

    Node* node_for_leaf(int leaf_id) const;
    Node* make_node(Node* parent);
    void maybe_split(Node* node);
    bool split_node(Node* node);
    void try_merge_up(Node* node);
    bool merge_node(Node* node);
    void recompute_leaf_cells(Node* node);
};

// Build a connected k-way assignment by recursively cutting a BFS spanning
// tree of the current part. Both sides of each cut are connected in the H3
// adjacency graph.
std::vector<int> build_connected_assignment(
    const CellCover& cover, const std::vector<double>& cell_weights,
    int target_partitions, std::uint32_t seed = 1);

bool split_connected_cells(const CellCover& cover,
                           const std::vector<CellId>& cells,
                           const std::vector<double>& cell_weights,
                           std::vector<CellId>& left,
                           std::vector<CellId>& right);

double distance_point_box(const Point& point, const Box& box);

}  // namespace spatial::h3
