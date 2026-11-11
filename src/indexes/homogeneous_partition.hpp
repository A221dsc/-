#ifndef SPATIAL_HOMOGENEOUS_PARTITION_HPP
#define SPATIAL_HOMOGENEOUS_PARTITION_HPP

#include "core/index.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace spatial {

// One PR Quadtree per partition with a minimal global owner map.
//
// The container keeps only id -> leaf id; the authoritative point lives in the
// leaf's Quad (which, like the direct Quad, keeps its own id->point map and
// tree entries). No per-leaf id vector and no second coordinate copy is added
// by the partition layer.
class HomogeneousPartitionIndex final : public Index {
public:
    HomogeneousPartitionIndex(Box world, int quad_capacity, bool profile = false);

    void set_partition(const std::vector<std::array<int, 4>>& rects,
                       const std::vector<unsigned>& levels);
    unsigned leaf_of(Point p) const;
    unsigned leaf_count() const { return static_cast<unsigned>(quads_.size()); }
    std::size_t entity_count(unsigned leaf) const;
    std::size_t index_nodes() const;
    void reset_window() {}

    void insert(Id, Point) override { throw std::logic_error("direct insert unused"); }
    void remove(Id, Point) override { throw std::logic_error("direct remove unused"); }
    void set(Id id, Point point) override;
    void erase(Id id) override;
    Answer query(Query q) const override;

    // Diagnostic counters (always on, cheap); the profile timers are only
    // active when the constructor's profile flag is set.
    std::uint64_t route_cells() const { return route_cells_; }
    std::uint64_t touched_leaves() const { return touched_leaves_; }
    std::uint64_t local_queries() const { return local_queries_; }
    double route_ms() const { return route_ms_; }
    double local_ms() const { return local_ms_; }

private:
    static constexpr int side = 64;
    Box world_;
    int capacity_{32};
    bool profile_{false};
    double cell_w_{1.0};
    double cell_h_{1.0};
    std::vector<std::array<int, 4>> rects_;
    std::vector<std::unique_ptr<Index>> quads_;
    std::vector<int> leaf_of_cell_;
    std::unordered_map<Id, unsigned> owner_;
    mutable std::vector<std::uint32_t> seen_;
    mutable std::vector<unsigned> scratch_;
    mutable std::uint32_t generation_{0};
    mutable std::uint64_t route_cells_{};
    mutable std::uint64_t touched_leaves_{};
    mutable std::uint64_t local_queries_{};
    mutable double route_ms_{};
    mutable double local_ms_{};

    unsigned cell_of(Point p) const;
    std::unique_ptr<Index> make_leaf(unsigned leaf) const;
};

}  // namespace spatial
#endif
