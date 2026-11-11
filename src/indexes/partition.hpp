#pragma once
#include "core/index.hpp"
#include "indexes/regional.hpp"
#include <array>
#include <optional>
#include <unordered_map>
#include <memory>
#include <vector>

namespace spatial {
// Load-aware adaptive partition. The world is divided into 64x64 sub-cells
// (the legacy 16x16 grid refined 4x4); a leaf covers an axis-aligned rectangle
// of sub-cells, so leaves can be larger than a legacy cell (merged) or smaller
// (split). The partition tree owns spatial routing and one leaf-local index
// (SCAN / FINE_GRID / LOCAL_QUAD); LOCAL_QUAD here is a leaf-local object index,
// not the partition tree itself.
class PartitionIndex final : public Index {
public:
    static constexpr unsigned base_side = 16;
    static constexpr unsigned sub = 4;
    static constexpr unsigned side = 64;

    struct LeafTimes {
        std::uint64_t queries{}, updates{}, local_updates{}, builds{}, switches{};
        std::int64_t query_ns{}, base_ns{}, local_ns{}, build_ns{}, switch_ns{};
    };

    PartitionIndex(Box world, int fine_cell, int quad_capacity, bool profile = false,
                   bool observe = false);
    void set_partition(const std::vector<std::array<int, 4>>& rects,
                       const std::vector<unsigned>& levels);
    void set_mode(unsigned leaf, Mode mode);

    unsigned leaf_of(Point p) const;
    unsigned leaf_count() const { return static_cast<unsigned>(leaves_.size()); }
    std::size_t entity_count(unsigned leaf) const { return leaves_.at(leaf).ids.size(); }
    Mode mode(unsigned leaf) const { return leaves_.at(leaf).mode; }
    const WindowLoad& window_load(unsigned leaf) const { return loads_.at(leaf); }
    const LeafTimes& leaf_times(unsigned leaf) const { return times_.at(leaf); }
    void reset_window();

    void set(Id id, Point point) override;
    void erase(Id id) override;
    Answer query(Query q) const override;

private:
    struct Leaf {
        Box box;
        unsigned level{1};
        std::vector<Id> ids;
        std::unique_ptr<Index> aux;
        Mode mode{Mode::Scan};
    };
    struct Record {
        Point point;
        unsigned leaf;
        std::size_t slot;
    };

    Box leaf_box(unsigned leaf) const;
    unsigned sub_of(Point p) const;
    void maintain(unsigned leaf, Id id, std::optional<Point> point);
    void detach(Leaf& leaf, const Record& record);
    void rebuild_leaf(unsigned leaf);
    void insert(Id id, Point point) override { set(id, point); }
    void remove(Id id, Point) override { erase(id); }

    Box world_;
    double width_{}, height_{};
    int fine_cell_, quad_capacity_;
    bool profile_, observe_;
    std::vector<Leaf> leaves_;
    std::vector<std::uint16_t> leaf_of_;
    std::unordered_map<Id, Record> objects_;
    mutable std::vector<WindowLoad> loads_;
    mutable std::vector<LeafTimes> times_;
};
}  // namespace spatial
