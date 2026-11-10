#pragma once
#include "core/index.hpp"
#include <array>

namespace spatial {
enum class Mode { Scan, FineGrid, LocalQuad };
struct WindowLoad {
    std::uint64_t queries{}, updates{};
    double coverage{};
};
struct RegionalStats {
    std::uint64_t queries{}, object_updates{}, local_updates{}, builds{}, switches{};
    std::int64_t query_ns{}, local_update_ns{}, build_ns{}, switch_ns{};
};
struct RegionalTimes {
    std::int64_t base_ns{}, local_ns{}, switch_ns{};
    std::uint64_t local_updates{}, switches{};
};

// Mutations and mode changes complete synchronously between replay events.
class AdaptiveRegionalIndex final : public Index {
public:
    static constexpr unsigned side = 16;
    static constexpr unsigned region_count = side * side;

    AdaptiveRegionalIndex(Box world, Mode mode, int fine_cell, int quad_capacity,
                          bool profile = false, bool observe = false);
    void set(Id id, Point point) override;
    void erase(Id id) override;
    Answer query(Query q) const override;
    void switch_mode(unsigned region, Mode mode);
    unsigned region_of(Point p) const;
    std::size_t entity_count(unsigned region) const;
    Mode mode(unsigned region) const { return modes_.at(region); }
    RegionalTimes times() const { return times_; }
    const RegionalStats& region_stats(unsigned region) const { return stats_.at(region); }
    const WindowLoad& window_load(unsigned region) const { return loads_.at(region); }
    void reset_window() { stats_ = {}; loads_ = {}; }

private:
    struct Record {
        Point point;
        unsigned region;
        std::size_t slot;
    };
    struct Region {
        std::vector<Id> ids;
        std::unique_ptr<Index> auxiliary;
    };

    Box world_;
    double width_, height_;
    int fine_cell_, quad_capacity_;
    bool profile_, observe_;
    mutable std::array<WindowLoad, region_count> loads_{};
    std::unordered_map<Id, Record> objects_;
    std::unordered_map<unsigned, Region> regions_;
    std::array<Mode, region_count> modes_;
    mutable std::array<RegionalStats, region_count> stats_{};
    RegionalTimes times_;

    Box region_box(unsigned region) const;
    std::unique_ptr<Index> build(unsigned region, Mode mode, const Region& bucket);
    void detach(Region& bucket, const Record& record);
    void maintain(unsigned region, Region& bucket, Id id, std::optional<Point> point);
    void insert(Id id, Point p) override;
    void remove(Id id, Point p) override;
};

std::optional<Mode> fixed_mode(std::string_view name);
std::string_view mode_name(Mode mode);
}
