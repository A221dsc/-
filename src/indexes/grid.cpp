#include "core/index.hpp"
#include <algorithm>

namespace spatial {
namespace {
struct Cell {
    std::int64_t x, y;
    bool operator==(const Cell&) const = default;
};
struct CellHash {
    std::size_t operator()(Cell c) const {
        auto x = std::hash<std::int64_t>{}(c.x);
        auto y = std::hash<std::int64_t>{}(c.y);
        return x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2));
    }
};
class Grid final : public Index {
    std::unordered_map<Cell, std::vector<Id>, CellHash> cells_;
    std::unordered_map<Id, std::size_t> slots_;
    double side_;
    Cell locate(Point p) const {
        return {static_cast<std::int64_t>(std::floor(p.x/side_)),
                static_cast<std::int64_t>(std::floor(p.y/side_))};
    }
    void insert(Id id, Point p) override {
        auto& ids = cells_[locate(p)];
        slots_.emplace(id, ids.size());
        ids.push_back(id);
    }
    void remove(Id id, Point p) override {
        auto cell = cells_.find(locate(p));
        auto& ids = cell->second;
        auto slot = slots_.find(id);
        ids[slot->second] = ids.back();
        slots_.at(ids.back()) = slot->second;
        slots_.erase(slot);
        ids.pop_back();
        if (ids.empty()) cells_.erase(cell);
    }
    void move(Id id, Point from, Point to) override {
        if (locate(from) != locate(to)) Index::move(id,from,to);
    }
public:
    explicit Grid(int side) : side_(side) {}
    Answer query(Query q) const override {
        Answer a;
        a.visits = 0;
        auto lo = locate({q.box.x0,q.box.y0}), hi = locate({q.box.x1,q.box.y1});
        auto scan = [&](const auto& ids) {
            for (Id id : ids) refine(a, q, id, points_.at(id));
        };
        const auto cell_count = (double(hi.x)-double(lo.x)+1)*(double(hi.y)-double(lo.y)+1);
        if (cell_count > 4.0*cells_.size()) {
            for (const auto& [cell, ids] : cells_) {
                ++*a.visits;
                if (lo.x <= cell.x && cell.x <= hi.x && lo.y <= cell.y && cell.y <= hi.y)
                    scan(ids);
            }
        } else {
            for (auto x = lo.x; x <= hi.x; ++x)
                for (auto y = lo.y; y <= hi.y; ++y) {
                    ++*a.visits;
                    auto it = cells_.find({x,y});
                    if (it != cells_.end()) scan(it->second);
                }
        }
        return a;
    }
};
}
std::unique_ptr<Index> make_grid(int side) { return std::make_unique<Grid>(side); }
}
