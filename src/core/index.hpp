#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spatial {
using Id = std::int64_t;
struct Point {
    double x{}, y{};
    bool operator==(const Point&) const = default;
};
struct Box {
    double x0{}, y0{}, x1{}, y1{};
    bool contains(Point p) const {
        return x0 <= p.x && p.x <= x1 && y0 <= p.y && p.y <= y1;
    }
    bool overlaps(Box b) const {
        return x0 <= b.x1 && b.x0 <= x1 && y0 <= b.y1 && b.y0 <= y1;
    }
};
struct Query {
    Box box;
    bool circular{};
    Point center;
    double radius{};
    bool matches(Point p) const {
        if (!box.contains(p)) return false;
        if (!circular) return true;
        const double dx = p.x - center.x, dy = p.y - center.y;
        return dx*dx + dy*dy <= radius*radius;
    }
};
inline Query circle(Point p, double r) {
    return {{p.x-r,p.y-r,p.x+r,p.y+r},true,p,r};
}
struct Answer {
    std::vector<Id> ids;
    std::uint64_t candidates{};
    std::optional<std::uint64_t> visits;
};

class Index {
public:
    virtual ~Index() = default;
    virtual void set(Id id, Point point) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y))
            throw std::invalid_argument("nonfinite point");
        auto old = points_.find(id);
        if (old == points_.end()) {
            insert(id, point);
            points_.emplace(id, point);
        } else if (old->second != point) {
            move(id, old->second, point);
            old->second = point;
        }
    }
    virtual void erase(Id id) {
        auto old = points_.find(id);
        if (old == points_.end()) return;
        remove(id, old->second);
        points_.erase(old);
    }
    virtual Answer query(Query q) const = 0;
    virtual std::size_t nodes() const { return 0; }
    std::size_t size() const { return points_.size(); }
    std::optional<Point> point_of(Id id) const {
        auto it = points_.find(id);
        if (it == points_.end()) return std::nullopt;
        return it->second;
    }
protected:
    std::unordered_map<Id, Point> points_;
    virtual void insert(Id, Point) = 0;
    virtual void remove(Id, Point) = 0;
    virtual void move(Id id, Point from, Point to) {
        remove(id, from);
        insert(id, to);
    }
    void refine(Answer& a, Query q, Id id, Point point) const {
        ++a.candidates;
        if (q.matches(point)) a.ids.push_back(id);
    }
};

std::unique_ptr<Index> make_grid(int cell_m);
std::unique_ptr<Index> make_quad(int capacity, Box bounds);
std::unique_ptr<Index> make_boost(int capacity);
std::unique_ptr<Index> make_lsi(int capacity);
inline std::unique_ptr<Index> make_index(std::string_view name, int param, Box bounds) {
    if (param <= 0) throw std::invalid_argument("index parameter must be positive");
    if (name == "UNIFORM_GRID") return make_grid(param);
    if (name == "PR_QUADTREE") return make_quad(param, bounds);
    if (name == "RSTAR_BOOST") return make_boost(param);
    if (name == "RSTAR_LIBSPATIALINDEX") return make_lsi(param);
    throw std::invalid_argument("unknown index");
}
}
