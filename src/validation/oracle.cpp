#include "oracle.hpp"
#include <algorithm>
namespace spatial {
std::vector<Id> Oracle::query(Query q) const {
    std::vector<Id> result;
    for (auto [id, p] : rows_) {
        if (q.circular) {
            if (p.x < q.center.x-q.radius || p.x > q.center.x+q.radius ||
                p.y < q.center.y-q.radius || p.y > q.center.y+q.radius)
                continue;
            const double x = q.center.x - p.x;
            const double y = q.center.y - p.y;
            if (x*x + y*y > q.radius*q.radius) continue;
        } else if (p.x < q.box.x0 || p.x > q.box.x1 ||
                   p.y < q.box.y0 || p.y > q.box.y1) {
            continue;
        }
        result.push_back(id);
    }
    return result;
}

void ArrayOracle::set(Id id, Point p) {
    auto found = slot_.find(id);
    if (found != slot_.end()) {
        xs_[found->second] = p.x;
        ys_[found->second] = p.y;
        alive_[found->second] = 1;
        return;
    }
    slot_.emplace(id, ids_.size());
    ids_.push_back(id);
    xs_.push_back(p.x);
    ys_.push_back(p.y);
    alive_.push_back(1);
}

void ArrayOracle::erase(Id id) {
    const auto found = slot_.find(id);
    if (found != slot_.end()) alive_[found->second] = 0;
}

std::vector<Id> ArrayOracle::query(Query q) const {
    std::vector<Id> result;
    const double left = q.box.x0, right = q.box.x1, bottom = q.box.y0, top = q.box.y1;
    if (!q.circular) {
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            const double x = xs_[i], y = ys_[i];
            if (alive_[i] && left <= x && x <= right && bottom <= y && y <= top)
                result.push_back(ids_[i]);
        }
    } else {
        const double dx = q.center.x, dy = q.center.y, r2 = q.radius * q.radius;
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            const double x = xs_[i], y = ys_[i];
            if (!alive_[i] || x < left || x > right || y < bottom || y > top) continue;
            const double ex = x - dx, ey = y - dy;
            if (ex * ex + ey * ey <= r2) result.push_back(ids_[i]);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}
}
