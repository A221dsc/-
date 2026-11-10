#include "oracle.hpp"
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
}
