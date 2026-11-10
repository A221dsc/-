#include "core/index.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace spatial {
namespace {
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using BP = bg::model::point<double, 2, bg::cs::cartesian>;
using Value = std::pair<BP,Id>;
class BoostRstar final : public Index {
    bgi::rtree<Value, bgi::dynamic_rstar> tree_;
    void insert(Id id, Point p) override { tree_.insert({BP(p.x,p.y),id}); }
    void remove(Id id, Point p) override {
        if (tree_.remove({BP(p.x,p.y),id}) != 1)
            throw std::logic_error("missing Boost R*-Tree entry");
    }
public:
    explicit BoostRstar(int cap) : tree_(bgi::dynamic_rstar(cap)) {}
    Answer query(Query q) const override {
        bg::model::box<BP> box(BP(q.box.x0,q.box.y0),BP(q.box.x1,q.box.y1));
        Answer a;
        for (auto it = tree_.qbegin(bgi::intersects(box)); it != tree_.qend(); ++it)
            refine(a,q,it->second,{bg::get<0>(it->first),bg::get<1>(it->first)});
        return a;
    }
};
}
std::unique_ptr<Index> make_boost(int capacity) {
    return std::make_unique<BoostRstar>(capacity);
}
}
