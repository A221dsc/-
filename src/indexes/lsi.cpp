#include "core/index.hpp"
#include <algorithm>
#include <spatialindex/capi/sidx_api.h>

namespace spatial {
namespace {
void check(RTError result) {
    if (result != RT_None) throw std::runtime_error("libspatialindex C API failed");
}
class LsiRstar final : public Index {
    IndexH tree_;
    void insert(Id id, Point p) override {
        double xy[]{p.x,p.y};
        check(Index_InsertData(tree_,id,xy,xy,2,nullptr,0));
    }
    void remove(Id id, Point p) override {
        double xy[]{p.x,p.y};
        check(Index_DeleteData(tree_,id,xy,xy,2));
    }
public:
    explicit LsiRstar(int cap) {
        std::unique_ptr<Tools_PropertySet, decltype(&IndexProperty_Destroy)> p(
            IndexProperty_Create(), IndexProperty_Destroy);
        if (!p) throw std::runtime_error("cannot create libspatialindex properties");
        check(IndexProperty_SetIndexType(p.get(),RT_RTree));
        check(IndexProperty_SetIndexVariant(p.get(),RT_Star));
        check(IndexProperty_SetIndexStorage(p.get(),RT_Memory));
        check(IndexProperty_SetDimension(p.get(),2));
        check(IndexProperty_SetIndexCapacity(p.get(),cap));
        check(IndexProperty_SetLeafCapacity(p.get(),cap));
        check(IndexProperty_SetNearMinimumOverlapFactor(p.get(),std::min(cap,32)));
        check(IndexProperty_SetFillFactor(p.get(),0.7));
        check(IndexProperty_SetSplitDistributionFactor(p.get(),0.4));
        check(IndexProperty_SetReinsertFactor(p.get(),0.3));
        tree_ = Index_Create(p.get());
        if (!tree_) throw std::runtime_error("cannot create libspatialindex R*-Tree");
    }
    ~LsiRstar() override { Index_Destroy(tree_); }
    Answer query(Query q) const override {
        double lo[]{q.box.x0,q.box.y0}, hi[]{q.box.x1,q.box.y1};
        Id* ids = nullptr;
        std::uint64_t size{};
        auto status = Index_Intersects_id(tree_,lo,hi,2,&ids,&size);
        std::unique_ptr<Id, decltype(&Index_Free)> owned(ids,Index_Free);
        check(status);
        Answer a;
        a.ids.reserve(size);
        for (std::uint64_t i = 0; i < size; ++i) refine(a,q,ids[i],points_.at(ids[i]));
        return a;
    }
};
}
std::unique_ptr<Index> make_lsi(int capacity) {
    if (capacity < 4) throw std::invalid_argument("libspatialindex capacity below 4");
    return std::make_unique<LsiRstar>(capacity);
}
}
