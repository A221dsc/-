#include "regional.hpp"
#include "core/clock.hpp"
#include <algorithm>

namespace spatial {
AdaptiveRegionalIndex::AdaptiveRegionalIndex(Box world, Mode mode, int fine_cell,
                                             int quad_capacity, bool profile)
    : world_(world), width_((world.x1-world.x0)/side), height_((world.y1-world.y0)/side),
      fine_cell_(fine_cell), quad_capacity_(quad_capacity), profile_(profile) {
    if (!(width_ > 0 && height_ > 0 && std::isfinite(width_) && std::isfinite(height_))
        || fine_cell <= 0 || quad_capacity <= 0)
        throw std::invalid_argument("invalid regional geometry or parameters");
    modes_.fill(mode);
}

unsigned AdaptiveRegionalIndex::region_of(Point p) const {
    // Extend edge regions to retain real trajectory points beyond the query world.
    auto bin = [](double value, double origin, double span) {
        return static_cast<unsigned>(std::clamp((value-origin)/span, 0.0, double(side-1)));
    };
    return side * bin(p.y,world_.y0,height_) + bin(p.x,world_.x0,width_);
}

Box AdaptiveRegionalIndex::region_box(unsigned r) const {
    const unsigned x = r%side, y = r/side;
    return {world_.x0+width_*x, world_.y0+height_*y,
            world_.x0+width_*(x+1), world_.y0+height_*(y+1)};
}

std::size_t AdaptiveRegionalIndex::entity_count(unsigned region) const {
    const auto found = regions_.find(region);
    return found == regions_.end() ? 0 : found->second.ids.size();
}

std::unique_ptr<Index> AdaptiveRegionalIndex::build(unsigned r, Mode mode, const Region& bucket) {
    if (mode == Mode::Scan || bucket.ids.empty()) return {};
    const auto begin = profile_ ? Clock::now() : decltype(Clock::now()){};
    auto next = mode == Mode::FineGrid ? make_grid(fine_cell_)
                                       : make_quad(quad_capacity_,region_box(r));
    for (Id id : bucket.ids) next->set(id,objects_.at(id).point);
    if (profile_) {
        stats_[r].build_ns += (Clock::now()-begin).count();
        ++stats_[r].builds;
    }
    return next;
}

void AdaptiveRegionalIndex::switch_mode(unsigned r, Mode target) {
    auto& mode = modes_.at(r);
    if (mode == target) return;
    const auto begin = Clock::now();
    auto found = regions_.find(r);
    if (found != regions_.end()) {
        auto next = build(r,target,found->second);
        // Publish only after the replacement is ready; assignment releases the old index.
        found->second.auxiliary = std::move(next);
    }
    mode = target;
    const auto elapsed = (Clock::now()-begin).count();
    times_.switch_ns += elapsed;
    ++times_.switches;
    if (profile_) {
        stats_[r].switch_ns += elapsed;
        ++stats_[r].switches;
    }
}

void AdaptiveRegionalIndex::detach(Region& bucket, const Record& record) {
    auto& ids = bucket.ids;
    if (record.slot != ids.size()-1) {
        ids[record.slot] = ids.back();
        objects_.at(ids.back()).slot = record.slot;
    }
    ids.pop_back();
}

void AdaptiveRegionalIndex::maintain(unsigned r, Region& bucket, Id id, std::optional<Point> point) {
    if (profile_) ++stats_[r].object_updates;
    if (modes_[r] == Mode::Scan) return;
    const auto begin = profile_ ? Clock::now() : decltype(Clock::now()){};
    if (!bucket.auxiliary) bucket.auxiliary = build(r,modes_[r],bucket);
    else if (point) bucket.auxiliary->set(id,*point);
    else bucket.auxiliary->erase(id);
    if (profile_) {
        const auto elapsed = (Clock::now()-begin).count();
        times_.local_ns += elapsed;
        ++times_.local_updates;
        stats_[r].local_update_ns += elapsed;
        ++stats_[r].local_updates;
    }
}

void AdaptiveRegionalIndex::set(Id id, Point point) {
    const auto begin = profile_ ? Clock::now() : decltype(Clock::now()){};
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
        throw std::invalid_argument("nonfinite point");
    auto old = objects_.find(id);
    if (old != objects_.end() && old->second.point == point) {
        if (profile_) times_.base_ns += (Clock::now()-begin).count();
        if (profile_) ++stats_[old->second.region].object_updates;
        return;
    }
    const unsigned next = region_of(point);
    Region* previous_bucket = nullptr;
    unsigned previous = next;
    Region* bucket;
    if (old != objects_.end() && old->second.region == next) {
        bucket = &regions_.at(next);
        old->second.point = point;
    } else {
        bucket = &regions_[next];
        const auto slot = bucket->ids.size();
        bucket->ids.push_back(id);
        if (old == objects_.end()) {
            objects_.emplace(id,Record{point,next,slot});
        } else {
            previous = old->second.region;
            previous_bucket = &regions_.at(previous);
            detach(*previous_bucket,old->second);
            old->second = {point,next,slot};
        }
    }
    if (profile_) times_.base_ns += (Clock::now()-begin).count();
    if (previous_bucket) maintain(previous,*previous_bucket,id,std::nullopt);
    maintain(next,*bucket,id,point);
}

void AdaptiveRegionalIndex::erase(Id id) {
    const auto begin = profile_ ? Clock::now() : decltype(Clock::now()){};
    const auto found = objects_.find(id);
    if (found == objects_.end()) {
        if (profile_) times_.base_ns += (Clock::now()-begin).count();
        return;
    }
    const unsigned r = found->second.region;
    auto& bucket = regions_.at(r);
    detach(bucket,found->second);
    objects_.erase(found);
    if (profile_) times_.base_ns += (Clock::now()-begin).count();
    maintain(r,bucket,id,std::nullopt);
}

void AdaptiveRegionalIndex::insert(Id id, Point point) { set(id,point); }
void AdaptiveRegionalIndex::remove(Id id, Point) { erase(id); }

Answer AdaptiveRegionalIndex::query(Query q) const {
    Answer result;
    result.visits = 0;
    const unsigned lo = region_of({q.box.x0,q.box.y0});
    const unsigned hi = region_of({q.box.x1,q.box.y1});
    for (unsigned y = lo/side; y <= hi/side; ++y) {
        for (unsigned x = lo%side; x <= hi%side; ++x) {
            const unsigned r = side*y+x;
            const auto begin = profile_ ? Clock::now() : decltype(Clock::now()){};
            ++*result.visits;
            const auto found = regions_.find(r);
            if (found != regions_.end()) {
                const auto& bucket = found->second;
                if (bucket.auxiliary) {
                    auto partial = bucket.auxiliary->query(q);
                    result.candidates += partial.candidates;
                    *result.visits += partial.visits.value();
                    if (result.ids.empty()) result.ids = std::move(partial.ids);
                    else result.ids.insert(result.ids.end(),partial.ids.begin(),partial.ids.end());
                } else {
                    for (Id id : bucket.ids) refine(result,q,id,objects_.at(id).point);
                }
            }
            if (profile_) {
                stats_[r].query_ns += (Clock::now()-begin).count();
                ++stats_[r].queries;
            }
        }
    }
    return result;
}

std::optional<Mode> fixed_mode(std::string_view name) {
    if (name == "S0_ALL_SCAN") return Mode::Scan;
    if (name == "S1_ALL_FINE") return Mode::FineGrid;
    if (name == "S2_ALL_QUAD") return Mode::LocalQuad;
    return std::nullopt;
}

std::string_view mode_name(Mode mode) {
    switch (mode) {
        case Mode::Scan: return "SCAN";
        case Mode::FineGrid: return "FINE_GRID";
        case Mode::LocalQuad: return "LOCAL_QUAD";
    }
    throw std::invalid_argument("unknown region mode");
}
} // namespace spatial
