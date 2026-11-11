#include "partition.hpp"
#include "core/clock.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spatial {
PartitionIndex::PartitionIndex(Box world, int fine_cell, int quad_capacity, bool profile,
                               bool observe)
    : world_(world), width_((world.x1 - world.x0) / side),
      height_((world.y1 - world.y0) / side), fine_cell_(fine_cell),
      quad_capacity_(quad_capacity), profile_(profile), observe_(observe || profile) {
    if (!(width_ > 0 && height_ > 0 && std::isfinite(width_) && std::isfinite(height_)) ||
        fine_cell <= 0 || quad_capacity <= 0)
        throw std::invalid_argument("invalid partition geometry");
    if (profile_) (void)Tick::ns_per_tick();
    leaf_of_.assign(side * side, 0);
}

unsigned PartitionIndex::sub_of(Point p) const {
    auto bin = [](double value, double origin, double span) {
        return static_cast<unsigned>(std::clamp((value - origin) / span, 0.0, double(side - 1)));
    };
    return side * bin(p.y, world_.y0, height_) + bin(p.x, world_.x0, width_);
}

unsigned PartitionIndex::leaf_of(Point p) const { return leaf_of_[sub_of(p)]; }

Box PartitionIndex::leaf_box(unsigned leaf) const { return leaves_.at(leaf).box; }

void PartitionIndex::set_partition(const std::vector<std::array<int, 4>>& rects,
                                   const std::vector<unsigned>& levels) {
    if (rects.empty() || rects.size() != levels.size())
        throw std::invalid_argument("partition rects and levels must match");
    if (rects.size() > 4096) throw std::invalid_argument("too many partition leaves");
    leaves_.clear();
    leaves_.resize(rects.size());
    leaf_of_.assign(side * side, 0xFFFF);
    for (std::size_t i = 0; i < rects.size(); ++i) {
        const auto& r = rects[i];
        if (r[0] < 0 || r[1] < 0 || r[2] <= r[0] || r[3] <= r[1] || r[2] > int(side) ||
            r[3] > int(side))
            throw std::invalid_argument("partition rect out of range");
        leaves_[i].level = levels[i];
        leaves_[i].box = {world_.x0 + r[0] * width_, world_.y0 + r[1] * height_,
                          world_.x0 + r[2] * width_, world_.y0 + r[3] * height_};
        for (int y = r[1]; y < r[3]; ++y) {
            for (int x = r[0]; x < r[2]; ++x) {
                const auto index = static_cast<std::size_t>(y) * side + x;
                if (leaf_of_[index] != 0xFFFF)
                    throw std::invalid_argument("partition rects overlap");
                leaf_of_[index] = static_cast<std::uint16_t>(i);
            }
        }
    }
    if (std::find(leaf_of_.begin(), leaf_of_.end(), 0xFFFF) != leaf_of_.end())
        throw std::invalid_argument("partition does not cover the world");
    loads_.assign(leaves_.size(), WindowLoad{});
    times_.assign(leaves_.size(), LeafTimes{});
}

void PartitionIndex::rebuild_leaf(unsigned index) {
    auto& leaf = leaves_.at(index);
    leaf.aux.reset();
    if (leaf.mode == Mode::Scan) return;
    const auto begin = profile_ ? Tick::now() : 0;
    auto next = leaf.mode == Mode::FineGrid ? make_grid(fine_cell_)
                                            : make_quad(quad_capacity_, leaf.box);
    for (Id id : leaf.ids) next->set(id, objects_.at(id).point);
    leaf.aux = std::move(next);
    if (profile_) {
        const auto elapsed = Tick::ns(Tick::now() - begin);
        times_[index].build_ns += elapsed;
        ++times_[index].builds;
    }
}

void PartitionIndex::set_mode(unsigned leaf, Mode mode) {
    auto& target = leaves_.at(leaf);
    if (target.mode == mode) return;
    const auto begin = Clock::now();
    target.mode = mode;
    rebuild_leaf(leaf);
    const auto elapsed = (Clock::now() - begin).count();
    if (profile_) {
        times_[leaf].switch_ns += elapsed;
        ++times_[leaf].switches;
    }
}

void PartitionIndex::detach(Leaf& leaf, const Record& record) {
    auto& ids = leaf.ids;
    if (record.slot != ids.size() - 1) {
        ids[record.slot] = ids.back();
        objects_.at(ids.back()).slot = record.slot;
    }
    ids.pop_back();
}

void PartitionIndex::maintain(unsigned leaf, Id id, std::optional<Point> point) {
    if (observe_) ++loads_[leaf].updates;
    auto& target = leaves_[leaf];
    if (target.mode == Mode::Scan) return;
    const auto begin = profile_ ? Tick::now() : 0;
    if (!target.aux) rebuild_leaf(leaf);
    else if (point) target.aux->set(id, *point);
    else target.aux->erase(id);
    if (profile_) {
        const auto elapsed = Tick::ns(Tick::now() - begin);
        times_[leaf].local_ns += elapsed;
        ++times_[leaf].local_updates;
    }
}

void PartitionIndex::set(Id id, Point point) {
    const auto begin = profile_ ? Tick::now() : 0;
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
        throw std::invalid_argument("nonfinite point");
    const unsigned next = leaf_of(point);
    auto old = objects_.find(id);
    if (old != objects_.end() && old->second.point == point) {
        if (profile_) times_[old->second.leaf].base_ns += Tick::ns(Tick::now() - begin);
        if (observe_) ++loads_[old->second.leaf].updates;
        return;
    }
    unsigned previous = next;
    bool moved = false;
    if (old != objects_.end() && old->second.leaf == next) {
        old->second.point = point;
    } else {
        auto& bucket = leaves_[next];
        bucket.ids.push_back(id);
        const auto slot = bucket.ids.size() - 1;
        if (old == objects_.end()) {
            objects_.emplace(id, Record{point, next, slot});
        } else {
            previous = old->second.leaf;
            detach(leaves_[previous], old->second);
            old->second = {point, next, slot};
            moved = true;
        }
    }
    if (profile_) times_[next].base_ns += Tick::ns(Tick::now() - begin);
    if (moved) maintain(previous, id, std::nullopt);
    maintain(next, id, point);
}

void PartitionIndex::erase(Id id) {
    const auto begin = profile_ ? Tick::now() : 0;
    const auto found = objects_.find(id);
    if (found == objects_.end()) {
        if (profile_) {
            // Nothing to attribute; keep the base clock running.
        }
        return;
    }
    const unsigned leaf = found->second.leaf;
    detach(leaves_[leaf], found->second);
    objects_.erase(found);
    if (profile_) times_[leaf].base_ns += Tick::ns(Tick::now() - begin);
    maintain(leaf, id, std::nullopt);
}

Answer PartitionIndex::query(Query q) const {
    Answer result;
    result.visits = 0;
    const unsigned lo = sub_of({q.box.x0, q.box.y0});
    const unsigned hi = sub_of({q.box.x1, q.box.y1});
    std::vector<unsigned> touched;
    for (unsigned y = lo / side; y <= hi / side; ++y) {
        for (unsigned x = lo % side; x <= hi % side; ++x) {
            const unsigned leaf = leaf_of_[y * side + x];
            if (std::find(touched.begin(), touched.end(), leaf) == touched.end())
                touched.push_back(leaf);
        }
    }
    for (unsigned leaf : touched) {
        const auto begin = profile_ ? Tick::now() : 0;
        ++*result.visits;
        const auto& bucket = leaves_[leaf];
        if (bucket.aux) {
            auto partial = bucket.aux->query(q);
            result.candidates += partial.candidates;
            *result.visits += partial.visits.value_or(0);
            if (result.ids.empty()) result.ids = std::move(partial.ids);
            else result.ids.insert(result.ids.end(), partial.ids.begin(), partial.ids.end());
        } else {
            for (Id id : bucket.ids) refine(result, q, id, objects_.at(id).point);
        }
        if (profile_) {
            times_[leaf].query_ns += Tick::ns(Tick::now() - begin);
            ++times_[leaf].queries;
        }
        if (observe_) {
            const Box b = bucket.box;
            const double w = std::max(0.0, std::min(b.x1, q.box.x1) - std::max(b.x0, q.box.x0));
            const double h = std::max(0.0, std::min(b.y1, q.box.y1) - std::max(b.y0, q.box.y0));
            const double area = (b.x1 - b.x0) * (b.y1 - b.y0);
            ++loads_[leaf].queries;
            if (area > 0) loads_[leaf].coverage += w * h / area;
        }
    }
    return result;
}

void PartitionIndex::reset_window() {
    std::fill(loads_.begin(), loads_.end(), WindowLoad{});
    std::fill(times_.begin(), times_.end(), LeafTimes{});
}
}  // namespace spatial
