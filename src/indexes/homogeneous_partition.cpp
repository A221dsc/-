#include "indexes/homogeneous_partition.hpp"
#include "core/clock.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spatial {
namespace {
unsigned clamp_cell(int v) { return static_cast<unsigned>(std::clamp(v, 0, 63)); }
}  // namespace

HomogeneousPartitionIndex::HomogeneousPartitionIndex(Box world, int quad_capacity, bool profile)
    : world_(world), capacity_(quad_capacity), profile_(profile) {
    if (world.x1 <= world.x0 || world.y1 <= world.y0)
        throw std::invalid_argument("invalid world box");
    cell_w_ = (world.x1 - world.x0) / side;
    cell_h_ = (world.y1 - world.y0) / side;
    if (cell_w_ <= 0.0 || cell_h_ <= 0.0) throw std::invalid_argument("empty world");
}

void HomogeneousPartitionIndex::set_partition(const std::vector<std::array<int, 4>>& rects,
                                              const std::vector<unsigned>& levels) {
    if (rects.empty() || rects.size() != levels.size())
        throw std::invalid_argument("partition rects/levels mismatch");
    rects_ = rects;
    quads_.clear();
    quads_.reserve(rects.size());
    for (unsigned leaf = 0; leaf < rects.size(); ++leaf) quads_.push_back(make_leaf(leaf));
    leaf_of_cell_.assign(side * side, -1);
    for (unsigned leaf = 0; leaf < rects.size(); ++leaf) {
        const auto& r = rects[leaf];
        for (int cy = r[1]; cy < r[3]; ++cy) {
            for (int cx = r[0]; cx < r[2]; ++cx) {
                if (cx < 0 || cy < 0 || cx >= side || cy >= side)
                    throw std::invalid_argument("partition rect outside the 64x64 grid");
                if (leaf_of_cell_[cy * side + cx] >= 0)
                    throw std::invalid_argument("overlapping partition rects");
                leaf_of_cell_[cy * side + cx] = static_cast<int>(leaf);
            }
        }
    }
    for (int cell = 0; cell < side * side; ++cell)
        if (leaf_of_cell_[cell] < 0) throw std::invalid_argument("partition leaves a hole");
    seen_.assign(rects.size(), 0);
    generation_ = 0;
}

std::unique_ptr<Index> HomogeneousPartitionIndex::make_leaf(unsigned leaf) const {
    const auto& r = rects_[leaf];
    Box box{world_.x0 + r[0] * cell_w_, world_.y0 + r[1] * cell_h_,
            world_.x0 + r[2] * cell_w_, world_.y0 + r[3] * cell_h_};
    return make_quad(capacity_, box);
}

unsigned HomogeneousPartitionIndex::cell_of(Point p) const {
    const int u = static_cast<int>(std::floor((p.x - world_.x0) / cell_w_));
    const int v = static_cast<int>(std::floor((p.y - world_.y0) / cell_h_));
    return clamp_cell(v) * side + clamp_cell(u);
}

unsigned HomogeneousPartitionIndex::leaf_of(Point p) const {
    return static_cast<unsigned>(leaf_of_cell_[cell_of(p)]);
}

std::size_t HomogeneousPartitionIndex::entity_count(unsigned leaf) const {
    const auto& points = quads_.at(leaf);
    // The Quad keeps the authoritative points in its own map; expose the count
    // through the Index point map where available.
    return points->size();
}

std::size_t HomogeneousPartitionIndex::index_nodes() const {
    std::size_t total = 0;
    for (const auto& leaf : quads_) total += leaf->nodes();
    return total;
}

void HomogeneousPartitionIndex::set(Id id, Point point) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
        throw std::invalid_argument("nonfinite point");
    const unsigned leaf = leaf_of(point);
    auto it = owner_.find(id);
    if (it == owner_.end()) {
        quads_[leaf]->set(id, point);
        owner_.emplace(id, leaf);
        return;
    }
    if (it->second == leaf) {
        quads_[leaf]->set(id, point);
        return;
    }
    const unsigned old = it->second;
    const auto previous = quads_[old]->point_of(id);
    quads_[old]->erase(id);
    try {
        quads_[leaf]->set(id, point);
    } catch (...) {
        // Restore the authoritative old point and ownership on failure.
        if (previous) quads_[old]->set(id, *previous);
        throw;
    }
    it->second = leaf;
}

void HomogeneousPartitionIndex::erase(Id id) {
    auto it = owner_.find(id);
    if (it == owner_.end()) return;
    quads_[it->second]->erase(id);
    owner_.erase(it);
}

Answer HomogeneousPartitionIndex::query(Query q) const {
    Answer result;
    result.visits = 0;
    const auto route_begin = Clock::now();
    ++generation_;
    const unsigned u0 = cell_of({q.box.x0, q.box.y0}) % side;
    const unsigned v0 = cell_of({q.box.x0, q.box.y0}) / side;
    const unsigned u1 = cell_of({q.box.x1, q.box.y1}) % side;
    const unsigned v1 = cell_of({q.box.x1, q.box.y1}) / side;
    scratch_.clear();
    for (unsigned v = v0; v <= v1; ++v) {
        for (unsigned u = u0; u <= u1; ++u) {
            ++route_cells_;
            const int leaf = leaf_of_cell_[v * side + u];
            if (seen_[leaf] == generation_) continue;
            seen_[leaf] = generation_;
            scratch_.push_back(static_cast<unsigned>(leaf));
            ++touched_leaves_;
            ++*result.visits;
        }
    }
    if (profile_) route_ms_ += (Clock::now() - route_begin).count() / 1e6;
    for (unsigned leaf : scratch_) {
        const auto local_begin = Clock::now();
        Answer partial = quads_[leaf]->query(q);
        result.candidates += partial.candidates;
        *result.visits += partial.visits.value_or(0);
        if (result.ids.empty()) result.ids = std::move(partial.ids);
        else result.ids.insert(result.ids.end(), partial.ids.begin(), partial.ids.end());
        ++local_queries_;
        if (profile_) local_ms_ += (Clock::now() - local_begin).count() / 1e6;
    }
    return result;
}

}  // namespace spatial
