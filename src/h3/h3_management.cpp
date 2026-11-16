#include "h3/h3_management.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace spatial::h3 {

namespace {

double point_distance(Point a, Point b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

BBox box_of(const Box& box) {
    BBox out;
    out.expand(box.x0, box.y0);
    out.expand(box.x1, box.y1);
    return out;
}

}  // namespace

double distance_point_box(const Point& point, const Box& box) {
    const double dx = std::max({box.x0 - point.x, 0.0, point.x - box.x1});
    const double dy = std::max({box.y0 - point.y, 0.0, point.y - box.y1});
    return std::hypot(dx, dy);
}

CellCover::CellCover(PartitionConfig config) : config_(std::move(config)) {
    if (config_.resolution < 0 || config_.resolution > 15) {
        throw std::invalid_argument("CellCover: invalid H3 resolution");
    }
    if (config_.map.x0 >= config_.map.x1 || config_.map.y0 >= config_.map.y1) {
        throw std::invalid_argument("CellCover: invalid map box");
    }

    // A polygon expanded by a margin larger than one H3 cell diameter, then
    // filtered by local cell bbox, gives a stable finite universe for the
    // theatre map. polygonToCells uses cell centroids; the margin prevents
    // dropping cells whose centroid is just outside the map.
    const Box poly_box{config_.map.x0 - config_.universe_margin_m,
                       config_.map.y0 - config_.universe_margin_m,
                       config_.map.x1 + config_.universe_margin_m,
                       config_.map.y1 + config_.universe_margin_m};
    std::vector<GeoPoint> corners{
        config_.projection.toGeo({poly_box.x0, poly_box.y0}),
        config_.projection.toGeo({poly_box.x1, poly_box.y0}),
        config_.projection.toGeo({poly_box.x1, poly_box.y1}),
        config_.projection.toGeo({poly_box.x0, poly_box.y1}),
        config_.projection.toGeo({poly_box.x0, poly_box.y0})};
    std::vector<LatLng> verts;
    verts.reserve(corners.size());
    for (const GeoPoint& point : corners) {
        verts.push_back({deg2rad(point.lat_deg), deg2rad(point.lng_deg)});
    }
    GeoLoop loop{static_cast<int>(verts.size()), verts.data()};
    GeoPolygon polygon{loop, 0, nullptr};

    std::int64_t max_cells = 0;
    h3_check(H3_EXPORT(maxPolygonToCellsSize)(
                 &polygon, config_.resolution, 0, &max_cells),
             "maxPolygonToCellsSize");
    if (max_cells <= 0) throw std::runtime_error("CellCover: empty polygon size");
    std::vector<H3Index> raw(static_cast<std::size_t>(max_cells), H3_NULL);
    h3_check(H3_EXPORT(polygonToCells)(&polygon, config_.resolution, 0,
                                       raw.data()),
             "polygonToCells");

    const BBox map_box = box_of(config_.map);
    std::vector<CellInfo> candidates;
    candidates.reserve(raw.size());
    for (const H3Index h3 : raw) {
        if (h3 == H3_NULL) continue;
        LatLng center_latlng{};
        h3_check(H3_EXPORT(cellToLatLng)(h3, &center_latlng), "cellToLatLng");
        const Point center =
            config_.projection.toLocal({rad2deg(center_latlng.lat),
                                        rad2deg(center_latlng.lng)});
        CellBoundary boundary{};
        h3_check(H3_EXPORT(cellToBoundary)(h3, &boundary), "cellToBoundary");
        BBox bbox;
        double radius = 0.0;
        for (int i = 0; i < boundary.numVerts; ++i) {
            const Point vertex =
                config_.projection.toLocal({rad2deg(boundary.verts[i].lat),
                                            rad2deg(boundary.verts[i].lng)});
            bbox.expand(vertex.x, vertex.y);
            radius = std::max(radius, point_distance(center, vertex));
        }
        if (!bbox.overlaps(map_box)) continue;
        CellInfo info;
        info.h3 = h3;
        info.center = center;
        info.bbox = bbox;
        info.radius = radius + config_.cell_margin_m;
        candidates.push_back(info);
    }
    if (candidates.empty()) {
        throw std::runtime_error("CellCover: no H3 cells cover the map");
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const CellInfo& a, const CellInfo& b) { return a.h3 < b.h3; });
    cells_ = std::move(candidates);
    h3_to_id_.reserve(cells_.size() * 2);
    for (std::size_t i = 0; i < cells_.size(); ++i) {
        cells_[i].id = static_cast<CellId>(i);
        h3_to_id_.emplace(cells_[i].h3, cells_[i].id);
    }

    neighbors_.resize(cells_.size());
    std::int64_t disk_size = 0;
    h3_check(H3_EXPORT(maxGridDiskSize)(1, &disk_size), "maxGridDiskSize");
    std::vector<H3Index> disk(static_cast<std::size_t>(disk_size), H3_NULL);
    for (const CellInfo& info : cells_) {
        h3_check(H3_EXPORT(gridDisk)(info.h3, 1, disk.data()), "gridDisk");
        for (const H3Index neighbor : disk) {
            if (neighbor == H3_NULL || neighbor == info.h3) continue;
            const auto it = h3_to_id_.find(neighbor);
            if (it != h3_to_id_.end()) {
                neighbors_[static_cast<std::size_t>(info.id)].push_back(it->second);
            }
        }
        std::sort(neighbors_[static_cast<std::size_t>(info.id)].begin(),
                  neighbors_[static_cast<std::size_t>(info.id)].end());
    }

    max_radius_ = 0.0;
    for (const CellInfo& info : cells_) max_radius_ = std::max(max_radius_, info.radius);
    grid_min_x_ = config_.map.x0 - max_radius_ - config_.cell_margin_m;
    grid_min_y_ = config_.map.y0 - max_radius_ - config_.cell_margin_m;
    const double grid_max_x = config_.map.x1 + max_radius_ + config_.cell_margin_m;
    const double grid_max_y = config_.map.y1 + max_radius_ + config_.cell_margin_m;
    grid_cell_ = std::max(2.0 * max_radius_, 64.0);
    grid_w_ = std::max(1, static_cast<int>(std::ceil((grid_max_x - grid_min_x_) / grid_cell_)));
    grid_h_ = std::max(1, static_cast<int>(std::ceil((grid_max_y - grid_min_y_) / grid_cell_)));
    grid_.assign(static_cast<std::size_t>(grid_w_ * grid_h_), {});
    for (CellInfo& info : cells_) {
        info.grid_x = std::clamp(
            static_cast<int>(std::floor((info.center.x - grid_min_x_) / grid_cell_)),
            0, grid_w_ - 1);
        info.grid_y = std::clamp(
            static_cast<int>(std::floor((info.center.y - grid_min_y_) / grid_cell_)),
            0, grid_h_ - 1);
        grid_[static_cast<std::size_t>(info.grid_y * grid_w_ + info.grid_x)]
            .push_back(info.id);
    }
}

int CellCover::id_of(H3Index h3) const {
    const auto it = h3_to_id_.find(h3);
    return it == h3_to_id_.end() ? -1 : it->second;
}

std::vector<CellId> CellCover::candidate_cells(const Query& query) const {
    std::vector<CellId> out;
    const double expand = max_radius_ + config_.cell_margin_m;
    const int gx0 = std::clamp(
        static_cast<int>(std::floor((query.box.x0 - expand - grid_min_x_) / grid_cell_)),
        0, grid_w_ - 1);
    const int gx1 = std::clamp(
        static_cast<int>(std::floor((query.box.x1 + expand - grid_min_x_) / grid_cell_)),
        0, grid_w_ - 1);
    const int gy0 = std::clamp(
        static_cast<int>(std::floor((query.box.y0 - expand - grid_min_y_) / grid_cell_)),
        0, grid_h_ - 1);
    const int gy1 = std::clamp(
        static_cast<int>(std::floor((query.box.y1 + expand - grid_min_y_) / grid_cell_)),
        0, grid_h_ - 1);
    for (int gy = gy0; gy <= gy1; ++gy) {
        for (int gx = gx0; gx <= gx1; ++gx) {
            for (const CellId id : grid_[static_cast<std::size_t>(gy * grid_w_ + gx)]) {
                const CellInfo& info = cells_[static_cast<std::size_t>(id)];
                if (query.circular) {
                    if (point_distance(info.center, query.center) <=
                        query.radius + info.radius) {
                        out.push_back(id);
                    }
                } else if (distance_point_box(info.center, query.box) <= info.radius) {
                    out.push_back(id);
                }
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

void H3EntityIndex::add(H3Index cell, Id id) {
    buckets_[cell].push_back(id);
    ++entity_count_;
}

void H3EntityIndex::remove(H3Index cell, Id id) {
    const auto it = buckets_.find(cell);
    if (it == buckets_.end()) throw std::logic_error("H3EntityIndex: missing bucket");
    auto& ids = it->second;
    const auto pos = std::find(ids.begin(), ids.end(), id);
    if (pos == ids.end()) throw std::logic_error("H3EntityIndex: missing id");
    *pos = ids.back();
    ids.pop_back();
    if (ids.empty()) buckets_.erase(it);
    --entity_count_;
}

const std::vector<Id>* H3EntityIndex::bucket(H3Index cell) const {
    const auto it = buckets_.find(cell);
    return it == buckets_.end() ? nullptr : &it->second;
}

std::size_t H3EntityIndex::estimated_bytes() const {
    std::size_t bytes = buckets_.size() * (sizeof(H3Index) + sizeof(std::vector<Id>) + 16);
    for (const auto& [cell, ids] : buckets_) {
        (void)cell;
        bytes += ids.capacity() * sizeof(Id);
    }
    return bytes;
}

GlobalH3Index::GlobalH3Index(std::shared_ptr<CellCover> cover)
    : cover_(std::move(cover)) {
    if (!cover_) throw std::invalid_argument("GlobalH3Index: null cover");
}

void GlobalH3Index::set(Id id, Point point) {
    const H3Index new_cell = [&] {
        const GeoPoint geo = cover_->config().projection.toGeo(point);
        LatLng ll{deg2rad(geo.lat_deg), deg2rad(geo.lng_deg)};
        H3Index cell = H3_NULL;
        h3_check(H3_EXPORT(latLngToCell)(&ll, cover_->config().resolution, &cell),
                 "latLngToCell");
        return cell;
    }();
    const auto old = points_.find(id);
    if (old == points_.end()) {
        points_.emplace(id, point);
        cell_of_.emplace(id, new_cell);
        index_.add(new_cell, id);
        return;
    }
    if (old->second == point) return;
    const H3Index old_cell = cell_of_.at(id);
    if (old_cell != new_cell) {
        index_.remove(old_cell, id);
        index_.add(new_cell, id);
        cell_of_[id] = new_cell;
    }
    points_[id] = point;
}

void GlobalH3Index::erase(Id id) {
    const auto it = points_.find(id);
    if (it == points_.end()) return;
    index_.remove(cell_of_.at(id), id);
    cell_of_.erase(id);
    points_.erase(it);
}

QueryResult GlobalH3Index::query(Query query) const {
    QueryResult result;
    const auto candidates = cover_->candidate_cells(query);
    result.stats.candidate_cells = candidates.size();
    result.stats.leaves_visited = 1;
    for (const CellId cell_id : candidates) {
        ++result.stats.bucket_lookups;
        const auto* ids = index_.bucket(cover_->cell(cell_id).h3);
        if (!ids) continue;
        for (const Id id : *ids) {
            ++result.stats.candidates;
            if (query.matches(points_.at(id))) result.ids.push_back(id);
        }
    }
    std::sort(result.ids.begin(), result.ids.end());
    return result;
}

std::optional<Point> GlobalH3Index::point_of(Id id) const {
    const auto it = points_.find(id);
    if (it == points_.end()) return std::nullopt;
    return it->second;
}

std::size_t GlobalH3Index::estimated_bytes() const {
    return points_.size() * (sizeof(Id) + sizeof(Point) + 24) +
           cell_of_.size() * (sizeof(Id) + sizeof(H3Index) + 24) +
           index_.estimated_bytes();
}

PartitionedBase::PartitionedBase(std::shared_ptr<CellCover> cover)
    : cover_(std::move(cover)) {
    if (!cover_) throw std::invalid_argument("PartitionedBase: null cover");
    cell_to_leaf_.assign(cover_->size(), -1);
}

void PartitionedBase::register_leaf(std::unique_ptr<PartitionLeaf> leaf) {
    if (!leaf || leaf->id < 0) throw std::invalid_argument("register_leaf: bad leaf");
    const int id = leaf->id;
    for (const CellId cell : leaf->cells) {
        if (cell < 0 || static_cast<std::size_t>(cell) >= cell_to_leaf_.size()) {
            throw std::logic_error("register_leaf: cell out of range");
        }
        if (cell_to_leaf_[static_cast<std::size_t>(cell)] != -1) {
            throw std::logic_error("register_leaf: cell already assigned");
        }
        cell_to_leaf_[static_cast<std::size_t>(cell)] = id;
    }
    leaves_[id] = std::move(leaf);
    ++active_leaves_;
}

PartitionLeaf* PartitionedBase::find_leaf(int leaf_id) const {
    const auto it = leaves_.find(leaf_id);
    return it == leaves_.end() ? nullptr : it->second.get();
}

int PartitionedBase::leaf_id_of(H3Index cell) const {
    const int cell_id = cover_->id_of(cell);
    if (cell_id < 0) throw std::logic_error("cell outside the theatre universe");
    const int leaf = cell_to_leaf_.at(static_cast<std::size_t>(cell_id));
    if (leaf < 0) throw std::logic_error("cell has no management leaf");
    return leaf;
}

PartitionLeaf& PartitionedBase::leaf_of(H3Index cell) const {
    PartitionLeaf* leaf = find_leaf(leaf_id_of(cell));
    if (!leaf) throw std::logic_error("leaf id has no leaf");
    return *leaf;
}

void PartitionedBase::set(Id id, Point point) {
    const GeoPoint geo = cover_->config().projection.toGeo(point);
    LatLng ll{deg2rad(geo.lat_deg), deg2rad(geo.lng_deg)};
    H3Index new_cell = H3_NULL;
    h3_check(H3_EXPORT(latLngToCell)(&ll, cover_->config().resolution, &new_cell),
             "latLngToCell");
    const auto old = points_.find(id);
    if (old == points_.end()) {
        const int leaf_id = leaf_id_of(new_cell);
        PartitionLeaf* leaf = find_leaf(leaf_id);
        leaf->index.add(new_cell, id);
        ++leaf->entity_count;
        points_.emplace(id, point);
        cell_of_.emplace(id, new_cell);
        on_insert(leaf_id, new_cell);
        return;
    }
    if (old->second == point) return;
    const H3Index old_cell = cell_of_.at(id);
    if (old_cell == new_cell) {
        points_[id] = point;
        return;
    }
    const int old_leaf_id = leaf_id_of(old_cell);
    if (old_leaf_id == leaf_id_of(new_cell)) {
        PartitionLeaf* leaf = find_leaf(old_leaf_id);
        leaf->index.remove(old_cell, id);
        leaf->index.add(new_cell, id);
        points_[id] = point;
        cell_of_[id] = new_cell;
        return;
    }

    // Remove first and let the old leaf merge before the entity is inserted
    // into the new leaf. If the old leaf's parent absorbs the new cell, the
    // new leaf id is re-read after the hook; this keeps capacity-tree counts
    // exact even when the two leaves are siblings.
    PartitionLeaf* old_leaf = find_leaf(old_leaf_id);
    old_leaf->index.remove(old_cell, id);
    --old_leaf->entity_count;
    on_erase(old_leaf_id, old_cell);

    const int new_leaf_id = leaf_id_of(new_cell);
    PartitionLeaf* new_leaf = find_leaf(new_leaf_id);
    new_leaf->index.add(new_cell, id);
    ++new_leaf->entity_count;
    points_[id] = point;
    cell_of_[id] = new_cell;
    on_insert(new_leaf_id, new_cell);
}

void PartitionedBase::erase(Id id) {
    const auto it = points_.find(id);
    if (it == points_.end()) return;
    const H3Index cell = cell_of_.at(id);
    const int leaf_id = leaf_id_of(cell);
    PartitionLeaf* leaf = find_leaf(leaf_id);
    leaf->index.remove(cell, id);
    --leaf->entity_count;
    points_.erase(it);
    cell_of_.erase(id);
    on_erase(leaf_id, cell);
}

QueryResult PartitionedBase::query(Query query) const {
    QueryResult result;
    const auto candidates = cover_->candidate_cells(query);
    result.stats.candidate_cells = candidates.size();
    // A small per-query visit list keeps the const query thread-safe and avoids
    // shared mutable state. Management fan-out is usually much smaller than the
    // number of candidate cells.
    std::vector<int> visited_leaves;
    visited_leaves.reserve(8);
    for (const CellId cell_id : candidates) {
        const int leaf_id = cell_to_leaf_.at(static_cast<std::size_t>(cell_id));
        if (leaf_id < 0) continue;
        if (std::find(visited_leaves.begin(), visited_leaves.end(), leaf_id) ==
            visited_leaves.end()) {
            visited_leaves.push_back(leaf_id);
            ++result.stats.leaves_visited;
        }
        const auto it = leaves_.find(leaf_id);
        if (it == leaves_.end() || !it->second) {
            throw std::logic_error("candidate cell has no active leaf");
        }
        ++result.stats.bucket_lookups;
        const auto* ids = it->second->index.bucket(cover_->cell(cell_id).h3);
        if (!ids) continue;
        for (const Id id : *ids) {
            ++result.stats.candidates;
            if (query.matches(points_.at(id))) result.ids.push_back(id);
        }
    }
    std::sort(result.ids.begin(), result.ids.end());
    return result;
}

std::size_t PartitionedBase::bucket_count() const {
    std::size_t count = 0;
    for (const auto& [id, leaf] : leaves_) {
        (void)id;
        count += leaf->index.bucket_count();
    }
    return count;
}

std::optional<Point> PartitionedBase::point_of(Id id) const {
    const auto it = points_.find(id);
    if (it == points_.end()) return std::nullopt;
    return it->second;
}

std::size_t PartitionedBase::estimated_bytes() const {
    std::size_t bytes = points_.size() * (sizeof(Id) + sizeof(Point) + 24) +
                        cell_of_.size() * (sizeof(Id) + sizeof(H3Index) + 24) +
                        cell_to_leaf_.size() * sizeof(int) +
                        leaves_.size() * (sizeof(PartitionLeaf) + 64);
    for (const auto& [id, leaf] : leaves_) {
        (void)id;
        bytes += leaf->cells.capacity() * sizeof(CellId) + 64;
        bytes += leaf->index.estimated_bytes();
    }
    return bytes;
}

FixedPartitionIndex::FixedPartitionIndex(std::shared_ptr<CellCover> cover,
                                         std::vector<int> assignment,
                                         int partition_count)
    : PartitionedBase(std::move(cover)) {
    if (partition_count <= 0) {
        throw std::invalid_argument("FixedPartitionIndex: bad partition count");
    }
    if (assignment.size() != cover_->size()) {
        throw std::invalid_argument("FixedPartitionIndex: assignment size");
    }
    std::vector<std::unique_ptr<PartitionLeaf>> leaves;
    leaves.reserve(static_cast<std::size_t>(partition_count));
    for (int i = 0; i < partition_count; ++i) {
        auto leaf = std::make_unique<PartitionLeaf>();
        leaf->id = i;
        leaves.push_back(std::move(leaf));
    }
    for (std::size_t i = 0; i < assignment.size(); ++i) {
        const int p = assignment[i];
        if (p < 0 || p >= partition_count) {
            throw std::invalid_argument("FixedPartitionIndex: bad assignment value");
        }
        leaves[static_cast<std::size_t>(p)]->cells.push_back(static_cast<CellId>(i));
    }
    for (auto& leaf : leaves) {
        if (leaf->cells.empty()) {
            throw std::invalid_argument("FixedPartitionIndex: empty partition");
        }
        register_leaf(std::move(leaf));
    }
}

WorkerShardedIndex::WorkerShardedIndex(std::shared_ptr<CellCover> cover,
                                               std::vector<int> region_assignment,
                                               std::vector<int> region_to_worker)
    : cover_(std::move(cover)),
      region_to_worker_(std::move(region_to_worker)) {
    if (!cover_) throw std::invalid_argument("WorkerShardedIndex: null cover");
    if (region_assignment.size() != cover_->size()) {
        throw std::invalid_argument("WorkerShardedIndex: assignment size");
    }
    if (region_to_worker_.empty()) {
        throw std::invalid_argument("WorkerShardedIndex: empty mapping");
    }
    const int regions =
        *std::max_element(region_assignment.begin(), region_assignment.end()) + 1;
    if (regions > static_cast<int>(region_to_worker_.size())) {
        throw std::invalid_argument("WorkerShardedIndex: mapping too short");
    }
    const int workers =
        *std::max_element(region_to_worker_.begin(), region_to_worker_.end()) + 1;
    if (workers <= 0) throw std::invalid_argument("WorkerShardedIndex: bad worker count");
    worker_indexes_.resize(static_cast<std::size_t>(workers));
    cell_to_worker_.assign(cover_->size(), -1);
    for (std::size_t cell = 0; cell < region_assignment.size(); ++cell) {
        const int region = region_assignment[cell];
        if (region < 0 || region >= static_cast<int>(region_to_worker_.size())) {
            throw std::invalid_argument("WorkerShardedIndex: bad region");
        }
        const int worker = region_to_worker_[static_cast<std::size_t>(region)];
        if (worker < 0 || worker >= workers) {
            throw std::invalid_argument("WorkerShardedIndex: bad worker");
        }
        cell_to_worker_[cell] = worker;
    }
}

void WorkerShardedIndex::set(Id id, Point point) {
    const GeoPoint geo = cover_->config().projection.toGeo(point);
    LatLng ll{deg2rad(geo.lat_deg), deg2rad(geo.lng_deg)};
    H3Index cell = H3_NULL;
    h3_check(H3_EXPORT(latLngToCell)(&ll, cover_->config().resolution, &cell),
             "latLngToCell");
    const int cell_id = cover_->id_of(cell);
    if (cell_id < 0) throw std::logic_error("WorkerShardedIndex: cell outside universe");
    const int worker = cell_to_worker_.at(static_cast<std::size_t>(cell_id));
    const auto old = points_.find(id);
    if (old == points_.end()) {
        worker_indexes_[static_cast<std::size_t>(worker)].add(cell, id);
        points_.emplace(id, point);
        cell_of_.emplace(id, cell);
        return;
    }
    if (old->second == point) return;
    const H3Index old_cell = cell_of_.at(id);
    const int old_cell_id = cover_->id_of(old_cell);
    const int old_worker = cell_to_worker_.at(static_cast<std::size_t>(old_cell_id));
    if (old_worker != worker) {
        worker_indexes_[static_cast<std::size_t>(old_worker)].remove(old_cell, id);
        worker_indexes_[static_cast<std::size_t>(worker)].add(cell, id);
    } else if (old_cell != cell) {
        worker_indexes_[static_cast<std::size_t>(worker)].remove(old_cell, id);
        worker_indexes_[static_cast<std::size_t>(worker)].add(cell, id);
    }
    points_[id] = point;
    cell_of_[id] = cell;
}

void WorkerShardedIndex::erase(Id id) {
    const auto it = points_.find(id);
    if (it == points_.end()) return;
    const H3Index cell = cell_of_.at(id);
    const int cell_id = cover_->id_of(cell);
    const int worker = cell_to_worker_.at(static_cast<std::size_t>(cell_id));
    worker_indexes_[static_cast<std::size_t>(worker)].remove(cell, id);
    points_.erase(it);
    cell_of_.erase(id);
}

QueryResult WorkerShardedIndex::query(Query query) const {
    QueryResult result;
    const auto candidates = cover_->candidate_cells(query);
    result.stats.candidate_cells = candidates.size();
    std::vector<std::vector<CellId>> by_worker(worker_indexes_.size());
    for (const CellId cell : candidates) {
        const int worker = cell_to_worker_.at(static_cast<std::size_t>(cell));
        if (worker >= 0) by_worker[static_cast<std::size_t>(worker)].push_back(cell);
    }
    for (std::size_t worker = 0; worker < by_worker.size(); ++worker) {
        if (by_worker[worker].empty()) continue;
        ++result.stats.leaves_visited;
        for (const CellId cell : by_worker[worker]) {
            ++result.stats.bucket_lookups;
            const auto* ids =
                worker_indexes_[worker].bucket(cover_->cell(cell).h3);
            if (!ids) continue;
            for (const Id id : *ids) {
                ++result.stats.candidates;
                if (query.matches(points_.at(id))) result.ids.push_back(id);
            }
        }
    }
    std::sort(result.ids.begin(), result.ids.end());
    return result;
}

std::size_t WorkerShardedIndex::worker_entity_count(int worker) const {
    if (worker < 0 || worker >= static_cast<int>(worker_indexes_.size())) return 0;
    return worker_indexes_[static_cast<std::size_t>(worker)].entity_count();
}

std::size_t WorkerShardedIndex::total_bucket_count() const {
    std::size_t count = 0;
    for (const auto& index : worker_indexes_) count += index.bucket_count();
    return count;
}

bool WorkerShardedIndex::validate(std::string* why) const {
    try {
        if (cell_to_worker_.size() != cover_->size()) {
            throw std::runtime_error("cell_to_worker size mismatch");
        }
        std::size_t total = 0;
        for (const auto& index : worker_indexes_) total += index.entity_count();
        if (total != points_.size()) {
            throw std::runtime_error("worker entity totals disagree with points");
        }
        std::unordered_map<Id, std::size_t> seen;
        for (std::size_t cell = 0; cell < cover_->size(); ++cell) {
            const int worker = cell_to_worker_[cell];
            if (worker < 0) throw std::runtime_error("cell without worker");
            const auto* ids = worker_indexes_[static_cast<std::size_t>(worker)].bucket(
                cover_->cell(static_cast<CellId>(cell)).h3);
            if (!ids) continue;
            for (const Id id : *ids) {
                if (++seen[id] != 1) throw std::runtime_error("duplicate entity id");
                const auto pit = points_.find(id);
                if (pit == points_.end()) throw std::runtime_error("entity without point");
                const GeoPoint geo = cover_->config().projection.toGeo(pit->second);
                LatLng ll{deg2rad(geo.lat_deg), deg2rad(geo.lng_deg)};
                H3Index actual_h3 = H3_NULL;
                h3_check(H3_EXPORT(latLngToCell)(
                             &ll, cover_->config().resolution, &actual_h3),
                         "latLngToCell");
                const int actual_cell = cover_->id_of(actual_h3);
                if (actual_cell != static_cast<int>(cell)) {
                    throw std::runtime_error("entity is in the wrong cell bucket");
                }
            }
        }
        if (seen.size() != points_.size()) {
            throw std::runtime_error("not all points are indexed");
        }
        return true;
    } catch (const std::exception& error) {
        if (why) *why = error.what();
        return false;
    }
}

CapacityPartitionIndex::CapacityPartitionIndex(std::shared_ptr<CellCover> cover,
                                               std::size_t split_threshold,
                                               std::size_t merge_threshold)
    : PartitionedBase(std::move(cover)),
      split_threshold_(split_threshold),
      merge_threshold_(merge_threshold) {
    if (split_threshold_ == 0 || merge_threshold_ >= split_threshold_) {
        throw std::invalid_argument("CapacityPartitionIndex: invalid thresholds");
    }
    root_ = std::make_unique<Node>();
    root_->id = next_node_id_++;
    root_->leaf = true;
    root_->leaf_id = next_leaf_id_++;
    root_->cells.resize(cover_->size());
    std::iota(root_->cells.begin(), root_->cells.end(), 0);
    auto leaf = std::make_unique<PartitionLeaf>();
    leaf->id = root_->leaf_id;
    leaf->cells = root_->cells;
    node_of_leaf_[root_->leaf_id] = root_.get();
    register_leaf(std::move(leaf));
    node_count_ = 1;
}

CapacityPartitionIndex::Node* CapacityPartitionIndex::node_for_leaf(
    int leaf_id) const {
    const auto it = node_of_leaf_.find(leaf_id);
    return it == node_of_leaf_.end() ? nullptr : it->second;
}

CapacityPartitionIndex::Node* CapacityPartitionIndex::make_node(Node* parent) {
    auto node = std::make_unique<Node>();
    node->id = next_node_id_++;
    node->parent = parent;
    ++node_count_;
    if (parent) {
        if (!parent->left) {
            parent->left = std::move(node);
            return parent->left.get();
        }
        if (!parent->right) {
            parent->right = std::move(node);
            return parent->right.get();
        }
        throw std::logic_error("make_node: parent already has two children");
    }
    if (root_) throw std::logic_error("make_node: second root");
    root_ = std::move(node);
    return root_.get();
}

void CapacityPartitionIndex::on_insert(int leaf_id, H3Index cell) {
    (void)cell;
    Node* node = node_for_leaf(leaf_id);
    if (!node) return;
    const auto it = leaves_.find(leaf_id);
    if (it == leaves_.end()) return;
    node->entity_count = it->second->entity_count;
    for (Node* parent = node->parent; parent; parent = parent->parent) {
        ++parent->entity_count;
    }
    maybe_split(node);
}

void CapacityPartitionIndex::on_erase(int leaf_id, H3Index cell) {
    (void)cell;
    Node* node = node_for_leaf(leaf_id);
    if (!node) return;
    const auto it = leaves_.find(leaf_id);
    if (it == leaves_.end()) return;
    node->entity_count = it->second->entity_count;
    for (Node* parent = node->parent; parent; parent = parent->parent) {
        if (parent->entity_count == 0) {
            throw std::logic_error("on_erase: entity count underflow");
        }
        --parent->entity_count;
    }
    if (node->entity_count <= merge_threshold_) {
        try_merge_up(node);
    }
}

void CapacityPartitionIndex::maybe_split(Node* node) {
    if (!node || !node->leaf) return;
    while (node->leaf && node->entity_count > split_threshold_ &&
           node->cells.size() > 1) {
        if (!split_node(node)) break;
    }
    if (node->left) maybe_split(node->left.get());
    if (node->right) maybe_split(node->right.get());
}

bool CapacityPartitionIndex::split_node(Node* node) {
    if (!node || !node->leaf || node->cells.size() < 2) return false;
    const auto leaf_it = leaves_.find(node->leaf_id);
    if (leaf_it == leaves_.end()) throw std::logic_error("split_node: no leaf");
    PartitionLeaf* old_leaf = leaf_it->second.get();
    std::vector<double> weights(cover_->size(), 0.0);
    for (const CellId cell : node->cells) {
        const H3Index h3 = cover_->cell(cell).h3;
        const auto* ids = old_leaf->index.bucket(h3);
        weights[static_cast<std::size_t>(cell)] = ids ? static_cast<double>(ids->size()) : 0.0;
    }
    std::vector<CellId> left_cells;
    std::vector<CellId> right_cells;
    if (!split_connected_cells(*cover_, node->cells, weights, left_cells, right_cells)) {
        return false;
    }

    Node* left_node = make_node(node);
    Node* right_node = make_node(node);
    node->leaf = false;
    node->cells.clear();
    node->leaf_id = -1;
    node_of_leaf_.erase(old_leaf->id);

    auto left_leaf = std::make_unique<PartitionLeaf>();
    left_leaf->id = next_leaf_id_++;
    left_leaf->cells = left_cells;
    auto right_leaf = std::make_unique<PartitionLeaf>();
    right_leaf->id = next_leaf_id_++;
    right_leaf->cells = right_cells;

    for (const CellId cell : left_cells) {
        const H3Index h3 = cover_->cell(cell).h3;
        const auto* ids = old_leaf->index.bucket(h3);
        if (ids) {
            for (const Id id : *ids) left_leaf->index.add(h3, id);
        }
    }
    for (const CellId cell : right_cells) {
        const H3Index h3 = cover_->cell(cell).h3;
        const auto* ids = old_leaf->index.bucket(h3);
        if (ids) {
            for (const Id id : *ids) right_leaf->index.add(h3, id);
        }
    }
    left_leaf->entity_count = left_leaf->index.entity_count();
    right_leaf->entity_count = right_leaf->index.entity_count();
    left_node->leaf = true;
    right_node->leaf = true;
    left_node->leaf_id = left_leaf->id;
    right_node->leaf_id = right_leaf->id;
    left_node->cells = left_cells;
    right_node->cells = right_cells;
    left_node->entity_count = left_leaf->entity_count;
    right_node->entity_count = right_leaf->entity_count;
    node_of_leaf_[left_leaf->id] = left_node;
    node_of_leaf_[right_leaf->id] = right_node;

    for (const CellId cell : old_leaf->cells) {
        cell_to_leaf_[static_cast<std::size_t>(cell)] = -1;
    }
    leaves_.erase(old_leaf->id);
    --active_leaves_;
    register_leaf(std::move(left_leaf));
    register_leaf(std::move(right_leaf));
    ++split_count_;
    return true;
}

void CapacityPartitionIndex::try_merge_up(Node* node) {
    Node* current = node;
    while (current && current->parent) {
        Node* parent = current->parent;
        if (parent->left && parent->right && parent->left->leaf &&
            parent->right->leaf && parent->entity_count <= merge_threshold_) {
            if (!merge_node(parent)) break;
            current = parent;
        } else {
            break;
        }
    }
}

bool CapacityPartitionIndex::merge_node(Node* node) {
    if (!node || node->leaf || !node->left || !node->right) return false;
    Node* left_node = node->left.get();
    Node* right_node = node->right.get();
    if (!left_node->leaf || !right_node->leaf) return false;
    if (node->entity_count > merge_threshold_) return false;
    const auto left_it = leaves_.find(left_node->leaf_id);
    const auto right_it = leaves_.find(right_node->leaf_id);
    if (left_it == leaves_.end() || right_it == leaves_.end()) return false;
    PartitionLeaf* left_leaf = left_it->second.get();
    PartitionLeaf* right_leaf = right_it->second.get();

    auto merged = std::make_unique<PartitionLeaf>();
    merged->id = next_leaf_id_++;
    merged->cells = left_node->cells;
    merged->cells.insert(merged->cells.end(), right_node->cells.begin(),
                         right_node->cells.end());
    for (const CellId cell : merged->cells) {
        const H3Index h3 = cover_->cell(cell).h3;
        const auto* left_ids = left_leaf->index.bucket(h3);
        if (left_ids) {
            for (const Id id : *left_ids) merged->index.add(h3, id);
        }
        const auto* right_ids = right_leaf->index.bucket(h3);
        if (right_ids) {
            for (const Id id : *right_ids) merged->index.add(h3, id);
        }
    }
    merged->entity_count = merged->index.entity_count();
    node_of_leaf_.erase(left_leaf->id);
    node_of_leaf_.erase(right_leaf->id);
    for (const CellId cell : left_leaf->cells) {
        cell_to_leaf_[static_cast<std::size_t>(cell)] = -1;
    }
    for (const CellId cell : right_leaf->cells) {
        cell_to_leaf_[static_cast<std::size_t>(cell)] = -1;
    }
    leaves_.erase(left_leaf->id);
    leaves_.erase(right_leaf->id);
    active_leaves_ -= 2;
    node->left.reset();
    node->right.reset();
    node->leaf = true;
    node->leaf_id = merged->id;
    node->cells = merged->cells;
    node->entity_count = merged->entity_count;
    node_of_leaf_[merged->id] = node;
    register_leaf(std::move(merged));
    node_count_ -= 2;
    ++merge_count_;
    return true;
}

void CapacityPartitionIndex::recompute_leaf_cells(Node* node) {
    if (!node) return;
    if (node->leaf) {
        const auto it = leaves_.find(node->leaf_id);
        if (it != leaves_.end()) node->cells = it->second->cells;
        return;
    }
    node->cells.clear();
    recompute_leaf_cells(node->left.get());
    recompute_leaf_cells(node->right.get());
}

bool split_connected_cells(const CellCover& cover,
                           const std::vector<CellId>& cells,
                           const std::vector<double>& cell_weights,
                           std::vector<CellId>& left,
                           std::vector<CellId>& right) {
    left.clear();
    right.clear();
    if (cells.size() < 2) return false;
    std::unordered_map<CellId, int> pos;
    pos.reserve(cells.size() * 2);
    for (std::size_t i = 0; i < cells.size(); ++i) pos.emplace(cells[i], static_cast<int>(i));
    if (pos.size() != cells.size()) throw std::logic_error("split: duplicate cells");

    const int n = static_cast<int>(cells.size());
    int root = 0;
    double best_root_weight = -1.0;
    for (int i = 0; i < n; ++i) {
        const double w = cell_weights.empty()
                             ? 1.0
                             : cell_weights[static_cast<std::size_t>(cells[static_cast<std::size_t>(i)])];
        if (w > best_root_weight) {
            best_root_weight = w;
            root = i;
        }
    }
    std::vector<int> parent(static_cast<std::size_t>(n), -1);
    std::vector<std::vector<int>> children(static_cast<std::size_t>(n));
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(n));
    std::queue<int> queue;
    parent[static_cast<std::size_t>(root)] = root;
    queue.push(root);
    while (!queue.empty()) {
        const int u = queue.front();
        queue.pop();
        order.push_back(u);
        for (const CellId neighbor : cover.neighbors(cells[static_cast<std::size_t>(u)])) {
            const auto it = pos.find(neighbor);
            if (it == pos.end()) continue;
            const int v = it->second;
            if (parent[static_cast<std::size_t>(v)] != -1) continue;
            parent[static_cast<std::size_t>(v)] = u;
            children[static_cast<std::size_t>(u)].push_back(v);
            queue.push(v);
        }
    }
    if (static_cast<int>(order.size()) != n) return false;  // disconnected input

    std::vector<double> subtree(static_cast<std::size_t>(n), 0.0);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int u = *it;
        double weight = cell_weights.empty()
                            ? 1.0
                            : cell_weights[static_cast<std::size_t>(cells[static_cast<std::size_t>(u)])];
        for (const int child : children[static_cast<std::size_t>(u)]) {
            weight += subtree[static_cast<std::size_t>(child)];
        }
        subtree[static_cast<std::size_t>(u)] = weight;
    }
    const double total = subtree[static_cast<std::size_t>(root)];
    int cut_child = -1;
    double best_score = std::numeric_limits<double>::infinity();
    for (const int v : order) {
        if (v == root) continue;
        const double a = subtree[static_cast<std::size_t>(v)];
        const double b = total - a;
        if (a <= 1e-12 || b <= 1e-12) continue;
        const double score = std::abs(a - b);
        if (score < best_score) {
            best_score = score;
            cut_child = v;
        }
    }
    if (cut_child < 0) return false;

    std::vector<char> in_left(static_cast<std::size_t>(n), 0);
    std::queue<int> stack;
    stack.push(cut_child);
    while (!stack.empty()) {
        const int u = stack.front();
        stack.pop();
        in_left[static_cast<std::size_t>(u)] = 1;
        for (const int child : children[static_cast<std::size_t>(u)]) stack.push(child);
    }
    left.reserve(static_cast<std::size_t>(n));
    right.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (in_left[static_cast<std::size_t>(i)]) {
            left.push_back(cells[static_cast<std::size_t>(i)]);
        } else {
            right.push_back(cells[static_cast<std::size_t>(i)]);
        }
    }
    if (left.empty() || right.empty()) return false;
    return true;
}

std::vector<int> build_connected_assignment(
    const CellCover& cover, const std::vector<double>& cell_weights,
    int target_partitions, std::uint32_t seed) {
    (void)seed;
    if (target_partitions <= 0) {
        throw std::invalid_argument("build_connected_assignment: bad k");
    }
    if (cell_weights.size() != cover.size()) {
        throw std::invalid_argument("build_connected_assignment: weight size");
    }
    struct Part {
        std::vector<CellId> cells;
        double weight{};
        bool splittable{true};
    };
    std::vector<Part> parts;
    Part all;
    all.cells.resize(cover.size());
    std::iota(all.cells.begin(), all.cells.end(), 0);
    all.weight = std::accumulate(cell_weights.begin(), cell_weights.end(), 0.0);
    parts.push_back(std::move(all));

    while (static_cast<int>(parts.size()) < target_partitions) {
        int pick = -1;
        double best_weight = -1.0;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (parts[i].splittable && parts[i].cells.size() > 1 &&
                parts[i].weight > best_weight) {
                best_weight = parts[i].weight;
                pick = static_cast<int>(i);
            }
        }
        if (pick < 0) break;
        std::vector<CellId> left_cells;
        std::vector<CellId> right_cells;
        if (!split_connected_cells(cover, parts[static_cast<std::size_t>(pick)].cells,
                                   cell_weights, left_cells, right_cells)) {
            parts[static_cast<std::size_t>(pick)].splittable = false;
            continue;
        }
        Part left;
        left.cells = std::move(left_cells);
        Part right;
        right.cells = std::move(right_cells);
        for (const CellId cell : left.cells) {
            left.weight += cell_weights[static_cast<std::size_t>(cell)];
        }
        for (const CellId cell : right.cells) {
            right.weight += cell_weights[static_cast<std::size_t>(cell)];
        }
        parts[static_cast<std::size_t>(pick)] = std::move(left);
        parts.push_back(std::move(right));
    }

    std::vector<int> assignment(cover.size(), -1);
    for (std::size_t p = 0; p < parts.size(); ++p) {
        for (const CellId cell : parts[p].cells) {
            assignment[static_cast<std::size_t>(cell)] = static_cast<int>(p);
        }
    }
    for (const int value : assignment) {
        if (value < 0) throw std::logic_error("assignment left an unassigned cell");
    }
    return assignment;
}

}  // namespace spatial::h3
