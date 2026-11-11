#include "h3/local_projection.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spatial::h3 {

double normalize_lng_deg(double lng_deg) {
    double value = std::fmod(lng_deg + 180.0, 360.0);
    if (value < 0.0) value += 360.0;
    return value - 180.0;
}

LocalProjection::LocalProjection(double origin_lat_deg,
                                 double origin_lng_deg,
                                 double meters_per_unit)
    : origin_lat_deg_(origin_lat_deg),
      origin_lng_deg_(normalize_lng_deg(origin_lng_deg)),
      origin_lat_rad_(deg2rad(origin_lat_deg)),
      origin_lng_rad_(deg2rad(normalize_lng_deg(origin_lng_deg))),
      meters_per_unit_(meters_per_unit),
      cos_lat0_(std::cos(deg2rad(origin_lat_deg))) {
    if (!std::isfinite(origin_lat_deg) || !std::isfinite(origin_lng_deg) ||
        !std::isfinite(meters_per_unit)) {
        throw std::invalid_argument("LocalProjection: non-finite parameter");
    }
    if (origin_lat_deg <= -89.0 || origin_lat_deg >= 89.0) {
        throw std::invalid_argument(
            "LocalProjection: origin latitude must stay away from the poles");
    }
    if (meters_per_unit <= 0.0) {
        throw std::invalid_argument("LocalProjection: metres per unit <= 0");
    }
}

GeoPoint LocalProjection::toGeo(Point local) const {
    if (!std::isfinite(local.x) || !std::isfinite(local.y)) {
        throw std::invalid_argument("LocalProjection::toGeo: non-finite point");
    }
    const double lat_rad =
        origin_lat_rad_ + (local.y * meters_per_unit_) / kEarthRadiusM;
    if (lat_rad < -kPi / 2.0 || lat_rad > kPi / 2.0) {
        throw std::invalid_argument(
            "LocalProjection::toGeo: point leaves the spherical domain");
    }
    const double lng_rad =
        origin_lng_rad_ +
        (local.x * meters_per_unit_) / (kEarthRadiusM * cos_lat0_);
    GeoPoint out;
    out.lat_deg = rad2deg(lat_rad);
    out.lng_deg = normalize_lng_deg(rad2deg(lng_rad));
    return out;
}

Point LocalProjection::toLocal(GeoPoint geo) const {
    if (!std::isfinite(geo.lat_deg) || !std::isfinite(geo.lng_deg)) {
        throw std::invalid_argument("LocalProjection::toLocal: non-finite point");
    }
    const double lat_rad = deg2rad(geo.lat_deg);
    double dlng_rad = deg2rad(normalize_lng_deg(geo.lng_deg - origin_lng_deg_));
    // Choose the shortest longitudinal displacement around the anchor.
    if (dlng_rad > kPi) dlng_rad -= 2.0 * kPi;
    if (dlng_rad < -kPi) dlng_rad += 2.0 * kPi;
    Point out;
    out.x = dlng_rad * kEarthRadiusM * cos_lat0_ / meters_per_unit_;
    out.y = (lat_rad - origin_lat_rad_) * kEarthRadiusM / meters_per_unit_;
    return out;
}

}  // namespace spatial::h3
