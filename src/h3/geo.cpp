#include "h3/geo.hpp"

#include <algorithm>

namespace spatial::h3 {

double haversine_m(GeoPoint a, GeoPoint b) {
    const double lat1 = deg2rad(a.lat_deg);
    const double lat2 = deg2rad(b.lat_deg);
    const double dlat = lat2 - lat1;
    const double dlng = deg2rad(b.lng_deg - a.lng_deg);
    const double s1 = std::sin(dlat / 2.0);
    const double s2 = std::sin(dlng / 2.0);
    const double h = std::clamp(
        s1 * s1 + std::cos(lat1) * std::cos(lat2) * s2 * s2, 0.0, 1.0);
    return 2.0 * kEarthRadiusM * std::asin(std::sqrt(h));
}

}  // namespace spatial::h3
