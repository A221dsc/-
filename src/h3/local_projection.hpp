#pragma once

#include "core/index.hpp"
#include "h3/geo.hpp"

namespace spatial::h3 {

// A deliberately small, auditable local projection for simulation maps.
//
// The simulation frame is a metric plane: x is east, y is north, one unit is
// `meters_per_unit` metres. The projection is an equirectangular (plate
// carree) map anchored at (origin_lat_deg, origin_lng_deg). It is intended for
// a bounded theatre map, not for global use. The inverse is exact up to
// floating point and longitude wrapping.
class LocalProjection {
public:
    LocalProjection(double origin_lat_deg, double origin_lng_deg,
                    double meters_per_unit = 1.0);

    GeoPoint toGeo(Point local) const;
    Point toLocal(GeoPoint geo) const;

    double meters_per_unit() const { return meters_per_unit_; }
    double origin_lat_deg() const { return origin_lat_deg_; }
    double origin_lng_deg() const { return origin_lng_deg_; }

private:
    double origin_lat_deg_{};
    double origin_lng_deg_{};
    double origin_lat_rad_{};
    double origin_lng_rad_{};
    double meters_per_unit_{1.0};
    double cos_lat0_{1.0};
};

double normalize_lng_deg(double lng_deg);

}  // namespace spatial::h3
