#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace spatial::h3 {

// H3 uses the WGS84 authalic radius for its spherical model. Keeping the same
// constant here avoids mixing two Earth radii in projection and distance code.
inline constexpr double kEarthRadiusM = 6371007.1809;
inline constexpr double kPi = 3.141592653589793238462643383279502884;

inline double deg2rad(double degrees) { return degrees * kPi / 180.0; }
inline double rad2deg(double radians) { return radians * 180.0 / kPi; }

struct GeoPoint {
    double lat_deg{};
    double lng_deg{};
    bool operator==(const GeoPoint&) const = default;
};

// Planar bounding box in the local simulation frame (metres).
struct BBox {
    double x0{0.0};
    double y0{0.0};
    double x1{0.0};
    double y1{0.0};
    bool valid{false};

    static BBox empty() { return {}; }

    void expand(double x, double y) {
        if (!valid) {
            x0 = x1 = x;
            y0 = y1 = y;
            valid = true;
            return;
        }
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
    }

    void expand(const BBox& other) {
        if (!other.valid) return;
        expand(other.x0, other.y0);
        expand(other.x1, other.y1);
    }

    bool overlaps(const BBox& other) const {
        return valid && other.valid && x0 <= other.x1 && other.x0 <= x1 &&
               y0 <= other.y1 && other.y0 <= y1;
    }

    bool overlaps(double ox0, double oy0, double ox1, double oy1) const {
        return valid && x0 <= ox1 && ox0 <= x1 && y0 <= oy1 && oy0 <= y1;
    }

    bool contains(double x, double y, double eps = 1e-9) const {
        return valid && x >= x0 - eps && x <= x1 + eps && y >= y0 - eps &&
               y <= y1 + eps;
    }
};

// Great-circle distance in metres, used only for validation/diagnostics.
double haversine_m(GeoPoint a, GeoPoint b);

}  // namespace spatial::h3
