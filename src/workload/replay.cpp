#include "replay.hpp"
#include <fstream>

namespace spatial {
Trace read_trace(const std::filesystem::path& path) {
    std::ifstream in(path);
    Trace trace;
    std::string signature;
    std::size_t initial_count{}, event_count{};
    auto& [x0,y0,x1,y1] = trace.bounds;
    if (!(in >> signature >> x0 >> y0 >> x1 >> y1 >> initial_count >> event_count) ||
        signature != "V2M1" || !(x0 < x1 && y0 < y1))
        throw std::runtime_error("invalid trace header");
    trace.initial.resize(initial_count);
    for (auto& [id,p] : trace.initial)
        if (!(in >> id >> p.x >> p.y)) throw std::runtime_error("invalid initial point");
    trace.events.resize(event_count);
    for (auto& e : trace.events) {
        if (!(in >> e.type)) throw std::runtime_error("missing event");
        auto& box = e.query.box;
        switch (e.type) {
            case 'I': case 'U': in >> e.id >> e.point.x >> e.point.y; break;
            case 'D': in >> e.id; break;
            case 'R':
                in >> box.x0 >> box.y0 >> box.x1 >> box.y1;
                if (!(box.x0 <= box.x1 && box.y0 <= box.y1))
                    throw std::runtime_error("inverted rectangle");
                break;
            case 'C': {
                Point center;
                double radius{};
                in >> center.x >> center.y >> radius;
                if (radius < 0) throw std::runtime_error("negative circle radius");
                e.query = circle(center,radius);
                break;
            }
            default: throw std::runtime_error("unknown event type");
        }
        if (!in) throw std::runtime_error("truncated event");
    }
    if (in >> signature) throw std::runtime_error("extra events in trace");
    return trace;
}
}
