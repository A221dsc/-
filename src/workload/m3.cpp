#include "m3.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace spatial {
namespace {
constexpr std::size_t header_size = 8 + 4 * sizeof(double) + 2 * sizeof(std::uint64_t);
constexpr std::size_t point_size = 2 * sizeof(double);
constexpr std::size_t event_size = 48;
constexpr char magic[8] = {'M', '3', 'B', 'I', 'N', '0', '0', '1'};

std::vector<std::uint64_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open M3 trace " + path.string());
    const auto size = static_cast<std::size_t>(in.tellg());
    if (size < header_size || (size - header_size) % 8 != 0)
        throw std::runtime_error("invalid M3 trace length " + path.string());
    std::vector<std::uint64_t> data(size / 8);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error("cannot read M3 trace " + path.string());
    return data;
}

template <class T>
T field(const std::vector<std::uint64_t>& data, std::size_t offset) {
    T value{};
    std::memcpy(&value, reinterpret_cast<const char*>(data.data()) + offset, sizeof(T));
    return value;
}
}

M3Trace read_m3_trace(const std::filesystem::path& path) {
    const auto data = read_bytes(path);
    const auto total = data.size() * 8;
    if (std::memcmp(data.data(), magic, sizeof(magic)) != 0)
        throw std::runtime_error("bad M3 magic in " + path.string());
    M3Trace trace;
    trace.bytes = total;
    const double x0 = field<double>(data, 8);
    const double y0 = field<double>(data, 16);
    const double x1 = field<double>(data, 24);
    const double y1 = field<double>(data, 32);
    trace.bounds = {x0, y0, x1, y1};
    const auto entities = field<std::uint64_t>(data, 40);
    const auto events = field<std::uint64_t>(data, 48);
    if (!(x1 > x0 && y1 > y0) || entities == 0 || events == 0)
        throw std::runtime_error("invalid M3 header in " + path.string());
    const std::size_t expected = header_size + entities * point_size + events * event_size;
    if (total != expected)
        throw std::runtime_error("M3 size mismatch: expected " + std::to_string(expected) +
                                 " bytes, found " + std::to_string(total));
    trace.points.reserve(entities);
    std::size_t offset = header_size;
    for (std::uint64_t i = 0; i < entities; ++i) {
        trace.points.push_back({field<double>(data, offset), field<double>(data, offset + 8)});
        offset += point_size;
    }
    trace.events.reserve(events);
    std::uint32_t last_step = 0;
    for (std::uint64_t i = 0; i < events; ++i) {
        const char type = static_cast<char>(field<std::uint8_t>(data, offset));
        const auto id = static_cast<std::int64_t>(field<std::uint64_t>(data, offset + 8));
        const double a = field<double>(data, offset + 16);
        const double b = field<double>(data, offset + 24);
        const double c = field<double>(data, offset + 32);
        const double d = field<double>(data, offset + 40);
        M3Event event;
        event.type = type;
        switch (type) {
            case 'U':
                if (id < 1 || static_cast<std::uint64_t>(id) > entities)
                    throw std::runtime_error("M3 update id out of range at event " +
                                             std::to_string(i));
                event.id = id;
                event.point = {a, b};
                ++trace.update_count;
                break;
            case 'D':
                if (id < 1 || static_cast<std::uint64_t>(id) > entities)
                    throw std::runtime_error("M3 delete id out of range at event " +
                                             std::to_string(i));
                event.id = id;
                ++trace.delete_count;
                break;
            case 'R':
                if (!(c >= a && d >= b))
                    throw std::runtime_error("M3 rectangle is inverted at event " + std::to_string(i));
                event.query = {{a, b, c, d}, false, {}, 0};
                ++trace.query_count;
                break;
            case 'C':
                if (!(c > 0)) throw std::runtime_error("M3 circle radius is not positive at event " +
                                                      std::to_string(i));
                event.query = circle({a, b}, c);
                ++trace.query_count;
                break;
            case 'P':
                if (id <= 0 || static_cast<std::uint64_t>(id) > ~std::uint32_t{0})
                    throw std::runtime_error("M3 boundary step out of range");
                event.step = static_cast<std::uint32_t>(id);
                if (event.step <= last_step)
                    throw std::runtime_error("M3 boundary steps are not increasing");
                last_step = event.step;
                ++trace.decision_points;
                break;
            default:
                throw std::runtime_error("unknown M3 event type at event " + std::to_string(i));
        }
        trace.events.push_back(event);
        offset += event_size;
    }
    if (!trace.query_count || !trace.update_count || !trace.decision_points)
        throw std::runtime_error("M3 trace lacks required event kinds");
    return trace;
}
}
