#include "h3/h3_p0_v2.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace spatial;
using namespace spatial::h3;
using namespace spatial::h3::p0v2;

namespace {
void require(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}

void check_frame(Replay& replay, const Frame& frame, std::vector<std::optional<Point>>& state,
                 const CellCover& cover) {
    const auto output = replay.run(frame, Clock::now());
    require(output.answers.size() == frame.queries.size(), "result count");
    std::vector<CellId> locations(state.size(), -1);
    for (std::size_t slot = 0; slot < state.size(); ++slot)
        if (state[slot]) locations[slot] = locate(cover, *state[slot]);
    for (std::size_t i = 0; i < frame.queries.size(); ++i) {
        const auto& query = frame.queries[i];
        auto ids = output.answers[i].ids;
        std::sort(ids.begin(), ids.end());
        std::vector<Id> expected;
        std::uint64_t candidates{};
        const auto cells = cover.candidate_cells(query);
        for (std::size_t slot = 0; slot < state.size(); ++slot) {
            if (!state[slot]) continue;
            if (query.matches(*state[slot])) expected.push_back(static_cast<Id>(slot));
            const auto cell = locations[slot];
            candidates += std::find(cells.begin(), cells.end(), cell) != cells.end();
        }
        require(std::adjacent_find(ids.begin(), ids.end()) == ids.end(), "duplicate hit");
        if (ids != expected) {
            std::vector<Id> missing, extra;
            std::set_difference(expected.begin(), expected.end(), ids.begin(), ids.end(),
                                std::back_inserter(missing));
            std::set_difference(ids.begin(), ids.end(), expected.begin(), expected.end(),
                                std::back_inserter(extra));
            std::cerr << "query=" << i << " actual=" << ids.size() << " expected="
                      << expected.size() << " missing:";
            for (const auto id : missing) std::cerr << ' ' << id << '@' << locations[id];
            std::cerr << " extra:";
            for (const auto id : extra) std::cerr << ' ' << id << '@' << locations[id];
            std::cerr << '\n';
            throw std::runtime_error("query is not the step-start snapshot");
        }
        require(output.answers[i].stats.candidate_cells == cells.size(), "candidate cell count");
        require(output.answers[i].stats.bucket_lookups == cells.size(), "bucket count");
        require(output.answers[i].stats.candidates == candidates, "candidate entity count");
    }
    for (const auto& update : frame.updates) state[update.slot] = update.point;
    replay.validate(state);
}

void stress(const CellCover& cover, Mode mode, int workers, int ticks, unsigned seed) {
    std::cout << "RUN " << name(mode) << " workers=" << workers
              << " seed=" << seed << std::endl;
    const Point a{1024, 1024};
    const Point b{3072, 3072};
    const Query whole{{0, 0, 4096, 4096}, false, {}, 0};
    std::vector<std::optional<Point>> state(192, a);
    state[0] = b;
    state[3] = std::nullopt;
    const bool profile = seed != 202;
    Replay replay(cover, state, mode, workers, profile);
    check_frame(replay, {{whole, circle(a, 40), circle(b, 40)},
                         {{0, a}, {1, b}, {2, std::nullopt}, {3, b}, {4, Point{1025, 1025}}}},
                state, cover);
    check_frame(replay, {{whole, circle(a, 40), circle(b, 40)},
                         {{0, b}, {1, a}, {2, b}, {3, std::nullopt}, {4, Point{1026, 1026}}}},
                state, cover);

    bool rejected = false;
    try {
        replay.run({{whole}, {{0, a}, {0, b}}}, Clock::now());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "duplicate update accepted");
    replay.validate(state);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        replay.run({{{{nan, 0, 4096, 4096}, false, {}, 0}}, {{0, a}}}, Clock::now());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "nonfinite query accepted");
    replay.validate(state);
    check_frame(replay, {{{{-1e100, -1e100, 1e100, 1e100}, false, {}, 0}}, {}}, state, cover);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> xy(0, 4096);
    std::vector<Slot> slots(state.size());
    std::iota(slots.begin(), slots.end(), 0);
    for (int tick = 0; tick < ticks; ++tick) {
        std::shuffle(slots.begin(), slots.end(), rng);
        Frame frame;
        frame.queries = {whole, circle(a, 150), circle(b, 300)};
        for (int k = 0; k < 6; ++k) {
            const Point p{xy(rng), xy(rng)};
            frame.queries.push_back(k % 2 == 0 ? circle(p, 400)
                : Query{{p.x - 400, p.y - 200, p.x + 400, p.y + 200}, false, {}, 0});
        }
        for (int k = 0; k < 144; ++k) {
            const std::array<Point, 3> choices{a, b, {xy(rng), xy(rng)}};
            frame.updates.push_back({slots[k], rng() % 7 == 0
                ? std::nullopt : std::optional<Point>{choices[rng() % choices.size()]}});
        }
        check_frame(replay, frame, state, cover);
    }
    std::cout << "PASS " << name(mode) << " workers=" << workers
              << " seed=" << seed << " profile=" << profile << " random_frames=" << ticks << std::endl;
}
}  // namespace

int main(int argc, char** argv) try {
    const int ticks = argc > 1 ? std::stoi(argv[1]) : 200;
    const int max_workers = argc > 2 ? std::stoi(argv[2]) : 16;
    require(ticks > 0 && max_workers > 0, "positive ticks and workers required");
    CellCover cover(PartitionConfig{});
    stress(cover, Mode::Serial, 1, ticks, 42);
    for (const Mode mode : {Mode::Lock, Mode::Static, Mode::Matched, Mode::Partition}) {
        for (const int workers : {1, 4, 8, 16, 32}) {
            if (workers > max_workers) continue;
            for (const unsigned seed : {101, 202, 303}) stress(cover, mode, workers, ticks, seed);
        }
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return EXIT_FAILURE;
}
