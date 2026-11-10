// Unit tests for the M3 trace reader, the selector configuration and the
// history-only decision rules of S3 and S4.
#include "core/clock.hpp"
#include "indexes/regional.hpp"
#include "indexes/selector.hpp"
#include "workload/m3.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace spatial;

namespace {
int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error("FAILED: " + message);
}

SelectorConfig base_config() {
    SelectorConfig config;
    config.threshold = {24, 25, 0.5, 1e9, 1.0, 1.0, 0};
    config.margin = 0.1;
    config.alpha = 1.0;
    // Scanning is linear in entities; a tree has a per-query floor and costs per update.
    config.cost.query[0][1] = 10;
    config.cost.query[1][0] = 0.5;
    config.cost.query[2][0] = 1.0;
    config.cost.update[0][0] = 0;
    config.cost.update[1][0] = 1;
    config.cost.update[2][0] = 1.2;
    for (unsigned from = 0; from < 3; ++from)
        for (unsigned to = 0; to < 3; ++to)
            if (from != to) config.cost.transition[from][to] = {0, 0};
    return config;
}

AdaptiveRegionalIndex make_index(Mode mode = Mode::Scan) {
    return AdaptiveRegionalIndex({0, 0, 1600, 1600}, mode, 20, 8, false, true);
}

void fill_region(AdaptiveRegionalIndex& index, unsigned region, unsigned count, Id base) {
    const double x0 = 100.0 * (region % 16), y0 = 100.0 * (region / 16);
    for (unsigned i = 0; i < count; ++i)
        index.set(base + i, {x0 + 1.0 + (i % 7), y0 + 1.0 + (i / 7)});
}

void query_region(AdaptiveRegionalIndex& index, unsigned region, unsigned count) {
    const double x0 = 100.0 * (region % 16) + 10, y0 = 100.0 * (region / 16) + 10;
    for (unsigned i = 0; i < count; ++i) index.query(circle({x0, y0}, 30));
}

void touch_region(AdaptiveRegionalIndex& index, unsigned region, unsigned count, Id base) {
    const double x0 = 100.0 * (region % 16), y0 = 100.0 * (region / 16);
    for (unsigned i = 0; i < count; ++i)
        index.set(base + i, {x0 + 2.0 + (i % 7), y0 + 2.0 + (i / 7)});
}

void test_threshold_and_cooldown() {
    auto index = make_index();
    fill_region(index, 0, 40, 1);
    fill_region(index, 5, 3, 1000);
    index.reset_window();
    query_region(index, 0, 60);
    Selector selector(Policy::Threshold, base_config());
    auto decisions = selector.decide(index, 50);
    require(index.mode(0) == Mode::FineGrid, "hot region promoted to FINE_GRID");
    require(index.mode(5) == Mode::Scan, "cold region stays in SCAN");
    require(selector.counters().promotions == 1, "one promotion counted");
    require(!decisions.empty(), "proposals reported");

    // No new query load: the region should be demoted, but only after two windows.
    index.reset_window();
    touch_region(index, 0, 40, 1);
    decisions = selector.decide(index, 100);
    require(index.mode(0) == Mode::FineGrid, "cooldown blocks demotion after one window");
    bool cooldown_seen = false;
    for (const auto& decision : decisions)
        if (decision.region == 0 && decision.outcome == Outcome::Cooldown) cooldown_seen = true;
    require(cooldown_seen, "cooldown reported on the blocked proposal");
    index.reset_window();
    touch_region(index, 0, 40, 1);
    selector.decide(index, 150);
    require(index.mode(0) == Mode::Scan, "demoted once the cooldown expires");
    require(selector.counters().demotions == 1, "one demotion counted");
}

void test_rebuild_budget_and_free_demotions() {
    {
        auto index = make_index();
        for (unsigned region = 0; region < 8; ++region) fill_region(index, region, 40, 1000 + 100 * region);
        index.reset_window();
        Selector selector(Policy::Threshold, base_config());
        for (unsigned region = 0; region < 8; ++region) query_region(index, region, 60);
        auto decisions = selector.decide(index, 50);
        unsigned applied = 0;
        for (const auto& decision : decisions) applied += decision.applied ? 1 : 0;
        require(applied == Selector::max_rebuilds_per_window, "rebuild budget caps one window");
        index.reset_window();
        for (unsigned region = 0; region < 8; ++region) query_region(index, region, 60);
        selector.decide(index, 150);
        unsigned indexed = 0;
        for (unsigned region = 0; region < 256; ++region) indexed += index.mode(region) != Mode::Scan;
        require(indexed == 8, "the next window promotes the remaining regions");
    }
    {
        auto index = make_index(Mode::LocalQuad);
        for (unsigned region = 0; region < 8; ++region) fill_region(index, region, 40, 1000 + 100 * region);
        index.reset_window();
        Selector selector(Policy::Threshold, base_config());
        for (unsigned region = 0; region < 8; ++region) touch_region(index, region, 40, 1000 + 100 * region);
        selector.decide(index, 50);
        unsigned demoted = 0;
        for (unsigned region = 0; region < 8; ++region) demoted += index.mode(region) == Mode::Scan;
        require(demoted == 8, "demotions are not limited by the rebuild budget");
        // All 256 regions start indexed, so the empty ones are released in the same window.
        require(selector.counters().demotions >= 8, "every copied region counted one demotion");
    }
}

void test_dense_floor() {
    auto index = make_index(Mode::LocalQuad);
    fill_region(index, 4, 600, 1);
    index.reset_window();
    auto config = base_config();
    config.threshold.dense_entities = 512;
    Selector selector(Policy::Greedy, config);
    for (unsigned i = 0; i < 40; ++i) index.set(1 + i, {402.0 + i % 7, 2.0 + i / 7});
    selector.decide(index, 50);
    require(index.mode(4) == Mode::LocalQuad, "the structural floor keeps a dense region indexed");

    auto cold = make_index(Mode::LocalQuad);
    fill_region(cold, 4, 100, 1);
    cold.reset_window();
    Selector plain(Policy::Greedy, base_config());
    for (unsigned i = 0; i < 40; ++i) cold.set(1 + i, {402.0 + i % 7, 2.0 + i / 7});
    plain.decide(cold, 50);
    require(cold.mode(4) == Mode::Scan, "without the floor a cold region is demoted");
}

void test_greedy_choice() {
    auto index = make_index();
    fill_region(index, 3, 100, 1);
    index.reset_window();
    query_region(index, 3, 40);
    Selector selector(Policy::Greedy, base_config());
    selector.decide(index, 50);
    require(index.mode(3) == Mode::FineGrid, "greedy indexes a query-heavy region at the cheaper mode");

    index.reset_window();
    for (unsigned i = 0; i < 40; ++i) index.set(1 + i, {300.0 + i % 10, 1.0 + i / 10});
    selector.decide(index, 150);
    require(index.mode(3) == Mode::Scan, "greedy drops the index when updates dominate");
}

void test_ewma_retention() {
    for (bool smooth : {false, true}) {
        auto index = make_index(Mode::LocalQuad);
        fill_region(index, 6, 100, 1);
        index.reset_window();
        auto config = base_config();
        config.cost.query[2][0] = 0.1;
        config.cost.update[1][0] = 1.5;   // the local quadtree is the cheapest indexed mode here
        config.alpha = smooth ? 0.8 : 1.0;
        Selector selector(Policy::Greedy, config);
        query_region(index, 6, 40);
        selector.decide(index, 50);
        require(index.mode(6) == Mode::LocalQuad, "hot region retained");
        index.reset_window();
        for (unsigned i = 0; i < 40; ++i) index.set(1 + i, {602.0 + i % 7, 2.0 + i / 7});
        selector.decide(index, 150);
        require(index.mode(6) == (smooth ? Mode::LocalQuad : Mode::Scan),
                smooth ? "smoothed load keeps the region indexed through one cold window"
                       : "raw last window demotes after one cold window");
    }
}

void test_history_only_and_empty_window() {
    auto index = make_index();
    fill_region(index, 7, 40, 1);
    index.reset_window();
    Selector selector(Policy::Greedy, base_config());
    require(selector.decide(index, 50).empty(), "an empty window cannot justify a switch");
    query_region(index, 7, 60);
    require(index.mode(7) == Mode::Scan, "no decision without a decision point");
    selector.decide(index, 100);
    require(index.mode(7) == Mode::FineGrid, "the accumulated window drives the next decision");
}

void write_config(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path);
    out << text;
}

void test_config_parser() {
    const auto path = std::filesystem::current_path() / "m3_selector_test.txt";
    const std::string good =
        "version 1\n"
        "margin 0.15\n"
        "alpha 0.85\n"
        "threshold 24 25 0.5 600 0.15 0.7 512\n"
        "query SCAN 0 10 0 0 0\n"
        "query FINE_GRID 0.5 0 1 0 0\n"
        "query LOCAL_QUAD 0.4 0 0 0 0\n"
        "update SCAN 0 0\n"
        "update FINE_GRID 1 0\n"
        "update LOCAL_QUAD 1.2 0\n"
        "switch SCAN FINE_GRID 0 0\n"
        "switch SCAN LOCAL_QUAD 0 0\n"
        "switch SCAN SCAN 0 0\n"
        "switch FINE_GRID SCAN 0 0\n"
        "switch FINE_GRID LOCAL_QUAD 0 0\n"
        "switch FINE_GRID FINE_GRID 0 0\n"
        "switch LOCAL_QUAD SCAN 0 0\n"
        "switch LOCAL_QUAD FINE_GRID 0 0\n"
        "switch LOCAL_QUAD LOCAL_QUAD 0 0\n";
    write_config(path, good);
    auto config = read_selector_config(path);
    require(std::abs(config.margin - 0.15) < 1e-12, "margin parsed");
    require(std::abs(config.alpha - 0.85) < 1e-12, "alpha parsed");
    require(std::abs(config.threshold.quad_entities - 600) < 1e-12, "threshold parsed");
    require(std::abs(config.threshold.dense_entities - 512) < 1e-12, "dense floor parsed");
    require(std::abs(config.cost.query[0][1] - 10) < 1e-12, "query coefficient parsed");
    require(std::abs(config.cost.update[2][0] - 1.2) < 1e-12, "update coefficient parsed");

    write_config(path, "version 1\nmargin 0.1\nthreshold 1 1 1 1 1 1 0\n");
    bool thrown = false;
    try { read_selector_config(path); } catch (const std::runtime_error&) { thrown = true; }
    require(thrown, "incomplete config rejected");

    std::string text = good;
    const std::string pair = "switch SCAN SCAN 0 0\n";
    text.erase(text.find(pair), pair.size());
    write_config(path, text);
    thrown = false;
    try { read_selector_config(path); } catch (const std::runtime_error&) { thrown = true; }
    require(thrown, "missing switch pair rejected");
    std::filesystem::remove(path);
}

void write_trace(const std::filesystem::path& path, bool truncate) {
    struct Header {
        char magic[8];
        double world[4];
        std::uint64_t entities, events;
    };
#pragma pack(push, 1)
    struct RawEvent {
        char type;
        char pad[7];
        std::int64_t id;
        double a, b, c, d;
    };
#pragma pack(pop)
    static_assert(sizeof(RawEvent) == 48);
    Header header{{'M', '3', 'B', 'I', 'N', '0', '0', '1'}, {0, 0, 100, 100}, 2, 5};
    const double points[4] = {10, 10, 20, 20};
    const RawEvent events[5] = {
        {'U', {}, 1, 12, 12, 0, 0},
        {'R', {}, 0, 0, 0, 30, 30},
        {'C', {}, 0, 15, 15, 5, 0},
        {'P', {}, 50, 0, 0, 0, 0},
        {'U', {}, 2, 21, 21, 0, 0},
    };
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create test trace " + path.string());
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(points), sizeof(points));
    if (!truncate) out.write(reinterpret_cast<const char*>(events), sizeof(events));
    if (!out) throw std::runtime_error("cannot write test trace " + path.string());
}

void test_trace_reader() {
    const auto path = std::filesystem::current_path() / "m3_trace_test.m3bin";
    write_trace(path, false);
    const auto trace = read_m3_trace(path);
    require(trace.points.size() == 2, "two initial points");
    require(trace.update_count == 2, "two updates");
    require(trace.query_count == 2, "two queries");
    require(trace.decision_points == 1, "one decision boundary");
    require(trace.events[1].type == 'R' && trace.events[1].query.box.x1 == 30, "rectangle payload");
    require(trace.events[2].query.circular && trace.events[2].query.radius == 5, "circle payload");
    require(trace.events[3].step == 50, "boundary step");

    write_trace(path, true);
    bool thrown = false;
    try { read_m3_trace(path); } catch (const std::runtime_error&) { thrown = true; }
    require(thrown, "truncated trace rejected");
    std::filesystem::remove(path);
}

void test_tick() {
    const auto first = Tick::now();
    volatile double sink = 0;
    for (int i = 0; i < 100000; ++i) sink = sink + i;
    const auto second = Tick::now();
    require(Tick::ns_per_tick() > 0, "TSC conversion factor is positive");
    require(Tick::ns(second - first) >= 0, "TSC duration is non-negative");
    (void)sink;
}
}

int main() {
    try {
        test_trace_reader();
        test_config_parser();
        test_tick();
        test_threshold_and_cooldown();
        test_rebuild_budget_and_free_demotions();
        test_dense_floor();
        test_greedy_choice();
        test_ewma_retention();
        test_history_only_and_empty_window();
        std::cout << checks << " M3 checks passed: trace reader, config parser, TSC timer, "
                     "threshold promotion/demotion, cooldown, rebuild budget, free demotions, "
                     "dense floor, greedy cost choice, EWMA retention, history-only decisions\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
