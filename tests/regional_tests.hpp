#pragma once
#include "indexes/regional.hpp"
#include "validation/oracle.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

inline void test_regions() {
    using namespace spatial;
    const Box world{-800,-800,800,800};
    for (Mode mode : {Mode::Scan, Mode::FineGrid, Mode::LocalQuad}) {
        AdaptiveRegionalIndex index(world,mode,25,8,true);
        Oracle oracle;
        auto set = [&](Id id, Point p) { index.set(id,p); oracle.set(id,p); };
        auto erase = [&](Id id) { index.erase(id); oracle.erase(id); };
        auto query = [&](Query q) {
            auto result = index.query(q).ids;
            std::sort(result.begin(),result.end());
            if (result != oracle.query(q)) throw std::runtime_error("regional oracle mismatch");
        };
        const Query whole{{-3000,-3000,3000,3000},false,{},0};
        const auto region = index.region_of({0,0});
        set(1,{-0.01,0}); set(2,{0,0}); set(3,{25,0}); set(4,{100,0});
        set(5,{-800,-800}); set(6,{800,800}); set(7,{2500,2500});
        query(whole);
        query({{0,0,100,0},false,{},0});
        query(circle({0,0},25));
        query(circle({0,0},0));
        set(8,{1e-13,5000});
        query(circle({-10000,5000},10000));
        erase(8);
        index.switch_mode(region,Mode::FineGrid);
        index.switch_mode(index.region_of({100,0}),Mode::LocalQuad);
        index.switch_mode(index.region_of({-0.01,0}),Mode::Scan);
        query(whole);
        query({{-1,0,100,0},false,{},0});
        set(1,{0,0}); set(4,{-0.01,0}); set(2,{100,0}); erase(3);
        query(whole);
        query(circle({0,0},1));
        index.switch_mode(region,Mode::Scan);
        index.reset_window();
        set(1,{5,5}); set(9000,{7,7}); erase(9000);
        const auto& cold = index.region_stats(region);
        if (cold.local_updates != 0 || cold.local_update_ns != 0 ||
            cold.object_updates != 3 || index.entity_count(region) != 1 ||
            index.mode(region) != Mode::Scan)
            throw std::runtime_error("SCAN accounted auxiliary updates");
        index.switch_mode(region,Mode::LocalQuad);
        index.switch_mode(region,Mode::LocalQuad);
        index.switch_mode(region,Mode::Scan);
        if (index.region_stats(region).switches != 2 ||
            index.region_stats(region).builds != 1)
            throw std::runtime_error("switch and build counters inconsistent");
        for (unsigned seed : {101,203,307,409,503}) {
            std::mt19937 rng(seed);
            for (unsigned i=0; i<4000; ++i) {
                auto id = rng()%512;
                double x = int(rng()%2401)-1200, y = int(rng()%2401)-1200;
                switch (rng()%6) {
                    case 0: index.switch_mode(rng()%256,Mode(rng()%3)); break;
                    case 1: erase(id); break;
                    case 2: case 3: set(id,{x,y}); break;
                    case 4: query(circle({x,y},rng()%450)); break;
                    case 5: query({{x,y,x+350,y+350},false,{},0}); break;
                }
            }
            query(whole);
        }
    }
}
