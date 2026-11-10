#include "core/index.hpp"
#include "validation/oracle.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace spatial;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void exercise(Index& index) {
    Oracle oracle;
    auto set = [&](Id id, Point p) { index.set(id, p); oracle.set(id, p); };
    auto del = [&](Id id) { index.erase(id); oracle.erase(id); };
    auto check = [&](Query q) {
        auto answer = index.query(q);
        std::sort(answer.ids.begin(), answer.ids.end());
        require(answer.ids == oracle.query(q), "query result mismatch");
        require(answer.candidates >= answer.ids.size(), "invalid candidate counter");
        return answer.ids;
    };
    const Query whole{{-500,-500,500,500}, false, {}, 0};
    require(check(whole).empty(), "nonempty new index");
    set(1, {-10, 0}); set(2, {10, 0}); set(3, {6, 8}); set(4, {10, 10});
    require(check(circle({0,0},10)) == std::vector<Id>{1,2,3}, "inclusive circle");
    require(check({{-10,0,10,10},false,{},0}).size() == 4, "inclusive rectangle");
    set(1, {2000, 2000}); set(1, {-400, -400});
    del(2); del(2);
    for (Id id = 100; id < 200; ++id) set(id, {0, 0});
    require(check(circle({0,0},0)).size() == 100, "duplicate point bucket");
    for (Id id = 100; id < 200; ++id) del(id);
    require(check(circle({0,0},0)).empty(), "delete duplicate point bucket");
    set(999, {1e-13, 5000});
    auto edge = check(circle({-10000,5000},10000));
    require(std::find(edge.begin(),edge.end(),999) == edge.end(),
            "circle edge rounded distance must not admit an out-of-bounds point");
    del(999);

    for (unsigned seed : {101,203,307,409,503}) {
        std::mt19937 rng(seed);
        auto xy = [&]() { return static_cast<double>(int(rng() % 801) - 400); };
        for (int op = 0; op < 2500; ++op) {
            Id id = rng() % 256;
            switch (op % 5) {
                case 0: del(id); break;
                case 1: case 2: set(id, {xy(),xy()}); break;
                case 3: check(circle({xy(),xy()}, rng() % 150)); break;
                case 4: {
                    double x = xy(), y = xy();
                    check({{x,y,x+100,y+100},false,{},0});
                    break;
                }
            }
        }
        check(whole);
    }
}

int main() {
    try {
        for (auto [name, param] : {std::pair{"UNIFORM_GRID",10},
                                  {"PR_QUADTREE",8},{"RSTAR_BOOST",16},
                                  {"RSTAR_LIBSPATIALINDEX",16}}) {
            exercise(*make_index(name,param,{-500,-500,500,500}));
            std::cout << name << " boundary and 5 x 2500 mixed operations passed\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
