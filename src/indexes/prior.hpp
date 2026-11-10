#pragma once
#include <cstddef>
#include <filesystem>
#include <vector>

namespace spatial {
// One region-window of the nominal scenario derivation. Values are expectations
// computed from planned data only: planned routes, phase schedule and periodic
// queries. No actual disturbance seed enters this table.
struct PriorLoad {
    double queries{};
    double entities{};
    double updates{};
    double coverage{};
};

class ScenarioPrior {
public:
    ScenarioPrior() = default;
    static ScenarioPrior read(const std::filesystem::path& path);
    bool valid() const { return regions_ > 0; }
    bool has(std::size_t window) const { return window < windows_; }
    unsigned regions() const { return regions_; }
    std::size_t windows() const { return windows_; }
    const PriorLoad& at(std::size_t window, unsigned region) const;

private:
    unsigned regions_{};
    std::size_t windows_{};
    std::vector<PriorLoad> data_;
};
}
