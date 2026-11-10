#include "prior.hpp"
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace spatial {
ScenarioPrior ScenarioPrior::read(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open scenario prior " + path.string());
    ScenarioPrior prior;
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty scenario prior");
    const auto header = line;
    if (header.find("window") == std::string::npos || header.find("region") == std::string::npos ||
        header.find("query_count") == std::string::npos || header.find("entity_count") == std::string::npos ||
        header.find("update_count") == std::string::npos || header.find("coverage") == std::string::npos)
        throw std::runtime_error("unexpected scenario prior header");
    const bool sampled = header.find("sample") != std::string::npos;
    struct Row { unsigned sample; std::size_t window; unsigned region; PriorLoad load; };
    std::vector<Row> rows;
    std::size_t window = 0;
    unsigned region = 0, sample = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream stream(line);
        std::string field;
        auto next = [&](double& value) {
            if (!std::getline(stream, field, ',')) throw std::runtime_error("short prior row");
            value = std::stod(field);
        };
        double s = 0, w = 0, r = 0;
        if (sampled) next(s);
        next(w); next(r);
        PriorLoad load;
        next(load.queries); next(load.entities); next(load.updates); next(load.coverage);
        if (s < 0 || w < 0 || r < 0) throw std::runtime_error("negative prior index");
        sample = std::max(sample, static_cast<unsigned>(s));
        window = std::max(window, static_cast<std::size_t>(w));
        region = std::max(region, static_cast<unsigned>(r));
        rows.push_back({static_cast<unsigned>(s), static_cast<std::size_t>(w),
                        static_cast<unsigned>(r), load});
    }
    if (rows.empty()) throw std::runtime_error("scenario prior has no rows");
    prior.samples_ = sample + 1;
    prior.regions_ = region + 1;
    prior.windows_ = window + 1;
    prior.data_.assign(prior.samples_ * prior.windows_ * prior.regions_, PriorLoad{});
    std::vector<char> seen(prior.data_.size(), 0);
    for (const Row& row : rows) {
        if (row.region >= prior.regions_ || row.window >= prior.windows_ || row.sample >= prior.samples_)
            throw std::runtime_error("prior index out of range");
        const auto index = (static_cast<std::size_t>(row.sample) * prior.windows_ + row.window) *
                           prior.regions_ + row.region;
        if (seen[index]) throw std::runtime_error("duplicate prior cell");
        seen[index] = 1;
        prior.data_[index] = row.load;
    }
    return prior;
}

const PriorLoad& ScenarioPrior::at(std::size_t window, unsigned region) const {
    if (window >= windows_) throw std::out_of_range("prior window out of range");
    if (region >= regions_) throw std::out_of_range("prior region out of range");
    return data_[window * regions_ + region];
}

const PriorLoad& ScenarioPrior::at(unsigned sample, std::size_t window, unsigned region) const {
    if (sample >= samples_) throw std::out_of_range("prior sample out of range");
    if (window >= windows_) throw std::out_of_range("prior window out of range");
    if (region >= regions_) throw std::out_of_range("prior region out of range");
    return data_[(static_cast<std::size_t>(sample) * windows_ + window) * regions_ + region];
}
}
