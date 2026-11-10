#pragma once
#include "core/index.hpp"
#include <map>
#include <unordered_map>
namespace spatial {
// Reference implementation used by the M1/M2 validation path.
class Oracle {
    std::map<Id, Point> rows_;
public:
    void set(Id id, Point p) { rows_.insert_or_assign(id,p); }
    void erase(Id id) { rows_.erase(id); }
    std::vector<Id> query(Query q) const;
};

// Structure-of-arrays brute force used by the long M3 replay. It keeps the same
// closed-boundary predicate as the index implementations but stores coordinates
// contiguously, so the full 43,200-query correctness pass stays affordable.
class ArrayOracle {
public:
    void set(Id id, Point p);
    void erase(Id id);
    std::size_t size() const { return ids_.size(); }
    // Returns matching ids in ascending order.
    std::vector<Id> query(Query q) const;
private:
    std::vector<Id> ids_;
    std::vector<double> xs_, ys_;
    std::vector<char> alive_;
    std::unordered_map<Id, std::size_t> slot_;
};
}
