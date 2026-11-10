#pragma once
#include "core/index.hpp"
#include <map>
namespace spatial {
class Oracle {
    std::map<Id, Point> rows_;
public:
    void set(Id id, Point p) { rows_.insert_or_assign(id,p); }
    void erase(Id id) { rows_.erase(id); }
    std::vector<Id> query(Query q) const;
};
}
