#include "core/index.hpp"
#include <algorithm>
#include <array>
#include <numeric>

namespace spatial {
namespace {
struct Entry { Id id; Point p; };
struct Node {
    Box box;
    std::size_t size{};
    std::vector<Entry> entries;
    std::unique_ptr<std::array<Node, 4>> children;
    explicit Node(Box bounds) : box(bounds) {}
    Point midpoint() const {
        return {std::midpoint(box.x0,box.x1), std::midpoint(box.y0,box.y1)};
    }
    unsigned quadrant(Point p) const {
        auto m = midpoint();
        return unsigned(p.x >= m.x) + 2 * unsigned(p.y >= m.y);
    }
    void split() {
        auto m = midpoint();
        children = std::make_unique<std::array<Node,4>>(std::array{
            Node({box.x0,box.y0,m.x,m.y}), Node({m.x,box.y0,box.x1,m.y}),
            Node({box.x0,m.y,m.x,box.y1}), Node({m.x,m.y,box.x1,box.y1})});
    }
};

class Quad final : public Index {
    static std::size_t count_nodes(const Node& n) {
        std::size_t total = 1;
        if (n.children)
            for (const auto& child : *n.children) total += count_nodes(child);
        return total;
    }

    Node root_;
    std::size_t cap_;
    void add(Node& n, Entry e, int depth) {
        ++n.size;
        if (!n.children) {
            n.entries.push_back(e);
            if (n.entries.size() <= cap_ || depth == 28) return;
            if (std::all_of(n.entries.begin(),n.entries.end(),
                            [&](Entry other) { return other.p == e.p; })) return;
            n.split();
            for (Entry item : n.entries)
                add((*n.children)[n.quadrant(item.p)], item, depth+1);
            n.entries.clear();
            return;
        }
        add((*n.children)[n.quadrant(e.p)],e,depth+1);
    }
    void collect(const Node& n, std::vector<Entry>& out) {
        if (n.children) {
            for (const auto& child : *n.children) collect(child,out);
        } else {
            out.insert(out.end(),n.entries.begin(),n.entries.end());
        }
    }
    void drop(Node& n, Entry e) {
        --n.size;
        if (n.children) {
            drop((*n.children)[n.quadrant(e.p)],e);
            if (n.size <= cap_/2) {
                for (const auto& child : *n.children) collect(child,n.entries);
                n.children.reset();
            }
        } else {
            auto it = std::find_if(n.entries.begin(),n.entries.end(),
                                   [&](Entry item) { return item.id == e.id; });
            if (it == n.entries.end()) throw std::logic_error("missing quadtree entry");
            *it = n.entries.back();
            n.entries.pop_back();
        }
    }
    void visit(const Node& n, Query q, Answer& a) const {
        ++*a.visits;
        if (n.size == 0 || !q.box.overlaps(n.box)) return;
        if (n.children) {
            for (const auto& child : *n.children) visit(child,q,a);
        } else {
            for (auto [id, p] : n.entries) refine(a,q,id,p);
        }
    }
    void insert(Id id, Point p) override {
        if (!root_.box.contains(p)) {
            Box b = root_.box;
            Node expanded({std::min(b.x0,p.x),std::min(b.y0,p.y),
                           std::max(b.x1,p.x),std::max(b.y1,p.y)});
            std::vector<Entry> existing;
            existing.reserve(root_.size);
            collect(root_,existing);
            for (Entry e : existing) add(expanded,e,0);
            root_ = std::move(expanded);
        }
        add(root_,{id,p},0);
    }
    void remove(Id id, Point p) override { drop(root_,{id,p}); }

public:
    Quad(int capacity, Box bounds) : root_(bounds), cap_(capacity) {}
    Answer query(Query q) const override {
        Answer a;
        a.visits = 0;
        visit(root_,q,a);
        return a;
    }
    std::size_t nodes() const override { return count_nodes(root_); }
};
}
std::unique_ptr<Index> make_quad(int capacity, Box bounds) {
    return std::make_unique<Quad>(capacity,bounds);
}
}
