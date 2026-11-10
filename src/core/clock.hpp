#pragma once
#include <chrono>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace spatial {
using Ns = std::chrono::nanoseconds;
#ifdef _WIN32
struct Clock {
    static Ns now() {
        LARGE_INTEGER counter{};
        QueryPerformanceCounter(&counter);
        static const auto epoch = counter.QuadPart;
        static const double ns_per_tick = [] {
            LARGE_INTEGER frequency{};
            if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0)
                throw std::runtime_error("QueryPerformanceCounter frequency unavailable");
            return 1e9 / frequency.QuadPart;
        }();
        return Ns(static_cast<Ns::rep>((counter.QuadPart - epoch) * ns_per_tick));
    }
};
#else
using Clock = std::chrono::steady_clock;
#endif
}
