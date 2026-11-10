#pragma once
#include <chrono>
#include <cstdint>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <x86intrin.h>
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

// Cheap per-event timer for the profiling path. QueryPerformanceCounter costs
// about 50 ns per call here, which inflated region update costs by roughly
// 90 ns per update; the time-stamp counter costs a few ns instead. The
// conversion factor is calibrated once against the steady clock.
struct Tick {
    static double ns_per_tick() {
        static const double value = calibrate();
        return value;
    }
    static std::int64_t now() { return static_cast<std::int64_t>(__rdtsc()); }
    static std::int64_t ns(std::int64_t ticks) {
        return static_cast<std::int64_t>(static_cast<double>(ticks) * ns_per_tick());
    }
#ifdef _WIN32
    static Ns duration(std::int64_t ticks) { return Ns(ns(ticks)); }
#else
    static Ns duration(std::int64_t ticks) { return Ns(ns(ticks)); }
#endif
private:
    static double calibrate() {
        // Two rounds of a 2 ms steady-clock interval.
        double best = 0;
        for (int round = 0; round < 3; ++round) {
            const auto wall0 = Clock::now();
            const auto tsc0 = now();
            while ((Clock::now() - wall0).count() < 2000000) {}
            const auto tsc1 = now();
            const auto wall1 = Clock::now();
            const double elapsed = static_cast<double>((wall1 - wall0).count());
            const double ticks = static_cast<double>(tsc1 - tsc0);
            if (ticks > 0) best = elapsed / ticks;
        }
        if (!(best > 0)) throw std::runtime_error("TSC calibration failed");
        return best;
    }
};
}
