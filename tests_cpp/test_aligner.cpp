// Synthetic oracle suite for WNetAligner<DIM> (aligner.hpp), standalone so it
// runs under every sanitizer and valgrind lane.  Instances are small: the
// slowest lanes (valgrind, MSan) must finish in a few minutes.
//
//   * every NetworkSimplex warm mode (including the repair_max_nodes gate)
//     and CostScaling give the same aligner cost, in 1-D and 2-D, for L1 and
//     LINF, single and asymmetric trash, one and two theoretical spectra;
//   * consensus pairs are one-to-one and index valid peaks;
//   * aligners built and solved on concurrent threads equal their
//     sequential results.
#include "wnetalign/aligner.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <thread>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const char* what, const char* where) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL %s: %s\n", where, what);
    }
}

template <size_t D>
Spectrum<D> peaks(std::mt19937_64& rng, size_t n, double extent) {
    std::uniform_real_distribution<double> pos(0.0, extent), inten(0.1, 5.0);
    std::vector<std::array<double, D>> p(n);
    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < D; ++d) p[i][d] = pos(rng);
        w[i] = inten(rng);
    }
    return Spectrum<D>(std::move(p), std::move(w));
}

// A warped, thinned, noisy copy: what an alignment target looks like.
template <size_t D>
Spectrum<D> warped(std::mt19937_64& rng, const Spectrum<D>& s, double shift) {
    std::normal_distribution<double> noise(0.0, shift / 4);
    std::uniform_real_distribution<double> keep(0.0, 1.0), scale(0.7, 1.3);
    std::vector<std::array<double, D>> p;
    std::vector<double> w;
    const auto& pos = s.get_positions();
    const auto in = s.intensities();
    for (size_t i = 0; i < pos.size(); ++i) {
        if (keep(rng) < 0.15) continue;
        auto q = pos[i];
        for (size_t d = 0; d < D; ++d) q[d] += shift + noise(rng);
        p.push_back(q);
        w.push_back(in[i] * scale(rng));
    }
    return Spectrum<D>(std::move(p), std::move(w));
}

// repair_max_nodes is newer than some wnet releases this builds against; the
// check must be on a dependent type to be a test rather than an error.
template <class Config>
void set_repair_gate(Config& c, long repair_max_nodes) {
    if constexpr (requires { c.repair_max_nodes; }) c.repair_max_nodes = repair_max_nodes;
}

NetworkSimplexConfig ns(NSWarmMode warm, long repair_max_nodes = -1) {
    NetworkSimplexConfig c;
    c.warm = warm;
    set_repair_gate(c, repair_max_nodes);
    return c;
}

std::vector<SolverConfig> configs() {
    std::vector<SolverConfig> out;
    for (auto warm : {NSWarmMode::None, NSWarmMode::Simple, NSWarmMode::Dual, NSWarmMode::Primal,
                      NSWarmMode::DualRatio, NSWarmMode::DualGreedy, NSWarmMode::LinkCut})
        out.push_back(ns(warm));
    out.push_back(ns(NSWarmMode::DualRatio, 0));
    out.push_back(ns(NSWarmMode::DualRatio, 20));
    out.push_back(CostScalingConfig{});
    return out;
}

std::vector<std::vector<double>> points(size_t k) {
    std::vector<std::vector<double>> pts;
    std::mt19937_64 rng(9);
    std::uniform_real_distribution<double> u(0.4, 1.6);
    for (int i = 0; i < 6; ++i) {
        std::vector<double> w(k);
        for (auto& x : w) x = u(rng);
        pts.push_back(w);
    }
    return pts;
}

template <size_t D>
void solvers_agree(int seed, DistanceMetric metric, bool asymmetric, size_t n_theo, const char* where) {
    std::mt19937_64 rng(seed);
    const double extent = D == 1 ? 40.0 : 20.0;
    auto emp = peaks<D>(rng, D == 1 ? 120 : 80, extent);
    std::vector<Spectrum<D>> theo_store;
    for (size_t t = 0; t < n_theo; ++t) theo_store.push_back(warped<D>(rng, emp, 0.05 * (t + 1)));
    std::vector<Spectrum<D>*> theos;
    for (auto& t : theo_store) theos.push_back(&t);
    const double max_d = 0.6;
    const auto pts = points(n_theo);

    auto run = [&](const SolverConfig& cfg) {
        WNetAligner<D> a = asymmetric
            ? WNetAligner<D>(emp, theos, metric, max_d, -1.0, 0, 0.5, 0.8, cfg)
            : WNetAligner<D>(emp, theos, metric, max_d, 0.6, 0, -1.0, -1.0, cfg);
        std::vector<double> costs;
        for (const auto& w : pts) {
            a.set_point(w);
            costs.push_back(a.total_cost());
        }
        // Consensus: one-to-one pairs of valid peaks.
        for (size_t t = 0; t < n_theo; ++t) {
            auto [e, th] = a.consensus_for_target(t);
            std::set<long long> se(e.begin(), e.end()), st(th.begin(), th.end());
            check(e.size() == th.size() && se.size() == e.size() && st.size() == th.size(),
                  "consensus is not one-to-one", where);
            for (auto x : e) check(x >= 0 && (size_t)x < emp.size(), "consensus empirical id out of range", where);
            for (auto x : th) check(x >= 0 && (size_t)x < theos[t]->size(), "consensus theoretical id out of range", where);
        }
        return costs;
    };
    const auto expect = run(ns(NSWarmMode::None));
    for (const auto& cfg : configs())
        check(run(cfg) == expect, "solver disagrees with cold NetworkSimplex", where);
}

void concurrent_aligners() {
    auto job = [](int seed, std::vector<double>& out) {
        std::mt19937_64 rng(seed);
        auto emp = peaks<2>(rng, 70, 20.0);
        auto t1 = warped<2>(rng, emp, 0.05);
        auto t2 = warped<2>(rng, emp, 0.1);
        std::vector<Spectrum<2>*> theos{&t1, &t2};
        WNetAligner<2> a(emp, theos, DistanceMetric::L1, 0.6, 0.6, 0, -1.0, -1.0, ns(NSWarmMode::DualRatio));
        for (const auto& w : points(2)) {
            a.set_point(w);
            out.push_back(a.total_cost());
        }
    };
    constexpr int N = 4;
    std::vector<std::vector<double>> seq(N), par(N);
    for (int i = 0; i < N; ++i) job(300 + i, seq[i]);
    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) threads.emplace_back(job, 300 + i, std::ref(par[i]));
    for (auto& t : threads) t.join();
    for (int i = 0; i < N; ++i)
        check(par[i] == seq[i], "threaded result differs from sequential", "concurrent_aligners");
}

}  // namespace

int main() {
    solvers_agree<1>(1, DistanceMetric::L1, false, 1, "1d L1 simple");
    solvers_agree<1>(2, DistanceMetric::L1, true, 2, "1d L1 asymmetric two targets");
    solvers_agree<2>(3, DistanceMetric::L1, false, 2, "2d L1 simple two targets");
    solvers_agree<2>(4, DistanceMetric::LINF, true, 1, "2d LINF asymmetric");
    concurrent_aligners();
    std::printf("%d checks, %d failures\n", checks, failures);
    std::printf("RESULT: %s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
