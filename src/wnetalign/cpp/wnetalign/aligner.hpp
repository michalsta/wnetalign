#ifndef WNETALIGN_ALIGNER_HPP
#define WNETALIGN_ALIGNER_HPP

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <tuple>
#include <unordered_set>

#include "spectrum.hpp"
#include <wnet/decompositable_graph.hpp>
#include <wnet/distances.hpp>
#include <wnet/distribution.hpp>

template<size_t DIM>
class WNetAligner {
    using VecDist = VectorDistribution<DIM, double, int64_t>;

    double scale_factor_;
    size_t no_theoretical_;
    WassersteinNetwork<int64_t, int64_t> network_;

    static VecDist scale_spectrum(const Spectrum<DIM>& spectrum, double scale_factor) {
        auto positions = spectrum.get_positions();
        for (auto& pos : positions) {
            for (auto& p : pos) {
                p *= scale_factor;
            }
        }
        const auto& intensities = spectrum.get_intensities();
        std::vector<int64_t> int_intensities;
        int_intensities.reserve(intensities.size());
        for (double intensity : intensities) {
            int_intensities.push_back(static_cast<int64_t>(intensity * scale_factor));
        }
        return VecDist(std::move(positions), std::move(int_intensities));
    }

    static double compute_scale_factor(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        double trash_cost,
        double experimental_trash_cost,
        double theoretical_trash_cost)
    {
        bool asymmetric = experimental_trash_cost >= 0 || theoretical_trash_cost >= 0;
        double min_cost;
        if (asymmetric) {
            double eff_exp  = experimental_trash_cost >= 0 ? experimental_trash_cost : trash_cost;
            double eff_theo = theoretical_trash_cost  >= 0 ? theoretical_trash_cost  : trash_cost;
            if (eff_exp >= 0 && eff_theo >= 0)
                min_cost = std::min(eff_exp, eff_theo);
            else if (eff_exp >= 0)
                min_cost = eff_exp;
            else
                min_cost = eff_theo;
        } else {
            min_cost = trash_cost;
        }
        if (min_cost <= 0)
            throw std::invalid_argument("resolved trash cost must be positive, got " + std::to_string(min_cost));
        constexpr int64_t ALMOST_MAXINT = 1LL << 60;
        double empirical_sum = empirical.sum_intensities();
        double theoretical_sum = 0;
        for (const auto* t : theoretical) {
            theoretical_sum += t->sum_intensities();
        }
        double max_sum = std::max(empirical_sum, theoretical_sum);
        if (max_sum <= 0)
            throw std::invalid_argument("max intensity sum must be positive (got " + std::to_string(max_sum) + "). Are all spectra empty?");
        double product = max_sum * min_cost;
        if (std::isinf(product))
            throw std::overflow_error("max_sum * min_trash_cost overflows double (" + std::to_string(max_sum) + " * " + std::to_string(min_cost) + ")");
        return std::sqrt(static_cast<double>(ALMOST_MAXINT) / product);
    }

    static WassersteinNetwork<int64_t, int64_t> build_network(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        DistanceMetric distance,
        double max_distance,
        double trash_cost,
        double scale_factor,
        double experimental_trash_cost,
        double theoretical_trash_cost,
        SolverConfig config)
    {
        if (theoretical.empty())
            throw std::invalid_argument("Need at least one theoretical spectrum");
        if (empirical.size() == 0)
            throw std::invalid_argument("Empirical spectrum is empty");

        VecDist emp_dist = scale_spectrum(empirical, scale_factor);

        std::vector<VecDist> theo_dists;
        theo_dists.reserve(theoretical.size());
        for (const auto* t : theoretical) {
            theo_dists.push_back(scale_spectrum(*t, scale_factor));
        }

        std::vector<VecDist*> theo_ptrs;
        theo_ptrs.reserve(theo_dists.size());
        for (auto& d : theo_dists) {
            theo_ptrs.push_back(&d);
        }

        auto network = WassersteinNetworkFactory<int64_t>::create(
            &emp_dist,
            theo_ptrs,
            distance,
            static_cast<int64_t>(max_distance * scale_factor)
        );
        bool asymmetric = experimental_trash_cost >= 0 || theoretical_trash_cost >= 0;
        if (asymmetric) {
            double eff_exp  = experimental_trash_cost >= 0 ? experimental_trash_cost : trash_cost;
            double eff_theo = theoretical_trash_cost  >= 0 ? theoretical_trash_cost  : trash_cost;
            if (eff_exp >= 0)
                network.add_experimental_trash(static_cast<int64_t>(eff_exp * scale_factor));
            if (eff_theo >= 0)
                network.add_theoretical_trash(static_cast<int64_t>(eff_theo * scale_factor));
        } else {
            network.add_simple_trash(static_cast<int64_t>(trash_cost * scale_factor));
        }
        network.build(config);
        return network;
    }

    static double resolve_scale_factor(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        double trash_cost,
        double scale_factor,
        double experimental_trash_cost,
        double theoretical_trash_cost)
    {
        bool asymmetric = experimental_trash_cost >= 0 || theoretical_trash_cost >= 0;
        if (!asymmetric && trash_cost < 0)
            throw std::invalid_argument("At least one of trash_cost, experimental_trash_cost, or theoretical_trash_cost must be provided.");
        if (scale_factor > 0)
            return scale_factor;
        return compute_scale_factor(empirical, theoretical, trash_cost, experimental_trash_cost, theoretical_trash_cost);
    }

public:
    WNetAligner(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        DistanceMetric distance,
        double max_distance,
        double trash_cost = -1.0,
        double scale_factor = 0,
        double experimental_trash_cost = -1.0,
        double theoretical_trash_cost  = -1.0,
        SolverConfig config = NetworkSimplexConfig{}
    ) : scale_factor_(resolve_scale_factor(empirical, theoretical, trash_cost, scale_factor, experimental_trash_cost, theoretical_trash_cost)),
        no_theoretical_(theoretical.size()),
        network_(build_network(empirical, theoretical, distance, max_distance, trash_cost, scale_factor_, experimental_trash_cost, theoretical_trash_cost, config))
    {}

    void set_point(const std::vector<double>& point) {
        network_.solve(point);
    }

    double total_cost() const {
        return network_.total_cost() / (scale_factor_ * scale_factor_);
    }

    double scale_factor() const {
        return scale_factor_;
    }

    size_t no_theoretical_spectra() const {
        return no_theoretical_;
    }

    std::tuple<std::vector<LEMON_INDEX>, std::vector<LEMON_INDEX>, std::vector<int64_t>>
    flows_for_target(size_t target_id) const {
        return network_.flows_for_target(target_id);
    }

    size_t no_subgraphs() const {
        return network_.no_subgraphs();
    }

    size_t count_empirical_nodes() const {
        return network_.template count_nodes_of_type<EmpiricalNode<int64_t>>();
    }

    size_t count_theoretical_nodes() const {
        return network_.template count_nodes_of_type<TheoreticalNode<int64_t>>();
    }

    double matching_density() const {
        return network_.matching_density();
    }

    const WassersteinNetworkSubgraph<int64_t, int64_t>& get_subgraph(size_t idx) const {
        return network_.get_subgraph(idx);
    }

    std::string to_string() const {
        return network_.to_string();
    }

    /// Greedy consensus matching for a given target spectrum.
    /// Returns (empirical_ids, theoretical_ids) of the 1-to-1 consensus pairs,
    /// selected greedily by descending flow magnitude.
    std::pair<std::vector<LEMON_INDEX>, std::vector<LEMON_INDEX>>
    consensus_for_target(size_t target_id) const {
        auto flow_result = network_.flows_for_target(target_id);
        auto& emp_ids = std::get<0>(flow_result);
        auto& theo_ids = std::get<1>(flow_result);
        auto& flows = std::get<2>(flow_result);
        // Sort indices by flow descending
        std::vector<size_t> order(emp_ids.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&flows](size_t a, size_t b) {
            return flows[a] > flows[b];
        });
        std::unordered_set<LEMON_INDEX> used_emp, used_theo;
        std::vector<LEMON_INDEX> cons_emp, cons_theo;
        for (size_t idx : order) {
            LEMON_INDEX e = emp_ids[idx];
            LEMON_INDEX t = theo_ids[idx];
            if (used_emp.find(e) == used_emp.end() && used_theo.find(t) == used_theo.end()) {
                used_emp.insert(e);
                used_theo.insert(t);
                cons_emp.push_back(e);
                cons_theo.push_back(t);
            }
        }
        return {cons_emp, cons_theo};
    }
};

#endif // WNETALIGN_ALIGNER_HPP
