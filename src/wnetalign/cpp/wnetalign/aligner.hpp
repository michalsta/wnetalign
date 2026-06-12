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
    // Real (double) intensities now: positions are scaled by scale_factor for
    // distance resolution, but intensities pass through untouched and are
    // quantized to integer supplies inside the network via set_intensity_scale.
    using VecDist = VectorDistribution<DIM, double, double>;

    double scale_factor_;
    size_t no_theoretical_;
    WassersteinNetwork<int64_t, double> network_;

    static VecDist scale_spectrum(const Spectrum<DIM>& spectrum, double scale_factor) {
        auto positions = spectrum.get_positions();
        for (auto& pos : positions) {
            for (auto& p : pos) {
                p *= scale_factor;
            }
        }
        // Intensities kept real; the network applies the intensity scale.
        std::vector<double> intensities = spectrum.get_intensities();
        return VecDist(std::move(positions), std::move(intensities));
    }

    static double compute_scale_factor(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        double max_distance,
        double trash_cost,
        double experimental_trash_cost,
        double theoretical_trash_cost)
    {
        // max_cost_per_unit_flow bounds the cost of any single unit of flow in
        // the scaled integer network: matching edges cost at most max_distance,
        // trash edges cost at most the relevant trash cost.  Using the maximum
        // (not minimum) keeps scale_factor small enough to avoid int64 overflow.
        bool asymmetric = experimental_trash_cost >= 0 || theoretical_trash_cost >= 0;
        double max_cost = max_distance;
        if (asymmetric) {
            double eff_exp  = experimental_trash_cost >= 0 ? experimental_trash_cost : trash_cost;
            double eff_theo = theoretical_trash_cost  >= 0 ? theoretical_trash_cost  : trash_cost;
            if (eff_exp  >= 0) max_cost = std::max(max_cost, eff_exp);
            if (eff_theo >= 0) max_cost = std::max(max_cost, eff_theo);
        } else {
            if (trash_cost >= 0) max_cost = std::max(max_cost, trash_cost);
        }
        if (max_cost <= 0)
            throw std::invalid_argument("max cost per unit flow must be positive, got " + std::to_string(max_cost));
        constexpr int64_t ALMOST_MAXINT = 1LL << 60;
        double empirical_sum = empirical.sum_intensities();
        double theoretical_sum = 0;
        for (const auto* t : theoretical) {
            theoretical_sum += t->sum_intensities();
        }
        double max_sum = std::max(empirical_sum, theoretical_sum);
        if (max_sum <= 0)
            throw std::invalid_argument("max intensity sum must be positive (got " + std::to_string(max_sum) + "). Are all spectra empty?");
        double product = max_sum * max_cost;
        if (std::isinf(product))
            throw std::overflow_error("max_sum * max_cost overflows double (" + std::to_string(max_sum) + " * " + std::to_string(max_cost) + ")");
        return std::sqrt(static_cast<double>(ALMOST_MAXINT) / product);
    }

    static WassersteinNetwork<int64_t, double> build_network(
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
        // Intensities were passed real; quantize them to integer supplies with
        // the same factor used for positions, so total_cost stays in
        // scale_factor**2 units (unchanged unscaling) but without the old
        // pre-truncation of intensities (single quantization, inside the
        // network, after the point weight is applied).
        network.set_intensity_scale(scale_factor);
        network.build(config);
        return network;
    }

    static double resolve_scale_factor(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        double max_distance,
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
        return compute_scale_factor(empirical, theoretical, max_distance, trash_cost, experimental_trash_cost, theoretical_trash_cost);
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
    ) : scale_factor_(resolve_scale_factor(empirical, theoretical, max_distance, trash_cost, scale_factor, experimental_trash_cost, theoretical_trash_cost)),
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
        return network_.template count_nodes_of_type<EmpiricalNode<double>>();
    }

    size_t count_theoretical_nodes() const {
        return network_.template count_nodes_of_type<TheoreticalNode<double>>();
    }

    double matching_density() const {
        return network_.matching_density();
    }

    const WassersteinNetworkSubgraph<int64_t, double>& get_subgraph(size_t idx) const {
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
