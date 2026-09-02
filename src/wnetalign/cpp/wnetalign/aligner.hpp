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
#include <wnet/scaling.hpp>

template<size_t DIM>
class WNetAligner {
    // Positions and intensities are both real doubles all the way to the
    // network, which quantizes each onto its OWN integer grid: intensities to
    // integer supplies via set_intensity_scale, distances to integer edge
    // costs via set_cost_scaling.  Nothing is pre-scaled here.
    //
    // This replaced a single tied `scale_factor` that was multiplied into the
    // positions (to buy distance resolution) and simultaneously handed to
    // set_intensity_scale (to buy intensity resolution).  One number cannot
    // serve both: sized as a joint overflow cap sqrt(2^60 / (max_sum *
    // max_cost)), it was a compromise neither side chose.  Normalized spectra
    // — what every published pipeline here feeds in — pushed intensities to
    // ~1e-8, where the tied factor truncated the faint tail to zero supply and
    // the peaks silently disappeared from the alignment.  Pre-multiplying the
    // positions was also lossy in its own right: it inflated coordinates to
    // ~1e15, where differencing two nearby peaks loses most of its significant
    // digits before the distance is ever computed.
    using VecDist = VectorDistribution<DIM, double, double>;

    double intensity_scale_;
    size_t no_theoretical_;
    WassersteinNetwork<int64_t, double> network_;

    // --- Budget allocation policy -------------------------------------------
    //
    // The int64 cost accumulator bounds the PRODUCT of the two scales, not
    // either one; untying them is exactly the freedom to split that budget
    // deliberately instead of down the middle.  We spend it as follows.
    //
    // Intensities are the conserved quantity: a peak quantized to zero supply
    // is *gone*, an unrecoverable failure.  So the intensity scale is sized to
    // give even the faintest peak in any spectrum enough integer units to be
    // matched with a bounded relative error.
    static constexpr double FAINT_PEAK_TARGET_UNITS = 1000.0;  // <= 0.1% on the faintest
    //
    // Distances degrade gracefully by comparison — a coarse cost grid makes the
    // solver indifferent between near-equal matches rather than deleting
    // anything — so when the budget cannot pay for both, intensity wins.  The
    // cost grid asks for COST_GRID_STEPS_WANT distinguishable steps across the
    // cost range and will settle for COST_GRID_STEPS_MIN; below that it simply
    // takes whatever the intensity scale leaves, and the caller can see the
    // result in cost_scale().
    // 1e6 steps is six significant digits on the largest cost in the network —
    // far finer than any real position measurement these spectra carry.
    static constexpr double COST_GRID_STEPS_WANT = 1.0e6;
    static constexpr double COST_GRID_STEPS_MIN  = 1.0e3;

    // Trash costs that are actually in force, in the network's cost units.
    static std::vector<double> active_trash_costs(
        double trash_cost, double experimental_trash_cost, double theoretical_trash_cost)
    {
        const bool asymmetric =
            experimental_trash_cost >= 0 || theoretical_trash_cost >= 0;
        std::vector<double> costs;
        if (asymmetric) {
            const double eff_exp =
                experimental_trash_cost >= 0 ? experimental_trash_cost : trash_cost;
            const double eff_theo =
                theoretical_trash_cost >= 0 ? theoretical_trash_cost : trash_cost;
            if (eff_exp  >= 0) costs.push_back(eff_exp);
            if (eff_theo >= 0) costs.push_back(eff_theo);
        } else if (trash_cost >= 0) {
            costs.push_back(trash_cost);
        }
        return costs;
    }

    static double compute_intensity_scale(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        DistanceMetric metric,
        double max_distance,
        const std::vector<double>& trash_costs)
    {
        std::vector<const VecDist*> theo_ptrs;
        theo_ptrs.reserve(theoretical.size());
        for (const auto* t : theoretical) theo_ptrs.push_back(t);

        // GenericScaler with quantile 1.0 anchors on the faintest positive peak
        // across all spectra rather than a p95 "signal band": for alignment the
        // faint tail is data, not noise to be rounded away.  The rounding-loss
        // guard is off because this policy exists precisely to avoid that loss —
        // and the caller gets the honest number back either way.
        GenericScaler<DIM> scaler(
            empirical, theo_ptrs, metric, max_distance, trash_costs,
            /*p95_frac=*/1.0,
            /*rounding_tol=*/1.0 / FAINT_PEAK_TARGET_UNITS,
            /*max_dropped_frac=*/1.0);
        // Never go BELOW unit scale.  The target is a floor on resolution, not
        // a ceiling: a spectrum whose intensities are uniformly large already
        // has all the resolution it needs, and shrinking it to hit the target
        // from above would throw away mass that costs nothing to keep.
        double sf = std::max(1.0, scaler.sf_intensity());

        // Reserve the cost grid's share.  The network picks its cost scale as
        // ~2^62 / (max_cost * 4 * total_flow * sf_intensity) (the 4x being its
        // default flow-budget headroom), so bounding sf here is what keeps that
        // from collapsing when one dust peak drags the intensity target up.
        // Asking for a fixed number of steps across the cost range cancels
        // max_cost out of the reservation entirely.
        double total_flow = empirical.sum_intensities();
        for (const auto* t : theoretical) total_flow += t->sum_intensities();
        if (total_flow > 0.0) {
            constexpr double ACCUMULATOR_TARGET = 4611686018427387904.0;  // 2^62
            for (double steps : {COST_GRID_STEPS_WANT, COST_GRID_STEPS_MIN}) {
                const double sf_cap =
                    ACCUMULATOR_TARGET / (4.0 * total_flow * steps);
                if (sf_cap >= 1.0) {
                    if (sf > sf_cap) sf = sf_cap;
                    break;  // the most generous affordable reservation wins
                }
            }
        }
        return sf;
    }

    static WassersteinNetwork<int64_t, double> build_network(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        DistanceMetric distance,
        double max_distance,
        double trash_cost,
        double intensity_scale,
        double experimental_trash_cost,
        double theoretical_trash_cost,
        SolverConfig config)
    {
        if (theoretical.empty())
            throw std::invalid_argument("Need at least one theoretical spectrum");
        if (empirical.size() == 0)
            throw std::invalid_argument("Empirical spectrum is empty");

        // Real positions, real max_distance: the network quantizes distances to
        // integer costs itself (set_cost_scaling below), so there is nothing to
        // pre-scale and no copy of the spectra to make.
        std::vector<Spectrum<DIM>*> theo_ptrs(theoretical.begin(), theoretical.end());
        auto network = WassersteinNetworkFactory<int64_t>::create(
            &empirical,
            theo_ptrs,
            distance,
            max_distance
        );
        const bool asymmetric =
            experimental_trash_cost >= 0 || theoretical_trash_cost >= 0;
        if (asymmetric) {
            const double eff_exp =
                experimental_trash_cost >= 0 ? experimental_trash_cost : trash_cost;
            const double eff_theo =
                theoretical_trash_cost >= 0 ? theoretical_trash_cost : trash_cost;
            if (eff_exp >= 0)  network.add_experimental_trash(eff_exp);
            if (eff_theo >= 0) network.add_theoretical_trash(eff_theo);
        } else {
            network.add_simple_trash(trash_cost);
        }
        // The two grids, chosen independently:
        //   * intensities -> integer supplies at intensity_scale, quantized once
        //     inside the network AFTER the point weight is applied;
        //   * distances   -> integer edge costs at a scale the network sizes
        //     against whatever int64 budget the intensity scale left it (0 =
        //     auto).  Without this opt-in, p == 1 keeps the legacy scale of 1,
        //     which truncates every real distance to a whole number.
        network.set_intensity_scale(intensity_scale);
        network.set_cost_scaling(0);
        network.build(config);
        return network;
    }

    static double resolve_intensity_scale(
        const Spectrum<DIM>& empirical,
        const std::vector<Spectrum<DIM>*>& theoretical,
        DistanceMetric metric,
        double max_distance,
        double trash_cost,
        double scale_factor,
        double experimental_trash_cost,
        double theoretical_trash_cost)
    {
        const bool asymmetric =
            experimental_trash_cost >= 0 || theoretical_trash_cost >= 0;
        if (!asymmetric && trash_cost < 0)
            throw std::invalid_argument("At least one of trash_cost, experimental_trash_cost, or theoretical_trash_cost must be provided.");
        // An explicit scale_factor now sets the INTENSITY scale only; cost
        // quantization is always the network's job.  It used to set both at
        // once, so a caller who passed one to buy distance resolution should
        // drop the argument and let the automatic policy size both grids.
        if (scale_factor > 0)
            return scale_factor;
        return compute_intensity_scale(
            empirical, theoretical, metric, max_distance,
            active_trash_costs(trash_cost, experimental_trash_cost, theoretical_trash_cost));
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
    ) : intensity_scale_(resolve_intensity_scale(empirical, theoretical, distance, max_distance, trash_cost, scale_factor, experimental_trash_cost, theoretical_trash_cost)),
        no_theoretical_(theoretical.size()),
        network_(build_network(empirical, theoretical, distance, max_distance, trash_cost, intensity_scale_, experimental_trash_cost, theoretical_trash_cost, config))
    {}

    void set_point(const std::vector<double>& point) {
        network_.solve(point);
    }

    double total_cost() const {
        // The scaled cost carries one factor of each grid: edge costs are in
        // cost-scale units, the flow through them in intensity-scale units.
        return network_.total_cost()
             / (static_cast<double>(network_.scale_factor())
                * network_.intensity_scale_factor());
    }

    /// Factor real intensities are multiplied by before truncation to integer
    /// supplies.  Flows come back in these units.
    double intensity_scale() const {
        return intensity_scale_;
    }

    /// Integer cost grid the network chose for the real distances.
    int64_t cost_scale() const {
        return network_.scale_factor();
    }

    /// Back-compatible alias for the intensity scale, which is the factor
    /// flows_for_target() results must be divided by.  It is no longer also
    /// the position/cost scale — those are separate grids now.
    double scale_factor() const {
        return intensity_scale_;
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
        // Sort indices by flow descending; stable so that ties between equal
        // flows resolve by original index order, deterministically across
        // platforms and standard library implementations.
        std::vector<size_t> order(emp_ids.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&flows](size_t a, size_t b) {
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
