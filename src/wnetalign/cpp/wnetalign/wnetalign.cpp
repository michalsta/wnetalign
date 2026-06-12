#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include "aligner.hpp"

// Spectrum<DIM> is now an alias for wnet's VectorDistribution<DIM,double,double>
// (bound by the wnet module as CVectorDistributionFloat{DIM}); it is not
// re-registered here.  The aligner accepts those objects directly (passed from
// Python as a Distribution/Spectrum's `vecdist`).

#define EXPOSE_ALIGNER(DIM) \
    using WNetAligner_##DIM = WNetAligner<DIM>; \
    nb::class_<WNetAligner_##DIM>(m, "WNetAligner" #DIM) \
        .def("__init__", [](WNetAligner_##DIM* self, \
                const Spectrum<DIM>& empirical, \
                nb::list theoretical_list, \
                int distance, \
                double max_distance, \
                double trash_cost, \
                double scale_factor, \
                double experimental_trash_cost, \
                double theoretical_trash_cost, \
                SolverConfig config) { \
            std::vector<Spectrum<DIM>*> theoretical; \
            theoretical.reserve(nb::len(theoretical_list)); \
            for (auto item : theoretical_list) { \
                theoretical.push_back(nb::cast<Spectrum<DIM>*>(item)); \
            } \
            new (self) WNetAligner_##DIM(empirical, theoretical, \
                static_cast<DistanceMetric>(distance), \
                max_distance, trash_cost, scale_factor, \
                experimental_trash_cost, theoretical_trash_cost, \
                config); \
        }, nb::arg("empirical"), nb::arg("theoretical"), \
           nb::arg("distance"), nb::arg("max_distance"), \
           nb::arg("trash_cost") = -1.0, nb::arg("scale_factor") = 0.0, \
           nb::arg("experimental_trash_cost") = -1.0, \
           nb::arg("theoretical_trash_cost") = -1.0, \
           nb::arg("solver") = NetworkSimplexConfig{}) \
        .def("set_point", &WNetAligner_##DIM::set_point) \
        .def("total_cost", &WNetAligner_##DIM::total_cost) \
        .def("scale_factor", &WNetAligner_##DIM::scale_factor) \
        .def("no_theoretical_spectra", &WNetAligner_##DIM::no_theoretical_spectra) \
        .def("flows_for_target", [](const WNetAligner_##DIM& self, size_t target_id) { \
            auto [emp, theo, flows] = self.flows_for_target(target_id); \
            return nb::make_tuple( \
                vector_to_numpy<LEMON_INDEX>(emp), \
                vector_to_numpy<LEMON_INDEX>(theo), \
                vector_to_numpy<int64_t>(flows)); \
        }, nb::rv_policy::move) \
        .def("no_subgraphs", &WNetAligner_##DIM::no_subgraphs) \
        .def("count_empirical_nodes", &WNetAligner_##DIM::count_empirical_nodes) \
        .def("count_theoretical_nodes", &WNetAligner_##DIM::count_theoretical_nodes) \
        .def("matching_density", &WNetAligner_##DIM::matching_density) \
        .def("consensus_for_target", [](const WNetAligner_##DIM& self, size_t target_id) { \
            auto [emp, theo] = self.consensus_for_target(target_id); \
            return nb::make_tuple( \
                vector_to_numpy<LEMON_INDEX>(emp), \
                vector_to_numpy<LEMON_INDEX>(theo)); \
        }, nb::rv_policy::move) \
        .def("__str__", &WNetAligner_##DIM::to_string);

NB_MODULE(wnetalign_cpp, m) {
    m.doc() = "WNetAlign C++ implementation module";
    m.def("wnetalign_cpp_hello", []() {
        std::cout << "Hello from WNetAlign (C++)!" << std::endl;
    }, "A simple hello world function for the WNetAlign (C++) extension");

    EXPOSE_ALIGNER(1)
    EXPOSE_ALIGNER(2)
    EXPOSE_ALIGNER(3)
    EXPOSE_ALIGNER(4)
    EXPOSE_ALIGNER(5)
    EXPOSE_ALIGNER(6)
    EXPOSE_ALIGNER(7)
    EXPOSE_ALIGNER(8)
    EXPOSE_ALIGNER(9)
    EXPOSE_ALIGNER(10)
    EXPOSE_ALIGNER(11)
    EXPOSE_ALIGNER(12)
    EXPOSE_ALIGNER(13)
    EXPOSE_ALIGNER(14)
    EXPOSE_ALIGNER(15)
    EXPOSE_ALIGNER(16)
    EXPOSE_ALIGNER(17)
    EXPOSE_ALIGNER(18)
    EXPOSE_ALIGNER(19)
    EXPOSE_ALIGNER(20)
}