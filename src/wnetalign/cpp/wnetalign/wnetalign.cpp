#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include "aligner.hpp"

#define EXPOSE_SPECTRUM(DIM) \
    using Spectrum_##DIM = Spectrum<DIM>; \
    nb::class_<Spectrum_##DIM>(m, "Spectrum" #DIM) \
        .def(nb::init<const nb::ndarray<double, nb::shape<DIM, -1>> &, const nb::ndarray<double, nb::shape<-1>> &>()) \
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, -1>> &, const nb::ndarray<double, nb::shape<-1>> &>()) \
        .def("sum_intensities", &Spectrum_##DIM::sum_intensities) \
        .def("scaled", &Spectrum_##DIM::scaled) \
        .def("normalized", &Spectrum_##DIM::normalized) \
        .def("to_vector_distribution", &Spectrum_##DIM::to_vector_distribution) \
        .def("size", &Spectrum_##DIM::size) \
        .def("__len__", &Spectrum_##DIM::size) \
        .def("py_get_positions", &Spectrum_##DIM::py_get_positions) \
        .def("py_get_intensities", &Spectrum_##DIM::py_get_intensities);

#define EXPOSE_ALIGNER(DIM) \
    using WNetAligner_##DIM = WNetAligner<DIM>; \
    nb::class_<WNetAligner_##DIM>(m, "WNetAligner" #DIM) \
        .def("__init__", [](WNetAligner_##DIM* self, \
                const Spectrum<DIM>& empirical, \
                nb::list theoretical_list, \
                int distance, \
                double max_distance, \
                double trash_cost, \
                double scale_factor) { \
            std::vector<Spectrum<DIM>*> theoretical; \
            theoretical.reserve(nb::len(theoretical_list)); \
            for (auto item : theoretical_list) { \
                theoretical.push_back(nb::cast<Spectrum<DIM>*>(item)); \
            } \
            new (self) WNetAligner_##DIM(empirical, theoretical, \
                static_cast<DistanceMetric>(distance), \
                max_distance, trash_cost, scale_factor); \
        }, nb::arg("empirical"), nb::arg("theoretical"), \
           nb::arg("distance"), nb::arg("max_distance"), \
           nb::arg("trash_cost"), nb::arg("scale_factor") = 0.0) \
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

    EXPOSE_SPECTRUM(1)
    EXPOSE_SPECTRUM(2)
    EXPOSE_SPECTRUM(3)
    EXPOSE_SPECTRUM(4)
    EXPOSE_SPECTRUM(5)
    EXPOSE_SPECTRUM(6)
    EXPOSE_SPECTRUM(7)
    EXPOSE_SPECTRUM(8)
    EXPOSE_SPECTRUM(9)
    EXPOSE_SPECTRUM(10)
    EXPOSE_SPECTRUM(11)
    EXPOSE_SPECTRUM(12)
    EXPOSE_SPECTRUM(13)
    EXPOSE_SPECTRUM(14)
    EXPOSE_SPECTRUM(15)
    EXPOSE_SPECTRUM(16)
    EXPOSE_SPECTRUM(17)
    EXPOSE_SPECTRUM(18)
    EXPOSE_SPECTRUM(19)
    EXPOSE_SPECTRUM(20)

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