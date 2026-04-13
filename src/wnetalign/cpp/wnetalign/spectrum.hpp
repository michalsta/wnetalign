#ifndef WNETALIGN_SPECTRUM_HPP
#define WNETALIGN_SPECTRUM_HPP

#include <array>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstring>

#ifdef INCLUDE_NANOBIND_STUFF
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <wnet/py_support.hpp>
#include <wnet/misc.hpp>
namespace nb = nanobind;
#endif

#include <wnet/distribution.hpp>

template<size_t DIM>
class Spectrum {
private:
    std::vector<std::array<double, DIM>> positions;
    std::vector<double> intensities;

public:
    Spectrum(const std::vector<std::array<double, DIM>> &positions, const std::vector<double> &intensities)
        : positions(positions), intensities(intensities) {}

#ifdef INCLUDE_NANOBIND_STUFF
    Spectrum(const nb::ndarray<double, nb::shape<DIM, -1>> &positions_arg, const nb::ndarray<double, nb::shape<-1>> &intensities_arg)
        : positions(numpy_to_vector_of_arrays<double, DIM>(positions_arg)),
          intensities(numpy_to_vector<double>(intensities_arg)) {}

    Spectrum(const nb::ndarray<double, nb::shape<-1, -1>> &positions_arg, const nb::ndarray<double, nb::shape<-1>> &intensities_arg)
        : positions(numpy_to_vector_of_arrays<double, DIM>(positions_arg)),
          intensities(numpy_to_vector<double>(intensities_arg)) {}
#endif

    double sum_intensities() const {
        return std::accumulate(intensities.begin(), intensities.end(), 0.0);
    }

    Spectrum scaled(double factor) const {
        std::vector<double> scaled_intensities;
        scaled_intensities.reserve(intensities.size());
        for (double intensity : intensities) {
            scaled_intensities.push_back(intensity * factor);
        }
        return Spectrum(positions, scaled_intensities);
    }

    Spectrum normalized() const {
        double total = sum_intensities();
        if (total == 0) {
            throw std::runtime_error("Cannot normalize a spectrum with total intensity of 0.");
        }
        std::vector<double> normalized_intensities;
        normalized_intensities.reserve(intensities.size());
        for (double intensity : intensities) {
            normalized_intensities.push_back(intensity / total);
        }
        return Spectrum(positions, normalized_intensities);
    }

    VectorDistribution<DIM, double, int64_t> to_vector_distribution() const {
        std::vector<int64_t> int_intensities;
        int_intensities.reserve(intensities.size());
        for (double intensity : intensities) {
            int_intensities.push_back(static_cast<int64_t>(intensity));
        }
        return VectorDistribution<DIM, double, int64_t>(
            std::vector<std::array<double, DIM>>(positions),
            std::move(int_intensities)
        );
    }

    size_t size() const {
        return intensities.size();
    }

    const std::vector<std::array<double, DIM>> &get_positions() const {
        return positions;
    }

    const std::vector<double> &get_intensities() const {
        return intensities;
    }

#ifdef INCLUDE_NANOBIND_STUFF
    nb::ndarray<nb::numpy, double, nb::shape<DIM, -1>> py_get_positions() const {
        return vector_of_arrays_to_numpy<double, DIM>(positions);
    }

    nb::ndarray<nb::numpy, double, nb::shape<-1>> py_get_intensities() const {
        double* data = new double[intensities.size()];
        std::memcpy(data, intensities.data(), intensities.size() * sizeof(double));
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double, nb::shape<-1>>(
            data,
            { intensities.size() },
            owner
        );
    }
#endif
};

#endif // WNETALIGN_SPECTRUM_HPP
