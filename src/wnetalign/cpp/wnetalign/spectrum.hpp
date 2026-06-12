#ifndef WNETALIGN_SPECTRUM_HPP
#define WNETALIGN_SPECTRUM_HPP

#include <wnet/distribution.hpp>

// wnetalign's Spectrum<DIM> used to be a standalone class duplicating wnet's
// double-intensity VectorDistribution.  It is now a compatibility alias for
// that type: positions are double, intensities are real (double) values that
// the aligner's WassersteinNetwork quantizes to integer supplies internally
// (via set_intensity_scale), instead of being pre-truncated to int64 here.
//
// The alias keeps existing C++ references to `Spectrum<DIM>` compiling.  On the
// Python side it means a wnetalign Spectrum is backed by the same C++ object as
// any wnet Distribution (CVectorDistributionFloat{DIM}); there is no separate
// `wnetalign_cpp.Spectrum{DIM}` binding anymore.
template<size_t DIM>
using Spectrum = VectorDistribution<DIM, double, double>;

#endif // WNETALIGN_SPECTRUM_HPP
