#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <filesystem>

#include "wnetalign/spectrum.hpp"
#include "wnetalign/aligner.hpp"

struct CsvData {
    std::vector<double> mz;
    std::vector<double> rt;
    std::vector<double> intensity;
};

CsvData read_csv(const std::string& path) {
    CsvData data;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::string line;
    // skip header
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        data.mz.push_back(std::stod(token));
        std::getline(ss, token, ',');
        data.rt.push_back(std::stod(token));
        std::getline(ss, token, ',');
        data.intensity.push_back(std::stod(token));
    }
    return data;
}

Spectrum<2> make_spectrum(const CsvData& data) {
    std::vector<std::array<double, 2>> positions;
    positions.reserve(data.mz.size());
    for (size_t i = 0; i < data.mz.size(); ++i) {
        positions.push_back({data.mz[i], data.rt[i]});
    }
    return Spectrum<2>(positions, data.intensity);
}

Spectrum<2> scale_mz_values(const Spectrum<2>& spectrum, double scale_factor) {
    auto positions = spectrum.get_positions();
    for (auto& pos : positions) {
        pos[0] *= scale_factor;
    }
    return Spectrum<2>(positions, spectrum.get_intensities());
}

int main(int argc, char* argv[]) {
    // Locate data directory relative to executable or via argument
    std::string datadir;
    if (argc > 1) {
        datadir = argv[1];
    } else {
        // Try relative to source tree
        namespace fs = std::filesystem;
        fs::path exe_path = fs::current_path();
        fs::path candidate = exe_path / "tutorials" / "lcms" / "data";
        if (fs::exists(candidate)) {
            datadir = candidate.string();
        } else {
            std::cerr << "Usage: " << argv[0] << " <path_to_lcms_data_dir>" << std::endl;
            return 1;
        }
    }

    std::string file1 = datadir + "/100825O2c1_MT-AU-0044-2010-08-15_038.csv";
    std::string file2 = datadir + "/100820O2c1_MT-AU-0044-2010-08-1_030.csv";

    std::cout << "Reading CSV files..." << std::endl;
    CsvData csv1 = read_csv(file1);
    CsvData csv2 = read_csv(file2);
    std::cout << "  S1: " << csv1.mz.size() << " peaks" << std::endl;
    std::cout << "  S2: " << csv2.mz.size() << " peaks" << std::endl;

    Spectrum<2> S1 = make_spectrum(csv1);
    Spectrum<2> S2 = make_spectrum(csv2);

    double max_mz_shift = 0.005;
    double max_rt_shift = 800;
    double scale_mz = max_rt_shift / max_mz_shift;
    int64_t mtd = static_cast<int64_t>(std::round(max_rt_shift));

    // Scale m/z dimension
    S1 = scale_mz_values(S1, scale_mz);
    S2 = scale_mz_values(S2, scale_mz);

    // Normalize
    S1 = S1.normalized();
    S2 = S2.normalized();

    std::cout << "Building aligner (LINF, max_dist=" << mtd << ", trash_cost=" << mtd << ")..." << std::endl;

    std::vector<Spectrum<2>*> theoretical = {&S2};
    WNetAligner<2> aligner(S1, theoretical, DistanceMetric::LINF, mtd, mtd);

    std::cout << "Solving..." << std::endl;
    aligner.set_point({1.0});

    std::cout << "  Scale factor: " << aligner.scale_factor() << std::endl;

    auto [cons_emp, cons_theo] = aligner.consensus_for_target(0);
    size_t consensus_count = cons_emp.size();

    std::cout << "  Consensus features: " << consensus_count << std::endl;
    assert(consensus_count > 25000);
    std::cout << "PASSED (consensus_count=" << consensus_count << " > 25000)" << std::endl;

    return 0;
}
