#ifndef RAND_TEMP_MASS_H
#define RAND_TEMP_MASS_H

#include <tuple>
#include <random>
#include <cmath>
#include <numbers>

// #include mateusz file


# define M_PI   3.141592653589793238462643383279502884L // shouldn't be needed for c++20 but my M_PI won't work :/

// tank constants
static const double R_tank = 2.19 / 2.0;                        // m
static const double L_tank = 5.5;                               // m
static const double V_max  = M_PI * R_tank * R_tank * L_tank;   // m^3

// mateusz change this however you like to pass desnity float 
double getDensityFromGUI();

// generateRandomMassTempRho()
inline std::tuple<double, double, double>
generateRandomMassTempRho(double min_fill = 0.7) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    //  Collect density from gui
    double rho = getDensityFromGUI();

    // min tank fill is 80%, so 70 for demonstration
    std::uniform_real_distribution<double>
    massDist(min_fill * rho * V_max, rho * V_max);
    
    double mass = massDist(gen);

    // rnad temp in operation range
    std::uniform_real_distribution<double> tempDist(-40.0, 60.0);
    double temp = tempDist(gen);

    return {mass, temp, rho};
}

#endif // RAND_TEMP_MASS_H