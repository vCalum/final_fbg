"""
Bridges mass and temperature generation in C++ with mass <-> strain conversion in physics

- random_mass_and_temp() calls C++ generateRandomMassTempRho() to obtain mass, temperature and density
- physics_strain_from_mass_density() and physics_mass_from_strain_density() represent physical relationship between mass and strain

Mateusz incorprates c++ extension, uncomment the import of generateRandomMassTempRho from rand_temp_mass
For python testing just use random_mass_temp.py
"""

import math


# from rand_temp_mass import generateRandomMassTempRho


import random_mass_temp
from random_mass_temp import generateRandomMassTempRho


# Tank geometry
R_tank = 2.19 / 2.0                     # radius in meters
L_tank = 5.5                            # length in meters
V_max = math.pi * R_tank**2 * L_tank    # maximum volume (m^3)

#BASE_EMPTY_STRAIN = 1.62e-5             # strain with empty tank
#MAX_STRAIN = 5.0e-5                     # strain at full tank

# beam & physics constants
g = 9.81                # m/s²
E_beam = 2e12           # Pa
I_beam = 0.2e-5         # m⁴
y_beam = 0.075          # m
L_beam = 0.36           # m

def random_mass_and_temp() -> tuple[float, float, float]:
    # calls values from rnad_temp_mass.h for floats
    rho = float(input("Enter liquid density (kg/m^3): "))
    mass, temp, rho = generateRandomMassTempRho(rho)
    return mass, temp, rho

"""
def physics_strain_from_mass_density(mass: float, density: float) -> float:
    fraction = max(0.0, min(mass / density / V_max, 1.0))
    return BASE_EMPTY_STRAIN + (MAX_STRAIN - BASE_EMPTY_STRAIN) * fraction

def physics_mass_from_strain_density(strain: float, density: float) -> float:
    fraction = (strain - BASE_EMPTY_STRAIN) / (MAX_STRAIN - BASE_EMPTY_STRAIN)
    fraction = max(0.0, min(fraction, 1.0))
    volume = fraction * V_max
    return volume * density
"""

def physics_strain_from_mass_density(mass: float, density: float) -> float:
    w = mass * g / L_beam
    M = w * L_beam**2 / 8
    return (M * y_beam) / (E_beam * I_beam)

def physics_mass_from_strain_density(strain: float, density: float) -> float:
    M = strain * E_beam * I_beam / y_beam
    w = 8 * M / (L_beam**2)
    mass = w * L_beam / g
    return mass

if __name__ == "__main__":
    m, t, r = random_mass_and_temp()
    print(f"mass={m:.2f} kg, temp={t:.2f} degrees C, rho={r:.1f} kg/m^3")
    eps = physics_strain_from_mass_density(m, r)
    print(f"-> strain = {eps:.6e}")