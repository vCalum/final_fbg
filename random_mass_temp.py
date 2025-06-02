import numpy as np
import math

# Tank geometry (local copy)
R_tank = 2.19 / 2.0
L_tank = 5.5
V_max   = math.pi * R_tank**2 * L_tank

def generateRandomMassTempRho(density: float, min_fill: float = 0.99) -> tuple[float, float, float]:
    """
    Generate a random tank mass and temperature, using user input liquid density
    density - float
    min_fill - float
    Minimum fraction of tank volume to use when sampling mass, default is 0.7 -> 70% of full tank as real ISO standard minimum fill is 80%

    Returns
    mass - float - random mass in kg
    temp - float - random temperature in degrees C, uniformly in [-40, 60].
    density - float - Same as the input density
    """
    mass = float(np.random.uniform(min_fill * density * V_max, density * V_max))
    temp = float(np.random.uniform(-40.0, 60.0))
    return mass, temp, density

if __name__ == "__main__":
    # test prompt the user for density
    rho = float(input("Enter liquid density (kg/m^3): "))
    mass, temp, rho = generateRandomMassTempRho(rho)
    print(f"Random mass = {mass:.2f} kg, temp = {temp:.2f} degrees C, rho = {rho:.1f} kg/m^3")
