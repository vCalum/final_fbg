# Final FBG Sensor Simulation

Provides complete simulation of Fiber Bragg Grating sensors in a liquid filled tank:

1. **C++ Header** (`rand_temp_mass.h`); Randomly generate liquid mass and temperature, as well as recieves liquid density from GUI. Return mass, temperature, and liquid density
2. **Python Mass to Strain Mapping** (`mass_strain.py`); Mass <--> strain linear model. Utilises mass, temperature and liquid density from `rand_temp_mass`
3. **FBG TMM Module** (`fbg_tmm.py`); Core Transfer Matrix Method (TMM) reflecitivty spectra simulation
   - Uses two stubs for front end calls
   - `generate_displacements(mass, temperature)`: returns a list of strains for all sensors  
   - `generate_raw_sensor_output(mass, temperature, sensor_index)`: returns a windowed `RAW_OUTPUT_SIZE` spectrum around each FBG’s central wavelength 
4. **Test Harness** (`test_stubs.py`); Demonstration use of stubs, plot each spectrum, compute liquid height, recovered mass, fill percentage, and summary statistics

## End-to-End Workflow
GUI: user inputs liquid density ρ, implemented in full group project
 --> C++: generateRandomMassTempRho() → (mass, temp, p)
 --> Python: random_mass_and_temp() calls C++ -> (mass, temp, p)
 --> Mass<->Strain: strain = physics_strain_from_mass_density(mass, p)
 --> FBG TMM:  
    - generate_displacements(strain) -> [eps_0,0,eps_1,0,...]  
    - generate_raw_sensor_output(...) -> 8 spectra
 --> Stub: decouple & compute ΔT, ε, heights, fill% -> floats
 --> GUI frontend receives results and displays liquid height and reflectvity spectra over each FBG

 Currently, different GUI was implemented for group project, goal is to reimplement individually with stronger focus on variable FBG types and options for SHM.