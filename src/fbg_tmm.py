"""
FBG Transfer Matrix Method

Simulates Fiber Bragg Grating spectra in from sensor attached to support beam of horizontal cylinder tank

The FBG experiences both strain and temperature defomration variations. Strain and temperature effects on FBG must be decoupled, done by having two FBG's in the same region, with one being strain independant

TMM simulates forward and backwards proagating light through an optical fiber, and decouples them over segments of the grating, ahering to physical laws and the parameters of the cable and interrogator.

Uses stubs to communicate with c++ front end
    - generate_displacements(mass, temp) - list[float]: supports changes in "dynamic" (strain and temperature) and "static" (temperature only) sensor strains to separate the sensors per region
    - generate_raw_sensor_output(mass, temp, sensor_index) -> list[float]: creates RAW_OUTPUT_SIZE sized window around the central wavelength for the strained spectral reflectivity peak
    - finally decouples the strain from the temperature per region to determine and accurate strain on that particular beam where central wavelength's are 'paired'
"""

import numpy as np
from typing import List, Tuple
from mass_strain import physics_strain_from_mass_density

DENSITY = 1000.0  # default value; stub will set this at runtime

# Simulation pramaters
InitialRefractiveIndex = 1.45
MeanChangeRefractiveIndex = 1e-4
FringeVisibility = 1
DirectionalRefractiveP11 = 0.121
DirectionalRefractiveP12 = 0.27
PoissonsCoefficient = 0.17
ThermoOptic = 5.2e-7
FBGLength = 12e-3
M = 200
SimulationResolution = 0.0005
use_apod = True
apod_sigma = 50.0
insertion_loss_db_per_m = 0.3

# Photoelastic constant
def compute_photoelastic_constant(n0, p11, p12, nu):
    return (n0**2 / 2) * (p12 - nu * (p11 + p12))

PhotoElasticParam = compute_photoelastic_constant(InitialRefractiveIndex, DirectionalRefractiveP11, DirectionalRefractiveP12, PoissonsCoefficient)

# CWL distribution, evenly spaced around the CWL of a particular FBG sensor, spaced evenly to greatly avoid cross talk
def generate_sensor_cwls(num_sensors: int, base_wavelength: float = 1550.0, delta_nm: float = 4.0) -> List[float]:
    start = base_wavelength - (num_sensors // 2) * delta_nm
    return [start + i * delta_nm for i in range(num_sensors)]

# TMM reflectivity
def simulate_reflectivity_vectorized(FBGPeriod_m, wavelengths_nm, n0, dneff, FV, M, use_apod=True, apod_sigma=50.0):
    """
    Vectorized TRansfer Matrix for FBG reflectivity simulation
    implemented oringally in pure python was simple and constly, for each wavelength -> segment -> update 2x2 matrix in `pure python`
    
    instead vectorize
        - packs all ALL wavelengths into single numpy array `lam_m`
        - coupling constants `kaa` and `base_sigma` are computed per segment as 2D arrays, M segments by N wavelengths
        - numpy's ufuncs and np.where find complex conjugation propogation constants for every wavelgnth in all segments at once with no python loops
        - 2x2 transfer matrix built with np.eisnums, operating over all segments at once
        - extracts final reflectivity vectroized over wavelength array 
        
    Signifcantly sped up simulation time, allowing for simulation over thousands of wavelength points to be achieved in milliseconds instead seconds (minutes for more intense sims)
    
    function returns array of |r(λ)|^2 for each input wavelength
    """
    
    # convert insertion loss dB/m -> Neper/m
    alpha_db_per_m = insertion_loss_db_per_m  
    alpha_np_per_m = alpha_db_per_m * np.log(10) / 20  
    
    L_seg = FBGLength / M
    
    # apodization weights across M segments
    weights = (np.exp(-((np.arange(M) - (M - 1)/2)**2) / (2 * apod_sigma**2))
        if 
            use_apod
        else
            np.ones(M))
    
    # convert wavelengths to meters and broadcast
    lam_m = wavelengths_nm * 1e-9
    inv_lam = 1.0 / lam_m
    
    # base detuning factor for each wavelength (MxN)
    base_sigma = (2 * np.pi * n0 * (inv_lam - 1 / (2 * n0 * FBGPeriod_m)) + 2 * np.pi * dneff * inv_lam)
    
    # coupling coefficient
    kaa = (np.pi * FV * dneff) * inv_lam[np.newaxis, :] * weights[:, None]
    
    # complex propagtion constant
    arg = kaa**2 - base_sigma[None, :]**2
    
    # Compute the complex propagation constant y (real or imaginary)
    gammab = np.where(
        arg >= 0,
        np.sqrt(arg),
        1j * np.sqrt(-arg)
    )
    
    # avoids exact zeros 
    gammab += (np.abs(gammab) < 1e-12) * 1e-12
    
    # global transfer matrix 
    T_global = np.tile(np.eye(2, dtype=complex)[:, :, None], (1, 1, lam_m.size))
    
    # step through M segments, updates T_global by einsum
    for seg in range(M):
        g = gammab[seg]
        c = np.cosh(g * L_seg)
        s = np.sinh(g * L_seg)
        kk = kaa[seg]
        sgm = base_sigma
        
        # 2x2 transfer matrix for current segment, at each wavelength all at once
        T_seg = np.array([
            [c - 1j*(sgm/g)*s, -1j*(kk/g)*s],
            [1j*(kk/g)*s,      c + 1j*(sgm/g)*s]
        ], dtype=complex)                           # shaped [2, 2, N]
        
        # applies loss per segment
        T_seg *= np.exp(-alpha_np_per_m * L_seg)
        
        # multiply into global matrix [2 2, N] eisnum over 2x2 matrix
        T_global = np.einsum('ijw, jkw -> ikw', T_global, T_seg)
    
    # reflectivity over each wavelength    
    r = T_global[1, 0] / T_global[0, 0]
    return np.abs(r)**2

# Spectra simulation
def simulate_spectra(cwl, eps_in, deltaT):
    """
    builds wavelength around central wavelgnth at +-1 nm and simulates reflectivity after just temperarture deformation as well as both temperature and strain deformationx
    """
    wl = np.arange(cwl - 1.0, cwl + 1.0, SimulationResolution)
    IL = 10**(-2 * insertion_loss_db_per_m * FBGLength / 10)    #insertion loss (dB/m)
    # temp only
    dT = cwl * (1 + ThermoOptic * deltaT) * 1e-9 / (2 * InitialRefractiveIndex)     # grating period under only temp
    RT = simulate_reflectivity_vectorized(dT, wl, InitialRefractiveIndex, MeanChangeRefractiveIndex, FringeVisibility, M, use_apod, apod_sigma) * IL
    # strain amd temperature
    dD = cwl * (1 + (1 - PhotoElasticParam) * eps_in + ThermoOptic * deltaT) * 1e-9 / (2 * InitialRefractiveIndex)      #grating period under strain + temp
    RD = simulate_reflectivity_vectorized(dD, wl, InitialRefractiveIndex, MeanChangeRefractiveIndex, FringeVisibility, M, use_apod, apod_sigma) * IL
    return {'wl': wl, 'RT': RT, 'RD': RD}
    # wl, wavelength array      R0, unperturbed reflectivity        RT, reflectivty after temp      RD, reflectivty after strain + temp

# stubs
NUM_REGIONS = 4
NUM_SENSORS = NUM_REGIONS * 2
# full CWL list
CWLS = generate_sensor_cwls(NUM_SENSORS)
# split dynamic strain + temp and temp only sensors over cwl's
DYN_CWLS = CWLS[0::2]  # i 0,2,4,6
STAT_CWLS = CWLS[1::2] # i 1,3,5,7

RAW_OUTPUT_SIZE = 1000

INITIAL_TEMPERATURE = 20 # degrees C

def generate_displacements(target_mass: float, target_temperature: float) -> List[float]:
    """
    returns strain values over strain effected sensors, ignoring temperature only sensors
    """
    eps = physics_strain_from_mass_density(target_mass, DENSITY)
    deltaT = target_temperature - INITIAL_TEMPERATURE
    out = []
    for region in range(NUM_REGIONS):
        out.append(eps)  # dynamic sensor
        out.append(deltaT)  # static sensor
    return out

def generate_raw_sensor_output(target_mass: float, target_temperature: float, sensor_index: int) -> List[float]:
    """
    Retuyrn RAW_OUTPU_SIZE window of reflectivity spectrum samples around the central wavelgnth of the particular FBG
    RD is defined as dynamic sensor, experiencing the most (strain induced) deformation, and RT as static, experiencing no (strain induced) deformation
    """
    dynamic = (sensor_index < NUM_REGIONS)
    # pick the correct CWL
    if dynamic:
        cwl = DYN_CWLS[sensor_index]
        #eps = physics_strain_from_mass(target_mass, 'water')    #physics_strain_from_mass_density
        eps = physics_strain_from_mass_density(target_mass, DENSITY)
    else:
        cwl = STAT_CWLS[sensor_index - NUM_REGIONS]
        eps = 0.0

    spec = simulate_spectra(cwl, eps, target_temperature)
    data = spec['RD'] if dynamic else spec['RT']

    peak_idx = np.argmax(data)
    half = RAW_OUTPUT_SIZE // 2
    start = max(0, peak_idx - half)
    end = min(data.size, start + RAW_OUTPUT_SIZE)
    segment = data[start:end]

    if segment.size < RAW_OUTPUT_SIZE:
        segment = np.concatenate([segment, np.zeros(RAW_OUTPUT_SIZE - segment.size)])
    return segment.tolist()

def decouple_temperature_strain(lambda_T: float,lambda_D: float,cwl_T: float,cwl_D: float) -> Tuple[float, float]:
    """
    Given a peak at lambda_T (temperature only sensor) and lambda_D (strain and temperature sensor) as well as their respective central wavelengths
    return change in temperature in degrees C and strain (epsilon)
    """
    deltaT = (lambda_T / cwl_T - 1) / ThermoOptic
    eps = (lambda_D / cwl_D - 1 - ThermoOptic * deltaT) / (1 - PhotoElasticParam)
    return deltaT, eps


def decouple_regions(measured_peaks: List[float]) -> List[Tuple[float, float]]:
    """
    Unpacks paired peaks [lambda_D0, Lambda_T0, 1, 1, 2, 2...]
    """
    results = []
    for region in range(NUM_REGIONS):
        lambda_D = measured_peaks[2*region]
        lambda_T = measured_peaks[2*region + 1]
        cwl_D = DYN_CWLS[region]
        cwl_T = STAT_CWLS[region]
        dT, eps = decouple_temperature_strain(lambda_T, lambda_D, cwl_T, cwl_D)
        results.append((dT, eps))
    return results