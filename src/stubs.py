import numpy as np
import matplotlib.pyplot as plt
from math import acos, sqrt, pi

from mass_strain import (physics_strain_from_mass_density,physics_mass_from_strain_density,R_tank, V_max)
import fbg_tmm
from fbg_tmm import (generate_displacements,generate_raw_sensor_output,simulate_reflectivity_vectorized,simulate_spectra,decouple_regions,
    RAW_OUTPUT_SIZE, SimulationResolution,NUM_REGIONS, DYN_CWLS, STAT_CWLS,InitialRefractiveIndex, MeanChangeRefractiveIndex,FringeVisibility, M, use_apod, apod_sigma,ThermoOptic, PhotoElasticParam,)

import random_mass_temp
from random_mass_temp import generateRandomMassTempRho

BASE_EMPTY_STRAIN = 1.62e-5

def height_from_strain(strain: float) -> float:         # this strain is now ε_rec from decoupling  instead of using thermal and mechanical displacements  
    """
    Convert recorded strain into liquid height within tank
    tank is a horizontal cylinder
    use circle segment inversion
    
    requires mapping of strain to a volume % of tank
    numerically inverts the circle segment area formula via bisection to find the liquid depth that correspeonds to the volume % 
    """
    frac = (strain - BASE_EMPTY_STRAIN) / (MAX_STRAIN - BASE_EMPTY_STRAIN)
    vol = np.clip(frac, 0., 1.) * V_max
    R = R_tank
    L = V_max / (pi * R**2)
    
    # circle-segment area
    def segA(h):
        theta = acos((R - h) / R)
        return R*R*theta - (R - h) * sqrt(2*R*h - h*h)
    
    # invert via bisection, using height [0, 2R]
    low, high = 0.0, 2*R
    f_low, f_high = segA(low)*L - vol, segA(high)*L - vol
    
    # fall back to simple hight if signs are same
    if f_low * f_high > 0:
        return vol / (pi * R**2)
    
    # bisection iteration to ~1e-6m precision
    for _ in range(50):
        mid = 0.5*(low + high)
        f_mid = segA(mid)*L - vol
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
            
    return 0.5*(low + high)


# verifies SLD output for spectroscopy 
def compute_fwhm(wl: np.ndarray, refl: np.ndarray) -> float:
    peak = refl.max()
    half = peak/2.0
    inds = np.where(refl >= half)[0]
    if len(inds) < 2:
        return 0.0
    return wl[inds[-1]] - wl[inds[0]]


def test_mass_and_temp():
    """
    Generate mass, temp and rho
    overridding fbg_tmm.DENSITY set to water  
    computes and verifies reference strain
    """
    rho = float(input("Enter liquid density (kg/m^3): "))
    mass, temp, rho = generateRandomMassTempRho(rho)
    fbg_tmm.DENSITY = rho
    global MAX_STRAIN
    MAX_STRAIN = physics_strain_from_mass_density(rho*V_max, rho)
    eps = physics_strain_from_mass_density(mass, rho)
    print(f"[mass_strain] mass={mass:.2f} kg, temp={temp:.2f} °C → ε={eps:.2e}")
    return mass, temp, rho


def test_fbg_stubs(mass, temp, rho):
    """
    Run the full FBG pipeline -
      generate_displacements. Convert mass, temp and liquid density into strain  
      generate raw sensor data from simulated reflectivity spectra  
      height, fill % and mass recovery
      reflcitivty spectra plot and tank summary (maybe helpful for SHM, no idea what mechs are doing yet (30/04/25 lol))
    """
    disps = generate_displacements(mass, temp)

    # Storage for averages
    in_strains = []
    rec_strains = []
    rec_masses = []
    rec_fills = []
    rec_heights = []
    rec_volumes = []

    # First pass: plot each sensor
    for idx, disp in enumerate(disps):
        dynamic = (idx % 2 == 0)
        region = idx//2
        cwl = DYN_CWLS[region] if dynamic else STAT_CWLS[region]

        # reflectivity with strain/temp
        spec = simulate_spectra(cwl, disp if dynamic else 0.0, temp)
        refl = (spec['RD'] if dynamic else spec['RT'])*100
        wl = spec['wl']

        # individual plot ±1 nm window
        plt.figure()
        plt.plot(wl, refl)
        plt.axvline(cwl, color='gray', linestyle=':')
        peak_idx = np.argmax(refl)
        plt.axvline(wl[peak_idx], color='red', linestyle=':')
        plt.title(f"Sensor {idx}|Region {region}|{'Strain & Temperature (Dynamic)' if dynamic else 'Temperature Only (Static)'}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectivity (%)")
        plt.xlim(cwl-1, cwl+1)
        plt.grid(True)

    # Summary pass
    print("=== Simulation Summary ===")
    peaks = []
    for idx, disp in enumerate(disps):
        dynamic = (idx % 2 == 0)
        region = idx//2
        cwl = DYN_CWLS[region] if dynamic else STAT_CWLS[region]

        # undeformed
        wl0 = np.linspace(cwl-1, cwl+1, RAW_OUTPUT_SIZE)
        R0 = simulate_reflectivity_vectorized(cwl*1e-9/(2*InitialRefractiveIndex),wl0, InitialRefractiveIndex,MeanChangeRefractiveIndex,FringeVisibility,M, use_apod, apod_sigma)*100
        R0p = R0.max()
        BW  = compute_fwhm(wl0, R0)

        # deformed
        spec = simulate_spectra(cwl, disp if dynamic else 0.0, temp)
        Rdef= (spec['RD'] if dynamic else spec['RT'])*100
        Rdp = Rdef.max()

        # record peak λ for decoupling later
        peaks.append(spec['wl'][np.argmax(Rdef if dynamic else spec['RT']*100)])

        if dynamic:
            print(f"(will decouple) Sensor {idx+1} dynamic…") 
        else:
            print(f"(will decouple) Sensor {idx+1} static…")

        # store summary lows
        R0_list, BW_list, Rdef_list = R0p, BW, Rdp

    # decouple all regions at once
    dec = decouple_regions(peaks) 

    # now print final lines
    for region,(ΔT, ε_rec) in enumerate(dec):
        idxD = 2*region
        cwlD = DYN_CWLS[region]

        # recover everything
        ε_in = disps[idxD]
        m_rec = physics_mass_from_strain_density(ε_rec, rho)
        fill = m_rec/(rho*V_max)*100
        h = height_from_strain(ε_rec)
        vol = m_rec/rho

        # store for averages
        in_strains.append(ε_in)
        rec_strains.append(ε_rec)
        rec_masses.append(m_rec)
        rec_fills.append(fill)
        rec_heights.append(h)
        rec_volumes.append(vol)

        # final per‐sensor print
        print(
            f"Sensor {idxD+1}: CWL={cwlD:.1f} nm  "
            f"R₀={R0_list:.1f}%  BW @ 3dB={BW_list:.3f} nm  "
            f"R_def={Rdef_list:.1f}%  "
            f"ε_in={ε_in:.2e}  ε_rec={ε_rec:.2e}  "
            f"Fill={fill:.1f}%  h={h:.3f} m  Vol={vol:.4f} m³"
        )
        # static sensor
        idxT = idxD+1
        cwlT = STAT_CWLS[region]
        print(
            f"Sensor {idxT+1} (T-only): CWL={cwlT:.1f} nm  ΔT={ΔT:.2f} °C"
        )

    # AVERAGES for the 4 dynamic sensors only:
    print("=== Averages (strain+temp sensors) ===")
    print(f"Average input strain: {np.mean(in_strains):.4e}")
    print(f"Average recovered strain: {np.mean(rec_strains):.4e}")
    print(f"Average recovered mass: {np.mean(rec_masses):.1f} kg")
    print(f"Average fill percent: {np.mean(rec_fills):.1f}%")
    print(f"Average height: {np.mean(rec_heights):.3f} m")
    print(f"Average volume: {np.mean(rec_volumes):.4f} m³")


    # —————— Full‐SLD‐40nm-range overlay ——————
    wl_sld = np.linspace(1530.0, 1570.0, 2000)
    plt.figure(figsize=(10,5))
    for idx in range(NUM_REGIONS*2):
        # pick the correct CWL/period for this sensor
        dynamic = (idx % 2 == 0)
        region = idx//2
        cwl = DYN_CWLS[region] if dynamic else STAT_CWLS[region]
        period = (cwl*1e-9)/(2*InitialRefractiveIndex)
        
        # simulate reflectivity over full SLD
        R_sld = simulate_reflectivity_vectorized(period,wl_sld,InitialRefractiveIndex,MeanChangeRefractiveIndex,FringeVisibility,M,use_apod,apod_sigma) * 100
        
        plt.plot(wl_sld, R_sld, label=f"S{idx+1} @ {cwl:.1f} nm")

    plt.title("Full 40 nm SLD‐Range Reflectivity for All 8 Sensors")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectivity (%)")
    plt.xlim(1530,1570)
    plt.ylim(0, np.max(R_sld)*1.1)
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    mass, temp, rho = test_mass_and_temp()
    test_fbg_stubs(mass, temp, rho)
