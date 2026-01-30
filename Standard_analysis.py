import os
import yaml
import argparse
import sys
import csv
from Modules import RateCalculator,GaussFitter, ExpoFitter
from Modules.DatasetAnalysis import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Circle
from tabulate import tabulate
from scipy.stats import norm,expon
import math

# ---------------- Utility functions ----------------

def parse_args():
    parser = argparse.ArgumentParser(description="BiPo analysis runner")
    parser.add_argument("--start", type=int, required=True, help="Starting run number")
    parser.add_argument("--end", type=int, required=True, help="Ending run number")
    parser.add_argument("--fit_type", type=str, default=None, help="Fit type (e.g. single, double)")
    parser.add_argument("--fit_range", type=float, nargs=2, default=None, help="Fit range [min max] in NHits")
    parser.add_argument("--save", action="store_true", help="Save figures and outputs instead of showing them")
    parser.add_argument("--output_name", type=str, help="Custom name for the output directory")
    parser.add_argument("--name_append", type=str, default="", help="Append string to default folder name")
    parser.add_argument("--config", type=str, default="configs/config_default.yaml", help="YAML configuration file")

    # --- FV CONTROL OPTION ---
    fv_group = parser.add_mutually_exclusive_group()
    fv_group.add_argument("--fv", dest="fv_enabled", action="store_true", help="Enable full FV analysis")
    fv_group.add_argument("--no-fv", dest="fv_enabled", action="store_false", help="Disable full FV analysis")
    parser.set_defaults(fv_enabled=None)  # Will fall back to config unless user overrides

    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_unique_dir(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    i = 1
    while True:
        new_dir = f"{base_dir}_{i}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1

# ---------------- Tee class ----------------

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# ---------------- Plotting functions ----------------

def PlotEventDistribution(Data, main_mask, cfg, output_dir=None, save=False):
    mask = (
        main_mask &
        (Data.NonUnCorrectedBi > cfg["Bi_min"]) &
        (Data.FinalPo > cfg["FinalPo_min"]) &
        (Data.FinalPo < cfg["FinalPo_max"])
    )

    viridis = plt.cm.viridis(np.linspace(0, 1, 256))
    viridis[0] = [1, 1, 1, 1]
    new_cmap = ListedColormap(viridis)

    plt.figure(figsize=(10, 7))
    plt.hist2d(
        Data.Recox[mask],
        Data.Recoz[mask],
        bins=100,
        cmap=new_cmap,
        range=[[-20000, 20000], [-20000, 20000]]
    )
    plt.colorbar(label="Number of events")

    for radius, color in zip(cfg["FV_circles"], cfg["FV_colors"]):
        circle = Circle((0, 0), radius, edgecolor=color, facecolor='none',
                        linestyle='--', linewidth=1.5, label=f"FV = {radius} mm")
        plt.gca().add_patch(circle)

    plt.title(f"Event distribution (FinalPo {cfg['FinalPo_min']}–{cfg['FinalPo_max']}, Bi>{cfg['Bi_min']})")
    plt.xlabel("Y [mm]")
    plt.ylabel("Z [mm]")
    plt.legend(loc="lower left")
    plt.axis('equal')

    if save and output_dir:
        plt.savefig(f"{output_dir}/Event_distribution.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def PlotEventDistribution_Z_Rho2(Data, main_mask, cfg, output_dir=None, save=False):
    mask = (
        main_mask &
        (Data.NonUnCorrectedBi > cfg["Bi_min"]) &
        (Data.FinalPo > cfg["FinalPo_min"]) &
        (Data.FinalPo < cfg["FinalPo_max"])
    )

    # Compute rho^2 = x^2 + y^2
    Rho2 = Data.Recox**2 + Data.Recoy**2

    viridis = plt.cm.viridis(np.linspace(0, 1, 256))
    viridis[0] = [1, 1, 1, 1]
    new_cmap = ListedColormap(viridis)

    plt.figure(figsize=(10, 7))
    plt.hist2d(
        Rho2[mask],
        Data.Recoz[mask],
        bins=100,
        cmap=new_cmap,
        range=[[0, (20000)**2], [-20000, 20000]]
    )
    plt.colorbar(label="Number of events")

    plt.title("ρ² vs Z Event Distribution")
    plt.xlabel("ρ² [mm²]  (x² + y²)")
    plt.ylabel("Z [mm]")

    if save and output_dir:
        plt.savefig(f"{output_dir}/Event_distribution_rho2_vs_Z.pdf",
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# ---------------- Main analysis ----------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Handle FV override logic
    if "FV_enabled" not in cfg:
        cfg["FV_enabled"] = True  # default enable

    if args.fv_enabled is not None:
        cfg["FV_enabled"] = args.fv_enabled

    if args.fit_type:
        cfg["fit_type"] = args.fit_type
    if args.fit_range:
        cfg["fit_range"] = list(args.fit_range)

    start_run = args.start
    end_run = args.end
    name_append = f"_{args.name_append}" if args.name_append else ""
    output_name = args.output_name or f"Std_analysis_{start_run}_{end_run}{name_append}"
    output_dir = ensure_unique_dir(os.path.join("Analyzed_runs", output_name))

    cfg_used_path = os.path.join(output_dir, "config_used.yaml")
    with open(cfg_used_path, "w") as f:
        yaml.safe_dump(cfg, f)

    if args.save:
        log_path = os.path.join(output_dir, "full_output.log")
        log_file = open(log_path, "w")
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[INFO] Starting analysis for runs {start_run}-{end_run}")
    print(f"[INFO] FV enabled? {cfg['FV_enabled']}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output directory: {output_dir}")

    Data = Dataset()
    print("[INFO] Reading data...")
    Data.loadFromFolder(
        cfg["data_folder"], start_run, end_run,
        withTimeHist=False, verbose=False
    )

    Data.PostProcessData(
        cfg["kernel_file"], cfg["lookup_table"],
        DN_NHits=cfg["DN_NHits"], delayMin=cfg["delay_min"],
        delayMax=cfg["delay_max"]
    )

    mask_muons = (Data.TimeSinceLastMuon > 0.001)
    main_mask = mask_muons & Data.DelaysCut
    muon_efficiency = 1 - (Data.nMuons * cfg["muon_veto_length"] / Data.Durations)
    total_efficiency = cfg["total_efficiency_factor"] * muon_efficiency

    results = []
    scale = 1e17

    # Standard FV radial cuts (BiPo212 only)
    for Rcut in cfg["FV_cuts"]:
        label = f"FV {Rcut/1000:.1f} m"
        mask = main_mask & (Data.R < Rcut) & (Data.NonUnCorrectedBi > cfg["Bi_min"]) & mask_muons

        values_BiPo212, err_BiPo212 =  GaussFitter.fit_and_plot(
            Data.FinalPo[mask], cfg["fit_range"], bins=100, fit_type=cfg["fit_type"],
            savefig=os.path.join(output_dir, f"Fit_{int(Rcut/1000)}m.pdf") if args.save else None
        )

        rate = RateCalculator.CalculateConcentrationWithRadialCut(
            Rcut/1000, values_BiPo212["signal"], err_BiPo212["signal"], total_efficiency, Data.Durations
        )
        results.append([label, rate[0], rate[1]])

    # Hemisphere cuts
    if cfg.get("FV_cuts_hemisphere_enabled", False):
        for Rcut in cfg.get("FV_cuts_hemisphere", []):
            for hemi, zmask in [("Z>0", Data.Recoz > 0), ("Z<0", Data.Recoz < 0)]:
                mask = main_mask & (Data.R < Rcut) & (Data.NonUnCorrectedBi > cfg["Bi_min"]) & zmask

                values_BiPo212, err_BiPo212 = GaussFitter.fit_and_plot(
                    Data.FinalPo[mask], cfg["fit_range"], bins=100, fit_type=cfg["fit_type"],
                    savefig=os.path.join(output_dir, f"Fit_{int(Rcut/1000)}m_{hemi}.pdf") if args.save else None
                )

                rate = RateCalculator.CalculateConcentrationWithRadialCutinHalf(
                    Rcut/1000, values_BiPo212["signal"], err_BiPo212["signal"], total_efficiency, Data.Durations
                )
                results.append([f"FV {Rcut/1000:.1f} m ({hemi})", rate[0], rate[1]])

    # Full FV BiPo212-only
    if cfg["FV_enabled"]:
        Rcut_FV = 16500
        Zmax_FV = 15500
        mask_FV = main_mask & (Data.R < Rcut_FV) & (Data.NonUnCorrectedBi > cfg["Bi_min"]) & (abs(Data.Recoz) < Zmax_FV)

        values_BiPo212, err_BiPo212 = GaussFitter.fit_and_plot(
            Data.FinalPo[mask_FV], cfg["fit_range"], bins=100, fit_type=cfg["fit_type"],
            savefig=os.path.join(output_dir, "Fit_FV.pdf") if args.save else None
        )

        rate = RateCalculator.CalculateConcentrationinFV(
            values_BiPo212["signal"], err_BiPo212["signal"], total_efficiency, Data.Durations
        )
        results.append(["FV R<16.5m & |z|<15.5m", rate[0], rate[1]])

    # Check decaying exponential fit 

    print("\n=================== Fit to determine expo fit extrema =======================================")

    mask_expo_gauss = main_mask & (Data.R < cfg["Expo_Rcut"]) & (Data.NonUnCorrectedBi > cfg["Bi_min"])

    values_fit, ____ = GaussFitter.fit_and_plot(
            Data.FinalPo[mask_expo_gauss], cfg["fit_range"],
            bins=100, fit_type=cfg["fit_type"],
            savefig=False, show_fig = False
    )

    tau = 299./math.log(2.)

    print("=================== End fit to determine expo fit extrema =======================================\n")

    if (cfg["Free_parameters"]) :
        mask_expo_fit = main_mask & (Data.FinalPo < values_fit["mu"] + cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.FinalPo > values_fit["mu"] - cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.NonUnCorrectedBi > cfg["Bi_min"])
        ExpoFitter.ExponentialFit(Data.Delays[mask_expo_fit],cut_range=[cfg["delay_min"],cfg["delay_max"]],reBin=cfg["Expo_reBin"],
                                       savefig=os.path.join(output_dir, f"Fit_expo_freeparameters.pdf") if args.save else None)
        
    if (cfg["Fixed_amplitude"]) :
        mask_expo_fit = main_mask & (Data.FinalPo < values_fit["mu"] + cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.FinalPo > values_fit["mu"] - cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.NonUnCorrectedBi > cfg["Bi_min"])

        # Evaluate the efficiency -> multiply per the efficiency of the gauss sigma cut and divide for the t interval considered
        sigma_cut_efficiency = norm.cdf(cfg["Expo_sigma_cut"]) - norm.cdf(-cfg["Expo_sigma_cut"])
        Fraction_in_time_cut = expon.cdf(cfg["delay_max"],scale=tau) - expon.cdf(cfg["delay_min"],scale=tau)
        Extrapolated_signal = values_fit["signal"]*sigma_cut_efficiency/Fraction_in_time_cut

        ExpoFitter.ExponentialFit(Data.Delays[mask_expo_fit],cut_range=[cfg["delay_min"],cfg["delay_max"]],
                                       reBin=cfg["Expo_reBin"], fix_signal = Extrapolated_signal,
                                       savefig=os.path.join(output_dir, f"Fit_expo_fixedA.pdf") if args.save else None)
        
    if (cfg["Fixed_halflife"]) :
        mask_expo_fit = main_mask & (Data.FinalPo < values_fit["mu"] + cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.FinalPo > values_fit["mu"] - cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.NonUnCorrectedBi > cfg["Bi_min"])
        ExpoFitter.ExponentialFit(Data.Delays[mask_expo_fit],cut_range=[cfg["delay_min"],cfg["delay_max"]],
                                       reBin=cfg["Expo_reBin"], fix_tau = tau,
                                       savefig=os.path.join(output_dir, f"Fit_expo_fixed_HalfLife.pdf") if args.save else None)
        
    if (cfg["Fixed_zero_C"]) :
        mask_expo_fit = main_mask & (Data.FinalPo < values_fit["mu"] + cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.FinalPo > values_fit["mu"] - cfg["Expo_sigma_cut"] * values_fit["sigma"]) & (Data.NonUnCorrectedBi > cfg["Bi_min"])
        ExpoFitter.ExponentialFit(Data.Delays[mask_expo_fit],cut_range=[cfg["delay_min"],cfg["delay_max"]],
                                       reBin=cfg["Expo_reBin"], fix_bkg = 0.,
                                       savefig=os.path.join(output_dir, f"Fit_expo_fixed_zeroC.pdf") if args.save else None)


    # Save results CSV
    csv_path = os.path.join(output_dir, "Rate_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Radial Cut (m)", "Rate (×10⁻¹⁷)", "Error (×10⁻¹⁷)"])
        for label, rate, err in results:
            writer.writerow([label, rate * scale, err * scale])
    print(f"[INFO] CSV summary saved: {csv_path}")

    # Save results TXT
    txt_path = os.path.join(output_dir, "Rate_summary.txt")
    with open(txt_path, "w") as f:
        f.write(tabulate([[l, r*scale, e*scale] for l,r,e in results],
                         headers=["Radial Cut (m)","Rate (×10⁻¹⁷)","Error (×10⁻¹⁷)"],
                         tablefmt="fancy_grid", floatfmt=".4f"))
    print(f"[INFO] TXT summary saved: {txt_path}")

    PlotEventDistribution(Data, main_mask, cfg, output_dir=output_dir, save=args.save)
    PlotEventDistribution_Z_Rho2(Data, main_mask, cfg, output_dir=output_dir, save=args.save)

    print("[INFO] Analysis completed successfully.")

    if args.save:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()

if __name__ == "__main__":
    main()
