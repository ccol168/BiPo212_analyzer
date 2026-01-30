#!/usr/bin/env python3
"""
Script to calculate Thorium concentrations from ROOT files
"""

import ROOT
import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

def get_volume(cut_type):
    """Calculate active volume in m^3 for different cuts"""
    if "R < 16500" in cut_type and "|RecoZ| < 15000" in cut_type:
        # Spherical cap cut at top and bottom
        # V = V_sphere - V_cap_top - V_cap_bottom
        # V_cap = π * h^2 * (3r - h) / 3, where h is height of cap
        r = 16.5  # meters
        z_cut = 15.0  # meters
        # Height of each cap: h = r - sqrt(r^2 - z_cut^2) doesn't apply here
        # Instead: the cap starts at z = z_cut, h = r - z_cut
        h_cap = r - z_cut
        v_sphere = (4.0/3.0) * np.pi * (r**3)
        v_cap = np.pi * (h_cap**2) * (3*r - h_cap) / 3.0
        return v_sphere - 2 * v_cap  # subtract both top and bottom caps
    elif "R < 16500" in cut_type:
        # Sphere: V = 4/3 * π * r^3, r=16.5m
        return (4.0/3.0) * np.pi * (16.5**3)
    elif "R < 15000" in cut_type and "Z > 0" in cut_type:
        # Hemisphere (upper): V = 2/3 * π * r^3, r=15m
        return (2.0/3.0) * np.pi * (15.0**3)
    elif "R < 15000" in cut_type and "Z < 0" in cut_type:
        # Hemisphere (lower): V = 2/3 * π * r^3, r=15m
        return (2.0/3.0) * np.pi * (15.0**3)
    elif "R < 15000" in cut_type:
        # Sphere: V = 4/3 * π * r^3, r=15m
        return (4.0/3.0) * np.pi * (15.0**3)
    elif "R < 13000" in cut_type:
        # Sphere: V = 4/3 * π * r^3, r=13m
        return (4.0/3.0) * np.pi * (13.0**3)
    elif "R < 10000" in cut_type:
        # Sphere: V = 4/3 * π * r^3, r=10m
        return (4.0/3.0) * np.pi * (10.0**3)
    return 0.0

def apply_position_cut(x, y, z, cut_type):
    """Apply position cuts based on cut type"""
    R = np.sqrt(x**2 + y**2 + z**2)
    
    if cut_type == "R < 16500":
        return R < 16500
    elif cut_type == "R < 16500 and |RecoZ| < 15000":
        return (R < 16500) & (np.abs(z) < 15000)
    elif cut_type == "R < 15000":
        return R < 15000
    elif cut_type == "R < 15000 and Z > 0":
        return (R < 15000) & (z > 0)
    elif cut_type == "R < 15000 and Z < 0":
        return (R < 15000) & (z < 0)
    elif cut_type == "R < 13000":
        return R < 13000
    elif cut_type == "R < 10000":
        return R < 10000
    return False

def process_file(filepath, nhits_bi_cut, nhits_po_min, nhits_po_max, time_since_muon_cut):
    """Process a single ROOT file and return event counts"""
    f = ROOT.TFile.Open(str(filepath))
    if not f or f.IsZombie():
        print(f"Error opening file: {filepath}")
        return None
    
    # Get CdEvents tree
    tree = f.Get("CdEvents")
    if not tree:
        print(f"CdEvents tree not found in {filepath}")
        f.Close()
        return None
    
    # Get summary tree
    summary_tree = f.Get("summary")
    if not summary_tree:
        print(f"summary tree not found in {filepath}")
        f.Close()
        return None
    
    # Read summary info
    summary_tree.GetEntry(0)
    n_muons = summary_tree.nMuonsTotal
    run_length = summary_tree.runLength
    
    # Get first timestamp for run start time
    tree.GetEntry(0)
    first_timestamp = tree.TimeStamp
    start_time = datetime.fromtimestamp(first_timestamp.GetSec())
    
    # Extract run number from filename
    filename = Path(filepath).name
    run_number = filename.replace("RUN", "").replace("_merged.root", "")
    
    # Apply base cuts and collect events
    x_vals, y_vals, z_vals = [], [], []
    
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
        # Base cuts
        if tree.NHitsBi <= nhits_bi_cut:
            continue
        if tree.NHitsPo <= nhits_po_min or tree.NHitsPo >= nhits_po_max:
            continue
        if tree.TimeSinceLastMuon <= time_since_muon_cut:
            continue
        
        x_vals.append(tree.Recox)
        y_vals.append(tree.Recoy)
        z_vals.append(tree.Recoz)
    
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)
    
    # Define position cuts
    position_cuts = [
        "R < 16500",
        "R < 16500 and |RecoZ| < 15000",
        "R < 15000",
        "R < 15000 and Z > 0",
        "R < 15000 and Z < 0",
        "R < 13000",
        "R < 10000"
    ]
    
    # Count events for each position cut
    results = {
        'run_number': run_number,
        'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'n_muons': n_muons,
        'run_length': run_length,
        'events_after_base_cuts': len(x_vals)
    }
    
    for cut in position_cuts:
        mask = apply_position_cut(x_vals, y_vals, z_vals, cut)
        count = np.sum(mask)
        results[f'count_{cut}'] = count
        results[f'error_{cut}'] = np.sqrt(count)  # Poisson error
    
    f.Close()
    return results

def calculate_concentrations(results_df, efficiency, muon_veto_time=2e-3):
    """Calculate Th concentrations from event counts"""
    
    position_cuts = [
        "R < 16500",
        "R < 16500 and |RecoZ| < 15000",
        "R < 15000",
        "R < 15000 and Z > 0",
        "R < 15000 and Z < 0",
        "R < 13000",
        "R < 10000"
    ]
    
    for cut in position_cuts:
        count_col = f'count_{cut}'
        error_col = f'error_{cut}'
        
        # Calculate livetime (corrected for muon veto)
        livetime = results_df['run_length'] - (muon_veto_time * results_df['n_muons'])
        
        # Get volume for this cut
        volume = get_volume(cut)
        
        # Calculate concentration: counts / efficiency / duration / volume / 4060 / 856 / 1e3
        concentration = (results_df[count_col] / efficiency / livetime / 
                        volume / 4060.0 / 856.0 / 1e3)
        
        # Propagate error (only from counting statistics, assuming efficiency is exact)
        concentration_error = (results_df[error_col] / efficiency / livetime / 
                              volume / 4060.0 / 856.0 / 1e3)
        
        results_df[f'concentration_{cut}'] = concentration
        results_df[f'concentration_error_{cut}'] = concentration_error
    
    return results_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Calculate Thorium concentrations from ROOT files'
    )
    parser.add_argument('folder_path', type=str, 
                       help='Path to folder containing ROOT files')
    parser.add_argument('--begin-run', type=int, default=None,
                       help='First run number to process (optional)')
    parser.add_argument('--end-run', type=int, default=None,
                       help='Last run number to process (optional)')
    parser.add_argument('--efficiency', type=float, required=True,
                       help='Detection efficiency (e.g., 0.85 for 85%%)')
    parser.add_argument('--output', type=str, default='thorium_concentrations.csv',
                       help='Output CSV filename (default: thorium_concentrations.csv)')
    parser.add_argument('--nhits-bi-cut', type=float, default=1000,
                       help='NHitsBi cut threshold (default: 1000)')
    parser.add_argument('--nhits-po-min', type=float, default=1550,
                       help='NHitsPo minimum threshold (default: 1550)')
    parser.add_argument('--nhits-po-max', type=float, default=2150,
                       help='NHitsPo maximum threshold (default: 2150)')
    parser.add_argument('--time-since-muon-cut', type=float, default=0.001,
                       help='TimeSinceLastMuon cut threshold in seconds (default: 0.001)')
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    begin_run = args.begin_run
    end_run = args.end_run
    efficiency = args.efficiency
    muon_veto = args.time_since_muon_cut
    
    # Find all ROOT files
    all_files = sorted(folder_path.glob("RUN*_merged.root"))
    
    # Filter by run number if specified
    if begin_run is not None or end_run is not None:
        filtered_files = []
        for f in all_files:
            run_num = int(f.name.replace("RUN", "").replace("_merged.root", ""))
            if begin_run is not None and run_num < begin_run:
                continue
            if end_run is not None and run_num > end_run:
                continue
            filtered_files.append(f)
        all_files = filtered_files
    
    print(f"Processing {len(all_files)} files...")
    
    # Process all files
    results = []
    for i, filepath in enumerate(all_files):
        print(f"Processing file {i+1}/{len(all_files)}: {filepath.name}")
        result = process_file(filepath, args.nhits_bi_cut, args.nhits_po_min, 
                            args.nhits_po_max, args.time_since_muon_cut)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate concentrations
    df = calculate_concentrations(df, efficiency, muon_veto)
    
    # Save to CSV
    output_file = args.output
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total runs processed: {len(df)}")
    print(f"Efficiency used: {efficiency}")
    print(f"Muon veto time: {muon_veto} s")
    print(f"Cuts applied:")
    print(f"  NHitsBi > {args.nhits_bi_cut}")
    print(f"  {args.nhits_po_min} < NHitsPo < {args.nhits_po_max}")
    print(f"  TimeSinceLastMuon > {args.time_since_muon_cut} s")

if __name__ == "__main__":
    main()