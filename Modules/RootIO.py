import ROOT
import numpy as np
import os
import re

class RootIO:
    """Handles fast loading of ROOT datasets."""

    @staticmethod
    def readRootFile(filename, treeName="CdEvents", withTimeHist=False):
        file = ROOT.TFile(filename, "READ")
        if not file or file.IsZombie():
            raise IOError(f"Could not open ROOT file: {filename}")
        if not file.Get(treeName):
            raise ValueError(f"Tree '{treeName}' not found in {filename}")

        tree = ROOT.RDataFrame(treeName, file)
        columns = list(tree.GetColumnNames())

        # --- Bulk read all needed columns ---
        needed_cols = [
            "PeakPositions", "NPE", "NPeaks", "PEPo", "PEBi",
            "Recox", "Recoy", "Recoz", "fSec", "fNanoSec",
            "NHitsBi", "NHitsPo", "TimeSinceLastMuon",
            "CorrTime", "DeconvolutedSignal", "TimeStamp", "PMTID"
        ]
        available_cols = [c for c in needed_cols if c in columns]
        arrays = tree.AsNumpy(available_cols)

        data = {}

        # --- Handle TimeStamp vs fSec/fNanoSec ---
        if "TimeStamp" in arrays:
            ts_array = arrays["TimeStamp"]
            data["fSec"] = np.array([ts.GetSec() for ts in ts_array], dtype=int)
            data["fNanoSec"] = np.array([ts.GetNanoSec() for ts in ts_array], dtype=int)
        else:
            data["fSec"] = arrays.get("fSec")
            data["fNanoSec"] = arrays.get("fNanoSec")

        # --- Copy per-event arrays ---
        for key in ["PeakPositions", "NPE", "NPeaks", "PEPo", "PEBi",
                    "Recox", "Recoy", "Recoz"]:
            if key in arrays:
                data[key] = arrays[key]

        # --- Optional NHits ---
        if "NHitsBi" in arrays and "NHitsPo" in arrays:
            data["NHitsBi"] = arrays["NHitsBi"]
            data["NHitsPo"] = arrays["NHitsPo"]
        elif "CorrTime" in arrays:
            # Compute NHitsBi/Po on the fly
            corrTime = arrays["CorrTime"]
            peakPositions = arrays["PeakPositions"]
            nEvents = len(corrTime)
            nhitsBi = np.zeros(nEvents, dtype=int)
            nhitsPo = np.zeros(nEvents, dtype=int)

            # Vectorized-ish computation
            for i in range(nEvents):
                times = np.array(corrTime[i])
                bi, po = peakPositions[i]
                nhitsBi[i] = np.count_nonzero((times > (bi*6-60)) & (times < (bi*6+120)))
                nhitsPo[i] = np.count_nonzero((times > (po*6-60)) & (times < (po*6+120)))

            data["NHitsBi"] = nhitsBi
            data["NHitsPo"] = nhitsPo
        
        if withTimeHist :
            data["TimeHistogram"] = arrays["CorrTime"]
            data["DeconvResult"] = arrays["DeconvolutedSignal"]
            data["PMTID"] = arrays["PMTID"]
 
        # --- TimeSinceLastMuon ---
        if "TimeSinceLastMuon" in arrays:
            data["TimeSinceLastMuon"] = arrays["TimeSinceLastMuon"]
        else:
            data["TimeSinceLastMuon"] = -10 * np.ones(len(data["fSec"]), dtype=int)

        # --- Summary info ---
        if file.Get("summary"):
            summary = ROOT.RDataFrame("summary", file)
            sarrays = summary.AsNumpy(["nMuonsTotal", "runLength"])
            data["nMuons"] = sarrays["nMuonsTotal"]
            data["Durations"] = sarrays["runLength"]

        file.Close()
        return data

    @staticmethod
    def collectFiles(folder, begin=None, end=None):
        pattern = re.compile(r"RUN(\d+)_merged\.root")
        files = []
        for filename in os.listdir(folder):
            match = pattern.match(filename)
            if match:
                runNum = int(match.group(1))
                if (begin is None or runNum >= begin) and (end is None or runNum <= end):
                    files.append(os.path.join(folder, filename))
        return sorted(files)
