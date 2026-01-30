import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import ROOT
from .RootIO import RootIO

class Dataset:
    """Event-level ROOT dataset with derived quantities and corrections."""

    def __init__(self):
        # Event arrays
        self.PeakPositions = None
        self.NPE = None
        self.NPeaks = None
        self.PEPo = None
        self.PEBi = None
        self.Recox = None
        self.Recoy = None
        self.Recoz = None
        self.fSec = None
        self.fNanoSec = None
        self.TimeSinceLastMuon = None
        self.NHitsBi = None
        self.NHitsPo = None
        self.TotalNHits = None
        self.TimeHistogram = None
        self.DeconvResult = None
        self.PMTID = None

        # Derived quantities
        self.Delays = None
        self.DelaysCut = None
        self.Residuals = None
        self.R = None
        self.Costheta = None
        self.PromptTime = None
        self.DelayTime = None

        # Summary info
        self.nMuons = 0
        self.Durations = 0

        # internal lists for efficient append
        self._buffers = {}

    def loadFromFile(self, filename, withTimeHist=False):
        data = RootIO.readRootFile(filename, withTimeHist=withTimeHist)
        self._appendData(data)

    def loadFromFolder(self, folder, begin=None, end=None, withTimeHist=False, verbose=False):
        files = RootIO.collectFiles(folder, begin, end)
        for i, file in enumerate(files):
            self.loadFromFile(file, withTimeHist)
            if verbose:
                print(f"Loaded file #{i}: {file}")
        self.sortByTime()

    def _appendData(self, data):
        for key, val in data.items():
            if key in ["nMuons", "Durations"]:
                setattr(self, key, getattr(self, key, 0) + np.sum(val))
                continue
            if getattr(self, key, None) is None:
                self._buffers[key] = [val]
            else:
                self._buffers[key].append(val)

        # flatten buffers into arrays
        for key, buf in self._buffers.items():
            setattr(self, key, np.concatenate(buf))

    def sortByTime(self):
        if self.fSec is None or self.fNanoSec is None:
            return
        indices = np.lexsort((self.fNanoSec, self.fSec))
        for key in self._buffers.keys():
            setattr(self, key, getattr(self, key)[indices])

    def computeDerived(self, delayMin=250, delayMax=600):
        
        pp = np.array([list(p) for p in self.PeakPositions])
    
        self.PromptTime = pp[:,0].astype(int) * 6.
        self.DelayTime = pp[:,1].astype(int) * 6.
	
        self.Delays = (self.DelayTime - self.PromptTime)
        self.DelaysCut = (self.Delays > delayMin) & (self.Delays < delayMax)
        self.Residuals = self.NPE - self.PEPo - self.PEBi
        self.R = np.sqrt(self.Recox**2 + self.Recoy**2 + self.Recoz**2)
        self.Costheta = self.Recoz / self.R
        
    def ApplyNonUniformityCorrection(self, interpFile, DN_NHits=343):
        """Apply non-uniformity correction using a 2D interpolator."""
        df = pd.read_csv(interpFile, sep=" ")
        interp_func = LinearNDInterpolator(
            list(zip(df["r_mm"].values, df["costh"].values)),
            df["par1_minus_dcr"].values
        )

        DN_in_window = DN_NHits * 180 / 1005
        effective_R = np.maximum(self.R, 2500)

        
        self.FinalPo = (self.NonUnCorrectedPo - DN_in_window) * interp_func(15000, 0.0) / interp_func(effective_R, self.Costheta) + DN_in_window
        self.NonUnCorrectedBi = (self.NHitsBi - DN_in_window) * interp_func(15000, 0.0) / interp_func(effective_R, self.Costheta) + DN_in_window

    def APrioriCorrection(self, kernelFile, histName="h_ideal_prompt"):
        """Apply the a priori kernel correction to NonUnCorrectedPo."""
        f = ROOT.TFile.Open(kernelFile)
        h = f.Get(histName)
        if not h:
            raise RuntimeError(f"Histogram '{histName}' not found in {kernelFile}")

        nbins = h.GetNbinsX()
        kernel_contents = np.array([h.GetBinContent(i) for i in range(1, nbins + 1)])
        kernel_edges = np.array([h.GetBinLowEdge(i) for i in range(1, nbins + 2)])

        # DN subtraction: mean between 100 and 200 ns
        begin = int(50. / 6.)
        end = int(150. / 6.)
        DN_per_bin = np.mean(kernel_contents[begin:end])
        kernel_minus_DN = np.maximum(kernel_contents - DN_per_bin, 0)
        max_position = np.argmax(kernel_minus_DN)
        bin_width = kernel_edges[1] - kernel_edges[0]

        # Compute corrected Po hits
        results = np.zeros_like(self.NHitsPo, dtype=float)
        for i, delay in enumerate(self.Delays):
            begin_window = int(delay / 6) + max_position - 10
            end_window = int(delay / 6) + max_position + 20
            window = np.array(kernel_minus_DN[begin_window:end_window]) * bin_width
            correction = np.sum(self.NHitsBi[i] * window)
            results[i] = self.NHitsPo[i] - correction

        self.NonUnCorrectedPo = results

    def PostProcessData (self, kernelFile, interpFile, delayMin = 250, delayMax = 600, DN_NHits = 343,histName = "h_ideal_prompt") :
        self.computeDerived(delayMin=delayMin,delayMax=delayMax)
        self.APrioriCorrection(kernelFile,histName=histName)
        self.ApplyNonUniformityCorrection(interpFile,DN_NHits=DN_NHits)
            