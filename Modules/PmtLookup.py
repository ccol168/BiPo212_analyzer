"""
PmtLookup.py — JUNO CD-LPMT mapping utility
"""

import pandas as pd
from functools import lru_cache

CSV_PATH = "/home/claudio/Documenti/Dottorato/Work/BiPo212/FFTW_test/PMTMaps/pmt_20250422_CDLPMT.csv"

@lru_cache(maxsize=1)
def _LoadData():
    """Load CSV once and return a dict indexed by CopyNo."""
    df = pd.read_csv(CSV_PATH, comment='#')
    return df.set_index('CopyNo').to_dict(orient='index')

def PmtExists(copyNo: int) -> bool:
    return copyNo in _LoadData()

def GetPmtInfo(copyNo: int) -> dict | None:
    return _LoadData().get(copyNo)

# ---- Position helpers ----

def GetPmtPosition(copyNo: int) -> tuple[float, float, float] | None:
    info = GetPmtInfo(copyNo)
    if info:
        return info["X"], info["Y"], info["Z"]
    return None

def GetPmtX(copyNo: int) -> float | None:
    info = GetPmtInfo(copyNo)
    return info["X"] if info else None

def GetPmtY(copyNo: int) -> float | None:
    info = GetPmtInfo(copyNo)
    return info["Y"] if info else None

def GetPmtZ(copyNo: int) -> float | None:
    info = GetPmtInfo(copyNo)
    return info["Z"] if info else None

# ---- PMT type helpers ----

def GetPmtType(copyNo: int) -> str | None:
    info = GetPmtInfo(copyNo)
    return info["PMTType"] if info else None

def IsPmtHamamatsu(copyNo: int) -> bool:
    return GetPmtType(copyNo) == "Hamamatsu"

def IsPmtNNVT(copyNo: int) -> bool:
    pmtType = GetPmtType(copyNo)
    return pmtType and "NNVT" in pmtType

def GetPmtTheta(copyNo: int) -> float | None:
    info = GetPmtInfo(copyNo)
    return info["Orientation_theta"] if info else None

def GetPmtPhi(copyNo: int) -> float | None:
    info = GetPmtInfo(copyNo)
    return info["Orientation_phi"] if info else None

def GetPmtDirection(copyNo: int) -> tuple[float, float] | None:
    """Return (theta, phi) in degrees."""
    info = GetPmtInfo(copyNo)
    if info:
        return info["Orientation_theta"], info["Orientation_phi"]
    return None
