#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PT ignition map (0D) + laminar flame speed map (1D FreeFlame) demo with Cantera.

- Sweep P0, T0 (and optionally additive fraction)
- Step 1: 0D "reacted or not" judgment with a simple spark model
- Step 2: If reacted, compute 1D laminar flame speed S_L using FreeFlame
- Output: CSV + contour plots

Notes / Limitations:
- Uses gri30.yaml by default (includes N2O/NOx). For serious N2O decomposition at MPa,
  you should validate/replace the mechanism.
- 0D ignition uses a simplified spark as an instantaneous energy deposition to the whole gas.
- 1D FreeFlame gives "planar, adiabatic, freely-propagating" laminar flame speed (baseline).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cantera as ct


# -----------------------------
# Configuration data structures
# -----------------------------


@dataclass
class MixSpec:
    """Mixture specification for N2O + additive.

    mode:
      - 'air': additive fraction is treated as air (O2: 0.21, N2: 0.79 by mole)
      - 'o2' : additive fraction is pure O2
    x_add: additive mole fraction in the final mixture (0..1)
    x_n2o: if None, set to (1 - x_add); else explicit (must be consistent with remaining)
    x_n2: optional additional diluent N2 mole fraction; if None, remainder goes to N2O
    """

    mode: str
    x_add: float
    x_n2o: Optional[float] = None
    x_n2: float = 0.0


@dataclass
class SparkSpec:
    """Simplified spark energy deposition model."""

    E_spark_J: float = 0.5  # total energy [J]
    efficiency: float = 1.0  # fraction deposited as thermal energy [0..1]
    # In this demo, energy is deposited uniformly (0D). If you later move to multi-zone,
    # you can reinterpret this as "kernel volume" etc.


@dataclass
class IgnitionSpec:
    """Ignition / reaction criteria for 0D."""

    reactor_mode: str = "constV"  # 'constV' or 'constP'
    t_end: float = 0.02  # [s] integrate up to this time
    max_steps: int = 200000  # safety cap
    dt_max: float = 5e-6  # [s] max timestep for sampling
    dT_crit: float = 200.0  # [K] ignition if Tmax - T_init > dT_crit
    # Alternative/additional criteria could be max(dT/dt) threshold; keep simple for demo.


@dataclass
class FlameSpec:
    """1D flame solver settings."""

    width: float = 0.06  # [m] initial domain width
    transport: str = "mixture-averaged"  # mixture-averaged or 'Multi'
    loglevel: int = 0  # Cantera solver verbosity
    refine_ratio: float = 3.0
    refine_slope: float = 0.06
    refine_curve: float = 0.12
    prune: float = 0.0
    max_time_step_count: int = 9000


# -----------------------------
# Mixture builder
# -----------------------------


def build_mole_fractions(spec: MixSpec) -> Dict[str, float]:
    mode = spec.mode.lower()
    x_add = float(spec.x_add)
    if not (0.0 <= x_add <= 1.0):
        raise ValueError(f"x_add must be in [0,1], got {x_add}")

    x_n2_dil = float(spec.x_n2)
    if x_n2_dil < 0.0:
        raise ValueError(f"x_n2 must be >= 0, got {x_n2_dil}")

    if spec.x_n2o is None:
        x_n2o = 1.0 - x_add - x_n2_dil
    else:
        x_n2o = float(spec.x_n2o)

    if x_n2o < 0.0:
        raise ValueError(
            f"Resulting x_n2o is negative ({x_n2o}). "
            f"Check x_add and x_n2 or set x_n2o explicitly."
        )

    X: Dict[str, float] = {}

    # Additive
    if mode == "air":
        # Simple dry air split
        X["O2"] = X.get("O2", 0.0) + 0.21 * x_add
        X["N2"] = X.get("N2", 0.0) + 0.79 * x_add
    elif mode == "o2":
        X["O2"] = X.get("O2", 0.0) + x_add
    else:
        raise ValueError("mode must be 'air' or 'o2'")

    # N2O
    X["N2O"] = X.get("N2O", 0.0) + x_n2o

    # Additional diluent N2
    if x_n2_dil > 0:
        X["N2"] = X.get("N2", 0.0) + x_n2_dil

    # Normalize (guard against tiny numerical drift)
    s = sum(X.values())
    if s <= 0:
        raise ValueError("Total mole fraction sum is zero.")
    for k in list(X.keys()):
        X[k] /= s

    return X


# -----------------------------
# 0D ignition / reaction test
# -----------------------------


def apply_spark_uniform_energy(gas: ct.Solution, spark: SparkSpec) -> float:
    """Deposit spark energy uniformly by raising internal energy.

    Returns:
        deltaT_est (rough estimate of temperature rise based on cv at initial state)
    """
    E = spark.E_spark_J * spark.efficiency
    if E <= 0:
        return 0.0

    # Mass in 1 m^3? Not defined. So we must define a reactor volume to convert energy to specific energy.
    # We'll assume a nominal volume V0 (handled in the reactor setup), and compute m from gas density.
    # Here: do nothing; actual adjustment is done in run_0d by computing m from chosen V.
    return 0.0


def run_0d_reaction_test(
    mech: str,
    P0_Pa: float,
    T0_K: float,
    mix: MixSpec,
    spark: SparkSpec,
    ign: IgnitionSpec,
    V_reactor: float = 1.0e-6,  # [m^3] nominal micro-liter scale; only used to map E->specific energy
) -> Tuple[bool, float, float]:
    """Run 0D reactor and decide if "reacted" based on Tmax - T_init.

    Spark model: instantaneous energy deposition to the reactor contents at t=0
    by increasing specific internal energy (constV) or specific enthalpy (constP).

    Returns:
        reacted (bool), tau_ign (s or nan), Tmax (K)
    """
    gas = ct.Solution(mech)
    X = build_mole_fractions(mix)
    gas.TPX = T0_K, P0_Pa, X

    # Set reactor mass via chosen volume
    rho0 = gas.density
    m = rho0 * V_reactor
    if m <= 0:
        return False, float("nan"), float("nan")

    # Apply spark as state jump at t=0 (uniform)
    Edep = spark.E_spark_J * spark.efficiency
    if Edep > 0:
        if ign.reactor_mode == "constV":
            u_target = gas.u + Edep / m  # specific internal energy [J/kg]
            v = 1.0 / gas.density  # specific volume [m^3/kg]
            # Set by (U, V) while keeping composition
            gas.UV = u_target, v
        elif ign.reactor_mode == "constP":
            h_target = gas.h + Edep / m  # specific enthalpy [J/kg]
            # Set by (H, P)
            gas.HP = h_target, P0_Pa
        else:
            raise ValueError("reactor_mode must be 'constV' or 'constP'")

    T_init = gas.T

    # Reactor setup (adiabatic by default unless walls/heat transfer are added)
    if ign.reactor_mode == "constV":
        r = ct.IdealGasReactor(gas, energy="on", volume=V_reactor)
    else:
        r = ct.IdealGasConstPressureReactor(gas, energy="on")

    net = ct.ReactorNet([r])

    t = 0.0
    Tmax = T_init
    tau_ign = float("nan")

    # Integrate with controlled sampling
    steps = 0
    while t < ign.t_end and steps < ign.max_steps:
        t_target = min(t + ign.dt_max, ign.t_end)
        t = net.advance(t_target)
        steps += 1

        T = r.T
        if T > Tmax:
            Tmax = T

        # Simple ignition criterion: temperature rise beyond threshold
        if math.isnan(tau_ign) and (Tmax - T_init) >= ign.dT_crit:
            tau_ign = t  # first time crossing (approx)
            # Keep integrating a little can help avoid false positives, but for demo this is OK.
            # You can break here if you want speed:
            # break

    reacted = (Tmax - T_init) >= ign.dT_crit
    return reacted, tau_ign, Tmax


# -----------------------------
# 1D flame speed (baseline)
# -----------------------------


def run_1d_freeflame_speed(
    mech: str,
    P0_Pa: float,
    T0_K: float,
    mix: MixSpec,
    flame_spec: FlameSpec,
) -> float:
    """Compute freely propagating planar laminar flame speed S_L [m/s].

    Returns NaN if solver fails.
    """
    gas = ct.Solution(mech)
    X = build_mole_fractions(mix)
    gas.TPX = T0_K, P0_Pa, X
    gas.transport_model = flame_spec.transport

    f = ct.FreeFlame(gas, width=flame_spec.width)

    # Refinement criteria
    f.set_refine_criteria(
        ratio=flame_spec.refine_ratio,
        slope=flame_spec.refine_slope,
        curve=flame_spec.refine_curve,
        prune=flame_spec.prune,
    )

    # Reasonable initial guess: enable solution of energy eqn
    f.solve(loglevel=flame_spec.loglevel, auto=True)

    # In Cantera, inlet velocity for FreeFlame equals the laminar flame speed
    # (relative to unburned gas).
    sL = float(f.velocity[0])
    return sL


# -----------------------------
# Sweep runner + plotting
# -----------------------------


def make_PT_grid(
    P_min_MPa: float, P_max_MPa: float, nP: int, T_min_K: float, T_max_K: float, nT: int
) -> Tuple[np.ndarray, np.ndarray]:
    P = np.logspace(np.log10(P_min_MPa), np.log10(P_max_MPa), nP) * 1e6
    T = np.linspace(T_min_K, T_max_K, nT)
    return P, T


def main():
    ap = argparse.ArgumentParser(
        description="Demo: P-T ignition + flame speed map for N2O + air/O2 in Cantera."
    )
    ap.add_argument(
        "--mech",
        type=str,
        default="gri30.yaml",
        help="Cantera mechanism file (default: gri30.yaml)",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="air",
        choices=["air", "o2"],
        help="Additive mode: air or o2",
    )
    ap.add_argument(
        "--x_add",
        type=float,
        default=0.0,
        help="Additive mole fraction (air or O2) in [0,1]",
    )
    ap.add_argument(
        "--x_n2",
        type=float,
        default=0.0,
        help="Extra N2 diluent mole fraction (optional)",
    )
    ap.add_argument(
        "--Pmin", type=float, default=0.1, help="Min pressure in MPa (log sweep)"
    )
    ap.add_argument(
        "--Pmax", type=float, default=5.0, help="Max pressure in MPa (log sweep)"
    )
    ap.add_argument("--nP", type=int, default=18, help="# pressure points")
    ap.add_argument("--Tmin", type=float, default=300.0, help="Min temperature in K")
    ap.add_argument("--Tmax", type=float, default=1200.0, help="Max temperature in K")
    ap.add_argument("--nT", type=int, default=20, help="# temperature points")
    ap.add_argument(
        "--reactor",
        type=str,
        default="constV",
        choices=["constV", "constP"],
        help="0D reactor mode",
    )
    ap.add_argument(
        "--V",
        type=float,
        default=1e-6,
        help="0D reactor nominal volume [m^3] for spark energy scaling",
    )
    ap.add_argument(
        "--Espark",
        type=float,
        default=0.5,
        help="Spark energy [J] deposited uniformly (demo)",
    )
    ap.add_argument(
        "--eta", type=float, default=1.0, help="Spark thermal efficiency [0..1]"
    )
    ap.add_argument(
        "--tend", type=float, default=0.02, help="0D integration end time [s]"
    )
    ap.add_argument(
        "--dTcrit",
        type=float,
        default=200.0,
        help="Ignition if Tmax - T_init >= dTcrit [K]",
    )
    ap.add_argument(
        "--width", type=float, default=0.06, help="1D flame domain width [m]"
    )
    ap.add_argument(
        "--transport",
        type=str,
        default="mixture-averaged",
        choices=["mixture-averaged", "Multi"],
        help="Transport model for 1D flame",
    )
    ap.add_argument(
        "--out", type=str, default="pt_map", help="Output prefix (CSV + PNG)"
    )
    args = ap.parse_args()

    mix = MixSpec(mode=args.mode, x_add=args.x_add, x_n2o=None, x_n2=args.x_n2)
    spark = SparkSpec(E_spark_J=args.Espark, efficiency=args.eta)
    ign = IgnitionSpec(
        reactor_mode=args.reactor,
        t_end=args.tend,
        dT_crit=args.dTcrit,
    )
    flame_spec = FlameSpec(width=args.width, transport=args.transport)

    P_arr, T_arr = make_PT_grid(
        args.Pmin, args.Pmax, args.nP, args.Tmin, args.Tmax, args.nT
    )

    # Result arrays
    reacted = np.zeros(
        (args.nT, args.nP), dtype=int
    )  # T index first for plotting convenience
    tau_ign = np.full((args.nT, args.nP), np.nan, dtype=float)
    Tmax = np.full((args.nT, args.nP), np.nan, dtype=float)
    sL = np.full((args.nT, args.nP), np.nan, dtype=float)
    sL_ok = np.zeros((args.nT, args.nP), dtype=int)

    # Sweep
    for iT, T0 in enumerate(T_arr):
        for iP, P0 in enumerate(P_arr):
            print(f"[INFO] Running T={T0:.1f} K, P={P0/1e6:.3f} MPa ...")
            try:
                ok, tau, tmax = run_0d_reaction_test(
                    mech=args.mech,
                    P0_Pa=float(P0),
                    T0_K=float(T0),
                    mix=mix,
                    spark=spark,
                    ign=ign,
                    V_reactor=float(args.V),
                )
                reacted[iT, iP] = 1 if ok else 0
                tau_ign[iT, iP] = tau
                Tmax[iT, iP] = tmax
            except Exception as e:
                # Keep going; mark as not reacted
                reacted[iT, iP] = 0
                tau_ign[iT, iP] = np.nan
                Tmax[iT, iP] = np.nan
                continue

            # If reacted, try 1D flame speed
            if reacted[iT, iP] == 1:
                print("  [INFO]  -> reacted; running 1D FreeFlame for S_L ...")
                try:
                    sL_val = run_1d_freeflame_speed(
                        mech=args.mech,
                        P0_Pa=float(P0),
                        T0_K=float(T0),
                        mix=mix,
                        flame_spec=flame_spec,
                    )
                    sL[iT, iP] = sL_val
                    sL_ok[iT, iP] = 1
                except Exception:
                    sL[iT, iP] = np.nan
                    sL_ok[iT, iP] = 0

    # Save CSV (long format)
    rows: List[Dict[str, float]] = []
    for iT, T0 in enumerate(T_arr):
        for iP, P0 in enumerate(P_arr):
            rows.append(
                {
                    "P_Pa": float(P0),
                    "P_MPa": float(P0 / 1e6),
                    "T_K": float(T0),
                    "reacted": int(reacted[iT, iP]),
                    "tau_ign_s": (
                        float(tau_ign[iT, iP])
                        if np.isfinite(tau_ign[iT, iP])
                        else np.nan
                    ),
                    "Tmax_K": (
                        float(Tmax[iT, iP]) if np.isfinite(Tmax[iT, iP]) else np.nan
                    ),
                    "S_L_mps": float(sL[iT, iP]) if np.isfinite(sL[iT, iP]) else np.nan,
                    "S_L_ok": int(sL_ok[iT, iP]),
                    "x_add": float(args.x_add),
                    "mode": (
                        0.0 if args.mode == "air" else 1.0
                    ),  # numeric tag for convenience
                }
            )
    df = pd.DataFrame(rows)
    csv_path = f"{args.out}.csv"
    df.to_csv(csv_path, index=False)

    # Plotting grids
    P_grid_MPa, T_grid = np.meshgrid(P_arr / 1e6, T_arr)

    # Figure 1: reacted map (binary)
    plt.figure()
    plt.contourf(P_grid_MPa, T_grid, reacted, levels=[-0.5, 0.5, 1.5])
    plt.xscale("log")
    plt.xlabel("Pressure [MPa]")
    plt.ylabel("Temperature [K]")
    plt.title(
        f"Reacted map (0D) | mode={args.mode}, x_add={args.x_add:.3f}, reactor={args.reactor}"
    )
    plt.colorbar(label="reacted (0/1)")
    plt.tight_layout()
    plt.savefig(f"{args.out}_reacted.png", dpi=200)
    plt.close()

    # Figure 2: flame speed map (only where available)
    # Mask NaNs for nicer contours
    sL_masked = np.ma.masked_invalid(sL)
    plt.figure()
    plt.contourf(P_grid_MPa, T_grid, sL_masked)
    plt.xscale("log")
    plt.xlabel("Pressure [MPa]")
    plt.ylabel("Temperature [K]")
    plt.title(
        f"Laminar flame speed S_L [m/s] (1D FreeFlame) | mode={args.mode}, x_add={args.x_add:.3f}"
    )
    plt.colorbar(label="S_L [m/s]")
    plt.tight_layout()
    plt.savefig(f"{args.out}_SL.png", dpi=200)
    plt.close()

    # Figure 3: ignition delay (where defined)
    tau_masked = np.ma.masked_invalid(tau_ign)
    plt.figure()
    plt.contourf(P_grid_MPa, T_grid, tau_masked)
    plt.xscale("log")
    plt.xlabel("Pressure [MPa]")
    plt.ylabel("Temperature [K]")
    plt.title(f"Ignition delay tau_ign [s] (0D threshold) | dTcrit={args.dTcrit:g} K")
    plt.colorbar(label="tau_ign [s]")
    plt.tight_layout()
    plt.savefig(f"{args.out}_tau.png", dpi=200)
    plt.close()

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {args.out}_reacted.png, {args.out}_SL.png, {args.out}_tau.png")


if __name__ == "__main__":
    main()
