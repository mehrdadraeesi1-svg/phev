"""
PHEVEnv_v13 -- unified multi-mode PHEV environment ("plant v3")
================================================================

ONE environment class for every scenario of the study.  Scenarios differ ONLY in
  (a) the action parameterisation  -> an action wrapper (threshold / argmax / Gumbel)
  (b) the reward formulation       -> RewardConfig.energy in {"cs", "sep", "efc"}
      (+ the SOC-regulation weight w_soc_delta for the Eco variant)
Physics, supervisor, switching constraint, observation, SOC limits and emission
accounting are identical for all of them (this removes the lineage differences of
v6+/v11/v12 found in the pre-submission audit).

Plant v3 (PhysicsModels_v10):
  * 1.25-L turbocharged ENG engine maps, speed range 1250-4000 rpm, WOT limit
  * 96s1p 50-Ah LFP pack (17.5 kWh nominal), liquid cooled (hA = 50 W/K)
  * engine-off cooling time constant 30 min
  * power-split topology of the Mitsubishi Outlander PHEV type:
      EV          : traction motor only
      SERIES      : engine at its best-efficiency point (35 kW @ 2500 rpm) drives the
                    generator; battery balances generator and traction power
      PARALLEL    : clutch closed, engine coupled to the wheels through ONE fixed
                    reduction (engine rpm = wheel rpm x 3.40); the continuous action
                    u in [0,1] is the engine LOAD FRACTION, P_ice = u * P_wot(rpm);
                    the motor supplies the rest (or absorbs the surplus = load-point
                    shifting, limited by motor power and SOC_max)
      ENGINE_ONLY : clutch closed, engine alone follows demand (motor assists only
                    above the WOT limit)
      REGEN       : braking, imposed by the supervisor
    Coupled modes are available above 50 km/h (engine >= 1250 rpm at 46 km/h).
  * GHG-only well-to-wheel CO2 (tailpipe CO2 x WTT factor + grid CO2); pollutant
    masses (NOx/HC/CO) are logged but are NOT converted to CO2-equivalent.

Switching constraint (identical for all scenarios and baselines):
  hard minimum dwell of `min_dwell` steps on agent-requested mode changes
  (supervisor overrides -- braking, SOC protection, speed limits -- always pass),
  plus the soft switching cost of the reward.

Author: Mehrdad Raeesi
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

PLANT = "v3"
# SERIES OOL sensitivity: engine power 10..60 kW (u = 0.5 -> 35 kW, the fixed point) on the OOL
SERIES_OOL_PMIN, SERIES_OOL_PMAX = 10.0, 60.0
DATA_DIR = "data_1s"

CYCLE_FILES = {
    "nedc":          "vel_power_nedc_refine.csv",
    "ftp75":         "vel_power_ftp75_refine.csv",
    "us06":          "vel_power_sftp_us06_refine.csv",
    "highway":       "vel_power_epa_highway_refine.csv",
    "birmangham":    "vel_power_birmangham_refine.csv",
    "tehran1":       "vel_power_tehran1_refine.csv",
    "tehran_shahab": "vel_power_tehran_shahab_refine.csv",
    "wltc":          "vel_power_wltc_refine.csv",
}
TRAIN_CYCLES = ["nedc", "ftp75", "us06", "highway"]


def load_cycle(filepath: str, vel_unit: str = "kmh") -> np.ndarray:
    df = pd.read_csv(filepath)
    for col in ["time", "vel", "power"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: '{col}'")
    if vel_unit.lower() == "kmh":
        df["vel"] = df["vel"] / 3.6
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "vel", "power"]].values


def load_cycles(names, data_dir: str = DATA_DIR) -> Dict[str, np.ndarray]:
    out = {}
    for n in names:
        p = os.path.join(data_dir, CYCLE_FILES[n])
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        out[n] = load_cycle(p)
    return out


# =============================================================================
# Parameters
# =============================================================================

class DriveMode(Enum):
    EV = 0
    SERIES = 1
    PARALLEL = 2
    ENGINE_ONLY = 3
    REGEN = 4


@dataclass
class VehicleParams:
    wheel_radius: float = 0.33
    coupled_ratio: float = 3.40          # engine-to-wheel fixed reduction (Outlander-type)
    ev_max_speed_kmh: float = 120.0
    parallel_min_speed_kmh: float = 50.0
    motor_efficiency: float = 0.93
    generator_efficiency: float = 0.92
    inverter_efficiency: float = 0.96
    driveline_efficiency: float = 0.95
    regen_efficiency: float = 0.70
    max_motor_power_kw: float = 80.0
    max_batt_power_kw: float = 80.0
    series_power_kw: float = 35.0
    series_rpm: float = 2500.0
    ice_min_rpm: float = 1250.0
    ice_max_rpm: float = 4000.0


@dataclass
class WTWParams:
    K_WTT_gasoline: float = 1.19
    K_WTT_electricity: float = 2.5
    gasoline_LHV: float = 43.4e6
    gamma_grid: float = 400.0            # g CO2 / kWh


@dataclass
class HealthParams:
    C_nominal: float = 50.0              # Ah (pack)
    V_nominal: float = 350.0
    E_nominal_kwh: float = 17.5
    beta_crate: float = 0.15
    max_crate: float = 3.0
    beta_fluct: float = 0.05
    max_delta_I: float = 100.0
    SOC_target: float = 0.50
    SOC_min: float = 0.25
    SOC_max: float = 0.85
    SOC_min_emergency: float = 0.10
    SOC_max_emergency: float = 0.87
    QLOSS_ref: float = 0.2


@dataclass
class ThermalParams:
    T_ambient: float = 25.0
    T_eng_cold: float = 60.0
    T_eng_emergency: float = 120.0
    T_batt_safe: float = 45.0
    T_batt_emergency: float = 65.0
    T_batt_optimal: float = 28.0
    T_batt_safe_delta: float = 7.0


@dataclass
class RewardConfig:
    energy: str = "sep"          # "sep" (S2), "efc" (S3/S4), "cs" (S1), "aligned" (metric-aligned)
    # metric-aligned: per-step cost = fuel_g - k_repl * E_nom * dSOC, whose sum over a cycle is
    # exactly the charge-sustaining-corrected fuel used for evaluation (k_repl measured on the
    # plant by recharge_v13.py).  SOC is only constrained by the window penalties.
    k_repl: float = 245.0
    # separate fuel + electricity (S2)
    w_fuel: float = 0.12
    w_elec: float = 0.03
    fuel_rate_norm: float = 3.0
    # equivalent fuel consumption (S3/S4)
    w_energy: float = 0.12
    elec_fuel_equiv: float = 170.0
    fuel_eq_norm: float = 5.0
    # common terms (S2-S4)
    w_regen: float = 0.02
    w_soc: float = 4.0
    w_soc_delta: float = 200.0
    w_thermal: float = 0.3
    w_switch_base: float = 0.10
    w_switch_fast: float = 0.40
    accel_factor: float = 0.15
    switch_window: int = 20
    # S1 (CS-SAC) cost-based reward
    cs_w_emission: float = 1.0
    cs_w_energy: float = 0.5
    cs_w_health: float = 0.7
    cs_w_soc: float = 1.5
    cs_w_thermal: float = 0.3
    cs_w_regen: float = 0.5
    cs_w_mode_penalty: float = 1.0
    cs_base_co2_rate: float = 50.0
    cs_base_energy_rate: float = 0.1
    cs_base_crate: float = 2.0
    cs_base_soc_dev: float = 0.3
    cs_switch_base: float = 0.05
    cs_switch_fast: float = 0.40
    cs_switch_window: int = 8
    clip_lo: float = -5.0
    clip_hi: float = 2.0


# =============================================================================
# Environment
# =============================================================================

class PHEVMultiModeEnv(gym.Env):
    """Dict action: {"mode": 0..3 (EV, SERIES, PARALLEL, ENGINE_ONLY), "split": [u]}.
    25-dim observation (15 + 2 x 5 look-ahead), identical to v9-v12."""

    metadata = {"render_modes": []}

    def __init__(self, list_of_cycles: List[np.ndarray], dt: float = 1.0,
                 reward_config: RewardConfig = None, min_dwell: int = 10,
                 lookahead_steps: int = 5, normalize_reward: bool = True,
                 vehicle_params: VehicleParams = None, wtw_params: WTWParams = None,
                 health_params: HealthParams = None, thermal_params: ThermalParams = None,
                 ice_data_folder: str = "data_engine", batt_data_dir: str = "utiles",
                 soc_init_range: Tuple[float, float] = (0.30, 0.80), series_ool: bool = False,
                 **_ignored):
        super().__init__()
        self.cycles = list_of_cycles
        self.dt = dt
        self.rc = reward_config or RewardConfig()
        self.min_dwell = int(min_dwell)
        self.lookahead_steps = lookahead_steps
        self.normalize_reward = normalize_reward
        self.veh = vehicle_params or VehicleParams()
        self.wtw = wtw_params or WTWParams()
        self.health = health_params or HealthParams()
        self.thermal = thermal_params or ThermalParams()
        self.soc_init_range = soc_init_range

        # physics models are mandatory (no silent fallback)
        from PhysicsModels_v10 import ICEModelRL, BatteryModelRL
        self._ice = ICEModelRL(data_folder=ice_data_folder,
                               optimal_rpm=self.veh.series_rpm,
                               optimal_power_kw=self.veh.series_power_kw,
                               max_power_kw=100.0,
                               max_rpm=self.veh.ice_max_rpm, min_rpm=self.veh.ice_min_rpm,
                               idle_rpm=800.0, Tamb=self.thermal.T_ambient)
        if self._ice._maps.get("fuel") is None:
            raise RuntimeError("engine fuel map not loaded (data_engine/conshot_1.data)")
        self._batt = BatteryModelRL(data_dir=batt_data_dir, C_nominal_Ah=self.health.C_nominal,
                                    V_nominal=self.health.V_nominal,
                                    soc_init=self.health.SOC_target, Tamb=self.thermal.T_ambient)
        if self._batt._maps.get("ocv") is None:
            raise RuntimeError("battery OCV map not loaded (utiles/OCV.data)")
        # optional variable SERIES operating point on the optimal operating line (OOL)
        self.series_ool = bool(series_ool)
        if self.series_ool:
            self._ool_P, self._ool_rpm, self._ool_bsfc = self._build_ool()

        self.speed_max_mps = 50.0
        self.P_dem_max_kw = 100.0
        self.I_max_abs = 200.0
        self.T_eng_max_norm = 120.0
        self.T_batt_max_norm = 60.0
        self.CO2_ref = 1000.0

        self.action_space = spaces.Dict({
            "mode": spaces.Discrete(4),
            "split": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)})
        self.obs_dim = 15 + 2 * lookahead_steps
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        self._init_state()

    # ------------------------------------------------------------------ state
    def _init_state(self):
        self.cycle_data = None
        self.steps = 0
        self.max_steps = 0
        self.soc = self.health.SOC_target
        self.soc_init = self.soc
        self.T_eng = self.thermal.T_ambient
        self.T_batt = self.thermal.T_ambient
        self.current_mode = DriveMode.EV
        self.prev_mode = DriveMode.EV
        self.prev_speed_mps = 0.0
        self.prev_current = 0.0
        self.qloss_total = 0.0
        self.co2_cumulative = 0.0
        self.mode_switches = 0
        self.dwell = 10 ** 6            # steps since last applied mode change
        self.n_vetoed = 0
        self.n_overridden = 0
        self.mode_counts = {m: 0 for m in DriveMode}
        self._log: List[Dict] = []
        self._termination_reason = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._init_state()
        options = options or {}
        idx = options.get("cycle_idx", None)
        if idx is None:
            idx = self.np_random.integers(0, len(self.cycles))
        self.cycle_data = self.cycles[int(idx)]
        self.max_steps = len(self.cycle_data)
        if "soc_init" in options:
            s = float(options["soc_init"])
            soc = s / 100.0 if s > 1.0 else s
        else:
            soc = float(self.np_random.uniform(*self.soc_init_range))
        self.soc = self.soc_init = float(np.clip(soc, 0.15, 0.90))
        self._ice.reset(Tamb=self.thermal.T_ambient)
        self._batt.reset(soc_init=self.soc, Tamb=self.thermal.T_ambient)
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------- step
    def step(self, action: Dict):
        if self.steps >= self.max_steps - 1:
            return self._get_obs(), 0.0, False, True, self._get_info()
        t, speed_mps, P_dem_kw = self.cycle_data[self.steps]
        accel = (speed_mps - self.prev_speed_mps) / self.dt

        requested = DriveMode(int(action["mode"]))
        split = float(np.clip(np.asarray(action["split"]).reshape(-1)[0], 0.0, 1.0))

        # hard minimum dwell on agent-requested switches (identical for all scenarios)
        if (self.dwell < self.min_dwell and requested != self.current_mode
                and self.current_mode != DriveMode.REGEN
                and self.soc > self.health.SOC_min_emergency + 0.05):
            requested = self.current_mode
            self.n_vetoed += 1

        final_mode, sup_pen = self._supervisor(requested, speed_mps, P_dem_kw)
        if final_mode != requested and P_dem_kw >= -0.5:
            self.n_overridden += 1
        phys = self._apply_physics(final_mode, split, speed_mps, P_dem_kw)
        reward, comp = self._calc_reward(phys, final_mode, sup_pen, accel)
        self._update_state(phys, final_mode, speed_mps)
        self.steps += 1
        self._log_step(t, speed_mps, P_dem_kw, requested, final_mode, split, phys, reward)
        terminated = self._check_termination()
        truncated = self.steps >= self.max_steps - 1
        info = self._get_info()
        info["reward_components"] = comp
        if terminated:
            info["termination_reason"] = self._termination_reason
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------- supervisor
    def _supervisor(self, requested: DriveMode, speed_mps: float, P_dem_kw: float):
        v = speed_mps * 3.6
        h, veh = self.health, self.veh
        mode, pen = requested, 0.0
        if P_dem_kw < -0.5:
            return (DriveMode.EV if self.soc > h.SOC_max else DriveMode.REGEN), 0.0
        coupled_ok = v >= veh.parallel_min_speed_kmh
        if P_dem_kw < 3.0 and mode in (DriveMode.ENGINE_ONLY, DriveMode.PARALLEL):
            mode, pen = DriveMode.EV, pen - 0.2
        if mode in (DriveMode.PARALLEL, DriveMode.ENGINE_ONLY) and not coupled_ok:
            mode, pen = DriveMode.SERIES, pen - 0.1
        if mode == DriveMode.EV and v > veh.ev_max_speed_kmh:
            mode, pen = DriveMode.PARALLEL, pen - 0.3
        if self.soc < 0.35 and mode == DriveMode.EV:
            mode, pen = DriveMode.SERIES, pen - 0.5
        if self.soc < h.SOC_min and mode == DriveMode.EV:
            mode, pen = DriveMode.SERIES, pen - 1.5
        if self.soc < h.SOC_min_emergency + 0.05 and mode in (DriveMode.EV, DriveMode.SERIES):
            if coupled_ok:
                mode, pen = DriveMode.ENGINE_ONLY, pen - 2.5
            else:
                mode = DriveMode.SERIES
        if P_dem_kw > 0.85 * veh.max_motor_power_kw and mode == DriveMode.EV:
            mode, pen = (DriveMode.PARALLEL if coupled_ok else DriveMode.SERIES), pen - 0.3
        if self.soc > h.SOC_max and mode in (DriveMode.SERIES, DriveMode.ENGINE_ONLY):
            if v <= veh.ev_max_speed_kmh:
                mode, pen = DriveMode.EV, pen - 0.5
        return mode, pen

    # ---------------------------------------------------------------- physics
    def _build_ool(self):
        """Optimal operating line: for each engine power, the speed with minimum warm fuel
        rate subject to the WOT limit (sensitivity variant of the SERIES mode)."""
        P_grid = np.arange(SERIES_OOL_PMIN, SERIES_OOL_PMAX + 0.5, 1.0)
        rpms = np.arange(self.veh.ice_min_rpm, self.veh.ice_max_rpm + 1, 50.0)
        r_opt, b_opt = [], []
        for P in P_grid:
            best = (np.inf, self.veh.series_rpm)
            for r in rpms:
                if P > 0.98 * self._ice.max_power_kw_at(r):
                    continue
                f = self.fuel_rate_warm(P, r)
                if 0 < f < best[0]:
                    best = (f, r)
            r_opt.append(best[1]); b_opt.append(best[0] * 3600.0 / P)
        return P_grid, np.array(r_opt), np.array(b_opt)

    def series_point(self, split: float):
        """(P_ice [kW], rpm) of the SERIES mode."""
        if not self.series_ool:
            return self.veh.series_power_kw, self.veh.series_rpm
        P = SERIES_OOL_PMIN + float(np.clip(split, 0, 1)) * (SERIES_OOL_PMAX - SERIES_OOL_PMIN)
        return P, float(np.interp(P, self._ool_P, self._ool_rpm))

    def coupled_rpm(self, speed_mps: float) -> float:
        return speed_mps * 60.0 / (2 * np.pi * self.veh.wheel_radius) * self.veh.coupled_ratio

    def power_flow(self, mode: DriveMode, split: float, speed_mps: float, P_dem_kw: float):
        """Pure (state-free except SOC_max check) power split.
        Returns P_ice_kw, rpm, ice_mode, P_mot_shaft_kw, P_batt_kw (terminal), P_regen_kw."""
        veh = self.veh
        eta_m = veh.motor_efficiency * veh.inverter_efficiency
        P_ice = P_mot = P_regen = 0.0
        rpm, ice_mode = 0.0, "off"
        if mode == DriveMode.REGEN:
            if self.soc < self.health.SOC_max:
                P_regen = min(abs(P_dem_kw) * veh.regen_efficiency, veh.max_motor_power_kw)
            P_batt = -P_regen
        elif mode == DriveMode.EV:
            if P_dem_kw >= 0:
                P_mot = P_dem_kw
                P_batt = P_mot / eta_m
            else:   # braking with a full pack (SOC > SOC_max): friction brakes only
                P_batt = 0.0
        elif mode == DriveMode.SERIES:
            P_ice, rpm = self.series_point(split)
            ice_mode = "series"
            P_mot = P_dem_kw
            P_batt = P_mot / eta_m - P_ice * veh.generator_efficiency
        else:
            rpm = float(np.clip(self.coupled_rpm(speed_mps), veh.ice_min_rpm, veh.ice_max_rpm))
            ice_mode = "coupled"
            P_wot = self._ice.max_power_kw_at(rpm)
            if mode == DriveMode.PARALLEL:
                P_ice = split * P_wot
                P_ice_follow = max(P_dem_kw, 0.0) / veh.driveline_efficiency
                # no charging surplus above SOC_max; motor can absorb at most its rating
                if self.soc >= self.health.SOC_max:
                    P_ice = min(P_ice, P_ice_follow)
                P_ice = min(P_ice, (P_dem_kw + veh.max_motor_power_kw) / veh.driveline_efficiency)
            else:  # ENGINE_ONLY
                P_ice = min(max(P_dem_kw, 0.0) / veh.driveline_efficiency, P_wot)
            P_ice = max(P_ice, 0.0)
            P_mot = P_dem_kw - P_ice * veh.driveline_efficiency
            P_batt = P_mot / eta_m if P_mot >= 0 else P_mot * eta_m
        P_batt = float(np.clip(P_batt, -veh.max_batt_power_kw, veh.max_batt_power_kw))
        if P_ice < 0.1:
            P_ice, ice_mode = 0.0, "off"
        return P_ice, rpm, ice_mode, P_mot, P_batt, P_regen

    def fuel_rate_warm(self, P_ice_kw: float, rpm: float) -> float:
        """Warm-engine fuel rate [g/s] at (P, rpm) -- for model-based baselines."""
        if P_ice_kw < 0.1:
            return 0.0
        r = float(np.clip(rpm, self._ice.min_rpm, self._ice.max_rpm))
        bmep = float(np.clip(P_ice_kw * 9549.0 / r * 4 * np.pi / (self._ice.V_d * 1e5), 0, 22.5))
        return self._ice.get_fuel_rate_gs(P_ice_kw, r, bmep, 90.0)

    def _apply_physics(self, mode, split, speed_mps, P_dem_kw):
        P_ice, rpm, ice_mode, P_mot, P_batt, P_regen = self.power_flow(mode, split, speed_mps, P_dem_kw)
        # engine
        T_eng_before = self.T_eng
        r = self._ice.step_rl(P_ice_kw=P_ice, rpm=(rpm if ice_mode != "off" else None),
                              T_eng=self.T_eng, dt=self.dt,
                              mode=("series" if ice_mode == "series" else "coupled"))
        fuel_g = r["fuel_g"]
        cf = self._cold_factor(T_eng_before)
        # battery (terminal power in, current out)
        self._batt.soc, self._batt.T_batt, self._batt.qloss = self.soc, self.T_batt, self.qloss_total
        b = self._batt.step_rl(P_batt, self.T_batt, self.dt)
        # GHG-only well-to-wheel accounting
        co2_tail = fuel_g * 3.17
        co2_grid = elec_MJ = 0.0
        if P_batt > 0:
            kwh = P_batt * self.dt / 3600.0
            co2_grid = kwh * self.wtw.gamma_grid
            elec_MJ = kwh * 3.6 * self.wtw.K_WTT_electricity
        fuel_MJ = fuel_g / 1000.0 * self.wtw.gasoline_LHV / 1e6
        return {
            "P_ice_kw": P_ice, "rpm": (r["rpm"] if P_ice > 0 else 0.0), "P_mot_kw": P_mot,
            "P_batt_kw": P_batt, "P_regen_kw": P_regen, "current_A": b["current_A"],
            "soc_new": b["soc"], "T_batt_new": b["T_batt"], "T_eng_new": r["T_eng"],
            "qloss_total": b["qloss_total"], "fuel_g": fuel_g,
            "co2_tailpipe_g": co2_tail,
            "nox_g": fuel_g * 0.015 * cf, "hc_g": fuel_g * 0.003 * cf, "co_g": fuel_g * 0.008 * cf,
            "co2_wtw_g": co2_tail * self.wtw.K_WTT_gasoline + co2_grid,
            "energy_wtw_MJ": fuel_MJ * self.wtw.K_WTT_gasoline + elec_MJ,
        }

    def _cold_factor(self, T_eng):
        tc = self.thermal.T_eng_cold
        return 2.0 + 2.0 * (tc - T_eng) / tc if T_eng < tc else 1.0

    # ----------------------------------------------------------------- reward
    def _calc_reward(self, phys, mode, sup_pen, accel):
        rc, h, dt = self.rc, self.health, self.dt
        soc = phys["soc_new"]
        switched = (mode != self.current_mode)
        comp = {}
        if rc.energy == "cs":
            C_em = phys["co2_wtw_g"] / dt / rc.cs_base_co2_rate * rc.cs_w_emission
            C_en = phys["energy_wtw_MJ"] / dt / rc.cs_base_energy_rate * rc.cs_w_energy
            I = phys["current_A"]
            C_rate = abs(I) / h.C_nominal
            C_he = ((C_rate / rc.cs_base_crate) ** 2 * h.beta_crate
                    + abs(I - self.prev_current) / h.max_delta_I * h.beta_fluct) * rc.cs_w_health
            C_soc = abs(soc - h.SOC_target) / rc.cs_base_soc_dev * rc.cs_w_soc
            if soc < h.SOC_min:
                C_soc += 3.0 * ((h.SOC_min - soc) / 0.05) ** 2
            if soc > h.SOC_max:
                C_soc += 3.0 * ((soc - h.SOC_max) / 0.05) ** 2
            th = self.thermal
            C_cold = ((th.T_eng_cold - phys["T_eng_new"]) / th.T_eng_cold * 0.5
                      if phys["T_eng_new"] < th.T_eng_cold and phys["P_ice_kw"] > 5 else 0.0)
            T_dev = abs(phys["T_batt_new"] - th.T_batt_optimal) / th.T_batt_safe_delta
            C_th = (C_cold + max(0.0, T_dev - 1.0) ** 2 * 0.1) * rc.cs_w_thermal
            R_rg = phys["P_regen_kw"] * dt / 3600.0 * 5.0 * rc.cs_w_regen
            C_sw = 0.0
            if switched:
                C_sw = rc.cs_switch_fast if self.dwell < rc.cs_switch_window else rc.cs_switch_base
            reward = -(C_em + C_en + C_he + C_soc + C_th + C_sw) + R_rg + sup_pen * rc.cs_w_mode_penalty
            comp = dict(C_emission=C_em, C_energy=C_en, C_health=C_he, C_soc=C_soc,
                        C_thermal=C_th, C_switch=C_sw, R_regen=R_rg, sup=sup_pen)
        else:
            if rc.energy == "efc":
                elec_eq = max(0.0, phys["P_batt_kw"]) * dt / 3600.0 * rc.elec_fuel_equiv
                r_en = -rc.w_energy * (phys["fuel_g"] + elec_eq) / rc.fuel_eq_norm
            elif rc.energy == "sep":
                r_en = (-rc.w_fuel * (phys["fuel_g"] / dt) / rc.fuel_rate_norm
                        - rc.w_elec * max(0.0, phys["P_batt_kw"]) / self.veh.max_batt_power_kw)
            elif rc.energy == "aligned":
                dsoc_ = soc - self.soc
                cost_g = phys["fuel_g"] - rc.k_repl * h.E_nominal_kwh * dsoc_
                r_en = -rc.w_energy * cost_g / rc.fuel_eq_norm
            else:
                raise ValueError(rc.energy)
            aligned = rc.energy == "aligned"
            r_soc = 0.0 if aligned else -rc.w_soc * (soc - h.SOC_target) ** 2
            if soc < h.SOC_min:
                r_soc -= 2.0 * ((h.SOC_min - soc) / 0.05) ** 2
            if soc > h.SOC_max:
                r_soc -= 2.0 * ((soc - h.SOC_max) / 0.05) ** 2
            r_dsoc = 0.0 if aligned else rc.w_soc_delta * (soc - self.soc) * np.sign(h.SOC_target - self.soc)
            r_rg = 0.0 if aligned else phys["P_regen_kw"] * dt / 3600.0 * rc.w_regen   # regen already priced via dSOC
            r_th = -rc.w_thermal * max(0.0, phys["T_batt_new"] - self.thermal.T_batt_safe) ** 2
            C_sw = 0.0
            if switched:
                base = rc.w_switch_fast if self.dwell < rc.switch_window else rc.w_switch_base
                C_sw = base * (1.0 + rc.accel_factor * abs(accel))
            reward = r_en + r_soc + r_dsoc + r_rg + r_th - C_sw + sup_pen
            comp = dict(r_energy=r_en, r_soc=r_soc, r_soc_delta=r_dsoc, r_regen=r_rg,
                        r_thermal=r_th, C_switch=C_sw, sup=sup_pen)
        if self.normalize_reward:
            reward = float(np.clip(reward, rc.clip_lo, rc.clip_hi))
        return reward, comp

    # ------------------------------------------------------------ bookkeeping
    def _update_state(self, phys, mode, speed_mps):
        if mode != self.current_mode:
            self.mode_switches += 1
            self.dwell = 0
        else:
            self.dwell += 1
        self.prev_mode = self.current_mode
        self.current_mode = mode
        self.prev_speed_mps = speed_mps
        self.mode_counts[mode] += 1
        self.soc = phys["soc_new"]
        self.T_batt = phys["T_batt_new"]
        self.T_eng = phys["T_eng_new"]
        self.prev_current = phys["current_A"]
        self.qloss_total = phys["qloss_total"]
        self.co2_cumulative += phys["co2_wtw_g"]

    def _get_obs(self):
        speed_norm, power_norm, prog = 0.0, 0.5, 0.0
        if self.cycle_data is not None and self.steps < self.max_steps:
            _, v, p = self.cycle_data[self.steps]
            speed_norm = float(np.clip(v / self.speed_max_mps, 0, 1))
            power_norm = float(np.clip((p + self.P_dem_max_kw) / (2 * self.P_dem_max_kw), 0, 1))
            prog = self.steps / max(self.max_steps - 1, 1)
        one_hot = [0.0] * 4
        if self.current_mode.value < 4:
            one_hot[self.current_mode.value] = 1.0
        obs = [float(np.clip(self.soc, 0, 1)), speed_norm, power_norm,
               float(np.clip(self.T_eng / self.T_eng_max_norm, 0, 1)),
               float(np.clip(self.T_batt / self.T_batt_max_norm, 0, 1)),
               float(np.clip((self.prev_current + self.I_max_abs) / (2 * self.I_max_abs), 0, 1)),
               float(np.clip(abs(self.prev_current) / self.health.C_nominal / self.health.max_crate, 0, 1))
               ] + one_hot + [
               float(np.clip(self.qloss_total / self.health.QLOSS_ref, 0, 1)),
               float(np.clip(self.co2_cumulative / self.CO2_ref, 0, 1)),
               float(np.clip(max(0.0, self.thermal.T_eng_cold - self.T_eng) / self.thermal.T_eng_cold, 0, 1)),
               float(prog)]
        for i in range(1, self.lookahead_steps + 1):
            if self.cycle_data is not None:
                _, lv, lp = self.cycle_data[min(self.steps + i, self.max_steps - 1)]
                obs += [float(np.clip(lv / self.speed_max_mps, 0, 1)),
                        float(np.clip((lp + self.P_dem_max_kw) / (2 * self.P_dem_max_kw), 0, 1))]
            else:
                obs += [0.0, 0.5]
        return np.array(obs, dtype=np.float32)

    def _check_termination(self):
        h, th = self.health, self.thermal
        for cond, why in ((self.soc < h.SOC_min_emergency, f"soc_low ({self.soc*100:.1f}%)"),
                          (self.soc > h.SOC_max_emergency, f"soc_high ({self.soc*100:.1f}%)"),
                          (self.T_eng > th.T_eng_emergency, f"T_eng ({self.T_eng:.1f})"),
                          (self.T_batt > th.T_batt_emergency, f"T_batt ({self.T_batt:.1f})")):
            if cond:
                self._termination_reason = why
                return True
        self._termination_reason = None
        return False

    def _pct(self, m):
        return 100.0 * self.mode_counts[m] / max(self.steps, 1)

    def _get_info(self):
        fuel = float(sum(s["fuel_g"] for s in self._log)) if self._log else 0.0
        return {
            "soc": self.soc, "T_batt": self.T_batt, "T_eng": self.T_eng,
            "mode": self.current_mode.name, "mode_switches": self.mode_switches,
            "total_steps": self.steps, "co2_cumulative": self.co2_cumulative,
            "qloss_total": self.qloss_total, "termination_reason": self._termination_reason,
            "total_fuel_g": fuel, "total_co2_g": self.co2_cumulative,
            "ev_percent": self._pct(DriveMode.EV), "series_percent": self._pct(DriveMode.SERIES),
            "parallel_percent": self._pct(DriveMode.PARALLEL),
            "engine_only_percent": self._pct(DriveMode.ENGINE_ONLY),
            "regen_percent": self._pct(DriveMode.REGEN),
        }

    def _log_step(self, t, v, P, req, mode, split, phys, reward):
        self._log.append({
            "time": t, "velocity_kmh": v * 3.6, "power_demand_kW": P,
            "req_mode": req.name, "mode": mode.name, "split": split,
            "soc": self.soc * 100, "T_engine": self.T_eng, "T_battery": self.T_batt,
            "P_ice_kW": phys["P_ice_kw"], "rpm": phys["rpm"], "P_mot_kW": phys["P_mot_kw"],
            "P_batt_kW": phys["P_batt_kw"], "P_regen_kW": phys["P_regen_kw"],
            "current_A": phys["current_A"], "fuel_g": phys["fuel_g"],
            "nox_g": phys["nox_g"], "hc_g": phys["hc_g"], "co_g": phys["co_g"],
            "co2_wtw_g": phys["co2_wtw_g"], "reward": reward,
        })

    def get_logs(self):
        return pd.DataFrame(self._log) if self._log else None

    def get_summary(self):
        lg = self.get_logs()
        s = {
            "total_steps": self.steps,
            "total_fuel_g": float(lg["fuel_g"].sum()) if lg is not None else 0.0,
            "total_co2_g": self.co2_cumulative,
            "total_regen_kWh": float(lg["P_regen_kW"].sum() / 3600) if lg is not None else 0.0,
            "initial_soc": self.soc_init * 100, "final_soc": self.soc * 100,
            "mode_switches": self.mode_switches, "qloss_total": self.qloss_total,
            "termination_reason": self._termination_reason,
            "n_vetoed": self.n_vetoed, "n_overridden": self.n_overridden,
            "ev_percent": self._pct(DriveMode.EV), "series_percent": self._pct(DriveMode.SERIES),
            "parallel_percent": self._pct(DriveMode.PARALLEL),
            "engine_only_percent": self._pct(DriveMode.ENGINE_ONLY),
            "regen_percent": self._pct(DriveMode.REGEN),
        }
        if lg is not None:
            s.update(total_nox_g=float(lg["nox_g"].sum()), total_hc_g=float(lg["hc_g"].sum()),
                     total_co_g=float(lg["co_g"].sum()),
                     engine_starts=int(((lg["P_ice_kW"] > 0) & (lg["P_ice_kW"].shift(1, fill_value=0) <= 0)).sum()),
                     peak_T_batt=float(lg["T_battery"].max()),
                     rms_current_A=float(np.sqrt((lg["current_A"] ** 2).mean())))
        return s


# =============================================================================
# Action wrappers (the only difference between action parameterisations)
# =============================================================================

def _inner(env):
    while hasattr(env, "env"):
        env = env.env
    return env


class ThresholdActionWrapper(gym.Wrapper):
    """S1 (CS-SAC): a = (mode_signal, u) in [0,1]^2; mode = quartile of mode_signal."""
    def __init__(self, env, **_):
        super().__init__(env)
        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)

    @staticmethod
    def decode(a):
        m = float(np.clip(a[0], 0, 1))
        mode = 0 if m < 0.25 else 1 if m < 0.5 else 2 if m < 0.75 else 3
        return mode, float(np.clip(a[1], 0, 1))

    def step(self, action):
        mode, u = self.decode(np.asarray(action, dtype=np.float32))
        return self.env.step({"mode": mode, "split": np.array([u], np.float32)})


class ArgmaxActionWrapper(gym.Wrapper):
    """S2 (DH-SAC): a = (l_EV, l_SER, l_PAR, l_EO, x) in [-3,3]^5; mode = argmax l."""
    def __init__(self, env, **_):
        super().__init__(env)
        self.action_space = spaces.Box(-3.0, 3.0, shape=(5,), dtype=np.float32)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        return self.env.step({"mode": int(np.argmax(a[:4])),
                              "split": np.array([np.clip((a[4] + 3) / 6, 0, 1)], np.float32)})


class GumbelActionWrapper(gym.Wrapper):
    """S3/S4 (GH-SAC): same action vector; during training the applied mode is the
    hard Gumbel-max sample argmax(l + g), g ~ Gumbel(0,1) (the temperature of a
    Gumbel-Softmax relaxation does not change the hard sample, so none is used);
    at test time (deterministic=True) mode = argmax l."""
    def __init__(self, env, deterministic: bool = False, **_):
        super().__init__(env)
        self.action_space = spaces.Box(-3.0, 3.0, shape=(5,), dtype=np.float32)
        self.deterministic = deterministic
        self._rng = np.random.default_rng()

    def reset(self, **kw):
        if kw.get("seed") is not None:
            self._rng = np.random.default_rng(kw["seed"])
        return self.env.reset(**kw)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        l = a[:4].astype(np.float64)
        if not self.deterministic:
            u = self._rng.uniform(1e-10, 1.0 - 1e-10, size=4)
            l = l - np.log(-np.log(u))
        return self.env.step({"mode": int(np.argmax(l)),
                              "split": np.array([np.clip((a[4] + 3) / 6, 0, 1)], np.float32)})


# =============================================================================
# Scenarios
# =============================================================================

SCENARIOS = {
    #  name          action       reward  w_soc_delta curriculum  paper label
    "CS-SAC":      ("threshold", "cs",    None,  False),   # S1
    "DH-SAC":      ("argmax",    "sep",   None,  True),    # S2
    "GH-SAC":      ("gumbel",    "efc",   None,  True),    # S3
    "GH-SAC-Eco":  ("gumbel",    "efc",   120.0, True),    # S4
    "DH-SAC-EFC":  ("argmax",    "efc",   None,  True),    # factorial cell
    "GH-SAC-SR":   ("gumbel",    "sep",   None,  True),    # factorial cell
    "DH-SAC-MA":   ("argmax",    "aligned", None, True),   # metric-aligned reward
    "GH-SAC-MA":   ("gumbel",    "aligned", None, True),   # metric-aligned reward
}
# sensitivity scenarios with a variable SERIES operating point (same agent/reward as the base)
SCENARIOS["GH-SAC-MA-OOL"] = SCENARIOS["GH-SAC-MA"]
SCENARIOS["DH-SAC-MA-OOL"] = SCENARIOS["DH-SAC-MA"]
OOL_SCENARIOS = {"GH-SAC-MA-OOL", "DH-SAC-MA-OOL"}

WRAPPERS = {"threshold": ThresholdActionWrapper, "argmax": ArgmaxActionWrapper,
            "gumbel": GumbelActionWrapper}
MIN_DWELL = 10


def reward_config_for(scenario: str) -> RewardConfig:
    _, energy, wsd, _ = SCENARIOS[scenario]
    rc = RewardConfig(energy=energy)
    if wsd is not None:
        rc.w_soc_delta = wsd
    return rc


def make_env(scenario: str, cycles: List[np.ndarray], deterministic: bool = False,
             min_dwell: int = MIN_DWELL, seed: Optional[int] = None):
    act = SCENARIOS[scenario][0]
    inner = PHEVMultiModeEnv(cycles, dt=1.0, lookahead_steps=5,
                             reward_config=reward_config_for(scenario), min_dwell=min_dwell,
                             series_ool=scenario in OOL_SCENARIOS)
    env = WRAPPERS[act](inner, deterministic=deterministic)
    if seed is not None:
        env.reset(seed=seed)
    return env


# =============================================================================
# Rule-based baseline (applied through the same environment)
# =============================================================================

def rb_action(env: PHEVMultiModeEnv) -> Dict:
    _, v, P = env.cycle_data[env.steps]
    v_kmh = v * 3.6
    if P < -0.5:
        m = 0
    elif env.soc > 0.65 and v_kmh <= 80:
        m = 0
    elif env.soc < 0.35:
        m = 1
    elif v_kmh > 60 and P > 20:
        m = 2
    else:
        m = 1
    return {"mode": m, "split": np.array([0.5], np.float32)}


def run_policy(env: PHEVMultiModeEnv, policy, soc_init: float):
    env.reset(options={"soc_init": soc_init})
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(policy(env))
        done = term or trunc
    return env.get_summary()


__all__ = ["PHEVMultiModeEnv", "DriveMode", "RewardConfig", "VehicleParams", "SCENARIOS",
           "make_env", "load_cycle", "load_cycles", "rb_action", "run_policy",
           "CYCLE_FILES", "TRAIN_CYCLES", "DATA_DIR", "MIN_DWELL"]
