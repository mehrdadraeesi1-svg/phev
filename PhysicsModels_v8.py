"""
Physics Models Adapter — v8
============================

این ماژول یک لایه نازک (Adapter) بین مدل‌های فیزیکی موجود
و محیط RL ایجاد می‌کند.

مشکل اصلی مدل‌های اصلی برای RL:
  • BatteryFile_v6 : آرایه‌های np.full(size, ...) از ابتدا
    → هزاران reset در آموزش → کند و حافظه‌بر
  • ICEFile_v4     : step(index, ...) → به index وابسته
    → در محیط RL index مفهوم ندارد

راه‌حل:
  • BatteryRL: مقدار OCV و R را از فایل‌های داده می‌خواند
    اما state را به‌صورت scalar نگه می‌دارد (بدون آرایه)
  • ICEModelRL: نقشه‌های موتور را می‌خواند، مصرف سوخت
    را از conshot_1.data (g/h) محاسبه می‌کند

هر دو کلاس:
  ① فقط یک‌بار فایل‌های داده را لود می‌کنند (class-level cache)
  ② reset() سریع است — فقط چند scalar به‌روز می‌شود
  ③ step_rl() خروجی dict دارد — سازگار با PHEVEnv_v8

نویسنده: مهرداد رعیسی
نسخه: 1.0  (برای استفاده در PHEVEnv_v8)
"""

from __future__ import annotations

import os
import numpy as np
import warnings
from typing import Dict, Optional, Tuple

# ─── تلاش برای import مدل‌های اصلی ──────────────────────────────────────────
try:
    from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator
    from scipy.interpolate import NearestNDInterpolator, interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy یافت نشد — از مدل‌های ساده استفاده می‌شود.")

try:
    from utiles.LoadFile import LoadFileData
    HAS_LOADFILE = True
except ImportError:
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utiles"))
        from LoadFile import LoadFileData
        HAS_LOADFILE = True
    except ImportError:
        HAS_LOADFILE = False
        warnings.warn("LoadFileData یافت نشد — از منحنی‌های پیش‌فرض استفاده می‌شود.")


# =============================================================================
# Cache سراسری — فایل‌ها یک‌بار خوانده می‌شوند
# =============================================================================
_BATTERY_CACHE: Dict = {}
_ICE_CACHE:     Dict = {}


def _load_2d_table(filepath: str) -> Optional[object]:
    """
    بارگذاری فایل 2DTable با فرمت Simcenter Amesim.
    خروجی: RectBivariateSpline یا None
    """
    if not HAS_SCIPY or not HAS_LOADFILE:
        return None
    if not os.path.exists(filepath):
        return None
    try:
        data = LoadFileData(filepath, "2DTable", "    ")
        z = np.array(data.z)
        x = np.array(data.x)
        y = np.array(data.y)
        z = z.reshape(len(x), len(y))
        return RectBivariateSpline(x, y, z, kx=2, ky=2)
    except Exception as e:
        warnings.warn(f"خطا در بارگذاری {filepath}: {e}")
        return None


def _load_ice_map(filepath: str):
    """
    بارگذاری نقشه موتور با فرمت M1DTable (rpm, bmep → مقدار).
    خروجی: LinearNDInterpolator + NearestNDInterpolator
    """
    if not HAS_SCIPY:
        return None, None
    if not os.path.exists(filepath):
        return None, None
    try:
        points, values = [], []
        with open(filepath, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    rpm = float(parts[0])
                    n   = int(parts[1])
                    for _ in range(n):
                        i += 1
                        if i >= len(lines):
                            break
                        dp = lines[i].strip().split()
                        if len(dp) >= 2:
                            points.append([rpm, float(dp[0])])
                            values.append(float(dp[1]))
                except ValueError:
                    pass
            i += 1

        if len(points) > 3:
            pts = np.array(points)
            vals = np.array(values)
            lin = LinearNDInterpolator(pts, vals)
            nn  = NearestNDInterpolator(pts, vals)
            return lin, nn
    except Exception as e:
        warnings.warn(f"خطا در بارگذاری ICE map {filepath}: {e}")
    return None, None


def _query_map(lin, nn, rpm, bmep) -> float:
    """پرسش از نقشه موتور با fallback به nearest."""
    if lin is None:
        return 0.0
    val = lin(rpm, bmep)
    if np.isnan(val):
        val = nn(rpm, bmep)
    return float(val) if not np.isnan(val) else 0.0


# =============================================================================
# ICE Model — Adapter برای RL
# =============================================================================

class ICEModelRL:
    """
    مدل موتور احتراقی — سازگار با RL.

    ویژگی‌ها:
    • نقشه‌های واقعی از data_engine/*.data
    • محاسبه BSFC و fuel_rate از conshot_1.data
    • مدل حرارتی ساده (scalar، بدون آرایه)
    • reset() در O(1) — مناسب برای هزاران episode

    استفاده در PHEVEnv_v8:
        self.ice_model_rl = ICEModelRL(data_folder="data_engine")
        ...
        result = self.ice_model_rl.step_rl(P_ice_kw, rpm, T_eng, dt)
        fuel_g   = result["fuel_g"]
        T_eng_new = result["T_eng"]
    """

    def __init__(
        self,
        data_folder: str = "data_engine",
        displacement_L: float = 1.8,
        num_cylinders: int = 4,
        max_power_kw: float = 80.0,
        max_rpm: float = 6000.0,
        min_rpm: float = 1000.0,
        idle_rpm: float = 800.0,
        optimal_rpm: float = 2500.0,
        optimal_power_kw: float = 35.0,
        Tamb: float = 25.0,
    ):
        self.data_folder     = data_folder
        self.displacement_L  = displacement_L
        self.V_d             = displacement_L / 1000.0   # m³
        self.num_cylinders   = num_cylinders
        self.max_power_kw    = max_power_kw
        self.max_rpm         = max_rpm
        self.min_rpm         = min_rpm
        self.idle_rpm        = idle_rpm
        self.optimal_rpm     = optimal_rpm
        self.optimal_power_kw = optimal_power_kw
        self.Tamb            = Tamb

        # ثوابت سوخت (بنزین)
        self.fuel_lhv    = 44.0e6   # J/kg
        self.co2_factor  = 2.31     # kg CO2 / kg fuel

        # بارگذاری یک‌بار (cache)
        self._load_maps()

        # state (scalar — بدون آرایه)
        self.T_eng = Tamb

    # ─── بارگذاری نقشه‌ها ────────────────────────────────────────────────────

    def _load_maps(self):
        """بارگذاری نقشه‌های موتور — از cache اگر قبلاً بارگذاری شده."""
        global _ICE_CACHE
        key = os.path.abspath(self.data_folder)

        if key not in _ICE_CACHE:
            cache = {}
            print(f"  [ICEModelRL] بارگذاری نقشه‌های موتور از {self.data_folder}")

            for name, fname in [
                ("fuel",    "conshot_1.data"),
                ("torque_max", "torque_1.data"),   # فقط برای حد بالا
                ("fmep",    "FMEP_1.data"),
                ("nox",     "NOxhot_1.data"),
                ("co",      "COhot_1.data"),
                ("hc",      "HChot_1.data"),
                ("co2_map", "CO2hot_1.data"),
            ]:
                filepath = os.path.join(self.data_folder, fname)
                if name in ("fmep",):
                    # فایل 1D: rpm → fmep
                    cache[name] = self._load_1d(filepath)
                else:
                    lin, nn = _load_ice_map(filepath)
                    cache[name]    = lin
                    cache[name+"_nn"] = nn

            _ICE_CACHE[key] = cache
            loaded = [k for k in cache if cache[k] is not None and not k.endswith("_nn")]
            print(f"  [ICEModelRL] نقشه‌های بارگذاری‌شده: {loaded}")

        self._maps = _ICE_CACHE[key]

    def _load_1d(self, filepath: str):
        """بارگذاری جدول 1D ساده."""
        if not os.path.exists(filepath) or not HAS_SCIPY:
            return None
        try:
            xs, ys = [], []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            xs.append(float(parts[0]))
                            ys.append(float(parts[1]))
                        except ValueError:
                            continue
            if len(xs) > 1:
                return interp1d(xs, ys, fill_value="extrapolate")
        except Exception:
            pass
        return None

    # ─── محاسبات اصلی ────────────────────────────────────────────────────────

    def power_to_rpm_torque(
        self, P_kw: float, mode: str = "series"
    ) -> Tuple[float, float, float]:
        """
        تبدیل توان به (rpm, torque, bmep).

        mode='series' : RPM بهینه (decoupled)
        mode='coupled': RPM از سرعت چرخ (در PHEVEnv_v8 از بیرون داده می‌شود)
        """
        if P_kw < 0.1:
            return self.idle_rpm, 0.0, 0.0

        if mode == "series":
            rpm = self.optimal_rpm
        else:
            rpm = float(np.clip(P_kw / self.optimal_power_kw * self.optimal_rpm,
                                self.min_rpm, self.max_rpm))

        # T = P*9549/rpm
        torque = float(np.clip(P_kw * 9549.0 / max(rpm, 1.0), 0.0, 250.0))

        # bmep = T * 4π / V_d (bar)
        bmep = torque * 4 * np.pi / (self.V_d * 1e5)
        bmep = float(np.clip(bmep, 0.0, 20.0))

        return rpm, torque, bmep

    def get_fuel_rate_gs(
        self, P_kw: float, rpm: float, bmep: float, T_eng: float
    ) -> float:
        """
        نرخ مصرف سوخت [g/s] از نقشه واقعی.

        نقشه conshot_1.data: (rpm, bmep) → مصرف سوخت [g/h]
        با cold start correction.
        """
        if P_kw < 0.1 or rpm < self.idle_rpm:
            return 0.0

        # ── از نقشه واقعی ──────────────────────────────────────────────────
        fuel_lin = self._maps.get("fuel")
        fuel_nn  = self._maps.get("fuel_nn")

        if fuel_lin is not None:
            fuel_g_h = _query_map(fuel_lin, fuel_nn, rpm, bmep)
            # [FIX v10] اگر نقشه NaN یا صفر برگرداند (خارج از محدوده)
            # به مدل BSFC تجربی fallback می‌کنیم
            if fuel_g_h is None or fuel_g_h <= 0 or np.isnan(fuel_g_h):
                deviation = abs(P_kw - self.optimal_power_kw)
                bsfc_val  = 260.0 + 1.5 * deviation
                fuel_g_s  = bsfc_val * P_kw / 3600.0
            else:
                fuel_g_s = fuel_g_h / 3600.0
        else:
            # fallback: مدل تجربی BSFC
            deviation = abs(P_kw - self.optimal_power_kw)
            bsfc      = 260.0 + 1.5 * deviation   # g/kWh
            fuel_g_s  = bsfc * P_kw / 3600.0

        # ── Cold start correction ──────────────────────────────────────────
        # موتور سرد → مصرف تا 30% بیشتر
        warmup  = float(np.clip((T_eng - self.Tamb) / (80.0 - self.Tamb), 0.0, 1.0))
        cs_factor = 1.0 + 0.30 * (1.0 - warmup)
        fuel_g_s *= cs_factor

        return max(fuel_g_s, 0.0)

    def get_fmep(self, rpm: float) -> float:
        """FMEP [bar] — از نقشه یا مدل Chen-Flynn."""
        fmep_fn = self._maps.get("fmep")
        if fmep_fn is not None:
            try:
                return float(np.clip(fmep_fn(rpm), 0.3, 2.5))
            except Exception:
                pass
        # Chen-Flynn fallback
        rpm_k = rpm / 1000.0
        return float(np.clip(0.4 + 0.005 * rpm_k + 0.001 * rpm_k**2, 0.3, 2.0))

    def _update_T_eng(self, P_kw: float, dt: float) -> float:
        """مدل حرارتی ساده (scalar) — بدون آرایه."""
        if P_kw < 0.5:
            dT = -0.014 * (self.T_eng - self.Tamb) * dt
        else:
            dT = (0.004 * P_kw - 0.014 * (self.T_eng - 90.0)) * dt
        self.T_eng = float(np.clip(self.T_eng + dT, self.Tamb - 2, 125.0))
        return self.T_eng

    # ─── رابط RL ─────────────────────────────────────────────────────────────

    def step_rl(
        self,
        P_ice_kw: float,
        rpm: float = None,
        T_eng: float = None,
        dt: float = 1.0,
        mode: str = "series",
    ) -> Dict:
        """
        یک گام محاسبه موتور — سازگار با PHEVEnv_v8.

        [FIX v10] برای مد coupled (PARALLEL):
          rpm از سرعت چرخ محاسبه می‌شود اما ممکن است خارج از محدوده نقشه باشد.
          در این حالت، rpm به محدوده بهینه موتور clamp می‌شود تا از NaN جلوگیری شود.
        """
        if T_eng is not None:
            self.T_eng = T_eng

        if P_ice_kw < 0.1:
            self._update_T_eng(0.0, dt)
            return {
                "fuel_g":    0.0,
                "T_eng":     self.T_eng,
                "rpm":       0.0,
                "bmep":      0.0,
                "bsfc":      0.0,
                "efficiency": 0.0,
            }

        # RPM و BMEP
        if rpm is None or rpm < self.min_rpm:
            rpm, torque, bmep = self.power_to_rpm_torque(P_ice_kw, mode)
        else:
            # [FIX v10] clamp rpm به محدوده عملی موتور
            # rpm بیش از max_rpm در coupled mode خارج از نقشه است → NaN
            rpm    = float(np.clip(rpm, self.min_rpm, self.max_rpm))
            torque = P_ice_kw * 9549.0 / max(rpm, 1.0)
            torque = float(np.clip(torque, 0.0, 250.0))
            bmep   = torque * 4 * np.pi / (self.V_d * 1e5)
            bmep   = float(np.clip(bmep, 0.0, 20.0))

        # مصرف سوخت
        fuel_g_s = self.get_fuel_rate_gs(P_ice_kw, rpm, bmep, self.T_eng)
        fuel_g   = fuel_g_s * dt

        # BSFC و بازده
        bsfc = (fuel_g_s * 3600.0) / max(P_ice_kw, 0.1)
        bsfc = float(np.clip(bsfc, 150.0, 800.0))
        efficiency = 3.6e6 / (bsfc * self.fuel_lhv / 1000.0)
        efficiency = float(np.clip(efficiency, 0.0, 0.45))

        # به‌روزرسانی دمای موتور
        self._update_T_eng(P_ice_kw, dt)

        return {
            "fuel_g":     fuel_g,
            "T_eng":      self.T_eng,
            "rpm":        rpm,
            "bmep":       bmep,
            "bsfc":       bsfc,
            "efficiency": efficiency,
        }

    def reset(self, Tamb: float = None):
        """ریست سریع — فقط scalar state."""
        if Tamb is not None:
            self.Tamb = Tamb
        self.T_eng = self.Tamb


# =============================================================================
# Battery Model — Adapter برای RL
# =============================================================================

class BatteryModelRL:
    """
    مدل باتری LFP — سازگار با RL.

    ویژگی‌ها:
    • OCV(T, SOC) از فایل‌های داده AMESim (OCV.data / Rdch.data)
    • مدل OCV-R لحظه‌ای (بدون آرایه تاریخچه)
    • مدل حرارتی scalar
    • reset() در O(1)

    استفاده در PHEVEnv_v8:
        self.batt_model_rl = BatteryModelRL(
            data_dir="utiles",
            ns=96, npar=3, C_nominal_Ah=50.0, V_nominal=350.0
        )
        ...
        result = self.batt_model_rl.step_rl(P_batt_kw, T_batt, dt)
        soc_new   = result["soc"]
        T_batt_new = result["T_batt"]
        current_A  = result["current_A"]
    """

    def __init__(
        self,
        data_dir: str = "utiles",
        ns: int = 96,
        npar: int = 3,
        C_nominal_Ah: float = 50.0,     # ظرفیت پک (Ah)
        V_nominal: float = 350.0,        # ولتاژ نامی پک (V)
        soc_init: float = 0.50,          # SOC اولیه [0,1]
        Tamb: float = 25.0,
        # پارامترهای حرارتی (scalar)
        batt_mass_kg: float = 150.0,
        batt_Cp: float = 1000.0,         # J/kg.K
        thermal_resistance: float = 0.5, # K/W
        # محدودیت‌ها
        V_min_cell: float = 2.5,
        V_max_cell: float = 4.2,
        I_max_A: float = 200.0,
    ):
        self.data_dir     = data_dir
        self.ns           = ns
        self.npar         = npar
        self.C_pack       = C_nominal_Ah * npar   # ظرفیت کل پک (Ah)
        self.V_nominal    = V_nominal
        self.soc_init     = soc_init
        self.Tamb         = Tamb
        self.batt_mass    = batt_mass_kg
        self.batt_Cp      = batt_Cp
        self.R_thermal    = thermal_resistance
        self.V_min_pack   = V_min_cell * ns
        self.V_max_pack   = V_max_cell * ns
        self.I_max        = I_max_A

        # بارگذاری یک‌بار
        self._load_maps()

        # state scalar
        self.soc   = soc_init          # [0, 1]
        self.T_batt = Tamb             # °C
        self.qloss  = 0.0              # کاهش ظرفیت تجمعی

    # ─── بارگذاری ─────────────────────────────────────────────────────────────

    def _load_maps(self):
        global _BATTERY_CACHE
        key = os.path.abspath(self.data_dir)

        if key not in _BATTERY_CACHE:
            cache = {}
            print(f"  [BatteryModelRL] بارگذاری نقشه‌های باتری از {self.data_dir}")

            for name, fname in [("ocv", "OCV.data"), ("res", "Rdch.data")]:
                fpath = os.path.join(self.data_dir, fname)
                spline = _load_2d_table(fpath)
                cache[name] = spline
                status = "✓" if spline is not None else "✗ (fallback)"
                print(f"    {status} {name}: {fname}")

            _BATTERY_CACHE[key] = cache

        self._maps = _BATTERY_CACHE[key]

    # ─── OCV و مقاومت از نقشه ────────────────────────────────────────────────

    def get_ocv(self, soc_pct: float, T: float) -> float:
        """
        OCV پک [V] از نقشه AMESim.

        OCV.data: 2DTable → محور X: دما [°C], محور Y: SOC [0..100]
        مقدار: ولتاژ یک سلول [V]
        """
        T_c   = float(np.clip(T, -20, 60))
        s_pct = float(np.clip(soc_pct * 100.0, 0.0, 100.0))

        spline = self._maps.get("ocv")
        if spline is not None:
            try:
                v_cell = float(spline(T_c, s_pct)[0][0])
            except Exception:
                v_cell = self._default_ocv_cell(s_pct)
        else:
            v_cell = self._default_ocv_cell(s_pct)

        v_pack = float(np.clip(self.ns * v_cell, self.V_min_pack, self.V_max_pack))
        return max(v_pack, 10.0)

    def get_resistance(self, soc_pct: float, T: float) -> float:
        """
        مقاومت داخلی پک [Ω] از نقشه AMESim.

        Rdch.data: 2DTable → همان ابعاد OCV
        مقدار: مقاومت یک سلول [Ω]
        """
        T_c   = float(np.clip(T, -20, 60))
        s_pct = float(np.clip(soc_pct * 100.0, 0.0, 100.0))

        spline = self._maps.get("res")
        if spline is not None:
            try:
                r_cell = float(spline(T_c, s_pct)[0][0])
            except Exception:
                r_cell = 0.002
        else:
            r_cell = 0.002 * (1.0 + 0.5 * (1.0 - soc_pct))

        # مقاومت پک: R_pack = ns/npar * R_cell
        R_pack = self.ns / max(self.npar, 1) * r_cell
        return max(R_pack, 1e-4)

    @staticmethod
    def _default_ocv_cell(soc_pct: float) -> float:
        """منحنی OCV پیش‌فرض LFP [V/cell]."""
        # نقاط مرجع LFP-C (LiFePO4)
        soc_pts = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
        ocv_pts = np.array([2.8, 3.1, 3.13, 3.20, 3.22, 3.23, 3.25,
                            3.27, 3.30, 3.33, 3.36, 3.40, 3.65])
        return float(np.interp(soc_pct, soc_pts, ocv_pts))

    # ─── محاسبه جریان از توان ────────────────────────────────────────────────

    def power_to_current(self, P_kw: float) -> float:
        """
        حل معادله درجه‌دو: P = V_oc*I - R*I²
        انتخاب ریشه با |I| کمتر.
        """
        V_oc = self.get_ocv(self.soc, self.T_batt)
        R    = self.get_resistance(self.soc, self.T_batt)
        P_w  = P_kw * 1000.0

        disc = V_oc**2 - 4 * R * P_w
        if disc < 0:
            return P_w / max(V_oc, 1.0)

        I1 = (V_oc + np.sqrt(disc)) / (2 * R)
        I2 = (V_oc - np.sqrt(disc)) / (2 * R)
        I  = I1 if abs(I1) < abs(I2) else I2
        return float(np.clip(I, -self.I_max, self.I_max))

    # ─── مدل حرارتی باتری (scalar) ───────────────────────────────────────────

    def _update_T_batt(self, current_A: float, dt: float) -> float:
        """
        دمای باتری با مدل اهمی + خنک‌کن طبیعی.
        Q_gen = R * I²
        Q_loss = (T - T_amb) / R_thermal
        """
        R     = self.get_resistance(self.soc, self.T_batt)
        Q_gen = R * current_A**2                         # W
        Q_loss = (self.T_batt - self.Tamb) / self.R_thermal  # W
        dT    = (Q_gen - Q_loss) * dt / (self.batt_mass * self.batt_Cp)
        self.T_batt = float(np.clip(self.T_batt + dT, self.Tamb - 5, 70.0))
        return self.T_batt

    # ─── رابط RL ─────────────────────────────────────────────────────────────

    def step_rl(
        self,
        P_batt_kw: float,
        T_batt: float = None,
        dt: float = 1.0,
    ) -> Dict:
        """
        یک گام شبیه‌سازی باتری — سازگار با PHEVEnv_v8.

        Args:
            P_batt_kw : توان الکتریکی (kW) — مثبت=دشارژ، منفی=شارژ
            T_batt    : دمای باتری (اختیاری — اگر None، از self.T_batt)
            dt        : گام زمانی (s)

        Returns:
            {soc, T_batt, current_A, voltage, heat_W, qloss_total}
        """
        if T_batt is not None:
            self.T_batt = T_batt

        # جریان
        current_A = self.power_to_current(P_batt_kw)

        # SOC جدید
        # delta_SOC = -I * dt / (C_pack * 3600)
        C_eff = self.C_pack * (1.0 - self.qloss)   # ظرفیت با پیرشدگی
        delta_soc = -(current_A * dt) / (C_eff * 3600.0)
        self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))

        # دما
        self._update_T_batt(current_A, dt)

        # گرمای تولیدی
        R     = self.get_resistance(self.soc, self.T_batt)
        Q_gen = R * current_A**2

        # پیرشدگی ساده (alpha * C_rate)
        C_rate     = abs(current_A) / max(C_eff, 1e-6)
        alpha_aging = 1.0e-5
        self.qloss += alpha_aging * C_rate * dt

        # ولتاژ ترمینال
        V_oc  = self.get_ocv(self.soc, self.T_batt)
        V_term = V_oc - current_A * R

        return {
            "soc":        self.soc,
            "T_batt":     self.T_batt,
            "current_A":  current_A,
            "voltage":    float(np.clip(V_term, self.V_min_pack, self.V_max_pack)),
            "heat_W":     Q_gen,
            "qloss_total": self.qloss,
        }

    def reset(self, soc_init: float = None, Tamb: float = None):
        """ریست سریع — فقط scalar state."""
        if soc_init is not None:
            self.soc_init = soc_init
        if Tamb is not None:
            self.Tamb = Tamb
        self.soc    = self.soc_init
        self.T_batt = self.Tamb
        self.qloss  = 0.0


# =============================================================================
# تابع کمکی برای PHEVEnv_v8
# =============================================================================

def build_physics_models(
    ice_data_folder: str = "data_engine",
    batt_data_dir:   str = "utiles",
    Tamb:            float = 25.0,
    verbose:         bool = True,
) -> Tuple["ICEModelRL", "BatteryModelRL"]:
    """
    ساخت هر دو مدل فیزیکی در یک فراخوانی.

    مثال استفاده در PHEVEnv_v8.__init__:
        from PhysicsModels_v8 import build_physics_models
        self.ice_model_rl, self.batt_model_rl = build_physics_models(
            ice_data_folder="data_engine",
            batt_data_dir="utiles",
        )
    """
    if verbose:
        print("\n[PhysicsModels] بارگذاری مدل‌های فیزیکی...")

    ice  = ICEModelRL(data_folder=ice_data_folder, Tamb=Tamb)
    batt = BatteryModelRL(data_dir=batt_data_dir,   Tamb=Tamb)

    if verbose:
        print("[PhysicsModels] آماده.\n")

    return ice, batt
