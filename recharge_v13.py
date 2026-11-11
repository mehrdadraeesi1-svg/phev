"""Plant-v3 replenishment cost k [g fuel per kWh of nominal pack energy]:
standstill SERIES recharge (engine 35 kW @ 2500 rpm, warm) from SOC_f to SOC_ref."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import PHEVEnv_v13 as E

def recharge(soc_f, soc_ref=0.5, T_eng0=90.0, T_batt0=30.0):
    cyc = np.array([[t, 0.0, 0.0] for t in range(10)], float)
    e = E.PHEVMultiModeEnv([cyc]); e.reset(options={"soc_init": soc_f})
    e.T_eng, e.T_batt = T_eng0, T_batt0
    fuel, n = 0.0, 0
    while e.soc < soc_ref and n < 10000:
        ph = e._apply_physics(E.DriveMode.SERIES, 0.5, 0.0, 0.0)
        e._update_state(ph, E.DriveMode.SERIES, 0.0); fuel += ph["fuel_g"]; n += 1
    dE = (e.soc - soc_f) * e.health.E_nominal_kwh
    return fuel / dE, n

if __name__ == "__main__":
    for sf in (0.30, 0.40, 0.45, 0.48):
        for ref in (0.5, 0.7):
            if sf < ref:
                k, n = recharge(sf, ref); print(f"SOC {sf:.2f}->{ref:.2f}: k={k:6.1f} g/kWh  ({n} s)")
