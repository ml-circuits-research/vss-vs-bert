"""power_energy.py (responses R12, Q8 d): integrates logs/power_<label>.csv (one sample per second from eval/power_sampler.sh) into GPU
energy: trapezoidal rule over the timestamps, in Wh, with the mean and peak power and the sampled duration. The power is that of the
whole GPU of this machine (GB10), idle draw included; the idle draw before the run is printed so that it can be subtracted.
Run: python3 eval/power_energy.py logs/power_<label>.csv"""
import csv, sys
from datetime import datetime

def energy(path):
    t, w = [], []
    for row in csv.DictReader(open(path)):
        try: t.append(datetime.strptime(row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()); w.append(float(row["watts"]))
        except (ValueError, KeyError): continue
    wh = sum((t[i + 1] - t[i]) * (w[i] + w[i + 1]) / 2 for i in range(len(t) - 1)) / 3600.0
    return {"samples": len(t), "seconds": (t[-1] - t[0]) if t else 0.0, "energy_Wh": wh, "mean_W": sum(w) / max(1, len(w)), "peak_W": max(w) if w else 0.0, "first_W": w[0] if w else None}

if __name__ == "__main__":
    for p in sys.argv[1:]:
        e = energy(p); print(f"{p}: {e['samples']} samples over {e['seconds']:.0f} s; GPU energy {e['energy_Wh']:.1f} Wh; mean {e['mean_W']:.1f} W, peak {e['peak_W']:.1f} W; first sample {e['first_W']} W (this machine, GB10, fp32)")
