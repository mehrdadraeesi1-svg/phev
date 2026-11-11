"""Graphical abstract for Applied Energy (1328 x 531 px aspect, readable at 13 x 5 cm)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
C=dict(blue="#2a78d6",orange="#eb6834",aqua="#1baf7a",yellow="#eda100",grey="#8a8985",violet="#4a3aa7",red="#e34948")
INK="#0b0b0b"; INK2="#52514e"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,
 "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(13.28/2.54*2,5.31/2.54*2))   # 26.56 x 10.62 cm -> 2.5:1
fig.text(0.5,0.955,"SAC energy management of a multi-mode PHEV: what matters, and how far from the DP bound?",
         ha="center",va="top",fontsize=12.5,fontweight="bold",color=INK)
# ---- panel 1: implied equivalence factor
ax1=fig.add_axes([0.075,0.17,0.24,0.62])
names=["Sep","EFC","EFC$_{120}$","MA"]; dis=[348,693,484,269]; chg=[-280,-353,-144,269]
x=range(4)
ax1.bar([i-0.18 for i in x],dis,0.36,color=C["orange"],label="SOC below target")
ax1.bar([i+0.18 for i in x],chg,0.36,color=C["blue"],label="SOC above target")
ax1.axhline(269,color=INK,lw=1.2,ls="--"); ax1.text(3.55,300,"plant cost\n269 g/kWh",fontsize=7.5,ha="right",va="bottom")
ax1.axhline(0,color=INK2,lw=0.6)
ax1.set_xticks(list(x)); ax1.set_xticklabels(names); ax1.set_ylabel("Implied price of\nbattery energy (g/kWh)")
ax1.set_title("1  Reward = implicit price",fontsize=10,fontweight="bold",loc="left",pad=14)
ax1.legend(fontsize=7.5,frameon=False,loc="upper right",ncol=1); ax1.set_ylim(-420,1000)
# ---- panel 2: variance explained
ax2=fig.add_axes([0.39,0.17,0.2,0.62])
lab=["Reward","Action\nmapping","Interaction"]; v50=[33.5,10.3,11.7]; v70=[53.7,7.2,5.0]
ax2.bar([i-0.18 for i in range(3)],v50,0.36,color=C["violet"],label="SOC$_0$ 50 %")
ax2.bar([i+0.18 for i in range(3)],v70,0.36,color=C["aqua"],label="SOC$_0$ 70 %")
ax2.text(0,58,"p ≤ 0.034",ha="center",fontsize=8,fontweight="bold"); ax2.text(1,15,"n.s.",ha="center",fontsize=8); ax2.text(2,15,"n.s.",ha="center",fontsize=8)
ax2.set_xticks(range(3)); ax2.set_xticklabels(lab,fontsize=8.5); ax2.set_ylabel("Share of fuel variance (%)"); ax2.set_ylim(0,66)
ax2.set_title("2  Reward dominates",fontsize=10,fontweight="bold",loc="left",pad=14); ax2.legend(fontsize=7.5,frameon=False,loc="upper right")
# ---- panel 3: optimality gap
ax3=fig.add_axes([0.72,0.17,0.26,0.62])
ctl=["DP policy on plant","DP-guided, 10-s dwell","Adaptive ECMS","Rule-based","Best SAC (G/MA)","Threshold SAC"]
gap=[1.3,4.8,6.2,8.8,9.5,13.9]; col=[C["grey"],C["grey"],C["blue"],C["yellow"],C["orange"],C["red"]]
y=list(range(len(ctl)))[::-1]
ax3.barh(y,gap,color=col,height=0.62)
for yi,g in zip(y,gap): ax3.text(g+0.25,yi,f"+{g:.1f} %",va="center",fontsize=8)
ax3.set_yticks(y); ax3.set_yticklabels(ctl,fontsize=8.5); ax3.set_xlabel("Fuel above relaxed DP bound (%)"); ax3.set_xlim(0,17)
ax3.set_title("3  Gap to the relaxed DP bound",fontsize=10,fontweight="bold",loc="left",pad=14)
fig.text(0.5,0.035,"Same plant, supervisor and switching constraint for all controllers  ·  8 cycles  ·  SAC converges to series charging at cruise; A-ECMS and DP use direct drive",ha="center",fontsize=8,color=INK2)
for ext in ("pdf","png","tiff"):
    fig.savefig(f"figs/graphical_abstract.{ext}",dpi=300 if ext!="pdf" else None)
print("ok")
