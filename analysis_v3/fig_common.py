import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C=dict(blue="#2a78d6",orange="#eb6834",aqua="#1baf7a",yellow="#eda100",magenta="#e87ba4",green="#008300",violet="#4a3aa7",red="#e34948")
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.edgecolor":INK2,"axes.labelcolor":INK,
 "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False,"lines.linewidth":2,
 "savefig.dpi":300,"figure.dpi":150,"axes.titlesize":9,"axes.titleweight":"bold"})
def save(fig,name):
    fig.savefig(f"figs/{name}.png",bbox_inches="tight"); fig.savefig(f"figs/{name}.pdf",bbox_inches="tight"); plt.close(fig)
