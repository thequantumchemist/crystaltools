"""
make_banner.py - banner for the crystaltools repository (1536 x 1024 px, style of the OVERTONE banner)
All settings in the SETTINGS section. Run with:  python make_banner.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# =============================================================================
#  SETTINGS
# =============================================================================
STRUCTURE_IMAGE = "structure.png"          # structure picture with transparent background
HEADLINE = ["Distributed harmonic frequencies",
            "and IR spectra of crystals"]
SUBTITLE = "scripts for CRYSTAL23 and phonopy"
KEYWORDS = "finite differences  ·  Berry-phase Born charges  ·  IR intensities  ·  free energies"
OUTPUT = "crystaltools_banner.png"

# colour scheme: graphite background, emerald -> lime -> amber accents
BG_TOP, BG_BOTTOM = "#0e1013", "#1b1f26"
GLOW_COLOURS = [("#10b981", 1150, 180, 520, 0.10), ("#f59e0b", 250, 900, 520, 0.05)]  # colour, x, y, radius, strength
ACCENTS = ["#34d399", "#a3e635", "#facc15", "#f59e0b"]
EQUATION_COLOUR = "#3b434f"
TEXT_SUB, TEXT_KEY = "#d1d5db", "#6ee7b7"
TASK_COLOURS = [("#34d399", 0.85), ("#f59e0b", 0.95), ("#374151", 0.7)]   # finished, running, waiting
WIDTH, HEIGHT = 1536, 1024             # px
FONT = "DejaVu Sans"

# =============================================================================
#  MAIN
# =============================================================================
plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "dejavusans"})
fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.axis("off")

# --- background: graphite gradient with soft coloured glows -----------------------------
from matplotlib.colors import to_rgb
yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
t = (yy / HEIGHT)[..., None]
base = np.array(to_rgb(BG_TOP)) * (1 - t) + np.array(to_rgb(BG_BOTTOM)) * t
for colour, gx, gy, r, s in GLOW_COLOURS:
    g = np.exp(-(((xx - gx) / r) ** 2 + ((yy - gy) / (0.7 * r)) ** 2))[..., None]
    base = base + s * g * np.array(to_rgb(colour))
ax.imshow(np.clip(base, 0, 1), extent=[0, WIDTH, 0, HEIGHT], origin="upper", zorder=0)

# --- faint equations ------------------------------------------------------------
eq_style = dict(color=EQUATION_COLOUR, alpha=0.9, fontsize=19, zorder=1)
ax.text(50, 960, r"$D_{i\alpha,j\beta}=\Phi_{i\alpha,j\beta}/\sqrt{m_i m_j}$", **eq_style)
ax.text(640, 940, r"$Z^{*}_{i,\alpha\beta}=\frac{\Omega}{e}\,\frac{\partial P_\alpha}{\partial u_{i\beta}}$", **eq_style)
ax.text(1060, 760, r"$I_n\propto\left|\sum_i Z^{*}_i\,\mathbf{e}_{n,i}/\sqrt{m_i}\right|^2$", **eq_style)
ax.text(40, 55, r"$F_{\rm vib}=\sum_n\left[\frac{\hbar\omega_n}{2}+k_BT\ln\left(1-e^{-\hbar\omega_n/k_BT}\right)\right]$", **eq_style)
ax.text(960, 55, r"$\Phi_{i\alpha,j\beta}\approx-\frac{F_{i\alpha}(+u_{j\beta})-F_{i\alpha}(-u_{j\beta})}{2u}$", **eq_style)

# --- job-array motif: tasks of a queue (finished, running, waiting) -------------------
rng = np.random.default_rng(3)
for r in range(3):
    for c in range(12):
        x0, y0 = 1080 + c * 34, 900 - r * 34
        state = rng.choice([0, 1, 2], p=[0.55, 0.2, 0.25]) if (c + 12 * r) > 6 else 0
        colour, alpha = TASK_COLOURS[state]
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x0, y0), 24, 24, boxstyle="round,pad=0,rounding_size=5", color=colour, alpha=alpha, lw=0, zorder=2))

# --- structure picture ----------------------------------------------------------------
img = Image.open(STRUCTURE_IMAGE).convert("RGBA")
w_img = 560; h_img = w_img * img.size[1] / img.size[0]
ax.imshow(np.array(img), extent=[60, 60 + w_img, 330, 330 + h_img], zorder=3, interpolation="lanczos")

# --- headline, subtitle, keywords ------------------------------------------------------
for k, line in enumerate(HEADLINE):
    ax.text(655, 650 - k * 58, line, fontsize=30, fontweight="bold", color="white", zorder=6, va="center")
ax.text(657, 515, SUBTITLE, fontsize=21, color=TEXT_SUB, zorder=6, va="center")
ax.text(657, 455, KEYWORDS, fontsize=13.5, color=TEXT_KEY, zorder=6, va="center")

# --- stylised IR spectrum with glowing gradient line ------------------------------------
x = np.linspace(0, WIDTH, 3000)
peaks = [(150, 0.55), (230, 0.35), (330, 0.95), (370, 0.7), (520, 1.0), (560, 0.8), (610, 0.45),
         (760, 0.75), (800, 0.4), (1010, 0.5), (1090, 0.25), (1300, 0.6), (1340, 0.35)]
g = 9.0
y = sum(h * g ** 2 / ((x - p) ** 2 + g ** 2) for p, h in peaks)
y = 110 + 210 * y / y.max()
pts = np.array([x, y]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
cmap = LinearSegmentedColormap.from_list("ct", ACCENTS)
for lw, alpha in [(16, 0.06), (9, 0.12), (4.5, 0.35), (2.2, 1.0)]:
    lc = LineCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha, zorder=4)
    lc.set_array(x[:-1] / WIDTH); ax.add_collection(lc)
for p, h in peaks:
    ax.plot([p, p], [110, 110 + 26 * h], color="white", alpha=0.25, lw=1.0, zorder=4)

fig.savefig(OUTPUT, dpi=100)
print("written:", OUTPUT)
