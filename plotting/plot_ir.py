"""
plot_ir.py

Computed IR spectrum of Bi2Al4O9 (HSE06, harmonic).

Upper panel : spectrum broadened with Lorentzian functions
Lower panel : stick spectrum, each mode coloured by the structural unit
              that carries most of the vibration (from mode_character.dat)

All settings are in the SETTINGS section below. Run with:

    python plot_ir.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
#  SETTINGS
# =============================================================================

# --- input -------------------------------------------------------------------

# mode data written by cryfreq.py (full file or the IR-active subset)
MODE_FILE = "mode_character.dat"            

# modes with a smaller IR intensity (km/mol) are ignored
MIN_INTENSITY = 0.01


# --- spectrum ----------------------------------------------------------------

FWHM = 10.0                     # Lorentzian full width at half maximum / cm-1
X_MIN, X_MAX = 75.0, 1050.0     # plotted wavenumber range / cm-1
X_STEP = 0.5                    # grid spacing / cm-1
REVERSED_AXIS = True            # True: high wavenumbers on the left (IR convention)
STICK_YMAX = 4200.0             # upper limit of the stick panel / (km/mol)


# --- labels in the upper panel: (position / cm-1, text) ----------------------

ANNOTATIONS = [
    (943, "ν$_{as}$(Al–O–Al)\n940/946"),
    (768, "768"),
    (707, "AlO$_4$\n707"),
    (496, "AlO$_6$\n483–537"),
    (368, "368"),
    (137, "T–O–T\nbend 137"),
]


# --- structural units used for the colours -----------------------------------
# name in the legend, colour, atom sets (column names in MODE_FILE) that belong to it

UNITS = [
    ("Bi lattice modes",                      "#7f7f7f", ["Bi5"]),
    ("AlO$_6$ octahedra (Al$_{oct}$, O13)",    "#2e86c1", ["Al1", "O13"]),
    ("AlO$_4$ tetrahedra (Al$_{tet}$, O17, O21)", "#d68910", ["Al9", "O17", "O21"]),
    ("T–O–T bridge (O29)",                    "#c0392b", ["O29"]),
]


# --- output ------------------------------------------------------------------

TITLE = "Bi$_2$Al$_4$O$_9$, HSE06 harmonic IR spectrum"
OUTPUT_FILES = ["Bi2Al4O9_IR_HSE06.png", "Bi2Al4O9_IR_HSE06.pdf"]
FIGURE_SIZE = (7.2, 4.6)        # inch
DPI = 300
FONT_SIZE = 9


# =============================================================================
#  FUNCTIONS
# =============================================================================

def read_modes(file_name, min_intensity):
    """
    Read mode_character.dat from cryfreq.py.

    Returns wavenumbers (cm-1), irreps, intensities (km/mol),
    the shares of every atom set (%) and the names of the atom sets.
    Only modes with an intensity above min_intensity are kept.
    """
    wavenumbers, irreps, intensities, shares = [], [], [], []
    set_names = None

    for line in open(file_name):

        if line.startswith("#"):
            if "freq" in line and "irrep" in line:
                set_names = line.split()[5:]
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        intensity = float(parts[3])
        if not np.isfinite(intensity) or intensity < min_intensity:
            continue

        wavenumbers.append(float(parts[1]))
        irreps.append(parts[2])
        intensities.append(intensity)
        shares.append([float(x) for x in parts[4:]])

    return (np.array(wavenumbers), irreps, np.array(intensities),
            np.array(shares), set_names)


def dominant_unit(shares, set_names, units):
    """Index of the structural unit with the largest summed share, for every mode."""
    column = {name: i for i, name in enumerate(set_names)}

    unit_shares = np.zeros((len(shares), len(units)))
    for k, (_, _, atom_sets) in enumerate(units):
        for atom_set in atom_sets:
            unit_shares[:, k] += shares[:, column[atom_set]]

    return np.argmax(unit_shares, axis=1)


def lorentzian_spectrum(x, wavenumbers, intensities, fwhm):
    """Sum of area-normalised Lorentzians, one per mode."""
    gamma = fwhm / 2.0
    spectrum = np.zeros_like(x)

    for position, intensity in zip(wavenumbers, intensities):
        spectrum += intensity * gamma / np.pi / ((x - position) ** 2 + gamma ** 2)

    return spectrum


# =============================================================================
#  MAIN
# =============================================================================

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_SIZE})


# --- data --------------------------------------------------------------------

wavenumbers, irreps, intensities, shares, set_names = read_modes(MODE_FILE, MIN_INTENSITY)
unit_index = dominant_unit(shares, set_names, UNITS)

x = np.arange(X_MIN, X_MAX + X_STEP, X_STEP)
spectrum = lorentzian_spectrum(x, wavenumbers, intensities, FWHM)
spectrum = spectrum / spectrum.max()


# --- figure ------------------------------------------------------------------

fig, (ax_spec, ax_stick) = plt.subplots(
    2, 1, figsize=FIGURE_SIZE, sharex=True,
    gridspec_kw=dict(height_ratios=[1.3, 1.0], hspace=0.05),
)


# upper panel: broadened spectrum

ax_spec.plot(x, spectrum, color="black", lw=1.0)

ax_spec.set_ylabel("absorbance / norm.")
ax_spec.set_ylim(0, 1.3)
ax_spec.set_title(f"{TITLE} (Lorentzian FWHM {FWHM:g} cm$^{{-1}}$)", fontsize=FONT_SIZE)

for position, text in ANNOTATIONS:
    y = spectrum[np.argmin(abs(x - position))]
    ax_spec.annotate(
        text, xy=(position, y), xytext=(position, min(y + 0.22, 1.08)),
        ha="center", fontsize=FONT_SIZE - 2,
        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"),
    )


# lower panel: sticks coloured by structural unit

for k, (label, colour, _) in enumerate(UNITS):
    selected = (unit_index == k) & (wavenumbers >= X_MIN) & (wavenumbers <= X_MAX)
    ax_stick.vlines(wavenumbers[selected], 0, intensities[selected],
                    color=colour, lw=1.6, label=label)

ax_stick.set_ylabel(r"IR intensity / (km mol$^{-1}$)")
ax_stick.set_xlabel(r"wavenumber / cm$^{-1}$")
ax_stick.set_ylim(0, STICK_YMAX)
ax_stick.legend(fontsize=FONT_SIZE - 2, frameon=False, loc="upper right")

if REVERSED_AXIS:
    ax_stick.set_xlim(X_MAX, X_MIN)
else:
    ax_stick.set_xlim(X_MIN, X_MAX)


# --- save --------------------------------------------------------------------

for file_name in OUTPUT_FILES:
    fig.savefig(file_name, dpi=DPI, bbox_inches="tight")
    print("written:", file_name)
