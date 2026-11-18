"""
plot_ir_overlay.py

Overlay of the computed IR spectrum of Bi2Al4O9 (HSE06, harmonic)
with the measured spectrum of the x = 0 sample.

Upper panel : measured spectrum and broadened computed spectrum
Lower panel : computed stick spectrum, coloured by structural unit

All settings are in the SETTINGS section below. Run with:

    python plot_ir_overlay.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


# =============================================================================
#  SETTINGS
# =============================================================================

# --- input -------------------------------------------------------------------

# mode data written by cryfreq.py (full file or the IR-active subset)
MODE_FILE = "mode_character.dat"            
MIN_INTENSITY = 0.01            # km/mol, weaker modes are ignored

# experimental spectrum: two columns, wavenumber / cm-1 and intensity
EXP_FILE = "measured_spectrum.dat"

# True  : EXP_FILE was digitized from a figure -> remove spikes, smooth lightly
# False : EXP_FILE contains measured data      -> use it unchanged
EXP_IS_DIGITIZED = False


# --- computed spectrum -------------------------------------------------------

FWHM = 30.0                     # Lorentzian full width at half maximum / cm-1
SCALE = 1.0                     # uniform frequency scaling factor (1.0 = unscaled)


# --- plot range --------------------------------------------------------------

X_MIN, X_MAX = 400.0, 1100.0    # cm-1
X_STEP = 0.5                    # cm-1
GAP_LIMIT = 4.0                 # cm-1, larger gaps in the measured data are not connected
STICK_YMAX = 4200.0             # km/mol


# --- labels in the upper panel: (position / cm-1, text) ----------------------

REGION_LABELS = [
    (925, "Q$^2$"),
    (735, "Q$^4$"),
    (510, "Q$^6$"),
]


# --- structural units used for the colours -----------------------------------
# name in the legend, colour, atom sets (column names in MODE_FILE) that belong to it

UNITS = [
    ("Bi lattice",                    "#7f7f7f", ["Bi5"]),
    ("AlO$_6$ (Al$_{oct}$, O13)",      "#2e86c1", ["Al1", "O13"]),
    ("AlO$_4$ (Al$_{tet}$, O17, O21)", "#d68910", ["Al9", "O17", "O21"]),
    ("T–O–T bridge (O29)",            "#c0392b", ["O29"]),
]


# --- output ------------------------------------------------------------------

EXP_LABEL = "experiment, x = 0"
EXP_COLOUR = "#6c3483"
OUTPUT_FILES = ["Bi2Al4O9_IR_exp_vs_HSE06.png", "Bi2Al4O9_IR_exp_vs_HSE06.pdf"]
FIGURE_SIZE = (7.2, 4.8)        # inch
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


def read_experiment(file_name, digitized):
    """
    Read the measured spectrum (two columns).

    For digitized data, points that deviate from a running median are removed
    (crossings with other curves in the original figure) and the rest is
    smoothed with a 5-point running mean.
    """
    data = np.loadtxt(file_name)
    data = data[np.argsort(data[:, 0])]

    if not digitized:
        return data[:, 0], data[:, 1]

    median = np.r_[data[:4, 1],
                   np.median(sliding_window_view(data[:, 1], 9), axis=1),
                   data[-4:, 1]]
    data = data[np.abs(data[:, 1] - median) < 0.03]

    smooth = np.r_[data[:2, 1],
                   np.mean(sliding_window_view(data[:, 1], 5), axis=1),
                   data[-2:, 1]]

    return data[:, 0], smooth


# =============================================================================
#  MAIN
# =============================================================================

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_SIZE})


# --- data --------------------------------------------------------------------

wavenumbers, irreps, intensities, shares, set_names = read_modes(MODE_FILE, MIN_INTENSITY)
wavenumbers = wavenumbers * SCALE
unit_index = dominant_unit(shares, set_names, UNITS)

x_exp, y_exp = read_experiment(EXP_FILE, EXP_IS_DIGITIZED)

x = np.arange(X_MIN - 20, X_MAX + 10, X_STEP)
spectrum = lorentzian_spectrum(x, wavenumbers, intensities, FWHM)

in_range = (x >= X_MIN) & (x <= X_MAX)
spectrum = spectrum / spectrum[in_range].max() * y_exp.max()     # same maximum as the experiment


# --- figure ------------------------------------------------------------------

fig, (ax_spec, ax_stick) = plt.subplots(
    2, 1, figsize=FIGURE_SIZE, sharex=True,
    gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.05),
)


# upper panel: experiment (gaps are not bridged) and computed spectrum

breaks = np.where(np.diff(x_exp) > GAP_LIMIT)[0] + 1

for k, (xs, ys) in enumerate(zip(np.split(x_exp, breaks), np.split(y_exp, breaks))):
    ax_spec.plot(xs, ys, color=EXP_COLOUR, lw=1.6, label=EXP_LABEL if k == 0 else None)

computed_label = "HSE06 harmonic"
if SCALE != 1.0:
    computed_label += f", scaled by {SCALE:g}"
computed_label += f", FWHM {FWHM:g} cm$^{{-1}}$"

ax_spec.plot(x, spectrum, color="black", lw=1.1, label=computed_label)

ax_spec.set_ylabel("normalized intensity")
ax_spec.set_ylim(0, 1.25)
ax_spec.legend(fontsize=FONT_SIZE - 1.5, frameon=False, loc="upper right")

for position, text in REGION_LABELS:
    ax_spec.text(position, 0.99, text, ha="center", fontsize=FONT_SIZE - 1)


# lower panel: sticks coloured by structural unit

for k, (label, colour, _) in enumerate(UNITS):
    selected = (unit_index == k) & (wavenumbers >= X_MIN) & (wavenumbers <= X_MAX)
    ax_stick.vlines(wavenumbers[selected], 0, intensities[selected],
                    color=colour, lw=1.6, label=label)

ax_stick.set_ylabel(r"I / (km mol$^{-1}$)")
ax_stick.set_xlabel(r"wavenumber / cm$^{-1}$")
ax_stick.set_xlim(X_MIN, X_MAX)
ax_stick.set_ylim(0, STICK_YMAX)
ax_stick.legend(fontsize=FONT_SIZE - 2, frameon=False, loc="upper right", ncol=2)


# --- save --------------------------------------------------------------------

for file_name in OUTPUT_FILES:
    fig.savefig(file_name, dpi=DPI, bbox_inches="tight")
    print("written:", file_name)
