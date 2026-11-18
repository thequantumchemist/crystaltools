IR plots
========

    python plot_ir.py            computed spectrum + sticks coloured by structural unit
    python plot_ir_overlay.py    computed spectrum on top of a measured one

All file names and parameters are set in the SETTINGS section at the top of each script.
MODE_FILE is the mode_character.dat written by "cryfreq.py analyze"; the atom-set names in
UNITS must match its column headers. EXP_FILE: two columns, wavenumber / cm-1 and intensity.
Requires numpy and matplotlib.
