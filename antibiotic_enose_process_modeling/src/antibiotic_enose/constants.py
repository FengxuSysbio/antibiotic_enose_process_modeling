SEED = 20260913
SENSOR_COLS = [f"Channel_{i}" for i in range(1, 17)]
GENTAMICIN_TARGETS = ["PMV", "RS", "TS", "NH4", "Titer"]
TRANSFER_TARGETS = {
    "Erythromycin": ["DCW", "Titer", "RS", "NH4"],
    "Cephalosporin": ["DCW", "Titer", "RS"],
    "Lincomycin": ["PMV", "Titer", "RS", "NH4"],
}
