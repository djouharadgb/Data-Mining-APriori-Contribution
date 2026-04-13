"""
Generate a WEKA-compatible ARFF file from medicine_dataset.csv.

Each transaction is a drug record. Each unique side effect becomes a binary
attribute (1 = present, 0 = absent).  Only records with >= 2 side effects
are included, matching the notebook's transaction extraction logic.

The last WINDOW_SIZE (3000) transactions are exported to match the sliding
window used in the WNAR-SW experiments.

Output: medicine_side_effects.arff
"""

import pandas as pd
import re

# ── Parameters (match the notebook) ──────────────────────────
WINDOW_SIZE = 3000
INPUT_CSV   = "medicine_dataset.csv"
OUTPUT_ARFF = "medicine_side_effects.arff"

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, low_memory=False)
se_cols = [c for c in df.columns if c.startswith("sideEffect")]

# Extract transactions: list of non-null side effects per row
transactions = []
for _, row in df[se_cols].iterrows():
    items = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
    if len(items) >= 2:
        transactions.append(items)

print(f"Total transactions with >= 2 side effects: {len(transactions)}")

# Take the last WINDOW_SIZE transactions (sliding window)
window = transactions[-WINDOW_SIZE:]
print(f"Window size used: {len(window)}")

# ── Collect all unique side effects in the window ────────────
all_effects = set()
for trans in window:
    all_effects.update(trans)

# Sort for deterministic attribute ordering
all_effects = sorted(all_effects)
print(f"Unique side effects in window: {len(all_effects)}")


def sanitize_name(name):
    """Make a side-effect name safe for WEKA attribute names."""
    # Replace problematic characters with underscores
    s = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Build attribute name mapping
attr_names = []
seen = set()
for effect in all_effects:
    base = sanitize_name(effect)
    if not base:
        base = "unknown"
    name = base
    counter = 2
    while name in seen:
        name = f"{base}_{counter}"
        counter += 1
    seen.add(name)
    attr_names.append(name)

# ── Write ARFF ───────────────────────────────────────────────
with open(OUTPUT_ARFF, "w") as f:
    f.write("% WEKA ARFF file generated from medicine_dataset.csv\n")
    f.write("% Sliding window: last {} transactions with >= 2 side effects\n".format(WINDOW_SIZE))
    f.write("% Each attribute is a binary flag for a specific side effect\n")
    f.write("%\n")
    f.write("@RELATION medicine_side_effects\n\n")

    # Attributes: one binary attribute per side effect
    for attr_name in attr_names:
        f.write(f"@ATTRIBUTE {attr_name} {{0,1}}\n")

    f.write("\n@DATA\n")

    # Build a lookup for fast indexing
    effect_to_idx = {effect: i for i, effect in enumerate(all_effects)}

    for trans in window:
        row = ["0"] * len(all_effects)
        for item in trans:
            if item in effect_to_idx:
                row[effect_to_idx[item]] = "1"
        f.write(",".join(row) + "\n")

print(f"\nARFF file written to: {OUTPUT_ARFF}")
print(f"  Attributes: {len(all_effects)}")
print(f"  Instances:  {len(window)}")
print(f"\nTo use in WEKA:")
print(f"  1. Open WEKA Explorer")
print(f"  2. Load '{OUTPUT_ARFF}' via the Preprocess tab")
print(f"  3. Go to Associate tab -> Choose Apriori")
print(f"  4. Set minMetric (confidence) and lowerBoundMinSupport as needed")
