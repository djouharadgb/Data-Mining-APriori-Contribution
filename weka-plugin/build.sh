#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Build and install WNAR-SW as a WEKA plugin
# ──────────────────────────────────────────────────────────────
set -e

WEKA_JAR="/home/nesrine/Downloads/weka-3-8-6/weka.jar"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$PLUGIN_DIR/src"
BUILD_DIR="$PLUGIN_DIR/build"
OUT_JAR="$PLUGIN_DIR/WNARSW.jar"

# WEKA packages directory
WEKA_PACKAGES="$HOME/wekafiles/packages/WNARSW"

echo "=== WNAR-SW WEKA Plugin Builder ==="
echo ""

# Check WEKA jar
if [ ! -f "$WEKA_JAR" ]; then
    echo "ERROR: WEKA jar not found at $WEKA_JAR"
    echo "       Edit WEKA_JAR in this script to point to your weka.jar"
    exit 1
fi

# Clean and compile
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "[1/3] Compiling WNARSW.java ..."
javac -source 11 -target 11 \
      -cp "$WEKA_JAR" \
      -d "$BUILD_DIR" \
      "$SRC_DIR/weka/associations/WNARSW.java"

echo "[2/3] Packaging WNARSW.jar ..."
cd "$BUILD_DIR"
jar cf "$OUT_JAR" weka/
cd "$PLUGIN_DIR"

echo "[3/3] Installing to WEKA packages directory ..."
mkdir -p "$WEKA_PACKAGES"
cp "$OUT_JAR" "$WEKA_PACKAGES/"

# Create Description.props for WEKA package manager
cat > "$WEKA_PACKAGES/Description.props" << 'EOF'
# WEKA package descriptor
PackageName=WNARSW
Version=1.0.0
Date=2025-01-01
Title=WNAR-SW: Weighted Negative Association Rule Mining with Regularity
Category=Associations
Author=WNAR-SW Research Team
Maintainer=WNAR-SW Research Team
License=GPL 3.0
Description=Weighted Negative Association Rule Mining with Regularity-Aware Sliding Windows for Real-Time Medical Alert Generation
URL=https://example.com
Depends=weka (>=3.8.0)
EOF

echo ""
echo "=== Build successful ==="
echo "  Plugin JAR: $OUT_JAR"
echo "  Installed:  $WEKA_PACKAGES/"
echo ""
echo "To use in WEKA:"
echo "  1. Restart WEKA"
echo "  2. Open Explorer -> load medicine_side_effects.arff"
echo "  3. Associate tab -> Choose -> weka.associations.WNARSW"
echo "  4. Configure thresholds in the GUI and click Start"
echo ""
echo "To run from command line:"
echo "  java -cp \"$WEKA_JAR:$OUT_JAR\" weka.associations.WNARSW \\"
echo "       -t ../medicine_side_effects.arff \\"
echo "       -S 0.02 -N 0.01 -C 0.25 -R 600"
