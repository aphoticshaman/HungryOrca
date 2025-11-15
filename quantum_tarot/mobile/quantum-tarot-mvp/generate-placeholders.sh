#!/bin/bash

# Generate placeholder icon (1024x1024)
convert -size 1024x1024 \
  -background "#0A0E27" \
  -fill "#00D9FF" \
  -gravity center \
  -font Arial-Bold \
  -pointsize 200 \
  label:"🔮" \
  assets/icon.png

# Generate adaptive icon (1024x1024)
convert -size 1024x1024 \
  -background "#0A0E27" \
  -fill "#00D9FF" \
  -gravity center \
  -font Arial-Bold \
  -pointsize 200 \
  label:"🔮" \
  assets/adaptive-icon.png

# Generate splash screen (1242x2688 - iPhone 12 Pro Max)
convert -size 1242x2688 \
  gradient:"#0A0E27-#1a1f3a" \
  -fill "#00D9FF" \
  -gravity center \
  -font Arial-Bold \
  -pointsize 120 \
  -annotate +0-200 "🔮" \
  -fill white \
  -pointsize 60 \
  -annotate +0+100 "Quantum Tarot" \
  assets/splash.png

echo "✅ Placeholder images generated!"
