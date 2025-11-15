#!/bin/bash
# Build PREMIUM version of Quantum Tarot

set -e

echo "🔮 Building Quantum Tarot - PREMIUM Edition..."
echo ""

# Backup current files
echo "📦 Backing up current config..."
if [ -f "app.json" ]; then
  cp app.json app.json.backup
fi
if [ -f "src/config/config.js" ]; then
  cp src/config/config.js src/config/config.js.backup
fi

# Copy PREMIUM configs
echo "👑 Switching to PREMIUM configuration..."
cp app.premium.json app.json
cp src/config/config.premium.js src/config/config.js

echo "✅ Config switched to PREMIUM version"
echo ""
echo "📱 App Details:"
echo "   Name: Quantum Tarot - Premium"
echo "   Package: com.aphoticshaman.quantumtarot.premium"
echo "   Features: Unlimited readings, all spreads, all features"
echo "   Price: $3.99 one-time purchase"
echo ""

# Build with EAS
echo "🏗️  Building APK with EAS..."
eas build --platform android --profile production --non-interactive

echo ""
echo "✨ PREMIUM build complete!"
echo ""
echo "Next steps:"
echo "1. Download APK from EAS"
echo "2. Test on device"
echo "3. Set price to $3.99 in Google Play Console"
echo "4. Upload to Google Play Console (Premium listing)"
echo ""
echo "Restoring original configs..."

# Restore backups
if [ -f "app.json.backup" ]; then
  mv app.json.backup app.json
fi
if [ -f "src/config/config.js.backup" ]; then
  mv src/config/config.js.backup src/config/config.js
fi

echo "✅ Done!"
