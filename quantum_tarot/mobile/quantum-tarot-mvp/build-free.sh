#!/bin/bash
# Build FREE version of Quantum Tarot

set -e

echo "🔮 Building Quantum Tarot - FREE Edition..."
echo ""

# Backup current files
echo "📦 Backing up current config..."
if [ -f "app.json" ]; then
  cp app.json app.json.backup
fi
if [ -f "src/config/config.js" ]; then
  cp src/config/config.js src/config/config.js.backup
fi

# Copy FREE configs
echo "🆓 Switching to FREE configuration..."
cp app.free.json app.json
cp src/config/config.free.js src/config/config.js

echo "✅ Config switched to FREE version"
echo ""
echo "📱 App Details:"
echo "   Name: Quantum Tarot - Free"
echo "   Package: com.aphoticshaman.quantumtarot.free"
echo "   Features: 1 reading/day, 2 spreads, 3 reading types"
echo ""

# Build with EAS
echo "🏗️  Building APK with EAS..."
eas build --platform android --profile preview --non-interactive

echo ""
echo "✨ FREE build complete!"
echo ""
echo "Next steps:"
echo "1. Download APK from EAS"
echo "2. Test on device"
echo "3. Upload to Google Play Console (Free listing)"
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
