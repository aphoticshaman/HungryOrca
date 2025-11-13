# Apply Quantum Tarot Fixes on Windows

## The Patch Contains:
1. **Card database + AGI query engine + flip UI** (001e64f5)
2. **Dual version system (Free vs Premium)** (68e12749)
3. **Manual push helper script** (3a2117cc)
4. **Fix syntax error: escape apostrophes in strings** (b924fe2a) ← THE BUNDLING FIX

## How to Apply on Windows:

### Option 1: Apply the Patch File
```bash
cd C:\Users\ryanj\HungryOrca
git apply quantum-tarot-fixes.patch
```

This will apply all the changes to your working directory. Then commit them:

```bash
git add .
git commit -m "Apply fixes from Linux: card database, dual version, syntax fix"
```

### Option 2: Pull from Remote (if push eventually succeeds)
```bash
cd C:\Users\ryanj\HungryOrca
git fetch origin claude/quantum-tarot-app-setup-011CV4XWLj8y1V5TvBkRgz5M
git pull origin claude/quantum-tarot-app-setup-011CV4XWLj8y1V5TvBkRgz5M
```

## What the Syntax Fix Does:
- **Fixed:** `'Each person's energy'` → `'Each person\'s energy'`
- **Fixed:** `"What's behind you"` → `"What is behind you"`
- **Location:** `src/data/spreadDefinitions.js` lines 338 and 498

## Test After Applying:
```bash
cd quantum_tarot\mobile\quantum-tarot-mvp
npm start
```

The bundling error should now be resolved!

## Files Modified in This Patch:
- `src/data/cardDatabase.js` (new)
- `src/services/cardQueryEngine.js` (new)
- `src/services/enhancedLunatiQ.js` (new)
- `src/components/FlippableCard.js` (new)
- `src/config/config.free.js` (new)
- `src/config/config.premium.js` (new)
- `src/utils/featureGate.js` (new)
- `app.free.json` (new)
- `app.premium.json` (new)
- `build-free.sh` (new)
- `build-premium.sh` (new)
- `CARD_DATABASE_ARCHITECTURE.md` (new)
- `DUAL_VERSION_STRATEGY.md` (new)
- `PUSH_COMMITS.sh` (new)
- `src/screens/IntentionScreen.js` (modified)
- `src/data/spreadDefinitions.js` (modified - SYNTAX FIX)
