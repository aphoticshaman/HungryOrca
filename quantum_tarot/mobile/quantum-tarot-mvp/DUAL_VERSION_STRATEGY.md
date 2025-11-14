# 📱 Dual Version Strategy - Play Store
## Free vs Premium Builds

**Business Model**:
- **Free**: 1 reading/day
- **Premium**: $3.99 one-time purchase, unlimited everything

**Strategy**: Two separate app listings on Google Play Store

---

## 🎯 Why Two Apps Works

### Advantages:
1. **Clear value proposition**: Users know exactly what they're getting
2. **No IAP complexity**: No in-app purchase code, receipt validation, etc.
3. **Better conversion funnel**: Free → see value → buy premium
4. **No subscription backlash**: One-time purchase is user-friendly
5. **Simpler codebase**: Feature flags instead of payment logic

### Disadvantages:
1. Two codebases to maintain (minimal with good architecture)
2. Two app listings to manage
3. Two sets of reviews/ratings

**Mitigation**: Share 95% of code via build variants, only toggle features

---

## 🏗️ Architecture

### Shared Codebase with Build Variants

```
quantum-tarot-mvp/
├── src/
│   ├── components/      # Shared
│   ├── screens/         # Shared
│   ├── services/        # Shared
│   ├── data/            # Shared
│   └── config/
│       ├── config.free.js      # Free version config
│       └── config.premium.js   # Premium version config
├── app.free.json         # Free app config
├── app.premium.json      # Premium app config
├── package.json
└── BUILD_INSTRUCTIONS.md
```

---

## 📝 Configuration Files

### `src/config/config.free.js`
```javascript
export const APP_CONFIG = {
  version: 'free',
  name: 'Quantum Tarot - Free',
  slug: 'quantum-tarot-free',
  bundleId: 'com.aphoticshaman.quantumtarot.free',

  features: {
    dailyReadingLimit: 1,
    unlimitedReadings: false,
    allSpreadTypes: false,
    themeSelection: true,
    readingHistory: false,
    advancedInterpretations: false,
    exportReadings: false,

    // Available spreads in free
    availableSpreads: ['single_card', 'three_card'],

    // Available reading types in free
    availableReadingTypes: ['career', 'romance', 'wellness']
  },

  monetization: {
    upgradePrompt: true,
    upgradeUrl: 'https://play.google.com/store/apps/details?id=com.aphoticshaman.quantumtarot.premium',
    upgradePrice: '$3.99'
  }
};
```

### `src/config/config.premium.js`
```javascript
export const APP_CONFIG = {
  version: 'premium',
  name: 'Quantum Tarot - Premium',
  slug: 'quantum-tarot-premium',
  bundleId: 'com.aphoticshaman.quantumtarot.premium',

  features: {
    dailyReadingLimit: null, // Unlimited
    unlimitedReadings: true,
    allSpreadTypes: true,
    themeSelection: true,
    readingHistory: true,
    advancedInterpretations: true,
    exportReadings: true,

    // All spreads available
    availableSpreads: [
      'single_card',
      'three_card',
      'goal_progress',
      'decision_analysis',
      'daily_checkin',
      'clairvoyant_predictive',
      'relationship',
      'celtic_cross',
      'horseshoe'
    ],

    // All reading types available
    availableReadingTypes: [
      'career',
      'romance',
      'wellness',
      'spiritual',
      'decision',
      'general',
      'shadow_work',
      'year_ahead'
    ]
  },

  monetization: {
    upgradePrompt: false,
    upgradeUrl: null,
    upgradePrice: null
  }
};
```

---

## 🔧 Dynamic Feature Access

### `src/utils/featureGate.js`
```javascript
import { APP_CONFIG } from '../config/config'; // Auto-selected by build

export class FeatureGate {
  static canDrawReading() {
    if (APP_CONFIG.features.unlimitedReadings) return true;

    // Check daily limit for free version
    return checkDailyLimit();
  }

  static isSpreadAvailable(spreadType) {
    return APP_CONFIG.features.availableSpreads.includes(spreadType);
  }

  static isReadingTypeAvailable(readingType) {
    return APP_CONFIG.features.availableReadingTypes.includes(readingType);
  }

  static canAccessReadingHistory() {
    return APP_CONFIG.features.readingHistory;
  }

  static canExportReading() {
    return APP_CONFIG.features.exportReadings;
  }

  static shouldShowUpgradePrompt() {
    return APP_CONFIG.monetization.upgradePrompt;
  }

  static getUpgradeUrl() {
    return APP_CONFIG.monetization.upgradeUrl;
  }
}

async function checkDailyLimit() {
  const lastReading = await getLastReadingDate();
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;

  return (now - lastReading) >= oneDayMs;
}
```

---

## 🎨 UI Differences

### Free Version - Upgrade Prompts

**IntentionScreen.js** (after daily limit reached):
```javascript
import { FeatureGate } from '../utils/featureGate';

async function handleDrawCards() {
  const canDraw = await FeatureGate.canDrawReading();

  if (!canDraw && FeatureGate.shouldShowUpgradePrompt()) {
    Alert.alert(
      'Daily Limit Reached',
      `You've used your free reading for today!\n\nUpgrade to Premium for ${APP_CONFIG.monetization.upgradePrice} and get:\n\n• Unlimited readings\n• All 9 spread types\n• Reading history\n• Export readings\n• No ads, ever`,
      [
        { text: 'Maybe Later', style: 'cancel' },
        {
          text: `Upgrade for ${APP_CONFIG.monetization.upgradePrice}`,
          onPress: () => Linking.openURL(FeatureGate.getUpgradeUrl())
        }
      ]
    );
    return;
  }

  // Continue with reading...
}
```

**ReadingTypeScreen.js** (locked reading types):
```javascript
{READING_TYPES.map(type => (
  <TouchableOpacity
    key={type.id}
    style={[
      styles.typeOption,
      !FeatureGate.isReadingTypeAvailable(type.id) && styles.lockedOption
    ]}
    onPress={() => {
      if (!FeatureGate.isReadingTypeAvailable(type.id)) {
        showUpgradePrompt();
      } else {
        handleSelectType(type.id);
      }
    }}
  >
    <Text style={styles.typeName}>
      {type.name}
      {!FeatureGate.isReadingTypeAvailable(type.id) && ' 🔒'}
    </Text>
  </TouchableOpacity>
))}
```

**SpreadSelectionScreen.js** (locked spreads):
```javascript
{SPREADS.map(spread => {
  const isAvailable = FeatureGate.isSpreadAvailable(spread.type);

  return (
    <TouchableOpacity
      key={spread.type}
      style={[
        styles.spreadOption,
        !isAvailable && styles.locked
      ]}
      onPress={() => {
        if (!isAvailable) {
          Alert.alert(
            'Premium Feature',
            `${spread.name} spread is available in Premium.\n\nUpgrade for ${APP_CONFIG.monetization.upgradePrice}?`,
            [
              { text: 'Not Now', style: 'cancel' },
              { text: 'Upgrade', onPress: openUpgradeUrl }
            ]
          );
        } else {
          selectSpread(spread.type);
        }
      }}
    >
      <Text style={styles.spreadName}>
        {spread.name} {!isAvailable && '🔒'}
      </Text>
    </TouchableOpacity>
  );
})}
```

### Premium Version - No Prompts
All features unlocked, no upgrade prompts, clean UX.

---

## 📦 Build Process

### Option 1: Manual Build Script
```bash
#!/bin/bash
# build-free.sh

echo "Building FREE version..."
cp app.free.json app.json
cp src/config/config.free.js src/config/config.js
eas build --platform android --profile preview
mv quantum-tarot-mvp.apk quantum-tarot-free.apk
echo "FREE build complete: quantum-tarot-free.apk"
```

```bash
#!/bin/bash
# build-premium.sh

echo "Building PREMIUM version..."
cp app.premium.json app.json
cp src/config/config.premium.js src/config/config.js
eas build --platform android --profile production
mv quantum-tarot-mvp.apk quantum-tarot-premium.apk
echo "PREMIUM build complete: quantum-tarot-premium.apk"
```

### Option 2: EAS Build Profiles
**eas.json**:
```json
{
  "build": {
    "free-preview": {
      "distribution": "internal",
      "android": {
        "buildType": "apk",
        "gradleCommand": ":app:assembleFreeRelease"
      },
      "env": {
        "APP_VARIANT": "free"
      }
    },
    "premium-preview": {
      "distribution": "internal",
      "android": {
        "buildType": "apk",
        "gradleCommand": ":app:assemblePremiumRelease"
      },
      "env": {
        "APP_VARIANT": "premium"
      }
    },
    "free-production": {
      "android": {
        "buildType": "app-bundle"
      },
      "env": {
        "APP_VARIANT": "free"
      }
    },
    "premium-production": {
      "android": {
        "buildType": "app-bundle"
      },
      "env": {
        "APP_VARIANT": "premium"
      }
    }
  }
}
```

Then build with:
```bash
eas build --platform android --profile free-preview
eas build --platform android --profile premium-preview
```

---

## 🎯 Feature Matrix

| Feature | Free | Premium |
|---------|------|---------|
| **Readings per day** | 1 | ∞ |
| **Spread types** | 2 (Single, 3-card) | 9 (All) |
| **Reading types** | 3 (Career, Romance, Wellness) | 8 (All) |
| **Theme selection** | ✅ | ✅ |
| **Reading history** | ❌ | ✅ |
| **Export readings** | ❌ | ✅ |
| **Advanced interpretations** | ❌ | ✅ |
| **Celtic Cross spread** | ❌ | ✅ |
| **Year Ahead spread** | ❌ | ✅ |
| **Card flip symbolism** | ✅ | ✅ |
| **Quantum randomness** | ✅ | ✅ |
| **Offline operation** | ✅ | ✅ |
| **No ads** | ✅ | ✅ |

---

## 🏪 Play Store Listings

### Free Version
**Title**: Quantum Tarot - Free Daily Reading
**Description**:
```
Experience genuine quantum tarot readings with AI-powered interpretations.

✨ FREE FEATURES:
• 1 free reading per day
• Quantum randomness from device hardware
• Offline AGI interpretation engine
• 2 spread types (Single card, Past-Present-Future)
• 3 reading types (Career, Romance, Wellness)
• 5 retro color themes
• No ads, ever

🔮 UPGRADE TO PREMIUM ($3.99 one-time):
• Unlimited readings
• All 9 spread types including Celtic Cross
• All 8 reading types
• Reading history
• Export readings
• Advanced meta-analysis

No subscriptions. No tracking. Pure tarot.
```

**Screenshots**: Show single card reading, 3-card spread, upgrade prompt

### Premium Version
**Title**: Quantum Tarot - Premium Edition
**Description**:
```
Professional quantum tarot with unlimited readings and advanced spreads.

🌟 PREMIUM FEATURES:
• Unlimited readings, anytime
• All 9 spread types including Celtic Cross & Horseshoe
• All 8 reading types (Career, Romance, Wellness, Shadow Work, Year Ahead, more)
• Reading history with search
• Export readings as text or image
• Advanced meta-analysis across cards
• Quantum randomness from device hardware
• Offline AGI interpretation engine
• 5 retro color themes
• No ads, no tracking, no subscriptions

ONE-TIME PURCHASE. YOURS FOREVER.

Built with genuine multi-agent AGI for sophisticated, personalized interpretations.
Not template-based like other apps.
```

**Screenshots**: Show Celtic Cross, relationship spread, reading history, export

---

## 💻 Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/config/config.free.js`
- [ ] Create `src/config/config.premium.js`
- [ ] Create `src/utils/featureGate.js`
- [ ] Create `app.free.json`
- [ ] Create `app.premium.json`
- [ ] Update `eas.json` with build profiles

### Phase 2: Feature Gating
- [ ] Add daily limit check to `IntentionScreen.js`
- [ ] Add spread locking to spread selection
- [ ] Add reading type locking to type selection
- [ ] Add upgrade prompts throughout free version
- [ ] Disable upgrade prompts in premium version
- [ ] Add 🔒 icons to locked features in free

### Phase 3: Build & Test
- [ ] Test free version on device
- [ ] Verify daily limit works
- [ ] Test upgrade URL opens Play Store
- [ ] Test premium version on device
- [ ] Verify all features unlocked
- [ ] Verify no upgrade prompts

### Phase 4: Deploy
- [ ] Build free APK with EAS
- [ ] Build premium APK with EAS
- [ ] Upload both to Google Play Console
- [ ] Create separate listings
- [ ] Link free → premium in upgrade prompts
- [ ] Submit for review

---

## 🎯 Estimated Effort

**Setup (one-time)**: 4-6 hours
- Config files: 1 hour
- Feature gating: 2 hours
- UI updates: 2 hours
- Testing: 1 hour

**Maintenance**: Minimal
- New features: Add to both configs
- Bug fixes: Fix once, deploy twice
- 95% code shared

---

## 💰 Revenue Projections

**Conversion funnel:**
1. **1000 free downloads** → try app
2. **30% use regularly** (300 users) → see value
3. **10% convert** (30 users) → buy premium

**Revenue**: 30 × $3.99 = **$119.70**

At scale:
- **10,000 downloads** → 300 conversions → **$1,197**
- **100,000 downloads** → 3,000 conversions → **$11,970**

**Google Play takes 15%** (after first $1M), you keep **$10,175**

Plus:
- Premium version direct sales (people who skip free)
- Word of mouth from satisfied customers
- App store featuring potential ("No subscription!")

---

## 🚀 Launch Strategy

### Week 1: Soft Launch
- Release free version only
- Gather feedback
- Fix bugs
- Build user base

### Week 2: Premium Launch
- Release premium version
- Add upgrade links to free version
- Monitor conversion rate
- Adjust pricing if needed (A/B test $2.99 vs $3.99 vs $4.99)

### Week 3+: Growth
- App store optimization
- Reddit/Discord promotion
- Influencer outreach
- Content marketing

---

## ✅ Bottom Line

**Two apps is 100% viable and actually BETTER than IAP for this use case.**

**Why it works:**
1. Simpler code (no payment validation)
2. Better UX (clear value, no subscription confusion)
3. Higher conversion (one-time purchase psychology)
4. Easier maintenance (feature flags, not payment logic)

**You're willing to do the work → this WILL succeed.**

**Next step**: Implement config files and feature gating. Then build both versions. 🚀

---

**Files to Create:**
1. `src/config/config.free.js`
2. `src/config/config.premium.js`
3. `src/utils/featureGate.js`
4. `app.free.json`
5. `app.premium.json`
6. `build-free.sh`
7. `build-premium.sh`

**Time to ship: 1 week** ⚡
