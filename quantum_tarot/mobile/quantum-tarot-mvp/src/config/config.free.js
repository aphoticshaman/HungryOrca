/**
 * FREE VERSION CONFIGURATION
 * Quantum Tarot - Free Daily Reading
 */

export const APP_CONFIG = {
  version: 'free',
  name: 'Quantum Tarot - Free',
  displayName: 'Quantum Tarot Free',
  slug: 'quantum-tarot-free',
  bundleId: 'com.aphoticshaman.quantumtarot.free',

  features: {
    // Reading limits
    dailyReadingLimit: null, // UNLIMITED with ads!
    unlimitedReadings: true, // Yes, unlimited - we make money from ads

    // Spreads (3 cards or fewer ONLY)
    allSpreadTypes: false,
    availableSpreads: [
      'single_card',
      'three_card'
    ],
    maxCardsPerSpread: 3, // Hard limit - bigger spreads are premium

    // Reading types
    allReadingTypes: false,
    availableReadingTypes: [
      'career',
      'romance',
      'wellness'
    ],

    // Features
    themeSelection: true,
    readingHistory: false, // BLOCKED - Premium only
    saveReadings: false, // BLOCKED - Premium only
    exportReadings: false, // BLOCKED - Premium only (copy/share)
    advancedInterpretations: false,
    metaAnalysis: false,
    cardFlip: true, // Can flip to see symbolism
    quantumSignature: true
  },

  // Ads configuration
  ads: {
    enabled: true,
    interstitialAfterCard: true, // Show ad between cards
    interstitialBeforeSynthesis: true, // Show ad before synthesis
    admobAppId: 'ca-app-pub-XXXXXXXXXXXXXXXX~XXXXXXXXXX', // TODO: Replace with your AdMob App ID
    admobInterstitialId: 'ca-app-pub-XXXXXXXXXXXXXXXX/XXXXXXXXXX', // TODO: Replace with your AdMob Interstitial ID
  },

  ui: {
    showPremiumBadges: true, // Show 🔒 on locked features
    showUpgradeButton: true, // Show upgrade button in settings
    upgradePromptFrequency: 'on_limit', // When to show upgrade prompt
  },

  monetization: {
    upgradePrompt: true,
    upgradeUrl: 'https://apps.apple.com/app/YOUR-APP-ID', // TODO: Update with actual App Store URL
    upgradePrice: '$3.99',
    upgradeMessage: 'Upgrade to Premium: Remove ads, unlock all spreads, save & share readings!',
  },

  branding: {
    tagline: 'Unlimited Readings • Ad-Supported',
    accent: 'Free Edition'
  }
};

export default APP_CONFIG;
