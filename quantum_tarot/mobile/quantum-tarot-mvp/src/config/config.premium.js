/**
 * PREMIUM VERSION CONFIGURATION
 * Quantum Tarot - Premium Edition
 */

export const APP_CONFIG = {
  version: 'premium',
  name: 'Quantum Tarot - Premium',
  displayName: 'Quantum Tarot Premium',
  slug: 'quantum-tarot-premium',
  bundleId: 'com.aphoticshaman.quantumtarot.premium',

  features: {
    // Reading limits
    dailyReadingLimit: null, // Unlimited
    unlimitedReadings: true,

    // Spreads (ALL)
    allSpreadTypes: true,
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
    maxCardsPerSpread: null, // No limit

    // Reading types (ALL)
    allReadingTypes: true,
    availableReadingTypes: [
      'career',
      'romance',
      'wellness',
      'spiritual',
      'decision',
      'general',
      'shadow_work',
      'year_ahead'
    ],

    // Features (ALL ENABLED)
    themeSelection: true,
    readingHistory: true,
    saveReadings: true, // ENABLED
    advancedInterpretations: true,
    exportReadings: true,
    metaAnalysis: true,
    cardFlip: true,
    quantumSignature: true
  },

  // Ads configuration
  ads: {
    enabled: false, // No ads for premium
    interstitialAfterCard: false,
    interstitialBeforeSynthesis: false,
    admobAppId: null,
    admobInterstitialId: null,
  },

  ui: {
    showPremiumBadges: false, // No locks, everything unlocked
    showUpgradeButton: false, // No upgrade needed
    upgradePromptFrequency: 'never',
  },

  monetization: {
    upgradePrompt: false,
    upgradeUrl: null,
    upgradePrice: null,
    upgradeMessage: null
  },

  branding: {
    tagline: 'Unlimited Quantum Readings',
    accent: 'Premium Edition'
  }
};

export default APP_CONFIG;
