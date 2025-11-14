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
    dailyReadingLimit: 1,
    unlimitedReadings: false,

    // Spreads
    allSpreadTypes: false,
    availableSpreads: [
      'single_card',
      'three_card'
    ],

    // Reading types
    allReadingTypes: false,
    availableReadingTypes: [
      'career',
      'romance',
      'wellness'
    ],

    // Features
    themeSelection: true,
    readingHistory: false,
    advancedInterpretations: false,
    exportReadings: false,
    metaAnalysis: false,
    cardFlip: true, // Can flip to see symbolism
    quantumSignature: true
  },

  ui: {
    showPremiumBadges: true, // Show 🔒 on locked features
    showUpgradeButton: true, // Show upgrade button in settings
    upgradePromptFrequency: 'on_limit', // When to show upgrade prompt
  },

  monetization: {
    upgradePrompt: true,
    upgradeUrl: 'https://play.google.com/store/apps/details?id=com.aphoticshaman.quantumtarot.premium',
    upgradePrice: '$3.99',
    upgradeMessage: 'Upgrade to Premium for unlimited readings and all features!',
  },

  branding: {
    tagline: 'One Free Reading Daily',
    accent: 'Free Edition'
  }
};

export default APP_CONFIG;
