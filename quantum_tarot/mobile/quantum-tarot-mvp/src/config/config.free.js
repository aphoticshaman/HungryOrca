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
    dailyReadingLimit: null, // UNLIMITED!
    unlimitedReadings: true, // Yes, unlimited readings

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

  // No ads - just polite upgrade prompts
  prompts: {
    showUpgradePrompts: true, // Show polite prompts to support development
    afterCardReading: true, // Prompt after card (occasionally)
    beforeSynthesis: true, // Prompt before synthesis (occasionally)
  },

  ui: {
    showPremiumBadges: true, // Show 🔒 on locked features
    showUpgradeButton: true, // Show upgrade button in settings
    upgradePromptFrequency: 'polite', // Polite, rotating messages
  },

  monetization: {
    upgradePrompt: true,
    inAppPurchaseProductId: 'com.aphoticshaman.lunatiq.premium', // IAP Product ID
    upgradePrice: '$3.99',
    upgradeMessage: 'Support indie development! Unlock all features for $3.99 (one-time payment)',
  },

  branding: {
    tagline: 'Unlimited Readings • Support Indie Development',
    accent: 'Free Edition'
  }
};

export default APP_CONFIG;
