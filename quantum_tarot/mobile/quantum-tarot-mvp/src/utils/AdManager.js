/**
 * AD MANAGER - LEGACY STUB
 * Now redirects to UpgradePromptManager for polite upgrade prompts
 * No ads - just gentle reminders to support indie development
 */

import UpgradePromptManager from './UpgradePromptManager';
import InAppPurchaseManager from './InAppPurchaseManager';

class AdManager {
  async initialize() {
    console.log('[AdManager] Legacy stub - using UpgradePromptManager instead');
  }

  async showAdAfterCard() {
    // Show polite upgrade prompt instead of ad
    await UpgradePromptManager.showAfterCardPrompt(() => {
      InAppPurchaseManager.purchasePremium();
    });
  }

  async showAdBeforeSynthesis() {
    // Show polite upgrade prompt instead of ad
    await UpgradePromptManager.showBeforeSynthesisPrompt(() => {
      InAppPurchaseManager.purchasePremium();
    });
  }

  isEnabled() {
    return false; // Ads are disabled
  }
}

// Export singleton instance
export default new AdManager();
