/**
 * AD MANAGER - STUB
 * AdMob removed for now - can add back later
 * All methods are no-ops
 */

class AdManager {
  async initialize() {
    console.log('[AdManager] Ads disabled - stub mode');
  }

  async showAdAfterCard() {
    // No-op
  }

  async showAdBeforeSynthesis() {
    // No-op
  }

  isEnabled() {
    return false;
  }
}

// Export singleton instance
export default new AdManager();
