/**
 * AD MANAGER
 * Handles Google AdMob interstitial ads for free version
 * Premium version has ads disabled
 */

import { Platform } from 'react-native';
import { InterstitialAd, AdEventType, TestIds } from 'react-native-google-mobile-ads';
import { APP_CONFIG } from '../config/config.free'; // Swapped during build

class AdManager {
  constructor() {
    this.interstitialAd = null;
    this.isAdLoaded = false;
    this.isAdLoading = false;
  }

  /**
   * Initialize AdMob
   * Call this once at app startup
   */
  async initialize() {
    if (!APP_CONFIG.ads.enabled) {
      console.log('[AdManager] Ads disabled (Premium version)');
      return;
    }

    console.log('[AdManager] Initializing AdMob...');

    try {
      // Note: react-native-google-mobile-ads auto-initializes
      // Just need to preload first ad
      await this.loadInterstitialAd();
    } catch (error) {
      console.error('[AdManager] Initialization error:', error);
    }
  }

  /**
   * Load interstitial ad
   */
  async loadInterstitialAd() {
    if (!APP_CONFIG.ads.enabled) {
      return;
    }

    if (this.isAdLoading || this.isAdLoaded) {
      console.log('[AdManager] Ad already loading or loaded');
      return;
    }

    this.isAdLoading = true;

    try {
      // Get ad unit ID from config
      // If config has placeholder IDs, use test ads
      const configAdId = APP_CONFIG.ads.admobInterstitialId;
      const isTestId = configAdId.includes('XXXXXXXXXXXXXXXX');

      const adUnitId = isTestId
        ? TestIds.INTERSTITIAL
        : configAdId;

      console.log('[AdManager] Loading interstitial ad...', { adUnitId: isTestId ? 'TEST' : 'REAL' });

      this.interstitialAd = InterstitialAd.createForAdRequest(adUnitId, {
        requestNonPersonalizedAdsOnly: false,
      });

      // Set up event listeners
      this.interstitialAd.addAdEventListener(AdEventType.LOADED, () => {
        console.log('[AdManager] Interstitial ad loaded');
        this.isAdLoaded = true;
        this.isAdLoading = false;
      });

      this.interstitialAd.addAdEventListener(AdEventType.CLOSED, () => {
        console.log('[AdManager] Interstitial ad closed');
        this.isAdLoaded = false;
        // Preload next ad
        setTimeout(() => this.loadInterstitialAd(), 1000);
      });

      this.interstitialAd.addAdEventListener(AdEventType.ERROR, (error) => {
        console.error('[AdManager] Ad error:', error);
        this.isAdLoaded = false;
        this.isAdLoading = false;
        // Retry after delay
        setTimeout(() => this.loadInterstitialAd(), 5000);
      });

      // Load the ad
      this.interstitialAd.load();

    } catch (error) {
      console.error('[AdManager] Error loading ad:', error);
      this.isAdLoading = false;
    }
  }

  /**
   * Show interstitial ad after card interpretation
   */
  async showAdAfterCard() {
    if (!APP_CONFIG.ads.interstitialAfterCard) {
      return;
    }

    return this.showInterstitialAd('after_card');
  }

  /**
   * Show interstitial ad before synthesis
   */
  async showAdBeforeSynthesis() {
    if (!APP_CONFIG.ads.interstitialBeforeSynthesis) {
      return;
    }

    return this.showInterstitialAd('before_synthesis');
  }

  /**
   * Show interstitial ad
   */
  async showInterstitialAd(placement = 'default') {
    if (!APP_CONFIG.ads.enabled) {
      console.log('[AdManager] Ads disabled (Premium version)');
      return;
    }

    console.log(`[AdManager] Attempting to show ad at: ${placement}`);

    if (!this.isAdLoaded || !this.interstitialAd) {
      console.log('[AdManager] Ad not ready, skipping...');
      // Try to load for next time
      this.loadInterstitialAd();
      return;
    }

    try {
      console.log('[AdManager] Showing interstitial ad...');
      await this.interstitialAd.show();
      this.isAdLoaded = false;
    } catch (error) {
      console.error('[AdManager] Error showing ad:', error);
      this.isAdLoaded = false;
      // Try to load next ad
      this.loadInterstitialAd();
    }
  }

  /**
   * Check if ads are enabled
   */
  isEnabled() {
    return APP_CONFIG.ads.enabled;
  }
}

// Export singleton instance
export default new AdManager();
