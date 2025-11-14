/**
 * FEATURE GATE
 * Controls access to premium features based on app version
 * Works with both Free and Premium configs
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

// Import config - will be swapped during build
import { APP_CONFIG } from '../config/config.free'; // Default to free, swap for premium build

const STORAGE_KEYS = {
  LAST_READING_DATE: '@last_reading_date',
  READING_COUNT_TODAY: '@reading_count_today'
};

/**
 * Feature Gate - Controls what users can access
 */
export class FeatureGate {
  // ═══════════════════════════════════════════════════════════
  // READING LIMITS
  // ═══════════════════════════════════════════════════════════

  static async canDrawReading() {
    // Premium: always allowed
    if (APP_CONFIG.features.unlimitedReadings) {
      return { allowed: true, reason: 'unlimited' };
    }

    // Free: check daily limit
    const limit = APP_CONFIG.features.dailyReadingLimit;
    const { count, lastDate } = await this.getTodayReadingCount();

    const today = this.getDateString(new Date());
    const isToday = lastDate === today;

    if (!isToday || count < limit) {
      return { allowed: true, reason: 'within_limit', remaining: isToday ? limit - count : limit };
    }

    const nextReading = this.getNextReadingTime();
    return {
      allowed: false,
      reason: 'limit_reached',
      nextReading,
      hoursUntil: this.getHoursUntil(nextReading)
    };
  }

  static async recordReading() {
    if (APP_CONFIG.features.unlimitedReadings) return; // Premium doesn't track

    const today = this.getDateString(new Date());
    const { count, lastDate } = await this.getTodayReadingCount();

    const newCount = (lastDate === today) ? count + 1 : 1;

    await AsyncStorage.setItem(STORAGE_KEYS.LAST_READING_DATE, today);
    await AsyncStorage.setItem(STORAGE_KEYS.READING_COUNT_TODAY, newCount.toString());
  }

  static async getTodayReadingCount() {
    try {
      const lastDate = await AsyncStorage.getItem(STORAGE_KEYS.LAST_READING_DATE);
      const count = await AsyncStorage.getItem(STORAGE_KEYS.READING_COUNT_TODAY);

      return {
        lastDate: lastDate || '',
        count: count ? parseInt(count) : 0
      };
    } catch (error) {
      console.error('Error getting reading count:', error);
      return { lastDate: '', count: 0 };
    }
  }

  static getNextReadingTime() {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);
    return tomorrow;
  }

  static getHoursUntil(futureDate) {
    const now = new Date();
    const diff = futureDate - now;
    return Math.ceil(diff / (1000 * 60 * 60));
  }

  static getDateString(date) {
    return date.toISOString().split('T')[0]; // YYYY-MM-DD
  }

  // ═══════════════════════════════════════════════════════════
  // SPREAD ACCESS
  // ═══════════════════════════════════════════════════════════

  static isSpreadAvailable(spreadType) {
    if (APP_CONFIG.features.allSpreadTypes) return true;
    return APP_CONFIG.features.availableSpreads.includes(spreadType);
  }

  static getAvailableSpreads() {
    return APP_CONFIG.features.availableSpreads;
  }

  static getLockedSpreads() {
    const all = [
      'single_card', 'three_card', 'goal_progress', 'decision_analysis',
      'daily_checkin', 'clairvoyant_predictive', 'relationship',
      'celtic_cross', 'horseshoe'
    ];
    return all.filter(s => !this.isSpreadAvailable(s));
  }

  // ═══════════════════════════════════════════════════════════
  // READING TYPE ACCESS
  // ═══════════════════════════════════════════════════════════

  static isReadingTypeAvailable(readingType) {
    if (APP_CONFIG.features.allReadingTypes) return true;
    return APP_CONFIG.features.availableReadingTypes.includes(readingType);
  }

  static getAvailableReadingTypes() {
    return APP_CONFIG.features.availableReadingTypes;
  }

  static getLockedReadingTypes() {
    const all = [
      'career', 'romance', 'wellness', 'spiritual',
      'decision', 'general', 'shadow_work', 'year_ahead'
    ];
    return all.filter(rt => !this.isReadingTypeAvailable(rt));
  }

  // ═══════════════════════════════════════════════════════════
  // FEATURE ACCESS
  // ═══════════════════════════════════════════════════════════

  static canAccessReadingHistory() {
    return APP_CONFIG.features.readingHistory;
  }

  static canExportReading() {
    return APP_CONFIG.features.exportReadings;
  }

  static hasAdvancedInterpretations() {
    return APP_CONFIG.features.advancedInterpretations;
  }

  static hasMetaAnalysis() {
    return APP_CONFIG.features.metaAnalysis;
  }

  static canSelectThemes() {
    return APP_CONFIG.features.themeSelection;
  }

  static canFlipCards() {
    return APP_CONFIG.features.cardFlip;
  }

  // ═══════════════════════════════════════════════════════════
  // UPGRADE PROMPTS
  // ═══════════════════════════════════════════════════════════

  static shouldShowUpgradePrompt() {
    return APP_CONFIG.monetization.upgradePrompt;
  }

  static getUpgradeUrl() {
    return APP_CONFIG.monetization.upgradeUrl;
  }

  static getUpgradePrice() {
    return APP_CONFIG.monetization.upgradePrice;
  }

  static getUpgradeMessage() {
    return APP_CONFIG.monetization.upgradeMessage;
  }

  static showPremiumBadges() {
    return APP_CONFIG.ui.showPremiumBadges;
  }

  static showUpgradeButton() {
    return APP_CONFIG.ui.showUpgradeButton;
  }

  // ═══════════════════════════════════════════════════════════
  // APP INFO
  // ═══════════════════════════════════════════════════════════

  static getAppVersion() {
    return APP_CONFIG.version;
  }

  static isPremium() {
    return APP_CONFIG.version === 'premium';
  }

  static isFree() {
    return APP_CONFIG.version === 'free';
  }

  static getAppName() {
    return APP_CONFIG.displayName;
  }

  static getTagline() {
    return APP_CONFIG.branding.tagline;
  }

  // ═══════════════════════════════════════════════════════════
  // FULL FEATURE LIST (for upgrade prompt)
  // ═══════════════════════════════════════════════════════════

  static getPremiumFeatures() {
    return [
      'Unlimited readings',
      'All 9 spread types',
      'All 8 reading types',
      'Reading history with search',
      'Export readings',
      'Advanced meta-analysis',
      'Celtic Cross spread',
      'Year Ahead reading',
      'No ads, ever'
    ];
  }

  static getFreeFeatures() {
    return [
      '1 reading per day',
      '2 spread types',
      '3 reading types',
      'Quantum randomness',
      'Offline operation',
      'Card symbolism',
      'No ads, no tracking'
    ];
  }
}

export default FeatureGate;
