/**
 * UPGRADE PROMPT MANAGER
 * Shows polite rotating messages encouraging users to upgrade
 * No ads - just gentle reminders that support helps indie devs
 */

import { Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const PROMPT_COUNT_KEY = '@lunatiq_prompt_count';

const POLITE_MESSAGES = [
  {
    title: "💜 Support Indie Development",
    message: "LunatIQ Tarot is built by a solo developer raising a kid. Your $3.99 one-time upgrade helps keep this app alive and ad-free forever.\n\n✨ Unlock all spreads, save readings, and remove these prompts!"
  },
  {
    title: "🌙 Enjoying Your Readings?",
    message: "If LunatIQ's readings have brought you clarity, consider supporting continued development with a one-time $3.99 upgrade.\n\n🔓 Get all features + support an independent creator!"
  },
  {
    title: "✨ Premium Unlocks Everything",
    message: "For the price of a coffee, unlock all 9 spread types, save your reading history, and support solo indie development.\n\n💝 One payment, lifetime access!"
  },
  {
    title: "🎯 Ready for Deeper Readings?",
    message: "3-card spreads are powerful, but Celtic Cross and Horseshoe spreads offer profound insights.\n\n🚀 Upgrade for $3.99 to unlock all spreads + help feed a developer's kid!"
  },
  {
    title: "💎 No Subscriptions, Ever",
    message: "Just $3.99 one time. No recurring charges, no tricks. Unlock everything forever and support independent software development.\n\n🙏 Your support matters!"
  },
  {
    title: "🔮 Help Keep This Free Version Free",
    message: "Premium users make it possible to offer unlimited free readings to everyone. Join them for $3.99 and unlock all features!\n\n💫 Support indie development!"
  }
];

class UpgradePromptManager {
  constructor() {
    this.currentIndex = 0;
    this.initialized = false;
  }

  /**
   * Initialize - load the prompt count from storage
   */
  async initialize() {
    if (this.initialized) return;

    try {
      const count = await AsyncStorage.getItem(PROMPT_COUNT_KEY);
      this.currentIndex = count ? parseInt(count, 10) % POLITE_MESSAGES.length : 0;
      this.initialized = true;
      console.log('[UpgradePrompt] Initialized, next message index:', this.currentIndex);
    } catch (error) {
      console.error('[UpgradePrompt] Init error:', error);
      this.currentIndex = 0;
      this.initialized = true;
    }
  }

  /**
   * Get the next polite message and rotate
   */
  async getNextMessage() {
    if (!this.initialized) {
      await this.initialize();
    }

    const message = POLITE_MESSAGES[this.currentIndex];

    // Rotate to next message
    this.currentIndex = (this.currentIndex + 1) % POLITE_MESSAGES.length;

    // Save progress
    try {
      await AsyncStorage.setItem(PROMPT_COUNT_KEY, this.currentIndex.toString());
    } catch (error) {
      console.error('[UpgradePrompt] Save error:', error);
    }

    return message;
  }

  /**
   * Show upgrade prompt after a card reading
   */
  async showAfterCardPrompt(onUpgrade) {
    const { title, message } = await this.getNextMessage();

    Alert.alert(
      title,
      message,
      [
        { text: 'Maybe Later', style: 'cancel' },
        {
          text: 'Upgrade for $3.99',
          onPress: () => {
            console.log('[UpgradePrompt] User tapped upgrade');
            if (onUpgrade) onUpgrade();
          }
        }
      ]
    );
  }

  /**
   * Show upgrade prompt before synthesis
   */
  async showBeforeSynthesisPrompt(onUpgrade) {
    const { title, message } = await this.getNextMessage();

    Alert.alert(
      title,
      message,
      [
        { text: 'Continue to Reading', style: 'cancel' },
        {
          text: 'Upgrade for $3.99',
          onPress: () => {
            console.log('[UpgradePrompt] User tapped upgrade');
            if (onUpgrade) onUpgrade();
          }
        }
      ]
    );
  }

  /**
   * Show feature-locked prompt (for save/share)
   */
  showFeatureLockedPrompt(featureName, onUpgrade) {
    Alert.alert(
      `🔒 ${featureName} - Premium Only`,
      `${featureName} is available in the Premium version.\n\nUpgrade once for $3.99 to unlock:\n\n• All 9 spread types\n• Save reading history\n• Share & export readings\n• Support indie development\n• No more prompts!`,
      [
        { text: 'Not Now', style: 'cancel' },
        {
          text: 'Upgrade for $3.99',
          onPress: () => {
            console.log('[UpgradePrompt] User tapped upgrade from feature lock');
            if (onUpgrade) onUpgrade();
          }
        }
      ]
    );
  }
}

// Export singleton
export default new UpgradePromptManager();
