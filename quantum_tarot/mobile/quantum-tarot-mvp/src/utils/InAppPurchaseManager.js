/**
 * IN-APP PURCHASE MANAGER
 * Handles one-time $3.99 premium upgrade for both iOS and Android
 * Uses react-native-iap for cross-platform purchases
 */

import { Platform, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as StoreReview from 'expo-store-review';

// Graceful IAP import for Expo Go
let RNIap;
try {
  RNIap = require('react-native-iap');
} catch (e) {
  console.warn('[IAP] react-native-iap not available (Expo Go mode)');
  RNIap = null;
}

const PREMIUM_STATUS_KEY = '@lunatiq_premium_status';

// Product IDs - must match App Store Connect and Google Play Console
const PRODUCT_IDS = {
  ios: 'com.aphoticshaman.lunatiq.premium',
  android: 'com.aphoticshaman.lunatiq.premium'
};

class InAppPurchaseManager {
  constructor() {
    this.isPremium = false;
    this.isInitialized = false;
    this.products = [];
  }

  /**
   * Initialize IAP connection and load premium status
   */
  async initialize() {
    if (this.isInitialized) return;

    // Load premium status from storage
    try {
      const status = await AsyncStorage.getItem(PREMIUM_STATUS_KEY);
      this.isPremium = status === 'true';
      console.log('[IAP] Premium status:', this.isPremium);
    } catch (error) {
      console.error('[IAP] Error loading premium status:', error);
    }

    // Initialize IAP if available
    if (!RNIap) {
      console.log('[IAP] Not available in Expo Go - using stored premium status only');
      this.isInitialized = true;
      return;
    }

    try {
      await RNIap.initConnection();
      console.log('[IAP] Connection initialized');

      // Get available products
      const productId = Platform.OS === 'ios' ? PRODUCT_IDS.ios : PRODUCT_IDS.android;
      this.products = await RNIap.getProducts({ skus: [productId] });
      console.log('[IAP] Products loaded:', this.products);

      // Check for previous purchases (restore)
      await this.restorePurchases();

      this.isInitialized = true;
    } catch (error) {
      console.error('[IAP] Initialization error:', error);
      this.isInitialized = true; // Continue anyway, use storage only
    }
  }

  /**
   * Check if user has premium
   */
  hasPremium() {
    return this.isPremium;
  }

  /**
   * Purchase premium upgrade
   */
  async purchasePremium() {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.isPremium) {
      Alert.alert('Already Premium', 'You already have premium access!');
      return false;
    }

    // Expo Go fallback - simulate purchase for testing
    if (!RNIap) {
      Alert.alert(
        'Test Mode',
        'In-app purchases are not available in Expo Go.\n\nIn production builds, this will open the App Store/Play Store payment flow.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Simulate Purchase (Test)',
            onPress: async () => {
              await this.setPremium(true);
              Alert.alert('Success!', 'Premium unlocked (test mode)');
            }
          }
        ]
      );
      return false;
    }

    try {
      const productId = Platform.OS === 'ios' ? PRODUCT_IDS.ios : PRODUCT_IDS.android;

      console.log('[IAP] Requesting purchase for:', productId);

      // Request purchase
      await RNIap.requestPurchase({
        sku: productId,
        andDangerouslyFinishTransactionAutomaticallyIOS: false,
      });

      // Purchase listener is handled in setup (see setupPurchaseListener)

      return true;
    } catch (error) {
      console.error('[IAP] Purchase error:', error);

      if (error.code === 'E_USER_CANCELLED') {
        console.log('[IAP] User cancelled purchase');
      } else {
        Alert.alert('Purchase Failed', 'Something went wrong. Please try again later.');
      }

      return false;
    }
  }

  /**
   * Restore previous purchases
   */
  async restorePurchases() {
    if (!RNIap) {
      console.log('[IAP] Cannot restore purchases in Expo Go');
      return;
    }

    try {
      console.log('[IAP] Restoring purchases...');

      const purchases = Platform.OS === 'ios'
        ? await RNIap.getAvailablePurchases()
        : await RNIap.getPurchaseHistory();

      console.log('[IAP] Found purchases:', purchases);

      const productId = Platform.OS === 'ios' ? PRODUCT_IDS.ios : PRODUCT_IDS.android;
      const hasPremiumPurchase = purchases.some(p => p.productId === productId);

      if (hasPremiumPurchase) {
        await this.setPremium(true);
        console.log('[IAP] Premium restored from previous purchase');
      }
    } catch (error) {
      console.error('[IAP] Restore error:', error);
    }
  }

  /**
   * Set premium status
   */
  async setPremium(isPremium) {
    this.isPremium = isPremium;

    try {
      await AsyncStorage.setItem(PREMIUM_STATUS_KEY, isPremium.toString());
      console.log('[IAP] Premium status saved:', isPremium);

      // Ask for rating if they just upgraded
      if (isPremium) {
        setTimeout(() => this.requestReview(), 2000);
      }
    } catch (error) {
      console.error('[IAP] Error saving premium status:', error);
    }
  }

  /**
   * Setup purchase update listener
   */
  setupPurchaseListener() {
    if (!RNIap) return;

    this.purchaseUpdateSubscription = RNIap.purchaseUpdatedListener(async (purchase) => {
      console.log('[IAP] Purchase update:', purchase);

      const receipt = purchase.transactionReceipt;

      if (receipt) {
        try {
          // Acknowledge the purchase
          await this.setPremium(true);

          // Finish the transaction
          if (Platform.OS === 'ios') {
            await RNIap.finishTransaction({ purchase, isConsumable: false });
          } else {
            await RNIap.acknowledgePurchaseAndroid({ token: purchase.purchaseToken });
          }

          Alert.alert(
            '🎉 Premium Unlocked!',
            'You now have access to all features! Thank you for supporting indie development.',
            [{ text: 'Awesome!' }]
          );

          console.log('[IAP] Purchase completed and acknowledged');
        } catch (error) {
          console.error('[IAP] Error finishing transaction:', error);
        }
      }
    });

    this.purchaseErrorSubscription = RNIap.purchaseErrorListener((error) => {
      console.error('[IAP] Purchase error listener:', error);
      if (error.code !== 'E_USER_CANCELLED') {
        Alert.alert('Purchase Error', 'Something went wrong. Please try again.');
      }
    });
  }

  /**
   * Clean up listeners
   */
  cleanup() {
    if (this.purchaseUpdateSubscription) {
      this.purchaseUpdateSubscription.remove();
    }
    if (this.purchaseErrorSubscription) {
      this.purchaseErrorSubscription.remove();
    }
    if (RNIap) {
      RNIap.endConnection();
    }
  }

  /**
   * Request app store review after upgrade
   */
  async requestReview() {
    try {
      const isAvailable = await StoreReview.isAvailableAsync();
      if (isAvailable) {
        await StoreReview.requestReview();
      }
    } catch (error) {
      console.log('[IAP] Review request not available');
    }
  }

  /**
   * Get product price info
   */
  getPrice() {
    if (this.products.length > 0) {
      return this.products[0].localizedPrice || '$3.99';
    }
    return '$3.99';
  }
}

// Export singleton
export default new InAppPurchaseManager();
