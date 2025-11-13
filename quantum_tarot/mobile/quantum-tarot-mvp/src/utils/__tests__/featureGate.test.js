/**
 * Unit tests for Feature Gate system
 */

import { FeatureGate } from '../featureGate';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock config - will be overridden per test
jest.mock('../../config/config', () => ({
  APP_CONFIG: {
    version: 'free',
    features: {
      unlimitedReadings: false,
      dailyReadingLimit: 1,
      availableSpreads: ['single_card', 'three_card'],
      availableReadingTypes: ['career', 'romance', 'wellness'],
      readingHistory: false,
      exportReadings: false
    },
    monetization: {
      upgradePrompt: true,
      upgradeUrl: 'https://play.google.com/store/apps/details?id=com.quantum.premium',
      upgradePrice: '$3.99'
    }
  }
}));

describe('FeatureGate', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Free Version', () => {
    describe('canDrawReading', () => {
      it('should allow reading if within daily limit', async () => {
        AsyncStorage.getItem.mockResolvedValue(null); // No previous reading
        const canDraw = await FeatureGate.canDrawReading();
        expect(canDraw).toBe(true);
      });

      it('should deny reading if daily limit reached', async () => {
        const oneHourAgo = Date.now() - (60 * 60 * 1000);
        AsyncStorage.getItem.mockResolvedValue(oneHourAgo.toString());

        const canDraw = await FeatureGate.canDrawReading();
        expect(canDraw).toBe(false);
      });

      it('should allow reading after 24 hours', async () => {
        const yesterdayPlus = Date.now() - (25 * 60 * 60 * 1000);
        AsyncStorage.getItem.mockResolvedValue(yesterdayPlus.toString());

        const canDraw = await FeatureGate.canDrawReading();
        expect(canDraw).toBe(true);
      });
    });

    describe('isSpreadAvailable', () => {
      it('should allow single_card spread', () => {
        expect(FeatureGate.isSpreadAvailable('single_card')).toBe(true);
      });

      it('should allow three_card spread', () => {
        expect(FeatureGate.isSpreadAvailable('three_card')).toBe(true);
      });

      it('should deny celtic_cross spread', () => {
        expect(FeatureGate.isSpreadAvailable('celtic_cross')).toBe(false);
      });

      it('should deny relationship spread', () => {
        expect(FeatureGate.isSpreadAvailable('relationship')).toBe(false);
      });
    });

    describe('isReadingTypeAvailable', () => {
      it('should allow career reading', () => {
        expect(FeatureGate.isReadingTypeAvailable('career')).toBe(true);
      });

      it('should allow romance reading', () => {
        expect(FeatureGate.isReadingTypeAvailable('romance')).toBe(true);
      });

      it('should allow wellness reading', () => {
        expect(FeatureGate.isReadingTypeAvailable('wellness')).toBe(true);
      });

      it('should deny spiritual reading', () => {
        expect(FeatureGate.isReadingTypeAvailable('spiritual')).toBe(false);
      });

      it('should deny shadow_work reading', () => {
        expect(FeatureGate.isReadingTypeAvailable('shadow_work')).toBe(false);
      });
    });

    describe('Premium Features', () => {
      it('should deny reading history access', () => {
        expect(FeatureGate.canAccessReadingHistory()).toBe(false);
      });

      it('should deny export reading access', () => {
        expect(FeatureGate.canExportReading()).toBe(false);
      });

      it('should show upgrade prompt', () => {
        expect(FeatureGate.shouldShowUpgradePrompt()).toBe(true);
      });

      it('should return upgrade URL', () => {
        const url = FeatureGate.getUpgradeUrl();
        expect(url).toContain('play.google.com');
        expect(url).toContain('premium');
      });
    });
  });

  describe('Premium Version', () => {
    beforeAll(() => {
      // Override config for premium tests
      jest.doMock('../../config/config', () => ({
        APP_CONFIG: {
          version: 'premium',
          features: {
            unlimitedReadings: true,
            availableSpreads: [
              'single_card', 'three_card', 'celtic_cross',
              'relationship', 'decision_analysis'
            ],
            availableReadingTypes: [
              'career', 'romance', 'wellness', 'spiritual',
              'shadow_work', 'year_ahead'
            ],
            readingHistory: true,
            exportReadings: true
          },
          monetization: {
            upgradePrompt: false,
            upgradeUrl: null,
            upgradePrice: null
          }
        }
      }));
    });

    it('should allow unlimited readings', async () => {
      // Even with recent reading
      AsyncStorage.getItem.mockResolvedValue(Date.now().toString());
      const canDraw = await FeatureGate.canDrawReading();
      expect(canDraw).toBe(true);
    });

    it('should allow all spreads', () => {
      expect(FeatureGate.isSpreadAvailable('celtic_cross')).toBe(true);
      expect(FeatureGate.isSpreadAvailable('relationship')).toBe(true);
      expect(FeatureGate.isSpreadAvailable('decision_analysis')).toBe(true);
    });

    it('should allow all reading types', () => {
      expect(FeatureGate.isReadingTypeAvailable('spiritual')).toBe(true);
      expect(FeatureGate.isReadingTypeAvailable('shadow_work')).toBe(true);
      expect(FeatureGate.isReadingTypeAvailable('year_ahead')).toBe(true);
    });

    it('should allow reading history access', () => {
      expect(FeatureGate.canAccessReadingHistory()).toBe(true);
    });

    it('should allow export reading', () => {
      expect(FeatureGate.canExportReading()).toBe(true);
    });

    it('should not show upgrade prompt', () => {
      expect(FeatureGate.shouldShowUpgradePrompt()).toBe(false);
    });
  });
});
