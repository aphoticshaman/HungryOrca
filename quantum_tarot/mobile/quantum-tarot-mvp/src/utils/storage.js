import AsyncStorage from '@react-native-async-storage/async-storage';

/**
 * User Profile Management
 */
export async function saveUserProfile(name, birthday, pronouns) {
  try {
    // Validate inputs
    if (!name || typeof name !== 'string' || name.trim().length === 0) {
      throw new Error('Invalid name');
    }
    if (!birthday || typeof birthday !== 'string') {
      throw new Error('Invalid birthday');
    }
    if (!pronouns || typeof pronouns !== 'string') {
      throw new Error('Invalid pronouns');
    }

    await AsyncStorage.multiSet([
      ['userName', name.trim()],
      ['userBirthday', birthday],
      ['userPronouns', pronouns]
    ]);
    return true;
  } catch (error) {
    console.error('Failed to save user profile:', error);
    return false;
  }
}

export async function getUserProfile() {
  try {
    const results = await AsyncStorage.multiGet([
      'userName',
      'userBirthday',
      'userPronouns'
    ]);

    // Validate results structure
    if (!results || !Array.isArray(results) || results.length !== 3) {
      throw new Error('Invalid storage results');
    }

    const [[, name], [, birthday], [, pronouns]] = results;
    return { name, birthday, pronouns };
  } catch (error) {
    console.error('Failed to load user profile:', error);
    return { name: null, birthday: null, pronouns: null };
  }
}

/**
 * Personality Profile Management
 */
export async function savePersonalityProfile(readingType, profile) {
  try {
    // Validate inputs
    if (!readingType || typeof readingType !== 'string') {
      throw new Error('Invalid reading type');
    }
    if (!profile || typeof profile !== 'object') {
      throw new Error('Invalid profile data');
    }

    const key = `profile_${readingType}`;
    await AsyncStorage.setItem(key, JSON.stringify(profile));
    return true;
  } catch (error) {
    console.error('Failed to save personality profile:', error);
    return false;
  }
}

export async function getPersonalityProfile(readingType) {
  try {
    const key = `profile_${readingType}`;
    const profileJson = await AsyncStorage.getItem(key);
    if (!profileJson) return null;

    const profile = JSON.parse(profileJson);
    // Validate profile structure
    if (!profile || typeof profile !== 'object') {
      throw new Error('Invalid profile data');
    }
    return profile;
  } catch (error) {
    console.error('Failed to load personality profile:', error);
    // Clear corrupted data
    try {
      await AsyncStorage.removeItem(`profile_${readingType}`);
    } catch (e) {
      // Ignore cleanup errors
    }
    return null;
  }
}

/**
 * Reading Management
 */
export async function saveReading(reading) {
  try {
    if (!reading || typeof reading !== 'object') {
      throw new Error('Invalid reading data');
    }

    const existingReadings = await AsyncStorage.getItem('readings');
    let readings = [];

    if (existingReadings) {
      try {
        readings = JSON.parse(existingReadings);
        if (!Array.isArray(readings)) {
          console.warn('Corrupted readings data, resetting');
          readings = [];
        }
      } catch (parseError) {
        console.warn('Failed to parse readings, resetting:', parseError);
        readings = [];
      }
    }

    readings.push({
      ...reading,
      savedAt: Date.now()
    });

    // Limit to last 100 readings to prevent storage bloat
    if (readings.length > 100) {
      readings = readings.slice(-100);
    }

    await AsyncStorage.setItem('readings', JSON.stringify(readings));
    return true;
  } catch (error) {
    console.error('Failed to save reading:', error);
    return false;
  }
}

export async function getReadings() {
  try {
    const readingsJson = await AsyncStorage.getItem('readings');
    if (!readingsJson) return [];

    const readings = JSON.parse(readingsJson);
    if (!Array.isArray(readings)) {
      throw new Error('Corrupted readings data');
    }
    return readings;
  } catch (error) {
    console.error('Failed to load readings:', error);
    // Clear corrupted data
    try {
      await AsyncStorage.removeItem('readings');
    } catch (e) {
      // Ignore cleanup errors
    }
    return [];
  }
}

/**
 * Daily Reading Limit (Free Tier)
 */
export async function canDrawReading() {
  try {
    const lastReading = await AsyncStorage.getItem('lastReadingTimestamp');
    if (!lastReading) {
      return true;
    }

    const lastReadingTime = parseInt(lastReading, 10);

    // Validate parsed timestamp
    if (isNaN(lastReadingTime) || lastReadingTime < 0) {
      console.warn('Corrupted timestamp, clearing');
      await AsyncStorage.removeItem('lastReadingTimestamp');
      return true;
    }

    const now = Date.now();
    const oneDayMs = 24 * 60 * 60 * 1000;

    // Check if 24 hours have passed
    return (now - lastReadingTime) >= oneDayMs;
  } catch (error) {
    console.error('Failed to check reading limit:', error);
    return true; // Allow on error
  }
}

export async function recordReading() {
  try {
    const timestamp = Date.now();
    if (!timestamp || timestamp < 0) {
      throw new Error('Invalid timestamp');
    }
    await AsyncStorage.setItem('lastReadingTimestamp', timestamp.toString());
    return true;
  } catch (error) {
    console.error('Failed to record reading:', error);
    return false;
  }
}

export async function getTimeUntilNextReading() {
  try {
    const lastReading = await AsyncStorage.getItem('lastReadingTimestamp');
    if (!lastReading) {
      return 0;
    }

    const lastReadingTime = parseInt(lastReading, 10);

    // Validate parsed timestamp
    if (isNaN(lastReadingTime) || lastReadingTime < 0) {
      console.warn('Corrupted timestamp, clearing');
      await AsyncStorage.removeItem('lastReadingTimestamp');
      return 0;
    }

    const now = Date.now();
    const oneDayMs = 24 * 60 * 60 * 1000;
    const timeSinceLastReading = now - lastReadingTime;

    if (timeSinceLastReading >= oneDayMs) {
      return 0;
    }

    const remaining = oneDayMs - timeSinceLastReading;
    return remaining > 0 ? remaining : 0;
  } catch (error) {
    console.error('Failed to calculate time until next reading:', error);
    return 0;
  }
}

/**
 * Premium Status
 */
export async function isPremiumUser() {
  try {
    const premium = await AsyncStorage.getItem('isPremium');
    if (!premium) {
      return false;
    }
    // Validate value is actually a boolean string
    if (premium !== 'true' && premium !== 'false') {
      console.warn('Corrupted premium status, resetting to false');
      await AsyncStorage.setItem('isPremium', 'false');
      return false;
    }
    return premium === 'true';
  } catch (error) {
    console.error('Failed to check premium status:', error);
    return false;
  }
}

export async function setPremiumStatus(isPremium) {
  try {
    // Validate input is boolean-like
    if (typeof isPremium !== 'boolean') {
      throw new Error('isPremium must be a boolean');
    }
    await AsyncStorage.setItem('isPremium', isPremium ? 'true' : 'false');
    return true;
  } catch (error) {
    console.error('Failed to set premium status:', error);
    return false;
  }
}
