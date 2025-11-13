import AsyncStorage from '@react-native-async-storage/async-storage';

/**
 * User Profile Management
 */
export async function saveUserProfile(name, birthday, pronouns) {
  try {
    await AsyncStorage.multiSet([
      ['userName', name],
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
    const [[, name], [, birthday], [, pronouns]] = await AsyncStorage.multiGet([
      'userName',
      'userBirthday',
      'userPronouns'
    ]);
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
    return profileJson ? JSON.parse(profileJson) : null;
  } catch (error) {
    console.error('Failed to load personality profile:', error);
    return null;
  }
}

/**
 * Reading Management
 */
export async function saveReading(reading) {
  try {
    const existingReadings = await AsyncStorage.getItem('readings');
    const readings = existingReadings ? JSON.parse(existingReadings) : [];

    readings.push({
      ...reading,
      savedAt: Date.now()
    });

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
    return readingsJson ? JSON.parse(readingsJson) : [];
  } catch (error) {
    console.error('Failed to load readings:', error);
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
    await AsyncStorage.setItem('lastReadingTimestamp', Date.now().toString());
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
    const now = Date.now();
    const oneDayMs = 24 * 60 * 60 * 1000;
    const timeSinceLastReading = now - lastReadingTime;

    if (timeSinceLastReading >= oneDayMs) {
      return 0;
    }

    return oneDayMs - timeSinceLastReading;
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
    return premium === 'true';
  } catch (error) {
    console.error('Failed to check premium status:', error);
    return false;
  }
}

export async function setPremiumStatus(isPremium) {
  try {
    await AsyncStorage.setItem('isPremium', isPremium ? 'true' : 'false');
    return true;
  } catch (error) {
    console.error('Failed to set premium status:', error);
    return false;
  }
}
