/**
 * XYZA CYCLE 2: Action Tracking System
 * Stores action completion data locally for behavioral pattern recognition
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@lunatiq_action_history';

/**
 * Action data structure:
 * {
 *   readingId: string (quantum seed),
 *   timestamp: number,
 *   intention: string,
 *   spreadType: string,
 *   actions: [
 *     {
 *       text: string,
 *       completed: boolean | null, // true=done, false=not done, null=skipped
 *       completedAt: number | null
 *     }
 *   ]
 * }
 */

/**
 * Save action completion status
 */
export async function saveActionStatus(readingId, actionIndex, completed) {
  try {
    const historyJson = await AsyncStorage.getItem(STORAGE_KEY);
    const history = historyJson ? JSON.parse(historyJson) : [];

    const readingIndex = history.findIndex(r => r.readingId === readingId);

    if (readingIndex >= 0) {
      history[readingIndex].actions[actionIndex] = {
        ...history[readingIndex].actions[actionIndex],
        completed,
        completedAt: completed !== null ? Date.now() : null
      };
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    }

    return true;
  } catch (error) {
    console.error('Error saving action status:', error);
    return false;
  }
}

/**
 * Initialize reading with actions
 */
export async function initializeReading(readingId, intention, spreadType, actions) {
  try {
    const historyJson = await AsyncStorage.getItem(STORAGE_KEY);
    const history = historyJson ? JSON.parse(historyJson) : [];

    const reading = {
      readingId,
      timestamp: Date.now(),
      intention,
      spreadType,
      actions: actions.map(text => ({
        text,
        completed: null,
        completedAt: null
      }))
    };

    history.unshift(reading); // Add to beginning

    // Keep only last 50 readings
    if (history.length > 50) {
      history.splice(50);
    }

    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    return true;
  } catch (error) {
    console.error('Error initializing reading:', error);
    return false;
  }
}

/**
 * Get action status for a reading
 */
export async function getReadingActions(readingId) {
  try {
    const historyJson = await AsyncStorage.getItem(STORAGE_KEY);
    const history = historyJson ? JSON.parse(historyJson) : [];

    const reading = history.find(r => r.readingId === readingId);
    return reading ? reading.actions : null;
  } catch (error) {
    console.error('Error getting reading actions:', error);
    return null;
  }
}

/**
 * Get full action history (for future analytics)
 */
export async function getActionHistory() {
  try {
    const historyJson = await AsyncStorage.getItem(STORAGE_KEY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error('Error getting action history:', error);
    return [];
  }
}

/**
 * Get completion stats (for future use)
 */
export async function getCompletionStats() {
  try {
    const history = await getActionHistory();

    let totalActions = 0;
    let completedActions = 0;
    let skippedActions = 0;

    history.forEach(reading => {
      reading.actions.forEach(action => {
        totalActions++;
        if (action.completed === true) completedActions++;
        if (action.completed === null) skippedActions++;
      });
    });

    return {
      total: totalActions,
      completed: completedActions,
      skipped: skippedActions,
      pending: totalActions - completedActions - skippedActions,
      completionRate: totalActions > 0 ? (completedActions / totalActions) * 100 : 0
    };
  } catch (error) {
    console.error('Error getting completion stats:', error);
    return { total: 0, completed: 0, skipped: 0, pending: 0, completionRate: 0 };
  }
}
