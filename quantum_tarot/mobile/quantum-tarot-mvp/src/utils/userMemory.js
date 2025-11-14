/**
 * XYZA-1: TEMPORAL MEMORY LAYER
 * Cross-references past readings to create "oracle that knows you" experience
 *
 * This closes the competitive gap vs ChatGPT by:
 * 1. Remembering user's history across sessions
 * 2. Adapting interpretations based on past patterns
 * 3. Tracking which advice actually gets followed
 */

import { getActionHistory, getCompletionStats } from './actionTracker';
import { CARD_DATABASE } from '../data/cardDatabase';

/**
 * Get user's memory context for interpretation personalization
 * @returns {Object} Memory context to inject into interpretations
 */
export async function getUserMemory() {
  try {
    const history = await getActionHistory();
    const stats = await getCompletionStats();

    if (!history || history.length === 0) {
      return {
        hasHistory: false,
        readingCount: 0,
        message: null
      };
    }

    const memory = {
      hasHistory: true,
      readingCount: history.length,

      // Behavioral patterns
      completionRate: stats.completionRate,
      followThroughStyle: determineFollowThroughStyle(stats),

      // Card patterns
      frequentCards: findFrequentCards(history),
      recurringThemes: findRecurringThemes(history),

      // Temporal patterns
      lastReading: history[0],
      daysSinceLastReading: getDaysSince(history[0].timestamp),
      readingFrequency: calculateReadingFrequency(history),

      // Intention patterns
      commonIntentionTypes: analyzeIntentionPatterns(history),

      // Generate personalized memory reference
      personalizedGreeting: generatePersonalizedGreeting(history, stats)
    };

    return memory;
  } catch (error) {
    console.error('Error loading user memory:', error);
    return { hasHistory: false, readingCount: 0, message: null };
  }
}

/**
 * Determine user's follow-through style from completion stats
 */
function determineFollowThroughStyle(stats) {
  const rate = stats.completionRate;

  if (rate >= 70) return 'high-executor'; // Takes action reliably
  if (rate >= 40) return 'selective-executor'; // Follows through on some
  if (rate >= 15) return 'contemplative'; // Reads but rarely acts
  return 'explorer'; // New or browsing
}

/**
 * Find cards that appear frequently in user's history
 */
function findFrequentCards(history) {
  const cardCounts = {};

  history.forEach(reading => {
    // Note: We don't have cards stored in action history yet
    // This will need to be enhanced when we store drawn cards
    // For now, return empty array
  });

  return []; // TODO: Store cards in reading history
}

/**
 * Find recurring themes in past intentions
 */
function findRecurringThemes(history) {
  const themePatterns = {
    career: /career|job|work|business|professional|boss|manager|promotion|salary/i,
    romance: /love|relationship|partner|dating|romance|boyfriend|girlfriend|husband|wife|marriage/i,
    personal_growth: /growth|healing|development|transform|self|consciousness|spiritual|therapy/i,
    decision: /should i|decide|choice|option|path|whether to/i,
    finance: /money|financial|income|debt|savings|invest|wealth|budget/i,
    wellness: /health|wellness|fitness|mental|anxiety|stress|energy|sleep/i,
    shadow: /fear|anxiety|stuck|block|pattern|wound|trauma|shadow/i
  };

  const themeCounts = {};

  history.forEach(reading => {
    const intention = reading.intention || '';
    Object.entries(themePatterns).forEach(([theme, pattern]) => {
      if (pattern.test(intention)) {
        themeCounts[theme] = (themeCounts[theme] || 0) + 1;
      }
    });
  });

  // Return themes that appear 2+ times, sorted by frequency
  return Object.entries(themeCounts)
    .filter(([_, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .map(([theme, count]) => ({ theme, count }));
}

/**
 * Calculate days since a timestamp
 */
function getDaysSince(timestamp) {
  const now = Date.now();
  const diff = now - timestamp;
  return Math.floor(diff / (1000 * 60 * 60 * 24));
}

/**
 * Calculate reading frequency (readings per week)
 */
function calculateReadingFrequency(history) {
  if (history.length < 2) return 'new-user';

  const oldest = history[history.length - 1].timestamp;
  const newest = history[0].timestamp;
  const daySpan = getDaysSince(oldest) - getDaysSince(newest);
  const weekSpan = daySpan / 7;

  if (weekSpan === 0) return 'very-frequent'; // Multiple in same week

  const readingsPerWeek = history.length / weekSpan;

  if (readingsPerWeek >= 3) return 'very-frequent';
  if (readingsPerWeek >= 1) return 'regular';
  if (readingsPerWeek >= 0.25) return 'occasional';
  return 'rare';
}

/**
 * Analyze patterns in user's intention types
 */
function analyzeIntentionPatterns(history) {
  const themes = findRecurringThemes(history);
  return themes.slice(0, 3); // Top 3 recurring themes
}

/**
 * Generate personalized greeting based on memory
 */
function generatePersonalizedGreeting(history, stats) {
  const daysSince = getDaysSince(history[0].timestamp);
  const completionRate = stats.completionRate;
  const readingCount = history.length;
  const themes = findRecurringThemes(history);

  let greeting = '';

  // Time-based acknowledgment
  if (daysSince === 0) {
    greeting = `Welcome back. `;
  } else if (daysSince === 1) {
    greeting = `Back for day 2. `;
  } else if (daysSince <= 7) {
    greeting = `${daysSince} days since your last reading. `;
  } else if (daysSince <= 30) {
    greeting = `${daysSince} days since your last reading—let's see what's changed. `;
  } else {
    greeting = `${daysSince} days since your last reading. A lot can shift in that time. `;
  }

  // Reading count acknowledgment
  if (readingCount >= 20) {
    greeting += `This is reading #${readingCount}. You're building serious history with this oracle. `;
  } else if (readingCount >= 10) {
    greeting += `Reading #${readingCount}. The patterns are starting to show. `;
  } else if (readingCount >= 5) {
    greeting += `Reading #${readingCount}. `;
  }

  // Completion rate acknowledgment (only if >5 readings)
  if (readingCount >= 5) {
    if (completionRate >= 70) {
      greeting += `You've completed ${Math.round(completionRate)}% of past actions—high-executor energy. The cards trust you to follow through. `;
    } else if (completionRate >= 40) {
      greeting += `You've completed ${Math.round(completionRate)}% of past actions—selective executor. You act when it resonates. `;
    } else if (completionRate >= 15) {
      greeting += `You've completed ${Math.round(completionRate)}% of past actions. Consider: what makes some advice stick and others slide? `;
    } else if (stats.total > 0) {
      greeting += `Few actions completed so far. These readings are contemplative for you—and that's valid. But try executing just ONE action from today's reading. `;
    }
  }

  // Recurring theme acknowledgment
  if (themes.length >= 2) {
    const themeNames = themes.slice(0, 2).map(t => t.theme).join(' and ');
    greeting += `I notice ${themeNames} keeps coming up for you. Today's reading will reference this pattern. `;
  } else if (themes.length === 1) {
    greeting += `${themes[0].theme.charAt(0).toUpperCase() + themes[0].theme.slice(1)} is a recurring theme in your readings. `;
  }

  return greeting.trim();
}

/**
 * Generate memory-aware context for a specific card interpretation
 * This gets injected into the interpretation layers
 */
export async function getCardMemoryContext(cardIndex, intention) {
  const memory = await getUserMemory();

  if (!memory.hasHistory || memory.readingCount < 3) {
    return null; // Not enough history yet
  }

  // Check if this card has appeared before
  // TODO: Implement when we store cards in reading history

  // Check if current intention relates to past patterns
  const relatedThemes = memory.commonIntentionTypes.filter(t => {
    const intentionLower = (intention || '').toLowerCase();
    return intentionLower.includes(t.theme);
  });

  if (relatedThemes.length > 0) {
    const theme = relatedThemes[0];
    return {
      hasPatternMatch: true,
      message: `This is the ${theme.count}${getOrdinalSuffix(theme.count)} time you've asked about ${theme.theme}. Notice the pattern?`
    };
  }

  return null;
}

/**
 * Get ordinal suffix (1st, 2nd, 3rd, 4th, etc.)
 */
function getOrdinalSuffix(n) {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return (s[(v - 20) % 10] || s[v] || s[0]);
}

/**
 * Generate memory-aware action recommendations
 * Adapts based on user's follow-through history
 */
export async function adaptActionsToMemory(actions, readingType) {
  const memory = await getUserMemory();

  if (!memory.hasHistory || memory.readingCount < 5) {
    return actions; // Not enough data to adapt yet
  }

  const style = memory.followThroughStyle;

  // High executors: Give them ambitious, multi-step actions
  if (style === 'high-executor') {
    // Already getting good actions, maybe add urgency
    return actions.map(action => {
      // Don't modify if already has timeframe
      if (action.includes('within') || action.includes('this week') || action.includes('today')) {
        return action;
      }
      return action + ' (You execute—do this within 48 hours.)';
    });
  }

  // Selective executors: Emphasize which action to prioritize
  if (style === 'selective-executor') {
    return [
      `[PRIORITY] ${actions[0]}`,
      actions[1],
      actions[2]
    ];
  }

  // Contemplative: Simplify and ask for just ONE action
  if (style === 'contemplative') {
    return [
      `Just do ONE: ${actions[0]}`,
      `(Optional) ${actions[1]}`,
      `(Optional) ${actions[2]}`
    ];
  }

  // Explorer: Keep standard actions
  return actions;
}
