/**
 * TEMPORAL PREDICTION ENGINE
 *
 * Professional tarot readers give SPECIFIC timeframes, not vague "soon" statements.
 * This engine generates concrete timing predictions based on:
 * - Card numerology (3 of Cups = 3 days/weeks/months)
 * - Astrological transits (Mercury retrograde dates, moon phases)
 * - Seasonal cycles
 * - Reading type context
 *
 * Makes readings feel MORE REAL and TESTABLE (which paradoxically increases belief)
 */

import { calculateMoonPhase, checkMercuryRetrograde } from './astrology';

/**
 * Generate specific timing prediction for a card or reading
 * @param {Object} card - Card data with index, reversed, position
 * @param {Object} astroContext - Astrological context
 * @param {String} readingType - Type of reading
 * @param {Number} quantumSeed - For variation
 * @returns {String} - Specific timing statement
 */
export function generateTimingPrediction(card, astroContext, readingType, quantumSeed) {
  const cardNumber = getCardNumber(card.cardIndex);
  const timeframe = getTimeframe(cardNumber, readingType, quantumSeed);
  const anchor = getTemporalAnchor(astroContext, quantumSeed);

  return constructTimingStatement(timeframe, anchor, card, quantumSeed);
}

/**
 * Extract numerological value from card
 */
function getCardNumber(cardIndex) {
  // Major Arcana: 0-21
  if (cardIndex <= 21) return cardIndex;

  // Minor Arcana: extract number from position
  // Cards 22-73 (52 minor arcana cards)
  const minorIndex = cardIndex - 22;
  const cardValue = (minorIndex % 14);

  if (cardValue === 0) return 11; // Page
  if (cardValue === 1) return 12; // Knight
  if (cardValue === 2) return 13; // Queen
  if (cardValue === 3) return 14; // King
  return cardValue; // Ace=1, 2-10=face value
}

/**
 * Convert card number to timeframe
 */
function getTimeframe(cardNumber, readingType, quantumSeed) {
  const urgencyMultiplier = {
    'crisis': 0.5,
    'decision': 0.7,
    'general': 1.0,
    'spiritual': 1.5,
    'career': 1.2,
    'romance': 0.9,
    'wellness': 1.1
  }[readingType] || 1.0;

  const adjustedNumber = Math.round(cardNumber * urgencyMultiplier);

  // Decide on units based on card number
  if (adjustedNumber <= 3) {
    return { value: adjustedNumber, unit: 'days', plural: adjustedNumber !== 1 };
  } else if (adjustedNumber <= 7) {
    return { value: adjustedNumber, unit: 'weeks', plural: adjustedNumber !== 1 };
  } else if (adjustedNumber <= 12) {
    return { value: adjustedNumber, unit: 'months', plural: adjustedNumber !== 1 };
  } else {
    // Court cards = seasons
    const seasons = ['spring', 'summer', 'fall', 'winter'];
    const seasonIndex = Math.floor(quantumSeed * seasons.length) % seasons.length;
    return { value: null, unit: 'season', season: seasons[seasonIndex] || 'spring' };
  }
}

/**
 * Get temporal anchor (specific date reference)
 */
function getTemporalAnchor(astroContext, quantumSeed) {
  const today = new Date();
  const anchors = [];

  // Next moon phase shift
  const nextNewMoon = getNextMoonPhase('New Moon', today);
  const nextFullMoon = getNextMoonPhase('Full Moon', today);

  if (nextNewMoon) {
    anchors.push({
      type: 'moon',
      date: nextNewMoon,
      description: 'the next New Moon',
      energy: 'new beginnings'
    });
  }

  if (nextFullMoon) {
    anchors.push({
      type: 'moon',
      date: nextFullMoon,
      description: 'the next Full Moon',
      energy: 'culmination'
    });
  }

  // Mercury direct/retrograde
  const mercuryStatus = checkMercuryRetrograde(today);
  if (mercuryStatus.isRetrograde) {
    // When does it go direct?
    const directDate = getMercuryDirectDate(today);
    if (directDate) {
      anchors.push({
        type: 'mercury',
        date: directDate,
        description: 'when Mercury goes direct',
        energy: 'clarity returns'
      });
    }
  }

  // Next solstice/equinox
  const nextSolarShift = getNextSolarEvent(today);
  if (nextSolarShift) {
    anchors.push(nextSolarShift);
  }

  // Current moon phase as anchor
  if (astroContext?.moonPhase) {
    anchors.push({
      type: 'current_moon',
      description: `this ${astroContext.moonPhase.name} phase`,
      energy: astroContext.moonPhase.influence
    });
  }

  // Pick one based on quantum seed
  if (anchors.length === 0) return null;
  const index = Math.floor(quantumSeed * anchors.length);
  return anchors[index];
}

/**
 * Calculate next occurrence of moon phase
 */
function getNextMoonPhase(phaseName, fromDate) {
  // Moon cycle is ~29.53 days
  const synodicMonth = 29.53;
  const today = fromDate || new Date();

  // Simple approximation - in production would use precise ephemeris
  const currentPhase = calculateMoonPhase(today);

  let daysToPhase;
  if (phaseName === 'New Moon') {
    daysToPhase = currentPhase.lunarAge > 14.76 ? (29.53 - currentPhase.lunarAge) : (14.76 - currentPhase.lunarAge + 14.76);
  } else if (phaseName === 'Full Moon') {
    daysToPhase = currentPhase.lunarAge < 14.76 ? (14.76 - currentPhase.lunarAge) : (29.53 - currentPhase.lunarAge + 14.76);
  }

  const nextDate = new Date(today);
  nextDate.setDate(nextDate.getDate() + Math.round(daysToPhase));
  return nextDate;
}

/**
 * Get when Mercury goes direct
 */
function getMercuryDirectDate(today) {
  const retrogradePeriods = [
    { start: new Date(2025, 2, 15), end: new Date(2025, 3, 7) },
    { start: new Date(2025, 6, 18), end: new Date(2025, 7, 11) },
    { start: new Date(2025, 10, 9), end: new Date(2025, 10, 29) }
  ];

  for (const period of retrogradePeriods) {
    if (today >= period.start && today <= period.end) {
      return period.end;
    }
  }
  return null;
}

/**
 * Get next solstice or equinox
 */
function getNextSolarEvent(today) {
  const year = today.getFullYear();
  const events = [
    { date: new Date(year, 2, 20), name: 'Spring Equinox', energy: 'renewal' },
    { date: new Date(year, 5, 21), name: 'Summer Solstice', energy: 'peak light' },
    { date: new Date(year, 8, 22), name: 'Fall Equinox', energy: 'harvest' },
    { date: new Date(year, 11, 21), name: 'Winter Solstice', energy: 'deepest dark' }
  ];

  // Find next one
  for (const event of events) {
    if (event.date > today) {
      return {
        type: 'solar',
        date: event.date,
        description: `the ${event.name}`,
        energy: event.energy
      };
    }
  }

  // If we're past all of them, return next year's spring equinox
  return {
    type: 'solar',
    date: new Date(year + 1, 2, 20),
    description: 'the Spring Equinox',
    energy: 'renewal'
  };
}

/**
 * Construct natural-sounding timing statement
 * HARDENING #3: Add fuzzy logic to make predictions less falsifiable
 */
function constructTimingStatement(timeframe, anchor, card, quantumSeed) {
  const templates = [];

  // Fuzzy modifiers to add hedging
  const fuzzyModifiers = [
    '', // No modifier (direct)
    'around ',
    'approximately ',
    'roughly ',
    'give or take a few days, '
  ];
  const modifierIdx = Math.floor((quantumSeed * 0.123) * fuzzyModifiers.length);
  const fuzzy = fuzzyModifiers[modifierIdx];

  if (timeframe.unit === 'season' && timeframe.season) {
    const season = timeframe.season || 'spring'; // Fallback
    const seasonCapitalized = season.charAt(0).toUpperCase() + season.slice(1);
    templates.push(
      `Watch for movement by ${fuzzy}${season}, or when synchronicity signals you.`,
      `${seasonCapitalized} brings the shift this card promises—though it may begin before you notice.`,
      `Before ${fuzzy}${season} ends, you'll see this manifest, unless resistance delays it.`
    );
  } else if (anchor) {
    templates.push(
      `Expect this to crystallize within ${fuzzy}${timeframe.value || ''} ${timeframe.unit || 'soon'}—${fuzzy}${anchor.description}, or when the sign appears.`,
      `The timeline here is ${fuzzy}${timeframe.value || ''} ${timeframe.unit || 'soon'}, with ${anchor.description} marking a likely turning point.`,
      `By ${fuzzy}${anchor.description}, this will have likely shifted. We're talking ${timeframe.value || ''} ${timeframe.unit || 'soon'}, give or take.`,
      `${anchor.description} is your rough marker. Count ${fuzzy}${timeframe.value || ''} ${timeframe.unit || 'time'} from now, but trust your gut over the calendar.`
    );
  } else if (timeframe.value) {
    const valueStr = (timeframe.value || '').toString();
    const valueCapitalized = valueStr ? (valueStr.charAt(0).toUpperCase() + valueStr.slice(1)) : '';
    templates.push(
      `Give this ${fuzzy}${valueStr} ${timeframe.unit || 'time'}. Could be shorter if you're ready, longer if resistance shows up.`,
      `Within ${fuzzy}${valueStr} ${timeframe.unit || 'time'}, you'll likely know—unless you're avoiding the answer.`,
      `The energy shifts in ${fuzzy}${valueStr} ${timeframe.unit || 'time'}. Mark it, but don't be rigid about it.`,
      valueCapitalized ? `${valueCapitalized} ${timeframe.unit || 'time'}, ${fuzzy}. That's your window, but synchronicity may speed it up.` : `${fuzzy}That's your window, but synchronicity may speed it up.`
    );
  }

  const index = Math.floor(quantumSeed * templates.length) % Math.max(1, templates.length);
  return templates[index] || '';
}

/**
 * Generate timing for overall reading arc
 */
export function generateReadingTimeframe(cards, astroContext, readingType, quantumSeed) {
  // Get average timing from all cards
  const numbers = cards.map(c => getCardNumber(c.cardIndex));
  const avgNumber = numbers.reduce((a, b) => a + b, 0) / numbers.length;

  const timeframe = getTimeframe(Math.round(avgNumber), readingType, quantumSeed);
  const anchor = getTemporalAnchor(astroContext, quantumSeed * 0.777);

  const statements = [
    `This entire cycle completes in ${timeframe.value} ${timeframe.unit}.`,
    `From beginning to resolution: ${timeframe.value} ${timeframe.unit}.`,
    `The full story unfolds over ${timeframe.value} ${timeframe.unit}.`
  ];

  if (anchor) {
    statements.push(
      `By ${anchor.description}, the picture becomes clear.`,
      `${anchor.description} marks the climax of this chapter.`
    );
  }

  const index = Math.floor((quantumSeed * 0.333) * statements.length) % Math.max(1, statements.length);
  return statements[index] || statements[0];
}

/**
 * Generate "what to watch for" predictive statement
 */
export function generatePredictiveMarker(card, readingType, quantumSeed) {
  const markers = [
    'Pay attention to synchronicities around',
    'You\'ll receive a sign related to',
    'Watch for confirmation through',
    'The universe will validate this via',
    'Expect a clear message about',
    'You\'ll know this is activating when you see/hear',
    'The external confirmation comes through'
  ];

  const specifics = [
    'a conversation that feels fated',
    'something you see three times',
    'a dream or vision',
    'an unexpected message or email',
    'a chance encounter',
    'media (song, article, post) that speaks directly to this',
    'physical sensations or gut feelings',
    'animal symbolism or nature signs',
    'technology glitches or unusual "coincidences"',
    'something someone says without knowing your situation'
  ];

  const markerIdx = Math.floor(quantumSeed * markers.length) % markers.length;
  const specificIdx = Math.floor((quantumSeed * 0.888) * specifics.length) % specifics.length;

  return `${markers[markerIdx] || markers[0]} ${specifics[specificIdx] || specifics[0]}.`;
}
