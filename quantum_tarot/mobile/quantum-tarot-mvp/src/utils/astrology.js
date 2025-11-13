/**
 * ASTROLOGY ENGINE - Calculate astrological influences
 * Solunar phases, planetary positions, Mercury retrograde, houses
 */

/**
 * Calculate moon phase (solunar phase)
 * @param {Date} date - Date to calculate for
 * @returns {Object} - Moon phase data
 */
export function calculateMoonPhase(date) {
  // Algorithm: Calculate days since known new moon (Jan 6, 2000)
  const knownNewMoon = new Date(2000, 0, 6, 18, 14);
  const synodicMonth = 29.53058867; // days

  const millisecondsSinceNewMoon = date - knownNewMoon;
  const daysSinceNewMoon = millisecondsSinceNewMoon / (1000 * 60 * 60 * 24);
  const lunarAge = daysSinceNewMoon % synodicMonth;

  // Calculate phase percentage (0 = new moon, 0.5 = full moon)
  const phasePercent = lunarAge / synodicMonth;

  // Determine phase name
  let phaseName, phaseEmoji, influence;

  if (phasePercent < 0.0625 || phasePercent >= 0.9375) {
    phaseName = 'New Moon';
    phaseEmoji = '🌑';
    influence = 'New beginnings, intention setting, fresh starts';
  } else if (phasePercent < 0.1875) {
    phaseName = 'Waxing Crescent';
    phaseEmoji = '🌒';
    influence = 'Growth, momentum building, taking action';
  } else if (phasePercent < 0.3125) {
    phaseName = 'First Quarter';
    phaseEmoji = '🌓';
    influence = 'Decision points, overcoming obstacles, commitment';
  } else if (phasePercent < 0.4375) {
    phaseName = 'Waxing Gibbous';
    phaseEmoji = '🌔';
    influence = 'Refinement, adjustment, preparation';
  } else if (phasePercent < 0.5625) {
    phaseName = 'Full Moon';
    phaseEmoji = '🌕';
    influence = 'Culmination, revelation, release, clarity';
  } else if (phasePercent < 0.6875) {
    phaseName = 'Waning Gibbous';
    phaseEmoji = '🌖';
    influence = 'Gratitude, sharing wisdom, teaching';
  } else if (phasePercent < 0.8125) {
    phaseName = 'Last Quarter';
    phaseEmoji = '🌗';
    influence = 'Letting go, forgiveness, reassessment';
  } else {
    phaseName = 'Waning Crescent';
    phaseEmoji = '🌘';
    influence = 'Rest, reflection, surrender, closure';
  }

  return {
    name: phaseName,
    emoji: phaseEmoji,
    illumination: Math.abs(Math.cos(phasePercent * 2 * Math.PI)),
    influence,
    lunarAge: Math.round(lunarAge * 10) / 10
  };
}

/**
 * Check if Mercury is in retrograde
 * @param {Date} date - Date to check
 * @returns {Object} - Mercury retrograde status
 */
export function checkMercuryRetrograde(date) {
  // Mercury retrograde periods for 2024-2025
  // In production, this would fetch from API or calculate astronomically
  const retrogradePeriods = [
    // 2024
    { start: new Date(2024, 3, 1), end: new Date(2024, 3, 25) },
    { start: new Date(2024, 7, 5), end: new Date(2024, 7, 28) },
    { start: new Date(2024, 10, 25), end: new Date(2024, 11, 15) },

    // 2025
    { start: new Date(2025, 2, 15), end: new Date(2025, 3, 7) },
    { start: new Date(2025, 6, 18), end: new Date(2025, 7, 11) },
    { start: new Date(2025, 10, 9), end: new Date(2025, 10, 29) }
  ];

  const isRetrograde = retrogradePeriods.some(period =>
    date >= period.start && date <= period.end
  );

  return {
    isRetrograde,
    influence: isRetrograde
      ? 'Communication challenges, tech issues, review and revise energy. Not ideal for new contracts or major decisions.'
      : 'Clear communication, forward momentum, good time for new projects and agreements.',
    warning: isRetrograde
      ? 'Mercury Retrograde active - double check details'
      : null
  };
}

/**
 * Calculate astrological house based on time and zodiac sign
 * Simplified house system (Equal House)
 * @param {Object} birthdate - Birth date info
 * @param {Date} readingDate - Current reading date
 * @returns {Object} - House data
 */
export function calculateAstrologicalHouses(birthdate, readingDate) {
  // Simplified: Use zodiac sign to determine rising sign approximation
  // In full implementation, would need birth time + location for accurate house calculation

  const zodiacHouses = {
    'Aries': { house: 1, ruler: 'Mars', focus: 'Self, identity, new beginnings' },
    'Taurus': { house: 2, ruler: 'Venus', focus: 'Values, resources, material security' },
    'Gemini': { house: 3, ruler: 'Mercury', focus: 'Communication, learning, local environment' },
    'Cancer': { house: 4, ruler: 'Moon', focus: 'Home, family, emotional foundation' },
    'Leo': { house: 5, ruler: 'Sun', focus: 'Creativity, romance, self-expression' },
    'Virgo': { house: 6, ruler: 'Mercury', focus: 'Health, service, daily routines' },
    'Libra': { house: 7, ruler: 'Venus', focus: 'Partnerships, relationships, balance' },
    'Scorpio': { house: 8, ruler: 'Pluto', focus: 'Transformation, shared resources, depth' },
    'Sagittarius': { house: 9, ruler: 'Jupiter', focus: 'Philosophy, travel, higher learning' },
    'Capricorn': { house: 10, ruler: 'Saturn', focus: 'Career, public image, achievement' },
    'Aquarius': { house: 11, ruler: 'Uranus', focus: 'Community, innovation, ideals' },
    'Pisces': { house: 12, ruler: 'Neptune', focus: 'Spirituality, unconscious, transcendence' }
  };

  const zodiacSign = birthdate.zodiacSign || 'Aries';
  const houseData = zodiacHouses[zodiacSign] || zodiacHouses.Aries;

  return {
    ...houseData,
    note: 'Based on birth zodiac (simplified). Full houses require birth time + location.'
  };
}

/**
 * Get current planetary energies (simplified positions)
 * @param {Date} date - Date to calculate for
 * @returns {Object} - Planetary influences
 */
export function getPlanetaryInfluences(date) {
  const month = date.getMonth();
  const day = date.getDate();

  // Simplified: Assign general planetary influence based on season
  // In production, would calculate actual planetary positions

  let dominantPlanet, energy;

  // Spring (Mar-May)
  if (month >= 2 && month <= 4) {
    dominantPlanet = 'Mars';
    energy = 'Action, initiation, courage, forward movement';
  }
  // Summer (Jun-Aug)
  else if (month >= 5 && month <= 7) {
    dominantPlanet = 'Sun';
    energy = 'Vitality, confidence, self-expression, leadership';
  }
  // Fall (Sep-Nov)
  else if (month >= 8 && month <= 10) {
    dominantPlanet = 'Venus';
    energy = 'Harmony, relationships, beauty, values';
  }
  // Winter (Dec-Feb)
  else {
    dominantPlanet = 'Saturn';
    energy = 'Structure, discipline, wisdom, patience';
  }

  return {
    dominantPlanet,
    energy,
    note: 'Seasonal dominant energy (simplified calculation)'
  };
}

/**
 * Check if we're in the Age of Aquarius
 * @returns {Object} - Astrological age info
 */
export function getAstrologicalAge() {
  // We are transitioning into the Age of Aquarius (2000-2150 approximate)
  const currentYear = new Date().getFullYear();

  return {
    age: 'Age of Aquarius (Transition)',
    startYear: 2000,
    influence: 'Humanitarian focus, technology, innovation, collective consciousness, breaking old structures',
    note: `Year ${currentYear} - In transition period from Pisces to Aquarius`,
    previousAge: 'Age of Pisces (faith, spirituality, sacrifice)'
  };
}

/**
 * Get comprehensive astrological context for a reading
 * @param {Object} params - { birthdate, zodiacSign, readingDate }
 * @returns {Object} - Complete astrological context
 */
export function getAstrologicalContext(params) {
  const { birthdate, zodiacSign } = params;
  const readingDate = new Date();

  const moonPhase = calculateMoonPhase(readingDate);
  const mercuryStatus = checkMercuryRetrograde(readingDate);
  const houses = calculateAstrologicalHouses({ zodiacSign }, readingDate);
  const planetaryInfluences = getPlanetaryInfluences(readingDate);
  const astrologicalAge = getAstrologicalAge();

  return {
    readingDate: readingDate.toISOString(),
    moonPhase,
    mercuryRetrograde: mercuryStatus,
    natalSign: zodiacSign,
    houses,
    planetaryInfluences,
    astrologicalAge,
    summary: generateAstroSummary(moonPhase, mercuryStatus, planetaryInfluences)
  };
}

/**
 * Generate a text summary of current astrological influences
 */
function generateAstroSummary(moonPhase, mercury, planetary) {
  let summary = `${moonPhase.name} energy (${moonPhase.influence}). `;

  if (mercury.isRetrograde) {
    summary += `Mercury Retrograde: ${mercury.influence} `;
  }

  summary += `Dominant planetary energy: ${planetary.dominantPlanet} - ${planetary.energy}.`;

  return summary;
}
