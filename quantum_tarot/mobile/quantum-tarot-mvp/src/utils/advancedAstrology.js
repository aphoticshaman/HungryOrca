/**
 * ADVANCED ASTROLOGICAL ANALYSIS
 * Beyond sun sign - includes Black Moon Lilith, Chiron, Nodes, asteroids, transits
 *
 * Used for hyper-personalized tarot synthesis that addresses:
 * - Shadow work (Lilith)
 * - Wounds and healing (Chiron)
 * - Life purpose and karma (North/South Node)
 * - Current cosmic weather (transits)
 * - Feminine power and relationships (Juno, Vesta, Ceres, Pallas)
 */

/**
 * Calculate Black Moon Lilith position by birth date
 * Lilith = repressed feminine rage, sexuality, what society told you to hide
 * 9-year cycle through zodiac
 */
export function calculateLilithSign(birthdate) {
  const birth = new Date(birthdate);
  const lilithEpoch = new Date('1900-01-01'); // Reference point
  const daysSinceEpoch = (birth - lilithEpoch) / (1000 * 60 * 60 * 24);

  // Lilith moves ~40 degrees per year, ~3.3 degrees per month
  const lilithDegrees = (daysSinceEpoch * 0.1107) % 360; // Approximate
  const signIndex = Math.floor(lilithDegrees / 30);

  const signs = [
    'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
  ];

  return {
    sign: signs[signIndex],
    meaning: getLilithMeaning(signs[signIndex])
  };
}

/**
 * Calculate Chiron position by birth date
 * Chiron = the wounded healer, where we're broken and how we heal others
 * ~50 year orbit
 */
export function calculateChironSign(birthdate) {
  const birth = new Date(birthdate);
  const chironEpoch = new Date('1977-01-01'); // Chiron in Taurus epoch
  const daysSinceEpoch = (birth - chironEpoch) / (1000 * 60 * 60 * 24);

  // Chiron moves ~7 degrees per year average (irregular orbit)
  const chironDegrees = (daysSinceEpoch * 0.0192) % 360;
  const signIndex = Math.floor(chironDegrees / 30);

  const signs = [
    'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra',
    'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces', 'Aries'
  ];

  return {
    sign: signs[signIndex],
    meaning: getChironMeaning(signs[signIndex])
  };
}

/**
 * Calculate North Node position (karmic destiny)
 * South Node is opposite (past life patterns to release)
 * ~18.6 year cycle (RETROGRADE motion)
 */
export function calculateNodalAxis(birthdate) {
  const birth = new Date(birthdate);
  const nodeEpoch = new Date('2000-01-01'); // Reference
  const daysSinceEpoch = (birth - nodeEpoch) / (1000 * 60 * 60 * 24);

  // Nodes move BACKWARD ~19 degrees per year
  const nodeDegrees = (360 - (daysSinceEpoch * 0.0521)) % 360;
  const northNodeIndex = Math.floor(nodeDegrees / 30);
  const southNodeIndex = (northNodeIndex + 6) % 12; // Opposite sign

  const signs = [
    'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
  ];

  return {
    northNode: {
      sign: signs[northNodeIndex],
      meaning: getNodeMeaning(signs[northNodeIndex], 'north')
    },
    southNode: {
      sign: signs[southNodeIndex],
      meaning: getNodeMeaning(signs[southNodeIndex], 'south')
    }
  };
}

/**
 * Get current moon phase
 * New = new beginnings, Full = culmination/release, Waxing = building, Waning = releasing
 */
export function getCurrentMoonPhase(currentDate = new Date()) {
  // Lunar cycle is ~29.53 days
  const knownNewMoon = new Date('2000-01-06'); // Reference new moon
  const daysSinceNew = (currentDate - knownNewMoon) / (1000 * 60 * 60 * 24);
  const lunarCycle = 29.53;
  const phase = (daysSinceNew % lunarCycle) / lunarCycle;

  let phaseName, phaseEnergy, phaseAdvice;

  if (phase < 0.03 || phase > 0.97) {
    phaseName = 'New Moon';
    phaseEnergy = 'New beginnings, setting intentions, planting seeds';
    phaseAdvice = 'What new chapter are you starting? Set intentions aligned with your cards.';
  } else if (phase < 0.22) {
    phaseName = 'Waxing Crescent';
    phaseEnergy = 'Building momentum, taking first steps, faith required';
    phaseAdvice = 'Take action on your intentions. Small steps build momentum.';
  } else if (phase < 0.28) {
    phaseName = 'First Quarter';
    phaseEnergy = 'Obstacles emerge, decisions needed, commitment tested';
    phaseAdvice = 'Face challenges head-on. Adjust your approach if needed.';
  } else if (phase < 0.47) {
    phaseName = 'Waxing Gibbous';
    phaseEnergy = 'Refinement, preparation, anticipation of culmination';
    phaseAdvice = 'Fine-tune your plans. The fruits of your labor are near.';
  } else if (phase < 0.53) {
    phaseName = 'Full Moon';
    phaseEnergy = 'Culmination, release, illumination, emotional peak';
    phaseAdvice = 'What needs to be released or celebrated? Full moons reveal what was hidden.';
  } else if (phase < 0.72) {
    phaseName = 'Waning Gibbous';
    phaseEnergy = 'Gratitude, sharing wisdom, processing lessons';
    phaseAdvice = 'Reflect on what you\'ve learned. Share your gifts with others.';
  } else if (phase < 0.78) {
    phaseName = 'Last Quarter';
    phaseEnergy = 'Letting go, forgiveness, breaking old patterns';
    phaseAdvice = 'Release what no longer serves you. Forgive yourself and others.';
  } else {
    phaseName = 'Waning Crescent';
    phaseEnergy = 'Rest, surrender, trust, spiritual renewal';
    phaseAdvice = 'Rest and recharge. Trust the process. A new cycle begins soon.';
  }

  return {
    phaseName,
    phaseEnergy,
    phaseAdvice,
    phasePercentage: Math.round(phase * 100)
  };
}

/**
 * Get current transits affecting user
 * What planets are doing NOW and how they aspect natal chart
 */
export function getCurrentTransits(birthdate, currentDate = new Date()) {
  const transits = {
    saturnReturn: isSaturnReturn(birthdate, currentDate),
    uranusOpposition: isUranusOpposition(birthdate, currentDate),
    jupiterTransit: getJupiterTransit(currentDate),
    marsRetrograde: isMarsRetrograde(currentDate),
    venusRetrograde: isVenusRetrograde(currentDate),
    mercuryRetrograde: isMercuryRetrograde(currentDate)
  };

  return transits;
}

/**
 * Saturn Return check (ages 27-30, 56-59, 84-87)
 * Major life restructuring, adulting hard
 */
function isSaturnReturn(birthdate, currentDate) {
  const age = (currentDate - new Date(birthdate)) / (1000 * 60 * 60 * 24 * 365.25);
  const inReturn = (age >= 27 && age <= 30) || (age >= 56 && age <= 59) || (age >= 84 && age <= 87);

  return {
    active: inReturn,
    age: Math.floor(age),
    meaning: inReturn ? 'MAJOR life restructuring. Saturn demands you face consequences, take responsibility, build real foundations. This is cosmic adulting - no shortcuts allowed. What legacy are you building?' : null
  };
}

/**
 * Uranus Opposition (age 38-42)
 * Midlife awakening, rebellion, divorce/crisis energy
 */
function isUranusOpposition(birthdate, currentDate) {
  const age = (currentDate - new Date(birthdate)) / (1000 * 60 * 60 * 24 * 365.25);
  const inOpposition = age >= 38 && age <= 42;

  return {
    active: inOpposition,
    age: Math.floor(age),
    meaning: inOpposition ? 'Midlife awakening - Uranus shakes up stagnation. Question everything. Rebel against what no longer fits. Divorce, career pivot, identity crisis energy. This is your chance to reclaim authenticity before it\'s too late.' : null
  };
}

/**
 * Jupiter transit (current sign)
 * Expansion, luck, growth opportunities - changes yearly
 */
function getJupiterTransit(currentDate) {
  const jupiterEpoch = new Date('2023-05-16'); // Jupiter in Taurus
  const daysSince = (currentDate - jupiterEpoch) / (1000 * 60 * 60 * 24);
  const jupiterDegrees = (daysSince * 0.0833) % 360; // ~12 year cycle
  const signIndex = Math.floor(jupiterDegrees / 30);

  const signs = [
    'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra',
    'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces', 'Aries'
  ];

  return {
    sign: signs[signIndex],
    meaning: `Jupiter in ${signs[signIndex]} expands ${getJupiterSignMeaning(signs[signIndex])}. Where can you grow? Luck favors bold action here.`
  };
}

/**
 * Mercury Retrograde check
 * Happens ~3 times/year for ~3 weeks
 * Communication chaos, tech fails, revisit/review/revise
 */
function isMercuryRetrograde(currentDate) {
  // 2025 Mercury retrogrades (approximate):
  // March 15 - April 7, July 18 - August 11, November 9 - November 29
  const year = currentDate.getFullYear();
  const month = currentDate.getMonth(); // 0-11
  const day = currentDate.getDate();

  let retrograde = false;

  if (year === 2025) {
    retrograde =
      (month === 2 && day >= 15) || (month === 3 && day <= 7) ||
      (month === 6 && day >= 18) || (month === 7 && day <= 11) ||
      (month === 10 && day >= 9) || (month === 10 && day <= 29);
  }
  // Add more years as needed

  return {
    active: retrograde,
    meaning: retrograde ? 'Mercury retrograde: Communication breakdowns, tech fails, travel delays. NOT the time for new contracts or big launches. Instead: REVIEW, REVISE, RECONNECT with old contacts. Re-do what needs fixing.' : null
  };
}

/**
 * Mars Retrograde check
 * Happens every ~2 years for ~2.5 months
 * Frustration, delayed action, internal anger work
 */
function isMarsRetrograde(currentDate) {
  // Mars retrogrades roughly: Dec 2024 - Feb 2025, etc.
  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();

  const retrograde = (year === 2024 && month >= 11) || (year === 2025 && month <= 1);

  return {
    active: retrograde,
    meaning: retrograde ? 'Mars retrograde: Forward action feels blocked. Anger turns inward or explodes sideways. Time for strategic planning, NOT forcing progress. Address passive aggression. What old battles need resolving?' : null
  };
}

/**
 * Venus Retrograde check
 * Happens every ~18 months for ~40 days
 * Relationship reevaluation, values shift, ex return energy
 */
function isVenusRetrograde(currentDate) {
  // Venus retrogrades roughly every 18 months
  // Next: March-April 2025
  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();
  const day = currentDate.getDate();

  const retrograde = (year === 2025 && month === 2 && day >= 1) || (year === 2025 && month === 3 && day <= 12);

  return {
    active: retrograde,
    meaning: retrograde ? 'Venus retrograde: Relationship review time. Exes resurface (not always to get back together - often for closure). Reevaluate your values, worth, and what you truly desire in love. NOT ideal for weddings or new romances - instead, heal old wounds.' : null
  };
}

// Meanings functions

function getLilithMeaning(sign) {
  const meanings = {
    'Aries': 'Repressed rage and assertion. You were told to be nice, not angry. Society punished your anger. Reclaim your righteous fury, your right to take up space, your warrior spirit.',
    'Taurus': 'Repressed sensuality and material desires. Shamed for wanting pleasure, comfort, beauty, or money. Told you\'re greedy or vain. Reclaim embodied pleasure and unapologetic comfort.',
    'Gemini': 'Repressed voice and curiosity. Told you talk too much, ask wrong questions, or are too scattered. Silenced for speaking truth. Reclaim your intellect and right to question everything.',
    'Cancer': 'Repressed needs and vulnerability. Told to stop being needy, sensitive, or emotional. Shamed for wanting care. Reclaim your emotional depth and need for nurturing.',
    'Leo': 'Repressed confidence and radiance. Told not to be attention-seeking, show-offy, or narcissistic. Dimmed your light for others. Reclaim your right to shine and be celebrated.',
    'Virgo': 'Repressed perfectionism turned to shame. Criticized for never being good enough. Internalized harsh judgment. Reclaim discernment without self-hatred.',
    'Libra': 'Repressed desires in favor of keeping peace. People-pleased your authentic wants away. Abandoned yourself for harmony. Reclaim your power to say no and create healthy conflict.',
    'Scorpio': 'Repressed sexuality, power, and intensity. Shamed for being "too much," too sexual, too dark. Told to lighten up. Reclaim your depth, magic, and transformative power.',
    'Sagittarius': 'Repressed freedom and wildness. Told to settle down, be practical, stop being reckless. Caged your spirit. Reclaim your expansive, untamed, adventurous nature.',
    'Capricorn': 'Repressed ambition and authority. Told not to be cold, ruthless, or power-hungry (especially if female). Shamed for wanting success. Reclaim your right to achieve and lead.',
    'Aquarius': 'Repressed weirdness and rebellion. Told to fit in, be normal, stop being so different. Punished for uniqueness. Reclaim your freak flag and revolutionary spirit.',
    'Pisces': 'Repressed intuition, mysticism, and sensitivity. Told you\'re too sensitive, delusional, or naive. Gaslit about your psychic gifts. Reclaim your magic and spiritual knowing.'
  };
  return meanings[sign] || 'Shadow feminine energy to reclaim.';
}

function getChironMeaning(sign) {
  const meanings = {
    'Aries': 'Wounded in: assertion, identity, right to exist. You struggle to stand up for yourself or claim space. You heal others by: teaching them courage and self-advocacy.',
    'Taurus': 'Wounded in: self-worth, security, body image. You struggle to feel "enough" or secure. You heal others by: teaching inherent value beyond achievement.',
    'Gemini': 'Wounded in: communication, being heard, intelligence. You struggle to feel understood or smart enough. You heal others by: helping them find their voice.',
    'Cancer': 'Wounded in: belonging, family, emotional safety. You struggle to feel at home anywhere. You heal others by: creating safe emotional spaces.',
    'Leo': 'Wounded in: self-expression, confidence, being seen. You struggle to take up space or feel special. You heal others by: celebrating their uniqueness.',
    'Virgo': 'Wounded in: perfectionism, usefulness, health. You struggle with never being good enough. You heal others by: teaching radical self-acceptance.',
    'Libra': 'Wounded in: relationships, fairness, beauty. You struggle with codependency or unfair treatment. You heal others by: modeling healthy boundaries.',
    'Scorpio': 'Wounded in: trust, intimacy, transformation. You struggle to let people in or feel safe in vulnerability. You heal others by: guiding them through darkness.',
    'Sagittarius': 'Wounded in: meaning, faith, freedom. You struggle with nihilism or feeling trapped. You heal others by: showing expansive possibility.',
    'Capricorn': 'Wounded in: authority, achievement, father. You struggle with imposter syndrome or lack of recognition. You heal others by: mentoring them to success.',
    'Aquarius': 'Wounded in: belonging to groups, authenticity, vision. You struggle to fit in anywhere. You heal others by: normalizing weirdness and creating inclusive spaces.',
    'Pisces': 'Wounded in: boundaries, victimhood, escapism. You struggle with martyrdom or addiction. You heal others by: teaching compassion WITH healthy limits.'
  };
  return meanings[sign] || 'Your core wound and healing gift.';
}

function getNodeMeaning(sign, node) {
  if (node === 'north') {
    const meanings = {
      'Aries': 'Destiny: Develop courage, independence, decisive action. STOP people-pleasing and over-compromising (Libra south node). Your growth edge is self-reliance.',
      'Taurus': 'Destiny: Build security, enjoy simple pleasures, slow down. STOP crisis addiction and emotional drama (Scorpio south node). Your growth edge is stability.',
      'Gemini': 'Destiny: Communicate, learn constantly, stay curious. STOP preaching or thinking you know it all (Sagittarius south node). Your growth edge is humility.',
      'Cancer': 'Destiny: Create home, feel feelings, nurture self/others. STOP workaholism and emotional unavailability (Capricorn south node). Your growth edge is softness.',
      'Leo': 'Destiny: Shine, create, lead from heart. STOP hiding in groups or detaching emotionally (Aquarius south node). Your growth edge is vulnerable self-expression.',
      'Virgo': 'Destiny: Serve with skill, perfect your craft, get grounded. STOP victim mentality and escapism (Pisces south node). Your growth edge is practical service.',
      'Libra': 'Destiny: Partner authentically, create beauty, find balance. STOP self-centeredness and going it alone (Aries south node). Your growth edge is collaboration.',
      'Scorpio': 'Destiny: Transform, go deep, merge courageously. STOP surface materialism and comfort-seeking (Taurus south node). Your growth edge is intensity.',
      'Sagittarius': 'Destiny: Seek truth, expand horizons, teach wisdom. STOP gossip and mental clutter (Gemini south node). Your growth edge is meaning-making.',
      'Capricorn': 'Destiny: Achieve mastery, lead with integrity, build legacy. STOP emotional dependency and clinging to past (Cancer south node). Your growth edge is authority.',
      'Aquarius': 'Destiny: Innovate, liberate, find your people. STOP ego attachment and need for validation (Leo south node). Your growth edge is detached service.',
      'Pisces': 'Destiny: Surrender to flow, create art, develop faith. STOP perfectionism and harsh self-judgment (Virgo south node). Your growth edge is trust.'
    };
    return meanings[sign] || 'Your soul\'s evolutionary direction.';
  } else {
    return `Past life mastery and comfort zone. You default to ${sign} patterns - release them to grow.`;
  }
}

function getJupiterSignMeaning(sign) {
  const meanings = {
    'Aries': 'bold action, entrepreneurship, leadership, physical courage',
    'Taurus': 'finances, sensual pleasure, material security, nature connection',
    'Gemini': 'communication, learning, networking, curiosity',
    'Cancer': 'home life, family bonds, emotional security, caregiving',
    'Leo': 'creativity, romance, self-expression, joy',
    'Virgo': 'health, daily routines, skill development, service',
    'Libra': 'partnerships, aesthetic beauty, justice, diplomacy',
    'Scorpio': 'transformation, depth psychology, sexuality, shared resources',
    'Sagittarius': 'travel, philosophy, higher education, adventure',
    'Capricorn': 'career advancement, public recognition, long-term goals',
    'Aquarius': 'innovation, community building, humanitarian work, technology',
    'Pisces': 'spirituality, artistic creation, compassion, mysticism'
  };
  return meanings[sign] || 'growth and opportunity';
}

/**
 * FULL ASTROLOGICAL CONTEXT for user
 * Combines everything into synthesis-ready package
 */
export function getFullAstrologicalContext(birthdate, zodiacSign, currentDate = new Date()) {
  const lilith = calculateLilithSign(birthdate);
  const chiron = calculateChironSign(birthdate);
  const nodes = calculateNodalAxis(birthdate);
  const transits = getCurrentTransits(birthdate, currentDate);
  const moonPhase = getCurrentMoonPhase(currentDate);

  return {
    sunSign: zodiacSign,
    lilith,
    chiron,
    northNode: nodes.northNode,
    southNode: nodes.southNode,
    moonPhase,
    currentTransits: transits,
    synthesisPrompts: {
      lilithPrompt: `Your Black Moon Lilith in ${lilith.sign} reveals: ${lilith.meaning}`,
      chironPrompt: `Your Chiron in ${chiron.sign} shows: ${chiron.meaning}`,
      nodalPrompt: `Your North Node in ${nodes.northNode.sign} is calling you to: ${nodes.northNode.meaning}`,
      moonPhasePrompt: `The ${moonPhase.phaseName} asks: ${moonPhase.phaseAdvice}`,
      transitPrompts: Object.entries(transits)
        .filter(([_, transit]) => transit.active || transit.sign)
        .map(([name, transit]) => transit.meaning || `${name}: ${transit.sign} - ${transit.meaning}`)
        .filter(Boolean)
    }
  };
}

/**
 * Get time of day energy
 * Morning/afternoon/evening/night have different vibes
 */
export function getTimeOfDayEnergy(currentDate = new Date()) {
  const hour = currentDate.getHours();

  if (hour >= 5 && hour < 12) {
    return {
      period: 'Morning',
      energy: 'Fresh starts, clarity, new possibilities emerging',
      advice: 'What seeds are you planting today? Morning energy supports new beginnings and clear vision.'
    };
  } else if (hour >= 12 && hour < 17) {
    return {
      period: 'Afternoon',
      energy: 'Peak productivity, action-taking, manifestation in motion',
      advice: 'How can you channel this active energy? Afternoon is for doing, building, making things happen.'
    };
  } else if (hour >= 17 && hour < 21) {
    return {
      period: 'Evening',
      energy: 'Winding down, reflection, integration of the day',
      advice: 'What did today teach you? Evening invites gentle reflection and processing.'
    };
  } else {
    return {
      period: 'Night',
      energy: 'Deep wisdom, shadow work, subconscious surfacing, dreamtime approaching',
      advice: 'What is your shadow revealing? Night is for facing what you avoid in daylight, for deep truth-telling.'
    };
  }
}
