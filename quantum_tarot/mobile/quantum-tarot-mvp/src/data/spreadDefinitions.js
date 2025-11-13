/**
 * Comprehensive Spread Definitions
 * Defines all tarot spreads with spatial coordinates, meanings, and UI metadata
 */

export const SPREAD_CATEGORIES = {
  LINEAR: 'linear',
  SPATIAL: 'spatial',
  DECISION_TREE: 'decision_tree',
  MATRIX: 'matrix'
};

export const NAVIGATION_MODES = {
  HORIZONTAL_SWIPE: 'horizontal_swipe',
  VERTICAL_SCROLL: 'vertical_scroll',
  PAN_2D: '2d_pan',
  PAGINATED: 'paginated',
  BRANCHING: 'branching'
};

/**
 * All spread definitions
 * Coordinates are normalized 0-1 for responsive positioning
 */
export const SPREAD_DEFINITIONS = {

  // ═══════════════════════════════════════════════════════════════
  // SINGLE CARD
  // ═══════════════════════════════════════════════════════════════
  single_card: {
    name: 'Single Card',
    description: 'Quick guidance or daily focus',
    cardCount: 1,
    category: SPREAD_CATEGORIES.LINEAR,
    navigationMode: NAVIGATION_MODES.VERTICAL_SCROLL,
    positions: [
      {
        index: 0,
        name: 'Focus',
        meaning: 'What you need to know right now',
        coordinates: { x: 0.5, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: []
      }
    ],
    viewModes: ['detail']
  },

  // ═══════════════════════════════════════════════════════════════
  // THREE-CARD: PAST-PRESENT-FUTURE
  // ═══════════════════════════════════════════════════════════════
  three_card: {
    name: 'Past-Present-Future',
    description: 'Classic timeline reading',
    cardCount: 3,
    category: SPREAD_CATEGORIES.LINEAR,
    navigationMode: NAVIGATION_MODES.HORIZONTAL_SWIPE,
    positions: [
      {
        index: 0,
        name: 'Past',
        meaning: 'What has led to this moment',
        coordinates: { x: 0.2, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1],
        colorAccent: '#9F7AEA' // Purple (fading)
      },
      {
        index: 1,
        name: 'Present',
        meaning: 'Current situation and energies',
        coordinates: { x: 0.5, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0, 2],
        colorAccent: '#F687B3' // Pink (active)
      },
      {
        index: 2,
        name: 'Future',
        meaning: 'Where things are heading',
        coordinates: { x: 0.8, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1],
        colorAccent: '#68D391' // Green (emerging)
      }
    ],
    viewModes: ['swipe', 'detail']
  },

  // ═══════════════════════════════════════════════════════════════
  // THREE-CARD: GOAL PROGRESS
  // ═══════════════════════════════════════════════════════════════
  goal_progress: {
    name: 'Goal Progress',
    description: 'Track your journey toward a goal',
    cardCount: 3,
    category: SPREAD_CATEGORIES.LINEAR,
    navigationMode: NAVIGATION_MODES.VERTICAL_SCROLL,
    positions: [
      {
        index: 0,
        name: 'Starting Point',
        meaning: 'Where you began, your foundation',
        coordinates: { x: 0.5, y: 0.8, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1],
        colorAccent: '#9F7AEA',
        progress: 0
      },
      {
        index: 1,
        name: 'Current Progress',
        meaning: 'Where you are now, your momentum',
        coordinates: { x: 0.5, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0, 2],
        colorAccent: '#F687B3',
        progress: 50
      },
      {
        index: 2,
        name: 'Goal Manifestation',
        meaning: 'Likely outcome, what manifests',
        coordinates: { x: 0.5, y: 0.2, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1],
        colorAccent: '#FFD700',
        progress: 100
      }
    ],
    viewModes: ['vertical', 'detail']
  },

  // ═══════════════════════════════════════════════════════════════
  // THREE-CARD: DECISION ANALYSIS
  // ═══════════════════════════════════════════════════════════════
  decision_analysis: {
    name: 'Decision Analysis',
    description: 'Explore two paths and their outcomes',
    cardCount: 3,
    category: SPREAD_CATEGORIES.DECISION_TREE,
    navigationMode: NAVIGATION_MODES.BRANCHING,
    positions: [
      {
        index: 0,
        name: 'The Crux',
        meaning: 'The core decision point, what matters most',
        coordinates: { x: 0.5, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1, 2],
        colorAccent: '#F687B3'
      },
      {
        index: 1,
        name: 'Option A',
        meaning: 'Outcome if you choose Path A',
        coordinates: { x: 0.3, y: 0.2, rotation: -15 },
        zIndex: 1,
        relatedPositions: [0],
        colorAccent: '#68D391',
        branch: 'A'
      },
      {
        index: 2,
        name: 'Option B',
        meaning: 'Outcome if you choose Path B',
        coordinates: { x: 0.7, y: 0.2, rotation: 15 },
        zIndex: 1,
        relatedPositions: [0],
        colorAccent: '#63B3ED',
        branch: 'B'
      }
    ],
    viewModes: ['branching', 'detail']
  },

  // ═══════════════════════════════════════════════════════════════
  // THREE-CARD: DAILY CHECK-IN
  // ═══════════════════════════════════════════════════════════════
  daily_checkin: {
    name: 'Daily Check-In',
    description: 'Morning guidance and reflection',
    cardCount: 3,
    category: SPREAD_CATEGORIES.LINEAR,
    navigationMode: NAVIGATION_MODES.HORIZONTAL_SWIPE,
    positions: [
      {
        index: 0,
        name: 'Focus',
        meaning: 'What deserves your attention today',
        coordinates: { x: 0.2, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1, 2],
        colorAccent: '#68D391' // Green
      },
      {
        index: 1,
        name: 'Avoid',
        meaning: 'What to be cautious of or release',
        coordinates: { x: 0.5, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0, 2],
        colorAccent: '#FC8181' // Red
      },
      {
        index: 2,
        name: 'Gift',
        meaning: 'Hidden opportunity or blessing available',
        coordinates: { x: 0.8, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0, 1],
        colorAccent: '#FFD700' // Gold
      }
    ],
    viewModes: ['swipe', 'detail']
  },

  // ═══════════════════════════════════════════════════════════════
  // THREE-CARD: CLAIRVOYANT PREDICTIVE
  // ═══════════════════════════════════════════════════════════════
  clairvoyant_predictive: {
    name: 'Clairvoyant Forecast',
    description: 'If I do X, what happens?',
    cardCount: 3,
    category: SPREAD_CATEGORIES.LINEAR,
    navigationMode: NAVIGATION_MODES.VERTICAL_SCROLL,
    positions: [
      {
        index: 0,
        name: 'Now / Choice',
        meaning: 'The energy around your intended action',
        coordinates: { x: 0.5, y: 0.2, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1],
        colorAccent: '#F687B3',
        timeframe: 'Present'
      },
      {
        index: 1,
        name: 'Near Future',
        meaning: 'Immediate ripples and reactions (days/weeks)',
        coordinates: { x: 0.5, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0, 2],
        colorAccent: '#63B3ED',
        timeframe: '2-4 weeks'
      },
      {
        index: 2,
        name: 'Far Outcome',
        meaning: 'Where this path ultimately leads (months)',
        coordinates: { x: 0.5, y: 0.8, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1],
        colorAccent: '#9F7AEA',
        timeframe: '3-6 months'
      }
    ],
    viewModes: ['vertical', 'detail'],
    warningMessage: 'Remember: you always have free will to change course'
  },

  // ═══════════════════════════════════════════════════════════════
  // SIX-CARD: RELATIONSHIP SPREAD
  // ═══════════════════════════════════════════════════════════════
  relationship: {
    name: 'Relationship',
    description: 'Deep dive into interpersonal dynamics',
    cardCount: 6,
    category: SPREAD_CATEGORIES.SPATIAL,
    navigationMode: NAVIGATION_MODES.PAGINATED,
    positions: [
      {
        index: 0,
        name: 'You',
        meaning: 'Your energy, what you bring',
        coordinates: { x: 0.3, y: 0.2, rotation: 0 },
        zIndex: 1,
        relatedPositions: [1, 2],
        colorAccent: '#F687B3',
        section: 'individuals'
      },
      {
        index: 1,
        name: 'Them',
        meaning: 'Their energy, what they bring',
        coordinates: { x: 0.7, y: 0.2, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0, 2],
        colorAccent: '#63B3ED',
        section: 'individuals'
      },
      {
        index: 2,
        name: 'Connection',
        meaning: 'The bond between you, relationship essence',
        coordinates: { x: 0.5, y: 0.4, rotation: 0 },
        zIndex: 2,
        relatedPositions: [0, 1, 3, 4, 5],
        colorAccent: '#FFD700',
        section: 'bond'
      },
      {
        index: 3,
        name: 'Challenge',
        meaning: "What you're working through together",
        coordinates: { x: 0.3, y: 0.6, rotation: 0 },
        zIndex: 1,
        relatedPositions: [2, 4],
        colorAccent: '#FC8181',
        section: 'guidance'
      },
      {
        index: 4,
        name: 'Advice',
        meaning: 'How to navigate this connection',
        coordinates: { x: 0.7, y: 0.6, rotation: 0 },
        zIndex: 1,
        relatedPositions: [2, 3],
        colorAccent: '#68D391',
        section: 'guidance'
      },
      {
        index: 5,
        name: 'Outcome',
        meaning: 'Where the relationship is heading',
        coordinates: { x: 0.5, y: 0.8, rotation: 0 },
        zIndex: 1,
        relatedPositions: [2],
        colorAccent: '#E2E8F0',
        section: 'outcome'
      }
    ],
    sections: [
      {
        name: 'The Individuals',
        cardIndices: [0, 1],
        description: 'Each person's energy and contribution'
      },
      {
        name: 'The Bond',
        cardIndices: [2],
        description: 'The relationship itself'
      },
      {
        name: 'Navigation',
        cardIndices: [3, 4],
        description: 'Challenges and how to move through them'
      },
      {
        name: 'Future',
        cardIndices: [5],
        description: 'Where things are going'
      }
    ],
    viewModes: ['overview', 'sectional', 'detail']
  },

  // ═══════════════════════════════════════════════════════════════
  // TEN-CARD: CELTIC CROSS (THE BIG ONE)
  // ═══════════════════════════════════════════════════════════════
  celtic_cross: {
    name: 'Celtic Cross',
    description: 'Comprehensive exploration of a situation',
    cardCount: 10,
    category: SPREAD_CATEGORIES.SPATIAL,
    navigationMode: NAVIGATION_MODES.PAN_2D,
    positions: [
      {
        index: 0,
        name: 'Present Situation',
        meaning: 'The heart of the matter, current state',
        coordinates: { x: 0.4, y: 0.5, rotation: 0 },
        zIndex: 2,
        relatedPositions: [1, 2, 3, 4, 5],
        colorAccent: '#F687B3',
        section: 'cross'
      },
      {
        index: 1,
        name: 'Challenge / Crossing',
        meaning: 'What opposes or complicates (may help or hinder)',
        coordinates: { x: 0.4, y: 0.5, rotation: 90 },
        zIndex: 3,
        relatedPositions: [0],
        colorAccent: '#FC8181',
        section: 'cross'
      },
      {
        index: 2,
        name: 'Foundation',
        meaning: 'Root cause, subconscious influence, distant past',
        coordinates: { x: 0.4, y: 0.7, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0],
        colorAccent: '#A0AEC0',
        section: 'cross'
      },
      {
        index: 3,
        name: 'Recent Past',
        meaning: 'Events leading here, what's leaving',
        coordinates: { x: 0.2, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0],
        colorAccent: '#9F7AEA',
        section: 'cross'
      },
      {
        index: 4,
        name: 'Crowning / Goal',
        meaning: 'Conscious goals, best possible outcome',
        coordinates: { x: 0.4, y: 0.3, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0],
        colorAccent: '#FFD700',
        section: 'cross'
      },
      {
        index: 5,
        name: 'Near Future',
        meaning: 'Coming soon, next 1-3 months',
        coordinates: { x: 0.6, y: 0.5, rotation: 0 },
        zIndex: 1,
        relatedPositions: [0],
        colorAccent: '#68D391',
        section: 'cross'
      },
      {
        index: 6,
        name: 'Self Perception',
        meaning: 'How you see yourself in this',
        coordinates: { x: 0.8, y: 0.7, rotation: 0 },
        zIndex: 1,
        relatedPositions: [7],
        colorAccent: '#F687B3',
        section: 'staff'
      },
      {
        index: 7,
        name: 'External Influences',
        meaning: 'How others see you, environmental factors',
        coordinates: { x: 0.8, y: 0.55, rotation: 0 },
        zIndex: 1,
        relatedPositions: [6, 8],
        colorAccent: '#63B3ED',
        section: 'staff'
      },
      {
        index: 8,
        name: 'Hopes & Fears',
        meaning: 'Secret wishes and anxieties (often contradictory)',
        coordinates: { x: 0.8, y: 0.4, rotation: 0 },
        zIndex: 1,
        relatedPositions: [7, 9],
        colorAccent: '#ECC94B',
        section: 'staff'
      },
      {
        index: 9,
        name: 'Final Outcome',
        meaning: 'Where all energies are leading, ultimate result',
        coordinates: { x: 0.8, y: 0.25, rotation: 0 },
        zIndex: 1,
        relatedPositions: [8],
        colorAccent: '#E2E8F0',
        section: 'staff'
      }
    ],
    sections: [
      {
        name: 'The Cross',
        cardIndices: [0, 1, 2, 3, 4, 5],
        description: 'The situation itself - what's happening'
      },
      {
        name: 'The Staff',
        cardIndices: [6, 7, 8, 9],
        description: 'Your journey through it - internal to external'
      }
    ],
    viewModes: ['overview', 'sectional', 'detail', 'synthesis']
  },

  // ═══════════════════════════════════════════════════════════════
  // SEVEN-CARD: HORSESHOE SPREAD
  // ═══════════════════════════════════════════════════════════════
  horseshoe: {
    name: 'Horseshoe',
    description: 'Explore multiple facets of a situation',
    cardCount: 7,
    category: SPREAD_CATEGORIES.SPATIAL,
    navigationMode: NAVIGATION_MODES.HORIZONTAL_SWIPE,
    positions: [
      {
        index: 0,
        name: 'Past',
        meaning: "What's behind you",
        coordinates: { x: 0.1, y: 0.7, rotation: -30 },
        zIndex: 1,
        relatedPositions: [1],
        colorAccent: '#9F7AEA',
        arcPosition: 0
      },
      {
        index: 1,
        name: 'Present',
        meaning: 'Where you are now',
        coordinates: { x: 0.25, y: 0.5, rotation: -20 },
        zIndex: 1,
        relatedPositions: [0, 2],
        colorAccent: '#F687B3',
        arcPosition: 1
      },
      {
        index: 2,
        name: 'Hidden Influences',
        meaning: "What you can't see",
        coordinates: { x: 0.4, y: 0.3, rotation: -10 },
        zIndex: 1,
        relatedPositions: [1, 3],
        colorAccent: '#A0AEC0',
        arcPosition: 2
      },
      {
        index: 3,
        name: 'Obstacles',
        meaning: "What stands in the way",
        coordinates: { x: 0.5, y: 0.2, rotation: 0 },
        zIndex: 1,
        relatedPositions: [2, 4],
        colorAccent: '#FC8181',
        arcPosition: 3
      },
      {
        index: 4,
        name: 'Environment',
        meaning: 'External factors',
        coordinates: { x: 0.6, y: 0.3, rotation: 10 },
        zIndex: 1,
        relatedPositions: [3, 5],
        colorAccent: '#63B3ED',
        arcPosition: 4
      },
      {
        index: 5,
        name: 'Advice',
        meaning: 'What action to take',
        coordinates: { x: 0.75, y: 0.5, rotation: 20 },
        zIndex: 1,
        relatedPositions: [4, 6],
        colorAccent: '#68D391',
        arcPosition: 5
      },
      {
        index: 6,
        name: 'Outcome',
        meaning: 'Where this leads',
        coordinates: { x: 0.9, y: 0.7, rotation: 30 },
        zIndex: 1,
        relatedPositions: [5],
        colorAccent: '#FFD700',
        arcPosition: 6
      }
    ],
    viewModes: ['carousel', 'detail']
  }
};

/**
 * Get spread definition by type
 */
export function getSpreadDefinition(spreadType) {
  return SPREAD_DEFINITIONS[spreadType] || null;
}

/**
 * Get all available spreads
 */
export function getAllSpreads() {
  return Object.keys(SPREAD_DEFINITIONS).map(key => ({
    type: key,
    ...SPREAD_DEFINITIONS[key]
  }));
}

/**
 * Get spreads by category
 */
export function getSpreadsByCategory(category) {
  return Object.keys(SPREAD_DEFINITIONS)
    .filter(key => SPREAD_DEFINITIONS[key].category === category)
    .map(key => ({
      type: key,
      ...SPREAD_DEFINITIONS[key]
    }));
}

/**
 * Get card position info within a spread
 */
export function getPositionInfo(spreadType, cardIndex) {
  const spread = SPREAD_DEFINITIONS[spreadType];
  if (!spread || !spread.positions || !spread.positions[cardIndex]) {
    return null;
  }
  return spread.positions[cardIndex];
}
