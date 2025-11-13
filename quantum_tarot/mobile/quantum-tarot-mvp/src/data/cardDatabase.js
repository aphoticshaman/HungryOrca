/**
 * Quantum Tarot - Comprehensive Card Knowledge Database
 * Rich metadata for AGI-powered interpretation
 * All 78 cards with queryable attributes
 */

/**
 * Card schema:
 * {
 *   id: number (0-77),
 *   name: string,
 *   arcana: 'major' | 'minor',
 *   suit: 'wands' | 'cups' | 'swords' | 'pentacles' | null,
 *   rank: 'ace' | '2'-'10' | 'page' | 'knight' | 'queen' | 'king' | null,
 *   number: number (card number in arcana),
 *   element: 'fire' | 'water' | 'air' | 'earth' | 'spirit',
 *   modality: 'cardinal' | 'fixed' | 'mutable' | null,
 *   astrology: string (planet/sign),
 *   numerology: number,
 *   kabbalah: string (Tree of Life path),
 *   symbols: string[] (key visual symbols),
 *   archetypes: string[] (Jungian archetypes),
 *   themes: string[] (core meanings),
 *   keywords: {
 *     upright: string[],
 *     reversed: string[]
 *   },
 *   jungian: string (primary Jungian archetype),
 *   chakra: string,
 *   seasonality: string,
 *   timeframe: string,
 *   advice: string (general guidance),
 *   shadow: string (shadow aspect),
 *   light: string (highest expression),
 *   questions: string[] (reflection prompts),
 *   description: string (symbolism & meaning)
 * }
 */

export const CARD_DATABASE = [
  // ═══════════════════════════════════════════════════════════
  // MAJOR ARCANA (0-21)
  // ═══════════════════════════════════════════════════════════

  {
    id: 0,
    name: "The Fool",
    arcana: "major",
    suit: null,
    rank: null,
    number: 0,
    element: "air",
    modality: null,
    astrology: "uranus",
    numerology: 0,
    kabbalah: "aleph",
    symbols: ["white dog", "cliff edge", "sun", "white rose", "bindle", "mountains"],
    archetypes: ["innocent", "wanderer", "divine_fool", "puer_aeternus"],
    themes: ["new beginnings", "innocence", "spontaneity", "faith", "potential"],
    keywords: {
      upright: ["beginnings", "innocence", "spontaneity", "free spirit", "leap of faith"],
      reversed: ["recklessness", "taken advantage of", "inconsideration", "naivety"]
    },
    jungian: "puer_aeternus",
    chakra: "crown",
    seasonality: "spring_equinox",
    timeframe: "immediate / new cycle beginning",
    advice: "Take the leap. Trust the universe. Begin without knowing the end.",
    shadow: "Reckless naivety, refusal to grow up, avoidance of responsibility",
    light: "Pure potential, divine trust, beginner's mind, fearless authenticity",
    questions: [
      "What new beginning calls to you?",
      "Where do you need to trust more and control less?",
      "What would you do if you knew you couldn't fail?"
    ],
    description: "The Fool stands at the precipice of a great adventure, one foot lifted to step into the unknown. The white dog represents instinct and loyalty, barking either warning or encouragement. The white rose symbolizes purity, the sun divine consciousness, and the mountains challenges yet to come. The Fool carries everything and nothing in his small bag—all he needs is faith. This card represents pure potential, the zero point before creation, and the courage to begin without guarantees."
  },

  {
    id: 1,
    name: "The Magician",
    arcana: "major",
    suit: null,
    rank: null,
    number: 1,
    element: "air",
    modality: null,
    astrology: "mercury",
    numerology: 1,
    kabbalah: "beth",
    symbols: ["infinity symbol", "four tools", "roses", "lilies", "ouroboros belt"],
    archetypes: ["magus", "creator", "manifestor", "trickster"],
    themes: ["manifestation", "power", "skill", "concentration", "action"],
    keywords: {
      upright: ["manifestation", "resourcefulness", "power", "inspired action", "mastery"],
      reversed: ["manipulation", "poor planning", "untapped talents", "illusion"]
    },
    jungian: "magician_archetype",
    chakra: "throat",
    seasonality: "all_seasons",
    timeframe: "present moment / active creation",
    advice: "You have all the tools you need. Focus your will. Speak it into being.",
    shadow: "Manipulation, using gifts for selfish ends, trickery, scattered energy",
    light: "Conscious creation, alignment of will and action, mastery of elements",
    questions: [
      "What are you ready to manifest?",
      "How can you better use the resources available to you?",
      "Where do thought and action need to align?"
    ],
    description: "The Magician stands before a table holding the four suits—pentacle, cup, sword, and wand—representing the four elements and all tools needed to manifest reality. One hand points to heaven, one to earth: 'As above, so below.' The infinity symbol above his head shows unlimited potential. Red roses climb behind him (desire manifested), white lilies at his feet (pure intention). He is the channel between spiritual and material, the one who speaks and makes it so."
  },

  {
    id: 2,
    name: "The High Priestess",
    arcana: "major",
    suit: null,
    rank: null,
    number: 2,
    element: "water",
    modality: null,
    astrology: "moon",
    numerology: 2,
    kabbalah: "gimel",
    symbols: ["two pillars", "veil of pomegranates", "crescent moon", "scroll", "cross"],
    archetypes: ["priestess", "oracle", "divine_feminine", "keeper_of_mysteries"],
    themes: ["intuition", "sacred knowledge", "divine feminine", "unconscious", "mystery"],
    keywords: {
      upright: ["intuition", "sacred knowledge", "divine feminine", "subconscious mind"],
      reversed: ["secrets", "disconnected from intuition", "withdrawal", "silence"]
    },
    jungian: "anima",
    chakra: "third_eye",
    seasonality: "full_moon",
    timeframe: "timeless / outside linear time",
    advice: "Listen to your inner voice. Trust what you know without knowing how you know it.",
    shadow: "Withholding knowledge, using mystery as power, disconnection from wisdom",
    light: "Deep intuition, access to unconscious wisdom, divine receptivity",
    questions: [
      "What does your intuition tell you?",
      "What mysteries are you being invited to explore?",
      "How can you better listen to your inner knowing?"
    ],
    description: "The High Priestess sits between two pillars—B (Boaz, strength) and J (Jachin, establishment)—guarding the threshold between visible and invisible worlds. Behind her hangs a veil decorated with pomegranates (fertility, Persephone's journey to the underworld). She holds the Torah scroll (divine law, hidden wisdom). The crescent moon at her feet shows lunar, receptive consciousness. She is the keeper of mysteries, the one who knows without logic, sees without eyes."
  },

  {
    id: 3,
    name: "The Empress",
    arcana: "major",
    suit: null,
    rank: null,
    number: 3,
    element: "earth",
    modality: null,
    astrology: "venus",
    numerology: 3,
    kabbalah: "daleth",
    symbols: ["venus symbol", "crown of stars", "wheat", "waterfall", "pomegranates"],
    archetypes: ["mother", "creator", "nurturer", "abundance"],
    themes: ["abundance", "nurturing", "fertility", "nature", "creativity"],
    keywords: {
      upright: ["femininity", "beauty", "nature", "nurturing", "abundance", "creativity"],
      reversed: ["creative block", "dependence", "smothering", "lack", "neglect"]
    },
    jungian: "great_mother",
    chakra: "sacral",
    seasonality: "summer / harvest",
    timeframe: "gestation period / 9 months",
    advice: "Create. Nurture. Allow abundance. Trust the fertile ground.",
    shadow: "Smothering love, creative blocks, attachment to outcomes, material focus",
    light: "Unconditional nurturing, abundant creativity, Mother Nature embodied",
    questions: [
      "What are you birthing into being?",
      "How can you nurture yourself and others?",
      "Where is abundance already present in your life?"
    ],
    description: "The Empress reclines on a throne in lush nature, pregnant with creative potential. Her crown bears twelve stars (zodiac, months, dominion over time). The Venus symbol marks her throne. Wheat grows abundantly (harvest), a waterfall flows (emotions, unconscious). Her white gown is decorated with pomegranates (fertility, feminine mysteries). She is Mother Nature herself—abundant, creative, nurturing, the feminine force that grows all things."
  },

  {
    id: 4,
    name: "The Emperor",
    arcana: "major",
    suit: null,
    rank: null,
    number: 4,
    element: "fire",
    modality: null,
    astrology: "aries",
    numerology: 4,
    kabbalah: "heh",
    symbols: ["ram heads", "ankh scepter", "orb", "armor", "stone throne", "barren mountains"],
    archetypes: ["father", "ruler", "authority", "king"],
    themes: ["authority", "structure", "control", "leadership", "father_figure"],
    keywords: {
      upright: ["authority", "establishment", "structure", "father figure", "leadership"],
      reversed: ["domination", "excessive control", "lack of discipline", "inflexibility"]
    },
    jungian: "senex",
    chakra: "solar_plexus",
    seasonality: "established / mature",
    timeframe: "long-term / structural",
    advice: "Build structure. Exercise authority. Create order from chaos.",
    shadow: "Tyranny, rigidity, domination, fear of vulnerability, over-control",
    light: "Just authority, protective strength, wise leadership, stable foundation",
    questions: [
      "Where do you need to establish boundaries?",
      "How can you create more structure in your life?",
      "What needs your leadership and authority?"
    ],
    description: "The Emperor sits on a massive stone throne carved with ram heads (Aries, initiative, leadership) against barren mountains (order imposed on chaos). He wears armor beneath red robes (readiness, action, life force). In his right hand, an ankh (Egyptian symbol of life and power); in his left, a golden orb (the world he rules). His expression is stern but not cruel—a just ruler who makes hard decisions. He is structure, law, paternal authority, the father who protects through strength."
  }

  // TODO: Add remaining 73 cards following this schema
  // This is a starter template - full database will be ~2000 lines
];

/**
 * Helper: Get card by ID
 */
export function getCardData(cardId) {
  return CARD_DATABASE.find(c => c.id === cardId) || null;
}

/**
 * Helper: Get all cards of a suit
 */
export function getCardsBySuit(suit) {
  return CARD_DATABASE.filter(c => c.suit === suit);
}

/**
 * Helper: Get all cards of an element
 */
export function getCardsByElement(element) {
  return CARD_DATABASE.filter(c => c.element === element);
}

/**
 * Helper: Get all Major Arcana
 */
export function getMajorArcana() {
  return CARD_DATABASE.filter(c => c.arcana === 'major');
}

/**
 * Helper: Get all Minor Arcana
 */
export function getMinorArcana() {
  return CARD_DATABASE.filter(c => c.arcana === 'minor');
}

/**
 * Helper: Search cards by theme
 */
export function searchCardsByTheme(theme) {
  return CARD_DATABASE.filter(c =>
    c.themes.some(t => t.toLowerCase().includes(theme.toLowerCase()))
  );
}

/**
 * Helper: Search cards by keyword
 */
export function searchCardsByKeyword(keyword, reversed = false) {
  return CARD_DATABASE.filter(c => {
    const keywords = reversed ? c.keywords.reversed : c.keywords.upright;
    return keywords.some(k => k.toLowerCase().includes(keyword.toLowerCase()));
  });
}
