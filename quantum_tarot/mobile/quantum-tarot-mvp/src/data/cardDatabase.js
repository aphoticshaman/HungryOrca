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
  },

  {
    id: 5,
    name: "The Hierophant",
    arcana: "major",
    suit: null,
    rank: null,
    number: 5,
    element: "earth",
    modality: null,
    astrology: "taurus",
    numerology: 5,
    kabbalah: "vav",
    symbols: ["twin pillars", "crossed keys", "papal crown", "religious devotees", "hand of blessing"],
    archetypes: ["priest", "teacher", "tradition keeper", "spiritual authority"],
    themes: ["tradition", "conformity", "education", "spiritual_wisdom", "institutions"],
    keywords: {
      upright: ["tradition", "conformity", "education", "spiritual wisdom", "institutions"],
      reversed: ["rebellion", "subversiveness", "new approaches", "freedom", "challenging tradition"]
    },
    jungian: "wise_old_man",
    chakra: "throat",
    seasonality: "established / traditional",
    timeframe: "long-term / enduring",
    advice: "Seek wisdom from tradition. Honor what has been proven. Learn from established systems.",
    shadow_work: "Dogma, blind faith, rigid thinking, fear of questioning authority",
    light: "Spiritual guidance, sacred knowledge, mentorship, moral compass",
    questions: [
      "What traditions serve you and which constrain you?",
      "Who are your teachers and guides?",
      "How can you honor the past while creating the future?"
    ],
    description: "The Hierophant sits between twin pillars (duality, gateway to higher knowledge) wearing a triple crown (three worlds: material, intellectual, spiritual). He raises his right hand in blessing while holding crossed keys (keys to heaven, esoteric knowledge). Two devotees kneel before him (the seeker and the initiate). He represents spiritual authority, tradition, conformity, and the established order—the bridge between divine and human."
  },

  {
    id: 6,
    name: "The Lovers",
    arcana: "major",
    suit: null,
    rank: null,
    number: 6,
    element: "air",
    modality: null,
    astrology: "gemini",
    numerology: 6,
    kabbalah: "zayin",
    symbols: ["angel", "naked figures", "tree of knowledge", "tree of life", "mountain"],
    archetypes: ["lovers", "union", "choice", "divine masculine and feminine"],
    themes: ["love", "harmony", "relationships", "choices", "values"],
    keywords: {
      upright: ["love", "harmony", "relationships", "values alignment", "choices"],
      reversed: ["disharmony", "imbalance", "misalignment", "disconnection", "difficult choices"]
    },
    jungian: "anima_animus",
    chakra: "heart",
    seasonality: "union / peak connection",
    timeframe: "present moment choice",
    advice: "Choose from your heart. Align your values. Embrace partnership and union.",
    shadow_work: "Codependency, fear of commitment, choosing based on fear, betrayal",
    light: "Sacred union, aligned values, conscious choice, divine love",
    questions: [
      "What choices align with your deepest values?",
      "Where do you need more harmony in relationships?",
      "What does authentic union look like for you?"
    ],
    description: "A man and woman stand naked beneath an angel (Archangel Raphael, divine blessing). Behind the woman is the Tree of Knowledge with serpent (temptation, consciousness); behind the man, the Tree of Life with flames (passion, vitality). A mountain rises between them (challenges overcome through partnership). This card represents choice, union, values alignment—the moment when two become one through conscious decision."
  },

  {
    id: 7,
    name: "The Chariot",
    arcana: "major",
    suit: null,
    rank: null,
    number: 7,
    element: "water",
    modality: "cardinal",
    astrology: "cancer",
    numerology: 7,
    kabbalah: "cheth",
    symbols: ["chariot", "sphinxes", "armor", "city walls", "starry canopy", "wand"],
    archetypes: ["warrior", "victor", "conqueror", "driver"],
    themes: ["willpower", "determination", "victory", "control", "direction"],
    keywords: {
      upright: ["control", "willpower", "success", "determination", "direction"],
      reversed: ["lack of control", "lack of direction", "aggression", "scattered energy"]
    },
    jungian: "hero",
    chakra: "solar_plexus",
    seasonality: "momentum / forward motion",
    timeframe: "immediate action / decisive moment",
    advice: "Take control. Focus your will. Move forward with determination and clarity.",
    shadow_work: "Over-control, aggression, directionlessness, scattered willpower",
    light: "Focused will, triumph through discipline, mastery of opposing forces",
    questions: [
      "Where do you need to take control?",
      "What opposing forces must you balance?",
      "What victory are you moving toward?"
    ],
    description: "A warrior sits in a chariot beneath a starry canopy (cosmic protection), wearing armor decorated with moons and stars. Two sphinxes (one black, one white—opposing forces) sit before him, ready to move. He holds a wand (willpower, direction) but no reins—he controls through mastery of self. The chariot represents triumph through willpower, controlling opposing forces through balance and determination."
  },

  {
    id: 8,
    name: "Strength",
    arcana: "major",
    suit: null,
    rank: null,
    number: 8,
    element: "fire",
    modality: "fixed",
    astrology: "leo",
    numerology: 8,
    kabbalah: "teth",
    symbols: ["woman", "lion", "infinity symbol", "flowers", "mountain"],
    archetypes: ["tamer", "gentle warrior", "compassionate power"],
    themes: ["courage", "compassion", "influence", "inner_strength", "patience"],
    keywords: {
      upright: ["strength", "courage", "compassion", "influence", "patience"],
      reversed: ["weakness", "self-doubt", "lack of discipline", "abuse of power"]
    },
    jungian: "self",
    chakra: "heart",
    seasonality: "sustained power / endurance",
    timeframe: "patient endurance",
    advice: "Lead with compassion. Trust your inner strength. Gentleness overcomes force.",
    shadow_work: "Brutality, weakness masked as aggression, lack of self-control",
    light: "Compassionate power, gentle influence, courage from the heart",
    questions: [
      "Where do you need gentle strength over force?",
      "How can you tame your inner beasts with love?",
      "What requires patient, compassionate power?"
    ],
    description: "A woman gently closes the mouth of a lion, crowned with flowers and infinity symbol (infinite patience, divine feminine strength). She needs no force—her compassion and courage tame the beast. Mountains rise behind (challenges overcome through inner strength). This card represents true power: not domination but gentle influence, courage born of compassion, strength through love."
  },

  {
    id: 9,
    name: "The Hermit",
    arcana: "major",
    suit: null,
    rank: null,
    number: 9,
    element: "earth",
    modality: "mutable",
    astrology: "virgo",
    numerology: 9,
    kabbalah: "yod",
    symbols: ["lantern", "staff", "mountain peak", "gray robes", "six-pointed star"],
    archetypes: ["sage", "seeker", "guide", "solitary"],
    themes: ["introspection", "soul_searching", "wisdom", "solitude", "guidance"],
    keywords: {
      upright: ["soul searching", "introspection", "inner guidance", "wisdom", "solitude"],
      reversed: ["isolation", "loneliness", "withdrawal", "lost your way"]
    },
    jungian: "wise_old_man",
    chakra: "third_eye",
    seasonality: "withdrawal / contemplation",
    timeframe: "pause for reflection",
    advice: "Withdraw to find clarity. Seek wisdom within. Be the light in darkness.",
    shadow_work: "Isolation, loneliness, running from connection, spiritual bypassing",
    light: "Inner wisdom, spiritual illumination, conscious solitude, self-discovery",
    questions: [
      "What do you need to withdraw from to find clarity?",
      "What wisdom lies in your solitude?",
      "How can you be both guide and seeker?"
    ],
    description: "An old man stands alone at a mountain peak holding a lantern containing a six-pointed star (seal of Solomon, divine wisdom, light in darkness). His gray robes represent invisibility in the material world. He has reached the summit through solitary journey. The Hermit represents withdrawal, introspection, seeking truth through solitude, being your own light."
  },

  {
    id: 10,
    name: "Wheel of Fortune",
    arcana: "major",
    suit: null,
    rank: null,
    number: 10,
    element: "fire",
    modality: null,
    astrology: "jupiter",
    numerology: 10,
    kabbalah: "kaph",
    symbols: ["wheel", "sphinx", "snake", "anubis", "TARO letters", "alchemical symbols"],
    archetypes: ["fate", "destiny", "fortune", "cycles"],
    themes: ["change", "cycles", "destiny", "luck", "karma"],
    keywords: {
      upright: ["good luck", "karma", "life cycles", "destiny", "turning point"],
      reversed: ["bad luck", "resistance to change", "breaking cycles"]
    },
    jungian: "self",
    chakra: "crown",
    seasonality: "turning point / seasonal shift",
    timeframe: "cycles completing",
    advice: "Accept change. Trust the cycles. What goes around comes around.",
    shadow_work: "Victim mentality, resistance to change, fatalism",
    light: "Divine timing, karmic lessons, embracing life's cycles",
    questions: [
      "What cycle is completing in your life?",
      "How can you work with fate rather than resist it?",
      "What have you set in motion that's now returning?"
    ],
    description: "A great wheel turns in the sky surrounded by clouds. A sphinx sits atop with sword (equilibrium), a snake descends on the left (material descent), and Anubis rises on the right (spiritual ascent). Hebrew letters spell YHVH (the divine name), Latin letters spell TARO/ROTA (the wheel). The wheel represents fate, karma, cycles—what rises must fall, what falls must rise again."
  },

  {
    id: 11,
    name: "Justice",
    arcana: "major",
    suit: null,
    rank: null,
    number: 11,
    element: "air",
    modality: "cardinal",
    astrology: "libra",
    numerology: 11,
    kabbalah: "lamed",
    symbols: ["scales", "sword", "throne", "pillars", "crown", "purple robe"],
    archetypes: ["judge", "arbiter", "balance keeper"],
    themes: ["justice", "fairness", "truth", "law", "cause_and_effect"],
    keywords: {
      upright: ["justice", "fairness", "truth", "cause and effect", "law"],
      reversed: ["unfairness", "lack of accountability", "dishonesty"]
    },
    jungian: "persona",
    chakra: "throat",
    seasonality: "balance point / equinox",
    timeframe: "karmic return / judgment day",
    advice: "Seek truth. Be fair. Accept consequences. Balance the scales.",
    shadow_work: "Judgment, harsh criticism, imbalance, dishonesty",
    light: "Divine justice, truth, karmic balance, fair judgment",
    questions: [
      "Where do you need to restore balance?",
      "What truth must be faced?",
      "How can you be both merciful and just?"
    ],
    description: "Justice sits between twin pillars wearing a crown and purple robe (spiritual authority), holding a raised sword (truth cutting through illusion) and balanced scales (perfect equilibrium). A square clasp holds her cloak (material stability, law). She represents karmic law, truth, fairness—the moment of reckoning where all is weighed and measured."
  },

  {
    id: 12,
    name: "The Hanged Man",
    arcana: "major",
    suit: null,
    rank: null,
    number: 12,
    element: "water",
    modality: null,
    astrology: "neptune",
    numerology: 12,
    kabbalah: "mem",
    symbols: ["upside down figure", "halo", "T-shaped tree", "rope", "serene expression"],
    archetypes: ["martyr", "mystic", "surrenderer", "Odin"],
    themes: ["surrender", "letting_go", "new_perspective", "sacrifice", "suspension"],
    keywords: {
      upright: ["surrender", "letting go", "new perspective", "sacrifice"],
      reversed: ["stalling", "needless sacrifice", "fear of sacrifice", "resistance"]
    },
    jungian: "transcendent_function",
    chakra: "third_eye",
    seasonality: "suspension / liminal space",
    timeframe: "pause between / limbo",
    advice: "Let go. Surrender control. See from a new angle. Sacrifice the lesser for the greater.",
    shadow_work: "Martyrdom, victim consciousness, stuck in limbo, fear of surrender",
    light: "Willing sacrifice, enlightened perspective, spiritual surrender",
    questions: [
      "What do you need to release?",
      "How does seeing upside-down change everything?",
      "What sacrifice leads to greater wisdom?"
    ],
    description: "A man hangs upside-down from a living tree, suspended by one foot in a figure-4 position (the number 4 inverted, material reality reversed). His free leg forms a cross with the other. A golden halo surrounds his head—he has achieved enlightenment through surrender. His expression is serene. The Hanged Man represents willing sacrifice, seeing the world anew, suspension between worlds."
  },

  {
    id: 13,
    name: "Death",
    arcana: "major",
    suit: null,
    rank: null,
    number: 13,
    element: "water",
    modality: "fixed",
    astrology: "scorpio",
    numerology: 13,
    kabbalah: "nun",
    symbols: ["skeleton", "armor", "white horse", "black flag", "white rose", "sun rising"],
    archetypes: ["transformer", "destroyer", "reaper", "phoenix"],
    themes: ["endings", "transformation", "transition", "letting_go", "rebirth"],
    keywords: {
      upright: ["endings", "transformation", "transition", "letting go", "rebirth"],
      reversed: ["resistance to change", "stagnation", "fear of endings"]
    },
    jungian: "shadow",
    chakra: "root",
    seasonality: "death and rebirth / winter",
    timeframe: "ending to allow beginning",
    advice: "Let what must die, die. Embrace transformation. Clear space for new life.",
    shadow_work: "Fear of death/change, clinging, stagnation, destroying without creating",
    light: "Necessary endings, transformation, phoenix rising, letting go",
    questions: [
      "What needs to die so something new can be born?",
      "What are you holding onto that must be released?",
      "How can you embrace transformation?"
    ],
    description: "A skeleton in black armor rides a white horse carrying a black flag with a white rose (purity in darkness, beauty in death). A king lies dead, a bishop prays, a child watches—death comes for all. Yet the sun rises between twin towers (rebirth). Death represents transformation, necessary endings, clearing the old to make way for new—not literal death but ego death, transformation, the end of cycles."
  },

  {
    id: 14,
    name: "Temperance",
    arcana: "major",
    suit: null,
    rank: null,
    number: 14,
    element: "fire",
    modality: "mutable",
    astrology: "sagittarius",
    numerology: 14,
    kabbalah: "samekh",
    symbols: ["angel", "two cups", "water flow", "triangle", "iris flowers", "mountain path"],
    archetypes: ["alchemist", "healer", "mediator", "angel"],
    themes: ["balance", "moderation", "patience", "alchemy", "healing"],
    keywords: {
      upright: ["balance", "moderation", "patience", "purpose", "meaning"],
      reversed: ["imbalance", "excess", "lack of harmony", "extremes"]
    },
    jungian: "transcendent_function",
    chakra: "heart",
    seasonality: "integration / alchemy",
    timeframe: "patient blending",
    advice: "Find the middle way. Mix opposing forces. Patience creates magic.",
    shadow_work: "Imbalance, extremism, impatience, forcing outcomes",
    light: "Divine alchemy, perfect balance, patient transformation",
    questions: [
      "What opposing forces need integration?",
      "Where do you need more balance?",
      "What is your unique alchemy?"
    ],
    description: "An angel with red wings (passion tempered by spirit) stands with one foot in water, one on land (balance of conscious/unconscious, physical/spiritual). She pours water between two cups in an impossible flow (alchemy, the mixing of opposites creates the new). A triangle on her chest (fire), iris flowers (Iris, goddess of rainbow bridge), mountain path winding upward. Temperance represents balance, moderation, the alchemical middle way."
  },

  {
    id: 15,
    name: "The Devil",
    arcana: "major",
    suit: null,
    rank: null,
    number: 15,
    element: "earth",
    modality: "cardinal",
    astrology: "capricorn",
    numerology: 15,
    kabbalah: "ayin",
    symbols: ["horned devil", "inverted pentagram", "torch", "chains", "naked figures"],
    archetypes: ["shadow", "tempter", "enslaver", "Pan"],
    themes: ["bondage", "addiction", "materialism", "shadow_self", "lust"],
    keywords: {
      upright: ["bondage", "addiction", "sexuality", "materialism", "shadow self"],
      reversed: ["release", "breaking free", "power reclaimed", "facing shadow"]
    },
    jungian: "shadow",
    chakra: "root",
    seasonality: "entrapment / winter solstice",
    timeframe: "bondage until awareness",
    advice: "Face your shadow. Own your desires. Recognize self-imposed chains.",
    shadow_work: "Addiction, materialism, lust, fear, denial of shadow",
    light: "Shadow integration, reclaimed power, conscious sexuality",
    questions: [
      "What enslaves you?",
      "What desires run your life unconsciously?",
      "How can you transform shadow into power?"
    ],
    description: "A horned devil figure (Baphomet, material world, base instincts) sits on a pedestal with inverted pentagram (spirit subjugated to matter). A man and woman stand chained before him with tails—they've become like him (horns sprouting). But the chains are loose—they could remove them. The torch points downward (inverted divine fire). The Devil represents bondage, addiction, shadow—but the bondage is self-imposed."
  },

  {
    id: 16,
    name: "The Tower",
    arcana: "major",
    suit: null,
    rank: null,
    number: 16,
    element: "fire",
    modality: null,
    astrology: "mars",
    numerology: 16,
    kabbalah: "peh",
    symbols: ["tower", "lightning", "crown", "falling figures", "flames"],
    archetypes: ["destroyer", "liberator", "chaos bringer"],
    themes: ["sudden_change", "upheaval", "revelation", "destruction", "liberation"],
    keywords: {
      upright: ["sudden change", "upheaval", "chaos", "revelation", "awakening"],
      reversed: ["avoiding disaster", "fear of change", "delayed disaster"]
    },
    jungian: "shadow",
    chakra: "crown",
    seasonality: "sudden upheaval / lightning strike",
    timeframe: "sudden / instantaneous",
    advice: "Let the false be destroyed. Embrace upheaval. Truth liberates even as it destroys.",
    shadow_work: "Resistance to necessary destruction, clinging to false structures",
    light: "Liberation through destruction, revelation, awakening",
    questions: [
      "What false structure must fall?",
      "What truth have you been avoiding?",
      "How can destruction be liberation?"
    ],
    description: "A tall tower built on a mountain is struck by lightning. The golden crown atop it flies off (ego's illusion of control shattered). Two figures fall headfirst from windows (forced awakening). Flames shoot from windows. The Tower represents sudden upheaval, destruction of false structures, the lightning bolt of truth—catastrophic change that liberates even as it destroys."
  },

  {
    id: 17,
    name: "The Star",
    arcana: "major",
    suit: null,
    rank: null,
    number: 17,
    element: "air",
    modality: "fixed",
    astrology: "aquarius",
    numerology: 17,
    kabbalah: "tzaddi",
    symbols: ["naked woman", "eight stars", "two urns", "water", "ibis bird", "tree"],
    archetypes: ["muse", "hope bringer", "divine feminine"],
    themes: ["hope", "renewal", "inspiration", "serenity", "spirituality"],
    keywords: {
      upright: ["hope", "faith", "renewal", "inspiration", "serenity"],
      reversed: ["lack of faith", "despair", "disconnection", "hopelessness"]
    },
    jungian: "anima",
    chakra: "crown",
    seasonality: "hope after darkness / spring promise",
    timeframe: "renewal beginning",
    advice: "Have faith. Trust the universe. Pour yourself out. Renewal is coming.",
    shadow_work: "Despair, lack of faith, spiritual disconnection",
    light: "Divine hope, spiritual renewal, faith in the universe",
    questions: [
      "What renews your faith?",
      "How can you be both vessel and gift?",
      "What hope guides you through darkness?"
    ],
    description: "A naked woman kneels by a pool, pouring water from two urns—one onto land (nourishing the material), one into water (returning to source). Above her, eight stars shine (seven chakras plus the soul). An ibis perches in a tree (Thoth, divine wisdom). She is completely vulnerable, open, giving. The Star represents hope, renewal, faith—after the Tower's destruction comes healing, the promise of new life."
  },

  {
    id: 18,
    name: "The Moon",
    arcana: "major",
    suit: null,
    rank: null,
    number: 18,
    element: "water",
    modality: null,
    astrology: "pisces",
    numerology: 18,
    kabbalah: "qoph",
    symbols: ["moon", "dog", "wolf", "crayfish", "path", "towers"],
    archetypes: ["unconscious", "shadow", "dream weaver"],
    themes: ["illusion", "intuition", "unconscious", "fear", "dreams"],
    keywords: {
      upright: ["illusion", "fear", "anxiety", "intuition", "unconscious"],
      reversed: ["release of fear", "clarity", "truth revealed"]
    },
    jungian: "shadow",
    chakra: "third_eye",
    seasonality: "darkness / full moon",
    timeframe: "night journey / confusion",
    advice: "Trust your intuition. Navigate illusion. Face your fears. The path through darkness leads to dawn.",
    shadow_work: "Fear, anxiety, illusion, denial of shadow",
    light: "Deep intuition, shadow work, navigating the unconscious",
    questions: [
      "What illusions must you see through?",
      "What does your unconscious reveal in dreams?",
      "How can you navigate fear to find truth?"
    ],
    description: "A full moon with a face drips dew (tears of the night). A path runs between two towers toward mountains (the journey through darkness). A dog and wolf howl (tamed and wild nature). A crayfish emerges from water (primordial unconscious rising). The Moon represents illusion, fear, the unconscious—the realm of dreams, intuition, and shadows. Nothing is as it seems. Trust deeper knowing."
  },

  {
    id: 19,
    name: "The Sun",
    arcana: "major",
    suit: null,
    rank: null,
    number: 19,
    element: "fire",
    modality: null,
    astrology: "sun",
    numerology: 19,
    kabbalah: "resh",
    symbols: ["radiant sun", "child", "white horse", "sunflowers", "red banner"],
    archetypes: ["child", "innocent", "light bringer"],
    themes: ["joy", "success", "vitality", "enlightenment", "innocence"],
    keywords: {
      upright: ["success", "vitality", "joy", "confidence", "enlightenment"],
      reversed: ["temporary depression", "lack of success", "sadness"]
    },
    jungian: "self",
    chakra: "solar_plexus",
    seasonality: "summer / peak vitality",
    timeframe: "fullest expression / now",
    advice: "Celebrate. Shine your light. Be joyfully yourself. Success is yours.",
    shadow_work: "False optimism, naivety, ego inflation",
    light: "Pure joy, authentic success, enlightened innocence",
    questions: [
      "Where can you be more authentically joyful?",
      "What success is ready to be celebrated?",
      "How can you reclaim childlike wonder?"
    ],
    description: "A radiant sun shines with a human face (conscious awareness). A naked child rides a white horse (innocent mastery, pure spirit in control). He holds a red banner (life force, victory). Sunflowers grow behind a brick wall (joy growing in the garden of consciousness). The Sun represents pure joy, success, vitality—enlightenment, the soul fully expressed, conscious awareness illuminating all."
  },

  {
    id: 20,
    name: "Judgement",
    arcana: "major",
    suit: null,
    rank: null,
    number: 20,
    element: "fire",
    modality: null,
    astrology: "pluto",
    numerology: 20,
    kabbalah: "shin",
    symbols: ["angel", "trumpet", "rising figures", "coffins", "mountains", "cross flag"],
    archetypes: ["resurrector", "caller", "awakener"],
    themes: ["judgement", "rebirth", "inner_calling", "absolution", "resurrection"],
    keywords: {
      upright: ["judgement", "rebirth", "inner calling", "absolution"],
      reversed: ["self-doubt", "refusal of calling", "lack of self-awareness"]
    },
    jungian: "self",
    chakra: "throat",
    seasonality: "resurrection / awakening",
    timeframe: "final reckoning / rebirth",
    advice: "Answer your calling. Rise renewed. Accept absolution. Be reborn.",
    shadow_work: "Self-judgment, refusing the call, spiritual bypassing",
    light: "Answering the call, resurrection, spiritual rebirth",
    questions: [
      "What is your higher calling?",
      "What must you forgive to be reborn?",
      "How are you being called to rise?"
    ],
    description: "Archangel Gabriel blows a trumpet with a cross flag (divine call to resurrection). Below, naked figures rise from coffins with arms outstretched (resurrection of the dead, spiritual rebirth). Mountains in the distance (challenges transcended). Judgement represents the moment of reckoning, answering your soul's calling, being reborn into higher purpose—resurrection, absolution, the final accounting before wholeness."
  },

  {
    id: 21,
    name: "The World",
    arcana: "major",
    suit: null,
    rank: null,
    number: 21,
    element: "earth",
    modality: null,
    astrology: "saturn",
    numerology: 21,
    kabbalah: "tav",
    symbols: ["wreath", "dancing figure", "four living creatures", "infinity ribbons"],
    archetypes: ["cosmic dancer", "world soul", "completion"],
    themes: ["completion", "accomplishment", "travel", "wholeness", "integration"],
    keywords: {
      upright: ["completion", "accomplishment", "travel", "wholeness"],
      reversed: ["incompletion", "lack of closure", "shortcuts"]
    },
    jungian: "self",
    chakra: "crown",
    seasonality: "completion / harvest",
    timeframe: "cycle complete / eternal",
    advice: "Celebrate completion. Embrace wholeness. The journey is done. You are the world.",
    shadow_work: "Fear of completion, never finishing, refusing wholeness",
    light: "Divine wholeness, cosmic consciousness, sacred completion",
    questions: [
      "What cycle is complete?",
      "How have you become whole?",
      "What mastery have you achieved?"
    ],
    description: "A naked dancer moves within a laurel wreath (victory, completion, eternal cycle). She holds two wands (active manifestation). In the four corners, four creatures (lion, bull, eagle, human—the four fixed signs, four elements, four evangelists—wholeness). Infinity ribbons flow (eternal return). The World represents completion, wholeness, cosmic consciousness—the end of the Fool's journey and the beginning of the next spiral. Integration of all parts. The universe itself."
  },

  // ═══════════════════════════════════════════════════════════
  // MINOR ARCANA: SWORDS (22-35)
  // Element: Air | Mental realm, thought, conflict, truth
  // ═══════════════════════════════════════════════════════════

  {
    id: 22,
    name: "Ace of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "ace",
    number: 1,
    element: "air",
    modality: "cardinal",
    astrology: "air signs",
    numerology: 1,
    kabbalah: "kether of air",
    symbols: ["upright sword", "crown", "mountains", "clouds"],
    archetypes: ["mental clarity", "truth seeker", "breakthrough"],
    themes: ["breakthrough", "clarity", "mental power", "truth", "new ideas"],
    keywords: {
      upright: ["breakthrough", "clarity", "truth", "mental power", "new ideas"],
      reversed: ["confusion", "mental fog", "lies", "blocked communication"]
    },
    jungian: "thinking",
    chakra: "throat",
    seasonality: "spring / dawn of thought",
    timeframe: "sudden / immediate",
    advice: "Cut through illusion. Speak your truth. Mental clarity is power.",
    shadow_work: "Using truth as a weapon, intellectual cruelty, cutting others down",
    light: "Pure truth, mental clarity, breakthrough insight",
    questions: [
      "What truth needs to be spoken?",
      "What illusion must be cut away?",
      "Where do you need mental clarity?"
    ],
    description: "A hand emerges from clouds gripping a sword crowned with laurel (victory through truth). Mountains in the background (mental challenges conquered). The Ace of Swords represents breakthrough, cutting through confusion, mental clarity—the moment when truth pierces illusion."
  },

  {
    id: 23,
    name: "Two of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "2",
    number: 2,
    element: "air",
    modality: "fixed",
    astrology: "moon in libra",
    numerology: 2,
    kabbalah: "chokmah of air",
    symbols: ["blindfolded figure", "crossed swords", "water", "crescent moon"],
    archetypes: ["indecision", "stalemate", "avoidance"],
    themes: ["indecision", "stalemate", "avoidance", "denial", "difficult choice"],
    keywords: {
      upright: ["indecision", "stalemate", "avoidance", "difficult choice"],
      reversed: ["information overload", "decision made", "removal of blindfold"]
    },
    jungian: "repression",
    chakra: "third eye",
    seasonality: "stagnation / crossroads",
    timeframe: "delayed / waiting",
    advice: "Remove the blindfold. Face what you're avoiding. Make the choice.",
    shadow_work: "Willful blindness, refusing to see truth, decision paralysis",
    light: "Peaceful contemplation, taking time to decide wisely",
    questions: [
      "What are you refusing to see?",
      "What decision are you avoiding?",
      "What happens if you remove the blindfold?"
    ],
    description: "A blindfolded figure sits holding two crossed swords (blocking out reality to avoid choice). Water and crescent moon behind (emotions suppressed by intellect). The Two of Swords represents indecision, stalemate, willful blindness—refusing to see what must be seen."
  },

  {
    id: 24,
    name: "Three of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "3",
    number: 3,
    element: "air",
    modality: "mutable",
    astrology: "saturn in libra",
    numerology: 3,
    kabbalah: "binah of air",
    symbols: ["heart pierced by three swords", "storm clouds", "rain"],
    archetypes: ["heartbreak", "grief", "painful truth"],
    themes: ["heartbreak", "sorrow", "painful truth", "grief", "separation"],
    keywords: {
      upright: ["heartbreak", "sorrow", "painful truth", "grief"],
      reversed: ["healing", "forgiveness", "releasing pain"]
    },
    jungian: "suffering",
    chakra: "heart",
    seasonality: "storm / crisis",
    timeframe: "painful present",
    advice: "Feel the pain. Truth hurts before it heals. Grief is love persisting.",
    shadow_work: "Holding onto pain, refusing to heal, weaponizing suffering",
    light: "Honest grief, truth even when painful, necessary heartbreak for growth",
    questions: [
      "What painful truth must you accept?",
      "Where is your heart wounded?",
      "What grief needs to be felt?"
    ],
    description: "Three swords pierce a red heart under stormy skies (painful truth, heartbreak, necessary wound). Rain falls (cleansing tears). The Three of Swords represents sorrow, painful truth, heartbreak—the moment when reality cuts through illusion and the heart must break open."
  },

  {
    id: 25,
    name: "Four of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "4",
    number: 4,
    element: "air",
    modality: "cardinal",
    astrology: "jupiter in libra",
    numerology: 4,
    kabbalah: "chesed of air",
    symbols: ["resting figure", "stained glass window", "three swords on wall", "one sword beneath"],
    archetypes: ["rest", "recuperation", "contemplation"],
    themes: ["rest", "recuperation", "meditation", "timeout", "recovery"],
    keywords: {
      upright: ["rest", "recuperation", "meditation", "timeout"],
      reversed: ["burnout", "restlessness", "stagnation"]
    },
    jungian: "integration",
    chakra: "crown",
    seasonality: "winter / dormancy",
    timeframe: "pause / recovery period",
    advice: "Rest is not weakness. Recuperate. Contemplation restores power.",
    shadow_work: "Avoiding life through rest, depression disguised as meditation",
    light: "Sacred rest, intentional recovery, meditative contemplation",
    questions: [
      "What needs to rest?",
      "Where are you depleted?",
      "What happens if you pause?"
    ],
    description: "A figure rests in prayer position, hands folded (peaceful recuperation). Three swords hang on wall (past battles), one beneath (remaining vigilance). Stained glass shows blessing (divine rest). The Four of Swords represents rest, recovery, contemplation—the necessary pause before re-engaging."
  },

  {
    id: 26,
    name: "Five of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "5",
    number: 5,
    element: "air",
    modality: "fixed",
    astrology: "venus in aquarius",
    numerology: 5,
    kabbalah: "geburah of air",
    symbols: ["figure collecting swords", "defeated opponents walking away", "stormy sky"],
    archetypes: ["conflict", "defeat", "hollow victory"],
    themes: ["conflict", "defeat", "hollow victory", "unfair tactics", "self-interest"],
    keywords: {
      upright: ["conflict", "defeat", "hollow victory", "self-interest"],
      reversed: ["reconciliation", "making amends", "moving on"]
    },
    jungian: "shadow",
    chakra: "solar plexus",
    seasonality: "aftermath / winter",
    timeframe: "recent past / consequences",
    advice: "Not every battle is worth winning. Know when victory is hollow. Choose your battles.",
    shadow_work: "Winning at all costs, cruelty, taking advantage, gloating",
    light: "Walking away from toxic battles, knowing when to surrender",
    questions: [
      "What victory would be hollow?",
      "What battle isn't worth fighting?",
      "Where have you won but lost yourself?"
    ],
    description: "A figure collects five swords with a smirk while two defeated figures walk away (hollow victory through unfair means). Stormy sky (conflict leaves damage). The Five of Swords represents conflict, defeat, the cost of winning through cruelty—victory that damages the victor."
  },

  {
    id: 27,
    name: "Six of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "6",
    number: 6,
    element: "air",
    modality: "mutable",
    astrology: "mercury in aquarius",
    numerology: 6,
    kabbalah: "tiphareth of air",
    symbols: ["boat", "ferryman", "woman and child", "six swords", "calm waters ahead"],
    archetypes: ["transition", "moving on", "journey"],
    themes: ["transition", "moving on", "travel", "healing journey", "leaving behind"],
    keywords: {
      upright: ["transition", "moving on", "travel", "healing journey"],
      reversed: ["stuck in the past", "unable to move on", "turbulent waters"]
    },
    jungian: "transition",
    chakra: "sacral",
    seasonality: "autumn / passage",
    timeframe: "gradual / transitional",
    advice: "Move forward. The journey heals. You cannot stay where you are.",
    shadow_work: "Refusing to move on, romanticizing the past, fear of the new",
    light: "Necessary departures, healing through movement, guided transitions",
    questions: [
      "What must you leave behind?",
      "Where are you being called to travel?",
      "What healing requires departure?"
    ],
    description: "A ferryman guides a boat carrying woman and child across water (guided transition). Six swords stand upright in boat (carrying wounds but moving forward). Rough waters behind, calm ahead (moving toward healing). The Six of Swords represents transition, moving on, the healing journey—leaving pain behind."
  },

  {
    id: 28,
    name: "Seven of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "7",
    number: 7,
    element: "air",
    modality: "cardinal",
    astrology: "moon in aquarius",
    numerology: 7,
    kabbalah: "netzach of air",
    symbols: ["figure sneaking away with swords", "military camp", "two swords left behind"],
    archetypes: ["deception", "strategy", "theft"],
    themes: ["deception", "strategy", "theft", "getting away with something", "betrayal"],
    keywords: {
      upright: ["deception", "strategy", "theft", "sneakiness"],
      reversed: ["getting caught", "confession", "coming clean"]
    },
    jungian: "trickster",
    chakra: "throat",
    seasonality: "night / covert action",
    timeframe: "secretive / hidden",
    advice: "Strategy is not the same as deception. Use intelligence wisely. Own your shadows.",
    shadow_work: "Deception, theft, betrayal, taking what isn't yours, sneakiness",
    light: "Strategic thinking, necessary secrets, protecting yourself",
    questions: [
      "Where are you being dishonest?",
      "What are you taking that isn't yours?",
      "What secret is eating at you?"
    ],
    description: "A figure sneaks away from camp carrying five swords, leaving two behind (theft, deception, getting away with something). Looking back over shoulder (guilty conscience). The Seven of Swords represents deception, strategy, betrayal—taking what isn't yours or hiding what is."
  },

  {
    id: 29,
    name: "Eight of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "8",
    number: 8,
    element: "air",
    modality: "fixed",
    astrology: "jupiter in gemini",
    numerology: 8,
    kabbalah: "hod of air",
    symbols: ["bound and blindfolded figure", "eight swords", "marshy ground", "castle in distance"],
    archetypes: ["imprisonment", "victim", "self-limitation"],
    themes: ["imprisonment", "restriction", "victim mentality", "self-limitation", "fear"],
    keywords: {
      upright: ["imprisonment", "restriction", "victim mentality", "self-limitation"],
      reversed: ["freedom", "release", "self-empowerment"]
    },
    jungian: "victim",
    chakra: "solar plexus",
    seasonality: "trapped / stagnation",
    timeframe: "feeling endless / stuck",
    advice: "You are freer than you think. The bindings are loose. Remove your own blindfold.",
    shadow_work: "Victim mentality, learned helplessness, refusing to free yourself",
    light: "Recognizing self-imposed limitations, choosing freedom",
    questions: [
      "What prison have you built?",
      "Where do you play the victim?",
      "What if you removed your own blindfold?"
    ],
    description: "A bound and blindfolded figure stands among eight swords stuck in muddy ground (self-imposed prison). Bindings are loose (could escape). Castle in distance (safety accessible). The Eight of Swords represents self-limitation, victim mentality—imprisonment by belief rather than reality."
  },

  {
    id: 30,
    name: "Nine of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "9",
    number: 9,
    element: "air",
    modality: "mutable",
    astrology: "mars in gemini",
    numerology: 9,
    kabbalah: "yesod of air",
    symbols: ["figure sitting up in bed", "head in hands", "nine swords on wall", "nightmare imagery"],
    archetypes: ["anxiety", "nightmare", "mental anguish"],
    themes: ["anxiety", "worry", "nightmare", "mental anguish", "insomnia"],
    keywords: {
      upright: ["anxiety", "worry", "nightmare", "mental anguish"],
      reversed: ["hope", "recovery", "light after darkness"]
    },
    jungian: "anxiety",
    chakra: "third eye",
    seasonality: "darkest hour / 3am",
    timeframe: "middle of the night / crisis",
    advice: "The nightmare is in your mind, not reality. Breathe. Dawn comes. Seek help.",
    shadow_work: "Catastrophizing, rumination, refusing comfort, anxiety spirals",
    light: "Facing fears, seeking help, recognizing thoughts aren't reality",
    questions: [
      "What nightmare keeps you awake?",
      "What anxiety is controlling you?",
      "What would happen if you asked for help?"
    ],
    description: "A figure sits up in bed, head in hands (mental anguish, anxiety, insomnia). Nine swords on wall (accumulated worries). Nightmare imagery on quilt (fears manifest). The Nine of Swords represents anxiety, worry, the dark night of the soul—when the mind becomes the tormentor."
  },

  {
    id: 31,
    name: "Ten of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "10",
    number: 10,
    element: "air",
    modality: "cardinal",
    astrology: "sun in gemini",
    numerology: 10,
    kabbalah: "malkuth of air",
    symbols: ["figure with ten swords in back", "black sky", "sunrise", "calm water"],
    archetypes: ["rock bottom", "endings", "betrayal"],
    themes: ["rock bottom", "painful ending", "betrayal", "defeat", "hitting bottom"],
    keywords: {
      upright: ["rock bottom", "painful ending", "betrayal", "defeat"],
      reversed: ["recovery", "regeneration", "resisting inevitable end"]
    },
    jungian: "death and rebirth",
    chakra: "root",
    seasonality: "darkest before dawn",
    timeframe: "the end / rock bottom",
    advice: "It cannot get worse. You've hit bottom. Now the only way is up. Dawn comes.",
    shadow_work: "Victimhood, dramatizing pain, refusing to rise, playing dead",
    light: "Necessary endings, clearing for rebirth, surrendering to what is",
    questions: [
      "What has reached its absolute end?",
      "Where have you hit rock bottom?",
      "What new beginning awaits after this ending?"
    ],
    description: "A figure lies face down with ten swords in back (complete defeat, rock bottom, betrayal). Black sky but sunrise on horizon (darkest before dawn). Calm water (peace after storm). The Ten of Swords represents painful endings, rock bottom—but also the promise that it cannot get worse, and dawn is coming."
  },

  {
    id: 32,
    name: "Page of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "page",
    number: 11,
    element: "air",
    modality: "mutable",
    astrology: "earth of air",
    numerology: 11,
    kabbalah: "princess of air",
    symbols: ["young figure with sword", "alert stance", "windswept trees", "birds"],
    archetypes: ["messenger", "student", "vigilance"],
    themes: ["new ideas", "vigilance", "curiosity", "mental energy", "communication"],
    keywords: {
      upright: ["new ideas", "vigilance", "curiosity", "mental energy"],
      reversed: ["all talk no action", "haste", "defensive"]
    },
    jungian: "eternal youth",
    chakra: "throat",
    seasonality: "spring / new thought",
    timeframe: "beginning / youthful",
    advice: "Stay alert. Ask questions. Guard your mind. Mental vigilance is strength.",
    shadow_work: "Gossip, hastiness, intellectual arrogance, all talk no action",
    light: "Curious mind, vigilant awareness, truth-seeking",
    questions: [
      "What new idea is emerging?",
      "Where do you need to be more vigilant?",
      "What truth are you seeking?"
    ],
    description: "A young figure holds sword upright in alert stance (mental vigilance, readiness). Windswept trees and birds (mental activity, communication). The Page of Swords represents curiosity, new ideas, vigilant mind—the mental student ready to learn and defend truth."
  },

  {
    id: 33,
    name: "Knight of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "knight",
    number: 12,
    element: "air",
    modality: "fixed",
    astrology: "fire of air",
    numerology: 12,
    kabbalah: "prince of air",
    symbols: ["charging knight", "sword raised", "white horse", "turbulent sky", "birds"],
    archetypes: ["warrior", "action", "haste"],
    themes: ["action", "haste", "direct approach", "assertiveness", "charging forward"],
    keywords: {
      upright: ["action", "haste", "direct approach", "assertiveness"],
      reversed: ["recklessness", "impulsiveness", "lack of direction"]
    },
    jungian: "warrior",
    chakra: "throat",
    seasonality: "storm / action",
    timeframe: "fast / immediate",
    advice: "Act decisively. But temper speed with wisdom. Haste makes waste.",
    shadow_work: "Recklessness, impulsiveness, aggression, bulldozing",
    light: "Decisive action, mental clarity in motion, truth in action",
    questions: [
      "Where do you need to act now?",
      "What requires decisive action?",
      "Are you being reckless or righteous?"
    ],
    description: "A knight charges forward on white horse, sword raised (decisive action, mental aggression). Turbulent sky and birds (air element in motion). The Knight of Swords represents action, assertiveness, charging forward—mental energy in rapid motion, for better or worse."
  },

  {
    id: 34,
    name: "Queen of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "queen",
    number: 13,
    element: "air",
    modality: "cardinal",
    astrology: "water of air",
    numerology: 13,
    kabbalah: "queen of air",
    symbols: ["seated queen", "upright sword", "raised hand", "clouds", "butterflies"],
    archetypes: ["wise woman", "clarity", "boundaries"],
    themes: ["clarity", "independence", "direct communication", "boundaries", "clear thinking"],
    keywords: {
      upright: ["clarity", "independence", "direct communication", "boundaries"],
      reversed: ["cold", "bitter", "harsh", "cutting"]
    },
    jungian: "wise woman",
    chakra: "throat",
    seasonality: "autumn / wisdom",
    timeframe: "experienced / mature",
    advice: "Speak truth clearly. Set boundaries firmly. Independence is not loneliness.",
    shadow_work: "Coldness, bitterness, using truth as weapon, isolation",
    light: "Clear boundaries, truthful communication, wisdom through pain",
    questions: [
      "What truth must you speak?",
      "What boundary must you set?",
      "Where do you need clarity over emotion?"
    ],
    description: "A queen sits alone holding upright sword and raising hand (commanding presence, clear boundaries). Clouds and butterflies (air element, transformation through thought). The Queen of Swords represents clarity, independence, direct communication—wisdom earned through pain, truth without sugar-coating."
  },

  {
    id: 35,
    name: "King of Swords",
    arcana: "minor",
    suit: "swords",
    rank: "king",
    number: 14,
    element: "air",
    modality: "fixed",
    astrology: "air of air",
    numerology: 14,
    kabbalah: "king of air",
    symbols: ["seated king", "upright sword", "butterflies", "clouds", "throne"],
    archetypes: ["judge", "authority", "intellectual power"],
    themes: ["authority", "truth", "judgment", "intellectual power", "legal matters"],
    keywords: {
      upright: ["authority", "truth", "judgment", "intellectual power"],
      reversed: ["tyranny", "manipulation", "abuse of power"]
    },
    jungian: "wise king",
    chakra: "throat",
    seasonality: "winter / judgment",
    timeframe: "authoritative / final",
    advice: "Judge fairly. Lead with truth. Power requires responsibility. Mental mastery is kingship.",
    shadow_work: "Tyranny, cold logic without compassion, manipulation, abuse of power",
    light: "Fair judgment, ethical leadership, truth as authority",
    questions: [
      "What requires your fair judgment?",
      "Where must you exercise authority?",
      "How do you lead with truth?"
    ],
    description: "A king sits enthroned holding upright sword (authority, judgment, intellectual power). Butterflies on throne (air element, transformation). Stern expression (impartial judgment). The King of Swords represents authority, truth, judgment—the mental ruler who leads with logic, ethics, and impartial fairness."
  },

  // ═══════════════════════════════════════════════════════════
  // WANDS - FIRE ELEMENT (36-49)
  // ═══════════════════════════════════════════════════════════

  {
    id: 36,
    name: "Ace of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "ace",
    number: 1,
    element: "fire",
    modality: "cardinal",
    astrology: "fire initiation",
    numerology: 1,
    kabbalah: "kether in fire",
    symbols: ["hand from cloud", "sprouting leaves", "castle", "mountains", "river"],
    archetypes: ["spark", "initiation", "raw power"],
    themes: ["inspiration", "new ventures", "creative spark", "enthusiasm", "potential"],
    keywords: {
      upright: ["inspiration", "creative spark", "new opportunity", "growth"],
      reversed: ["delays", "lack of direction", "creative blocks"]
    },
    jungian: "creative impulse",
    chakra: "solar plexus",
    seasonality: "spring / initiation",
    timeframe: "immediate opportunity",
    advice: "Seize the moment. Act on inspiration. Channel creative fire into form.",
    shadow_work: "Creative blocks, missed opportunities, passion without direction",
    light: "Pure creative potential, divine inspiration, bold initiation",
    questions: [
      "What new venture calls to you?",
      "Where must you take bold action?",
      "What creative spark wants expression?"
    ],
    description: "A hand emerges from clouds grasping a wand sprouting leaves (divine creative power, raw potential). Castle in distance (manifestation goals). The Ace of Wands represents inspiration, creative spark, bold new beginnings—pure fire energy ready to be directed."
  },

  {
    id: 37,
    name: "Two of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "2",
    number: 2,
    element: "fire",
    modality: "cardinal",
    astrology: "mars in aries",
    numerology: 2,
    kabbalah: "chokmah in fire",
    symbols: ["man with globe", "two wands", "castle", "sea"],
    archetypes: ["planner", "future vision", "decision"],
    themes: ["planning", "future vision", "personal power", "bold decisions"],
    keywords: {
      upright: ["planning", "making decisions", "leaving comfort", "taking risks"],
      reversed: ["fear of change", "playing it safe", "bad planning"]
    },
    jungian: "individuation journey",
    chakra: "solar plexus",
    seasonality: "early spring / planning",
    timeframe: "weeks to months",
    advice: "Make the plan. Take the risk. The world awaits your boldness.",
    shadow_work: "Fear of leaving comfort zone, paralysis by analysis, playing small",
    light: "Visionary planning, calculated risk-taking, personal power",
    questions: [
      "What bold plan must you execute?",
      "What comfort zone must you leave?",
      "Where do you need to think bigger?"
    ],
    description: "A figure holds globe in hand while grasping wand (mastery of personal world, planning power). Looking out to sea (future possibilities). The Two of Wands represents planning, vision, decision-making—standing at the threshold of expansion."
  },

  {
    id: 38,
    name: "Three of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "3",
    number: 3,
    element: "fire",
    modality: "mutable",
    astrology: "sun in aries",
    numerology: 3,
    kabbalah: "binah in fire",
    symbols: ["figure watching ships", "three wands", "mountains", "sea"],
    archetypes: ["explorer", "visionary", "enterprise"],
    themes: ["expansion", "foresight", "overseas opportunities", "progress"],
    keywords: {
      upright: ["expansion", "foresight", "overseas opportunities", "progress"],
      reversed: ["obstacles", "delays", "lack of foresight"]
    },
    jungian: "expansion phase",
    chakra: "solar plexus",
    seasonality: "late spring / growth",
    timeframe: "3-6 months",
    advice: "Expand your reach. Think globally. Your ships are coming in.",
    shadow_work: "Impatience, overextension, lack of follow-through",
    light: "Strategic expansion, manifesting vision, global thinking",
    questions: [
      "Where must you expand your reach?",
      "What distant opportunities call?",
      "How can you think more globally?"
    ],
    description: "Figure stands watching ships sail (ventures launched, waiting for results). Three wands planted (established foundation). The Three of Wands represents expansion, foresight, watching plans unfold—the exciting phase after initial action."
  },

  {
    id: 39,
    name: "Four of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "4",
    number: 4,
    element: "fire",
    modality: "fixed",
    astrology: "venus in aries",
    numerology: 4,
    kabbalah: "chesed in fire",
    symbols: ["four wands with garland", "celebrating figures", "castle", "bouquets"],
    archetypes: ["celebration", "homecoming", "harvest"],
    themes: ["celebration", "harmony", "homecoming", "milestone reached"],
    keywords: {
      upright: ["celebration", "harmony", "homecoming", "milestone"],
      reversed: ["lack of support", "instability", "broken relationships"]
    },
    jungian: "integration",
    chakra: "heart",
    seasonality: "summer / celebration",
    timeframe: "milestone achieved",
    advice: "Celebrate wins. Honor milestones. Community supports your fire.",
    shadow_work: "Inability to celebrate, isolation, perfectionism blocking joy",
    light: "Joyful celebration, community support, stability achieved",
    questions: [
      "What milestone deserves celebration?",
      "Where do you need community support?",
      "How can you honor your progress?"
    ],
    description: "Four wands adorned with garland, figures celebrating (achievement, harmony, joy). Castle in background (home base established). The Four of Wands represents celebration, homecoming, stability—the harvest festival after hard work."
  },

  {
    id: 40,
    name: "Five of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "5",
    number: 5,
    element: "fire",
    modality: "mutable",
    astrology: "saturn in leo",
    numerology: 5,
    kabbalah: "geburah in fire",
    symbols: ["five figures with wands clashing", "struggle", "competition"],
    archetypes: ["competition", "conflict", "struggle"],
    themes: ["conflict", "competition", "disagreement", "tension"],
    keywords: {
      upright: ["conflict", "competition", "disagreements", "tension"],
      reversed: ["avoiding conflict", "inner conflict", "end of struggle"]
    },
    jungian: "shadow confrontation",
    chakra: "solar plexus",
    seasonality: "conflict phase",
    timeframe: "short-term struggle",
    advice: "Engage the conflict. Competition sharpens you. May the best idea win.",
    shadow_work: "Avoidance of necessary conflict, ego battles, wasted energy",
    light: "Healthy competition, creative friction, growth through challenge",
    questions: [
      "What conflict must you engage?",
      "Where is competition healthy?",
      "What struggle sharpens your skills?"
    ],
    description: "Five figures with wands appearing to clash (competition, conflict, struggle). May be playful or serious (context matters). The Five of Wands represents conflict, competition, tension—the struggle that either wastes energy or catalyzes growth."
  },

  {
    id: 41,
    name: "Six of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "6",
    number: 6,
    element: "fire",
    modality: "mutable",
    astrology: "jupiter in leo",
    numerology: 6,
    kabbalah: "tiphareth in fire",
    symbols: ["victorious rider", "wreath", "followers", "six wands"],
    archetypes: ["victor", "leader", "hero"],
    themes: ["victory", "public recognition", "success", "self-confidence"],
    keywords: {
      upright: ["victory", "public recognition", "progress", "self-confidence"],
      reversed: ["arrogance", "lack of recognition", "fall from grace"]
    },
    jungian: "hero's return",
    chakra: "solar plexus",
    seasonality: "peak success",
    timeframe: "recent victory",
    advice: "Own your victory. Accept recognition. Leadership is service.",
    shadow_work: "Arrogance, need for external validation, ego inflation",
    light: "Confident leadership, earned recognition, inspiring others",
    questions: [
      "What victory must you own?",
      "Where do you deserve recognition?",
      "How can you lead with humility?"
    ],
    description: "Victorious rider with wreath-crowned wand, followers celebrating (public recognition, success). The Six of Wands represents victory, public success, leadership—the moment of triumph and recognition for efforts."
  },

  {
    id: 42,
    name: "Seven of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "7",
    number: 7,
    element: "fire",
    modality: "cardinal",
    astrology: "mars in leo",
    numerology: 7,
    kabbalah: "netzach in fire",
    symbols: ["figure defending high ground", "six wands attacking", "mismatched shoes"],
    archetypes: ["defender", "warrior", "challenger"],
    themes: ["defense", "maintaining position", "challenge", "perseverance"],
    keywords: {
      upright: ["defense", "maintaining position", "challenged", "perseverance"],
      reversed: ["giving up", "overwhelmed", "losing ground"]
    },
    jungian: "defending ego boundaries",
    chakra: "solar plexus",
    seasonality: "under siege",
    timeframe: "current challenge",
    advice: "Hold your ground. Defend what matters. Courage under fire defines character.",
    shadow_work: "Defensiveness, siege mentality, unnecessary battles",
    light: "Righteous defense, perseverance, courage under pressure",
    questions: [
      "What ground must you hold?",
      "Where are you under attack?",
      "What requires your fierce defense?"
    ],
    description: "Figure on high ground defending against six wands (holding position under challenge). Mismatched shoes (caught off-guard but fighting anyway). The Seven of Wands represents defense, perseverance, holding ground—courage when challenged."
  },

  {
    id: 43,
    name: "Eight of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "8",
    number: 8,
    element: "fire",
    modality: "mutable",
    astrology: "mercury in sagittarius",
    numerology: 8,
    kabbalah: "hod in fire",
    symbols: ["eight wands flying", "clear sky", "river", "landscape"],
    archetypes: ["messenger", "speed", "action"],
    themes: ["speed", "rapid action", "air travel", "communication"],
    keywords: {
      upright: ["speed", "rapid action", "movement", "progress"],
      reversed: ["delays", "frustration", "resisting change"]
    },
    jungian: "rapid development",
    chakra: "solar plexus",
    seasonality: "rapid change",
    timeframe: "days to weeks",
    advice: "Move fast. Strike while hot. Momentum compounds.",
    shadow_work: "Impatience, lack of grounding, burnout from speed",
    light: "Swift action, perfect timing, momentum mastery",
    questions: [
      "What requires swift action?",
      "Where must you accelerate?",
      "What momentum must you ride?"
    ],
    description: "Eight wands flying through clear sky (rapid movement, messages, travel). No obstacles (clear path forward). The Eight of Wands represents speed, rapid progress, swift communication—things moving fast in your favor."
  },

  {
    id: 44,
    name: "Nine of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "9",
    number: 9,
    element: "fire",
    modality: "fixed",
    astrology: "moon in sagittarius",
    numerology: 9,
    kabbalah: "yesod in fire",
    symbols: ["wounded figure", "eight wands behind", "one wand in hand", "bandage"],
    archetypes: ["survivor", "wounded warrior", "resilience"],
    themes: ["resilience", "persistence", "last stand", "wounded but standing"],
    keywords: {
      upright: ["resilience", "persistence", "test of faith", "boundaries"],
      reversed: ["paranoia", "stubbornness", "refusing help"]
    },
    jungian: "wounded healer",
    chakra: "root",
    seasonality: "final test",
    timeframe: "near completion",
    advice: "One more push. You've survived worse. Finish strong.",
    shadow_work: "Paranoia, refusal to heal, martyr complex, isolation",
    light: "Resilience through adversity, wisdom from wounds, final courage",
    questions: [
      "What final test faces you?",
      "Where must you persist despite wounds?",
      "What wisdom comes from your scars?"
    ],
    description: "Wounded but standing figure, bandage on head, guarding eight wands (battle-scarred but unbroken). The Nine of Wands represents resilience, persistence, wounded warrior—you've been through hell but you're still standing."
  },

  {
    id: 45,
    name: "Ten of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "10",
    number: 10,
    element: "fire",
    modality: "cardinal",
    astrology: "saturn in sagittarius",
    numerology: 10,
    kabbalah: "malkuth in fire",
    symbols: ["figure carrying ten wands", "burden", "town ahead", "bent posture"],
    archetypes: ["burden bearer", "martyr", "over-commitment"],
    themes: ["burden", "responsibility", "hard work", "nearing completion"],
    keywords: {
      upright: ["burden", "responsibility", "hard work", "accomplishment near"],
      reversed: ["release", "delegation", "burnout avoided"]
    },
    jungian: "inflation / carrying too much",
    chakra: "root",
    seasonality: "harvest burden",
    timeframe: "near end of cycle",
    advice: "Delegate. Release unnecessary burdens. Success shouldn't break your back.",
    shadow_work: "Martyr complex, inability to delegate, self-imposed suffering",
    light: "Responsible completion, strength to finish, wisdom to delegate",
    questions: [
      "What burden must you release?",
      "Where can you delegate?",
      "What's the difference between responsibility and martyrdom?"
    ],
    description: "Figure bent under burden of ten wands, town ahead (so close to finish but heavy burden). The Ten of Wands represents burden, over-responsibility, hard work—carrying too much alone when delegation would help."
  },

  {
    id: 46,
    name: "Page of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "page",
    number: 11,
    element: "fire",
    modality: "mutable",
    astrology: "earth of fire",
    numerology: 11,
    kabbalah: "princess of fire",
    symbols: ["young figure with wand", "desert", "pyramids", "sprouting wand"],
    archetypes: ["explorer", "student", "messenger"],
    themes: ["exploration", "enthusiasm", "discovery", "free spirit"],
    keywords: {
      upright: ["exploration", "enthusiasm", "discovery", "free spirit"],
      reversed: ["lack of direction", "procrastination", "laziness"]
    },
    jungian: "puer / puella",
    chakra: "sacral",
    seasonality: "youthful exploration",
    timeframe: "beginning phase",
    advice: "Explore freely. Learn by doing. Enthusiasm is its own reward.",
    shadow_work: "Lack of follow-through, scattered energy, perpetual beginner",
    light: "Enthusiastic exploration, courageous learning, authentic curiosity",
    questions: [
      "What new passion calls you?",
      "Where must you explore?",
      "What message of enthusiasm do you carry?"
    ],
    description: "Young figure holding wand observing sprouting leaves (curious exploration, messages of passion). Desert setting (uncharted territory). The Page of Wands represents enthusiasm, exploration, discovery—the youthful fire of new interests."
  },

  {
    id: 47,
    name: "Knight of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "knight",
    number: 12,
    element: "fire",
    modality: "cardinal",
    astrology: "air of fire",
    numerology: 12,
    kabbalah: "prince of fire",
    symbols: ["knight on rearing horse", "armor with salamanders", "desert", "pyramids"],
    archetypes: ["adventurer", "warrior", "hero"],
    themes: ["action", "adventure", "fearlessness", "passion"],
    keywords: {
      upright: ["action", "adventure", "fearlessness", "passion"],
      reversed: ["recklessness", "haste", "scattered energy"]
    },
    jungian: "warrior archetype",
    chakra: "solar plexus",
    seasonality: "action phase",
    timeframe: "immediate / sudden",
    advice: "Act boldly. Move fast. Fortune favors the brave.",
    shadow_work: "Recklessness, impulsiveness, lack of planning, burnout",
    light: "Courageous action, passionate pursuit, bold adventure",
    questions: [
      "Where must you act boldly?",
      "What adventure calls you?",
      "When is speed more important than planning?"
    ],
    description: "Knight on rearing horse, armor decorated with salamanders (fire element symbol, passionate action). The Knight of Wands represents action, adventure, fearlessness—charging forward with passion and courage."
  },

  {
    id: 48,
    name: "Queen of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "queen",
    number: 13,
    element: "fire",
    modality: "fixed",
    astrology: "water of fire",
    numerology: 13,
    kabbalah: "queen of fire",
    symbols: ["queen on throne", "black cat", "sunflowers", "lions"],
    archetypes: ["sovereign", "empress", "magnetic presence"],
    themes: ["confidence", "independence", "determination", "vivaciousness"],
    keywords: {
      upright: ["confidence", "independence", "determination", "vivaciousness"],
      reversed: ["jealousy", "selfishness", "demanding"]
    },
    jungian: "empress / sovereignty",
    chakra: "sacral",
    seasonality: "full bloom",
    timeframe: "mature / established",
    advice: "Own your power. Shine without apology. Your confidence inspires others.",
    shadow_work: "Jealousy, need for attention, dominating energy, insecurity masked as confidence",
    light: "Authentic confidence, warm leadership, magnetic presence",
    questions: [
      "Where must you own your power?",
      "How can you inspire through confidence?",
      "What makes you uniquely magnetic?"
    ],
    description: "Queen enthroned with sunflowers, black cat, lions (confidence, warmth, sovereignty). Wand in hand (command of creative power). The Queen of Wands represents confidence, independence, charismatic leadership—the sovereign who rules with warmth and strength."
  },

  {
    id: 49,
    name: "King of Wands",
    arcana: "minor",
    suit: "wands",
    rank: "king",
    number: 14,
    element: "fire",
    modality: "cardinal",
    astrology: "fire of fire",
    numerology: 14,
    kabbalah: "king of fire",
    symbols: ["king on throne", "lions", "salamanders", "crown"],
    archetypes: ["leader", "visionary", "entrepreneur"],
    themes: ["leadership", "vision", "entrepreneurship", "honor"],
    keywords: {
      upright: ["leadership", "vision", "entrepreneurship", "honor"],
      reversed: ["tyranny", "arrogance", "impulsiveness"]
    },
    jungian: "king archetype",
    chakra: "solar plexus",
    seasonality: "full mastery",
    timeframe: "leadership established",
    advice: "Lead boldly. Vision without action is fantasy. Be the king of your domain.",
    shadow_work: "Tyranny, ego inflation, reckless leadership, domination",
    light: "Visionary leadership, honorable command, inspiring action",
    questions: [
      "Where must you lead boldly?",
      "What vision requires your leadership?",
      "How do you wield power with honor?"
    ],
    description: "King enthroned with lions, salamanders (fire mastery, bold leadership). Commanding presence, wand of power. The King of Wands represents leadership, vision, entrepreneurship—the sovereign who leads with bold vision and honorable action."
  },

  // ═══════════════════════════════════════════════════════════
  // CUPS - WATER ELEMENT (50-63)
  // ═══════════════════════════════════════════════════════════

  {
    id: 50,
    name: "Ace of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "ace",
    number: 1,
    element: "water",
    modality: "cardinal",
    astrology: "water initiation",
    numerology: 1,
    kabbalah: "kether in water",
    symbols: ["hand from cloud", "overflowing cup", "dove", "lotus", "five streams"],
    archetypes: ["divine love", "spiritual opening", "compassion"],
    themes: ["new love", "emotional awakening", "compassion", "intuition", "creativity"],
    keywords: {
      upright: ["new love", "emotional awakening", "intuition", "spiritual connection"],
      reversed: ["emotional loss", "blocked creativity", "emptiness"]
    },
    jungian: "anima / animus awakening",
    chakra: "heart",
    seasonality: "spring / emotional renewal",
    timeframe: "new emotional beginning",
    advice: "Open your heart. Receive love. Let emotions flow freely.",
    shadow_work: "Closed heart, fear of vulnerability, emotional unavailability",
    light: "Divine love, open heart, emotional receptivity",
    questions: [
      "What new emotional experience calls you?",
      "Where must you open your heart?",
      "What prevents you from receiving love?"
    ],
    description: "Hand from clouds offering overflowing cup, dove descending, lotus blooms (divine love, spiritual awakening). Five streams (five senses awakened to love). The Ace of Cups represents new love, emotional opening, divine connection—the heart's first stirring."
  },

  {
    id: 51,
    name: "Two of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "2",
    number: 2,
    element: "water",
    modality: "cardinal",
    astrology: "venus in cancer",
    numerology: 2,
    kabbalah: "chokmah in water",
    symbols: ["two figures", "caduceus", "lion head", "mutual gaze"],
    archetypes: ["union", "partnership", "soul connection"],
    themes: ["partnership", "mutual respect", "attraction", "unity"],
    keywords: {
      upright: ["partnership", "mutual attraction", "unity", "connection"],
      reversed: ["disharmony", "imbalance", "broken relationship"]
    },
    jungian: "syzygy / sacred marriage",
    chakra: "heart",
    seasonality: "union / partnership",
    timeframe: "forming connection",
    advice: "Honor partnership. Mutual respect creates magic. Two become one.",
    shadow_work: "Codependency, loss of self, idealization, projection",
    light: "Sacred partnership, mutual honoring, balanced union",
    questions: [
      "What partnership must you honor?",
      "Where do you project instead of see?",
      "How can two remain two while becoming one?"
    ],
    description: "Two figures facing each other, cups raised (partnership, mutual respect, attraction). Caduceus above (healing through union). The Two of Cups represents partnership, attraction, sacred union—when two hearts meet as equals."
  },

  {
    id: 52,
    name: "Three of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "3",
    number: 3,
    element: "water",
    modality: "mutable",
    astrology: "mercury in cancer",
    numerology: 3,
    kabbalah: "binah in water",
    symbols: ["three dancing figures", "raised cups", "harvest", "celebration"],
    archetypes: ["celebration", "community", "friendship"],
    themes: ["celebration", "friendship", "community", "creativity"],
    keywords: {
      upright: ["celebration", "friendship", "creativity", "community"],
      reversed: ["overindulgence", "gossip", "isolation"]
    },
    jungian: "collective feminine",
    chakra: "sacral",
    seasonality: "harvest / celebration",
    timeframe: "social gathering",
    advice: "Celebrate with others. Community amplifies joy. Friendship is medicine.",
    shadow_work: "Codependency on social validation, gossip, superficial connections",
    light: "Authentic community, joyful celebration, creative collaboration",
    questions: [
      "What deserves celebration?",
      "Where do you need community?",
      "How can you co-create joy?"
    ],
    description: "Three figures dancing, cups raised in celebration (community, friendship, harvest joy). The Three of Cups represents celebration, friendship, creative collaboration—the joy of connection and shared success."
  },

  {
    id: 53,
    name: "Four of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "4",
    number: 4,
    element: "water",
    modality: "fixed",
    astrology: "moon in cancer",
    numerology: 4,
    kabbalah: "chesed in water",
    symbols: ["seated figure", "three cups", "hand offering fourth", "tree"],
    archetypes: ["contemplation", "apathy", "meditation"],
    themes: ["contemplation", "apathy", "reevaluation", "meditation"],
    keywords: {
      upright: ["contemplation", "apathy", "reevaluation", "meditation"],
      reversed: ["awareness", "acceptance", "new perspectives"]
    },
    jungian: "withdrawal / introv

ersion",
    chakra: "third eye",
    seasonality: "introspective phase",
    timeframe: "pause / reflection",
    advice: "Pause to reflect. What you seek may be right before you. Discontent is a teacher.",
    shadow_work: "Apathy, taking blessings for granted, grass-is-greener syndrome",
    light: "Deep contemplation, wise discernment, spiritual introspection",
    questions: [
      "What are you not seeing?",
      "Where is discontent a teacher?",
      "What new offering do you ignore?"
    ],
    description: "Figure sits cross-armed, three cups before them, hand from cloud offering fourth (apathy, contemplation, missing opportunities). The Four of Cups represents contemplation, apathy, reevaluation—sometimes missing the gift right in front of you."
  },

  {
    id: 54,
    name: "Five of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "5",
    number: 5,
    element: "water",
    modality: "mutable",
    astrology: "mars in scorpio",
    numerology: 5,
    kabbalah: "geburah in water",
    symbols: ["cloaked figure", "three spilled cups", "two standing cups", "bridge"],
    archetypes: ["grief", "loss", "regret"],
    themes: ["loss", "grief", "regret", "disappointment"],
    keywords: {
      upright: ["loss", "grief", "regret", "disappointment"],
      reversed: ["acceptance", "moving on", "finding peace"]
    },
    jungian: "shadow work / grief",
    chakra: "heart",
    seasonality: "winter / loss",
    timeframe: "recent loss",
    advice: "Grieve fully. Honor loss. But see: two cups still stand, and a bridge awaits.",
    shadow_work: "Dwelling in victimhood, refusing to see remaining blessings, stuck in grief",
    light: "Honest grief, acknowledging loss, finding gifts in pain",
    questions: [
      "What must you grieve?",
      "What two cups still stand?",
      "When does grief become dwelling?"
    ],
    description: "Cloaked figure mourning three spilled cups, two cups remain standing, bridge in distance (loss, grief, but hope remains). The Five of Cups represents loss, regret, grief—but also the choice to see what remains and move forward."
  },

  {
    id: 55,
    name: "Six of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "6",
    number: 6,
    element: "water",
    modality: "mutable",
    astrology: "sun in scorpio",
    numerology: 6,
    kabbalah: "tiphareth in water",
    symbols: ["children", "flowers in cups", "home", "nostalgia"],
    archetypes: ["innocence", "nostalgia", "childhood"],
    themes: ["nostalgia", "childhood memories", "innocence", "reunion"],
    keywords: {
      upright: ["nostalgia", "childhood memories", "innocence", "reunion"],
      reversed: ["stuck in past", "rose-colored glasses", "childhood wounds"]
    },
    jungian: "inner child",
    chakra: "heart",
    seasonality: "returning home",
    timeframe: "past revisited",
    advice: "Honor your past. Your inner child needs attention. Innocence is not lost, only forgotten.",
    shadow_work: "Living in past, idealizing childhood, refusing present, unhealed wounds",
    light: "Healed inner child, genuine innocence, wisdom from past",
    questions: [
      "What does your inner child need?",
      "Where are you stuck in the past?",
      "How does nostalgia serve or hinder you?"
    ],
    description: "Children exchanging flowers in cups (nostalgia, innocence, childhood memories). Home in background (returning to roots). The Six of Cups represents nostalgia, reunion, inner child work—revisiting the past to heal and integrate."
  },

  {
    id: 56,
    name: "Seven of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "7",
    number: 7,
    element: "water",
    modality: "cardinal",
    astrology: "venus in scorpio",
    numerology: 7,
    kabbalah: "netzach in water",
    symbols: ["silhouette", "seven cups with visions", "clouds", "choices"],
    archetypes: ["illusion", "fantasy", "wishful thinking"],
    themes: ["choices", "illusion", "fantasy", "wishful thinking"],
    keywords: {
      upright: ["choices", "fantasy", "illusion", "opportunities"],
      reversed: ["clarity", "reality check", "determination"]
    },
    jungian: "inflation / illusion",
    chakra: "third eye",
    seasonality: "confusion phase",
    timeframe: "decision paralysis",
    advice: "Choose wisely. Not all that glitters is gold. Fantasy must meet reality.",
    shadow_work: "Escapism, analysis paralysis, avoiding choice, living in fantasy",
    light: "Visionary dreaming, wise discernment, grounding dreams in reality",
    questions: [
      "Which choice is real vs. fantasy?",
      "Where are you avoiding decision?",
      "What dream must you ground in reality?"
    ],
    description: "Figure faces seven cups floating in clouds, each containing a vision (choices, illusions, fantasies). The Seven of Cups represents choices, fantasy, illusion—the challenge of discerning real opportunities from mirages."
  },

  {
    id: 57,
    name: "Eight of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "8",
    number: 8,
    element: "water",
    modality: "mutable",
    astrology: "saturn in pisces",
    numerology: 8,
    kabbalah: "hod in water",
    symbols: ["figure walking away", "eight cups", "moon", "mountains"],
    archetypes: ["journey", "abandonment", "search"],
    themes: ["walking away", "spiritual journey", "seeking deeper meaning"],
    keywords: {
      upright: ["walking away", "seeking truth", "spiritual journey", "abandonment"],
      reversed: ["fear of change", "stuck", "avoiding loss"]
    },
    jungian: "individuation journey",
    chakra: "crown",
    seasonality: "pilgrim phase",
    timeframe: "transition / departure",
    advice: "Walk away from what no longer serves. The spiritual journey requires leaving comfort.",
    shadow_work: "Fear of loss, attachment to familiar suffering, avoiding growth",
    light: "Courageous departure, spiritual seeking, choosing growth over comfort",
    questions: [
      "What must you walk away from?",
      "What deeper truth calls you?",
      "When is leaving an act of self-love?"
    ],
    description: "Figure walks away from eight cups toward mountains under moon (abandonment, spiritual quest, seeking deeper meaning). The Eight of Cups represents walking away, spiritual journey, seeking truth—leaving what's comfortable for what's meaningful."
  },

  {
    id: 58,
    name: "Nine of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "9",
    number: 9,
    element: "water",
    modality: "fixed",
    astrology: "jupiter in pisces",
    numerology: 9,
    kabbalah: "yesod in water",
    symbols: ["satisfied figure", "nine cups arranged", "smile", "arms crossed"],
    archetypes: ["contentment", "wish fulfillment", "satisfaction"],
    themes: ["contentment", "satisfaction", "wishes fulfilled", "happiness"],
    keywords: {
      upright: ["contentment", "satisfaction", "wish fulfillment", "happiness"],
      reversed: ["greed", "dissatisfaction", "materialism"]
    },
    jungian: "self-satisfaction",
    chakra: "solar plexus",
    seasonality: "abundance peak",
    timeframe: "wish granted",
    advice: "Savor satisfaction. You earned this. Gratitude amplifies abundance.",
    shadow_work: "Greed, never enough, smugness, attachment to comfort",
    light: "Genuine contentment, gratitude, celebrating success",
    questions: [
      "What wish has been fulfilled?",
      "Where can you savor success?",
      "How does contentment differ from complacency?"
    ],
    description: "Figure sits contentedly, nine cups arranged (wish fulfillment, satisfaction, emotional abundance). The Nine of Cups represents contentment, wishes fulfilled, satisfaction—the 'wish card' where dreams manifest."
  },

  {
    id: 59,
    name: "Ten of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "10",
    number: 10,
    element: "water",
    modality: "cardinal",
    astrology: "mars in pisces",
    numerology: 10,
    kabbalah: "malkuth in water",
    symbols: ["family", "rainbow", "home", "children", "ten cups"],
    archetypes: ["harmony", "family", "completion"],
    themes: ["harmony", "family", "emotional fulfillment", "lasting happiness"],
    keywords: {
      upright: ["harmony", "family", "happiness", "alignment"],
      reversed: ["broken family", "disconnection", "misalignment"]
    },
    jungian: "wholeness / family archetype",
    chakra: "heart",
    seasonality: "completion / harvest",
    timeframe: "lasting fulfillment",
    advice: "This is it—the happy ending. Family (chosen or blood) is everything. Cherish harmony.",
    shadow_work: "Idealization of family, ignoring dysfunction, superficial harmony",
    light: "Genuine harmony, loving family, emotional completion",
    questions: [
      "What does 'home' mean to you?",
      "Where have you found your tribe?",
      "How do you maintain harmony without sacrificing truth?"
    ],
    description: "Family embracing, rainbow and ten cups above (emotional fulfillment, harmony, lasting happiness). The Ten of Cups represents family harmony, emotional completion, lasting joy—the happily ever after of emotional fulfillment."
  },

  {
    id: 60,
    name: "Page of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "page",
    number: 11,
    element: "water",
    modality: "mutable",
    astrology: "earth of water",
    numerology: 11,
    kabbalah: "princess of water",
    symbols: ["young figure with cup", "fish emerging", "ocean", "dreamy expression"],
    archetypes: ["creative", "intuitive", "messenger"],
    themes: ["creativity", "intuition", "good news", "new feelings"],
    keywords: {
      upright: ["creativity", "intuition", "good news", "inner child"],
      reversed: ["emotional immaturity", "creative blocks", "bad news"]
    },
    jungian: "inner child / intuition",
    chakra: "sacral",
    seasonality: "emotional awakening",
    timeframe: "new emotional message",
    advice: "Trust intuition. Emotions bring messages. Play like a child, create like an artist.",
    shadow_work: "Emotional immaturity, ignoring intuition, creative blocks",
    light: "Intuitive wisdom, creative play, emotional honesty",
    questions: [
      "What does your intuition whisper?",
      "What wants to be created?",
      "How does your inner child want to play?"
    ],
    description: "Young figure holding cup, fish emerging (intuition, creativity, emotional messages). The Page of Cups represents intuition, creativity, good news—the youthful, dreamy messenger of the heart."
  },

  {
    id: 61,
    name: "Knight of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "knight",
    number: 12,
    element: "water",
    modality: "cardinal",
    astrology: "air of water",
    numerology: 12,
    kabbalah: "prince of water",
    symbols: ["knight on white horse", "cup", "winged helmet", "flowing water"],
    archetypes: ["romantic", "idealist", "artist"],
    themes: ["romance", "charm", "idealism", "proposals"],
    keywords: {
      upright: ["romance", "charm", "idealism", "following heart"],
      reversed: ["moodiness", "unrealistic expectations", "disappointment"]
    },
    jungian: "animus / romantic ideal",
    chakra: "heart",
    seasonality: "courtship phase",
    timeframe: "romantic gesture",
    advice: "Follow your heart. Romance is sacred. But ground idealism in reality.",
    shadow_work: "Unrealistic expectations, love addiction, avoiding commitment, fantasy over reality",
    light: "Authentic romance, creative expression, emotional courage",
    questions: [
      "What romantic gesture calls you?",
      "Where are expectations unrealistic?",
      "How do you balance idealism with reality?"
    ],
    description: "Knight on white horse offering cup (romance, proposals, idealism). Winged helmet (elevated emotions). The Knight of Cups represents romance, charm, following the heart—the poet, artist, and romantic idealist."
  },

  {
    id: 62,
    name: "Queen of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "queen",
    number: 13,
    element: "water",
    modality: "fixed",
    astrology: "water of water",
    numerology: 13,
    kabbalah: "queen of water",
    symbols: ["queen on throne", "ornate cup", "water", "angels", "shells"],
    archetypes: ["psychic", "empath", "healer"],
    themes: ["empathy", "intuition", "emotional depth", "compassion"],
    keywords: {
      upright: ["empathy", "intuition", "compassion", "emotional security"],
      reversed: ["emotional manipulation", "codependency", "martyrdom"]
    },
    jungian: "great mother / healer",
    chakra: "heart",
    seasonality: "emotional mastery",
    timeframe: "mature emotional wisdom",
    advice: "Trust your empathy. Emotional wisdom is power. But maintain boundaries.",
    shadow_work: "Emotional manipulation, martyrdom, porous boundaries, vicarious living",
    light: "Psychic sensitivity, compassionate healing, emotional mastery",
    questions: [
      "Where must you trust your intuition?",
      "How do you honor emotions without drowning?",
      "What's the difference between empathy and enmeshment?"
    ],
    description: "Queen enthroned by water, ornate cup, angels and shells (empathy, intuition, emotional depth). The Queen of Cups represents empathy, psychic ability, compassionate wisdom—the emotional healer and intuitive guide."
  },

  {
    id: 63,
    name: "King of Cups",
    arcana: "minor",
    suit: "cups",
    rank: "king",
    number: 14,
    element: "water",
    modality: "cardinal",
    astrology: "fire of water",
    numerology: 14,
    kabbalah: "king of water",
    symbols: ["king on throne", "cup", "turbulent seas", "ship", "fish"],
    archetypes: ["wise counselor", "emotional mastery", "diplomacy"],
    themes: ["emotional control", "diplomacy", "compassion", "wise counsel"],
    keywords: {
      upright: ["emotional control", "diplomacy", "compassion", "calm"],
      reversed: ["emotional manipulation", "coldness", "repression"]
    },
    jungian: "wise king / emotional mastery",
    chakra: "throat",
    seasonality: "emotional sovereignty",
    timeframe: "mature emotional leadership",
    advice: "Master emotions without suppressing them. Lead with compassion. Calm in the storm.",
    shadow_work: "Emotional suppression, manipulation through caring, passive aggression",
    light: "Emotional sovereignty, wise counseling, compassionate leadership",
    questions: [
      "How do you lead with heart?",
      "Where do you confuse control with mastery?",
      "What wisdom comes from emotional balance?"
    ],
    description: "King enthroned amid turbulent seas, calm demeanor (emotional mastery, calm in chaos). Fish jumping (emotional depths surfacing). The King of Cups represents emotional control, diplomacy, wise counsel—the master of the emotional realm."
  },

  // ═══════════════════════════════════════════════════════════
  // PENTACLES - EARTH ELEMENT (64-77)
  // ═══════════════════════════════════════════════════════════

  {
    id: 64,
    name: "Ace of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "ace",
    number: 1,
    element: "earth",
    modality: "cardinal",
    astrology: "earth initiation",
    numerology: 1,
    kabbalah: "kether in earth",
    symbols: ["hand from cloud", "pentacle", "garden", "archway", "mountains"],
    archetypes: ["manifestation", "prosperity", "grounding"],
    themes: ["opportunity", "prosperity", "manifestation", "new financial venture"],
    keywords: {
      upright: ["opportunity", "prosperity", "manifestation", "abundance"],
      reversed: ["missed opportunity", "lack of planning", "greed"]
    },
    jungian: "embodiment / manifestation",
    chakra: "root",
    seasonality: "planting season",
    timeframe: "new material beginning",
    advice: "Plant seeds now. Material opportunity is here. Ground dreams in reality.",
    shadow_work: "Greed, materialism, refusing to take practical action",
    light: "Grounded manifestation, practical abundance, embodied prosperity",
    questions: [
      "What material opportunity presents itself?",
      "Where must you take practical action?",
      "What dream needs grounding?"
    ],
    description: "Hand from clouds offering pentacle, lush garden below (material opportunity, prosperity, manifestation). The Ace of Pentacles represents new financial ventures, practical opportunities, grounded beginnings—spirit made manifest."
  },

  {
    id: 65,
    name: "Two of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "2",
    number: 2,
    element: "earth",
    modality: "cardinal",
    astrology: "jupiter in capricorn",
    numerology: 2,
    kabbalah: "chokmah in earth",
    symbols: ["juggling figure", "two pentacles", "infinity loop", "ships on waves"],
    archetypes: ["balance", "adaptability", "multitasking"],
    themes: ["balance", "adaptability", "time management", "priorities"],
    keywords: {
      upright: ["balance", "adaptability", "time management", "priorities"],
      reversed: ["overwhelm", "disorganization", "loss of balance"]
    },
    jungian: "balancing opposites",
    chakra: "sacral",
    seasonality: "active juggling",
    timeframe: "short-term balancing",
    advice: "Stay flexible. Balance competing priorities. Dance with change.",
    shadow_work: "Overwhelm, scattered focus, inability to prioritize, chaos",
    light: "Graceful balance, adaptive flow, masterful juggling",
    questions: [
      "What needs balancing?",
      "Where are you juggling too much?",
      "How can you prioritize better?"
    ],
    description: "Figure juggling two pentacles in infinity loop, ships on turbulent waves (balance, adaptability, managing multiple priorities). The Two of Pentacles represents balance, time management, juggling responsibilities—finding flow amid chaos."
  },

  {
    id: 66,
    name: "Three of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "3",
    number: 3,
    element: "earth",
    modality: "mutable",
    astrology: "mars in capricorn",
    numerology: 3,
    kabbalah: "binah in earth",
    symbols: ["craftsperson", "architect", "monk", "cathedral", "pentacles"],
    archetypes: ["craftsperson", "teamwork", "mastery"],
    themes: ["teamwork", "collaboration", "learning", "implementation"],
    keywords: {
      upright: ["teamwork", "collaboration", "learning", "implementation"],
      reversed: ["lack of teamwork", "disharmony", "poor quality"]
    },
    jungian: "skill development",
    chakra: "solar plexus",
    seasonality: "building phase",
    timeframe: "learning / apprenticeship",
    advice: "Master your craft. Collaborate. Quality requires skill, teamwork, and time.",
    shadow_work: "Lone wolf syndrome, refusing to learn, sloppy work, ego over excellence",
    light: "Masterful collaboration, dedicated learning, quality craftsmanship",
    questions: [
      "What skill must you master?",
      "Where do you need collaboration?",
      "How can you improve quality?"
    ],
    description: "Craftsperson collaborating with architect and monk in cathedral (teamwork, skill, implementation). The Three of Pentacles represents collaboration, mastery, building—bringing vision into form through skilled teamwork."
  },

  {
    id: 67,
    name: "Four of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "4",
    number: 4,
    element: "earth",
    modality: "fixed",
    astrology: "sun in capricorn",
    numerology: 4,
    kabbalah: "chesed in earth",
    symbols: ["figure clutching pentacles", "crown with pentacle", "city behind"],
    archetypes: ["control", "security", "possessiveness"],
    themes: ["control", "security", "possessiveness", "conservation"],
    keywords: {
      upright: ["control", "stability", "security", "conservatism"],
      reversed: ["greed", "materialism", "self-protection gone too far"]
    },
    jungian: "ego control / security shadow",
    chakra: "root",
    seasonality: "winter / preservation",
    timeframe: "holding phase",
    advice: "Protect what's yours. But know: grasping kills. Security requires flow, not death grip.",
    shadow_work: "Greed, control, fear-based hoarding, inability to receive",
    light: "Healthy boundaries, wise conservation, secure foundation",
    questions: [
      "What are you clutching too tightly?",
      "Where does security become prison?",
      "How do you balance saving with flow?"
    ],
    description: "Figure clutching pentacles tightly, crown with pentacle on head (control, security, possessiveness). The Four of Pentacles represents security, control, conservation—the fine line between protection and imprisonment."
  },

  {
    id: 68,
    name: "Five of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "5",
    number: 5,
    element: "earth",
    modality: "mutable",
    astrology: "mercury in taurus",
    numerology: 5,
    kabbalah: "geburah in earth",
    symbols: ["two beggars", "stained glass window", "snow", "church"],
    archetypes: ["poverty", "isolation", "hardship"],
    themes: ["poverty", "hardship", "isolation", "material loss"],
    keywords: {
      upright: ["poverty", "hardship", "isolation", "material loss"],
      reversed: ["recovery", "spiritual growth from hardship", "help arrives"]
    },
    jungian: "dark night of soul / material crisis",
    chakra: "root",
    seasonality: "winter / hardship",
    timeframe: "difficult period",
    advice: "Ask for help. You're not alone. The church door is open—walk through it.",
    shadow_work: "Victim mentality, refusing help, pride blocking receiving, poverty consciousness",
    light: "Vulnerability, asking for help, spiritual growth through hardship",
    questions: [
      "Where do you need help?",
      "What prevents you from asking?",
      "How does hardship serve your growth?"
    ],
    description: "Two figures in snow outside church with stained glass (hardship, isolation, but help is near). The Five of Pentacles represents material hardship, poverty, isolation—but also the reminder that help is available if you reach for it."
  },

  {
    id: 69,
    name: "Six of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "6",
    number: 6,
    element: "earth",
    modality: "mutable",
    astrology: "moon in taurus",
    numerology: 6,
    kabbalah: "tiphareth in earth",
    symbols: ["wealthy figure", "beggars", "scales", "giving"],
    archetypes: ["generosity", "charity", "sharing"],
    themes: ["generosity", "charity", "sharing wealth", "financial help"],
    keywords: {
      upright: ["generosity", "charity", "sharing", "financial help"],
      reversed: ["strings attached", "power imbalance", "debt"]
    },
    jungian: "healthy exchange",
    chakra: "heart",
    seasonality: "harvest sharing",
    timeframe: "giving / receiving",
    advice: "Give freely. Receive graciously. Wealth flows when shared. But guard against control disguised as generosity.",
    shadow_work: "Using money for control, refusing to receive, guilt around wealth, power dynamics",
    light: "Generous sharing, balanced exchange, wealth consciousness",
    questions: [
      "Where can you give generously?",
      "How do you receive without shame?",
      "Does your giving create dependence or empowerment?"
    ],
    description: "Wealthy figure with scales giving to beggars (generosity, charity, balanced giving). The Six of Pentacles represents generosity, sharing, financial help—but also the power dynamics in giving and receiving."
  },

  {
    id: 70,
    name: "Seven of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "7",
    number: 7,
    element: "earth",
    modality: "cardinal",
    astrology: "saturn in taurus",
    numerology: 7,
    kabbalah: "netzach in earth",
    symbols: ["farmer resting", "pentacles on vine", "contemplation"],
    archetypes: ["patience", "assessment", "long-term vision"],
    themes: ["patience", "assessment", "long-term investment", "evaluation"],
    keywords: {
      upright: ["patience", "long-term view", "assessment", "perseverance"],
      reversed: ["impatience", "lack of rewards", "poor investment"]
    },
    jungian: "evaluation phase",
    chakra: "solar plexus",
    seasonality: "mid-growth assessment",
    timeframe: "long-term investment",
    advice: "Pause. Assess. Are your efforts bearing fruit? Course-correct if needed.",
    shadow_work: "Impatience, giving up too soon, not assessing progress",
    light: "Patient perseverance, wise assessment, long-term thinking",
    questions: [
      "What investment needs more time?",
      "Where should you course-correct?",
      "How do you balance patience with action?"
    ],
    description: "Farmer leaning on hoe, observing pentacles growing on vine (assessment, patience, long-term vision). The Seven of Pentacles represents patience, evaluation, long-term investment—the pause to assess if your efforts are working."
  },

  {
    id: 71,
    name: "Eight of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "8",
    number: 8,
    element: "earth",
    modality: "mutable",
    astrology: "sun in virgo",
    numerology: 8,
    kabbalah: "hod in earth",
    symbols: ["craftsperson at work", "pentacles on wall", "town in distance"],
    archetypes: ["mastery", "diligence", "skill development"],
    themes: ["mastery", "diligence", "skill development", "attention to detail"],
    keywords: {
      upright: ["mastery", "skill development", "diligence", "detail work"],
      reversed: ["perfectionism", "lack of ambition", "poor quality"]
    },
    jungian: "individuation through work",
    chakra: "solar plexus",
    seasonality: "deep work phase",
    timeframe: "mastery development",
    advice: "Master your craft. Ten thousand hours. Quality over quantity. Excellence is the path.",
    shadow_work: "Perfectionism, workaholic, meaningless toil, lack of vision",
    light: "Joyful mastery, dedicated practice, craftsmanship",
    questions: [
      "What skill deserves your dedication?",
      "Where does perfectionism block progress?",
      "How does your work serve something larger?"
    ],
    description: "Craftsperson diligently working, finished pentacles displayed (mastery, skill development, dedication). The Eight of Pentacles represents mastery, diligence, skill development—the path of dedicated practice and excellence."
  },

  {
    id: 72,
    name: "Nine of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "9",
    number: 9,
    element: "earth",
    modality: "fixed",
    astrology: "venus in virgo",
    numerology: 9,
    kabbalah: "yesod in earth",
    symbols: ["wealthy figure", "garden", "falcon", "pentacles", "luxury"],
    archetypes: ["self-sufficiency", "luxury", "independence"],
    themes: ["self-sufficiency", "luxury", "financial independence", "self-discipline"],
    keywords: {
      upright: ["self-sufficiency", "luxury", "financial independence", "self-discipline"],
      reversed: ["over-investment in image", "financial dependence", "lack of discipline"]
    },
    jungian: "individuated self / sovereignty",
    chakra: "solar plexus",
    seasonality: "harvest / abundance",
    timeframe: "established success",
    advice: "Enjoy the fruits of your labor. You built this. Self-sufficiency is freedom.",
    shadow_work: "Isolation from self-sufficiency, materialism, image over substance",
    light: "Earned luxury, genuine independence, disciplined success",
    questions: [
      "What have you built alone?",
      "How do you balance independence with connection?",
      "Where can you enjoy your success?"
    ],
    description: "Wealthy figure in abundant garden with falcon (self-sufficiency, luxury, financial independence). The Nine of Pentacles represents independence, luxury, self-made success—the sovereignty of one who built their own empire."
  },

  {
    id: 73,
    name: "Ten of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "10",
    number: 10,
    element: "earth",
    modality: "cardinal",
    astrology: "mercury in virgo",
    numerology: 10,
    kabbalah: "malkuth in earth",
    symbols: ["family", "elder", "archway", "pentacles", "dogs"],
    archetypes: ["legacy", "inheritance", "completion"],
    themes: ["legacy", "inheritance", "family wealth", "long-term success"],
    keywords: {
      upright: ["legacy", "inheritance", "family wealth", "long-term success"],
      reversed: ["family disputes", "financial failure", "lost inheritance"]
    },
    jungian: "wholeness / completion",
    chakra: "root",
    seasonality: "completion / legacy",
    timeframe: "generational",
    advice: "Build for generations. Legacy matters. What you create outlives you.",
    shadow_work: "Dynasty thinking, family dysfunction, wealth without values",
    light: "Lasting legacy, family prosperity, wealth with wisdom",
    questions: [
      "What legacy do you leave?",
      "How do you balance individual success with family harmony?",
      "What values do you pass down?"
    ],
    description: "Multi-generational family, elder at gate, pentacles throughout (legacy, inheritance, lasting wealth). The Ten of Pentacles represents legacy, family wealth, generational prosperity—the completion of material success passing to the next generation."
  },

  {
    id: 74,
    name: "Page of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "page",
    number: 11,
    element: "earth",
    modality: "mutable",
    astrology: "earth of earth",
    numerology: 11,
    kabbalah: "princess of earth",
    symbols: ["young figure", "pentacle", "green field", "mountains"],
    archetypes: ["student", "manifestation", "opportunity"],
    themes: ["new financial opportunity", "manifestation", "skill development", "grounding"],
    keywords: {
      upright: ["opportunity", "new venture", "manifestation", "study"],
      reversed: ["lack of progress", "procrastination", "get-rich-quick schemes"]
    },
    jungian: "eager student",
    chakra: "root",
    seasonality: "planting / learning",
    timeframe: "new beginning",
    advice: "Study diligently. Opportunity knocks. Ground your dreams in practical action.",
    shadow_work: "Lack of follow-through, procrastination, distraction from real work",
    light: "Eager learning, practical dreaming, diligent study",
    questions: [
      "What opportunity presents itself?",
      "What skill must you learn?",
      "How do you ground your vision?"
    ],
    description: "Young figure studying pentacle intently, green field (new opportunity, learning, manifestation). The Page of Pentacles represents new financial ventures, study, practical opportunities—the eager student of material mastery."
  },

  {
    id: 75,
    name: "Knight of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "knight",
    number: 12,
    element: "earth",
    modality: "cardinal",
    astrology: "air of earth",
    numerology: 12,
    kabbalah: "prince of earth",
    symbols: ["knight on horse", "pentacle", "plowed field", "deliberate pace"],
    archetypes: ["worker", "reliability", "persistence"],
    themes: ["hard work", "responsibility", "reliability", "routine"],
    keywords: {
      upright: ["hard work", "reliability", "routine", "persistence"],
      reversed: ["laziness", "obsession with work", "stuck in routine"]
    },
    jungian: "reliable worker",
    chakra: "root",
    seasonality: "steady progress",
    timeframe: "slow but sure",
    advice: "Slow and steady. Reliability wins. Do the work, every day, no drama.",
    shadow_work: "Workaholism, stuck in routine, lack of vision, plodding without purpose",
    light: "Reliable dedication, steady progress, grounded action",
    questions: [
      "Where must you show up consistently?",
      "What work requires patient dedication?",
      "How do you balance routine with flexibility?"
    ],
    description: "Knight on stationary horse holding pentacle, plowed field (hard work, reliability, steady progress). The Knight of Pentacles represents reliability, hard work, dedication—the worker who shows up every day without fail."
  },

  {
    id: 76,
    name: "Queen of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "queen",
    number: 13,
    element: "earth",
    modality: "fixed",
    astrology: "water of earth",
    numerology: 13,
    kabbalah: "queen of earth",
    symbols: ["queen on throne", "pentacle", "rabbit", "lush garden", "roses"],
    archetypes: ["nurturer", "provider", "earth mother"],
    themes: ["nurturing", "practical", "working parent", "financial security"],
    keywords: {
      upright: ["nurturing", "practical", "providing", "working parent"],
      reversed: ["smothering", "neglect", "work-life imbalance"]
    },
    jungian: "great mother / provider",
    chakra: "heart",
    seasonality: "abundant harvest",
    timeframe: "established prosperity",
    advice: "Nurture through provision. Care for self and others. Abundance flows from grounded care.",
    shadow_work: "Martyr mom, neglecting self, using provision for control",
    light: "Abundant nurturer, grounded care, prosperity with warmth",
    questions: [
      "How do you nurture through practical means?",
      "Where must you provide for yourself first?",
      "How do you balance care with boundaries?"
    ],
    description: "Queen enthroned in lush garden, pentacle in lap, rabbit at feet (nurturing, practical care, abundant provision). The Queen of Pentacles represents nurturing, practical care, financial security—the earth mother who provides with warmth."
  },

  {
    id: 77,
    name: "King of Pentacles",
    arcana: "minor",
    suit: "pentacles",
    rank: "king",
    number: 14,
    element: "earth",
    modality: "cardinal",
    astrology: "fire of earth",
    numerology: 14,
    kabbalah: "king of earth",
    symbols: ["king on throne", "pentacle", "castle", "bull", "grapes"],
    archetypes: ["provider", "leader", "wealth"],
    themes: ["wealth", "business", "leadership", "security"],
    keywords: {
      upright: ["wealth", "business success", "leadership", "security"],
      reversed: ["greed", "corruption", "financial failure", "stubbornness"]
    },
    jungian: "wise king / material mastery",
    chakra: "root",
    seasonality: "established empire",
    timeframe: "mastery achieved",
    advice: "Lead through provision. Build empires. But never lose touch with earth beneath throne.",
    shadow_work: "Greed, workaholic, empire over people, losing touch with why",
    light: "Wise provision, benevolent leadership, material mastery with heart",
    questions: [
      "What empire must you build?",
      "How do you lead with provision?",
      "Where does success distance you from humanity?"
    ],
    description: "King enthroned in opulent setting, pentacle, bull, grapes (wealth, business mastery, established power). The King of Pentacles represents material success, business mastery, providing leadership—the sovereign who built an empire with his own hands."
  }
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
