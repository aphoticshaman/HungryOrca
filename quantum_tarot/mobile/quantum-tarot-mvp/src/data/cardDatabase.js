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
  }

  // TODO: Add 56 minor arcana cards (Wands, Cups, Swords, Pentacles)
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
