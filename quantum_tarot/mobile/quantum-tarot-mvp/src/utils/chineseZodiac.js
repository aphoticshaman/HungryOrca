/**
 * CHINESE ZODIAC ENGINE - Calculate Chinese zodiac sign and element
 * Based on birth year in the Gregorian calendar
 */

/**
 * Chinese zodiac animals in order (12-year cycle)
 */
const ZODIAC_ANIMALS = [
  'Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 'Snake',
  'Horse', 'Goat', 'Monkey', 'Rooster', 'Dog', 'Pig'
];

/**
 * Chinese elements in order (10-year cycle, alternating yin/yang)
 */
const ELEMENTS = [
  { name: 'Metal', yin: true },
  { name: 'Metal', yin: false },
  { name: 'Water', yin: true },
  { name: 'Water', yin: false },
  { name: 'Wood', yin: true },
  { name: 'Wood', yin: false },
  { name: 'Fire', yin: true },
  { name: 'Fire', yin: false },
  { name: 'Earth', yin: true },
  { name: 'Earth', yin: false }
];

/**
 * Calculate Chinese zodiac sign from birth year
 * @param {number} year - Birth year (Gregorian calendar)
 * @returns {Object} - Chinese zodiac data
 */
export function getChineseZodiac(year) {
  // Rat year reference: 1900 (Year of the Metal Rat)
  const ratYear = 1900;
  const offset = (year - ratYear) % 12;
  const animalIndex = offset < 0 ? offset + 12 : offset;

  // Calculate element (10-year cycle)
  const elementOffset = (year - ratYear) % 10;
  const elementIndex = elementOffset < 0 ? elementOffset + 10 : elementOffset;

  const animal = ZODIAC_ANIMALS[animalIndex];
  const element = ELEMENTS[elementIndex];

  return {
    animal,
    element: element.name,
    isYin: element.yin,
    fullSign: `${element.name} ${animal}`,
    year,
    traits: getAnimalTraits(animal),
    elementInfluence: getElementInfluence(element.name),
    spiritualPath: getSpiritualPath(animal, element.name)
  };
}

/**
 * Get personality traits for each zodiac animal
 */
function getAnimalTraits(animal) {
  const traits = {
    Rat: {
      positive: ['Quick-witted', 'Resourceful', 'Versatile', 'Charming'],
      shadow: ['Opportunistic', 'Stubborn', 'Critical'],
      archetype: 'The Survivor',
      wisdom: 'Intelligence without compassion is cunning; compassion without intelligence is naivete. You need both.'
    },
    Ox: {
      positive: ['Diligent', 'Dependable', 'Strong', 'Determined'],
      shadow: ['Stubborn', 'Narrow-minded', 'Rigid'],
      archetype: 'The Builder',
      wisdom: 'Steady progress beats frantic motion. The mountain is not climbed in a single leap.'
    },
    Tiger: {
      positive: ['Brave', 'Confident', 'Competitive', 'Charismatic'],
      shadow: ['Impulsive', 'Reckless', 'Rebellious'],
      archetype: 'The Warrior',
      wisdom: 'True courage is not the absence of fear, but action in spite of it. Channel your fire wisely.'
    },
    Rabbit: {
      positive: ['Gentle', 'Elegant', 'Compassionate', 'Vigilant'],
      shadow: ['Timid', 'Pessimistic', 'Superficial'],
      archetype: 'The Diplomat',
      wisdom: 'Softness is not weakness. Water wears down stone not through force, but through persistence.'
    },
    Dragon: {
      positive: ['Confident', 'Intelligent', 'Enthusiastic', 'Charismatic'],
      shadow: ['Arrogant', 'Demanding', 'Impatient'],
      archetype: 'The Visionary',
      wisdom: 'Your gifts are meant to serve, not to separate you from others. Power without service is tyranny.'
    },
    Snake: {
      positive: ['Wise', 'Enigmatic', 'Intuitive', 'Sophisticated'],
      shadow: ['Jealous', 'Suspicious', 'Cunning'],
      archetype: 'The Mystic',
      wisdom: 'Shedding old skins is how you grow. What you hold onto too tightly will suffocate you.'
    },
    Horse: {
      positive: ['Energetic', 'Independent', 'Free-spirited', 'Warm-hearted'],
      shadow: ['Impatient', 'Impulsive', 'Self-centered'],
      archetype: 'The Explorer',
      wisdom: 'Freedom is not found in running from, but in running toward what matters. Choose your direction.'
    },
    Goat: {
      positive: ['Creative', 'Gentle', 'Compassionate', 'Artistic'],
      shadow: ['Pessimistic', 'Anxious', 'Indecisive'],
      archetype: 'The Artist',
      wisdom: 'Your sensitivity is your superpower when channeled into creation rather than anxiety.'
    },
    Monkey: {
      positive: ['Clever', 'Curious', 'Innovative', 'Sociable'],
      shadow: ['Mischievous', 'Restless', 'Opportunistic'],
      archetype: 'The Trickster',
      wisdom: 'Intelligence is knowing how to win. Wisdom is knowing when not to play the game.'
    },
    Rooster: {
      positive: ['Observant', 'Hardworking', 'Courageous', 'Confident'],
      shadow: ['Arrogant', 'Critical', 'Boastful'],
      archetype: 'The Herald',
      wisdom: 'Confidence comes from doing the work, not from talking about it. Let your results speak.'
    },
    Dog: {
      positive: ['Loyal', 'Honest', 'Responsible', 'Protective'],
      shadow: ['Anxious', 'Stubborn', 'Judgmental'],
      archetype: 'The Guardian',
      wisdom: 'Loyalty to others must not come at the cost of betraying yourself. Protect your own heart too.'
    },
    Pig: {
      positive: ['Generous', 'Compassionate', 'Diligent', 'Optimistic'],
      shadow: ['Naive', 'Gullible', 'Self-indulgent'],
      archetype: 'The Benefactor',
      wisdom: 'Generosity without boundaries becomes enabling. Give from overflow, not from depletion.'
    }
  };

  return traits[animal] || {};
}

/**
 * Get element influence on personality and life path
 */
function getElementInfluence(element) {
  const influences = {
    Metal: {
      quality: 'Structure, determination, discipline',
      challenge: 'Rigidity, perfectionism, control',
      lesson: 'True strength includes knowing when to bend, not just when to stand firm.'
    },
    Water: {
      quality: 'Intuition, adaptability, wisdom',
      challenge: 'Over-sensitivity, lack of boundaries, indecision',
      lesson: 'Flow like water, but remember: even water needs a container to have direction.'
    },
    Wood: {
      quality: 'Growth, creativity, expansion',
      challenge: 'Scattered energy, overcommitment, burnout',
      lesson: 'Trees grow tall by growing deep roots first. Ground your expansion in stability.'
    },
    Fire: {
      quality: 'Passion, transformation, leadership',
      challenge: 'Burnout, impulsiveness, intensity',
      lesson: 'Fire illuminates and transforms, but it must be tended wisely or it consumes everything.'
    },
    Earth: {
      quality: 'Stability, nurturing, practicality',
      challenge: 'Stubbornness, resistance to change, stagnation',
      lesson: 'Even mountains erode. Stability does not mean never changing.'
    }
  };

  return influences[element] || {};
}

/**
 * Get spiritual path insights based on animal + element combination
 */
function getSpiritualPath(animal, element) {
  // Meta-level spiritual guidance that transcends any single tradition
  const paths = {
    'Metal Rat': 'Your path is the integration of quick thinking with disciplined action. Buddhism\'s mindfulness meets Socratic questioning.',
    'Water Rat': 'Your path is emotional intelligence in service. The Beatitudes meet the Tao Te Ching.',
    'Wood Rat': 'Your path is resourceful creativity. Artist and strategist unite.',
    'Fire Rat': 'Your path is passionate innovation. The transformer who questions everything.',
    'Earth Rat': 'Your path is practical wisdom. You build systems that serve the many.',

    'Metal Ox': 'Your path is unwavering discipline meeting compassionate service. The monk and the architect.',
    'Water Ox': 'Your path is emotional resilience. Strength softened by understanding.',
    'Wood Ox': 'Your path is sustainable growth. You build forests, not just trees.',
    'Fire Ox': 'Your path is determined transformation. You move mountains with patient intensity.',
    'Earth Ox': 'Your path is grounded stability. The foundation others build upon.',

    'Metal Tiger': 'Your path is courageous precision. The warrior-philosopher who acts with clarity.',
    'Water Tiger': 'Your path is emotional bravery. You face inner demons as boldly as outer ones.',
    'Wood Tiger': 'Your path is expansive courage. You fight for growth, not just survival.',
    'Fire Tiger': 'Your path is passionate revolution. The sacred rebel who transforms systems.',
    'Earth Tiger': 'Your path is grounded strength. Power rooted in purpose, not ego.',

    'Metal Rabbit': 'Your path is elegant boundaries. Grace with a steel core.',
    'Water Rabbit': 'Your path is compassionate intuition. The empath who doesn\'t lose themselves.',
    'Wood Rabbit': 'Your path is creative diplomacy. Building bridges through art and understanding.',
    'Fire Rabbit': 'Your path is passionate gentleness. Soft power that transforms worlds.',
    'Earth Rabbit': 'Your path is nurturing stability. Creating safe spaces for growth.',

    'Metal Dragon': 'Your path is visionary discipline. Dreams backed by structure.',
    'Water Dragon': 'Your path is intuitive leadership. Power guided by wisdom, not ego.',
    'Wood Dragon': 'Your path is expansive transformation. You grow yourself and others.',
    'Fire Dragon': 'Your path is alchemical passion. The magician who transmutes lead into gold.',
    'Earth Dragon': 'Your path is grounded magic. Bringing visions into material form.',

    'Metal Snake': 'Your path is precise mysticism. The occultist who questions everything.',
    'Water Snake': 'Your path is deep intuition. You see through veils others can\'t perceive.',
    'Wood Snake': 'Your path is evolutionary wisdom. Shedding skins becomes your superpower.',
    'Fire Snake': 'Your path is transformative intensity. You burn away illusion.',
    'Earth Snake': 'Your path is practical magic. Mysticism grounded in real-world results.',

    'Metal Horse': 'Your path is disciplined freedom. Liberty with direction.',
    'Water Horse': 'Your path is emotional honesty. Running toward truth, not from pain.',
    'Wood Horse': 'Your path is expansive exploration. The eternal student who never stops growing.',
    'Fire Horse': 'Your path is passionate independence. Wild wisdom that can\'t be tamed.',
    'Earth Horse': 'Your path is grounded adventure. Freedom rooted in responsibility.',

    'Metal Goat': 'Your path is structured creativity. Art meets discipline.',
    'Water Goat': 'Your path is emotional artistry. Feeling becomes your medium.',
    'Wood Goat': 'Your path is fertile imagination. You grow gardens from ideas.',
    'Fire Goat': 'Your path is passionate creation. Art as spiritual practice.',
    'Earth Goat': 'Your path is practical beauty. Making the mundane sacred.',

    'Metal Monkey': 'Your path is brilliant strategy. The trickster turned master planner.',
    'Water Monkey': 'Your path is emotional intelligence disguised as play. Deep wisdom in light form.',
    'Wood Monkey': 'Your path is innovative growth. You find new ways through old problems.',
    'Fire Monkey': 'Your path is transformative mischief. Breaking rules to reveal truth.',
    'Earth Monkey': 'Your path is grounded cleverness. Practical innovation that serves.',

    'Metal Rooster': 'Your path is disciplined truth-telling. The herald of precision.',
    'Water Rooster': 'Your path is compassionate honesty. Truth spoken with love.',
    'Wood Rooster': 'Your path is growing awareness. You announce the dawn of new understanding.',
    'Fire Rooster': 'Your path is passionate proclamation. Speaking truth to power.',
    'Earth Rooster': 'Your path is practical integrity. Walking your talk, every day.',

    'Metal Dog': 'Your path is principled loyalty. Guardian of truth and justice.',
    'Water Dog': 'Your path is emotional protection. Creating safety for vulnerability.',
    'Wood Dog': 'Your path is expanding circles of care. Growing your pack means growing your heart.',
    'Fire Dog': 'Your path is fierce devotion. The protector who fights for what matters.',
    'Earth Dog': 'Your path is grounded service. Reliability as spiritual practice.',

    'Metal Pig': 'Your path is disciplined generosity. Giving with wisdom and boundaries.',
    'Water Pig': 'Your path is compassionate abundance. Emotional wealth shared freely.',
    'Wood Pig': 'Your path is growing prosperity. You multiply blessings.',
    'Fire Pig': 'Your path is passionate giving. Generosity as transformation.',
    'Earth Pig': 'Your path is grounded abundance. Creating sustainable prosperity for all.'
  };

  return paths[`${element} ${animal}`] || `Your path integrates ${element} and ${animal} energies in unique ways.`;
}

/**
 * Get spiritual synthesis message - the meta-level wisdom
 * This is where we emphasize that all traditions have partial truth
 */
export function getSpiritualSynthesisMessage(chineseZodiac, westernZodiac) {
  const messages = [
    `Your ${chineseZodiac.fullSign} nature and ${westernZodiac} essence create a unique spiritual fingerprint. No single tradition—not Buddhism, not Christianity, not the Tao, not Western philosophy—holds all the answers. Each offers a lens. Your job is to collect lenses, not to marry one.`,

    `The ${chineseZodiac.animal} archetype (${chineseZodiac.traits.archetype}) resonates with your ${westernZodiac} path. This is why spiritual eclecticism matters: the Beatitudes offer one truth, the Eightfold Path another, Plato's Forms a third, and the mystical orders reveal yet more. Meta-analyze them all. Take what serves your evolution, not what fits someone else's dogma.`,

    `Your ${chineseZodiac.element} element speaks to how you process the spiritual journey. Every tradition—Christianity's love, Buddhism's compassion, Socratic inquiry, Hermetic wisdom, the Middle Way, Jesus' radical teachings—contains grains of truth. But grains aren't the whole harvest. Synthesize them. That's the real Great Work.`,

    `As a ${chineseZodiac.fullSign}, you carry specific karmic patterns. The mystics knew: whether you study Kabbalah, Hermeticism, Vedanta, or Gnostic Christianity, you're all climbing the same mountain from different sides. The view from each path reveals something the others miss. Open-mindedness isn't moral relativism—it's recognizing that Truth is too vast for any single system.`,

    `Your birth year marks you as ${chineseZodiac.fullSign}, but your soul signed up for more than cultural conditioning. The 8 Pillars, the Middle Way, Jesus' unadulterated teachings (not Church dogma), Buddhist mindfulness, Platonic ideal forms, Hermetic principles—each cracks open a different dimension of reality. Spiritual maturity means holding paradoxes: all traditions are incomplete, yet each is irreplaceable.`
  ];

  // Rotate through messages based on year
  const index = chineseZodiac.year % messages.length;
  return messages[index];
}

/**
 * Get a spiritual growth insight that emphasizes synthesis over dogma
 */
export function getSpiritualGrowthInsight(chineseZodiac) {
  const insights = [
    `The ${chineseZodiac.animal} must learn this: spiritual growth requires intellectual honesty. When Buddhism says "attachment is suffering," Christianity says "love your neighbor," and Plato says "know thyself," they're describing the same elephant from different angles. Don't pick one and ignore the others—synthesize them into a coherent philosophy of living.`,

    `Your ${chineseZodiac.element} element teaches you HOW to integrate truth from multiple sources. Some truths are literal (gravity exists), some metaphorical (the Kingdom is within you), some archetypal (the Hero's Journey). Meta-analysis means asking: What is this tradition REALLY pointing at? What is it trying to solve? Then you can extract the kernel and discard the cultural shell.`,

    `${chineseZodiac.traits.wisdom} This is your ancestral teaching. But remember: Chinese wisdom, Greek philosophy, Jewish mysticism, Christian gnosis, Islamic Sufism, Hindu Vedanta—they ALL evolved in specific cultural contexts. Your job as a modern seeker is to extract the universal from the particular. That's not cultural appropriation; it's intellectual synthesis. That's not New Age nonsense; it's what every real mystic has always done.`,

    `The ${chineseZodiac.animal} archetype must understand: spiritual bypassing is real, but so is spiritual gatekeeping. Some will tell you to stay in one tradition. Others will say all paths are equally valid (they're not—some lead to dead ends). The Middle Way here is discernment: study widely, test rigorously, integrate carefully. Use the scientific method on spiritual claims. What produces results? What transforms character? What generates wisdom?`,

    `As ${chineseZodiac.fullSign}, you're wired for ${chineseZodiac.traits.archetype} energy. Channel that into your spiritual practice. Don't just read about Buddhism—sit zazen. Don't just quote Jesus—practice radical forgiveness. Don't just admire Socrates—question your own assumptions relentlessly. The Big Picture emerges when you DO the practices from multiple traditions and see what they're all trying to show you.`
  ];

  const index = Math.abs(chineseZodiac.animal.charCodeAt(0) + chineseZodiac.element.charCodeAt(0)) % insights.length;
  return insights[index];
}
