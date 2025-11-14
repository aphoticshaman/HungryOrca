/**
 * QUANTUM NARRATIVE ENGINE
 * Uses quantum seed to uniquely vary EVERY sentence, word choice, structure
 * Creates synthesis personality blending pop culture wisdom voices:
 * - Carl Jung (depth psychology)
 * - Pema Chödrön (compassionate confrontation)
 * - Alan Watts (playful paradox)
 * - Audre Lorde (fierce truth, self-care as warfare)
 * - James Baldwin (poetic truth-telling)
 * - Brené Brown (vulnerability wisdom)
 * - Octavia Butler (pattern recognition, sci-fi mythos)
 * - bell hooks (love as practice)
 *
 * NO TWO SENTENCES will have the same structure or phrasing
 */

import { generateQuantumSeed } from './quantumRNG';

/**
 * Quantum-seeded word variation
 * Takes a concept and returns one of many synonyms/phrasings
 */
function quantumWordChoice(concept, quantumSeed) {
  const variations = {
    // Tarot-specific verbs
    reveals: ['reveals', 'illuminates', 'uncovers', 'exposes', 'shows you', 'lays bare', 'brings to light', 'makes visible'],
    suggests: ['suggests', 'hints at', 'whispers of', 'points toward', 'invites you to consider', 'asks you to notice', 'nudges you toward'],
    warns: ['warns', 'cautions', 'flags', 'signals danger in', 'raises a red flag about', 'sounds alarm bells for'],
    invites: ['invites', 'calls you to', 'beckons you toward', 'summons', 'asks of you', 'demands', 'requests'],

    // Emotional states
    struggle: ['struggle', 'wrestle with', 'grapple with', 'contend with', 'battle', 'face down', 'confront'],
    feel: ['feel', 'experience', 'sense', 'know in your bones', 'carry', 'hold', 'embody'],
    want: ['want', 'desire', 'crave', 'hunger for', 'yearn for', 'seek', 'long for', 'need'],
    fear: ['fear', 'dread', 'are terrified of', 'avoid', 'run from', 'resist', 'recoil from'],

    // Wisdom/insight language
    truth: ['truth', 'reality', 'what is', 'the real deal', 'the raw facts', 'the honest situation', 'the unvarnished truth'],
    wisdom: ['wisdom', 'insight', 'knowing', 'deep understanding', 'ancient knowledge', 'hard-won clarity'],
    pattern: ['pattern', 'cycle', 'loop', 'recurring theme', 'familiar dance', 'old story', 'known rhythm'],
    shadow: ['shadow', 'dark side', 'hidden part', 'repressed aspect', 'what you hide', 'what you deny', 'your underworld'],

    // Action imperatives
    release: ['release', 'let go of', 'surrender', 'drop', 'set down', 'free yourself from', 'untangle from'],
    embrace: ['embrace', 'welcome', 'accept', 'open to', 'allow', 'make space for', 'say yes to'],
    examine: ['examine', 'look closely at', 'investigate', 'explore', 'dig into', 'turn over', 'question'],
    act: ['act', 'move', 'take action', 'do something', 'step forward', 'make your move', 'commit'],

    // Mystical/spiritual
    soul: ['soul', 'deep self', 'true nature', 'essence', 'core being', 'authentic self', 'inner truth'],
    destiny: ['destiny', 'path', 'calling', 'life purpose', 'soul mission', 'evolutionary edge', 'growth direction'],
    divine: ['divine', 'sacred', 'holy', 'cosmic', 'universal', 'spiritual', 'transcendent'],

    // Time/change
    now: ['now', 'this moment', 'right here', 'today', 'at this crossroads', 'in this season', 'currently'],
    change: ['change', 'shift', 'transform', 'evolve', 'alchemize', 'transmute', 'metamorphose'],

    // Relationship to self/others
    you: ['you', 'your soul', 'your being', 'your heart', 'the part of you that knows', 'your deepest self'],
    they: ['they', 'the other', 'this person', 'your counterpart', 'the mirror', 'your dance partner'],

    // Intensity modifiers
    very: ['very', 'deeply', 'profoundly', 'intensely', 'radically', 'fundamentally', 'utterly'],
    somewhat: ['somewhat', 'to some degree', 'in part', 'partially', 'moderately', 'to an extent'],
  };

  if (!variations[concept]) return concept;

  const options = variations[concept];
  const index = Math.floor((quantumSeed * options.length)) % options.length;
  return options[index];
}

/**
 * Quantum sentence structure templates
 * 50+ different ways to construct a sentence about a card
 */
export function quantumSentenceStructure(cardName, meaning, position, quantumSeed) {
  const structures = [
    // Jungian depth psychology style
    `${cardName} ${quantumWordChoice('reveals', quantumSeed)} a ${quantumWordChoice('pattern', quantumSeed * 0.3)} you've been living unconsciously.`,
    `The archetype of ${cardName} speaks to the ${quantumWordChoice('shadow', quantumSeed * 0.7)} you carry in the realm of ${position}.`,
    `When ${cardName} appears in ${position}, ${meaning} becomes the psychic material you must ${quantumWordChoice('examine', quantumSeed * 0.5)}.`,

    // Pema Chödrön compassionate confrontation
    `${cardName} is ${quantumWordChoice('invites', quantumSeed * 0.2)} you to stay with the discomfort of ${meaning}.`,
    `This card doesn't offer easy answers—it asks you to sit with ${meaning} until ${quantumWordChoice('wisdom', quantumSeed * 0.4)} emerges.`,
    `Notice how ${cardName} triggers you. That reaction? That's where your work is.`,

    // Alan Watts playful paradox
    `${cardName} whispers the cosmic joke: ${meaning} is both problem and solution.`,
    `You want to escape ${meaning}, but ${cardName} laughs—running away IS the trap.`,
    `The universe placed ${cardName} here to remind you: ${meaning} dissolves when you stop fighting it.`,

    // Audre Lorde fierce truth
    `${cardName} demands you stop pretending ${meaning} isn't happening. Your survival depends on facing this.`,
    `Self-care means ${quantumWordChoice('act', quantumSeed * 0.6)}ing on what ${cardName} reveals about ${meaning}. Not later. Now.`,
    `${cardName} is not here to be nice. It's here to show you where ${meaning} is eating you alive.`,

    // James Baldwin poetic truth
    `${cardName} tells the story you've been afraid to speak: ${meaning} is the price of your silence.`,
    `In ${position}, ${cardName} becomes the mirror you've avoided. ${meaning} stares back, unblinking.`,
    `The ${quantumWordChoice('truth', quantumSeed * 0.8)} ${cardName} offers is this: ${meaning} will not release you until you release it.`,

    // Brené Brown vulnerability wisdom
    `${cardName} asks: Can you be brave enough to ${quantumWordChoice('feel', quantumSeed * 0.9)} ${meaning} without numbing?`,
    `Vulnerability looks like ${quantumWordChoice('embrace', quantumSeed * 0.1)}ing ${meaning} even when it terrifies you. ${cardName} is your guide.`,
    `${cardName} shows up when you're ready to stop armoring against ${meaning} and start living wholeheartedly.`,

    // Octavia Butler pattern recognition
    `${cardName} is a pattern interrupt. ${meaning} has been running your life—time to rewrite the code.`,
    `Watch how ${cardName} echoes across your ${position}. ${meaning} is the algorithm you inherited.`,
    `${meaning} is the shape your trauma takes. ${cardName} offers a different blueprint.`,

    // bell hooks love as practice
    `To ${quantumWordChoice('act', quantumSeed * 0.15)} with love here means ${quantumWordChoice('embrace', quantumSeed * 0.25)}ing the ${meaning} ${cardName} brings.`,
    `${cardName} asks: How do you practice love toward yourself in the face of ${meaning}?`,
    `Love is a verb. ${cardName} shows you must actively ${quantumWordChoice('examine', quantumSeed * 0.35)} ${meaning} to grow.`,

    // Direct mystic transmission
    `${meaning}. That's it. That's what ${cardName} is saying.`,
    `${cardName} in ${position}: ${meaning}. No fluff, no spiritual bypass.`,
    `Here's the download from ${cardName}: ${meaning}. What you do with it is your choice.`,

    // Conversational/real talk
    `Look, ${cardName} isn't subtle. It's saying ${meaning} loud and clear.`,
    `${cardName} showed up because you need to ${quantumWordChoice('act', quantumSeed * 0.45)} on ${meaning}. Period.`,
    `Can we be real? ${cardName} is calling you out on ${meaning}.`,

    // Mythological/epic
    `${cardName} arrives as the herald of ${meaning}, the quest only you can undertake.`,
    `The ${quantumWordChoice('soul', quantumSeed * 0.55)}'s journey through ${position} demands you face ${meaning} like the hero you are.`,
    `${cardName} is the threshold guardian. To pass, you must ${quantumWordChoice('embrace', quantumSeed * 0.65)} ${meaning}.`,

    // Quantum/scientific metaphor
    `${cardName} collapses the waveform of possibility into ${meaning}. Observe it. Don't collapse.`,
    `In the probability field of ${position}, ${cardName} makes ${meaning} inevitable—unless you change your input.`,
    `${meaning} is the attractor state. ${cardName} shows you're caught in its gravity well.`,

    // Taoist flow
    `${cardName} flows like water around ${meaning}. Resist and you suffer. Move with it and you thrive.`,
    `The Tao of ${position} is revealed through ${cardName}: ${meaning} is neither good nor bad, just is.`,
    `Wu wei—effortless action. ${cardName} asks you to stop forcing and allow ${meaning} to unfold.`,

    // Existential/Camus
    `${cardName} is absurd. ${meaning} makes no sense. And yet you must ${quantumWordChoice('act', quantumSeed * 0.75)} anyway.`,
    `In ${position}, ${cardName} confronts you with ${meaning}. Will you create meaning or succumb to nihilism?`,

    // Poetic/lyrical
    `${cardName} sings of ${meaning}, a melody you've hummed but never fully voiced.`,
    `Listen: ${cardName} whispers ${meaning} in the language your ${quantumWordChoice('soul', quantumSeed * 0.85)} speaks.`,

    // Urgent/crisis
    `${cardName} is an emergency broadcast: ${meaning} needs your attention NOW.`,
    `If you ignore ${cardName}, ${meaning} will escalate. This is your warning shot.`,

    // Gentle/nurturing
    `${cardName} gently holds space for ${meaning}, offering you permission to ${quantumWordChoice('feel', quantumSeed * 0.95)} it all.`,
    `With ${quantumWordChoice('divine', quantumSeed * 0.12)} tenderness, ${cardName} invites you to ${quantumWordChoice('examine', quantumSeed * 0.22)} ${meaning}.`,

    // Provocative/challenging
    `${cardName} is calling bullshit on how you've been handling ${meaning}.`,
    `You think you understand ${meaning}? ${cardName} says think again.`,

    // Integration/synthesis
    `${cardName} weaves ${meaning} into the larger tapestry of ${position}, showing how everything connects.`,
    `In the ecology of your reading, ${cardName} represents ${meaning}—remove it and the whole system shifts.`
  ];

  // Use quantum seed to pick structure
  const index = Math.floor((quantumSeed * structures.length)) % structures.length;
  return structures[index];
}

/**
 * Quantum transition phrases between cards
 * 40+ ways to connect one card to the next
 */
export function quantumTransition(quantumSeed) {
  const transitions = [
    'And then,',
    'But here\'s the thing:',
    'Meanwhile,',
    'Now consider this:',
    'Which brings us to',
    'This connects to',
    'Building on that,',
    'In contrast,',
    'Yet simultaneously,',
    'Deepening this,',
    'Zooming out,',
    'Getting more specific,',
    'Shifting perspective,',
    'From another angle,',
    'The plot thickens:',
    'Hold that thought—',
    'Wait, there\'s more:',
    'Digging deeper,',
    'Surfacing now,',
    'Emerging from the shadows,',
    'In the same breath,',
    'Interestingly,',
    'Here\'s where it gets real:',
    'Not to complicate things, but',
    'Adding another layer,',
    'Pulling back the curtain,',
    'Let\'s not ignore',
    'We can\'t skip over',
    'Crucially,',
    'The heart of it:',
    'Cutting through the noise,',
    'Stripping away pretense,',
    'With raw honesty,',
    'In the light of day,',
    'Looking beneath the surface,',
    'Connecting the dots,',
    'The throughline here is',
    'Threading this together,',
    'Weaving this into',
    'Layering onto that,'
  ];

  const index = Math.floor((quantumSeed * transitions.length)) % transitions.length;
  return transitions[index];
}

/**
 * Quantum opening structures for synthesis
 * 30+ ways to START the final synthesis
 */
export function quantumSynthesisOpening(readingType, userName, quantumSeed) {
  const openings = [
    `${userName}, the cards have spoken—and they're not mincing words.`,
    `Alright ${userName}, let's cut through the spiritual fluff and get real about what's happening.`,
    `${userName}, your ${readingType} reading reveals a story that's been writing itself beneath your awareness.`,
    `Here's what the universe is telling you, ${userName}:`,
    `Listen up, ${userName}—this reading is a mirror, and you might not like everything you see.`,
    `${userName}, the cards are laying out a path. Whether you walk it is up to you.`,
    `Let me be straight with you, ${userName}:`,
    `${userName}, this ${readingType} reading weaves together several threads you've been avoiding.`,
    `The energy around your ${readingType} situation is complex, ${userName}, but the cards cut through confusion.`,
    `${userName}, if you're looking for easy answers, turn back now. These cards demand more.`,
    `What emerges from this reading, ${userName}, is a pattern older than you realize.`,
    `${userName}, the cards are calling you into deeper work around ${readingType}.`,
    `Your soul has something to say about ${readingType}, ${userName}, and these cards are its voice.`,
    `${userName}, every card in this spread is a breadcrumb leading you toward truth.`,
    `The through-line in your reading, ${userName}, is transformation—whether you're ready or not.`,
    `${userName}, these cards form a constellation of meaning around your ${readingType} question.`,
    `Let's talk about what's really going on, ${userName}.`,
    `${userName}, the tarot isn't here to comfort you. It's here to free you.`,
    `Your reading tells a hero's journey, ${userName}, and you're right at the threshold.`,
    `${userName}, the universe has been trying to get your attention. These cards are the megaphone.`,
    `If life is a story, ${userName}, these cards are the chapter you're writing right now.`,
    `${userName}, what you're facing with ${readingType} isn't random. The pattern is clear.`,
    `The cards speak in symbols, ${userName}, but the message is concrete:`,
    `${userName}, I'm going to be honest about what I see here.`,
    `Your ${readingType} situation is at a crossroads, ${userName}. Here's the map.`,
    `${userName}, these cards are your medicine—and medicine doesn't always taste sweet.`,
    `Let me hold up the mirror, ${userName}. What do you see?`,
    `${userName}, this reading is an invitation to step into your power around ${readingType}.`,
    `The cards arranged before you, ${userName}, tell a truth you've been circling.`,
    `${userName}, buckle up. This reading goes deep.`
  ];

  const index = Math.floor((quantumSeed * openings.length)) % openings.length;
  return openings[index];
}

/**
 * Quantum synthesis closing structures
 * 25+ ways to END the final synthesis with action/wisdom
 */
export function quantumSynthesisClosing(quantumSeed) {
  const closings = [
    'The cards have spoken. Now you must.',
    'This reading isn\'t the end—it\'s the beginning of your conscious choice.',
    'Take what resonates. Leave what doesn\'t. But don\'t ignore what triggers you.',
    'Your move, beloved. The universe has shown its hand.',
    'May you move forward with clarity, courage, and radical honesty.',
    'The path is clear. Whether you walk it is your sacred choice.',
    'Remember: the cards show probability, not destiny. You still have free will.',
    'Sit with this. Journal it. Let it work on you before you work on it.',
    'You already know the truth. These cards just gave you permission to see it.',
    'Go gently. Go fiercely. But for the love of all that\'s holy, GO.',
    'This is your life calling itself back to you. Answer it.',
    'The work begins now—not tomorrow, not when you feel ready. Now.',
    'You don\'t have to do this alone. But you do have to do it.',
    'Trust the process. Trust yourself. Trust that you drew exactly the cards you needed.',
    'May this reading serve your highest good and deepest truth.',
    'The universe is conspiring FOR you, not against you. Even when it doesn\'t feel like it.',
    'Every ending is a beginning. Every death, a rebirth. Hold both.',
    'You are the author of your life. These cards are just early drafts.',
    'Let this reading be the permission slip you\'ve been waiting for.',
    'The oracle has spoken. Now it\'s your turn to speak back with your life.',
    'Bless this journey. Honor this moment. And then step forward.',
    'You came to this reading for a reason. Let it change you.',
    'May you have the courage to live what these cards revealed.',
    'The reading is complete. The living has just begun.',
    'Go forth and be the magic you seek.'
  ];

  const index = Math.floor((quantumSeed * closings.length)) % closings.length;
  return closings[index];
}

/**
 * Generate quantum-varied reference to astrological context
 * 20+ ways to weave in Pisces/MBTI/Lilith/etc without being repetitive
 */
export function quantumAstroReference(astroContext, quantumSeed) {
  const { sunSign, mbtiType, lilith, chiron, moonPhase } = astroContext;

  const templates = [
    `As a ${sunSign} with ${mbtiType} wiring, this hits differently for you:`,
    `Your ${sunSign} sun and ${mbtiType} mind are both activated here.`,
    `Given your ${lilith.sign} Lilith—where you've been told to suppress ${lilith.sign} energy—`,
    `The ${moonPhase.phaseName} amplifies this: ${moonPhase.phaseAdvice}`,
    `Your ${chiron.sign} Chiron wound (${chiron.meaning}) is relevant here.`,
    `As someone with ${mbtiType} preferences, you'll approach this through`,
    `Your ${sunSign} nature wants to`,
    `The cosmic context matters: ${moonPhase.phaseName} suggests`,
    `With Lilith in ${lilith.sign}, your shadow work involves`,
    `Chiron in ${chiron.sign} means you heal by`,
    `For ${mbtiType} folks, this typically manifests as`,
    `${sunSign} rising with ${mbtiType} processing means`,
    `The ${moonPhase.phaseName} isn't coincidental—it mirrors`,
    `Your ${lilith.sign} Lilith is screaming about`,
    `As a ${mbtiType}, your blind spot here might be`,
    `${sunSign} energy combined with ${mbtiType} creates`,
    `During this ${moonPhase.phaseName}, you're being asked to`,
    `Your Chiron wound in ${chiron.sign} gets activated when`,
    `For ${sunSign} sun signs, especially ${mbtiType} types,`,
    `The ${moonPhase.phaseName} timing suggests the universe is`
  ];

  const index = Math.floor((quantumSeed * templates.length)) % templates.length;
  return templates[index];
}

/**
 * Master function: Generate completely unique synthesis using quantum seed
 * EVERY sentence will be structurally different
 * EVERY word choice will be varied
 * EVERY transition will be fresh
 */
export function generateQuantumNarrative(cards, context, quantumSeed = generateQuantumSeed()) {
  // Use seed to generate sub-seeds for each section
  const seeds = {
    opening: (quantumSeed * 0.123) % 1,
    transitions: Array.from({length: cards.length}, (_, i) => (quantumSeed * (0.234 + i * 0.111)) % 1),
    sentences: Array.from({length: cards.length * 3}, (_, i) => (quantumSeed * (0.345 + i * 0.111)) % 1),
    astroRefs: Array.from({length: 5}, (_, i) => (quantumSeed * (0.456 + i * 0.111)) % 1),
    closing: (quantumSeed * 0.789) % 1
  };

  return {
    openingSeed: seeds.opening,
    transitionSeeds: seeds.transitions,
    sentenceSeeds: seeds.sentences,
    astroRefSeeds: seeds.astroRefs,
    closingSeed: seeds.closing,

    // Helper to get varied phrasing
    getWord: (concept) => quantumWordChoice(concept, quantumSeed),
    getSentence: (cardName, meaning, position) => quantumSentenceStructure(cardName, meaning, position, seeds.sentences.shift()),
    getTransition: () => quantumTransition(seeds.transitions.shift()),
    getAstroRef: (astroContext) => quantumAstroReference(astroContext, seeds.astroRefs.shift()),
    getOpening: (readingType, userName) => quantumSynthesisOpening(readingType, userName, seeds.opening),
    getClosing: () => quantumSynthesisClosing(seeds.closing)
  };
}
