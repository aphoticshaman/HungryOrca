/**
 * Quantum Tarot - Adaptive Language Engine (JavaScript)
 * Ported from Python - Runs entirely on phone
 * Delivers same card in different voices based on personality
 */

/**
 * Communication Voices
 */
export const CommunicationVoice = {
  ANALYTICAL_GUIDE: 'analytical_guide',
  INTUITIVE_MYSTIC: 'intuitive_mystic',
  SUPPORTIVE_FRIEND: 'supportive_friend',
  DIRECT_COACH: 'direct_coach',
  GENTLE_NURTURER: 'gentle_nurturer',
  WISE_MENTOR: 'wise_mentor',
  PLAYFUL_EXPLORER: 'playful_explorer',
  BALANCED_SAGE: 'balanced_sage'
};

/**
 * Aesthetic Profiles
 */
export const AestheticProfile = {
  MINIMAL_MODERN: 'minimal_modern',
  SOFT_MYSTICAL: 'soft_mystical',
  BOLD_AUTHENTIC: 'bold_authentic',
  ELEGANT_CLASSIC: 'elegant_classic',
  WITCHY_EARTHY: 'witchy_earthy'
};

/**
 * User Communication Profile
 */
export class UserCommunicationProfile {
  constructor(primaryVoice, aesthetic) {
    this.primaryVoice = primaryVoice;
    this.secondaryVoice = null;
    this.aesthetic = aesthetic || AestheticProfile.SOFT_MYSTICAL;

    // Delivery preferences
    this.sentenceLength = 'medium';  // short, medium, long
    this.metaphorDensity = 'medium';  // low, medium, high
    this.therapeuticExplicitness = 'subtle';  // hidden, subtle, explicit
    this.spiritualLanguage = 'moderate';  // minimal, moderate, rich
    this.emojiUse = true;  // Gen Z/Millennial preference

    // Tone modifiers (0-1 scales)
    this.warmthLevel = 0.7;
    this.directnessLevel = 0.5;
    this.empowermentVsComfort = 0.5;  // 0=comfort, 1=challenge
  }
}

/**
 * Adaptive Language Engine
 */
export class AdaptiveLanguageEngine {
  /**
   * Determine communication voice from personality profile
   */
  static determineVoice(personalityProfile) {
    // Extract key traits
    const analytical = personalityProfile.analyticalIntuitive < 0.4;
    const intuitive = personalityProfile.analyticalIntuitive > 0.6;
    const highEmotion = personalityProfile.emotionalRegulation < 0.4;
    const actionOriented = personalityProfile.actionOrientation > 0.6;
    const sensitive = personalityProfile.emotionalRegulation < 0.3;
    const structured = personalityProfile.structureFlexibility < 0.4;
    const flexible = personalityProfile.structureFlexibility > 0.6;

    // Decision tree for voice selection
    if (sensitive || highEmotion) {
      return CommunicationVoice.GENTLE_NURTURER;
    } else if (analytical && structured) {
      return CommunicationVoice.ANALYTICAL_GUIDE;
    } else if (intuitive && flexible) {
      return CommunicationVoice.INTUITIVE_MYSTIC;
    } else if (actionOriented && personalityProfile.optimismRealism > 0.6) {
      return CommunicationVoice.DIRECT_COACH;
    } else if (structured && personalityProfile.internalExternalLocus > 0.5) {
      return CommunicationVoice.WISE_MENTOR;
    } else if (flexible && personalityProfile.riskTolerance > 0.6) {
      return CommunicationVoice.PLAYFUL_EXPLORER;
    }
    return CommunicationVoice.BALANCED_SAGE;
  }

  /**
   * Determine aesthetic profile
   */
  static determineAesthetic(personalityProfile, birthYear = null) {
    // Generational defaults
    const genZ = birthYear && birthYear >= 1997;
    const millennial = birthYear && birthYear >= 1981 && birthYear < 1997;

    if (genZ && personalityProfile.analyticalIntuitive < 0.4) {
      return AestheticProfile.MINIMAL_MODERN;
    } else if (genZ) {
      return AestheticProfile.BOLD_AUTHENTIC;
    } else if (millennial) {
      return AestheticProfile.SOFT_MYSTICAL;
    } else if (personalityProfile.analyticalIntuitive < 0.3) {
      return AestheticProfile.MINIMAL_MODERN;
    } else if (personalityProfile.structureFlexibility < 0.4) {
      return AestheticProfile.ELEGANT_CLASSIC;
    }
    return AestheticProfile.WITCHY_EARTHY;
  }

  /**
   * Build complete communication profile
   */
  static buildCommunicationProfile(personalityProfile, birthYear = null, genderIdentity = null) {
    const voice = this.determineVoice(personalityProfile);
    const aesthetic = this.determineAesthetic(personalityProfile, birthYear);

    const profile = new UserCommunicationProfile(voice, aesthetic);

    // Sentence length
    if (personalityProfile.analyticalIntuitive < 0.4) {
      profile.sentenceLength = 'medium';  // Analytical wants clarity
    } else if (personalityProfile.analyticalIntuitive > 0.6) {
      profile.sentenceLength = 'long';  // Intuitive enjoys flow
    }

    // Metaphor density
    if (personalityProfile.analyticalIntuitive < 0.3) {
      profile.metaphorDensity = 'low';  // Give it straight
    } else if (personalityProfile.analyticalIntuitive > 0.7) {
      profile.metaphorDensity = 'high';  // They love symbolic language
    }

    // Therapeutic explicitness
    if (personalityProfile.primaryFramework === 'CBT' ||
        personalityProfile.primaryFramework === 'DBT') {
      profile.therapeuticExplicitness = 'subtle';
    } else {
      profile.therapeuticExplicitness = 'hidden';
    }

    // Spiritual language
    if (personalityProfile.analyticalIntuitive < 0.3) {
      profile.spiritualLanguage = 'minimal';  // Less woo, more practical
    } else if (personalityProfile.analyticalIntuitive > 0.7) {
      profile.spiritualLanguage = 'rich';  // Full mystical experience
    }

    // Emoji use (generational)
    const genZMillennial = birthYear && birthYear >= 1981;
    profile.emojiUse = genZMillennial;

    // Tone modifiers
    profile.warmthLevel = 0.9 - personalityProfile.emotionalRegulation;  // Lower reg = more warmth
    profile.directnessLevel = personalityProfile.actionOrientation;  // Action-oriented want direct
    profile.empowermentVsComfort = personalityProfile.internalExternalLocus;  // Internal locus = can handle challenge

    return profile;
  }

  /**
   * Voice-specific templates
   */
  static getVoiceTemplates() {
    return {
      [CommunicationVoice.ANALYTICAL_GUIDE]: {
        opening: 'In the {position} position, {reversed}{cardName} indicates:',
        connector: 'This suggests that',
        closing: null
      },
      [CommunicationVoice.INTUITIVE_MYSTIC]: {
        opening: 'The {reversed}{cardName} appears in your {position}, whispering:',
        connector: 'The universe is showing you that',
        closing: 'Trust what you already know.'
      },
      [CommunicationVoice.SUPPORTIVE_FRIEND]: {
        opening: 'Hey, so {reversed}{cardName} showed up in your {position}, and here\'s what I\'m seeing:',
        connector: 'What this means for you is',
        closing: 'You\'ve got this.'
      },
      [CommunicationVoice.DIRECT_COACH]: {
        opening: '{reversed}{cardName} in the {position} position. Here\'s what you need to know:',
        connector: 'Bottom line:',
        closing: 'Now take action.'
      },
      [CommunicationVoice.GENTLE_NURTURER]: {
        opening: 'Sweetie, {reversed}{cardName} has come through in your {position}, gently reminding you:',
        connector: 'This is inviting you to',
        closing: 'Be gentle with yourself through this.'
      },
      [CommunicationVoice.WISE_MENTOR]: {
        opening: 'The {reversed}{cardName} appears in your {position} as a teacher, offering this wisdom:',
        connector: 'What this lesson brings is',
        closing: 'Reflect on this truth.'
      },
      [CommunicationVoice.PLAYFUL_EXPLORER]: {
        opening: 'Ooh, {reversed}{cardName} in your {position}! Here\'s what\'s up:',
        connector: 'This is your invitation to',
        closing: 'Have fun with this!'
      },
      [CommunicationVoice.BALANCED_SAGE]: {
        opening: '{reversed}{cardName} in the {position} position speaks to:',
        connector: 'This brings both challenge and opportunity:',
        closing: 'Balance is the key.'
      }
    };
  }

  /**
   * Card emoji mapping
   */
  static getCardEmoji(card) {
    const emojiMap = {
      'The Fool': '🌟',
      'The Magician': '✨',
      'The High Priestess': '🌙',
      'The Empress': '🌸',
      'The Emperor': '👑',
      'The Sun': '☀️',
      'The Moon': '🌕',
      'The Star': '⭐',
      'The World': '🌍',
      'Ace': '🎯'
    };

    // Check for specific card names
    for (const [key, emoji] of Object.entries(emojiMap)) {
      if (card.name.includes(key)) {
        return emoji;
      }
    }

    // Default by suit
    if (card.suit === 'wands') return '🔥';
    if (card.suit === 'cups') return '💧';
    if (card.suit === 'swords') return '⚔️';
    if (card.suit === 'pentacles') return '🌿';

    return '✨';
  }

  /**
   * Generate card interpretation adapted to user
   */
  static generateCardInterpretation(card, positionMeaning, isReversed, commProfile, readingType) {
    // Select base meaning
    let baseMeaning;
    switch (readingType) {
      case 'career':
        baseMeaning = card.careerInterpretation;
        break;
      case 'romance':
        baseMeaning = card.romanceInterpretation;
        break;
      case 'wellness':
        baseMeaning = card.wellnessInterpretation;
        break;
      case 'family':
        baseMeaning = card.familyInterpretation;
        break;
      case 'self_growth':
        baseMeaning = card.selfGrowthInterpretation;
        break;
      case 'school':
        baseMeaning = card.schoolInterpretation;
        break;
      default:
        baseMeaning = isReversed ? card.reversedMeaning : card.uprightMeaning;
    }

    // Get voice template
    const templates = this.getVoiceTemplates();
    const template = templates[commProfile.primaryVoice];

    // Build interpretation
    const parts = [];

    // Opening
    const opening = template.opening
      .replace('{position}', positionMeaning)
      .replace('{cardName}', card.name)
      .replace('{reversed}', isReversed ? 'reversed ' : '');
    parts.push(opening);

    // Core meaning
    parts.push(baseMeaning);

    // Psychological insight (if appropriate)
    if (commProfile.therapeuticExplicitness !== 'hidden' && card.psychology) {
      const psychInsight = this.generatePsychologicalInsight(
        card,
        commProfile.primaryVoice,
        commProfile.therapeuticExplicitness
      );
      if (psychInsight) {
        parts.push(psychInsight);
      }
    }

    // Growth prompt (if empowerment > comfort)
    if (commProfile.empowermentVsComfort > 0.5 && card.psychology && card.psychology.growthPrompt) {
      parts.push(card.psychology.growthPrompt);
    }

    // Closing
    if (template.closing) {
      parts.push(template.closing);
    }

    // Add emoji if appropriate
    if (commProfile.emojiUse && this.shouldUseEmoji(commProfile.primaryVoice)) {
      const emoji = this.getCardEmoji(card);
      parts[parts.length - 1] = parts[parts.length - 1] + ' ' + emoji;
    }

    return parts.join(' ');
  }

  static shouldUseEmoji(voice) {
    return [
      CommunicationVoice.PLAYFUL_EXPLORER,
      CommunicationVoice.SUPPORTIVE_FRIEND,
      CommunicationVoice.GENTLE_NURTURER
    ].includes(voice);
  }

  static generatePsychologicalInsight(card, voice, explicitness) {
    if (!card.psychology || !card.psychology.coreMetaphor) {
      return null;
    }

    const metaphor = card.psychology.coreMetaphor;

    if (explicitness === 'hidden') {
      return null;
    } else if (explicitness === 'subtle') {
      // Integrate without naming framework
      if (voice === CommunicationVoice.SUPPORTIVE_FRIEND) {
        return `Kind of like when ${metaphor.toLowerCase()}`;
      } else if (voice === CommunicationVoice.ANALYTICAL_GUIDE) {
        return `This pattern mirrors ${metaphor.toLowerCase()}`;
      }
    } else {
      // Explicit
      const frameworks = [
        ...(card.psychology.dbtSkills || []),
        ...(card.psychology.cbtConcepts || [])
      ].join(', ');
      if (frameworks) {
        return `(Relates to: ${frameworks})`;
      }
    }

    return null;
  }
}

// Example usage
export function testAdaptiveLanguage() {
  console.log('=== Adaptive Language Engine Test ===\n');

  // Mock personality profiles
  const analyticalProfile = {
    analyticalIntuitive: 0.2,  // Very analytical
    emotionalRegulation: 0.7,  // Good regulation
    actionOrientation: 0.6,
    optimismRealism: 0.7,
    riskTolerance: 0.5,
    structureFlexibility: 0.3,  // Structured
    internalExternalLocus: 0.8,
    primaryFramework: 'CBT'
  };

  const intuitiveProfile = {
    analyticalIntuitive: 0.8,  // Very intuitive
    emotionalRegulation: 0.2,  // Needs support
    actionOrientation: 0.3,
    optimismRealism: 0.6,
    riskTolerance: 0.7,
    structureFlexibility: 0.8,  // Flexible
    internalExternalLocus: 0.6,
    primaryFramework: 'DBT'
  };

  const comm1 = AdaptiveLanguageEngine.buildCommunicationProfile(
    analyticalProfile,
    1995
  );

  const comm2 = AdaptiveLanguageEngine.buildCommunicationProfile(
    intuitiveProfile,
    2000
  );

  console.log('Analytical Person:');
  console.log(`  Voice: ${comm1.primaryVoice}`);
  console.log(`  Aesthetic: ${comm1.aesthetic}`);
  console.log(`  Metaphor Density: ${comm1.metaphorDensity}`);

  console.log('\nIntuitive Person:');
  console.log(`  Voice: ${comm2.primaryVoice}`);
  console.log(`  Aesthetic: ${comm2.aesthetic}`);
  console.log(`  Metaphor Density: ${comm2.metaphorDensity}`);

  // Mock card for demonstration
  const mockCard = {
    name: 'The Fool',
    suit: 'major_arcana',
    careerInterpretation: 'A new job or career change beckons. Trust your instincts even without all answers.',
    psychology: {
      coreMetaphor: 'starting therapy: vulnerable but necessary',
      growthPrompt: 'What would you attempt if you knew you couldn\'t fail?',
      dbtSkills: ['Wise Mind', 'Willingness'],
      cbtConcepts: ['Growth Mindset']
    }
  };

  console.log('\n=== Same Card, Different Voices ===\n');

  const interp1 = AdaptiveLanguageEngine.generateCardInterpretation(
    mockCard,
    'Present',
    false,
    comm1,
    'career'
  );

  const interp2 = AdaptiveLanguageEngine.generateCardInterpretation(
    mockCard,
    'Present',
    false,
    comm2,
    'career'
  );

  console.log('Analytical Voice:');
  console.log(interp1);

  console.log('\nIntuitive Voice:');
  console.log(interp2);
}

// Uncomment to test:
// testAdaptiveLanguage();
