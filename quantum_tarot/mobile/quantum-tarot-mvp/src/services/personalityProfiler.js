/**
 * Quantum Tarot - Personality Profiling System (JavaScript)
 * Ported from Python - Runs entirely on phone
 * Analyzes user responses to create psychological profile
 */

/**
 * Response Types
 */
export const ResponseType = {
  MULTIPLE_CHOICE: 'multiple_choice',
  LIKERT_SCALE: 'likert',
  BINARY: 'binary'
};

/**
 * Question definition
 */
export class Question {
  constructor(id, text, responseType, options, measuresTrait, dbtSkill = null, cbtPattern = null, mrtPillar = null) {
    this.id = id;
    this.text = text;
    this.responseType = responseType;
    this.options = options;
    this.measuresTrait = measuresTrait;
    this.dbtSkill = dbtSkill;
    this.cbtPattern = cbtPattern;
    this.mrtPillar = mrtPillar;
  }
}

/**
 * Personality Profile
 */
export class PersonalityProfile {
  constructor(userId, readingType) {
    this.userId = userId;
    this.readingType = readingType;
    this.timestamp = Date.now();

    // Raw responses
    this.responses = {};

    // Calculated traits (0.0 to 1.0)
    this.emotionalRegulation = 0.5;
    this.actionOrientation = 0.5;
    this.internalExternalLocus = 0.5;
    this.optimismRealism = 0.5;
    this.analyticalIntuitive = 0.5;
    this.riskTolerance = 0.5;
    this.socialOrientation = 0.5;
    this.structureFlexibility = 0.5;
    this.pastFutureFocus = 0.5;
    this.avoidanceApproach = 0.5;

    // Derived insights
    this.primaryFramework = null;  // DBT, CBT, MRT, Integrative
    this.interventionStyle = null; // directive, exploratory, supportive

    // Astrological
    this.sunSign = null;
    this.moonSign = null;
    this.risingSign = null;
  }
}

/**
 * Question Bank - All questions for each reading type
 */
export class QuestionBank {
  static getCareerQuestions() {
    return [
      new Question(
        'career_1',
        'When facing a difficult work decision, you typically:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Make a quick decision and adjust if needed',
          'Analyze all options thoroughly before choosing',
          'Seek advice from trusted colleagues',
          'Trust your gut feeling'
        ],
        'action_orientation',
        null,
        'decision_making_style',
        'mental_agility'
      ),

      new Question(
        'career_2',
        'On a scale of 1-5, how much control do you feel over your career path?',
        ResponseType.LIKERT_SCALE,
        ['1 - Very little', '2 - Some', '3 - Moderate', '4 - Quite a bit', '5 - Complete control'],
        'internal_external_locus',
        null,
        'locus_of_control',
        'optimism'
      ),

      new Question(
        'career_3',
        'When you receive critical feedback at work, you:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Feel hurt but try to learn from it',
          'Immediately start planning improvements',
          'Question the feedback giver\'s motives',
          'Need time alone to process your emotions'
        ],
        'emotional_regulation',
        'distress_tolerance',
        'cognitive_reappraisal',
        null
      ),

      new Question(
        'career_4',
        'Your ideal work environment has:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Clear structure and defined expectations',
          'Flexibility to create your own approach',
          'Mix of both structure and freedom',
          'Constantly changing challenges'
        ],
        'structure_flexibility',
        null,
        null,
        'self_regulation'
      ),

      new Question(
        'career_5',
        'When considering a career change, you think most about:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'What hasn\'t worked in the past',
          'Exciting possibilities ahead',
          'Current skills and resources',
          'What others expect of you'
        ],
        'past_future_focus',
        null,
        'temporal_orientation',
        'optimism'
      ),

      new Question(
        'career_6',
        'In team projects, you naturally tend to:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Take charge and organize',
          'Support others\' ideas',
          'Contribute ideas but let others lead',
          'Work independently on your part'
        ],
        'social_orientation',
        null,
        null,
        'relationship_building'
      ),

      new Question(
        'career_7',
        'True or False: I often worry about making the wrong career move.',
        ResponseType.BINARY,
        ['True', 'False'],
        'risk_tolerance',
        'mindfulness',
        'catastrophizing',
        null
      ),

      new Question(
        'career_8',
        'When facing a career challenge, your first instinct is to:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Research and gather information',
          'Take immediate action',
          'Talk it through with someone',
          'Reflect and meditate on it'
        ],
        'analytical_intuitive',
        null,
        'coping_style',
        null
      ),

      new Question(
        'career_9',
        'Rate your agreement: "Success comes from hard work, not luck."',
        ResponseType.LIKERT_SCALE,
        ['1 - Strongly disagree', '2 - Disagree', '3 - Neutral', '4 - Agree', '5 - Strongly agree'],
        'internal_external_locus',
        null,
        'attribution_style',
        'character_strengths'
      ),

      new Question(
        'career_10',
        'When work stress builds up, you cope by:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Pushing through and working harder',
          'Taking breaks for self-care',
          'Venting to friends or family',
          'Avoiding thinking about it'
        ],
        'avoidance_approach',
        'distress_tolerance',
        null,
        'self_regulation'
      )
    ];
  }

  static getRomanceQuestions() {
    return [
      new Question(
        'romance_1',
        'In relationships, you value most:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Emotional intimacy and deep connection',
          'Excitement and passion',
          'Stability and commitment',
          'Independence within togetherness'
        ],
        'social_orientation',
        'interpersonal_effectiveness',
        null,
        null
      ),

      new Question(
        'romance_2',
        'When conflict arises with a partner, you typically:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Address it immediately and directly',
          'Need space before discussing',
          'Try to smooth things over quickly',
          'Analyze what went wrong first'
        ],
        'emotional_regulation',
        'emotion_regulation',
        'conflict_resolution',
        null
      ),

      new Question(
        'romance_3',
        'Rate: "I trust my instincts about potential partners."',
        ResponseType.LIKERT_SCALE,
        ['1 - Not at all', '2 - Slightly', '3 - Moderately', '4 - Quite a bit', '5 - Completely'],
        'analytical_intuitive',
        null,
        'self_trust',
        null
      ),

      new Question(
        'romance_4',
        'You feel most loved when your partner:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Tells you verbally',
          'Shows physical affection',
          'Does helpful things for you',
          'Spends quality time with you'
        ],
        'social_orientation',
        'interpersonal_effectiveness',
        null,
        null
      ),

      new Question(
        'romance_5',
        'In past relationships, you\'ve struggled most with:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Communication breakdowns',
          'Trust issues',
          'Different life goals',
          'Emotional unavailability (yours or theirs)'
        ],
        'past_future_focus',
        'mindfulness',
        'relationship_patterns',
        null
      ),

      new Question(
        'romance_6',
        'True or False: I often sacrifice my needs for relationship harmony.',
        ResponseType.BINARY,
        ['True', 'False'],
        'internal_external_locus',
        'interpersonal_effectiveness',
        'boundaries',
        null
      ),

      new Question(
        'romance_7',
        'When dating, you prefer to:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Take things slowly and see what develops',
          'Know quickly if there\'s potential',
          'Let the other person set the pace',
          'Have clear milestones and timeline'
        ],
        'action_orientation',
        null,
        'relationship_pacing',
        null
      ),

      new Question(
        'romance_8',
        'Rate: "I\'m optimistic about finding/maintaining lasting love."',
        ResponseType.LIKERT_SCALE,
        ['1 - Very pessimistic', '2 - Somewhat pessimistic', '3 - Neutral', '4 - Somewhat optimistic', '5 - Very optimistic'],
        'optimism_realism',
        null,
        'relationship_schemas',
        'optimism'
      ),

      new Question(
        'romance_9',
        'You\'re most attracted to partners who are:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Emotionally expressive and open',
          'Intellectual and thought-provoking',
          'Adventurous and spontaneous',
          'Steady and reliable'
        ],
        'structure_flexibility',
        null,
        null,
        null
      ),

      new Question(
        'romance_10',
        'When single, you:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Actively seek partnership',
          'Focus on yourself and let love find you',
          'Feel incomplete without a partner',
          'Enjoy the freedom and aren\'t in a rush'
        ],
        'avoidance_approach',
        null,
        'relationship_beliefs',
        'character_strengths'
      )
    ];
  }

  static getWellnessQuestions() {
    return [
      new Question(
        'wellness_1',
        'When you feel physically unwell, you first:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Research symptoms and possible causes',
          'Rest and let your body heal naturally',
          'Seek professional medical advice',
          'Try alternative/holistic remedies'
        ],
        'analytical_intuitive',
        null,
        'health_beliefs',
        null
      ),

      new Question(
        'wellness_2',
        'Rate: "My mental health significantly impacts my physical health."',
        ResponseType.LIKERT_SCALE,
        ['1 - Not at all', '2 - Slightly', '3 - Moderately', '4 - Significantly', '5 - Completely'],
        'emotional_regulation',
        'mindfulness',
        'mind_body_connection',
        null
      ),

      new Question(
        'wellness_3',
        'Your self-care routine is best described as:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Consistent and structured',
          'Intuitive based on what I need',
          'Something I struggle to maintain',
          'Non-existent - I put others first'
        ],
        'structure_flexibility',
        'please_skills',
        null,
        'self_regulation'
      ),

      new Question(
        'wellness_4',
        'When stressed, your body tells you through:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Tension/pain (headaches, back pain, etc.)',
          'Digestive issues',
          'Sleep disturbances',
          'I don\'t notice physical signs'
        ],
        'emotional_regulation',
        'mindfulness',
        'interoceptive_awareness',
        null
      ),

      new Question(
        'wellness_5',
        'True or False: I often ignore health warning signs until they become serious.',
        ResponseType.BINARY,
        ['True', 'False'],
        'avoidance_approach',
        null,
        'health_avoidance',
        'self_awareness'
      ),

      new Question(
        'wellness_6',
        'Your relationship with exercise is:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'I love it and do it regularly',
          'I know I should but struggle with consistency',
          'I prefer gentle movement like yoga or walking',
          'Physical limitations prevent regular exercise'
        ],
        'action_orientation',
        'building_mastery',
        'behavioral_activation',
        null
      ),

      new Question(
        'wellness_7',
        'Rate your agreement: "I\'m in tune with what my body needs."',
        ResponseType.LIKERT_SCALE,
        ['1 - Strongly disagree', '2 - Disagree', '3 - Neutral', '4 - Agree', '5 - Strongly agree'],
        'analytical_intuitive',
        'mindfulness',
        'body_awareness',
        null
      ),

      new Question(
        'wellness_8',
        'When making health decisions, you most trust:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Medical professionals and research',
          'Your own intuition and body wisdom',
          'Alternative practitioners and natural approaches',
          'Combination of conventional and alternative'
        ],
        'internal_external_locus',
        null,
        'health_locus_control',
        null
      ),

      new Question(
        'wellness_9',
        'Your biggest wellness challenge is:',
        ResponseType.MULTIPLE_CHOICE,
        [
          'Managing stress and anxiety',
          'Physical health issues or chronic pain',
          'Motivation and consistency',
          'Balancing wellness with other responsibilities'
        ],
        'emotional_regulation',
        null,
        null,
        null
      ),

      new Question(
        'wellness_10',
        'Rate: "I believe I can significantly improve my health through my choices."',
        ResponseType.LIKERT_SCALE,
        ['1 - Not at all', '2 - Slightly', '3 - Moderately', '4 - Quite a bit', '5 - Completely'],
        'optimism_realism',
        null,
        'self_efficacy',
        'optimism'
      )
    ];
  }

  static getQuestionsForType(readingType) {
    switch (readingType) {
      case 'career':
        return this.getCareerQuestions();
      case 'romance':
        return this.getRomanceQuestions();
      case 'wellness':
        return this.getWellnessQuestions();
      // Add other types as needed
      default:
        return this.getCareerQuestions(); // Default
    }
  }
}

/**
 * Personality Analyzer
 * Calculates traits from responses
 */
export class PersonalityAnalyzer {
  /**
   * Calculate complete profile from responses
   */
  static calculateProfile(readingType, responses, birthday = null, name = null) {
    const profile = new PersonalityProfile(name || 'user', readingType);
    profile.responses = responses;

    // Get questions for this type
    const questions = QuestionBank.getQuestionsForType(readingType);

    // Initialize trait accumulators
    const traitScores = {
      emotional_regulation: [],
      action_orientation: [],
      internal_external_locus: [],
      optimism_realism: [],
      analytical_intuitive: [],
      risk_tolerance: [],
      social_orientation: [],
      structure_flexibility: [],
      past_future_focus: [],
      avoidance_approach: []
    };

    // Process each response
    questions.forEach(question => {
      if (!responses[question.id]) return;

      const response = responses[question.id];
      const trait = question.measuresTrait;

      if (!traitScores[trait]) return;

      // Score based on response type
      let score = null;

      if (question.responseType === ResponseType.LIKERT_SCALE) {
        // Extract numeric value (1-5) -> normalize to 0-1
        const match = response.match(/^(\d)/);
        if (match) {
          score = parseInt(match[1]) / 5.0;
        }
      } else if (question.responseType === ResponseType.MULTIPLE_CHOICE) {
        // Map position in options to score
        const index = question.options.indexOf(response);
        if (index !== -1) {
          score = index / (question.options.length - 1);
        }
      } else if (question.responseType === ResponseType.BINARY) {
        // True/False -> 0 or 1
        score = response.toLowerCase() === 'true' ? 1.0 : 0.0;

        // Reverse score for negative questions
        if (question.text.toLowerCase().includes('worry') ||
            question.text.toLowerCase().includes('sacrifice') ||
            question.text.toLowerCase().includes('ignore')) {
          score = 1.0 - score;
        }
      }

      if (score !== null) {
        traitScores[trait].push(score);
      }
    });

    // Average scores for each trait
    Object.keys(traitScores).forEach(trait => {
      const scores = traitScores[trait];
      if (scores.length > 0) {
        const avg = scores.reduce((sum, val) => sum + val, 0) / scores.length;
        profile[this.traitToCamelCase(trait)] = avg;
      }
    });

    // Identify primary framework and intervention style
    profile.primaryFramework = this.identifyFramework(profile);
    profile.interventionStyle = this.identifyInterventionStyle(profile);

    // Calculate astrological signs if birthday provided
    if (birthday) {
      const signs = this.calculateAstrologicalSigns(birthday);
      profile.sunSign = signs.sunSign;
      profile.moonSign = signs.moonSign;
      profile.risingSign = signs.risingSign;
    }

    return profile;
  }

  static traitToCamelCase(trait) {
    return trait.replace(/_([a-z])/g, (g) => g[1].toUpperCase());
  }

  static identifyFramework(profile) {
    // DBT: High emotional dysregulation, relationship issues
    if (profile.emotionalRegulation < 0.4 && profile.socialOrientation > 0.6) {
      return 'DBT';
    }
    // CBT: Analytical style, thought-focused
    if (profile.analyticalIntuitive < 0.4) {
      return 'CBT';
    }
    // MRT: Action-oriented, resilience focus
    if (profile.actionOrientation > 0.6 && profile.optimismRealism > 0.5) {
      return 'MRT';
    }
    return 'Integrative';
  }

  static identifyInterventionStyle(profile) {
    if (profile.structureFlexibility < 0.4) {
      return 'directive';  // Clear guidance needed
    }
    if (profile.analyticalIntuitive < 0.4) {
      return 'exploratory';  // Questions, self-discovery
    }
    return 'supportive';  // Validate, encourage
  }

  static calculateAstrologicalSigns(birthdayString) {
    // Parse birthday (format: YYYY-MM-DD or ISO string)
    const date = new Date(birthdayString);
    const month = date.getMonth() + 1; // JavaScript months are 0-indexed
    const day = date.getDate();

    const sunSign = this.getSunSign(month, day);

    // For moon and rising, we'd need birth time and location
    // For now, return null (can add later)
    return {
      sunSign,
      moonSign: null,
      risingSign: null
    };
  }

  static getSunSign(month, day) {
    if ((month === 3 && day >= 21) || (month === 4 && day <= 19)) return 'Aries';
    if ((month === 4 && day >= 20) || (month === 5 && day <= 20)) return 'Taurus';
    if ((month === 5 && day >= 21) || (month === 6 && day <= 20)) return 'Gemini';
    if ((month === 6 && day >= 21) || (month === 7 && day <= 22)) return 'Cancer';
    if ((month === 7 && day >= 23) || (month === 8 && day <= 22)) return 'Leo';
    if ((month === 8 && day >= 23) || (month === 9 && day <= 22)) return 'Virgo';
    if ((month === 9 && day >= 23) || (month === 10 && day <= 22)) return 'Libra';
    if ((month === 10 && day >= 23) || (month === 11 && day <= 21)) return 'Scorpio';
    if ((month === 11 && day >= 22) || (month === 12 && day <= 21)) return 'Sagittarius';
    if ((month === 12 && day >= 22) || (month === 1 && day <= 19)) return 'Capricorn';
    if ((month === 1 && day >= 20) || (month === 2 && day <= 18)) return 'Aquarius';
    return 'Pisces'; // Feb 19 - Mar 20
  }
}

// Example usage
export function testPersonalityProfiler() {
  console.log('=== Personality Profiler Test ===\n');

  const responses = {
    'career_1': 'Analyze all options thoroughly before choosing',
    'career_2': '4 - Quite a bit',
    'career_3': 'Immediately start planning improvements',
    'career_4': 'Mix of both structure and freedom',
    'career_5': 'Exciting possibilities ahead',
    'career_6': 'Take charge and organize',
    'career_7': 'False',
    'career_8': 'Research and gather information',
    'career_9': '5 - Strongly agree',
    'career_10': 'Taking breaks for self-care'
  };

  const profile = PersonalityAnalyzer.calculateProfile(
    'career',
    responses,
    '1990-06-15',
    'Test User'
  );

  console.log(`User: ${profile.userId}`);
  console.log(`Reading Type: ${profile.readingType}`);
  console.log(`Sun Sign: ${profile.sunSign}\n`);
  console.log(`Primary Framework: ${profile.primaryFramework}`);
  console.log(`Intervention Style: ${profile.interventionStyle}\n`);
  console.log('Trait Scores (0-1 scale):');
  console.log(`  Emotional Regulation: ${profile.emotionalRegulation.toFixed(2)}`);
  console.log(`  Action Orientation: ${profile.actionOrientation.toFixed(2)}`);
  console.log(`  Internal Locus: ${profile.internalExternalLocus.toFixed(2)}`);
  console.log(`  Optimism: ${profile.optimismRealism.toFixed(2)}`);
  console.log(`  Analytical: ${profile.analyticalIntuitive.toFixed(2)}`);
}

// Uncomment to test:
// testPersonalityProfiler();
