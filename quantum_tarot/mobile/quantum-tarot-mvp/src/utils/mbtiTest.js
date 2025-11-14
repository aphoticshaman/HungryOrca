/**
 * MBTI PERSONALITY TEST
 * 16 Personalities integration for hyper-personalized tarot interpretations
 *
 * Determines user's MBTI type across 4 dimensions:
 * - E/I: Extraversion vs Introversion
 * - S/N: Sensing vs Intuition
 * - T/F: Thinking vs Feeling
 * - J/P: Judging vs Perceiving
 *
 * 40 questions (10 per dimension) for accurate typing
 */

/**
 * MBTI Question Battery
 * Each question maps to one of the 4 dimensions
 */
export const MBTI_QUESTIONS = [
  // ═══════════════════════════════════════════════════════════
  // EXTRAVERSION (E) vs INTROVERSION (I) - 10 questions
  // ═══════════════════════════════════════════════════════════
  {
    id: 'ei_1',
    dimension: 'EI',
    question: 'After a long day, you prefer to:',
    options: [
      { text: 'Go out with friends or attend social events', score: 2, pole: 'E' },
      { text: 'Spend quiet time alone to recharge', score: -2, pole: 'I' },
      { text: 'Depends on my energy level', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_2',
    dimension: 'EI',
    question: 'In group settings, you typically:',
    options: [
      { text: 'Actively participate and energize the conversation', score: 2, pole: 'E' },
      { text: 'Observe and contribute selectively', score: -2, pole: 'I' },
      { text: 'Mix of both depending on the topic', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_3',
    dimension: 'EI',
    question: 'You gain clarity and process ideas best by:',
    options: [
      { text: 'Talking them through with others', score: 2, pole: 'E' },
      { text: 'Reflecting internally before sharing', score: -2, pole: 'I' },
      { text: 'Combination of both', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_4',
    dimension: 'EI',
    question: 'Your ideal weekend involves:',
    options: [
      { text: 'Multiple social activities and being around people', score: 2, pole: 'E' },
      { text: 'Solo hobbies, reading, or quiet personal projects', score: -2, pole: 'I' },
      { text: 'One social event plus downtime', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_5',
    dimension: 'EI',
    question: 'When meeting new people, you:',
    options: [
      { text: 'Easily strike up conversations and feel energized', score: 2, pole: 'E' },
      { text: 'Wait for them to approach or need warm-up time', score: -2, pole: 'I' },
      { text: 'Comfortable but not overly enthusiastic', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_6',
    dimension: 'EI',
    question: 'Your social circle is typically:',
    options: [
      { text: 'Wide network of many friends and acquaintances', score: 2, pole: 'E' },
      { text: 'Small group of very close, deep friendships', score: -2, pole: 'I' },
      { text: 'Mix of close friends and casual connections', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_7',
    dimension: 'EI',
    question: 'During conflicts or stress, you prefer to:',
    options: [
      { text: 'Talk it out immediately with someone', score: 2, pole: 'E' },
      { text: 'Retreat and process alone first', score: -2, pole: 'I' },
      { text: 'Depends on the situation', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_8',
    dimension: 'EI',
    question: 'You feel most alive when:',
    options: [
      { text: 'Surrounded by action, people, and stimulation', score: 2, pole: 'E' },
      { text: 'In peaceful solitude or intimate one-on-one time', score: -2, pole: 'I' },
      { text: 'Balance of both', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_9',
    dimension: 'EI',
    question: 'Your communication style is:',
    options: [
      { text: 'Think out loud, verbal processing', score: 2, pole: 'E' },
      { text: 'Think deeply first, then speak concisely', score: -2, pole: 'I' },
      { text: 'Varies by context', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'ei_10',
    dimension: 'EI',
    question: 'After socializing for hours, you typically feel:',
    options: [
      { text: 'Energized and wanting more interaction', score: 2, pole: 'E' },
      { text: 'Drained and needing to recharge alone', score: -2, pole: 'I' },
      { text: 'Satisfied but ready for quiet', score: 0, pole: 'neutral' }
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // SENSING (S) vs INTUITION (N) - 10 questions
  // ═══════════════════════════════════════════════════════════
  {
    id: 'sn_1',
    dimension: 'SN',
    question: 'When solving problems, you focus on:',
    options: [
      { text: 'Concrete facts, proven methods, what worked before', score: 2, pole: 'S' },
      { text: 'Patterns, possibilities, innovative approaches', score: -2, pole: 'N' },
      { text: 'Mix of practical and creative', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_2',
    dimension: 'SN',
    question: 'You trust:',
    options: [
      { text: 'Direct experience and tangible evidence', score: 2, pole: 'S' },
      { text: 'Gut feelings and theoretical frameworks', score: -2, pole: 'N' },
      { text: 'Both equally', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_3',
    dimension: 'SN',
    question: 'When learning something new, you prefer:',
    options: [
      { text: 'Step-by-step instructions with practical examples', score: 2, pole: 'S' },
      { text: 'Understanding the big picture and theory first', score: -2, pole: 'N' },
      { text: 'Combination approach', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_4',
    dimension: 'SN',
    question: 'Your conversations tend to focus on:',
    options: [
      { text: 'Real events, practical matters, specific details', score: 2, pole: 'S' },
      { text: 'Ideas, concepts, future possibilities', score: -2, pole: 'N' },
      { text: 'Depends on the topic', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_5',
    dimension: 'SN',
    question: 'You\'re more interested in:',
    options: [
      { text: 'What is actually happening right now', score: 2, pole: 'S' },
      { text: 'What could be or might happen', score: -2, pole: 'N' },
      { text: 'Both present and future', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_6',
    dimension: 'SN',
    question: 'When reading, you prefer:',
    options: [
      { text: 'Clear, literal, practical information', score: 2, pole: 'S' },
      { text: 'Metaphorical, symbolic, theoretical content', score: -2, pole: 'N' },
      { text: 'Mix of both styles', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_7',
    dimension: 'SN',
    question: 'You remember events as:',
    options: [
      { text: 'Accurate details of what actually happened', score: 2, pole: 'S' },
      { text: 'Overall impressions and meanings', score: -2, pole: 'N' },
      { text: 'Some details, some impressions', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_8',
    dimension: 'SN',
    question: 'Your ideal work involves:',
    options: [
      { text: 'Tangible results, hands-on tasks, proven processes', score: 2, pole: 'S' },
      { text: 'Innovation, strategy, exploring new ideas', score: -2, pole: 'N' },
      { text: 'Balance of routine and creativity', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_9',
    dimension: 'SN',
    question: 'When someone asks "How was your day?" you tend to:',
    options: [
      { text: 'Describe specific events chronologically', score: 2, pole: 'S' },
      { text: 'Give overall impression or meaning', score: -2, pole: 'N' },
      { text: 'Varies', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'sn_10',
    dimension: 'SN',
    question: 'You\'re drawn to:',
    options: [
      { text: 'Realistic, practical, applicable knowledge', score: 2, pole: 'S' },
      { text: 'Abstract, theoretical, visionary concepts', score: -2, pole: 'N' },
      { text: 'Both depending on context', score: 0, pole: 'neutral' }
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // THINKING (T) vs FEELING (F) - 10 questions
  // ═══════════════════════════════════════════════════════════
  {
    id: 'tf_1',
    dimension: 'TF',
    question: 'When making decisions, you prioritize:',
    options: [
      { text: 'Logical analysis and objective criteria', score: 2, pole: 'T' },
      { text: 'How it affects people and harmony', score: -2, pole: 'F' },
      { text: 'Both logic and feelings', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_2',
    dimension: 'TF',
    question: 'In disagreements, you value:',
    options: [
      { text: 'Being right and finding the truth', score: 2, pole: 'T' },
      { text: 'Maintaining the relationship and understanding', score: -2, pole: 'F' },
      { text: 'Depends on the situation', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_3',
    dimension: 'TF',
    question: 'People describe you as:',
    options: [
      { text: 'Logical, fair, direct', score: 2, pole: 'T' },
      { text: 'Compassionate, warm, empathetic', score: -2, pole: 'F' },
      { text: 'Mix of both', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_4',
    dimension: 'TF',
    question: 'When someone is upset, your instinct is to:',
    options: [
      { text: 'Help them solve the problem logically', score: 2, pole: 'T' },
      { text: 'Offer emotional support and understanding', score: -2, pole: 'F' },
      { text: 'Ask what they need', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_5',
    dimension: 'TF',
    question: 'You make choices based more on:',
    options: [
      { text: 'Head (rational analysis)', score: 2, pole: 'T' },
      { text: 'Heart (personal values)', score: -2, pole: 'F' },
      { text: 'Integrated approach', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_6',
    dimension: 'TF',
    question: 'In your work, you\'re motivated by:',
    options: [
      { text: 'Competence, efficiency, achievement', score: 2, pole: 'T' },
      { text: 'Helping others, making a difference, connection', score: -2, pole: 'F' },
      { text: 'Both results and relationships', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_7',
    dimension: 'TF',
    question: 'You prefer feedback that is:',
    options: [
      { text: 'Honest and direct, even if critical', score: 2, pole: 'T' },
      { text: 'Gentle and considerate of feelings', score: -2, pole: 'F' },
      { text: 'Honest but tactful', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_8',
    dimension: 'TF',
    question: 'When evaluating ideas, you focus on:',
    options: [
      { text: 'Logical consistency and effectiveness', score: 2, pole: 'T' },
      { text: 'Human impact and values alignment', score: -2, pole: 'F' },
      { text: 'Both criteria', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_9',
    dimension: 'TF',
    question: 'You\'re more comfortable with:',
    options: [
      { text: 'Debate and intellectual sparring', score: 2, pole: 'T' },
      { text: 'Collaborative discussion and consensus', score: -2, pole: 'F' },
      { text: 'Depends on context', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'tf_10',
    dimension: 'TF',
    question: 'Criticism affects you by:',
    options: [
      { text: 'Analyzing if it\'s accurate and useful', score: 2, pole: 'T' },
      { text: 'Feeling hurt even if logically justified', score: -2, pole: 'F' },
      { text: 'Both reactions', score: 0, pole: 'neutral' }
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // JUDGING (J) vs PERCEIVING (P) - 10 questions
  // ═══════════════════════════════════════════════════════════
  {
    id: 'jp_1',
    dimension: 'JP',
    question: 'Your living/work space is typically:',
    options: [
      { text: 'Organized, structured, everything has a place', score: 2, pole: 'J' },
      { text: 'Flexible, adaptable, organized chaos', score: -2, pole: 'P' },
      { text: 'Organized in some areas, flexible in others', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_2',
    dimension: 'JP',
    question: 'You prefer to:',
    options: [
      { text: 'Plan ahead and stick to the plan', score: 2, pole: 'J' },
      { text: 'Keep options open and adapt as you go', score: -2, pole: 'P' },
      { text: 'Loose plan with flexibility', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_3',
    dimension: 'JP',
    question: 'Deadlines make you feel:',
    options: [
      { text: 'Motivated - I finish early or on time', score: 2, pole: 'J' },
      { text: 'Pressured - I work best under time pressure', score: -2, pole: 'P' },
      { text: 'Neutral - depends on the task', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_4',
    dimension: 'JP',
    question: 'Your approach to projects is:',
    options: [
      { text: 'Create detailed plan, execute systematically', score: 2, pole: 'J' },
      { text: 'Dive in and figure it out as I go', score: -2, pole: 'P' },
      { text: 'General plan with room for improvisation', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_5',
    dimension: 'JP',
    question: 'You feel best when:',
    options: [
      { text: 'Things are decided, settled, complete', score: 2, pole: 'J' },
      { text: 'Options remain open, things are flexible', score: -2, pole: 'P' },
      { text: 'Balance of closure and flexibility', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_6',
    dimension: 'JP',
    question: 'Your calendar/schedule is:',
    options: [
      { text: 'Detailed and structured', score: 2, pole: 'J' },
      { text: 'Loose or nonexistent', score: -2, pole: 'P' },
      { text: 'Some structure, some spontaneity', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_7',
    dimension: 'JP',
    question: 'When traveling, you:',
    options: [
      { text: 'Create detailed itinerary with reservations', score: 2, pole: 'J' },
      { text: 'Book flight and figure out rest spontaneously', score: -2, pole: 'P' },
      { text: 'Book key things, leave room for adventure', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_8',
    dimension: 'JP',
    question: 'Unfinished tasks make you feel:',
    options: [
      { text: 'Anxious - I need to complete them ASAP', score: 2, pole: 'J' },
      { text: 'Fine - I\'ll get to them when I get to them', score: -2, pole: 'P' },
      { text: 'Slightly bothered but manageable', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_9',
    dimension: 'JP',
    question: 'You work best with:',
    options: [
      { text: 'Clear structure, timelines, and expectations', score: 2, pole: 'J' },
      { text: 'Freedom to explore and adapt', score: -2, pole: 'P' },
      { text: 'Structure with some flexibility', score: 0, pole: 'neutral' }
    ]
  },
  {
    id: 'jp_10',
    dimension: 'JP',
    question: 'Your decision-making style is:',
    options: [
      { text: 'Decide quickly to create closure', score: 2, pole: 'J' },
      { text: 'Keep gathering info and delaying decision', score: -2, pole: 'P' },
      { text: 'Decide when I have enough information', score: 0, pole: 'neutral' }
    ]
  }
];

/**
 * Calculate MBTI type from answers
 * @param {Array} answers - Array of { questionId, selectedOption } objects
 * @returns {Object} - { type: 'INTJ', scores: { E: -12, S: 4, T: 10, J: 8 }, ... }
 */
export function calculateMBTI(answers) {
  // Initialize dimension scores
  const scores = {
    EI: 0,  // Positive = E, Negative = I
    SN: 0,  // Positive = S, Negative = N
    TF: 0,  // Positive = T, Negative = F
    JP: 0   // Positive = J, Negative = P
  };

  // Tally scores
  answers.forEach(answer => {
    const question = MBTI_QUESTIONS.find(q => q.id === answer.questionId);
    if (!question) return;

    const option = question.options[answer.selectedOptionIndex];
    if (!option) return;

    scores[question.dimension] += option.score;
  });

  // Determine type letters
  const type =
    (scores.EI >= 0 ? 'E' : 'I') +
    (scores.SN >= 0 ? 'S' : 'N') +
    (scores.TF >= 0 ? 'T' : 'F') +
    (scores.JP >= 0 ? 'J' : 'P');

  // Calculate strength of preferences (0-100%)
  const maxScore = 20; // 10 questions * 2 points max per question
  const strengths = {
    EI: Math.abs(scores.EI) / maxScore * 100,
    SN: Math.abs(scores.SN) / maxScore * 100,
    TF: Math.abs(scores.TF) / maxScore * 100,
    JP: Math.abs(scores.JP) / maxScore * 100
  };

  return {
    type,
    scores,
    strengths,
    description: getMBTIDescription(type)
  };
}

/**
 * Get MBTI type description and tarot interpretation style
 */
function getMBTIDescription(type) {
  const descriptions = {
    // Analysts
    INTJ: {
      nickname: 'The Architect',
      core: 'Strategic, independent, visionary thinker',
      tarotStyle: 'Prefers deep symbolic analysis, long-term patterns, systems thinking. Wants interpretations that challenge them intellectually and offer strategic insights.',
      communicationPreference: 'Direct, precise, focuses on underlying patterns and future implications'
    },
    INTP: {
      nickname: 'The Logician',
      core: 'Analytical, curious, theory-focused',
      tarotStyle: 'Loves exploring abstract connections, philosophical depth, logical frameworks. Wants interpretations that satisfy intellectual curiosity and reveal hidden structures.',
      communicationPreference: 'Theoretical, exploratory, enjoys complexity and nuance'
    },
    ENTJ: {
      nickname: 'The Commander',
      core: 'Decisive, strategic, leadership-oriented',
      tarotStyle: 'Action-focused, wants clear strategy and concrete steps. Values efficiency and results-oriented guidance.',
      communicationPreference: 'Bold, direct, focused on goals and execution'
    },
    ENTP: {
      nickname: 'The Debater',
      core: 'Innovative, argumentative, possibility-seeker',
      tarotStyle: 'Loves exploring multiple perspectives, alternative interpretations, creative possibilities. Wants interpretations that challenge assumptions.',
      communicationPreference: 'Playful, provocative, enjoys intellectual sparring'
    },

    // Diplomats
    INFJ: {
      nickname: 'The Advocate',
      core: 'Idealistic, empathetic, purpose-driven',
      tarotStyle: 'Seeks deep meaning, spiritual significance, personal growth insights. Wants interpretations connecting to life purpose and authenticity.',
      communicationPreference: 'Compassionate, visionary, focuses on meaning and transformation'
    },
    INFP: {
      nickname: 'The Mediator',
      core: 'Idealistic, values-driven, introspective',
      tarotStyle: 'Emotional resonance, authenticity, personal values alignment. Wants interpretations honoring feelings and inner truth.',
      communicationPreference: 'Gentle, poetic, focuses on emotions and values'
    },
    ENFJ: {
      nickname: 'The Protagonist',
      core: 'Charismatic, inspiring, people-focused',
      tarotStyle: 'Relationship dynamics, empowering others, social harmony. Wants interpretations supporting connection and positive impact.',
      communicationPreference: 'Warm, encouraging, focuses on potential and relationships'
    },
    ENFP: {
      nickname: 'The Campaigner',
      core: 'Enthusiastic, creative, free-spirited',
      tarotStyle: 'Possibilities, creative expression, authentic self-discovery. Wants interpretations that inspire and liberate.',
      communicationPreference: 'Enthusiastic, imaginative, celebrates uniqueness'
    },

    // Sentinels
    ISTJ: {
      nickname: 'The Logistician',
      core: 'Practical, reliable, detail-oriented',
      tarotStyle: 'Clear, practical guidance with specific steps. Values tradition and proven methods.',
      communicationPreference: 'Factual, organized, focuses on concrete details and duty'
    },
    ISFJ: {
      nickname: 'The Defender',
      core: 'Caring, loyal, tradition-oriented',
      tarotStyle: 'Supportive, nurturing guidance focused on caring for self and others. Values stability and kindness.',
      communicationPreference: 'Warm, practical, focuses on caregiving and responsibility'
    },
    ESTJ: {
      nickname: 'The Executive',
      core: 'Organized, efficient, takes charge',
      tarotStyle: 'Direct action plans, clear structure, efficiency. Wants no-nonsense practical guidance.',
      communicationPreference: 'Direct, structured, focuses on logic and results'
    },
    ESFJ: {
      nickname: 'The Consul',
      core: 'Social, helpful, harmony-seeking',
      tarotStyle: 'Relationship guidance, community harmony, helping others. Wants warm, supportive interpretations.',
      communicationPreference: 'Friendly, supportive, focuses on social connection'
    },

    // Explorers
    ISTP: {
      nickname: 'The Virtuoso',
      core: 'Practical, hands-on, flexible',
      tarotStyle: 'Concrete, action-oriented, focused on immediate practical application. Minimal fluff.',
      communicationPreference: 'Concise, practical, focuses on what works'
    },
    ISFP: {
      nickname: 'The Adventurer',
      core: 'Artistic, spontaneous, present-focused',
      tarotStyle: 'Aesthetic, experiential, honors present moment feelings. Wants authentic, sensory-rich interpretations.',
      communicationPreference: 'Gentle, artistic, focuses on beauty and experience'
    },
    ESTP: {
      nickname: 'The Entrepreneur',
      core: 'Bold, energetic, action-taker',
      tarotStyle: 'Fast-paced, opportunity-focused, risk-taking guidance. Wants immediate actionable insights.',
      communicationPreference: 'Bold, direct, focuses on seizing opportunities'
    },
    ESFP: {
      nickname: 'The Entertainer',
      core: 'Enthusiastic, fun-loving, spontaneous',
      tarotStyle: 'Playful, experiential, celebrates joy and connection. Wants uplifting, present-focused guidance.',
      communicationPreference: 'Enthusiastic, warm, focuses on fun and experience'
    }
  };

  return descriptions[type] || {
    nickname: 'Unique Individual',
    core: 'Multifaceted personality',
    tarotStyle: 'Balanced interpretation approach',
    communicationPreference: 'Adaptable communication'
  };
}

/**
 * Get MBTI-specific interpretation variations
 * Used to customize language/focus for each type
 */
export function getMBTIInterpretationGuidelines(mbtiType) {
  const guidelines = {
    // Analysts - Want depth, systems, strategy
    INTJ: {
      emphasize: ['long-term patterns', 'strategic implications', 'systems thinking', 'mastery'],
      avoid: ['emotional appeals', 'vague platitudes', 'surface-level advice'],
      tone: 'intellectual, strategic, future-focused'
    },
    INTP: {
      emphasize: ['logical frameworks', 'theoretical connections', 'abstract patterns', 'curiosity'],
      avoid: ['rigid action plans', 'emotional pressure', 'oversimplification'],
      tone: 'analytical, exploratory, complex'
    },
    ENTJ: {
      emphasize: ['clear goals', 'actionable strategy', 'efficiency', 'leadership'],
      avoid: ['indecisiveness', 'excessive emotion', 'lack of direction'],
      tone: 'decisive, bold, results-oriented'
    },
    ENTP: {
      emphasize: ['multiple perspectives', 'innovation', 'possibilities', 'debate'],
      avoid: ['rigid rules', 'single interpretations', 'lack of creativity'],
      tone: 'playful, provocative, expansive'
    },

    // Diplomats - Want meaning, growth, authenticity
    INFJ: {
      emphasize: ['deeper meaning', 'personal growth', 'authenticity', 'vision'],
      avoid: ['superficiality', 'inauthentic positivity', 'ignoring intuition'],
      tone: 'meaningful, visionary, transformative'
    },
    INFP: {
      emphasize: ['values alignment', 'emotional truth', 'authenticity', 'ideals'],
      avoid: ['harsh criticism', 'forcing practicality', 'dismissing feelings'],
      tone: 'gentle, authentic, values-centered'
    },
    ENFJ: {
      emphasize: ['relationship harmony', 'helping others', 'potential', 'connection'],
      avoid: ['selfishness', 'conflict', 'isolation'],
      tone: 'warm, encouraging, people-focused'
    },
    ENFP: {
      emphasize: ['possibilities', 'authenticity', 'creativity', 'inspiration'],
      avoid: ['rigid structure', 'crushing spontaneity', 'limiting options'],
      tone: 'enthusiastic, imaginative, liberating'
    },

    // Sentinels - Want structure, duty, practicality
    ISTJ: {
      emphasize: ['concrete steps', 'practical application', 'duty', 'tradition'],
      avoid: ['abstract theory', 'impractical advice', 'chaos'],
      tone: 'clear, organized, reliable'
    },
    ISFJ: {
      emphasize: ['caring for others', 'stability', 'responsibility', 'support'],
      avoid: ['selfishness', 'disruption', 'harsh criticism'],
      tone: 'warm, nurturing, practical'
    },
    ESTJ: {
      emphasize: ['efficiency', 'structure', 'clear plans', 'logic'],
      avoid: ['disorganization', 'emotional excess', 'impracticality'],
      tone: 'direct, organized, no-nonsense'
    },
    ESFJ: {
      emphasize: ['social harmony', 'helping others', 'tradition', 'community'],
      avoid: ['conflict', 'selfishness', 'social rejection'],
      tone: 'friendly, supportive, conventional'
    },

    // Explorers - Want action, experience, flexibility
    ISTP: {
      emphasize: ['practical action', 'hands-on solutions', 'flexibility', 'efficiency'],
      avoid: ['emotional drama', 'abstract theory', 'rigid plans'],
      tone: 'concise, practical, action-focused'
    },
    ISFP: {
      emphasize: ['present experience', 'aesthetics', 'authenticity', 'sensory'],
      avoid: ['harsh judgment', 'rigid structure', 'dismissing feelings'],
      tone: 'gentle, experiential, authentic'
    },
    ESTP: {
      emphasize: ['immediate action', 'opportunities', 'risk-taking', 'results'],
      avoid: ['over-planning', 'analysis paralysis', 'waiting'],
      tone: 'bold, fast-paced, opportunistic'
    },
    ESFP: {
      emphasize: ['joy', 'experience', 'connection', 'spontaneity'],
      avoid: ['heavy negativity', 'rigid plans', 'missing fun'],
      tone: 'enthusiastic, experiential, joyful'
    }
  };

  return guidelines[mbtiType] || {
    emphasize: ['balance', 'clarity', 'actionable insights'],
    avoid: ['extremes', 'vagueness'],
    tone: 'balanced, clear, supportive'
  };
}
