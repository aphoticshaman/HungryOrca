/**
 * INTENTION VALIDATOR: 5W+H Analysis
 * Validates user intentions for depth and specificity
 * Provides feedback when context is insufficient
 */

/**
 * Validate intention using 5W+H framework
 * Who, What, When, Where, Why, How
 *
 * @param {string} intention - User's typed intention
 * @returns {Object} - { valid, score, feedback, missing, details }
 */
export function validateIntention(intention) {
  if (!intention || intention.trim().length === 0) {
    return {
      valid: false,
      score: 0,
      feedback: "No intention provided. Type your question or situation to receive guidance.",
      missing: ['who', 'what', 'when', 'where', 'why', 'how'],
      details: {}
    };
  }

  const text = intention.toLowerCase();
  const analysis = {
    who: analyzeWho(text, intention),
    what: analyzeWhat(text, intention),
    when: analyzeWhen(text, intention),
    where: analyzeWhere(text, intention),
    why: analyzeWhy(text, intention),
    how: analyzeHow(text, intention)
  };

  // Calculate present elements
  const present = Object.entries(analysis).filter(([_, val]) => val.present).map(([key]) => key);
  const missing = Object.entries(analysis).filter(([_, val]) => !val.present).map(([key]) => key);

  const score = present.length / 6; // 0-1 scale
  const valid = score >= 0.33; // Need at least 2/6 elements (Who/What are most critical)

  // Generate feedback
  const feedback = generateFeedback(analysis, present, missing, text);

  return {
    valid,
    score,
    feedback,
    missing,
    present,
    details: analysis
  };
}

/**
 * WHO: Identify subjects (people, entities, relationships)
 */
function analyzeWho(text, original) {
  const whoPatterns = [
    /\b(i|me|my|myself)\b/,
    /\b(we|us|our)\b/,
    /\b(he|she|they|him|her|them)\b/,
    // Romantic/intimate relationships
    /\b(partner|boyfriend|girlfriend|husband|wife|spouse|fianc[eé]|lover)\b/,
    /\b(ex|ex-girlfriend|ex-boyfriend|ex-wife|ex-husband|ex-partner)\b/,
    /\b(dating|crush|hookup|situationship)\b/,
    // Professional relationships
    /\b(boss|manager|supervisor|coworker|colleague|employee|subordinate)\b/,
    /\b(client|customer|vendor|supplier|contractor|freelancer)\b/,
    /\b(potential customer|prospective client|lead)\b/,
    /\b(mentor|coach|advisor|consultant|therapist|counselor)\b/,
    /\b(business partner|investor|stakeholder|shareholder)\b/,
    // Family relationships
    /\b(mother|mom|father|dad|parent|parents|stepmother|stepfather|step-parent)\b/,
    /\b(son|daughter|child|children|kid|kids|stepson|stepdaughter|stepchild)\b/,
    /\b(sibling|sister|brother|half-sister|half-brother|stepsister|stepbrother)\b/,
    /\b(grandmother|grandma|grandfather|grandpa|grandparent|grandparents)\b/,
    /\b(grandchild|grandchildren|grandson|granddaughter)\b/,
    /\b(aunt|auntie|uncle|niece|nephew)\b/,
    /\b(cousin|cousins)\b/,
    /\b(in-law|mother-in-law|father-in-law|sister-in-law|brother-in-law)\b/,
    /\b(family|relatives|extended family)\b/,
    // Social/community
    /\b(friend|best friend|close friend|old friend|childhood friend)\b/,
    /\b(acquaintance|neighbor|roommate|housemate)\b/,
    /\b(enemy|rival|competitor|adversary)\b/,
    /\b(stranger|someone|person|people)\b/,
    // Service/authority
    /\b(doctor|physician|nurse|dentist|therapist|psychiatrist|psychologist)\b/,
    /\b(teacher|professor|instructor|tutor)\b/,
    /\b(lawyer|attorney|accountant|financial advisor)\b/,
    /\b(landlord|tenant|property manager)\b/,
    /\b(police|officer|authority)\b/,
    // Contact status
    /\b(no contact|low contact|estranged)\b/,
    /\b(stay friends|stayed friends|friendly ex)\b/,
    /[A-Z][a-z]+\b/ // Proper names (must be last to avoid false positives)
  ];

  const matches = whoPatterns.filter(pattern => pattern.test(text));

  return {
    present: matches.length > 0,
    strength: matches.length > 1 ? 'strong' : matches.length === 1 ? 'weak' : 'none',
    details: matches.length > 0 ? 'Subject identified' : 'No clear subject'
  };
}

/**
 * WHAT: Identify action, situation, decision, or topic
 */
function analyzeWhat(text, original) {
  const whatPatterns = [
    /\b(should|can|could|would|will)\s+(i|we)\s+\w+/,
    /\b(quit|leave|start|join|move|apply|ask|tell|confront|end|begin|change)\b/,
    /\b(job|career|work|business|startup|project|relationship|marriage|house|apartment)\b/,
    /\b(decision|choice|opportunity|offer|problem|issue|situation|question)\b/,
    /\b(want|need|trying|hoping|planning|considering)\s+to\b/
  ];

  const matches = whatPatterns.filter(pattern => pattern.test(text));

  return {
    present: matches.length > 0,
    strength: matches.length > 1 ? 'strong' : matches.length === 1 ? 'weak' : 'none',
    details: matches.length > 0 ? 'Action/topic identified' : 'No clear action or topic'
  };
}

/**
 * WHEN: Identify timeframe or temporal context
 */
function analyzeWhen(text, original) {
  const whenPatterns = [
    /\b(now|today|tomorrow|tonight|this week|this month|this year|soon|eventually|immediately)\b/,
    /\b(next|last|past|future|current|recent|upcoming)\b/,
    /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/,
    /\b(january|february|march|april|may|june|july|august|september|october|november|december)\b/,
    /\b(within|by|before|after|until|during)\s+\w+/,
    /\b\d+\s+(days?|weeks?|months?|years?)\b/
  ];

  const matches = whenPatterns.filter(pattern => pattern.test(text));

  return {
    present: matches.length > 0,
    strength: matches.length > 1 ? 'strong' : matches.length === 1 ? 'weak' : 'none',
    details: matches.length > 0 ? 'Timeframe mentioned' : 'No timeframe specified'
  };
}

/**
 * WHERE: Identify location, context, or domain
 */
function analyzeWhere(text, original) {
  const wherePatterns = [
    /\b(at|in|to|from)\s+(work|home|school|office|city|country|place)\b/,
    /\b(here|there|everywhere|nowhere|anywhere)\b/,
    /\b(relationship|career|business|family|workplace|community)\b/,
    /\b(online|remote|virtual|physical|in-person)\b/,
    /\b[A-Z][a-z]+\s+(city|state|country|university|company|corporation)\b/ // Place names
  ];

  const matches = wherePatterns.filter(pattern => pattern.test(text));

  return {
    present: matches.length > 0,
    strength: matches.length > 1 ? 'strong' : matches.length === 1 ? 'weak' : 'none',
    details: matches.length > 0 ? 'Location/context mentioned' : 'No location or context'
  };
}

/**
 * WHY: Identify motivation, reasoning, or emotional context
 */
function analyzeWhy(text, original) {
  const whyPatterns = [
    /\bbecause\b/,
    /\b(feel|feeling|felt)\s+\w+/,
    /\b(afraid|scared|worried|anxious|excited|hopeful|confused|stuck|lost|hurt|angry|frustrated)\b/,
    /\b(want|need|desire|wish|hope|fear|love|hate)\b/,
    /\b(in order to|so that|to)\b/,
    /\b(reason|purpose|goal|motivation|drive)\b/
  ];

  const matches = whyPatterns.filter(pattern => pattern.test(text));

  return {
    present: matches.length > 0,
    strength: matches.length > 1 ? 'strong' : matches.length === 1 ? 'weak' : 'none',
    details: matches.length > 0 ? 'Motivation/emotion present' : 'No clear motivation or emotion'
  };
}

/**
 * HOW: Identify process, method, or manner
 */
function analyzeHow(text, original) {
  const howPatterns = [
    /\bhow\s+(do|can|should|could|would|will)\b/,
    /\b(by|through|via|using|with)\s+\w+/,
    /\b(way|method|approach|strategy|process|plan)\b/,
    /\b(step|action|move|decision)\b/
  ];

  const matches = howPatterns.filter(pattern => pattern.test(text));

  return {
    present: matches.length > 0,
    strength: matches.length > 1 ? 'strong' : matches.length === 1 ? 'weak' : 'none',
    details: matches.length > 0 ? 'Process/method mentioned' : 'No process or method'
  };
}

/**
 * Generate human-readable feedback (max 100 words)
 */
function generateFeedback(analysis, present, missing, text) {
  // Excellent - 5-6 elements present
  if (present.length >= 5) {
    return "Excellent! Your intention is clear and specific. The cards will provide detailed, actionable guidance.";
  }

  // Good - 3-4 elements present
  if (present.length >= 3) {
    const missingStr = missing.slice(0, 2).join(' and ');
    return `Good specificity. Adding ${missingStr} would help: ${getMissingHelp(missing[0])}`;
  }

  // Weak - 2 elements present
  if (present.length === 2) {
    const critical = ['who', 'what'].filter(w => !present.includes(w));
    if (critical.length > 0) {
      return `Your question needs more context. Missing: ${critical[0].toUpperCase()} - ${getMissingHelp(critical[0])}. Also consider: ${missing.filter(m => !critical.includes(m)).slice(0, 2).join(', ')}.`;
    }
    return `More context needed. Add: ${missing.slice(0, 3).join(', ')}. Example: "Should I (WHAT) quit my job at Google (WHERE) to start a bakery (WHAT) because I'm burned out (WHY) this month (WHEN)?"`;
  }

  // Very weak - 0-1 elements
  return `Too vague. Your intention needs WHO (you? someone else?), WHAT (what action/decision?), and WHY (what's the motivation?). Example: "Should I ask my boss for a raise because I've exceeded targets this quarter?" Be specific.`;
}

/**
 * Get specific help for missing element
 */
function getMissingHelp(element) {
  const help = {
    who: "Who is involved? You? Someone else?",
    what: "What's the specific action, decision, or situation?",
    when: "When is this happening? What's the timeframe?",
    where: "Where or in what context? Career, relationship, etc.?",
    why: "Why does this matter? What are you feeling?",
    how: "How are you approaching this? What's the process?"
  };
  return help[element] || "Add more context.";
}

/**
 * Get examples for improving intention
 */
export function getIntentionExamples() {
  return [
    {
      weak: "Should I quit?",
      strong: "Should I quit my job at Microsoft to start a coffee shop because I'm burned out and unfulfilled?",
      reason: "Added: WHO (I), WHAT (quit job at Microsoft, start coffee shop), WHERE (Microsoft, coffee shop), WHY (burned out, unfulfilled)"
    },
    {
      weak: "What about my relationship?",
      strong: "Should I have a difficult conversation with my partner Sarah about our lack of intimacy this weekend?",
      reason: "Added: WHO (my partner Sarah), WHAT (difficult conversation about intimacy), WHEN (this weekend)"
    },
    {
      weak: "Help with my career",
      strong: "How can I negotiate a 20% raise with my manager before my annual review in March given that I exceeded all Q4 targets?",
      reason: "Added: WHO (my manager), WHAT (negotiate 20% raise), WHEN (before annual review in March), WHY (exceeded Q4 targets), HOW (negotiate)"
    }
  ];
}
