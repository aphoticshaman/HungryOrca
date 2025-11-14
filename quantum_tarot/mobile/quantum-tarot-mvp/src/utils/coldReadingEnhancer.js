/**
 * COLD READING ENHANCER
 *
 * Professional psychics and tarot readers use cold reading techniques to create
 * the impression of specific, personal knowledge. This isn't "cheating"—it's
 * understanding human psychology and making readings feel PERSONAL and REAL.
 *
 * Techniques used:
 * - Barnum statements (universally true but feel specific)
 * - Astrological anchoring (Pisces → intuitive, Aries → impulsive)
 * - Sensory specifics (colors, numbers, letters, symbols)
 * - Progressive revelation (start vague, get specific based on "feedback")
 * - Flattery with edge (you're special but struggling)
 *
 * This makes readings feel like the reader KNOWS YOU, which increases trust
 * and emotional engagement.
 */

/**
 * Generate Barnum statements based on MBTI and astrology
 * (Universally true statements that feel personally targeted)
 * HARDENING #2: Layered Barnum - Universal + MBTI-specific elaboration
 */
export function generateBarnumStatements(userProfile, quantumSeed) {
  const { mbtiType, zodiacSign } = userProfile;

  const statements = [];

  // MBTI-based Barnum statements WITH LAYERING
  // Structure: Universal truth + MBTI-specific elaboration (still broad but feels targeted)
  const mbtiBarnum = {
    // Introverts
    'INFP': [
      'You have a rich inner world that few people truly understand. As an INFP, you\'ve tried to translate it through art, writing, or music—but even those feel inadequate to what you\'re actually experiencing.',
      'You\'re more idealistic than you let on. Part of you still believes in magic, even if you\'ve been disappointed. That\'s not naivete—it\'s your Fi refusing to accept a world without meaning.',
      'People think you\'re gentle, but you have a spine of steel when it comes to your values. You just don\'t advertise it. When someone crosses your moral line, though—they see a side of you they didn\'t know existed.'
    ],
    'INFJ': [
      'You see patterns others miss—not just surface behaviors, but the deeper currents beneath. Your Ni knows how things will play out before they do. It\'s both a gift and a burden, because you can see the train wreck coming but can\'t always stop it.',
      'You have at least one relationship where you gave too much and got too little. As an INFJ, you\'re still learning that boundaries aren\'t walls—they\'re the foundation for sustainable love. Your Fe wants to save everyone. Your Ni knows you can\'t.',
      'People come to you for advice, but who do YOU go to? You carry more alone than anyone knows. That\'s the INFJ burden: being everyone\'s therapist while your own wounds stay unspoken.'
    ],
    'INTP': [
      'Your mind moves faster than you can articulate. People mistake your silence for agreement when you\'re really five steps ahead, running mental simulations they can\'t see. Ti-Ne means you\'re playing 4D chess while they\'re playing checkers.',
      'You have a collection—physical or mental—of things others would find random but you see the pattern. As an INTP, you\'re not a hoarder, you\'re an archivist of connections. Each piece is a node in your mental map.',
      'You\'ve been called cold or detached, but you feel deeply. You just process through analysis first. Your Ti dissects emotions like a surgeon, not because you don\'t care—because that\'s how you understand them.'
    ],
    'INTJ': [
      'You knew you were different from a young age. Not better, just... operating on a different wavelength. While others were playing, you were analyzing systems and seeing three moves ahead.',
      'You have a vision of how things SHOULD work, and reality\'s inefficiency frustrates you daily. Your Ni-Te stack means you don\'t just see what\'s broken—you see exactly how to fix it. Getting others to listen is the hard part.',
      'People underestimate you until they realize you\'ve been running the long game all along. What looks like sudden success to others? You\'ve been building it for years in silence.'
    ],

    // Extroverts
    'ENFP': [
      'You start projects with explosive enthusiasm, then struggle to finish. Your graveyard of "almost" accomplishments haunts you. That\'s Ne-Fi—you see infinite possibilities, but each new one makes the last one fade. You\'re not flaky, you\'re a serial monogamist with ideas.',
      'You\'ve been told you\'re "too much"—too loud, too emotional, too intense. But dimming yourself feels like dying. As an ENFP, your authenticity is non-negotiable. You\'d rather be alone than perform a version of yourself that fits their comfort zone.',
      'You collect people the way others collect stamps. But how many truly know the real you? Your Fi keeps that hidden, even from people who think they\'re close. You show them the fun ENFP. The depth? That\'s invite-only.'
    ],
    'ENFJ': [
      'You give and give until you\'re empty, then feel guilty for having needs. When does YOUR healing happen? As an ENFJ, your Fe-Ni means you see everyone\'s potential and exhaust yourself trying to midwife it. But who\'s holding space for YOU?',
      'You can read a room in seconds. It\'s exhausting being everyone\'s emotional translator. You feel the undercurrents, the tension, the unspoken—and your Fe won\'t let you ignore it. So you fix it. Every. Single. Time.',
      'You have a wound around not feeling "good enough" despite evidence to the contrary. Who told you that lie? ENFJs carry this wound like a birthmark—the fear that love is earned through performance, not given freely.'
    ],
    'ENTP': [
      'You argue for sport. People think you\'re combative, but you\'re just testing ideas. Ne-Ti means you\'re not attacking THEM, you\'re stress-testing their logic. They don\'t see the difference. Still, it\'s cost you relationships.',
      'Boredom is your kryptonite. You\'d rather fail spectacularly than succeed safely. As an ENTP, routine is death. You need novelty, challenge, intellectual stimulation—or you become a chaos agent just to feel alive.',
      'You have unfinished business with someone from your past. It surfaces in quiet moments. Your Fe is your weakest function, but when it activates? It haunts you. That person represents something you didn\'t say or do when it mattered.'
    ],
    'ENTJ': [
      'You\'ve been called intimidating. You\'re not trying to be—you just don\'t suffer inefficiency or dishonesty. Te-Ni means you see the most direct path to the goal and can\'t understand why people want to take scenic routes through incompetence.',
      'You have a soft side you protect fiercely. Only 1-3 people have ever seen it fully. As an ENTJ, vulnerability feels like weakness—even though intellectually you know it\'s strength. Your Fi is locked in a vault with one key.',
      'You measure yourself against impossible standards. When\'s the last time you celebrated a win without immediately moving to the next goal? ENTJs don\'t rest on laurels. You see achievement as baseline, not victory. That drive built your success. It also costs you peace.'
    ],

    // Sensors
    'ISFP': [
      'You express yourself better through action/art than words. People misread your quiet as agreement. Fi-Se means you FEEL intensely but communicate through creation, not conversation. Your art speaks what your mouth can\'t.',
      'You have a strong aesthetic sense—your environment deeply affects your mood. As an ISFP, beauty isn\'t luxury—it\'s oxygen. Ugly spaces drain you. Harmonious ones restore you. People don\'t get that it\'s not shallow, it\'s survival.',
      'You\'ve been hurt by someone who took your gentle nature for weakness. You\'re not weak. You\'re selective. ISFPs are steel wrapped in silk. You choose non-confrontation until you don\'t. Then the claws come out.'
    ],
    'ISFJ': [
      'You remember birthdays, preferences, small details about people. Do they notice what you notice about them? Si-Fe means you archive every kindness you give and every slight you receive. You remember who showed up. And who didn\'t.',
      'You\'ve sacrificed for others in ways they\'ll never fully understand or appreciate. As an ISFJ, you carry the family, the team, the friendship group on your back—quietly, without complaint. Until one day you realize you\'re exhausted and no one even saw the weight.',
      'You have a memory from childhood that still shapes how you show love today. ISFJs are built on Si-Fe: past experiences create templates for how to care. That memory—good or bad—is your blueprint for connection.'
    ],
    'ISTJ': [
      'You do what you say you\'ll do. Why doesn\'t everyone? Their lack of follow-through baffles and frustrates you. Si-Te means your word is contract. When others break theirs, it\'s not just annoying—it\'s a moral failure in your book.',
      'You\'re more sentimental than people think. You have something you\'ve kept for years that others would call "junk." As an ISTJ, Si makes you a curator of meaningful objects. That "junk" is a time capsule, a memory anchor. You know exactly why you kept it.',
      'You value competence above almost everything. Incompetence feels like disrespect. ISTJs respect those who show up prepared, follow through, and do quality work. Everything else? Noise. If you can\'t do it right, don\'t waste their time.'
    ],
    'ISTP': [
      'You need freedom more than security. The thought of being trapped—literally or metaphorically—terrifies you. Ti-Se means you need autonomy and sensory stimulation. A stable job in a cubicle? That\'s not safety, that\'s a slow death.',
      'You fix things. Machines, problems, people. But who fixes YOU? As an ISTP, you\'re the troubleshooter everyone calls. Ti reverse-engineers anything. But emotional breakdowns? Those don\'t have manuals. And you don\'t ask for help.',
      'You\'ve walked away from something good because it started feeling like a cage. ISTPs value freedom over comfort. The moment a relationship, job, or situation demands too much compromise, you\'re gone. No drama. Just gone.'
    ],

    'ESFP': [
      'You light up rooms, but sometimes wonder who you\'d be without an audience. Se-Fi means you shine in the moment, feeding off energy around you. But in the quiet? There\'s a question: is this performance or authenticity? Both?',
      'You live for experiences. Your photo gallery is FULL. Your bank account... less so. As an ESFP, Se demands you LIVE—travel, taste, touch, feel. Money is for memories, not hoarding. Retirement planning? That\'s future-you\'s problem.',
      'You\'ve been told to "be more serious" or "settle down." Screw that. But also... when? ESFPs resist being boxed in. But part of you wonders if perpetual motion is freedom or avoidance. What are you afraid of catching if you stop running?'
    ],
    'ESFJ': [
      'Harmony matters to you—sometimes too much. You\'ve swallowed your truth to keep peace. Fe-Si means you prioritize group cohesion and learned social norms. But whose comfort are you protecting? And at what cost to your own?',
      'You know everyone\'s business. It\'s not gossip if you care (right?). As an ESFJ, you\'re the social hub—connecting people, tracking details, maintaining relationships. You call it caring. Critics call it meddling. Both are true.',
      'You judge yourself by how others see you. What would change if you stopped? ESFJs derive value from external validation. Being needed, appreciated, respected—that\'s how you know you matter. But what if you mattered regardless?'
    ],
    'ESTJ': [
      'You get shit done. But recognition? Often goes to someone more political or charming. Te-Si means you execute flawlessly, follow process, deliver results—and watch someone else get promoted because they schmooze better. It\'s infuriating.',
      'You value tradition and structure, but even you wonder if some rules are just... stupid. As an ESTJ, you respect proven systems—until they stop working. Then your Te says \"this is inefficient\" and you want to burn it down and rebuild it properly.',
      'You have strong opinions about "the way things should be done." You\'re usually right. Doesn\'t mean people listen. ESTJs see the obvious solution everyone else ignores. Your Te-Si knows what works. But people don\'t want efficiency. They want comfort.'
    ],
    'ESTP': [
      'You thrive in crisis. Normal life bores you. This has consequences. Se-Ti means you need adrenaline, risk, immediacy—or you feel dead inside. Stability? That\'s stagnation. But the chaos you create to feel alive? Yeah, consequences.',
      'You have at least one story that starts with "so we probably shouldn\'t have..." but you did anyway. As an ESTP, rules are suggestions. Ti evaluates risk-reward in real time. Se says \"let\'s find out.\" The stories you tell later? Worth whatever damage you took.',
      'You hate being told what to do. Authority is earned, not given. This has created conflict. ESTPs respect competence, not titles. If you can\'t back up your authority with skill, you\'re just noise. This attitude? Cost you jobs. But you\'d do it again.'
    ]
  };

  // Get MBTI statements
  const mbtiStatementSet = mbtiBarnum[mbtiType] || mbtiBarnum['INFP'];
  const mbtiIdx = Math.floor(quantumSeed * mbtiStatementSet.length) % mbtiStatementSet.length;
  const selectedStatement = mbtiStatementSet[mbtiIdx] || mbtiStatementSet[0];
  statements.push(selectedStatement);

  // Zodiac-based Barnum statements
  const zodiacBarnum = {
    'Aries': 'You have a scar—physical or emotional—on your head or face that tells a story of moving too fast.',
    'Taurus': 'There\'s something you\'ve held onto for years—a possession, grudge, or habit—that you know you should release.',
    'Gemini': 'You have unfinished writing projects or messages you never sent. Words live in your head louder than you speak them.',
    'Cancer': 'Your relationship with your mother or maternal figure is complex. Healing that wound is part of your soul work.',
    'Leo': 'You\'ve been dimmed by someone\'s jealousy or criticism. You\'re remembering how to shine again.',
    'Virgo': 'Your perfectionism protects you from vulnerability. But it also keeps intimacy at arm\'s length.',
    'Libra': 'You\'ve stayed in a situation too long because leaving felt mean or unfair. Kindness to others became cruelty to self.',
    'Scorpio': 'You have a secret you\'ve never told anyone. It feels like if people knew, they\'d see you differently.',
    'Sagittarius': 'You have a place you dream of but haven\'t been to yet. It represents freedom you haven\'t claimed.',
    'Capricorn': 'You achieved something that looked impressive from the outside but felt empty. Success doesn\'t equal fulfillment.',
    'Aquarius': 'You feel like an alien. You\'ve always felt like you\'re FROM somewhere else, just temporarily here.',
    'Pisces': 'Your dreams are vivid and sometimes prophetic. You know things you have no logical way of knowing.'
  };

  if (zodiacSign && zodiacBarnum[zodiacSign]) {
    statements.push(zodiacBarnum[zodiacSign]);
  }

  return statements;
}

/**
 * Generate sensory/symbolic specifics
 * Colors, numbers, letters, symbols that create "hits"
 */
export function generateSensoryDetails(userProfile, cards, quantumSeed) {
  const details = [];

  // Color associations (from cards + astrology)
  const colors = [
    'emerald green', 'deep purple', 'crimson red', 'midnight blue',
    'silver', 'gold', 'burnt orange', 'pearl white', 'charcoal gray'
  ];
  const colorIdx = Math.floor(quantumSeed * colors.length) % colors.length;
  const selectedColor = colors[colorIdx] || colors[0];
  details.push({
    type: 'color',
    text: `The color ${selectedColor} keeps appearing around this situation. Pay attention when you see it.`
  });

  // Number synchronicity
  const significantNumbers = [3, 7, 11, 12, 22, 33, 44];
  const numIdx = Math.floor((quantumSeed * 0.777) * significantNumbers.length) % significantNumbers.length;
  const selectedNumber = significantNumbers[numIdx] || significantNumbers[0];
  details.push({
    type: 'number',
    text: `Watch for the number ${selectedNumber}. It will show up as confirmation.`
  });

  // Letter/initial (based on zodiac or card)
  const letters = ['J', 'M', 'A', 'R', 'S', 'K', 'L', 'D', 'C'];
  const letterIdx = Math.floor((quantumSeed * 0.333) * letters.length) % letters.length;
  const letter1 = letters[letterIdx] || letters[0];
  const letter2 = letters[(letterIdx + 1) % letters.length] || letters[1];
  details.push({
    type: 'letter',
    text: `Someone whose name starts with ${letter1} or ${letter2} is relevant to this reading. Either helper or lesson.`
  });

  // Animal symbolism
  const animals = [
    'butterfly', 'raven', 'owl', 'snake', 'wolf', 'deer',
    'hummingbird', 'hawk', 'fox', 'bear', 'cat', 'dog'
  ];
  const animalIdx = Math.floor((quantumSeed * 0.555) * animals.length);
  const animal = animals[animalIdx] || 'owl'; // Fallback to owl if undefined
  details.push({
    type: 'animal',
    text: `${animal.charAt(0).toUpperCase() + animal.slice(1)} medicine is with you. Look up its spiritual meaning—it's your totem for this chapter.`
  });

  // Time of day significance
  const times = [
    { time: '3:33 AM', meaning: 'spiritual awakening hour' },
    { time: '11:11', meaning: 'portal opening, manifestation' },
    { time: 'dawn', meaning: 'new beginnings' },
    { time: 'dusk', meaning: 'liminal transition' },
    { time: 'midnight', meaning: 'shadow work time' }
  ];
  const timeIdx = Math.floor((quantumSeed * 0.888) * times.length) % times.length;
  const selectedTime = times[timeIdx] || times[0]; // Fallback to first element
  details.push({
    type: 'time',
    text: `Pay attention to what comes through at ${selectedTime.time}—${selectedTime.meaning}.`
  });

  // Physical sensation
  const sensations = [
    'tingling in your hands or fingertips',
    'warmth in your chest or heart center',
    'tightness in your throat (unexpressed truth)',
    'pressure at your third eye (psychic activation)',
    'butterflies or knots in your stomach (gut knowing)',
    'chills or goosebumps (confirmation from Spirit)'
  ];
  const sensationIdx = Math.floor((quantumSeed * 0.666) * sensations.length) % sensations.length;
  const selectedSensation = sensations[sensationIdx] || sensations[0];
  details.push({
    type: 'sensation',
    text: `You might feel ${selectedSensation} when you're on the right path. That's your body's yes.`
  });

  return details;
}

/**
 * Generate "I'm getting a feeling" intuitive statements
 * Creates emotional hooks and sense of psychic connection
 */
export function generateIntuitiveHooks(readingType, quantumSeed) {
  const hooks = [
    'I\'m getting a strong pull toward the left side of your body. There\'s either a physical issue there or it represents past/feminine energy that needs attention.',
    'This reading has a heavy energy. Like you\'ve been carrying something that isn\'t yours to carry.',
    'I keep seeing water. Either you live near it, it\'s significant in your dreams, or you need to cry and haven\'t let yourself.',
    'There\'s an older woman—grandmother, aunt, mentor—whose spirit is with you in this. She wants you to know you\'re on the right path.',
    'I\'m getting "contract" or "agreement." Either a literal document or a soul agreement you made that you\'ve outgrown.',
    'Music is significant. A song will come on that\'s a direct message. You\'ll know it when you hear it.',
    'You have a gift you\'re sitting on. Psychic ability, creative talent, healing hands—something you downplay that Spirit wants you to claim.',
    'There\'s unfinished business with a father figure. Biological or symbolic. Forgiveness might not be for them—it\'s to free YOU.',
    'I\'m seeing books or writing. You\'re meant to learn something specific OR share your story. Both?',
    'Your sleep is disrupted for a reason. The veil is thin. Your dreams are trying to tell you something. Start journaling them.',
  ];

  const idx = Math.floor(quantumSeed * hooks.length) % hooks.length;
  return hooks[idx] || hooks[0];
}

/**
 * Generate "yes/no" confirmation prompts
 * These create engagement and the illusion of psychic accuracy through feedback
 */
export function generateConfirmationPrompts(cards, userProfile, quantumSeed) {
  const prompts = [
    'Does the number 7 mean something to you? July, 7 years, 7 people? It\'s showing up.',
    'Have you been waking up between 3-4 AM? That\'s significant.',
    'Is there a crossroads or actual intersection that\'s meaningful? I keep seeing literal crossed paths.',
    'Someone said something recently that cut deep. It\'s still echoing. Sound familiar?',
    'You have a decision you\'ve been avoiding. The universe is saying the deadline is approaching.',
    'Is your left knee, ankle, or hip bothering you? Or metaphorically, is your foundation on the "past" side shaky?',
    'There\'s a photo or memento you look at when you\'re feeling nostalgic or sad. What is it showing you?',
    'You\'ve been thinking about someone from your past more than usual. There\'s a reason. What is it?'
  ];

  const idx = Math.floor(quantumSeed * prompts.length) % prompts.length;
  return prompts[idx] || prompts[0];
}

/**
 * Generate flattery-with-edge statements
 * Make user feel special but also called out
 */
export function generateFlatteryWithEdge(mbtiType, quantumSeed) {
  const statements = [
    'You\'re more psychic/intuitive than you give yourself credit for. But you rationalize it away instead of trusting it. Why?',
    'You have natural leadership ability, but self-doubt has kept you playing small. Who convinced you that you weren\'t ready?',
    'People underestimate you. You let them. It\'s a protection mechanism. But it\'s also a cage.',
    'You\'re smarter than most people in the room, and you know it. But you\'ve learned to hide it to avoid threatening egos.',
    'You have healing hands/energy. People feel better after being around you. But you deplete yourself. Where are YOUR boundaries?',
    'You see through bullshit instantly. It\'s a gift and a curse. Makes genuine connection rare. But when you find it—magic.',
    'You\'re capable of SO much more than you\'re currently doing. Fear or loyalty is holding you back. Which one?',
    'You\'re more sensitive than you appear. You\'ve built armor. It protected you once. Does it still serve you?'
  ];

  const idx = Math.floor(quantumSeed * statements.length) % statements.length;
  return statements[idx] || statements[0];
}

/**
 * Master function: Weave cold reading elements into synthesis
 */
export function weaveColdReadingElements(userProfile, cards, readingType, quantumSeed) {
  const elements = {};

  // Barnum statements (2-3 per reading)
  elements.barnum = generateBarnumStatements(userProfile, quantumSeed);

  // Sensory details (2-3 specific details)
  elements.sensory = generateSensoryDetails(userProfile, cards, quantumSeed);

  // Intuitive hook (1 per reading, placed strategically)
  elements.intuitiveHook = generateIntuitiveHooks(readingType, quantumSeed * 0.111);

  // Confirmation prompt (1 per reading)
  elements.confirmationPrompt = generateConfirmationPrompts(cards, userProfile, quantumSeed * 0.222);

  // Flattery with edge (1 per reading)
  elements.flattery = generateFlatteryWithEdge(userProfile.mbtiType, quantumSeed * 0.333);

  return elements;
}
