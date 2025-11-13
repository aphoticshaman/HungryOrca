/**
 * LUNATIQ INTERPRETATION AGENTS
 * ==============================
 *
 * Multi-modal interpretation agents for tarot reading synthesis
 * Each agent specializes in a different reasoning modality
 *
 * Architecture:
 * - 5 specialized agents (archetypal, practical, psychological, relational, mystical)
 * - Activation-weighted ensemble blending
 * - Adaptive communication voice integration
 * - Offline, no LLM dependencies
 */

import { CommunicationVoice } from './adaptiveLanguage.js';

// ═══════════════════════════════════════════════════════════════════════════════
// ARCHETYPAL AGENT - Deep symbolic/Jungian interpretation
// ═══════════════════════════════════════════════════════════════════════════════

export class ArchetypalAgent {
  /**
   * Generate deep archetypal interpretation
   * High activation → mythological, symbolic, collective unconscious
   */
  generateInterpretation(card, position, isReversed, activation, context) {
    if (activation < 0.3) {
      return null; // Not activated enough
    }

    const archetypes = this.getCardArchetype(card);
    const journey = this.getJourneyStage(card.number);
    const shadow = isReversed ? this.getShadowAspect(card) : null;

    const parts = [];

    // Opening with mythological framing
    if (activation > 0.7) {
      parts.push(`${card.name} emerges as ${archetypes.primary}, a primal force in the collective unconscious.`);
    } else {
      parts.push(`${card.name} carries the energy of ${archetypes.primary}.`);
    }

    // Journey context (Hero's Journey / Fool's Journey)
    if (journey && activation > 0.5) {
      parts.push(journey);
    }

    // Position meaning with archetypal lens
    parts.push(this.getArchetypalPositionMeaning(position, archetypes, isReversed));

    // Shadow work (if reversed)
    if (shadow && activation > 0.6) {
      parts.push(shadow);
    }

    // Mythological parallel (if high activation)
    if (activation > 0.8 && archetypes.myth) {
      parts.push(archetypes.myth);
    }

    return parts.join(' ');
  }

  getCardArchetype(card) {
    const archetypeMap = {
      'The Fool': {
        primary: 'the Innocent Wanderer',
        myth: 'Like Parsifal stepping into the unknown forest, you embody pure potential.',
        shadow: 'the Trickster who leads astray'
      },
      'The Magician': {
        primary: 'the Alchemist of Manifestation',
        myth: 'Hermes Trismegistus at his altar, channeling heaven to earth.',
        shadow: 'the Manipulator who bends reality for ego'
      },
      'The High Priestess': {
        primary: 'the Guardian of Mysteries',
        myth: 'Persephone between worlds, keeper of secret wisdom.',
        shadow: 'the Witch who hoards knowledge'
      },
      'The Empress': {
        primary: 'the Great Mother',
        myth: 'Demeter in her abundance, birthing all creation.',
        shadow: 'the Devouring Mother who smothers'
      },
      'The Emperor': {
        primary: 'the Divine King',
        myth: 'Zeus on his throne, ordering chaos into cosmos.',
        shadow: 'the Tyrant who rules through fear'
      },
      'The Hierophant': {
        primary: 'the Sacred Teacher',
        myth: 'Chiron the wounded healer, bridging mortal and divine.',
        shadow: 'the Dogmatist who imprisons in doctrine'
      },
      'The Lovers': {
        primary: 'the Sacred Union',
        myth: 'Eros and Psyche, choosing love despite the fall.',
        shadow: 'the Codependent who loses self'
      },
      'The Chariot': {
        primary: 'the Victorious Warrior',
        myth: 'Apollo driving the sun chariot, mastering opposing forces.',
        shadow: 'the Conqueror who crushes all opposition'
      },
      'Strength': {
        primary: 'the Gentle Power',
        myth: 'Beauty and the Beast—compassion transforming the savage.',
        shadow: 'the Martyr who endures too much'
      },
      'The Hermit': {
        primary: 'the Wise Solitary',
        myth: 'Diogenes with his lantern, seeking truth in darkness.',
        shadow: 'the Recluse who abandons community'
      },
      'Wheel of Fortune': {
        primary: 'the Cosmic Cycle',
        myth: 'The Moirai spinning fate\'s thread—what rises must fall, what falls shall rise.',
        shadow: 'the Victim of circumstance'
      },
      'Justice': {
        primary: 'the Divine Balance',
        myth: 'Ma\'at weighing the heart against a feather.',
        shadow: 'the Judge who cannot forgive'
      },
      'The Hanged Man': {
        primary: 'the Willing Sacrifice',
        myth: 'Odin on the World Tree, surrendering to receive wisdom.',
        shadow: 'the Martyr who suffers for attention'
      },
      'Death': {
        primary: 'the Great Transformer',
        myth: 'The Phoenix in flames, dying to be reborn.',
        shadow: 'the Reaper who clings to decay'
      },
      'Temperance': {
        primary: 'the Divine Alchemist',
        myth: 'Iris the rainbow bridge, uniting heaven and earth.',
        shadow: 'the Suppressor who denies passion'
      },
      'The Devil': {
        primary: 'the Shadow Self',
        myth: 'Pan in the wilderness—raw desire unchained.',
        shadow: 'the Addict who serves only appetite'
      },
      'The Tower': {
        primary: 'the Divine Disruption',
        myth: 'The Tower of Babel falling—false structures must collapse.',
        shadow: 'the Destroyer who razes without rebuilding'
      },
      'The Star': {
        primary: 'the Cosmic Hope',
        myth: 'Pandora\'s box—when all is lost, hope remains.',
        shadow: 'the Dreamer who never acts'
      },
      'The Moon': {
        primary: 'the Keeper of Illusions',
        myth: 'Hecate at the crossroads, guiding through darkness.',
        shadow: 'the Deceiver lost in delusion'
      },
      'The Sun': {
        primary: 'the Radiant Child',
        myth: 'Ra ascending—consciousness illuminating all.',
        shadow: 'the Narcissist who blinds'
      },
      'Judgement': {
        primary: 'the Great Awakening',
        myth: 'Gabriel\'s trumpet—the dead rising to new life.',
        shadow: 'the Accuser who condemns'
      },
      'The World': {
        primary: 'the Cosmic Dancer',
        myth: 'Shiva Nataraja—completion of the cycle, readiness for the next.',
        shadow: 'the Perfectionist who cannot begin again'
      }
    };

    // For minor arcana, derive from suit
    if (!archetypeMap[card.name]) {
      return this.getMinorArcanaArchetype(card);
    }

    return archetypeMap[card.name];
  }

  getMinorArcanaArchetype(card) {
    const suitArchetypes = {
      'wands': { primary: 'the Creative Fire', shadow: 'the Destroyer' },
      'cups': { primary: 'the Emotional Depths', shadow: 'the Overwhelmed' },
      'swords': { primary: 'the Mental Clarity', shadow: 'the Overthinking Mind' },
      'pentacles': { primary: 'the Material Mastery', shadow: 'the Greedy One' }
    };

    return suitArchetypes[card.suit] || { primary: 'the Seeker', shadow: 'the Lost' };
  }

  getJourneyStage(cardNumber) {
    if (cardNumber <= 21) {
      // Major Arcana = Fool's Journey
      if (cardNumber <= 7) {
        return 'You stand in the realm of outer teachers—learning how the world works.';
      } else if (cardNumber <= 14) {
        return 'You have entered the realm of inner trials—the soul forging itself.';
      } else {
        return 'You walk the realm of cosmic mysteries—the great spiritual reckoning.';
      }
    }
    return null;
  }

  getShadowAspect(card) {
    const archetypes = this.getCardArchetype(card);
    if (archetypes.shadow) {
      return `In reversal, beware ${archetypes.shadow}—this is the shadow you must integrate.`;
    }
    return 'In reversal, this energy turns inward or becomes blocked.';
  }

  getArchetypalPositionMeaning(position, archetypes, isReversed) {
    const positionMap = {
      'Past': `In your past, ${archetypes.primary} shaped your foundation.`,
      'Present': `Now, ${archetypes.primary} is the dominant force in your life.`,
      'Future': `${archetypes.primary} is emerging on your horizon.`,
      'You': `Your essence currently embodies ${archetypes.primary}.`,
      'Situation': `The situation is ruled by ${archetypes.primary}.`,
      'Challenge': `${archetypes.primary} presents your current trial.`,
      'Outcome': `${archetypes.primary} will define your resolution.`
    };

    return positionMap[position] || `${archetypes.primary} influences this aspect.`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRACTICAL AGENT - Actionable, grounded advice
// ═══════════════════════════════════════════════════════════════════════════════

export class PracticalAgent {
  /**
   * Generate concrete, actionable guidance
   * High activation → specific steps, timelines, practical wisdom
   */
  generateInterpretation(card, position, isReversed, activation, context) {
    if (activation < 0.3) {
      return null;
    }

    const action = this.getActionableAdvice(card, context.intention, isReversed);
    const timing = this.getTimingGuidance(card, position);
    const warning = isReversed ? this.getPracticalWarning(card) : null;

    const parts = [];

    // Direct action steps
    if (activation > 0.6) {
      parts.push(`Practical guidance: ${action}`);
    } else {
      parts.push(action);
    }

    // Timing if relevant
    if (timing && activation > 0.5) {
      parts.push(timing);
    }

    // Warning if reversed
    if (warning && activation > 0.4) {
      parts.push(warning);
    }

    return parts.join(' ');
  }

  getActionableAdvice(card, intention, isReversed) {
    // Context-aware advice based on intention
    const intentionLower = intention.toLowerCase();

    // Career advice
    if (intentionLower.includes('career') || intentionLower.includes('work') || intentionLower.includes('job')) {
      return this.getCareerAction(card, isReversed);
    }

    // Relationship advice
    if (intentionLower.includes('love') || intentionLower.includes('relationship')) {
      return this.getRelationshipAction(card, isReversed);
    }

    // Financial advice
    if (intentionLower.includes('money') || intentionLower.includes('financial')) {
      return this.getFinancialAction(card, isReversed);
    }

    // General advice
    return this.getGeneralAction(card, isReversed);
  }

  getCareerAction(card, isReversed) {
    const actions = {
      'The Fool': 'Apply for that unconventional role. Update your resume with skills you\'re still learning.',
      'The Magician': 'Pitch your idea this week. You have all the tools—execute now.',
      'The Empress': 'Focus on collaborative projects. Your leadership through nurturing wins respect.',
      'The Emperor': 'Create structure: set clear goals, timelines, boundaries with colleagues.',
      'The Chariot': 'Push through resistance. Work early mornings when focus is sharpest.',
      'The Hermit': 'Request solo project time. Your best work comes from deep focus.',
      'The Tower': isReversed ? 'Update your LinkedIn—change is coming whether you\'re ready or not.' : 'Start building your exit strategy. This structure won\'t hold.',
      'The Star': 'Network genuinely. Share your vision without agenda—opportunities follow.',
      'Three of Pentacles': 'Schedule that collaboration meeting. Present your expertise clearly.',
      'Eight of Pentacles': 'Skill up: take the course, get the certification, practice deliberately.'
    };

    return actions[card.name] || 'Take initiative in your sphere of influence. Document your wins.';
  }

  getRelationshipAction(card, isReversed) {
    const actions = {
      'The Lovers': 'Have the conversation you\'ve been avoiding. Choose with your whole heart.',
      'Two of Cups': 'Plan intentional quality time—phones away, presence full.',
      'The Empress': 'Show care through tangible acts: cook a meal, create beauty together.',
      'The Hermit': 'Take solo time without guilt. Your relationship needs your wholeness.',
      'The Devil': isReversed ? 'Name the pattern: what keeps you hooked?' : 'Get support. Couples therapy or a trusted mentor.',
      'The Tower': 'Stay present through the disruption. Don\'t make big decisions for 2 weeks.',
      'Temperance': 'Practice the pause. Count to 10 before responding in conflict.',
      'Ten of Cups': 'Create new traditions. Build the relationship you want deliberately.'
    };

    return actions[card.name] || 'Communicate your needs clearly. Ask what they need in return.';
  }

  getFinancialAction(card, isReversed) {
    const actions = {
      'Ace of Pentacles': 'Open that savings account. Set up automatic transfers today.',
      'Four of Pentacles': isReversed ? 'Calculate what you can afford to invest in yourself.' : 'Review your budget. Where can you loosen your grip?',
      'Nine of Pentacles': 'Invest in quality over quantity. That expensive tool pays for itself.',
      'The Emperor': 'Create a 6-month financial plan with specific milestones.',
      'The Devil': 'List all debts. Make a payoff strategy. Cut one subscription this week.',
      'The Tower': 'Build your emergency fund now—3 months expenses minimum.'
    };

    return actions[card.name] || 'Track your spending for 7 days. Awareness precedes change.';
  }

  getGeneralAction(card, isReversed) {
    // Fallback general advice
    if (isReversed) {
      return `Reverse course: the upright path isn't working. Try the opposite approach.`;
    }
    return `Lean into this energy. Small consistent actions compound.`;
  }

  getTimingGuidance(card, position) {
    const timingMap = {
      'The Fool': 'Act within the next new moon. Momentum fades fast.',
      'The Chariot': 'This week. Speed matters now.',
      'The Hermit': 'Take your time. 40 days of reflection yields clarity.',
      'The Hanged Man': 'Wait. Do nothing for 2 weeks—clarity comes in suspension.',
      'Death': 'The transition is already underway. Let go by the next full moon.',
      'The Tower': 'Immediate. The structure is already falling.',
      'The Star': 'Plant seeds now; harvest in 6 months.'
    };

    return timingMap[card.name] || null;
  }

  getPracticalWarning(card) {
    const warnings = {
      'The Fool': 'Warning: Don\'t skip the research phase. "Leap" doesn\'t mean "reckless."',
      'The Magician': 'Warning: Ensure you\'re not manipulating outcomes. Clean hands matter.',
      'The Emperor': 'Warning: Rigid control breaks relationships. Leave room for others.',
      'The Devil': 'Warning: If you\'re lying to yourself, you\'ll repeat the cycle.',
      'The Tower': 'Warning: Accept help. Lone wolf energy will amplify the crash.',
      'The Moon': 'Warning: Don\'t make major decisions in this fog. Wait for clarity.'
    };

    return warnings[card.name] || null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PSYCHOLOGICAL AGENT - CBT/DBT therapeutic integration
// ═══════════════════════════════════════════════════════════════════════════════

export class PsychologicalAgent {
  /**
   * Generate psychologically-informed interpretation
   * High activation → explicit CBT/DBT skills, cognitive patterns, emotional regulation
   */
  generateInterpretation(card, position, isReversed, activation, context) {
    if (activation < 0.3) {
      return null;
    }

    const pattern = this.getCognitivePattern(card, isReversed);
    const dbtSkill = this.getRelevantDBTSkill(card, context.profile);
    const emotionalWork = this.getEmotionalRegulationGuidance(card, isReversed);

    const parts = [];

    // Cognitive pattern identification
    if (activation > 0.5 && pattern) {
      parts.push(pattern);
    }

    // DBT skill recommendation
    if (activation > 0.6 && dbtSkill) {
      parts.push(`DBT skill: ${dbtSkill}`);
    } else if (dbtSkill && activation > 0.4) {
      parts.push(dbtSkill); // Subtle integration
    }

    // Emotional regulation guidance
    if (emotionalWork && activation > 0.5) {
      parts.push(emotionalWork);
    }

    // If nothing generated, provide generic psychological insight
    if (parts.length === 0 && activation > 0.4) {
      return `Notice what feelings arise with this card. What is your body telling you?`;
    }

    return parts.join(' ');
  }

  getCognitivePattern(card, isReversed) {
    const patterns = {
      'The Fool': {
        upright: 'You\'re in a growth mindset—embracing uncertainty as possibility.',
        reversed: 'Notice if you\'re avoiding due to catastrophizing: "If I try, I\'ll fail."'
      },
      'The Magician': {
        upright: 'You\'re recognizing your agency—internal locus of control activating.',
        reversed: 'Watch for all-or-nothing thinking: "I must control everything or I\'m powerless."'
      },
      'The High Priestess': {
        upright: 'You\'re accessing intuitive knowing—trusting non-verbal processing.',
        reversed: 'Are you intellectualizing emotions to avoid feeling them?'
      },
      'The Empress': {
        upright: 'You\'re practicing self-compassion and radical acceptance.',
        reversed: 'Notice if you\'re people-pleasing at the cost of your own needs.'
      },
      'The Emperor': {
        upright: 'You\'re setting healthy boundaries—protective structure emerging.',
        reversed: 'Watch for control as anxiety management. What are you afraid will happen if you let go?'
      },
      'The Lovers': {
        upright: 'You\'re integrating different parts of yourself—internal alignment.',
        reversed: 'Notice if you\'re seeking external validation to feel whole.'
      },
      'The Chariot': {
        upright: 'You\'re using opposite action—moving despite fear.',
        reversed: 'Is pushing through actually avoidance of needed rest? Check your window of tolerance.'
      },
      'Strength': {
        upright: 'You\'re regulating through gentleness, not force—wise mind activating.',
        reversed: 'Are you spiritually bypassing difficult emotions?'
      },
      'The Hermit': {
        upright: 'You\'re engaging healthy solitude for integration.',
        reversed: 'Is this solitude healing, or isolation to avoid vulnerability?'
      },
      'The Devil': {
        upright: 'You\'re facing your attachment patterns and behavioral chains.',
        reversed: 'Notice what triggers your addictive patterns. Map the urge cycle.'
      },
      'The Tower': {
        upright: 'Your schemas are breaking—painful but necessary cognitive restructuring.',
        reversed: 'Are you clinging to beliefs that no longer serve you out of fear of the unknown?'
      },
      'The Star': {
        upright: 'You\'re building hope as a skill, not just a feeling.',
        reversed: 'Notice if you\'re using "hope" to avoid present-moment action.'
      },
      'The Moon': {
        upright: 'You\'re learning to sit with ambiguity—distress tolerance building.',
        reversed: 'Your amygdala is hijacking your prefrontal cortex. Name it: "This is anxiety, not truth."'
      }
    };

    const pattern = patterns[card.name];
    if (!pattern) return null;

    return isReversed ? pattern.reversed : pattern.upright;
  }

  getRelevantDBTSkill(card, profile) {
    const skills = {
      'The Fool': 'Willingness (saying yes to the present moment)',
      'The Magician': 'Opposite Action (acting against ineffective urges)',
      'The High Priestess': 'Wise Mind (synthesizing emotion + reason)',
      'The Empress': 'Self-Soothe (engaging the five senses)',
      'The Emperor': 'DEAR MAN (assertiveness with relationship effectiveness)',
      'The Lovers': 'Dialectics (holding two truths simultaneously)',
      'The Chariot': 'Build Mastery (doing difficult things to build confidence)',
      'Strength': 'TIPP (Temperature, Intense exercise, Paced breathing, Progressive relaxation)',
      'The Hermit': 'Observe (watching thoughts without judgment)',
      'Wheel of Fortune': 'Ride the Wave (accepting impermanence)',
      'The Hanged Man': 'Radical Acceptance (ceasing to fight reality)',
      'Death': 'Turning the Mind (choosing acceptance again and again)',
      'Temperance': 'Middle Path (avoiding extremes)',
      'The Devil': 'Alternate Rebellion (meeting needs in non-destructive ways)',
      'The Tower': 'Reality Acceptance Skills (accepting reality to change it)',
      'The Moon': 'Checking the Facts (testing anxious thoughts against evidence)',
      'The Sun': 'Accumulate Positives (building a life worth living)'
    };

    const skill = skills[card.name];
    if (!skill) return null;

    // If profile shows low emotional regulation, make skills more explicit
    if (profile && profile.emotionalRegulation < 0.4) {
      return `Practice ${skill}—your nervous system needs regulation tools right now.`;
    }

    return skill;
  }

  getEmotionalRegulationGuidance(card, isReversed) {
    const guidance = {
      'The Fool': 'Notice excitement vs. anxiety in your body. Where do you feel it?',
      'The Tower': 'Your nervous system is in fight/flight. Breathe: 4 counts in, 6 counts out.',
      'The Moon': 'Fear is a visitor, not a commander. What does it want you to know?',
      'The Devil': 'Shame is showing up. Speak it aloud to someone safe—it loses power in light.',
      'The Star': 'Let yourself feel hope without protecting against disappointment.',
      'The Sun': 'Joy is safe to feel. Your past doesn\'t negate this present moment.'
    };

    return guidance[card.name] || null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RELATIONAL AGENT - Connection patterns, systemic dynamics
// ═══════════════════════════════════════════════════════════════════════════════

export class RelationalAgent {
  /**
   * Generate interpretation focused on relationship patterns and systems
   * High activation → attachment theory, family systems, interpersonal dynamics
   */
  generateInterpretation(card, position, isReversed, activation, context) {
    if (activation < 0.3) {
      return null;
    }

    const pattern = this.getRelationalPattern(card, isReversed);
    const dynamic = this.getSystemicDynamic(card, context.cards);

    const parts = [];

    if (pattern && activation > 0.5) {
      parts.push(pattern);
    }

    if (dynamic && activation > 0.6) {
      parts.push(dynamic);
    }

    return parts.join(' ');
  }

  getRelationalPattern(card, isReversed) {
    const patterns = {
      'The Empress': 'You\'re in caretaker mode—ensure it\'s reciprocal, not one-sided.',
      'The Emperor': 'You\'re setting boundaries. Healthy relationships have container and space.',
      'The Lovers': 'Notice: are you choosing from wholeness or trying to complete yourself through another?',
      'The Hermit': 'Solitude is repair time. Your relationships will be stronger when you return.',
      'Two of Cups': 'True partnership: you see them, they see you. Mutual recognition.',
      'Three of Cups': 'Community is medicine. Who are your people?',
      'Five of Cups': 'Grief in relationship: what you\'re mourning matters. Honor it.',
      'The Devil': 'Codependency alert: where do you end and they begin?'
    };

    return patterns[card.name] || null;
  }

  getSystemicDynamic(card, allCards) {
    // Analyze card in context of full spread
    // Look for patterns: lots of cups = emotional focus, etc.
    return null; // Placeholder for systemic analysis
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MYSTICAL AGENT - Esoteric, spiritual, energetic interpretation
// ═══════════════════════════════════════════════════════════════════════════════

export class MysticalAgent {
  /**
   * Generate mystical/spiritual interpretation
   * High activation → chakras, elements, moon phases, energetic work
   */
  generateInterpretation(card, position, isReversed, activation, context) {
    if (activation < 0.3) {
      return null;
    }

    const energetic = this.getEnergeticGuidance(card, isReversed);
    const spiritual = this.getSpiritualMessage(card, context.moonPhase);

    const parts = [];

    if (energetic && activation > 0.5) {
      parts.push(energetic);
    }

    if (spiritual && activation > 0.7) {
      parts.push(spiritual);
    }

    return parts.join(' ');
  }

  getEnergeticGuidance(card, isReversed) {
    const guidance = {
      'The Fool': 'Clear your crown chakra. Burn sage in doorways—you\'re beginning.',
      'The High Priestess': 'Third eye activation. Meditate in darkness. Trust your downloads.',
      'The Empress': 'Heart chakra opening. Wear green. Spend time with growing things.',
      'The Magician': 'Solar plexus power. Visualize golden light in your core.',
      'The Star': 'You\'re a channel right now. Create, write, move—let it flow through you.'
    };

    return guidance[card.name] || null;
  }

  getSpiritualMessage(card, moonPhase) {
    // Placeholder for moon phase integration
    return null;
  }
}
