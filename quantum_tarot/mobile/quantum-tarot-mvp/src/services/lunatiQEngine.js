/**
 * LUNATIQ AGI ENGINE
 * ==================
 *
 * Main LunatiQ AGI engine for quantum tarot interpretation
 * Integrates fuzzy orchestrator + interpretation agents + adaptive language
 *
 * Architecture:
 * 1. Fuzzy Orchestrator computes activation levels (multi-modal feature extraction)
 * 2. Interpretation Agents generate modality-specific interpretations
 * 3. Ensemble Blender synthesizes weighted final interpretation
 * 4. Adaptive Language applies communication voice
 *
 * Based on Ryan's METAMORPHOSIS + Fuzzy Meta-Controller frameworks
 * Offline, no LLM dependencies
 */

import { LunatiQOrchestrator } from './fuzzyOrchestrator.js';
import {
  ArchetypalAgent,
  PracticalAgent,
  PsychologicalAgent,
  RelationalAgent,
  MysticalAgent
} from './interpretationAgents.js';
import { AdaptiveLanguageEngine } from './adaptiveLanguage.js';

// ═══════════════════════════════════════════════════════════════════════════════
// ENSEMBLE BLENDER - Multi-agent interpretation synthesis
// ═══════════════════════════════════════════════════════════════════════════════

export class EnsembleBlender {
  /**
   * Blend multiple agent interpretations into coherent final reading
   * Uses activation weights + communication profile for synthesis
   */
  blendInterpretations(agentOutputs, activations, commProfile, card, position) {
    // Filter out null outputs and sort by activation
    const validOutputs = [];

    if (agentOutputs.archetypal && activations.archetypal > 0.3) {
      validOutputs.push({
        text: agentOutputs.archetypal,
        weight: activations.archetypal,
        type: 'archetypal'
      });
    }

    if (agentOutputs.practical && activations.practical > 0.3) {
      validOutputs.push({
        text: agentOutputs.practical,
        weight: activations.practical,
        type: 'practical'
      });
    }

    if (agentOutputs.psychological && activations.psychological > 0.3) {
      validOutputs.push({
        text: agentOutputs.psychological,
        weight: activations.psychological,
        type: 'psychological'
      });
    }

    if (agentOutputs.relational && activations.relational > 0.3) {
      validOutputs.push({
        text: agentOutputs.relational,
        weight: activations.relational,
        type: 'relational'
      });
    }

    if (agentOutputs.mystical && activations.mystical > 0.3) {
      validOutputs.push({
        text: agentOutputs.mystical,
        weight: activations.mystical,
        type: 'mystical'
      });
    }

    // Sort by weight (highest activation first)
    validOutputs.sort((a, b) => b.weight - a.weight);

    // Blend strategy based on communication profile
    return this.synthesizeByVoice(validOutputs, commProfile, card, position);
  }

  synthesizeByVoice(outputs, commProfile, card, position) {
    if (outputs.length === 0) {
      return `${card.name} appears, offering its guidance.`;
    }

    // Different synthesis strategies by voice
    const voice = commProfile.primaryVoice;

    // Analytical: prioritize practical + psychological
    if (voice === 'analytical_guide') {
      return this.synthesizeAnalytical(outputs);
    }

    // Intuitive/Mystical: prioritize archetypal + mystical
    if (voice === 'intuitive_mystic') {
      return this.synthesizeMystical(outputs);
    }

    // Direct Coach: prioritize practical only
    if (voice === 'direct_coach') {
      return this.synthesizeDirect(outputs);
    }

    // Gentle Nurturer: blend psychological + practical with soft language
    if (voice === 'gentle_nurturer') {
      return this.synthesizeNurturing(outputs);
    }

    // Default: balanced blend of top 2-3 agents
    return this.synthesizeBalanced(outputs);
  }

  synthesizeAnalytical(outputs) {
    // Prioritize practical and psychological
    const parts = [];

    const practical = outputs.find(o => o.type === 'practical');
    const psychological = outputs.find(o => o.type === 'psychological');
    const archetypal = outputs.find(o => o.type === 'archetypal');

    if (practical) parts.push(practical.text);
    if (psychological) parts.push(psychological.text);
    if (archetypal && archetypal.weight > 0.6) parts.push(archetypal.text);

    return parts.join(' ');
  }

  synthesizeMystical(outputs) {
    // Prioritize archetypal and mystical
    const parts = [];

    const archetypal = outputs.find(o => o.type === 'archetypal');
    const mystical = outputs.find(o => o.type === 'mystical');
    const psychological = outputs.find(o => o.type === 'psychological');

    if (archetypal) parts.push(archetypal.text);
    if (mystical) parts.push(mystical.text);
    if (psychological && psychological.weight > 0.5) {
      // Extract only the poetic parts of psychological
      const poeticPsych = this.extractPoetic(psychological.text);
      if (poeticPsych) parts.push(poeticPsych);
    }

    return parts.join(' ');
  }

  synthesizeDirect(outputs) {
    // Only practical, most direct form
    const practical = outputs.find(o => o.type === 'practical');
    if (practical) {
      // Strip to just the action
      return practical.text;
    }

    // Fallback to highest weighted
    return outputs[0].text;
  }

  synthesizeNurturing(outputs) {
    // Blend psychological + practical with gentle transitions
    const parts = [];

    const psychological = outputs.find(o => o.type === 'psychological');
    const practical = outputs.find(o => o.type === 'practical');
    const archetypal = outputs.find(o => o.type === 'archetypal');

    if (psychological) parts.push(psychological.text);
    if (practical) {
      // Soften the practical advice
      const softenedPractical = this.softenLanguage(practical.text);
      parts.push(softenedPractical);
    }
    if (archetypal && archetypal.weight > 0.5) {
      parts.push(archetypal.text);
    }

    return parts.join(' ');
  }

  synthesizeBalanced(outputs) {
    // Take top 2-3 agents by weight
    const parts = outputs.slice(0, 3).map(o => o.text);
    return parts.join(' ');
  }

  extractPoetic(text) {
    // Extract only metaphorical/poetic language from psychological output
    // Remove explicit CBT/DBT skill references
    const lines = text.split('.');
    const poetic = lines.filter(line => {
      const lower = line.toLowerCase();
      return !lower.includes('dbt skill') &&
             !lower.includes('practice') &&
             !lower.includes('notice if');
    });

    return poetic.join('.').trim();
  }

  softenLanguage(text) {
    // Make language more gentle
    return text
      .replace('You must', 'You might consider')
      .replace('Do this', 'It could help to')
      .replace('Warning:', 'Gentle reminder:')
      .replace('Bottom line:', 'What this means is');
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LUNATIQ ENGINE - Main orchestrator
// ═══════════════════════════════════════════════════════════════════════════════

export class LunatiQEngine {
  constructor() {
    this.orchestrator = new LunatiQOrchestrator();
    this.blender = new EnsembleBlender();

    // Initialize agents
    this.agents = {
      archetypal: new ArchetypalAgent(),
      practical: new PracticalAgent(),
      psychological: new PsychologicalAgent(),
      relational: new RelationalAgent(),
      mystical: new MysticalAgent()
    };
  }

  /**
   * Generate interpretation for a single card in a spread
   *
   * @param {Object} card - The tarot card object
   * @param {String} position - Position meaning (e.g., "Past", "Present", "Future")
   * @param {Boolean} isReversed - Whether card is reversed
   * @param {Array} allCards - All cards in the spread (for relational analysis)
   * @param {Array} positions - All position objects (for orchestrator)
   * @param {Object} userProfile - User's personality profile
   * @param {String} intention - User's reading intention
   * @param {Object} commProfile - Communication profile from AdaptiveLanguageEngine
   * @returns {String} - Final synthesized interpretation
   */
  generateCardInterpretation(
    card,
    position,
    isReversed,
    allCards,
    positions,
    userProfile,
    intention,
    commProfile
  ) {
    try {
      // Validate inputs
      if (!card || !position || !commProfile) {
        throw new Error('Missing required parameters for interpretation');
      }

      // Step 1: Compute activation levels via fuzzy orchestrator
      const { activations, features } = this.orchestrator.computeActivations(
        allCards,
        positions,
        userProfile,
        intention
      );

      // Step 2: Generate interpretation from each agent
      const context = {
        intention,
        profile: userProfile,
        cards: allCards,
        features,
        moonPhase: null // TODO: integrate moon phase if desired
      };

      const agentOutputs = {
        archetypal: this.agents.archetypal.generateInterpretation(
          card,
          position,
          isReversed,
          activations.archetypal,
          context
        ),
        practical: this.agents.practical.generateInterpretation(
          card,
          position,
          isReversed,
          activations.practical,
          context
        ),
        psychological: this.agents.psychological.generateInterpretation(
          card,
          position,
          isReversed,
          activations.psychological,
          context
        ),
        relational: this.agents.relational.generateInterpretation(
          card,
          position,
          isReversed,
          activations.relational,
          context
        ),
        mystical: this.agents.mystical.generateInterpretation(
          card,
          position,
          isReversed,
          activations.mystical,
          context
        )
      };

      // Step 3: Blend agent outputs via ensemble
      const blendedInterpretation = this.blender.blendInterpretations(
        agentOutputs,
        activations,
        commProfile,
        card,
        position
      );

      // Step 4: Apply final voice styling
      const finalInterpretation = this.applyVoiceStyling(
        blendedInterpretation,
        commProfile,
        card
      );

      return finalInterpretation;

    } catch (error) {
      console.error('LunatiQ interpretation failed:', error);
      // Fallback to basic interpretation
      return this.generateFallbackInterpretation(card, position, isReversed);
    }
  }

  applyVoiceStyling(interpretation, commProfile, card) {
    // Add emoji if appropriate
    if (commProfile.emojiUse && this.shouldAddEmoji(commProfile.primaryVoice)) {
      const emoji = AdaptiveLanguageEngine.getCardEmoji(card);
      return `${interpretation} ${emoji}`;
    }

    return interpretation;
  }

  shouldAddEmoji(voice) {
    return [
      'playful_explorer',
      'supportive_friend',
      'gentle_nurturer',
      'intuitive_mystic'
    ].includes(voice);
  }

  generateFallbackInterpretation(card, position, isReversed) {
    const meaning = isReversed ? card.reversedMeaning : card.uprightMeaning;
    return `${card.name} in the ${position} position: ${meaning}`;
  }

  /**
   * Generate interpretation for entire spread
   * Returns array of card interpretations
   */
  generateSpreadInterpretation(
    cards,
    positions,
    userProfile,
    intention,
    commProfile
  ) {
    try {
      if (!cards || cards.length === 0) {
        throw new Error('No cards provided for interpretation');
      }

      return cards.map((cardData, index) => {
        const position = positions[index];
        return {
          card: cardData,
          position: position.position,
          reversed: position.reversed,
          interpretation: this.generateCardInterpretation(
            cardData,
            position.position,
            position.reversed,
            cards,
            positions,
            userProfile,
            intention,
            commProfile
          )
        };
      });

    } catch (error) {
      console.error('Spread interpretation failed:', error);
      throw error;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

export default LunatiQEngine;
