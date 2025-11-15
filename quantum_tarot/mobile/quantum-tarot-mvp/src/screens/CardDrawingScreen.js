/**
 * CARD DRAWING SCREEN - Card shuffle and draw
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Dimensions, Platform } from 'react-native';
import { MorphText, MatrixRain, NeonText, LPMUDText } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';
import { performReading } from '../utils/quantumRNG';
import { interpretCard } from '../utils/agiInterpretation';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Uniform monospace font
const MONOSPACE_FONT = Platform.select({
  ios: 'Courier',
  android: 'monospace',
  default: 'Courier New',
});

/**
 * Format interpretation object into readable text for encrypted reveal
 */
function formatInterpretation(interpretation) {
  if (!interpretation) {
    console.error('formatInterpretation: interpretation is undefined');
    return 'ERROR: No interpretation generated';
  }

  const { cardData, layers, position, reversed } = interpretation;

  if (!cardData || !layers) {
    console.error('formatInterpretation: Missing cardData or layers', interpretation);
    return `ERROR: Invalid interpretation structure\n${JSON.stringify(interpretation, null, 2)}`;
  }

  try {
    let text = '';

    // Card header
    text += `${cardData.name || 'Unknown Card'}${reversed ? ' (Reversed)' : ''}\n`;
    text += `Position: ${position || 'Unknown'}\n`;
    text += `Element: ${layers.archetypal?.element || 'Spirit'}\n\n`;

    // Archetypal layer
    text += `━━ ARCHETYPAL MEANING ━━\n\n`;
    text += `${layers.archetypal?.core_meaning || 'No meaning available'}\n\n\n`;

    // Contextual layer
    text += `━━ IN YOUR SITUATION ━━\n\n`;
    text += `${layers.contextual?.position_significance || 'No context'}\n\n`;
    text += `${layers.contextual?.intention_alignment || 'No alignment'}\n\n\n`;

    // Psychological layer
    text += `━━ DEEPER INSIGHT ━━\n\n`;
    text += `${layers.psychological?.shadow_work || 'None'}\n\n`;
    text += `${layers.psychological?.integration_path || 'None'}\n\n\n`;

    // Practical layer
    text += `━━ ACTION STEPS ━━\n\n`;
    const steps = layers.practical?.action_steps || [];
    if (steps.length > 0) {
      steps.forEach((step, i) => {
        text += `${i + 1}. ${step}\n`;
      });
      text += `\n`;
    }
    text += `${layers.practical?.what_to_focus_on || 'General focus'}\n\n\n`;

    // Synthesis
    text += `━━ KEY MESSAGE ━━\n\n`;
    text += `${layers.synthesis?.core_message || 'No message'}\n\n`;
    text += `${layers.synthesis?.next_steps || 'Continue forward'}`;

    return text;
  } catch (error) {
    console.error('formatInterpretation error:', error);
    return `ERROR: Failed to format interpretation\n${error.message}`;
  }
}

// 31 Pro-tips inspired by CBT, DBT, Army MRT, and psychology
const PRO_TIPS = [
  '💡 Tarot works best when you\'re honest with yourself about what you\'re feeling.',
  '💡 Notice your first reaction to each card - it reveals what you need to hear.',
  '💡 Ask yourself: What can I control? Focus your energy there.',
  '💡 Readings aren\'t predictions - they\'re mirrors reflecting your current path.',
  '💡 Reversed cards aren\'t \'bad\' - they show what needs attention or balance.',
  '💡 Write down your intention before reading to stay focused.',
  '💡 Check the facts: Is your interpretation based on emotion or evidence?',
  '💡 Practice radical acceptance of what appears - then decide what to do next.',
  '💡 Your thoughts create your reality - examine your thinking patterns.',
  '💡 Major Arcana cards signal life-changing themes worth deep reflection.',
  '💡 Focus on what you can change, accept what you can\'t, find the wisdom between.',
  '💡 When stuck, ask: What would my wisest self do in this situation?',
  '💡 Notice patterns across readings - they reveal your blind spots.',
  '💡 Breathe deeply before viewing your spread to center yourself.',
  '💡 Difficult cards often carry the most important messages for growth.',
  '💡 The cards show possibilities, not certainties. You choose your path.',
  '💡 Trust your intuition - your subconscious knows more than you think.',
  '💡 Compare how you felt then vs. now to measure real progress.',
  '💡 Every ending in tarot creates space for a new beginning.',
  '💡 The universe speaks in symbols - what does this card mean to YOU?',
  '💡 Small daily actions compound into massive life changes over time.',
  '💡 Your past doesn\'t define you - your response to it does.',
  '💡 When emotions run high, pause and name what you\'re feeling.',
  '💡 Challenge catastrophic thinking: What\'s the evidence? What else could be true?',
  '💡 Progress isn\'t linear - setbacks are part of the growth process.',
  '💡 You can\'t control outcomes, but you can control your effort and attitude.',
  '💡 Self-compassion isn\'t self-indulgence - it\'s a requirement for growth.',
  '💡 The only constant is change - resistance creates suffering.',
  '💡 Your interpretation matters more than traditional meanings.',
  '💡 Celebrate small wins - they\'re proof you\'re moving forward.',
  '💡 When in doubt, act according to your values, not your fears.',
];

export default function CardDrawingScreen({ route, navigation }) {
  const { spreadType, intention, readingType, zodiacSign, birthdate, userProfile } = route.params;
  const [phase, setPhase] = useState('initializing'); // initializing, shuffling, drawing, complete
  const [statusLines, setStatusLines] = useState([]);
  const [cardCount, setCardCount] = useState(0);
  const [totalCards, setTotalCards] = useState(0);
  const [tipIndex, setTipIndex] = useState(0);

  useEffect(() => {
    performQuantumDraw();
  }, []);

  useEffect(() => {
    // Rotate pro-tips sequentially every 6 seconds max
    const tipInterval = setInterval(() => {
      setTipIndex((prev) => (prev + 1) % PRO_TIPS.length);
    }, 6000);
    return () => clearInterval(tipInterval);
  }, []);

  async function performQuantumDraw() {
    try {
      // Phase 1: Initializing
      setPhase('initializing');
      setStatusLines([
        'LunatIQ Tarot',
        '',
        'Preparing your reading...',
        '',
        'Thank you for your patience.',
      ]);
      await sleep(1200);

      // Phase 2: Shuffling
      setPhase('shuffling');
      setStatusLines([
        'LunatIQ Tarot',
        '',
        'Shuffling the deck...',
        'Mixing your intention...',
        '',
        'Thank you for your patience.',
      ]);
      await sleep(1500);

      // Phase 3: Drawing cards
      setPhase('drawing');
      const readingData = await performReading(spreadType, intention);
      const { cards, quantumSeed, timestamp } = readingData;
      setTotalCards(cards.length);

      // Simple card draw messages
      const drawMessages = [
        'Your first card is emerging...',
        'The second card reveals itself...',
        'Drawing your third card...',
        'The fourth card appears...',
        'Your fifth card is materializing...',
        'The sixth card shows its face...',
        'Drawing the seventh card...',
        'Your eighth card emerges...',
        'The ninth card completes the pattern...',
        'Your final card brings it all together...'
      ];

      // Animate card draws with personalized messages
      for (let i = 0; i < cards.length; i++) {
        setCardCount(i + 1);
        const messageIndex = Math.min(i, drawMessages.length - 1);
        setStatusLines([
          'LunatIQ Tarot',
          '',
          drawMessages[messageIndex],
          'Position: ' + cards[i].position,
          'Card ' + (i + 1) + ' of ' + cards.length,
          (cards[i].reversed ? '⟲ REVERSED' : '⟳ UPRIGHT'),
          '',
          'Thank you for your patience.',
        ]);
        await sleep(600);
      }

      // Phase 4: Complete
      setPhase('complete');
      setStatusLines([
        'LunatIQ Tarot',
        '',
        'All cards drawn! ✓',
        '',
        'Generating interpretations...',
        'Quantum processing...',
        '',
        'Thank you for your patience!',
      ]);
      await sleep(1000);

      // Generate individual card interpretations
      console.log('🎴 Generating interpretations for', cards.length, 'cards');
      const interpretations = cards.map((card, index) => {
        console.log(`🎴 Card ${index + 1}:`, card);
        const interpretation = interpretCard(
          card,
          intention,
          readingType,
          { zodiacSign, birthdate, userProfile }
        );
        console.log(`🎴 Interpretation ${index + 1}:`, interpretation ? 'Generated' : 'UNDEFINED');

        // Convert interpretation to readable text format
        const formatted = formatInterpretation(interpretation);
        console.log(`🎴 Formatted ${index + 1} length:`, formatted?.length);
        console.log(`🎴 Formatted ${index + 1} FULL TEXT:`, formatted);
        return formatted;
      });

      await sleep(500);

      // Navigate to card interpretation screen (card-by-card with MCQs)
      navigation.replace('CardInterpretation', {
        cards,
        interpretations,
        spreadType,
        intention,
        readingType,
        zodiacSign,
        birthdate,
        userProfile: userProfile || { zodiacSign, birthdate },
        quantumSeed,
        timestamp
      });

    } catch (error) {
      console.error('Draw error:', error);
      setStatusLines([
        'LunatIQ Tarot',
        '',
        'ERROR: Drawing failed',
        error.message,
        '',
        'Retrying...'
      ]);
      await sleep(2000);
      performQuantumDraw();
    }
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  const getPhaseColor = () => {
    switch (phase) {
      case 'initializing': return NEON_COLORS.hiYellow;
      case 'shuffling': return NEON_COLORS.hiCyan;
      case 'drawing': return NEON_COLORS.hiMagenta;
      case 'complete': return NEON_COLORS.hiGreen;
      default: return NEON_COLORS.hiWhite;
    }
  };

  const getPhaseTitle = () => {
    switch (phase) {
      case 'initializing': return 'INITIALIZING...';
      case 'shuffling': return 'SHUFFLING...';
      case 'drawing': return 'DRAWING... [' + cardCount + '/' + totalCards + ']';
      case 'complete': return 'COMPLETE';
      default: return 'PROCESSING...';
    }
  };

  return (
    <View style={styles.container}>
      {/* Matrix rain background */}
      <MatrixRain width={SCREEN_WIDTH} height={SCREEN_HEIGHT} speed={30} />

      {/* Status display - positioned to stay on screen */}
      <View style={styles.centerContent}>
        <View style={styles.statusBox}>
          <MorphText
            color={getPhaseColor()}
            style={styles.statusTitle}
            morphSpeed={400}
          >
            {getPhaseTitle()}
          </MorphText>

          <View style={styles.statusContent}>
            {statusLines.map((line, i) => (
              <LPMUDText key={i} style={styles.statusLine}>
                $HIC${line}$NOR$
              </LPMUDText>
            ))}
          </View>

          {/* Pro-tip rotating display */}
          <View style={styles.proTipContainer}>
            <NeonText color={NEON_COLORS.hiYellow} style={styles.proTip}>
              {PRO_TIPS[tipIndex]}
            </NeonText>
          </View>

          {/* Progress bar */}
          {phase === 'drawing' && cardCount > 0 && (
            <View style={styles.progressContainer}>
              <View style={styles.progressBar}>
                <View
                  style={[
                    styles.progressFill,
                    {
                      width: ((cardCount / totalCards) * 100) + '%',
                      backgroundColor: getPhaseColor()
                    }
                  ]}
                />
              </View>
            </View>
          )}

          {/* Card count inside status box */}
          {phase === 'drawing' && cardCount > 0 && (
            <View style={styles.cardCountContainer}>
              <NeonText color={getPhaseColor()} style={styles.cardCountText}>
                CARD {cardCount} / {totalCards}
              </NeonText>
            </View>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  statusBox: {
    width: SCREEN_WIDTH - 40,
    maxHeight: SCREEN_HEIGHT - 120,
    padding: 30,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
  },
  statusTitle: {
    fontSize: 20,
    fontFamily: MONOSPACE_FONT,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  statusContent: {
    // gap not supported in RN StyleSheet - use marginBottom on children
  },
  statusLine: {
    fontSize: 12,
    fontFamily: MONOSPACE_FONT,
    lineHeight: 16,
    marginBottom: 8, // Instead of gap on parent
  },
  progressContainer: {
    marginTop: 20,
  },
  progressBar: {
    height: 6,
    backgroundColor: NEON_COLORS.dimCyan,
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
  cardCountContainer: {
    marginTop: 20,
    alignItems: 'center',
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: NEON_COLORS.dimCyan,
  },
  cardCountText: {
    fontSize: 18,
    fontFamily: MONOSPACE_FONT,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  proTipContainer: {
    marginTop: 20,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: NEON_COLORS.dimYellow,
  },
  proTip: {
    fontSize: 11,
    fontFamily: MONOSPACE_FONT,
    lineHeight: 16,
    textAlign: 'center',
  },
});
