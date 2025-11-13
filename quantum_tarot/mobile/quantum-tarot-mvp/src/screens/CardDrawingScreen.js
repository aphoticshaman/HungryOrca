/**
 * CARD DRAWING SCREEN - Quantum shuffle and draw
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Dimensions, Platform } from 'react-native';
import { MorphText, MatrixRain, NeonText, LPMUDText } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';
import { performReading } from '../utils/quantumRNG';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Uniform monospace font
const MONOSPACE_FONT = Platform.select({
  ios: 'Courier',
  android: 'monospace',
  default: 'Courier New',
});

export default function CardDrawingScreen({ route, navigation }) {
  const { spreadType, intention, readingType, zodiacSign, birthdate } = route.params;
  const [phase, setPhase] = useState('initializing'); // initializing, shuffling, drawing, complete
  const [statusLines, setStatusLines] = useState([]);
  const [cardCount, setCardCount] = useState(0);
  const [totalCards, setTotalCards] = useState(0);

  useEffect(() => {
    performQuantumDraw();
  }, []);

  async function performQuantumDraw() {
    try {
      // Phase 1: Initializing
      setPhase('initializing');
      setStatusLines([
        '> Initializing quantum entropy generator...',
        '> Hardware RNG active',
        '> Mixing intention vector...',
        '> ',
        '> Thank you for your patience...',
        '> We\'re preparing something special for you!'
      ]);
      await sleep(1200); // Faster init

      // Phase 2: Shuffling
      setPhase('shuffling');
      setStatusLines([
        '> Generating quantum random bytes...',
        '> Applying Fisher-Yates shuffle...',
        '> SHA-256 entropy mixing...',
        '> Deck randomization complete!',
        '> ',
        '> Your unique quantum signature is being created...'
      ]);
      await sleep(1500); // Faster shuffle

      // Phase 3: Drawing cards
      setPhase('drawing');
      const readingData = await performReading(spreadType, intention);
      const { cards, quantumSeed, timestamp } = readingData;
      setTotalCards(cards.length);

      // Personalized messages for each card draw
      const drawMessages = [
        'Your first card is emerging from the quantum field...',
        'The second card reveals itself to you...',
        'Drawing your third card with intention...',
        'The fourth card appears in your spread...',
        'Your fifth card is materializing...',
        'The sixth card shows its face...',
        'Drawing the seventh card for deeper insight...',
        'Your eighth card emerges from possibility...',
        'The ninth card completes the pattern...',
        'Your final card brings the reading together...'
      ];

      // Animate card draws with personalized messages
      for (let i = 0; i < cards.length; i++) {
        setCardCount(i + 1);
        const messageIndex = Math.min(i, drawMessages.length - 1);
        setStatusLines([
          `> ${drawMessages[messageIndex]}`,
          `> `,
          `> Position: ${cards[i].position}`,
          `> Card ${i + 1} of ${cards.length}`,
          `> Orientation: ${cards[i].reversed ? '⟲ REVERSED' : '⟳ UPRIGHT'}`,
          `> `,
          `> Thank you for your patience...`
        ]);
        await sleep(600); // Faster = smoother UX
      }

      // Phase 4: Complete
      setPhase('complete');
      setStatusLines([
        '> All cards drawn successfully! ✓',
        `> `,
        `> Quantum signature: ${quantumSeed.substring(0, 20)}...`,
        '> ',
        '> Initializing LunatiQ AGI interpretation engine...',
        '> Loading astrological correspondences...',
        '> Analyzing spread patterns...',
        '> Personalizing your reading...',
        '> ',
        '> Thank you for your patience!',
        '> We look forward to sharing your insights...'
      ]);
      await sleep(1500); // Faster completion

      // Navigate to reading
      navigation.replace('Reading', {
        cards,
        spreadType,
        intention,
        readingType,
        zodiacSign,
        birthdate,
        quantumSeed,
        timestamp
      });

    } catch (error) {
      console.error('Quantum draw error:', error);
      setStatusLines([
        '> ERROR: Quantum draw failed',
        '> ' + error.message,
        '> Retrying...'
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
      case 'shuffling': return 'QUANTUM SHUFFLE...';
      case 'drawing': return `DRAWING CARDS... [${cardCount}/${totalCards}] ${Math.round((cardCount / totalCards) * 100)}%`;
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
            morphSpeed={60}
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

          {/* Progress bar */}
          {phase === 'drawing' && cardCount > 0 && (
            <View style={styles.progressContainer}>
              <View style={styles.progressBar}>
                <View
                  style={[
                    styles.progressFill,
                    {
                      width: `${(cardCount / totalCards) * 100}%`,
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
});
