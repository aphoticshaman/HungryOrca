/**
 * CARD DRAWING SCREEN - Quantum shuffle and draw
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { MorphText, MatrixRain, NeonText, LPMUDText } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';
import { performReading } from '../utils/quantumRNG';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

export default function CardDrawingScreen({ route, navigation }) {
  const { spreadType, intention, readingType, zodiacSign, birthdate } = route.params;
  const [phase, setPhase] = useState('initializing'); // initializing, shuffling, drawing, complete
  const [statusLines, setStatusLines] = useState([]);
  const [cardCount, setCardCount] = useState(0);

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
        '> Mixing intention vector...'
      ]);
      await sleep(1500);

      // Phase 2: Shuffling
      setPhase('shuffling');
      setStatusLines([
        '> Generating quantum random bytes...',
        '> Applying Fisher-Yates shuffle...',
        '> SHA-256 entropy mixing...',
        '> Deck randomization complete'
      ]);
      await sleep(2000);

      // Phase 3: Drawing cards
      setPhase('drawing');
      const readingData = await performReading(spreadType, intention);
      const { cards, quantumSeed, timestamp } = readingData;

      // Animate card draws
      for (let i = 0; i < cards.length; i++) {
        setCardCount(i + 1);
        setStatusLines([
          `> Drawing card ${i + 1}/${cards.length}...`,
          `> Position: ${cards[i].position}`,
          `> Card index: ${cards[i].cardIndex}`,
          `> Orientation: ${cards[i].reversed ? 'REVERSED' : 'UPRIGHT'}`
        ]);
        await sleep(800);
      }

      // Phase 4: Complete
      setPhase('complete');
      setStatusLines([
        '> All cards drawn',
        `> Quantum seed: ${quantumSeed.substring(0, 20)}...`,
        '> Initializing AGI interpretation...',
        '> Loading astrological data...',
        '> Preparing reading...'
      ]);
      await sleep(2000);

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
      case 'drawing': return `DRAWING CARDS... [${cardCount}]`;
      case 'complete': return 'COMPLETE';
      default: return 'PROCESSING...';
    }
  };

  return (
    <View style={styles.container}>
      {/* Matrix rain background */}
      <MatrixRain width={SCREEN_WIDTH} height={SCREEN_HEIGHT} speed={30} />

      {/* Status display */}
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
                    width: `${(cardCount / statusLines.length) * 100}%`,
                    backgroundColor: getPhaseColor()
                  }
                ]}
              />
            </View>
          </View>
        )}
      </View>

      {/* Bottom info */}
      <View style={styles.bottomBox}>
        <NeonText color={NEON_COLORS.dimCyan} style={styles.bottomText}>
          {'>'} QUANTUM RNG | HARDWARE ENTROPY
        </NeonText>
        <NeonText color={NEON_COLORS.dimYellow} style={styles.bottomText}>
          TRUE RANDOMNESS | ZERO BIAS
        </NeonText>
        <LPMUDText style={styles.intentionText}>
          $HIM$INTENTION:$NOR$ {intention.substring(0, 40)}...
        </LPMUDText>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  statusBox: {
    width: SCREEN_WIDTH - 40,
    padding: 30,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
  },
  statusTitle: {
    fontSize: 20,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  statusContent: {
    gap: 8,
  },
  statusLine: {
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 16,
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
  bottomBox: {
    position: 'absolute',
    bottom: 40,
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 20,
  },
  bottomText: {
    fontSize: 10,
    fontFamily: 'monospace',
  },
  intentionText: {
    fontSize: 9,
    fontFamily: 'monospace',
    marginTop: 5,
    textAlign: 'center',
    lineHeight: 13,
  },
});
