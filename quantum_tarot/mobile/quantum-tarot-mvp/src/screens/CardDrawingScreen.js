/**
 * CARD DRAWING SCREEN - Quantum shuffle and draw
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { MorphText, MatrixRain, NeonText, LPMUDText } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

export default function CardDrawingScreen({ route, navigation }) {
  const { spread, intention } = route.params;
  const [phase, setPhase] = useState('shuffling'); // shuffling, drawing, complete

  useEffect(() => {
    // Simulate quantum shuffle
    setTimeout(() => setPhase('drawing'), 2000);
    setTimeout(() => {
      // Navigate to reading with drawn cards
      // TODO: Implement actual quantum RNG
      const drawnCards = [
        { cardIndex: 0, reversed: false, position: 'Past' },
        { cardIndex: 1, reversed: true, position: 'Present' },
        { cardIndex: 2, reversed: false, position: 'Future' }
      ];
      navigation.replace('Reading', { cards: drawnCards, spread, intention });
    }, 4000);
  }, []);

  return (
    <View style={styles.container}>
      {/* Matrix rain background */}
      <MatrixRain width={SCREEN_WIDTH} height={SCREEN_HEIGHT} speed={30} />

      {/* Status display */}
      <View style={styles.statusBox}>
        {phase === 'shuffling' && (
          <>
            <MorphText
              color={NEON_COLORS.hiGreen}
              style={styles.statusTitle}
              morphSpeed={60}
            >
              SHUFFLING DECK...
            </MorphText>

            <LPMUDText style={styles.statusText}>
              $HIC${'>'} Generating quantum entropy...$NOR${'\n'}
              $HIC${'>'} Mixing with SHA-256...$NOR${'\n'}
              $HIC${'>'} Integrating intention vector...$NOR$
            </LPMUDText>
          </>
        )}

        {phase === 'drawing' && (
          <>
            <MorphText
              color={NEON_COLORS.hiCyan}
              style={styles.statusTitle}
              morphSpeed={60}
            >
              DRAWING CARDS...
            </MorphText>

            <LPMUDText style={styles.statusText}>
              $HIY${'>'} Card 1/3 drawn...$NOR${'\n'}
              $HIY${'>'} Card 2/3 drawn...$NOR${'\n'}
              $HIY${'>'} Card 3/3 drawn...$NOR${'\n\n'}
              $HIG${'>'} Initializing AGI interpretation...$NOR$
            </LPMUDText>
          </>
        )}
      </View>

      {/* Bottom info */}
      <View style={styles.bottomBox}>
        <NeonText color={NEON_COLORS.dimCyan} style={styles.bottomText}>
          {'>'} QUANTUM RNG ACTIVE
        </NeonText>
        <NeonText color={NEON_COLORS.dimYellow} style={styles.bottomText}>
          TRUE RANDOMNESS | ZERO BIAS
        </NeonText>
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
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    alignItems: 'center',
  },
  statusTitle: {
    fontSize: 20,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  statusText: {
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 18,
    textAlign: 'left',
  },
  bottomBox: {
    position: 'absolute',
    bottom: 40,
    alignItems: 'center',
    gap: 5,
  },
  bottomText: {
    fontSize: 10,
    fontFamily: 'monospace',
  },
});
