/**
 * READING SCREEN - Display cards and interpretation
 */

import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import CyberpunkCard from '../components/CyberpunkCard';
import { NeonText, LPMUDText, FlickerText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

export default function ReadingScreen({ route, navigation }) {
  const { cards, spread, intention } = route.params;
  const [currentCardIndex, setCurrentCardIndex] = useState(0);

  const handleNextCard = () => {
    if (currentCardIndex < cards.length - 1) {
      setCurrentCardIndex(currentCardIndex + 1);
    }
  };

  const handlePrevCard = () => {
    if (currentCardIndex > 0) {
      setCurrentCardIndex(currentCardIndex - 1);
    }
  };

  const handleFinish = () => {
    navigation.navigate('Welcome');
  };

  const currentCard = cards[currentCardIndex];

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIC${'>'} YOUR READING$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.headerSubtitle}>
            CARD {currentCardIndex + 1} OF {cards.length}
          </NeonText>
        </View>

        {/* Intention reminder */}
        <View style={styles.intentionBox}>
          <LPMUDText style={styles.intentionText}>
            $HIY$INTENTION:$NOR$ {intention || 'General guidance'}
          </LPMUDText>
        </View>

        {/* Card display */}
        <CyberpunkCard
          cardIndex={currentCard.cardIndex}
          reversed={currentCard.reversed}
          position={currentCard.position}
        />

        {/* AGI Interpretation */}
        <View style={styles.interpretationBox}>
          <LPMUDText style={styles.interpretationTitle}>
            $HIM${'>'} AGI INTERPRETATION$NOR$
          </LPMUDText>

          <LPMUDText style={styles.interpretationText}>
            $HIW$POSITION:$NOR$ {currentCard.position}{'\n'}
            $HIW$ORIENTATION:$NOR$ {currentCard.reversed ? '$HIR$REVERSED$NOR$' : '$HIG$UPRIGHT$NOR$'}{'\n\n'}

            $HIC$ARCHETYPAL LAYER:$NOR${'\n'}
            This card speaks to the universal pattern of [archetypal meaning].
            In your context, it suggests...{'\n\n'}

            $HIG$PRACTICAL LAYER:$NOR${'\n'}
            Actionable insight: [concrete guidance based on personality profile]{'\n\n'}

            $HIM$PSYCHOLOGICAL LAYER:$NOR${'\n'}
            Shadow work: [deep psychological insight]{'\n\n'}

            $HIY$SYNTHESIS:$NOR${'\n'}
            [Integrated interpretation from all 5 agents + personality weighting]
          </LPMUDText>
        </View>

        {/* Navigation */}
        <View style={styles.navRow}>
          {currentCardIndex > 0 && (
            <TouchableOpacity onPress={handlePrevCard} style={styles.navButton}>
              <NeonText color={NEON_COLORS.dimCyan} style={styles.navButtonText}>
                {'[ ← PREV ]'}
              </NeonText>
            </TouchableOpacity>
          )}

          {currentCardIndex < cards.length - 1 ? (
            <TouchableOpacity onPress={handleNextCard} style={styles.navButton}>
              <FlickerText color={NEON_COLORS.hiCyan} style={styles.navButtonText}>
                {'[ NEXT → ]'}
              </FlickerText>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity onPress={handleFinish} style={styles.finishButton}>
              <FlickerText color={NEON_COLORS.hiGreen} style={styles.finishButtonText}>
                {'[ COMPLETE ]'}
              </FlickerText>
            </TouchableOpacity>
          )}
        </View>

        {/* Card position indicators */}
        <View style={styles.indicatorRow}>
          {cards.map((_, i) => (
            <View
              key={i}
              style={[
                styles.indicator,
                {
                  borderColor: i === currentCardIndex ? NEON_COLORS.hiCyan : NEON_COLORS.dimCyan,
                  backgroundColor: i === currentCardIndex ? NEON_COLORS.cyan : 'transparent',
                }
              ]}
            />
          ))}
        </View>

        <View style={styles.spacer} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  content: {
    padding: 20,
  },
  header: {
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  headerTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 5,
    lineHeight: 22,
  },
  headerSubtitle: {
    fontSize: 11,
    fontFamily: 'monospace',
  },
  intentionBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.dimYellow,
    padding: 12,
    marginBottom: 20,
    backgroundColor: '#0a0a00',
  },
  intentionText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  interpretationBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimMagenta,
    padding: 15,
    marginVertical: 20,
    backgroundColor: '#000000',
  },
  interpretationTitle: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 12,
    lineHeight: 18,
  },
  interpretationText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 17,
  },
  navRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    gap: 10,
  },
  navButton: {
    flex: 1,
    padding: 15,
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
  },
  navButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
  finishButton: {
    flex: 1,
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiGreen,
    alignItems: 'center',
  },
  finishButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  indicatorRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
  },
  indicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
  },
  spacer: {
    height: 40,
  },
});
