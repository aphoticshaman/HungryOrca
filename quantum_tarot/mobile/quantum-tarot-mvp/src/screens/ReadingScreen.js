/**
 * READING SCREEN - Display cards and AGI interpretation
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import CyberpunkCard from '../components/CyberpunkCard';
import { NeonText, LPMUDText, FlickerText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';
import { interpretReading } from '../utils/agiInterpretation';

export default function ReadingScreen({ route, navigation }) {
  const { cards, spreadType, intention, readingType, zodiacSign, birthdate, quantumSeed, timestamp } = route.params;
  const [currentCardIndex, setCurrentCardIndex] = useState(0);
  const [reading, setReading] = useState(null);

  useEffect(() => {
    // Generate AGI interpretation
    const fullReading = interpretReading(cards, spreadType, intention, {
      readingType,
      zodiacSign,
      birthdate
    });
    setReading(fullReading);
  }, []);

  if (!reading) {
    return (
      <View style={styles.container}>
        <ScanLines />
        <View style={styles.loadingContainer}>
          <FlickerText color={NEON_COLORS.hiCyan} style={styles.loadingText}>
            GENERATING INTERPRETATION...
          </FlickerText>
        </View>
      </View>
    );
  }

  const currentInterpretation = reading.interpretations[currentCardIndex];
  const currentCard = cards[currentCardIndex];
  const astroContext = reading.astrologicalContext;

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

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header with quantum seed */}
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIC${'>'} YOUR READING$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.headerSubtitle}>
            CARD {currentCardIndex + 1} OF {cards.length}
          </NeonText>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.quantumSeed}>
            QUANTUM SEED: {quantumSeed}
          </NeonText>
        </View>

        {/* Astrological Context Banner */}
        <View style={styles.astroBox}>
          <LPMUDText style={styles.astroText}>
            $HIM$ASTRO CONTEXT:$NOR${'\n'}
            $HIY${astroContext.moonPhase.name}$NOR$ |
            $HIC${astroContext.natalSign}$NOR$ |
            {astroContext.mercuryRetrograde.isRetrograde ?
              ' $HIR$MERCURY RETROGRADE$NOR$' :
              ' $HIG$Mercury Direct$NOR$'}
            {'\n'}
            $NOR${astroContext.planetaryInfluences.dominantPlanet} energy - {astroContext.planetaryInfluences.energy}
          </LPMUDText>
        </View>

        {/* Intention reminder */}
        <View style={styles.intentionBox}>
          <LPMUDText style={styles.intentionText}>
            $HIY$INTENTION:$NOR$ {intention}
          </LPMUDText>
        </View>

        {/* Card display */}
        <CyberpunkCard
          cardIndex={currentCard.cardIndex}
          reversed={currentCard.reversed}
          position={currentCard.position}
        />

        {/* AGI Interpretation - 5 Layers */}
        <View style={styles.interpretationBox}>
          <LPMUDText style={styles.interpretationTitle}>
            $HIM${'>'} LUNATIQ AGI INTERPRETATION$NOR$
          </LPMUDText>

          {/* Card Data Header */}
          <LPMUDText style={styles.interpretationSection}>
            $HIC${'>'} CARD DATA STREAM {'<'}$NOR${'\n'}
            $HIY$━━━━━━━━━━━━━━━━━━━━$NOR${'\n\n'}
            $HIM$IDENTITY:$NOR$ {currentInterpretation.cardData.name}{'\n'}
            $HIM$ARCANA:$NOR$ {currentInterpretation.cardData.arcana?.toUpperCase()}{'\n'}
            $HIM$ELEMENT:$NOR$ {currentInterpretation.cardData.element?.toUpperCase() || 'SPIRIT'}{'\n'}
            $HIM$NUMEROLOGY:$NOR$ {currentInterpretation.cardData.numerology}{'\n'}
            $HIM$POSITION:$NOR$ {currentCard.position}{'\n'}
            $HIM$ORIENTATION:$NOR$ {currentCard.reversed ? '$HIR$REVERSED$NOR$' : '$HIG$UPRIGHT$NOR$'}{'\n\n'}
            $HIC$SYMBOLS:$NOR$ {currentInterpretation.cardData.symbols?.slice(0, 4).join(', ') || 'N/A'}{'\n\n'}
            $HIY$━━━━━━━━━━━━━━━━━━━━$NOR$
          </LPMUDText>

          {/* Layer 1: Archetypal */}
          <LPMUDText style={styles.interpretationSection}>
            $HIC$━━ ARCHETYPAL LAYER ━━$NOR${'\n'}
            $NOR${currentInterpretation.layers.archetypal.core_meaning}{'\n\n'}
            $HIG$UPRIGHT KEYWORDS:$NOR$ {currentInterpretation.cardData.keywords?.upright?.slice(0, 5).join(', ') || 'N/A'}{'\n\n'}
            $HIR$REVERSED KEYWORDS:$NOR$ {currentInterpretation.cardData.keywords?.reversed?.slice(0, 5).join(', ') || 'N/A'}
          </LPMUDText>

          {/* Layer 2: Contextual */}
          <LPMUDText style={styles.interpretationSection}>
            $HIC$━━ CONTEXTUAL LAYER ━━$NOR${'\n'}
            $NOR${currentInterpretation.layers.contextual.position_significance}{'\n\n'}
            {currentInterpretation.layers.contextual.intention_alignment}
          </LPMUDText>

          {/* Layer 3: Psychological */}
          <LPMUDText style={styles.interpretationSection}>
            $HIC$━━ PSYCHOLOGICAL LAYER ━━$NOR${'\n'}
            $HIM$Shadow Work:$NOR$ {currentInterpretation.layers.psychological.shadow_work}{'\n\n'}
            $HIM$Integration:$NOR$ {currentInterpretation.layers.psychological.integration_path}
          </LPMUDText>

          {/* Layer 4: Practical */}
          <LPMUDText style={styles.interpretationSection}>
            $HIC$━━ PRACTICAL LAYER ━━$NOR${'\n'}
            $HIG$Action Steps:$NOR${'\n'}
            {currentInterpretation.layers.practical.action_steps.map((step, i) =>
              `  ${i + 1}. ${step}\n`
            ).join('')}
            {'\n'}$HIY$Focus On:$NOR$ {currentInterpretation.layers.practical.what_to_focus_on}
          </LPMUDText>

          {/* Layer 5: Synthesis */}
          <LPMUDText style={styles.interpretationSection}>
            $HIC$━━ SYNTHESIS ━━$NOR${'\n'}
            $HIW${currentInterpretation.layers.synthesis.core_message}$NOR${'\n\n'}
            $HIG${currentInterpretation.layers.synthesis.next_steps}$NOR$
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  header: {
    marginBottom: 15,
    paddingBottom: 12,
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
    marginBottom: 5,
  },
  quantumSeed: {
    fontSize: 8,
    fontFamily: 'monospace',
    marginTop: 5,
  },
  astroBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.hiMagenta,
    padding: 12,
    marginBottom: 15,
    backgroundColor: '#0a000a',
  },
  astroText: {
    fontSize: 10,
    fontFamily: 'monospace',
    lineHeight: 15,
  },
  intentionBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.dimYellow,
    padding: 12,
    marginBottom: 15,
    backgroundColor: '#0a0a00',
  },
  intentionText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  interpretationBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 15,
    marginVertical: 15,
    backgroundColor: '#000000',
  },
  interpretationTitle: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 15,
    lineHeight: 18,
  },
  interpretationSection: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 17,
    marginBottom: 15,
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
