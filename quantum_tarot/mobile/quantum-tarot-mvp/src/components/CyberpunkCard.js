/**
 * CYBERPUNK TAROT CARD
 * Terminal hacker aesthetic with neon glows and Matrix effects
 */

import React, { useState, useRef, useEffect } from 'react';
import { View, TouchableOpacity, Animated, StyleSheet, Dimensions } from 'react-native';
import { NeonText, LPMUDText, FlickerText, GlitchText, MorphText, MatrixRain, ScanLines } from './TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';
import { CARD_DATABASE } from '../data/cardDatabase';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

/**
 * CYBERPUNK CARD - Flippable card with terminal effects
 */
export default function CyberpunkCard({ cardIndex, reversed, position, onReveal }) {
  const [isFlipped, setIsFlipped] = useState(false);
  const [isRevealing, setIsRevealing] = useState(false);
  const flipAnim = useRef(new Animated.Value(0)).current;
  const glowAnim = useRef(new Animated.Value(0)).current;

  const card = CARD_DATABASE[cardIndex] || CARD_DATABASE[0];

  useEffect(() => {
    // Pulsing glow effect
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: false,
        }),
        Animated.timing(glowAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: false,
        }),
      ])
    ).start();
  }, []);

  const handleFlip = () => {
    Animated.spring(flipAnim, {
      toValue: isFlipped ? 0 : 180,
      friction: 8,
      tension: 10,
      useNativeDriver: true,
    }).start();
    setIsFlipped(!isFlipped);
  };

  const handleReveal = () => {
    setIsRevealing(true);
    setTimeout(() => {
      setIsRevealing(false);
      onReveal && onReveal();
    }, 1500);
  };

  const frontInterpolate = flipAnim.interpolate({
    inputRange: [0, 180],
    outputRange: ['0deg', '180deg'],
  });

  const backInterpolate = flipAnim.interpolate({
    inputRange: [0, 180],
    outputRange: ['180deg', '360deg'],
  });

  const glowColor = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['rgba(0, 255, 255, 0.2)', 'rgba(255, 0, 255, 0.8)'],
  });

  const cardColor = card.element === 'fire' ? NEON_COLORS.hiRed :
                     card.element === 'water' ? NEON_COLORS.hiCyan :
                     card.element === 'air' ? NEON_COLORS.hiYellow :
                     card.element === 'earth' ? NEON_COLORS.hiGreen :
                     NEON_COLORS.hiMagenta;

  return (
    <View style={styles.cardContainer}>
      {/* Matrix rain background during reveal */}
      {isRevealing && (
        <View style={StyleSheet.absoluteFill}>
          <MatrixRain width={SCREEN_WIDTH - 40} height={400} speed={30} />
        </View>
      )}

      {/* Front of card */}
      <Animated.View
        style={[
          styles.cardFace,
          styles.cardFront,
          {
            transform: [{ rotateY: frontInterpolate }],
            borderColor: glowColor,
          },
        ]}
      >
        <ScanLines />

        {/* Card title with glitch effect */}
        <View style={styles.cardHeader}>
          <GlitchText
            style={styles.cardNumber}
            glitchChance={0.05}
          >
            {card.arcana === 'major' ? `[${card.id}]` : `[${card.suit?.toUpperCase()}]`}
          </GlitchText>

          <NeonText
            color={cardColor}
            style={styles.cardTitle}
          >
            {card.name.toUpperCase()}
          </NeonText>

          {reversed && (
            <FlickerText
              color={NEON_COLORS.hiRed}
              style={styles.reversedLabel}
            >
              [REVERSED]
            </FlickerText>
          )}
        </View>

        {/* ASCII Art placeholder */}
        <View style={styles.asciiContainer}>
          <LPMUDText style={styles.asciiArt}>
            {generateCyberpunkASCII(card)}
          </LPMUDText>
        </View>

        {/* Position label */}
        <NeonText
          color={NEON_COLORS.dimCyan}
          style={styles.positionLabel}
        >
          {'>'} {position} {'<'}
        </NeonText>

        {/* Flip button */}
        <TouchableOpacity onPress={handleFlip} style={styles.flipButton}>
          <FlickerText color={NEON_COLORS.hiYellow} style={styles.flipButtonText}>
            [ REVEAL DATA ] ▶
          </FlickerText>
        </TouchableOpacity>
      </Animated.View>

      {/* Back of card */}
      <Animated.View
        style={[
          styles.cardFace,
          styles.cardBack,
          {
            transform: [{ rotateY: backInterpolate }],
            borderColor: glowColor,
          },
        ]}
      >
        <ScanLines />

        {/* Card data with LPMUD color codes */}
        <View style={styles.cardData}>
          <LPMUDText style={styles.dataText}>
            $HIC${'>'} CARD DATA STREAM {'<'}$NOR${'\n'}
            $HIY$━━━━━━━━━━━━━━━━━━━━$NOR${'\n\n'}

            $HIM$IDENTITY:$NOR$ {card.name}{'\n'}
            $HIM$ELEMENT:$NOR$ {card.element?.toUpperCase() || 'N/A'}{'\n'}
            $HIM$NUMEROLOGY:$NOR$ {card.numerology}{'\n\n'}

            $HIC$SYMBOLS:$NOR${'\n'}
            {card.symbols?.slice(0, 3).map(s => `  • ${s}`).join('\n') || 'N/A'}{'\n\n'}

            $HIW$MEANING:$NOR${'\n'}
            {card.description?.substring(0, 150) || 'No data available'}...{'\n\n'}

            $HIG$UPRIGHT:$NOR${'\n'}
            {card.keywords?.upright?.slice(0, 3).join(', ') || 'N/A'}{'\n\n'}

            $HIR$REVERSED:$NOR${'\n'}
            {card.keywords?.reversed?.slice(0, 3).join(', ') || 'N/A'}{'\n\n'}

            $HIY$ADVICE:$NOR${'\n'}
            {card.advice || 'Trust the flow'}{'\n\n'}

            $HIY$━━━━━━━━━━━━━━━━━━━━$NOR$
          </LPMUDText>
        </View>

        {/* Flip back button */}
        <TouchableOpacity onPress={handleFlip} style={styles.flipButton}>
          <FlickerText color={NEON_COLORS.hiCyan} style={styles.flipButtonText}>
            ◀ [ CLOSE DATA ]
          </FlickerText>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
}

/**
 * Generate cyberpunk ASCII art placeholder
 * TODO: Replace with actual card-specific ASCII when ready
 */
function generateCyberpunkASCII(card) {
  const isMajor = card.arcana === 'major';

  if (isMajor) {
    return `$HIC$
    ╔═══════════════╗
    ║   $HIY$◢◣$HIC$   $HIY$◢◣$HIC$   ║
    ║  $HIM$◢$HIW$███$HIM$◣$HIC$ $HIM$◢$HIW$███$HIM$◣$HIC$  ║
    ║  $HIW$█$HIC$░$HIY$☆$HIC$░$HIW$█$HIC$ $HIW$█$HIC$░$HIY$☆$HIC$░$HIW$█$HIC$  ║
    ║  $HIM$◥$HIW$███$HIM$◤$HIC$ $HIM$◥$HIW$███$HIM$◤$HIC$  ║
    ║   $HIY$◥◤$HIC$   $HIY$◥◤$HIC$   ║
    ║               ║
    ║  $HIY$M A J O R$HIC$  ║
    ║  $HIM$A R C A N A$HIC$  ║
    ╚═══════════════╝
$NOR$`;
  } else {
    const suitSymbol = card.suit === 'wands' ? '|' :
                      card.suit === 'cups' ? '◡' :
                      card.suit === 'swords' ? '†' :
                      card.suit === 'pentacles' ? '◯' : '?';

    return `$HIG$
    ┌───────────────┐
    │   $HIY$${suitSymbol}$HIG$   $HIY$${suitSymbol}$HIG$   $HIY$${suitSymbol}$HIG$   │
    │               │
    │   $HIW$${suitSymbol}$HIG$   $HIM$${suitSymbol}$HIG$   $HIW$${suitSymbol}$HIG$   │
    │               │
    │   $HIY$${suitSymbol}$HIG$   $HIY$${suitSymbol}$HIG$   $HIY$${suitSymbol}$HIG$   │
    └───────────────┘
$NOR$`;
  }
}

const styles = StyleSheet.create({
  cardContainer: {
    width: SCREEN_WIDTH - 40,
    height: 450,
    marginVertical: 10,
    alignSelf: 'center',
  },
  cardFace: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    backgroundColor: '#000000',
    borderWidth: 2,
    borderRadius: 8,
    padding: 20,
    backfaceVisibility: 'hidden',
  },
  cardFront: {
    justifyContent: 'space-between',
  },
  cardBack: {
    justifyContent: 'space-between',
  },
  cardHeader: {
    alignItems: 'center',
    marginBottom: 15,
  },
  cardNumber: {
    fontSize: 16,
    fontFamily: 'monospace',
    marginBottom: 5,
  },
  cardTitle: {
    fontSize: 24,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  reversedLabel: {
    fontSize: 14,
    fontFamily: 'monospace',
    marginTop: 5,
  },
  asciiContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  asciiArt: {
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  positionLabel: {
    fontSize: 14,
    fontFamily: 'monospace',
    textAlign: 'center',
    marginVertical: 10,
  },
  flipButton: {
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    borderRadius: 4,
  },
  flipButtonText: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  cardData: {
    flex: 1,
  },
  dataText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
});
