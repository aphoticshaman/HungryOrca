/**
 * CYBERPUNK TAROT CARD
 * Terminal hacker aesthetic with neon glows - simplified display
 */

import React, { useRef, useEffect } from 'react';
import { View, Animated, StyleSheet, Dimensions, Platform } from 'react-native';
import { NeonText, FlickerText, GlitchText, ScanLines } from './TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';
import { CARD_DATABASE } from '../data/cardDatabase';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

/**
 * CYBERPUNK CARD - Simplified card display
 */
export default function CyberpunkCard({ cardIndex, reversed, position }) {
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
      <Animated.View
        style={[
          styles.cardFace,
          {
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

          {/* Element badge */}
          <NeonText
            color={cardColor}
            style={styles.elementBadge}
          >
            {'['} {card.element?.toUpperCase() || 'SPIRIT'} {']'}
          </NeonText>
        </View>

        {/* Position label */}
        <NeonText
          color={NEON_COLORS.dimCyan}
          style={styles.positionLabel}
        >
          {'>'} {position} {'<'}
        </NeonText>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  cardContainer: {
    width: SCREEN_WIDTH - 40,
    height: 180,
    marginVertical: 10,
    alignSelf: 'center',
  },
  cardFace: {
    width: '100%',
    height: '100%',
    backgroundColor: '#000000',
    borderWidth: 2,
    borderRadius: 8,
    padding: 20,
    justifyContent: 'center',
  },
  cardHeader: {
    alignItems: 'center',
  },
  cardNumber: {
    fontSize: 16,
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    marginBottom: 5,
  },
  cardTitle: {
    fontSize: 28,
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    fontWeight: 'bold',
    textAlign: 'center',
  },
  reversedLabel: {
    fontSize: 14,
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    marginTop: 5,
  },
  elementBadge: {
    fontSize: 12,
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    marginTop: 8,
  },
  positionLabel: {
    fontSize: 14,
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    textAlign: 'center',
    marginTop: 15,
  },
});
