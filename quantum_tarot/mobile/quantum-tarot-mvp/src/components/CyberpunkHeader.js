/**
 * CYBERPUNK HEADER - Wave-animated neon title
 */

import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, Dimensions, Animated, Text } from 'react-native';
import { NeonText, LPMUDText } from './TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function CyberpunkHeader({ showMatrixBg = false, compact = false }) {
  const [colorIndex, setColorIndex] = useState(0);

  const colors = [
    NEON_COLORS.hiCyan,
    NEON_COLORS.hiMagenta,
    NEON_COLORS.hiYellow,
    NEON_COLORS.hiGreen,
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setColorIndex((prev) => (prev + 1) % colors.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <LPMUDText style={styles.compactTitle}>
          $HIC$LunatiQ$NOR$ $HIM$TAROT SYSTEM$NOR$
        </LPMUDText>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        {/* Main title with wave animation */}
        <WaveText
          text="LunatiQ"
          color={colors[colorIndex]}
          style={styles.mainTitle}
        />

        <NeonText
          color={NEON_COLORS.hiMagenta}
          style={styles.subtitle}
        >
          TAROT SYSTEM
        </NeonText>
      </View>
    </View>
  );
}

/**
 * Wave animated text - each letter bobs up and down with glitch variations
 */
function WaveText({ text, color, style }) {
  const [glitchIndex, setGlitchIndex] = useState(0);

  // Glitch text variations (all must be 7 chars!)
  const glitchVariations = [
    'LunatIQ',
    'Lun4t1Q',
    'LunatiQ',
    'LxnxtxQ',
    'LunatIQ',
    'L00n4tQ',
    'LunatIQ',
  ];

  const currentText = glitchVariations[glitchIndex];
  const letters = currentText.split('');

  const animations = useRef(
    Array.from({ length: 7 }, () => new Animated.Value(0))
  ).current;

  useEffect(() => {
    // Create wave animation for each letter
    const waveAnimations = animations.map((anim, index) => {
      return Animated.loop(
        Animated.sequence([
          Animated.delay(index * 100), // Stagger the waves
          Animated.timing(anim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      );
    });

    // Start all animations
    Animated.stagger(0, waveAnimations).start();

    return () => {
      animations.forEach(anim => anim.stopAnimation());
    };
  }, []);

  useEffect(() => {
    // Cycle through glitch variations
    const interval = setInterval(() => {
      setGlitchIndex((prev) => (prev + 1) % glitchVariations.length);
    }, 800); // Faster glitch cycling

    return () => clearInterval(interval);
  }, []);

  return (
    <View style={styles.waveContainer}>
      {letters.map((letter, index) => {
        const translateY = animations[index].interpolate({
          inputRange: [0, 1],
          outputRange: [0, -15], // Wave amplitude
        });

        return (
          <Animated.View
            key={index}
            style={{
              transform: [{ translateY }],
            }}
          >
            {/* Layered glow effect for cel-shaded neon look */}
            {/* Outer glow - big bloom */}
            <Text
              style={[
                style,
                {
                  position: 'absolute',
                  color,
                  textShadowColor: color,
                  textShadowOffset: { width: 0, height: 0 },
                  textShadowRadius: 30,
                  opacity: 0.8,
                }
              ]}
            >
              {letter}
            </Text>
            {/* Middle glow */}
            <Text
              style={[
                style,
                {
                  position: 'absolute',
                  color,
                  textShadowColor: color,
                  textShadowOffset: { width: 0, height: 0 },
                  textShadowRadius: 15,
                  opacity: 0.9,
                }
              ]}
            >
              {letter}
            </Text>
            {/* Inner glow - sharp core */}
            <Text
              style={[
                style,
                {
                  color,
                  textShadowColor: color,
                  textShadowOffset: { width: 0, height: 0 },
                  textShadowRadius: 8,
                }
              ]}
            >
              {letter}
            </Text>
          </Animated.View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    paddingVertical: 40,
    paddingHorizontal: 20,
    backgroundColor: '#000000',
  },
  content: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  waveContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 15,
  },
  mainTitle: {
    fontSize: 48,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 20,
    letterSpacing: 4,
  },
  subtitle: {
    fontSize: 32,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
    letterSpacing: 12,
    textShadowColor: NEON_COLORS.glowMagenta,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 15,
  },
  compactContainer: {
    width: '100%',
    paddingVertical: 15,
    paddingHorizontal: 20,
    backgroundColor: '#000000',
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
  },
  compactTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
