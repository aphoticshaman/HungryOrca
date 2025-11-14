/**
 * CYBERPUNK HEADER - Wave-animated neon title
 */

import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, Dimensions, Animated, Text } from 'react-native';
import { NeonText, LPMUDText } from './TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function CyberpunkHeader({ showMatrixBg = false, compact = false }) {
  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <LPMUDText style={styles.compactTitle}>
          $HIC$LunatIQ Tarot$NOR$
        </LPMUDText>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        {/* Main title with wave animation and rainbow colors */}
        <WaveText
          text="LunatIQ Tarot"
          style={styles.mainTitle}
        />
      </View>
    </View>
  );
}

/**
 * Wave animated text - each letter bobs up and down with rainbow colors
 */
function WaveText({ text, style }) {
  const [colorOffset, setColorOffset] = useState(0);

  const colorPalette = [
    NEON_COLORS.hiCyan,    // Cyan
    NEON_COLORS.hiMagenta, // Magenta
    NEON_COLORS.hiYellow,  // Yellow
    NEON_COLORS.hiGreen,   // Green
    '#FF00FF',             // Purple
    '#FF6EC7',             // Pink
    '#00FFFF',             // Bright cyan
    '#FFFF00',             // Bright yellow
  ];

  const letters = text.split('');

  const animations = useRef(
    Array.from({ length: text.length }, () => new Animated.Value(0))
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
    // Rotate rainbow colors
    const interval = setInterval(() => {
      setColorOffset((prev) => (prev + 1) % colorPalette.length);
    }, 300);

    return () => clearInterval(interval);
  }, []);

  return (
    <View style={styles.waveContainer}>
      {letters.map((letter, index) => {
        const translateY = animations[index].interpolate({
          inputRange: [0, 1],
          outputRange: [0, -15], // Wave amplitude
        });

        // Each character gets a different color from the palette
        const colorIndex = (index + colorOffset) % colorPalette.length;
        const color = colorPalette[colorIndex];

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
    overflow: 'visible',
  },
  content: {
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'visible',
  },
  waveContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 12,
    marginBottom: 15,
    paddingHorizontal: 10,
    overflow: 'visible',
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
