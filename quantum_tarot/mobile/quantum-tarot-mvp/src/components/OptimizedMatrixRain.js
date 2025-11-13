/**
 * OPTIMIZED MATRIX RAIN - Smooth performance on all devices
 * Uses React Native Animated API + reduced render count
 */

import React, { useEffect, useRef, useMemo } from 'react';
import { View, Text, Animated, StyleSheet, Platform } from 'react-native';
import { MATRIX_CHARS } from '../styles/cyberpunkColors';

/**
 * Performance-optimized Matrix Rain
 * Runs at 60fps on cheap phones and flagships
 */
export const OptimizedMatrixRain = React.memo(({ width = 100, height = 200, speed = 50 }) => {
  // Reduce columns for performance (instead of width/12)
  const numColumns = Math.min(Math.floor(width / 20), 30); // Max 30 columns
  const charsPerColumn = 15; // Reduced from 40

  // Memoize column setup (doesn't change)
  const columnSetup = useMemo(() => {
    return Array.from({ length: numColumns }, (_, i) => ({
      x: i * (width / numColumns),
      animValue: new Animated.Value(0),
      chars: generateColumnChars(charsPerColumn),
      speed: 0.8 + Math.random() * 0.4, // Less speed variation
    }));
  }, [numColumns, width, charsPerColumn]);

  useEffect(() => {
    // Start all column animations
    const animations = columnSetup.map(col => {
      return Animated.loop(
        Animated.timing(col.animValue, {
          toValue: height,
          duration: (height / col.speed) * 20, // Speed based on height
          useNativeDriver: true, // GPU acceleration!
        })
      );
    });

    animations.forEach(anim => anim.start());

    return () => {
      animations.forEach(anim => anim.stop());
    };
  }, [columnSetup, height]);

  return (
    <View style={[styles.container, { width, height }]} pointerEvents="none">
      {columnSetup.map((col, colIndex) => (
        <Animated.View
          key={colIndex}
          style={[
            styles.column,
            {
              left: col.x,
              transform: [
                {
                  translateY: col.animValue.interpolate({
                    inputRange: [0, height],
                    outputRange: [-charsPerColumn * 20, height], // 20px per char
                  }),
                },
              ],
            },
          ]}
        >
          {col.chars.map((charData, charIndex) => (
            <Text
              key={charIndex}
              style={[
                styles.char,
                {
                  color: charData.color,
                  opacity: charData.opacity,
                },
              ]}
            >
              {charData.char}
            </Text>
          ))}
        </Animated.View>
      ))}
    </View>
  );
});

/**
 * Generate static character column (doesn't change during animation)
 */
function generateColumnChars(count) {
  const colors = [
    '#00FF0088', // Green (50% opacity)
    '#00FFFF66', // Cyan (40%)
    '#FFFFFF44', // White (25%)
    '#0099FF55', // Blue (33%)
    '#FFFF0066', // Yellow (40%)
  ];

  return Array.from({ length: count }, (_, i) => {
    // Fade from bright at top to dim at bottom
    const opacityFactor = 1 - (i / count) * 0.7;

    return {
      char: MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)],
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: 0.3 + opacityFactor * 0.5, // 0.3 to 0.8 range
    };
  });
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    overflow: 'hidden',
    backgroundColor: 'transparent',
  },
  column: {
    position: 'absolute',
    top: 0,
  },
  char: {
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    fontSize: 14,
    fontWeight: 'bold',
    lineHeight: 20,
    // NO text shadow - huge performance win!
  },
});
