/**
 * CYBERPUNK TERMINAL EFFECTS
 * Neon glow, flicker, glitch, morph - Matrix-style hacker aesthetic
 */

import React, { useState, useEffect, useRef } from 'react';
import { Text, View, Animated, StyleSheet, Platform } from 'react-native';
import { NEON_COLORS, parseLPMUDColors, GLITCH_COLORS, MATRIX_CHARS } from '../styles/cyberpunkColors';

/**
 * NEON TEXT - Glowing cyberpunk text with shadow effects
 */
export function NeonText({ children, color = NEON_COLORS.cyan, glowColor, style, ...props }) {
  const glow = glowColor || color;

  return (
    <Text
      style={[
        styles.neonBase,
        {
          color: color,
          textShadowColor: glow,
          textShadowOffset: { width: 0, height: 0 },
          textShadowRadius: 10,
        },
        style
      ]}
      {...props}
    >
      {children}
    </Text>
  );
}

/**
 * LPMUD TEXT - Parse and render LPMUD color codes
 * Supports $HIY$, $HIC$, $HIM$, etc.
 */
export function LPMUDText({ children, style, ...props }) {
  // Convert children to string (handles both strings and template literals)
  let text;
  if (typeof children === 'string') {
    text = children;
  } else if (Array.isArray(children)) {
    // Handle array of children (from template literals)
    // Filter out undefined, null, and empty values before joining
    text = children
      .filter(child => child != null && child !== '')
      .map(child => String(child))
      .join('');
  } else if (children != null) {
    // Handle other types (numbers, etc)
    text = String(children);
  } else {
    return null;
  }

  // If text is empty after processing, return null
  if (!text || text.trim() === '') {
    return null;
  }

  const segments = parseLPMUDColors(text);

  // If no segments were parsed, return null to avoid empty boxes
  if (!segments || segments.length === 0) {
    return null;
  }

  return (
    <Text style={[{ color: NEON_COLORS.dimWhite }, style]} {...props}>
      {segments.map((segment, i) => (
        <NeonText
          key={i}
          color={segment.color}
          glowColor={segment.glow}
          style={style}
        >
          {segment.text}
        </NeonText>
      ))}
    </Text>
  );
}

/**
 * FLICKER TEXT - Random intensity flickering like old CRT monitors
 */
export function FlickerText({ children, color = NEON_COLORS.cyan, flickerSpeed = 100, style, ...props }) {
  const [opacity, setOpacity] = useState(1);

  useEffect(() => {
    const interval = setInterval(() => {
      // Random flicker between 0.7 and 1.0
      setOpacity(0.7 + Math.random() * 0.3);
    }, flickerSpeed);

    return () => clearInterval(interval);
  }, [flickerSpeed]);

  return (
    <NeonText
      color={color}
      style={[style, { opacity }]}
      {...props}
    >
      {children}
    </NeonText>
  );
}

/**
 * GLITCH TEXT - Random character substitution and color shifts
 */
export function GlitchText({
  children,
  glitchChance = 0.1,
  glitchSpeed = 100,
  style,
  ...props
}) {
  const [displayText, setDisplayText] = useState(children);
  const [glitchColor, setGlitchColor] = useState(NEON_COLORS.cyan);

  useEffect(() => {
    const interval = setInterval(() => {
      if (typeof children !== 'string') return;

      if (Math.random() < glitchChance) {
        // Glitch some characters
        const chars = children.split('');
        const glitchedChars = chars.map(char => {
          if (Math.random() < 0.3) {
            return MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)];
          }
          return char;
        });
        setDisplayText(glitchedChars.join(''));
        setGlitchColor(GLITCH_COLORS[Math.floor(Math.random() * GLITCH_COLORS.length)]);

        // Reset after short delay
        setTimeout(() => setDisplayText(children), 50);
      }
    }, glitchSpeed);

    return () => clearInterval(interval);
  }, [children, glitchChance, glitchSpeed]);

  return (
    <NeonText
      color={glitchColor}
      style={style}
      {...props}
    >
      {displayText}
    </NeonText>
  );
}

/**
 * MORPH TEXT - Character-by-character transformation (Matrix style)
 */
export function MorphText({
  children,
  morphSpeed = 50,
  color = NEON_COLORS.hiGreen,
  style,
  onMorphComplete,
  ...props
}) {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (typeof children !== 'string') {
      setDisplayText(children);
      return;
    }

    if (currentIndex >= children.length) {
      onMorphComplete && onMorphComplete();
      return;
    }

    const interval = setInterval(() => {
      // Randomly cycle through chars before settling
      const iterations = 5;
      let iteration = 0;

      const cycleInterval = setInterval(() => {
        if (iteration < iterations) {
          const randomChar = MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)];
          setDisplayText(children.substring(0, currentIndex) + randomChar);
          iteration++;
        } else {
          setDisplayText(children.substring(0, currentIndex + 1));
          setCurrentIndex(currentIndex + 1);
          clearInterval(cycleInterval);
        }
      }, 10);
    }, morphSpeed);

    return () => clearInterval(interval);
  }, [currentIndex, children, morphSpeed]);

  return (
    <NeonText
      color={color}
      style={style}
      {...props}
    >
      {displayText}
    </NeonText>
  );
}

/**
 * MATRIX RAIN - Ultra-optimized using Animated API + GPU acceleration
 * 30 columns max (vs 100+), 15 chars each (vs 40), no text shadows = 60fps everywhere
 */
export const MatrixRain = React.memo(({ width = 100, height = 200, speed = 50 }) => {
  // PERFORMANCE: Drastically reduce elements
  const numColumns = Math.min(Math.floor(width / 20), 30); // Max 30 (was ~100)
  const charsPerColumn = 15; // Was 40

  // PERFORMANCE: Memoize setup (never changes)
  const columnSetup = React.useMemo(() => {
    return Array.from({ length: numColumns }, (_, i) => ({
      x: i * (width / numColumns),
      animValue: new Animated.Value(0),
      chars: generateMatrixColumnChars(charsPerColumn),
      speed: 0.8 + Math.random() * 0.4,
    }));
  }, [numColumns, width]);

  useEffect(() => {
    // PERFORMANCE: Native driver = GPU = smooth!
    const animations = columnSetup.map(col => {
      return Animated.loop(
        Animated.timing(col.animValue, {
          toValue: height,
          duration: (height / col.speed) * 20,
          useNativeDriver: true, // THE KEY TO SMOOTHNESS
        })
      );
    });

    animations.forEach(anim => anim.start());

    return () => {
      animations.forEach(anim => anim.stop());
    };
  }, [columnSetup, height]);

  return (
    <View style={[styles.matrixContainer, { width, height }]} pointerEvents="none">
      {columnSetup.map((col, colIndex) => (
        <Animated.View
          key={colIndex}
          style={[
            styles.matrixColumn,
            {
              left: col.x,
              transform: [
                {
                  translateY: col.animValue.interpolate({
                    inputRange: [0, height],
                    outputRange: [-charsPerColumn * 20, height],
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
                styles.matrixChar,
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

// Helper: Pre-generate static chars (no regeneration each frame!)
function generateMatrixColumnChars(count) {
  const colors = [
    '#00FF0088', // Green with alpha baked in
    '#00FFFF66', // Cyan
    '#FFFFFF44', // White
    '#0099FF55', // Blue
    '#FFFF0066', // Yellow
  ];

  return Array.from({ length: count }, (_, i) => {
    const opacityFactor = 1 - (i / count) * 0.7; // Fade toward bottom

    return {
      char: MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)],
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: 0.3 + opacityFactor * 0.5, // 0.3 to 0.8
    };
  });
}

/**
 * SCAN LINE EFFECT - Old CRT monitor horizontal lines
 */
export function ScanLines({ style }) {
  const opacity = useRef(new Animated.Value(0.05)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, {
          toValue: 0.15,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 0.05,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  return (
    <Animated.View style={[styles.scanLines, { opacity }, style]} />
  );
}

const styles = StyleSheet.create({
  neonBase: {
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    fontWeight: 'bold',
  },
  matrixContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    overflow: 'hidden',
    backgroundColor: 'transparent',
  },
  matrixColumn: {
    position: 'absolute',
    top: 0,
  },
  matrixChar: {
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    fontSize: 14,
    fontWeight: 'bold',
    lineHeight: 20,
    // NO TEXT SHADOW = MASSIVE PERFORMANCE WIN!
    // Text shadows are GPU killers, especially on 450+ elements
  },
  scanLines: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#000',
    pointerEvents: 'none',
  },
});
