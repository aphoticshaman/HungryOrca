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
  if (typeof children !== 'string') {
    return <Text style={style} {...props}>{children}</Text>;
  }

  const segments = parseLPMUDColors(children);

  return (
    <Text style={style} {...props}>
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
 * MATRIX RAIN - Multicolor falling character columns
 */
export function MatrixRain({ width = 100, height = 200, speed = 50 }) {
  const [columns, setColumns] = useState([]);

  // Multicolor palette - 2025 cyberpunk aesthetic
  const RAIN_COLORS = [
    '#FFFFFF', // white
    '#00FFFF', // cyan
    '#0099FF', // blue
    '#0044AA', // dark blue
    '#FF0000', // red
    '#DC143C', // crimson
    '#888888', // grey
    '#FFFF00', // yellow
    '#FFA500', // orange
    '#FF69B4', // pink
    '#9370DB', // purple
    '#8B4513', // brown
  ];

  useEffect(() => {
    // Initialize columns
    const numColumns = Math.floor(width / 12); // ~12px per char
    const initialColumns = Array.from({ length: numColumns }, (_, i) => ({
      x: i * 12,
      chars: [],
      speed: 0.5 + Math.random() * 1.5,
    }));
    setColumns(initialColumns);

    const interval = setInterval(() => {
      setColumns(prev => prev.map(col => {
        // Add new char at top with random color
        const newChars = [{
          char: MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)],
          y: 0,
          opacity: 1,
          color: RAIN_COLORS[Math.floor(Math.random() * RAIN_COLORS.length)],
        }, ...col.chars];

        // Update positions and fade
        return {
          ...col,
          chars: newChars
            .map(c => ({
              ...c,
              y: c.y + col.speed * 5,
              opacity: c.opacity - 0.02,
            }))
            .filter(c => c.y < height && c.opacity > 0)
            .slice(0, 20), // Max 20 chars per column
        };
      }));
    }, speed);

    return () => clearInterval(interval);
  }, [width, height, speed]);

  return (
    <View style={[styles.matrixContainer, { width, height }]}>
      {columns.map((col, colIndex) => (
        <View key={colIndex} style={{ position: 'absolute', left: col.x }}>
          {col.chars.map((charObj, charIndex) => (
            <Text
              key={charIndex}
              style={[
                styles.matrixChar,
                {
                  position: 'absolute',
                  top: charObj.y,
                  opacity: charObj.opacity,
                  color: charObj.color,
                }
              ]}
            >
              {charObj.char}
            </Text>
          ))}
        </View>
      ))}
    </View>
  );
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
    overflow: 'hidden',
    backgroundColor: 'transparent',
  },
  matrixChar: {
    fontFamily: Platform.select({
      ios: 'Courier',
      android: 'monospace',
      default: 'Courier New',
    }),
    fontSize: 14,
    fontWeight: 'bold',
    textShadowColor: NEON_COLORS.glowGreen,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
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
