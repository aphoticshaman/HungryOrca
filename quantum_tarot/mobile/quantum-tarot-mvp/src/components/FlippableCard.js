/**
 * Flippable Card Component
 * Shows card front (ASCII art + title) and back (symbolism + meaning)
 * Tappable flip animation
 */

import React, { useState, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, Platform } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { getAsciiCard } from '../data/asciiCards';
import { getCardData } from '../data/cardDatabase';

// Uniform monospace font for ASCII art
const MONOSPACE_FONT = Platform.select({
  ios: 'Courier',
  android: 'monospace',
  default: 'Courier New',
});

export default function FlippableCard({ cardIndex, reversed, position, style }) {
  const { theme } = useTheme();
  const [isFlipped, setIsFlipped] = useState(false);
  const flipAnim = useRef(new Animated.Value(0)).current;
  const styles = createStyles(theme);

  // Get card data
  const cardData = getCardData(cardIndex);
  if (!cardData) {
    return <Text style={styles.error}>Card data not found</Text>;
  }

  // Card title
  const cardTitle = cardData.arcana === 'major'
    ? `${cardData.number} - ${cardData.name.toUpperCase()}`
    : cardData.name.toUpperCase();

  // Handle flip
  const handleFlip = () => {
    const toValue = isFlipped ? 0 : 180;
    Animated.spring(flipAnim, {
      toValue,
      friction: 8,
      tension: 10,
      useNativeDriver: true
    }).start();
    setIsFlipped(!isFlipped);
  };

  // Front interpolation
  const frontInterpolate = flipAnim.interpolate({
    inputRange: [0, 180],
    outputRange: ['0deg', '180deg']
  });

  const frontOpacity = flipAnim.interpolate({
    inputRange: [0, 90, 180],
    outputRange: [1, 0, 0]
  });

  // Back interpolation
  const backInterpolate = flipAnim.interpolate({
    inputRange: [0, 180],
    outputRange: ['180deg', '360deg']
  });

  const backOpacity = flipAnim.interpolate({
    inputRange: [0, 90, 180],
    outputRange: [0, 0, 1]
  });

  return (
    <View style={[styles.container, style]}>
      {/* Flip Button */}
      <TouchableOpacity style={styles.flipButton} onPress={handleFlip}>
        <Text style={styles.flipButtonText}>
          {isFlipped ? '◀ FLIP TO FRONT' : 'FLIP TO BACK ▶'}
        </Text>
      </TouchableOpacity>

      {/* Card Container */}
      <View style={styles.cardContainer}>
        {/* FRONT SIDE */}
        <Animated.View
          style={[
            styles.cardFace,
            styles.cardFront,
            {
              opacity: frontOpacity,
              transform: [{ rotateY: frontInterpolate }]
            }
          ]}
          pointerEvents={isFlipped ? 'none' : 'auto'}
        >
          {/* Card Title */}
          <Text style={styles.cardTitle}>{cardTitle}</Text>
          {reversed && <Text style={styles.reversedLabel}>(REVERSED)</Text>}

          {/* ASCII Art */}
          <Text style={styles.asciiArt}>
            {getAsciiCard(cardIndex, reversed)}
          </Text>

          {/* Position */}
          {position && (
            <Text style={styles.position}>— {position.toUpperCase()} —</Text>
          )}
        </Animated.View>

        {/* BACK SIDE */}
        <Animated.View
          style={[
            styles.cardFace,
            styles.cardBack,
            {
              opacity: backOpacity,
              transform: [{ rotateY: backInterpolate }]
            }
          ]}
          pointerEvents={isFlipped ? 'auto' : 'none'}
        >
          {/* Title */}
          <Text style={styles.backTitle}>{cardTitle}</Text>
          <Text style={styles.backSubtitle}>SYMBOLISM & MEANING</Text>

          {/* Scrollable content */}
          <View style={styles.backContent}>
            {/* Description */}
            <Text style={styles.sectionTitle}>Description:</Text>
            <Text style={styles.description}>{cardData.description}</Text>

            {/* Keywords */}
            <Text style={styles.sectionTitle}>Keywords:</Text>
            <Text style={styles.keywords}>
              {reversed
                ? cardData.keywords.reversed.join(', ')
                : cardData.keywords.upright.join(', ')}
            </Text>

            {/* Symbols */}
            {cardData.symbols && cardData.symbols.length > 0 && (
              <>
                <Text style={styles.sectionTitle}>Key Symbols:</Text>
                <Text style={styles.symbols}>
                  {cardData.symbols.join(' • ')}
                </Text>
              </>
            )}

            {/* Element & Astrology */}
            <View style={styles.metadata}>
              <Text style={styles.metaItem}>Element: {cardData.element}</Text>
              {cardData.astrology && (
                <Text style={styles.metaItem}>Astrology: {cardData.astrology}</Text>
              )}
              {cardData.chakra && (
                <Text style={styles.metaItem}>Chakra: {cardData.chakra}</Text>
              )}
              {cardData.numerology !== undefined && (
                <Text style={styles.metaItem}>Number: {cardData.numerology}</Text>
              )}
            </View>

            {/* Themes */}
            {cardData.themes && cardData.themes.length > 0 && (
              <>
                <Text style={styles.sectionTitle}>Themes:</Text>
                <Text style={styles.themes}>
                  {cardData.themes.map(t => t.replace(/_/g, ' ')).join(' • ')}
                </Text>
              </>
            )}

            {/* Advice */}
            {cardData.advice && (
              <>
                <Text style={styles.sectionTitle}>Advice:</Text>
                <Text style={styles.advice}>"{cardData.advice}"</Text>
              </>
            )}

            {/* Shadow/Light */}
            {reversed && cardData.shadow && (
              <>
                <Text style={styles.sectionTitle}>Shadow Aspect:</Text>
                <Text style={styles.shadow}>{cardData.shadow}</Text>
              </>
            )}

            {!reversed && cardData.light && (
              <>
                <Text style={styles.sectionTitle}>Highest Expression:</Text>
                <Text style={styles.light}>{cardData.light}</Text>
              </>
            )}
          </View>
        </Animated.View>
      </View>
    </View>
  );
}

function createStyles(theme) {
  return StyleSheet.create({
    container: {
      width: '100%',
      marginVertical: 20
    },
    flipButton: {
      borderWidth: 1,
      borderColor: theme.border,
      paddingVertical: 10,
      paddingHorizontal: 15,
      marginBottom: 15,
      alignSelf: 'center'
    },
    flipButtonText: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 12,
      color: theme.accent,
      textAlign: 'center'
    },
    cardContainer: {
      minHeight: 400,
      position: 'relative'
    },
    cardFace: {
      position: 'absolute',
      width: '100%',
      minHeight: 400,
      backfaceVisibility: 'hidden',
      borderWidth: 2,
      borderColor: theme.border,
      padding: 15,
      backgroundColor: theme.background
    },
    cardFront: {
      justifyContent: 'center',
      alignItems: 'center'
    },
    cardBack: {
      justifyContent: 'flex-start'
    },
    cardTitle: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 14,
      color: theme.accent,
      textAlign: 'center',
      marginBottom: 5,
      fontWeight: 'bold'
    },
    reversedLabel: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: theme.textDim,
      textAlign: 'center',
      marginBottom: 10
    },
    asciiArt: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 9,
      color: theme.text,
      textAlign: 'center',
      lineHeight: 12,
      marginVertical: 15
    },
    position: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 11,
      color: theme.textDim,
      textAlign: 'center',
      marginTop: 10
    },
    // BACK SIDE STYLES
    backTitle: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 13,
      color: theme.accent,
      textAlign: 'center',
      marginBottom: 3,
      fontWeight: 'bold'
    },
    backSubtitle: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 9,
      color: theme.textDim,
      textAlign: 'center',
      marginBottom: 15,
      borderBottomWidth: 1,
      borderBottomColor: theme.border,
      paddingBottom: 10
    },
    backContent: {
      flex: 1
    },
    sectionTitle: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: theme.accent,
      marginTop: 12,
      marginBottom: 5,
      textTransform: 'uppercase'
    },
    description: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: theme.text,
      lineHeight: 16,
      marginBottom: 8
    },
    keywords: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: theme.text,
      fontStyle: 'italic'
    },
    symbols: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: theme.text
    },
    metadata: {
      marginTop: 10,
      marginBottom: 5,
      paddingTop: 10,
      borderTopWidth: 1,
      borderTopColor: theme.border
    },
    metaItem: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 9,
      color: theme.textDim,
      marginBottom: 3
    },
    themes: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: theme.text
    },
    advice: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: theme.accent,
      fontStyle: 'italic',
      lineHeight: 15
    },
    shadow: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: '#FC8181', // Red tint for shadow
      lineHeight: 15
    },
    light: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 10,
      color: '#68D391', // Green tint for light
      lineHeight: 15
    },
    error: {
      fontFamily: MONOSPACE_FONT,
      fontSize: 12,
      color: '#FC8181',
      textAlign: 'center'
    }
  });
}
