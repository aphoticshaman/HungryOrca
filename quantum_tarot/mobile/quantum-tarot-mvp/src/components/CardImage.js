/**
 * CardImage Component
 * ===================
 *
 * Renders tarot card with image-first approach and ASCII art fallback
 *
 * Priority:
 * 1. Try to load PNG image from /assets/cards/
 * 2. On error or missing file → fallback to ASCII art
 *
 * This proves the graceful degradation works even without images.
 */

import React, { useState } from 'react';
import { View, Image, Text, StyleSheet } from 'react-native';
import { getAsciiCard } from '../data/asciiCards';

/**
 * Get image source for a card
 * Images should be stored as: /assets/cards/0.png, /assets/cards/1.png, etc.
 * Reversed cards use same image (rotation handled in rendering)
 */
function getCardImageSource(cardIndex) {
  try {
    // Attempt to dynamically require the image
    // In production, you'd have all 78 images: 0.png through 77.png
    const imageMap = {
      0: require('../../assets/cards/0.png'),
      1: require('../../assets/cards/1.png'),
      2: require('../../assets/cards/2.png'),
      3: require('../../assets/cards/3.png'),
      4: require('../../assets/cards/4.png'),
      5: require('../../assets/cards/5.png'),
      6: require('../../assets/cards/6.png'),
      // Add all 78 cards here when images are ready
    };

    return imageMap[cardIndex] || null;
  } catch (error) {
    console.log(`Image not found for card ${cardIndex}, will use ASCII`);
    return null;
  }
}

export default function CardImage({ cardIndex, reversed = false, style }) {
  const [useAscii, setUseAscii] = useState(false);
  const imageSource = getCardImageSource(cardIndex);

  // If no image source available, use ASCII immediately
  if (!imageSource || useAscii) {
    return (
      <View style={[styles.asciiContainer, style]}>
        <Text style={styles.asciiArt}>
          {getAsciiCard(cardIndex, reversed)}
        </Text>
      </View>
    );
  }

  // Try to render image with error fallback
  return (
    <View style={[styles.imageContainer, style]}>
      <Image
        source={imageSource}
        style={[
          styles.cardImage,
          reversed && styles.reversed
        ]}
        onError={(error) => {
          console.log(`Failed to load image for card ${cardIndex}:`, error);
          setUseAscii(true);
        }}
        resizeMode="contain"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  imageContainer: {
    width: 200,
    height: 350,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardImage: {
    width: '100%',
    height: '100%',
  },
  reversed: {
    transform: [{ rotate: '180deg' }],
  },
  asciiContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
  },
  asciiArt: {
    fontFamily: 'Courier',
    fontSize: 10,
    lineHeight: 12,
    color: '#e0e0e0',
  },
});
