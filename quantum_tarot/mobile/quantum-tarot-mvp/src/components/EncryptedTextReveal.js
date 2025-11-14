import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { generateQuantumSeed } from '../utils/quantumRNG';

/**
 * ENCRYPTED TEXT REVEAL ANIMATION
 *
 * Creates a card-flip-like reveal animation:
 * 1. Fade out (1s)
 * 2. Black screen (0.5s)
 * 3. Fade in to encrypted text (0.5s)
 * 4. Decrypt/unscramble animation (2s)
 *
 * Uses quantum seed to pseudo-randomly encrypt text while preserving:
 * - Spaces
 * - Punctuation (. , ! ? ; : - — ')
 * - Line breaks
 * - Formatting
 *
 * Only encrypts alphanumeric characters, which then gradually reveal
 * in a random order determined by quantum seed.
 */

const EncryptedTextReveal = ({
  children,
  trigger = true,
  quantumSeed = null,
  onComplete = () => {},
  style = {},
}) => {
  const [displayText, setDisplayText] = useState('');
  const [isRevealing, setIsRevealing] = useState(false);
  const fadeAnim = useRef(new Animated.Value(1)).current;
  const originalText = typeof children === 'string' ? children : '';

  useEffect(() => {
    if (trigger && originalText) {
      startRevealSequence();
    }
  }, [trigger, originalText]);

  const startRevealSequence = async () => {
    const seed = quantumSeed || generateQuantumSeed();

    // Phase 1: CHARACTER RAIN OUT - Text dissolves character by character, raining down and out (2s)
    await characterRainOut(originalText, seed);

    // Phase 2: SACRED PAUSE - Black screen, breathe, reset (1.5s)
    await wait(1500);

    // Phase 3: Generate encrypted starting point for rain in
    const encrypted = encryptText(originalText, seed);
    setDisplayText(encrypted);

    // Phase 4: CHARACTER RAIN IN - New wisdom rains down from above, decrypting (3s)
    setIsRevealing(true);
    await characterRainIn(originalText, encrypted, seed);
    setIsRevealing(false);

    // Phase 5: INTEGRATION PAUSE - Let the message land (1s)
    await wait(1000);

    onComplete();
  };

  /**
   * CHARACTER RAIN OUT - The old wisdom dissolves, characters rain down and out
   * Sacred dissolution: 2 seconds of intentional release
   */
  const characterRainOut = async (text, seed) => {
    const duration = 2000; // 2 seconds - slow, ritualistic
    const frameInterval = 60; // Slightly slower frame rate for smoothness
    const totalFrames = duration / frameInterval;

    const alphanumericIndices = [];
    for (let i = 0; i < text.length; i++) {
      if (isAlphanumeric(text[i])) {
        alphanumericIndices.push(i);
      }
    }

    // Shuffle for quantum randomness
    const shuffledIndices = shuffleArray([...alphanumericIndices], seed);
    const charsPerFrame = Math.ceil(shuffledIndices.length / totalFrames);

    let currentText = text.split('');
    let dissolvedCount = 0;

    // Start fade out animation - slow, reverent
    Animated.timing(fadeAnim, {
      toValue: 0,
      duration: duration,
      useNativeDriver: true,
    }).start();

    // Characters rain down: randomly cycle through glitch chars before disappearing
    for (let frame = 0; frame < totalFrames; frame++) {
      for (let i = 0; i < charsPerFrame && dissolvedCount < shuffledIndices.length; i++) {
        const index = shuffledIndices[dissolvedCount];

        // Flash through 2-3 random characters (Matrix rain effect)
        for (let glitch = 0; glitch < 2; glitch++) {
          currentText[index] = getRandomChar(seed + dissolvedCount + glitch);
          setDisplayText(currentText.join(''));
          await wait(20); // Quick glitch flash
        }

        // Then make it a space (character "falls out")
        currentText[index] = ' ';
        dissolvedCount++;
      }

      setDisplayText(currentText.join(''));
      await wait(frameInterval);
    }

    // Final state: all spaces, fully dissolved
    setDisplayText(' '.repeat(text.length));
  };

  /**
   * CHARACTER RAIN IN - New wisdom rains down from above, sacred revelation
   * Holy unveiling: 3 seconds of intentional manifestation
   */
  const characterRainIn = async (originalText, encryptedText, seed) => {
    const duration = 3000; // 3 seconds - slow, ceremonial revelation
    const frameInterval = 60; // Smooth, meditative
    const totalFrames = duration / frameInterval;

    const alphanumericIndices = [];
    for (let i = 0; i < originalText.length; i++) {
      if (isAlphanumeric(originalText[i])) {
        alphanumericIndices.push(i);
      }
    }

    // Shuffle for quantum randomness
    const shuffledIndices = shuffleArray([...alphanumericIndices], seed);
    const charsPerFrame = Math.ceil(shuffledIndices.length / totalFrames);

    let currentText = encryptedText.split('');
    let revealedCount = 0;

    // Start fade in animation - slow emergence
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: duration,
      useNativeDriver: true,
    }).start();

    // Characters rain IN: cycle through glitch chars before solidifying into truth
    for (let frame = 0; frame < totalFrames; frame++) {
      for (let i = 0; i < charsPerFrame && revealedCount < shuffledIndices.length; i++) {
        const index = shuffledIndices[revealedCount];

        // Flash through 3-4 random characters (raining down)
        for (let glitch = 0; glitch < 3; glitch++) {
          currentText[index] = getRandomChar(seed + revealedCount + glitch + 1000);
          setDisplayText(currentText.join(''));
          await wait(25); // Slightly slower glitch (more ceremonial)
        }

        // Then reveal the TRUE character (wisdom solidifies)
        currentText[index] = originalText[index];
        revealedCount++;
      }

      setDisplayText(currentText.join(''));
      await wait(frameInterval);
    }

    // Final state: complete truth revealed
    setDisplayText(originalText);
  };

  /**
   * Get random character using seed
   */
  const getRandomChar = (seed) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
    const seedValue = (seed * 9301 + 49297) % 233280;
    const randomIndex = Math.floor((seedValue / 233280) * chars.length);
    return chars[randomIndex];
  };

  const fadeOut = () => {
    return new Promise((resolve) => {
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 1000,
        useNativeDriver: true,
      }).start(resolve);
    });
  };

  const fadeIn = () => {
    return new Promise((resolve) => {
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }).start(resolve);
    });
  };

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /**
   * Encrypt text by replacing alphanumeric chars with random chars
   * Preserves: spaces, punctuation, line breaks, formatting
   */
  const encryptText = (text, seed) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
    let result = '';
    let seedCounter = seed;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      if (isAlphanumeric(char)) {
        // Replace with random character
        seedCounter = (seedCounter * 9301 + 49297) % 233280; // Linear congruential generator
        const randomIndex = Math.floor((seedCounter / 233280) * chars.length);
        result += chars[randomIndex];
      } else {
        // Preserve non-alphanumeric (spaces, punctuation, line breaks)
        result += char;
      }
    }

    return result;
  };

  /**
   * Decrypt animation: gradually reveal characters in random order
   * determined by quantum seed
   */
  const decryptAnimation = async (originalText, encryptedText, seed) => {
    const duration = 2000; // 2 seconds
    const frameInterval = 50; // Update every 50ms (20 FPS)
    const totalFrames = duration / frameInterval;

    // Get indices of all alphanumeric characters
    const alphanumericIndices = [];
    for (let i = 0; i < originalText.length; i++) {
      if (isAlphanumeric(originalText[i])) {
        alphanumericIndices.push(i);
      }
    }

    // Shuffle indices using quantum seed for random reveal order
    const shuffledIndices = shuffleArray([...alphanumericIndices], seed);

    // Calculate how many characters to reveal per frame
    const charsPerFrame = Math.ceil(shuffledIndices.length / totalFrames);

    let currentText = encryptedText.split('');
    let revealedCount = 0;

    for (let frame = 0; frame < totalFrames; frame++) {
      // Reveal next batch of characters
      for (let i = 0; i < charsPerFrame && revealedCount < shuffledIndices.length; i++) {
        const index = shuffledIndices[revealedCount];
        currentText[index] = originalText[index];
        revealedCount++;
      }

      setDisplayText(currentText.join(''));
      await wait(frameInterval);
    }

    // Ensure all characters are revealed
    setDisplayText(originalText);
  };

  /**
   * Check if character is alphanumeric
   */
  const isAlphanumeric = (char) => {
    return /[a-zA-Z0-9]/.test(char);
  };

  /**
   * Shuffle array using quantum seed (Fisher-Yates with seeded random)
   */
  const shuffleArray = (array, seed) => {
    let currentSeed = seed;

    for (let i = array.length - 1; i > 0; i--) {
      // Seeded random number
      currentSeed = (currentSeed * 9301 + 49297) % 233280;
      const j = Math.floor((currentSeed / 233280) * (i + 1));

      // Swap
      [array[i], array[j]] = [array[j], array[i]];
    }

    return array;
  };

  return (
    <Animated.View style={[style, { opacity: fadeAnim }]}>
      <Text style={[styles.text, isRevealing && styles.revealingText]}>
        {displayText || children}
      </Text>
    </Animated.View>
  );
};

/**
 * MARKDOWN-AWARE ENCRYPTED TEXT REVEAL
 *
 * For use with synthesis text that contains markdown formatting.
 * Preserves markdown syntax during encryption.
 */
export const MarkdownEncryptedReveal = ({
  children,
  trigger = true,
  quantumSeed = null,
  onComplete = () => {},
  style = {},
}) => {
  const [displayText, setDisplayText] = useState('');
  const [isRevealing, setIsRevealing] = useState(false);
  const fadeAnim = useRef(new Animated.Value(1)).current;
  const originalText = typeof children === 'string' ? children : '';

  useEffect(() => {
    if (trigger && originalText) {
      startRevealSequence();
    }
  }, [trigger, originalText]);

  const startRevealSequence = async () => {
    const seed = quantumSeed || generateQuantumSeed();

    await fadeOut();
    await wait(500);

    const encrypted = encryptMarkdownText(originalText, seed);
    setDisplayText(encrypted);

    await fadeIn();

    setIsRevealing(true);
    await decryptMarkdownAnimation(originalText, encrypted, seed);
    setIsRevealing(false);

    onComplete();
  };

  const fadeOut = () => {
    return new Promise((resolve) => {
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 1000,
        useNativeDriver: true,
      }).start(resolve);
    });
  };

  const fadeIn = () => {
    return new Promise((resolve) => {
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }).start(resolve);
    });
  };

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /**
   * Encrypt markdown text while preserving:
   * - Markdown syntax (**, *, _, #, -, >, etc.)
   * - URLs
   * - Punctuation
   * - Spaces and line breaks
   */
  const encryptMarkdownText = (text, seed) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
    let result = '';
    let seedCounter = seed;
    let inMarkdownSyntax = false;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const nextChar = text[i + 1] || '';

      // Detect markdown syntax patterns
      if (
        char === '*' ||
        char === '_' ||
        char === '#' ||
        char === '-' ||
        char === '>' ||
        char === '[' ||
        char === ']' ||
        char === '(' ||
        char === ')'
      ) {
        result += char;
        continue;
      }

      // Detect URLs (http://, https://)
      if (text.slice(i, i + 7) === 'http://' || text.slice(i, i + 8) === 'https://') {
        const urlEndIndex = text.indexOf(' ', i);
        const url = text.slice(i, urlEndIndex === -1 ? text.length : urlEndIndex);
        result += url;
        i += url.length - 1;
        continue;
      }

      if (isAlphanumeric(char)) {
        seedCounter = (seedCounter * 9301 + 49297) % 233280;
        const randomIndex = Math.floor((seedCounter / 233280) * chars.length);
        result += chars[randomIndex];
      } else {
        result += char;
      }
    }

    return result;
  };

  const decryptMarkdownAnimation = async (originalText, encryptedText, seed) => {
    const duration = 2000;
    const frameInterval = 50;
    const totalFrames = duration / frameInterval;

    const alphanumericIndices = [];
    for (let i = 0; i < originalText.length; i++) {
      if (isAlphanumeric(originalText[i])) {
        alphanumericIndices.push(i);
      }
    }

    const shuffledIndices = shuffleArray([...alphanumericIndices], seed);
    const charsPerFrame = Math.ceil(shuffledIndices.length / totalFrames);

    let currentText = encryptedText.split('');
    let revealedCount = 0;

    for (let frame = 0; frame < totalFrames; frame++) {
      for (let i = 0; i < charsPerFrame && revealedCount < shuffledIndices.length; i++) {
        const index = shuffledIndices[revealedCount];
        currentText[index] = originalText[index];
        revealedCount++;
      }

      setDisplayText(currentText.join(''));
      await wait(frameInterval);
    }

    setDisplayText(originalText);
  };

  const isAlphanumeric = (char) => {
    return /[a-zA-Z0-9]/.test(char);
  };

  const shuffleArray = (array, seed) => {
    let currentSeed = seed;

    for (let i = array.length - 1; i > 0; i--) {
      currentSeed = (currentSeed * 9301 + 49297) % 233280;
      const j = Math.floor((currentSeed / 233280) * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }

    return array;
  };

  return (
    <Animated.View style={[style, { opacity: fadeAnim }]}>
      <Text style={[styles.text, isRevealing && styles.revealingText]}>
        {displayText || children}
      </Text>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  text: {
    fontSize: 16,
    lineHeight: 24,
    color: '#ffffff',
  },
  revealingText: {
    // Optional: add subtle glow effect during reveal
  },
});

export default EncryptedTextReveal;
