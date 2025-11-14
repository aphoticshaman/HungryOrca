import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import EncryptedTextReveal from '../components/EncryptedTextReveal';
import { CARD_DATABASE } from '../data/cardDatabase';
import { generateQuantumSeed } from '../utils/quantumRNG';

/**
 * CARD INTERPRETATION SCREEN
 *
 * Displays individual card interpretations with:
 * - Encrypted text reveal animation (Matrix-style decryption)
 * - Auto-scroll to top when moving to next card
 * - Quantum seed-based pseudo-random encryption
 * - Preserves formatting, punctuation, line breaks
 *
 * Animation sequence per card:
 * 1. Fade out (1s)
 * 2. Black screen (0.5s)
 * 3. Fade in to encrypted text (0.5s)
 * 4. Decrypt/unscramble (2s)
 * 5. User reads → clicks "Next Card"
 * 6. Scroll to top → repeat for next card
 */

const CardInterpretationScreen = ({ route, navigation }) => {
  const { cards, interpretations } = route.params || {};

  const [currentCardIndex, setCurrentCardIndex] = useState(0);
  const [revealTrigger, setRevealTrigger] = useState(false);
  const [quantumSeed, setQuantumSeed] = useState(generateQuantumSeed());

  const scrollViewRef = useRef(null);

  // Trigger reveal animation on mount
  useEffect(() => {
    setRevealTrigger(true);
  }, []);

  // Reset and animate when card changes
  useEffect(() => {
    // Scroll to top IMMEDIATELY when card changes
    scrollViewRef.current?.scrollTo({ y: 0, animated: false });

    // Then trigger reveal animation
    setQuantumSeed(generateQuantumSeed()); // New seed for new card
    setRevealTrigger(false); // Reset trigger

    // Small delay to ensure DOM updates, then re-trigger
    setTimeout(() => {
      setRevealTrigger(true);
    }, 50);
  }, [currentCardIndex]);

  const handleNextCard = () => {
    if (currentCardIndex < cards.length - 1) {
      setCurrentCardIndex(currentCardIndex + 1);
    } else {
      // All cards viewed - navigate to synthesis
      navigation.navigate('Synthesis');
    }
  };

  const handlePreviousCard = () => {
    if (currentCardIndex > 0) {
      setCurrentCardIndex(currentCardIndex - 1);
    }
  };

  const currentCard = cards[currentCardIndex];
  const currentInterpretation = interpretations[currentCardIndex];
  const cardData = CARD_DATABASE[currentCard.cardIndex];

  const cardName = `${cardData.name}${currentCard.reversed ? ' (Reversed)' : ''}`;

  return (
    <LinearGradient
      colors={['#1a0033', '#330066', '#4d0099']}
      style={styles.container}
    >
      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.cardNumber}>
            Card {currentCardIndex + 1} of {cards.length}
          </Text>
          <Text style={styles.cardName}>{cardName}</Text>
          <Text style={styles.position}>{currentCard.positionMeaning || currentCard.position}</Text>
        </View>

        {/* Progress bar */}
        <View style={styles.progressBarContainer}>
          <View
            style={[
              styles.progressBar,
              { width: `${((currentCardIndex + 1) / cards.length) * 100}%` },
            ]}
          />
        </View>

        {/* Scrollable interpretation with encrypted reveal */}
        <ScrollView
          ref={scrollViewRef}
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <EncryptedTextReveal
            trigger={revealTrigger}
            quantumSeed={quantumSeed}
            onComplete={() => console.log('Reveal complete for card', currentCardIndex + 1)}
            style={styles.interpretationContainer}
          >
            {currentInterpretation}
          </EncryptedTextReveal>
        </ScrollView>

        {/* Navigation buttons */}
        <View style={styles.footer}>
          <TouchableOpacity
            style={[styles.navButton, currentCardIndex === 0 && styles.navButtonDisabled]}
            onPress={handlePreviousCard}
            disabled={currentCardIndex === 0}
          >
            <Text
              style={[
                styles.navButtonText,
                currentCardIndex === 0 && styles.navButtonTextDisabled,
              ]}
            >
              ← Previous
            </Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.nextButton} onPress={handleNextCard}>
            <Text style={styles.nextButtonText}>
              {currentCardIndex === cards.length - 1 ? 'View Synthesis →' : 'Next Card →'}
            </Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </LinearGradient>
  );
};

/**
 * USAGE EXAMPLE: Card-by-card reveal in reading flow
 */
export const useCardRevealFlow = (cards) => {
  const [currentCardIndex, setCurrentCardIndex] = useState(0);
  const [interpretations, setInterpretations] = useState([]);
  const scrollRef = useRef(null);

  const addCardInterpretation = (interpretation) => {
    setInterpretations([...interpretations, interpretation]);
  };

  const moveToNextCard = () => {
    // Scroll to top BEFORE changing card
    scrollRef.current?.scrollTo({ y: 0, animated: true });

    // Wait for scroll, then change card (triggers new encryption animation)
    setTimeout(() => {
      setCurrentCardIndex(currentCardIndex + 1);
    }, 300);
  };

  const moveToPreviousCard = () => {
    scrollRef.current?.scrollTo({ y: 0, animated: true });

    setTimeout(() => {
      setCurrentCardIndex(currentCardIndex - 1);
    }, 300);
  };

  return {
    currentCardIndex,
    interpretations,
    scrollRef,
    addCardInterpretation,
    moveToNextCard,
    moveToPreviousCard,
  };
};

/**
 * INTEGRATION WITH MCQ MODAL:
 * Show encrypted card interpretation → MCQ modal → Next card (scroll to top + new animation)
 */
export const CardWithMCQFlow = ({ card, interpretation, mcqQuestions, onComplete }) => {
  const [showMCQ, setShowMCQ] = useState(false);
  const [revealComplete, setRevealComplete] = useState(false);
  const scrollRef = useRef(null);

  const handleRevealComplete = () => {
    setRevealComplete(true);
    // Auto-show MCQ modal after interpretation reveal completes
    setTimeout(() => setShowMCQ(true), 1000);
  };

  const handleMCQComplete = (answers) => {
    setShowMCQ(false);
    // Scroll to top before moving to next card
    scrollRef.current?.scrollTo({ y: 0, animated: true });
    setTimeout(() => onComplete(answers), 300);
  };

  return (
    <View>
      <ScrollView ref={scrollRef}>
        <EncryptedTextReveal
          trigger={true}
          quantumSeed={generateQuantumSeed()}
          onComplete={handleRevealComplete}
        >
          {interpretation}
        </EncryptedTextReveal>
      </ScrollView>

      {/* MCQModal would go here */}
      {showMCQ && <Text>MCQ Modal Here</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  header: {
    padding: 20,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  cardNumber: {
    fontSize: 12,
    color: '#9966ff',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  cardName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  position: {
    fontSize: 14,
    color: '#ccccff',
    fontStyle: 'italic',
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    marginHorizontal: 20,
    borderRadius: 2,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#9966ff',
    borderRadius: 2,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  interpretationContainer: {
    // EncryptedTextReveal will apply its own styles
  },
  footer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  navButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#9966ff',
    alignItems: 'center',
  },
  navButtonDisabled: {
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  navButtonText: {
    fontSize: 16,
    color: '#9966ff',
    fontWeight: 'bold',
  },
  navButtonTextDisabled: {
    color: 'rgba(255, 255, 255, 0.3)',
  },
  nextButton: {
    flex: 2,
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#9966ff',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  nextButtonText: {
    fontSize: 16,
    color: '#ffffff',
    fontWeight: 'bold',
  },
});

export default CardInterpretationScreen;
