import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import EncryptedTextReveal from '../components/EncryptedTextReveal';
import MCQModal from '../components/MCQModal';
import { CARD_DATABASE } from '../data/cardDatabase';
import { generateQuantumSeed } from '../utils/quantumRNG';
import { generatePostCardQuestions } from '../utils/postCardQuestions';
import { generateMegaSynthesis } from '../utils/megaSynthesisEngine';
import { NeonText, LPMUDText, MatrixRain, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

/**
 * CARD INTERPRETATION SCREEN WITH MCQ INTEGRATION
 *
 * Flow per card:
 * 1. Show encrypted reveal animation (7.5s sacred ritual)
 * 2. After reveal complete → auto-show MCQ modal (1-3 questions)
 * 3. User answers MCQs → modal closes
 * 4. User clicks "Next Card" → repeat
 * 5. After all cards + MCQs → generate mega synthesis → navigate to synthesis screen
 */

const CardInterpretationScreen = ({ route, navigation }) => {
  const {
    cards,
    interpretations,
    spreadType,
    intention,
    readingType,
    zodiacSign,
    birthdate,
    userProfile // Full profile with MBTI, personality, etc.
  } = route.params || {};

  const [currentCardIndex, setCurrentCardIndex] = useState(0);
  const [revealTrigger, setRevealTrigger] = useState(false);
  const [quantumSeed, setQuantumSeed] = useState(generateQuantumSeed());
  const [showMCQModal, setShowMCQModal] = useState(false);
  const [currentMCQs, setCurrentMCQs] = useState([]);
  const [allMCQAnswers, setAllMCQAnswers] = useState([]);
  const [isGeneratingSynthesis, setIsGeneratingSynthesis] = useState(false);

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

  // After encrypted reveal completes, show MCQ modal
  const handleRevealComplete = () => {
    console.log('Reveal complete for card', currentCardIndex + 1);

    // Generate MCQ questions for this card
    const questions = generatePostCardQuestions(
      cards[currentCardIndex],
      intention,
      readingType,
      currentCardIndex + 1, // Card number (1-indexed)
      cards.length,
      allMCQAnswers
    );

    setCurrentMCQs(questions);

    // Show MCQ modal after a brief pause (1 second)
    setTimeout(() => {
      setShowMCQModal(true);
    }, 1000);
  };

  // User completes MCQs for current card
  const handleMCQComplete = (answers) => {
    // Store answers with card context
    const answersWithContext = answers.map(answer => ({
      ...answer,
      cardIndex: currentCardIndex,
      cardName: CARD_DATABASE[cards[currentCardIndex].cardIndex].name,
      position: cards[currentCardIndex].position
    }));

    setAllMCQAnswers([...allMCQAnswers, ...answersWithContext]);
    setShowMCQModal(false);
  };

  // User skips MCQs for current card
  const handleMCQSkip = () => {
    setShowMCQModal(false);
  };

  const handleNextCard = async () => {
    if (currentCardIndex < cards.length - 1) {
      // Move to next card
      setCurrentCardIndex(currentCardIndex + 1);
    } else {
      // All cards + MCQs complete - generate mega synthesis
      await generateAndNavigateToSynthesis();
    }
  };

  const handlePreviousCard = () => {
    if (currentCardIndex > 0) {
      setCurrentCardIndex(currentCardIndex - 1);
    }
  };

  const generateAndNavigateToSynthesis = async () => {
    setIsGeneratingSynthesis(true);

    try {
      // Generate mega synthesis with all context
      const synthesis = await generateMegaSynthesis({
        cards,
        mcqAnswers: allMCQAnswers,
        userProfile: userProfile || { zodiacSign, birthdate },
        intention,
        readingType,
        spreadType
      });

      // Navigate to synthesis display screen
      navigation.navigate('Synthesis', {
        synthesis,
        cards,
        intention,
        readingType,
        spreadType
      });
    } catch (error) {
      console.error('Synthesis generation error:', error);
      // Fall back to simple synthesis screen
      navigation.navigate('Synthesis', {
        synthesis: 'Error generating synthesis. Please try again.',
        cards,
        intention,
        readingType,
        spreadType
      });
    } finally {
      setIsGeneratingSynthesis(false);
    }
  };

  const currentCard = cards[currentCardIndex];
  const currentInterpretation = interpretations?.[currentCardIndex];
  const cardData = CARD_DATABASE[currentCard?.cardIndex];

  const cardName = cardData ? `${cardData.name}${currentCard.reversed ? ' (Reversed)' : ''}` : 'Card';

  // Debug logging
  useEffect(() => {
    console.log('🎴 CardInterpretation Debug:', {
      currentCardIndex,
      hasInterpretations: !!interpretations,
      interpretationsLength: interpretations?.length,
      currentInterpretation: currentInterpretation ? currentInterpretation.substring(0, 100) + '...' : 'MISSING',
      hasCardData: !!cardData
    });
  }, [currentCardIndex, interpretations, currentInterpretation]);

  // Safety check
  if (!cards || !interpretations || !currentCard || !cardData) {
    return (
      <View style={styles.container}>
        <View style={styles.safeArea}>
          <LPMUDText style={styles.errorText}>
            $HIR$ERROR: Missing card data or interpretations$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.errorDetails}>
            Cards: {cards?.length || 0} | Interpretations: {interpretations?.length || 0}
          </NeonText>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Matrix rain background */}
      <View style={StyleSheet.absoluteFill}>
        <MatrixRain width={SCREEN_WIDTH} height={SCREEN_HEIGHT} speed={30} />
      </View>
      <ScanLines />

      <View style={styles.safeArea}>
        {/* Header */}
        <View style={styles.header}>
          <LPMUDText style={styles.cardNumber}>
            $HIY$CARD {currentCardIndex + 1} / {cards.length}$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.hiCyan} style={styles.cardName}>
            {cardName}
          </NeonText>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.position}>
            {currentCard.positionMeaning || currentCard.position}
          </NeonText>
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
            onComplete={handleRevealComplete}
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
            disabled={currentCardIndex === 0 || isGeneratingSynthesis}
          >
            <LPMUDText
              style={[
                styles.navButtonText,
                currentCardIndex === 0 && styles.navButtonTextDisabled,
              ]}
            >
              {currentCardIndex === 0 ? '$DIM$← Previous$NOR$' : '$HIC$← Previous$NOR$'}
            </LPMUDText>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.nextButton}
            onPress={handleNextCard}
            disabled={isGeneratingSynthesis}
          >
            <LPMUDText style={styles.nextButtonText}>
              {isGeneratingSynthesis ? (
                '$DIM$Generating Synthesis...$NOR$'
              ) : currentCardIndex === cards.length - 1 ? (
                '$HIG$[ VIEW SYNTHESIS → ]$NOR$'
              ) : (
                '$HIC$Next Card →$NOR$'
              )}
            </LPMUDText>
          </TouchableOpacity>
        </View>
      </View>

      {/* MCQ Modal */}
      <MCQModal
        visible={showMCQModal}
        questions={currentMCQs}
        cardName={cardData?.name || 'Card'}
        cardNumber={currentCardIndex + 1}
        totalCards={cards.length}
        onComplete={handleMCQComplete}
        onSkip={handleMCQSkip}
      />
    </View>
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
    backgroundColor: '#000000',
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
