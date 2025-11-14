import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import { NeonText, LPMUDText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

/**
 * MCQ MODAL COMPONENT
 *
 * Displays 1-3 multiple choice questions after each card draw
 * to detect:
 * - Cognitive dissonance (stated priorities vs emotional reactions)
 * - Resonance levels
 * - Avoidance patterns
 * - Action readiness
 *
 * Question types:
 * - Resonance (1-5 scale)
 * - Aspect (element, keywords, archetype, symbols)
 * - Emotion (excitement, resistance, validation, confusion)
 * - Confirmation (amplify, contradict, expand)
 * - Situation (work, relationships, internal patterns)
 * - Action (immediate, planned, reflect, not ready)
 * - Takeaway (pattern recognition)
 * - Readiness (ready, process, explore, overwhelmed, skeptical)
 */

const MCQModal = ({
  visible,
  questions,
  cardName,
  cardNumber,
  totalCards,
  onComplete,
  onSkip,
}) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState([]);

  const currentQuestion = questions?.[currentQuestionIndex];
  const isLastQuestion = currentQuestionIndex === questions?.length - 1;
  const progress = ((currentQuestionIndex + 1) / (questions?.length || 1)) * 100;

  const handleAnswer = (optionIndex) => {
    const newAnswers = [
      ...answers,
      {
        questionType: currentQuestion.type,
        question: currentQuestion.question,
        selectedOption: currentQuestion.options[optionIndex],
        selectedOptionIndex: optionIndex,
      },
    ];

    setAnswers(newAnswers);

    if (isLastQuestion) {
      // All questions answered - complete
      onComplete(newAnswers);
      resetModal();
    } else {
      // Move to next question
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
      setAnswers(answers.slice(0, -1));
    }
  };

  const handleSkip = () => {
    onSkip();
    resetModal();
  };

  const resetModal = () => {
    setCurrentQuestionIndex(0);
    setAnswers([]);
  };

  if (!visible || !currentQuestion) {
    return null;
  }

  return (
    <Modal
      animationType="slide"
      transparent={false}
      visible={visible}
      onRequestClose={handleSkip}
    >
      <View style={styles.container}>
        <ScanLines />
        <SafeAreaView style={styles.safeArea}>
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.headerTop}>
              <LPMUDText style={styles.cardContext}>
                $HIC$Card {cardNumber} of {totalCards}: {cardName}$NOR$
              </LPMUDText>
              <TouchableOpacity onPress={handleSkip} style={styles.skipButton}>
                <LPMUDText style={styles.skipButtonText}>$HIY$Skip All$NOR$</LPMUDText>
              </TouchableOpacity>
            </View>

            <View style={styles.progressContainer}>
              <NeonText color={NEON_COLORS.dimCyan} style={styles.progressText}>
                Question {currentQuestionIndex + 1} of {questions.length}
              </NeonText>
              <View style={styles.progressBarContainer}>
                <View style={[styles.progressBar, { width: `${progress}%` }]} />
              </View>
            </View>
          </View>

          {/* Question Content */}
          <ScrollView
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={false}
          >
            <View style={styles.questionContainer}>
              <LPMUDText style={styles.questionType}>$HIY${getQuestionTypeLabel(currentQuestion.type)}$NOR$</LPMUDText>
              <NeonText color={NEON_COLORS.hiWhite} style={styles.questionText}>{currentQuestion.question}</NeonText>

              {currentQuestion.subtext && (
                <NeonText color={NEON_COLORS.dimCyan} style={styles.questionSubtext}>{currentQuestion.subtext}</NeonText>
              )}
            </View>

            <View style={styles.optionsContainer}>
              {currentQuestion.options.map((option, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.optionButton}
                  onPress={() => handleAnswer(index)}
                  activeOpacity={0.8}
                >
                  <View style={styles.optionContent}>
                    {currentQuestion.type === 'resonance' && (
                      <View style={styles.scaleIndicator}>
                        <NeonText color={NEON_COLORS.hiBlack} style={styles.scaleNumber}>{index + 1}</NeonText>
                      </View>
                    )}
                    <NeonText color={NEON_COLORS.hiCyan} style={styles.optionText}>{option.text || option}</NeonText>
                  </View>
                  {option.description && (
                    <NeonText color={NEON_COLORS.dimCyan} style={styles.optionDescription}>{option.description}</NeonText>
                  )}
                </TouchableOpacity>
              ))}
            </View>

            {currentQuestionIndex > 0 && (
              <TouchableOpacity style={styles.backButton} onPress={handleBack}>
                <LPMUDText style={styles.backButtonText}>$HIC$← Previous Question$NOR$</LPMUDText>
              </TouchableOpacity>
            )}
          </ScrollView>

          {/* Footer hint */}
          <View style={styles.footer}>
            <NeonText color={NEON_COLORS.dimYellow} style={styles.footerText}>
              {isLastQuestion
                ? 'Last question for this card'
                : `${questions.length - currentQuestionIndex - 1} more question${
                    questions.length - currentQuestionIndex - 1 === 1 ? '' : 's'
                  }`}
            </NeonText>
          </View>
        </SafeAreaView>
      </View>
    </Modal>
  );
};

/**
 * Get human-readable label for question type
 */
function getQuestionTypeLabel(type) {
  const labels = {
    resonance: 'Resonance Check',
    aspect: 'What Stands Out',
    emotion: 'Emotional Response',
    confirmation: 'Card Relationship',
    situation: 'Life Context',
    action: 'Action Readiness',
    takeaway: 'Pattern Recognition',
    readiness: 'Next Steps',
  };

  return labels[type] || 'Question';
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  safeArea: {
    flex: 1,
  },
  header: {
    padding: 16,
    paddingTop: 8,
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  cardContext: {
    fontSize: 12,
    fontWeight: 'bold',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    flex: 1,
  },
  skipButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: NEON_COLORS.hiYellow,
  },
  skipButtonText: {
    fontSize: 12,
  },
  progressContainer: {
    gap: 6,
  },
  progressText: {
    fontSize: 12,
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: 'rgba(0, 255, 255, 0.2)',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: NEON_COLORS.hiCyan,
    borderRadius: 2,
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  questionContainer: {
    marginBottom: 28,
  },
  questionType: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  questionText: {
    fontSize: 20,
    lineHeight: 28,
    fontWeight: '500',
    marginBottom: 8,
  },
  questionSubtext: {
    fontSize: 14,
    lineHeight: 20,
    fontStyle: 'italic',
  },
  optionsContainer: {
    gap: 12,
  },
  optionButton: {
    backgroundColor: 'rgba(0, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 18,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
  },
  optionContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  scaleIndicator: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: NEON_COLORS.hiCyan,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scaleNumber: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  optionText: {
    fontSize: 16,
    lineHeight: 22,
    flex: 1,
  },
  optionDescription: {
    fontSize: 13,
    marginTop: 6,
    lineHeight: 18,
    fontStyle: 'italic',
  },
  backButton: {
    marginTop: 24,
    padding: 12,
    alignItems: 'center',
  },
  backButtonText: {
    fontSize: 14,
  },
  footer: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    fontStyle: 'italic',
  },
});

export default MCQModal;
