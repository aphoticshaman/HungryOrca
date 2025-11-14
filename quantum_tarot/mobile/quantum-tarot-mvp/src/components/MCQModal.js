import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

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
      <LinearGradient
        colors={['#1a0033', '#330066', '#4d0099']}
        style={styles.container}
      >
        <SafeAreaView style={styles.safeArea}>
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.headerTop}>
              <Text style={styles.cardContext}>
                Card {cardNumber} of {totalCards}: {cardName}
              </Text>
              <TouchableOpacity onPress={handleSkip} style={styles.skipButton}>
                <Text style={styles.skipButtonText}>Skip All</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.progressContainer}>
              <Text style={styles.progressText}>
                Question {currentQuestionIndex + 1} of {questions.length}
              </Text>
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
              <Text style={styles.questionType}>{getQuestionTypeLabel(currentQuestion.type)}</Text>
              <Text style={styles.questionText}>{currentQuestion.question}</Text>

              {currentQuestion.subtext && (
                <Text style={styles.questionSubtext}>{currentQuestion.subtext}</Text>
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
                        <Text style={styles.scaleNumber}>{index + 1}</Text>
                      </View>
                    )}
                    <Text style={styles.optionText}>{option.text || option}</Text>
                  </View>
                  {option.description && (
                    <Text style={styles.optionDescription}>{option.description}</Text>
                  )}
                </TouchableOpacity>
              ))}
            </View>

            {currentQuestionIndex > 0 && (
              <TouchableOpacity style={styles.backButton} onPress={handleBack}>
                <Text style={styles.backButtonText}>← Previous Question</Text>
              </TouchableOpacity>
            )}
          </ScrollView>

          {/* Footer hint */}
          <View style={styles.footer}>
            <Text style={styles.footerText}>
              {isLastQuestion
                ? 'Last question for this card'
                : `${questions.length - currentQuestionIndex - 1} more question${
                    questions.length - currentQuestionIndex - 1 === 1 ? '' : 's'
                  }`}
            </Text>
          </View>
        </SafeAreaView>
      </LinearGradient>
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
  },
  safeArea: {
    flex: 1,
  },
  header: {
    padding: 16,
    paddingTop: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  cardContext: {
    fontSize: 12,
    color: '#9966ff',
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
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  skipButtonText: {
    fontSize: 12,
    color: '#ccccff',
  },
  progressContainer: {
    gap: 6,
  },
  progressText: {
    fontSize: 12,
    color: '#ccccff',
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#9966ff',
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
    color: '#9966ff',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  questionText: {
    fontSize: 20,
    color: '#ffffff',
    lineHeight: 28,
    fontWeight: '500',
    marginBottom: 8,
  },
  questionSubtext: {
    fontSize: 14,
    color: '#ccccff',
    lineHeight: 20,
    fontStyle: 'italic',
  },
  optionsContainer: {
    gap: 12,
  },
  optionButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 18,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
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
    backgroundColor: '#9966ff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scaleNumber: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  optionText: {
    fontSize: 16,
    color: '#ffffff',
    lineHeight: 22,
    flex: 1,
  },
  optionDescription: {
    fontSize: 13,
    color: '#ccccff',
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
    color: '#ccccff',
  },
  footer: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: '#9966ff',
    fontStyle: 'italic',
  },
});

export default MCQModal;
