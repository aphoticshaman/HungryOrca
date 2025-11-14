import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Modal,
  Pressable,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MBTI_QUESTIONS, calculateMBTI } from '../utils/mbtiTest';

/**
 * MBTI Personality Test Screen
 *
 * Features:
 * - 40-question MBTI battery across 4 dimensions (E/I, S/N, T/F, J/P)
 * - "Vibe Mode" checkbox to skip testing (with warning popup)
 * - Progress tracking
 * - Results calculation and storage
 *
 * Flow:
 * 1. User sees intro with vibe mode option
 * 2. If vibe mode checked → show warning popup
 * 3. If skip confirmed → proceed with basic profile
 * 4. If not skipped → complete 40 questions
 * 5. Calculate MBTI type and save to profile
 */

const MBTITestScreen = ({ navigation, route }) => {
  const { userProfile, onComplete } = route.params || {};

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [showIntro, setShowIntro] = useState(true);
  const [vibeModeChecked, setVibeModeChecked] = useState(false);
  const [showSkipWarning, setShowSkipWarning] = useState(false);

  const currentQuestion = MBTI_QUESTIONS[currentQuestionIndex];
  const progress = ((currentQuestionIndex + 1) / MBTI_QUESTIONS.length) * 100;

  const handleVibeModeToggle = () => {
    if (!vibeModeChecked) {
      // User is trying to enable vibe mode - show warning
      setShowSkipWarning(true);
    } else {
      // User is unchecking vibe mode - just toggle
      setVibeModeChecked(false);
    }
  };

  const handleSkipConfirm = () => {
    // User confirmed they want to skip MBTI test
    setVibeModeChecked(true);
    setShowSkipWarning(false);

    // Proceed with basic profile (no MBTI type)
    if (onComplete) {
      onComplete({
        ...userProfile,
        mbtiType: null,
        vibeModeEnabled: true,
      });
    }
    navigation.goBack();
  };

  const handleSkipCancel = () => {
    // User wants to complete their profile
    setShowSkipWarning(false);
    setVibeModeChecked(false);
  };

  const handleStartTest = () => {
    if (vibeModeChecked) {
      // Should not happen, but safety check
      setShowSkipWarning(true);
    } else {
      setShowIntro(false);
    }
  };

  const handleAnswer = (optionIndex) => {
    const newAnswers = [
      ...answers,
      {
        questionId: currentQuestion.id,
        selectedOptionIndex: optionIndex,
      },
    ];
    setAnswers(newAnswers);

    if (currentQuestionIndex < MBTI_QUESTIONS.length - 1) {
      // Move to next question
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      // Test complete - calculate results
      const result = calculateMBTI(newAnswers);

      if (onComplete) {
        onComplete({
          ...userProfile,
          mbtiType: result.type,
          mbtiScores: result.scores,
          mbtiStrengths: result.strengths,
          vibeModeEnabled: false,
        });
      }
      navigation.goBack();
    }
  };

  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
      setAnswers(answers.slice(0, -1));
    } else {
      setShowIntro(true);
    }
  };

  if (showIntro) {
    return (
      <LinearGradient
        colors={['#1a0033', '#330066', '#4d0099']}
        style={styles.container}
      >
        <SafeAreaView style={styles.safeArea}>
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <Text style={styles.title}>Personality Profile</Text>

            <Text style={styles.subtitle}>
              Complete this 40-question assessment to receive deeply personalized readings
              that speak your unique psychological language.
            </Text>

            <View style={styles.benefitsContainer}>
              <Text style={styles.benefitTitle}>What You'll Get:</Text>
              <Text style={styles.benefitText}>
                • Interpretations tailored to your cognitive patterns
              </Text>
              <Text style={styles.benefitText}>
                • Communication style that resonates with how you process information
              </Text>
              <Text style={styles.benefitText}>
                • Guidance that honors your natural strengths and blind spots
              </Text>
              <Text style={styles.benefitText}>
                • Truly unique syntheses (no two readings ever the same)
              </Text>
            </View>

            <View style={styles.estimateContainer}>
              <Text style={styles.estimateText}>
                Estimated time: 8-12 minutes
              </Text>
            </View>

            {/* Vibe Mode Checkbox */}
            <TouchableOpacity
              style={styles.vibeModeContainer}
              onPress={handleVibeModeToggle}
              activeOpacity={0.7}
            >
              <View style={[styles.checkbox, vibeModeChecked && styles.checkboxChecked]}>
                {vibeModeChecked && <Text style={styles.checkmark}>✓</Text>}
              </View>
              <View style={styles.vibeModeTextContainer}>
                <Text style={styles.vibeModeLabel}>Vibe Mode</Text>
                <Text style={styles.vibeModeSubtext}>
                  Skip personality test (less accurate readings)
                </Text>
              </View>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.startButton}
              onPress={handleStartTest}
              disabled={vibeModeChecked}
            >
              <Text style={styles.startButtonText}>
                {vibeModeChecked ? 'Uncheck Vibe Mode to Start' : 'Begin Assessment'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.skipButton}
              onPress={() => navigation.goBack()}
            >
              <Text style={styles.skipButtonText}>Maybe Later</Text>
            </TouchableOpacity>
          </ScrollView>
        </SafeAreaView>

        {/* Skip Warning Modal */}
        <Modal
          animationType="fade"
          transparent={true}
          visible={showSkipWarning}
          onRequestClose={() => setShowSkipWarning(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>⚠️ Less Accurate Readings</Text>

              <Text style={styles.modalText}>
                Skipping personality profiling means your readings will lack the deep
                personalization that makes them truly transformative.
              </Text>

              <Text style={styles.modalText}>
                Without knowing your cognitive patterns, the synthesis will be:
              </Text>

              <Text style={styles.modalBullet}>• Less accurate to your psychology</Text>
              <Text style={styles.modalBullet}>• Less unique and personalized</Text>
              <Text style={styles.modalBullet}>• More generic in tone and guidance</Text>

              <Text style={styles.modalQuestion}>
                Are you sure you want to proceed with Vibe Mode?
              </Text>

              <View style={styles.modalButtons}>
                <TouchableOpacity
                  style={[styles.modalButton, styles.modalButtonNo]}
                  onPress={handleSkipCancel}
                >
                  <Text style={styles.modalButtonTextNo}>
                    No, Complete My Profile
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.modalButton, styles.modalButtonYes]}
                  onPress={handleSkipConfirm}
                >
                  <Text style={styles.modalButtonTextYes}>
                    Yes, Proceed Anyway
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient
      colors={['#1a0033', '#330066', '#4d0099']}
      style={styles.container}
    >
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.header}>
          <TouchableOpacity onPress={handleBack} style={styles.backButton}>
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>

          <Text style={styles.progressText}>
            Question {currentQuestionIndex + 1} of {MBTI_QUESTIONS.length}
          </Text>
        </View>

        <View style={styles.progressBarContainer}>
          <View style={[styles.progressBar, { width: `${progress}%` }]} />
        </View>

        <ScrollView contentContainerStyle={styles.questionScrollContent}>
          <Text style={styles.dimensionLabel}>{currentQuestion.dimension}</Text>

          <Text style={styles.questionText}>{currentQuestion.question}</Text>

          <View style={styles.optionsContainer}>
            {currentQuestion.options.map((option, index) => (
              <TouchableOpacity
                key={index}
                style={styles.optionButton}
                onPress={() => handleAnswer(index)}
                activeOpacity={0.8}
              >
                <Text style={styles.optionText}>{option.text}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  questionScrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 16,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#ccccff',
    marginBottom: 24,
    textAlign: 'center',
    lineHeight: 24,
  },
  benefitsContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  benefitTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  benefitText: {
    fontSize: 14,
    color: '#e6e6ff',
    marginBottom: 8,
    lineHeight: 20,
  },
  estimateContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 8,
    padding: 12,
    marginBottom: 24,
  },
  estimateText: {
    fontSize: 14,
    color: '#ccccff',
    textAlign: 'center',
    fontStyle: 'italic',
  },
  vibeModeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  checkbox: {
    width: 28,
    height: 28,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#ccccff',
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  checkboxChecked: {
    backgroundColor: '#9966ff',
    borderColor: '#9966ff',
  },
  checkmark: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  vibeModeTextContainer: {
    flex: 1,
  },
  vibeModeLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  vibeModeSubtext: {
    fontSize: 12,
    color: '#ccccff',
    fontStyle: 'italic',
  },
  startButton: {
    backgroundColor: '#9966ff',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  startButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  skipButton: {
    padding: 12,
    alignItems: 'center',
  },
  skipButtonText: {
    fontSize: 14,
    color: '#ccccff',
    textDecorationLine: 'underline',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: 8,
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 16,
    color: '#ccccff',
  },
  progressText: {
    fontSize: 14,
    color: '#ccccff',
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    marginHorizontal: 16,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#9966ff',
    borderRadius: 2,
  },
  dimensionLabel: {
    fontSize: 12,
    color: '#9966ff',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  questionText: {
    fontSize: 20,
    color: '#ffffff',
    marginBottom: 32,
    lineHeight: 28,
    fontWeight: '500',
  },
  optionsContainer: {
    gap: 12,
  },
  optionButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 20,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  optionText: {
    fontSize: 16,
    color: '#ffffff',
    lineHeight: 22,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    backgroundColor: '#1a0033',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
    borderWidth: 2,
    borderColor: '#9966ff',
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 16,
    textAlign: 'center',
  },
  modalText: {
    fontSize: 16,
    color: '#ccccff',
    marginBottom: 12,
    lineHeight: 24,
  },
  modalBullet: {
    fontSize: 14,
    color: '#e6e6ff',
    marginBottom: 8,
    marginLeft: 12,
    lineHeight: 20,
  },
  modalQuestion: {
    fontSize: 16,
    color: '#ffffff',
    marginTop: 16,
    marginBottom: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  modalButtons: {
    gap: 12,
  },
  modalButton: {
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  modalButtonNo: {
    backgroundColor: '#9966ff',
  },
  modalButtonYes: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: '#666699',
  },
  modalButtonTextNo: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  modalButtonTextYes: {
    fontSize: 16,
    color: '#ccccff',
  },
});

export default MBTITestScreen;
