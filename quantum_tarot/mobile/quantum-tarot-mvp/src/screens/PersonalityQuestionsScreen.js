/**
 * PERSONALITY QUESTIONS SCREEN - Psych profiling
 */

import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NeonText, LPMUDText, FlickerText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const PROFILES_KEY = '@lunatiq_profiles';
const ACTIVE_PROFILE_KEY = '@lunatiq_active_profile';

// 10 personality profiling questions (DBT/CBT/MRT framework)
const QUESTIONS = [
  {
    id: 'emotional_regulation',
    question: 'When facing emotional intensity, I tend to:',
    options: [
      { value: 0.2, text: 'Shut down completely' },
      { value: 0.4, text: 'Struggle but try to cope' },
      { value: 0.6, text: 'Process it with some difficulty' },
      { value: 0.8, text: 'Navigate it fairly well' },
      { value: 1.0, text: 'Flow through it skillfully' }
    ]
  },
  {
    id: 'pattern_recognition',
    question: 'I notice repeating patterns in my life:',
    options: [
      { value: 0.2, text: 'Rarely or never' },
      { value: 0.4, text: 'Occasionally' },
      { value: 0.6, text: 'Sometimes' },
      { value: 0.8, text: 'Often' },
      { value: 1.0, text: 'Constantly' }
    ]
  },
  {
    id: 'authenticity',
    question: 'Living authentically (being my true self):',
    options: [
      { value: 0.2, text: 'Terrifies me' },
      { value: 0.4, text: 'Makes me uncomfortable' },
      { value: 0.6, text: 'Is challenging but important' },
      { value: 0.8, text: 'Feels natural most of the time' },
      { value: 1.0, text: 'Is my default state' }
    ]
  },
  {
    id: 'shadow_awareness',
    question: 'I am aware of my shadow aspects (hidden/denied parts):',
    options: [
      { value: 0.2, text: 'Not at all' },
      { value: 0.4, text: 'Vaguely' },
      { value: 0.6, text: 'Somewhat' },
      { value: 0.8, text: 'Quite well' },
      { value: 1.0, text: 'Intimately' }
    ]
  },
  {
    id: 'intuition_trust',
    question: 'I trust my intuition:',
    options: [
      { value: 0.2, text: 'Never' },
      { value: 0.4, text: 'Rarely' },
      { value: 0.6, text: 'Sometimes' },
      { value: 0.8, text: 'Usually' },
      { value: 1.0, text: 'Always' }
    ]
  },
  {
    id: 'change_adaptability',
    question: 'When life changes dramatically, I:',
    options: [
      { value: 0.2, text: 'Resist and struggle' },
      { value: 0.4, text: 'Find it very difficult' },
      { value: 0.6, text: 'Adapt eventually' },
      { value: 0.8, text: 'Adapt fairly quickly' },
      { value: 1.0, text: 'Flow with it naturally' }
    ]
  },
  {
    id: 'self_reflection',
    question: 'I engage in deep self-reflection:',
    options: [
      { value: 0.2, text: 'Never' },
      { value: 0.4, text: 'Rarely' },
      { value: 0.6, text: 'Sometimes' },
      { value: 0.8, text: 'Regularly' },
      { value: 1.0, text: 'Daily' }
    ]
  },
  {
    id: 'boundary_setting',
    question: 'Setting and maintaining boundaries:',
    options: [
      { value: 0.2, text: 'Impossible for me' },
      { value: 0.4, text: 'Very difficult' },
      { value: 0.6, text: 'Challenging but doable' },
      { value: 0.8, text: 'Fairly easy' },
      { value: 1.0, text: 'Natural and effortless' }
    ]
  },
  {
    id: 'ambiguity_tolerance',
    question: 'I handle uncertainty and ambiguity:',
    options: [
      { value: 0.2, text: 'Terribly' },
      { value: 0.4, text: 'Poorly' },
      { value: 0.6, text: 'Okay' },
      { value: 0.8, text: 'Well' },
      { value: 1.0, text: 'Excellently' }
    ]
  },
  {
    id: 'integration',
    question: 'I integrate insights into action:',
    options: [
      { value: 0.2, text: 'Never' },
      { value: 0.4, text: 'Rarely' },
      { value: 0.6, text: 'Sometimes' },
      { value: 0.8, text: 'Usually' },
      { value: 1.0, text: 'Consistently' }
    ]
  }
];

export default function PersonalityQuestionsScreen({ route, navigation }) {
  const { profileName, birthdate, zodiacSign } = route.params;
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});

  const question = QUESTIONS[currentQuestion];
  const isLastQuestion = currentQuestion === QUESTIONS.length - 1;

  const handleAnswer = async (value) => {
    const newAnswers = {
      ...answers,
      [question.id]: value
    };
    setAnswers(newAnswers);

    if (isLastQuestion) {
      // Save profile and set as active
      await saveProfile(newAnswers);
    } else {
      setCurrentQuestion(currentQuestion + 1);
    }
  };

  async function saveProfile(personalityProfile) {
    try {
      // Load existing profiles
      const profilesData = await AsyncStorage.getItem(PROFILES_KEY);
      const profiles = profilesData ? JSON.parse(profilesData) : [];

      // Create new profile
      const newProfile = {
        id: Date.now().toString(),
        name: profileName,
        birthdate,
        zodiacSign,
        personality: personalityProfile,
        createdAt: new Date().toISOString()
      };

      // Add to profiles array
      profiles.push(newProfile);
      await AsyncStorage.setItem(PROFILES_KEY, JSON.stringify(profiles));

      // Set as active profile
      await AsyncStorage.setItem(ACTIVE_PROFILE_KEY, newProfile.id);

      // Navigate to Welcome
      navigation.navigate('Welcome');
    } catch (error) {
      console.error('Error saving profile:', error);
    }
  }

  const handleBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    } else {
      navigation.goBack();
    }
  };

  const progress = ((currentQuestion + 1) / QUESTIONS.length) * 100;

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header with progress */}
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIC${'>'} PERSONALITY PROFILING$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.progress}>
            QUESTION {currentQuestion + 1} / {QUESTIONS.length}
          </NeonText>

          {/* Progress bar */}
          <View style={styles.progressBarBg}>
            <View
              style={[
                styles.progressBarFill,
                { width: `${progress}%` }
              ]}
            />
          </View>
        </View>

        {/* Question */}
        <View style={styles.questionBox}>
          <NeonText color={NEON_COLORS.hiWhite} style={styles.questionText}>
            {question.question}
          </NeonText>
        </View>

        {/* Options */}
        <View style={styles.optionsContainer}>
          {question.options.map((option, index) => (
            <TouchableOpacity
              key={index}
              onPress={() => handleAnswer(option.value)}
              style={styles.optionButton}
            >
              <LPMUDText style={styles.optionText}>
                $HIC$[{index + 1}]$NOR$ {option.text}
              </LPMUDText>
            </TouchableOpacity>
          ))}
        </View>

        {/* Back button */}
        <TouchableOpacity onPress={handleBack} style={styles.backButton}>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.backButtonText}>
            {'[ ← BACK ]'}
          </NeonText>
        </TouchableOpacity>

        {/* Context note */}
        <View style={styles.noteBox}>
          <LPMUDText style={styles.noteText}>
            $HIY$Your answers shape the interpretation.$NOR${'\n'}
            Be honest.
          </LPMUDText>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  content: {
    padding: 20,
  },
  header: {
    marginBottom: 30,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  headerTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 8,
    lineHeight: 22,
  },
  progress: {
    fontSize: 11,
    fontFamily: 'monospace',
    marginBottom: 10,
  },
  progressBarBg: {
    height: 4,
    backgroundColor: NEON_COLORS.dimCyan,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: NEON_COLORS.hiCyan,
  },
  questionBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 20,
    marginBottom: 25,
    backgroundColor: '#000000',
  },
  questionText: {
    fontSize: 15,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    lineHeight: 22,
  },
  optionsContainer: {
    gap: 12,
    marginBottom: 30,
  },
  optionButton: {
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    backgroundColor: '#000000',
  },
  optionText: {
    fontSize: 13,
    fontFamily: 'monospace',
    lineHeight: 18,
  },
  backButton: {
    padding: 15,
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
    marginBottom: 20,
  },
  backButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
  noteBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.dimYellow,
    padding: 12,
    backgroundColor: '#0a0a00',
  },
  noteText: {
    fontSize: 10,
    fontFamily: 'monospace',
    lineHeight: 15,
  },
});
