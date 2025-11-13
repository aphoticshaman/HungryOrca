import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { QuestionBank, PersonalityAnalyzer } from '../services/personalityProfiler';
import { savePersonalityProfile, getUserProfile } from '../utils/storage';

export default function PersonalityQuestionsScreen({ route, navigation }) {
  const { readingType } = route.params;
  const { theme } = useTheme();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [responses, setResponses] = useState({});

  const questions = QuestionBank.getQuestionsForType(readingType);
  const currentQuestion = questions[currentIndex];
  const progress = ((currentIndex + 1) / questions.length) * 100;

  const styles = createStyles(theme);

  async function handleAnswer(answer) {
    const newResponses = {
      ...responses,
      [currentQuestion.id]: answer
    };
    setResponses(newResponses);

    if (currentIndex < questions.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      // All questions answered
      await finishQuestionnaire(newResponses);
    }
  }

  async function finishQuestionnaire(finalResponses) {
    const userProfile = await getUserProfile();

    const profile = PersonalityAnalyzer.calculateProfile(
      readingType,
      finalResponses,
      userProfile.birthday,
      userProfile.name
    );

    await savePersonalityProfile(readingType, profile);
    navigation.navigate('Intention', { readingType, profile });
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>
          Question {currentIndex + 1} of {questions.length}
        </Text>
        <View style={styles.progressBar}>
          <View style={[styles.progressFill, { width: `${progress}%` }]} />
        </View>
      </View>

      <Text style={styles.question}>{currentQuestion.text}</Text>

      <View style={styles.options}>
        {currentQuestion.options.map((option, index) => (
          <TouchableOpacity
            key={index}
            style={styles.option}
            onPress={() => handleAnswer(option)}
          >
            <Text style={styles.optionText}>{option}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </ScrollView>
  );
}

function createStyles(theme) {
  return StyleSheet.create({
    container: {
      flexGrow: 1,
      backgroundColor: theme.background,
      padding: 20
    },
    header: {
      marginTop: 40,
      marginBottom: 30
    },
    title: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      textAlign: 'center',
      marginBottom: 10
    },
    progressBar: {
      height: 4,
      backgroundColor: theme.textDim,
      width: '100%'
    },
    progressFill: {
      height: 4,
      backgroundColor: theme.accent
    },
    question: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.text,
      marginBottom: 30,
      lineHeight: 22
    },
    options: {
      gap: 15
    },
    option: {
      borderWidth: 1,
      borderColor: theme.border,
      padding: 15
    },
    optionText: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text
    }
  });
}
