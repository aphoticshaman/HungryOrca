import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useTheme } from '../context/ThemeContext';

const READING_TYPES = [
  { type: 'career', label: 'Career', icon: '💼', description: 'Work, ambition, purpose' },
  { type: 'romance', label: 'Romance', icon: '❤️', description: 'Love, relationships, connection' },
  { type: 'wellness', label: 'Wellness', icon: '🧘', description: 'Health, balance, self-care' },
  { type: 'family', label: 'Family', icon: '👨‍👩‍👧', description: 'Kin, roots, belonging' },
  { type: 'self_growth', label: 'Self-Growth', icon: '🌱', description: 'Inner work, evolution' },
  { type: 'school', label: 'Learning', icon: '📚', description: 'Education, skills, mastery' },
  { type: 'general', label: 'General', icon: '✨', description: 'Life guidance' },
  { type: 'shadow', label: 'Shadow Work', icon: '🌑', description: 'Deep exploration' }
];

export default function ReadingTypeScreen({ navigation }) {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  function selectType(type) {
    navigation.navigate('Questions', { readingType: type });
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>
{`╔═══════════════════════════════╗
║   CHOOSE YOUR PATH            ║
╚═══════════════════════════════╝`}
      </Text>

      <Text style={styles.subtitle}>
        What area of life calls to you?
      </Text>

      <View style={styles.grid}>
        {READING_TYPES.map((rt) => (
          <TouchableOpacity
            key={rt.type}
            style={styles.card}
            onPress={() => selectType(rt.type)}
          >
            <Text style={styles.icon}>{rt.icon}</Text>
            <Text style={styles.cardTitle}>{rt.label}</Text>
            <Text style={styles.cardDesc}>{rt.description}</Text>
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
    title: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.text,
      textAlign: 'center',
      marginBottom: 10,
      marginTop: 40
    },
    subtitle: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.textDim,
      textAlign: 'center',
      marginBottom: 30
    },
    grid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      justifyContent: 'space-between'
    },
    card: {
      width: '48%',
      borderWidth: 1,
      borderColor: theme.border,
      padding: 15,
      marginBottom: 15,
      alignItems: 'center'
    },
    icon: {
      fontSize: 32,
      marginBottom: 8
    },
    cardTitle: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      marginBottom: 4,
      textAlign: 'center'
    },
    cardDesc: {
      fontFamily: 'monospace',
      fontSize: 9,
      color: theme.textDim,
      textAlign: 'center'
    }
  });
}
