import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { saveUserProfile } from '../utils/storage';

export default function OnboardingScreen({ navigation }) {
  const { theme } = useTheme();
  const [name, setName] = useState('');
  const [birthday, setBirthday] = useState('');
  const [pronouns, setPronouns] = useState('');

  const styles = createStyles(theme);

  async function handleContinue() {
    if (!name || !birthday) {
      alert('Please enter your name and birthday');
      return;
    }

    await saveUserProfile(name, birthday, pronouns);
    navigation.navigate('ReadingType');
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>
{`╔═══════════════════════════════╗
║   WELCOME TO THE QUANTUM      ║
╚═══════════════════════════════╝`}
      </Text>

      <Text style={styles.subtitle}>
        To personalize your readings,{'\n'}tell us a bit about yourself
      </Text>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Your name:</Text>
        <TextInput
          style={styles.input}
          value={name}
          onChangeText={setName}
          placeholder="Luna"
          placeholderTextColor={theme.textDim}
        />
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Birthday (YYYY-MM-DD):</Text>
        <TextInput
          style={styles.input}
          value={birthday}
          onChangeText={setBirthday}
          placeholder="1995-06-15"
          placeholderTextColor={theme.textDim}
        />
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Pronouns (optional):</Text>
        <TextInput
          style={styles.input}
          value={pronouns}
          onChangeText={setPronouns}
          placeholder="she/her, they/them, he/him, etc."
          placeholderTextColor={theme.textDim}
        />
      </View>

      <TouchableOpacity style={styles.button} onPress={handleContinue}>
        <Text style={styles.buttonText}>CONTINUE ▶</Text>
      </TouchableOpacity>

      <Text style={styles.privacy}>
        🔒 All data stays on your device.{'\n'}
        Nothing is sent to servers.
      </Text>
    </ScrollView>
  );
}

function createStyles(theme) {
  return StyleSheet.create({
    container: {
      flexGrow: 1,
      backgroundColor: theme.background,
      padding: 20,
      justifyContent: 'center'
    },
    title: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.text,
      textAlign: 'center',
      marginBottom: 10
    },
    subtitle: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.textDim,
      textAlign: 'center',
      marginBottom: 30
    },
    inputContainer: {
      marginBottom: 20
    },
    label: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      marginBottom: 8
    },
    input: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.text,
      borderWidth: 1,
      borderColor: theme.border,
      padding: 12,
      backgroundColor: theme.background
    },
    button: {
      borderWidth: 2,
      borderColor: theme.border,
      paddingVertical: 15,
      marginTop: 10,
      marginBottom: 20
    },
    buttonText: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.text,
      textAlign: 'center'
    },
    privacy: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.textDim,
      textAlign: 'center'
    }
  });
}
