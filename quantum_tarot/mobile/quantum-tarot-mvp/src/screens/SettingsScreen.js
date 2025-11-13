import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useTheme, ASCII_THEMES } from '../context/ThemeContext';

export default function SettingsScreen({ navigation }) {
  const { theme, changeTheme, themes } = useTheme();
  const styles = createStyles(theme);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>
{`╔═══════════════════════════════╗
║   SETTINGS                    ║
╚═══════════════════════════════╝`}
      </Text>

      <Text style={styles.section}>COLOR THEME</Text>

      {Object.values(themes).map((themeOption) => (
        <TouchableOpacity
          key={themeOption.id}
          style={[
            styles.themeOption,
            { borderColor: themeOption.border }
          ]}
          onPress={() => changeTheme(themeOption.id)}
        >
          <View style={[styles.themeSwatch, { backgroundColor: themeOption.text }]} />
          <Text style={[styles.themeName, { color: themeOption.text }]}>
            {themeOption.name}
            {theme.id === themeOption.id && ' ✓'}
          </Text>
        </TouchableOpacity>
      ))}

      <Text style={styles.section}>ABOUT</Text>
      <Text style={styles.about}>
        Quantum Tarot: Retro Edition{'\n'}
        Version 1.0.0{'\n\n'}

        Built with genuine quantum randomness,{'\n'}
        personality profiling, and adaptive{'\n'}
        language delivery.{'\n\n'}

        No servers. No subscriptions.{'\n'}
        No data collection.{'\n\n'}

        Just pure tarot.
      </Text>

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.buttonText}>◀ BACK</Text>
      </TouchableOpacity>
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
      marginTop: 40,
      marginBottom: 30
    },
    section: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      marginTop: 20,
      marginBottom: 10
    },
    themeOption: {
      flexDirection: 'row',
      alignItems: 'center',
      borderWidth: 1,
      padding: 15,
      marginBottom: 10
    },
    themeSwatch: {
      width: 20,
      height: 20,
      marginRight: 15
    },
    themeName: {
      fontFamily: 'monospace',
      fontSize: 12
    },
    about: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.textDim,
      lineHeight: 16
    },
    button: {
      borderWidth: 2,
      borderColor: theme.border,
      paddingVertical: 15,
      marginTop: 30
    },
    buttonText: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.text,
      textAlign: 'center'
    }
  });
}
