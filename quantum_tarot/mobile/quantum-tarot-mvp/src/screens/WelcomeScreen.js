import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { CARD_BACK } from '../data/asciiCards';

export default function WelcomeScreen({ navigation }) {
  const { theme } = useTheme();

  const styles = createStyles(theme);

  return (
    <View style={styles.container}>
      <Text style={styles.logo}>
{`
╔═══════════════════════════════╗
║                               ║
║    ██████  ██    ██  █████    ║
║   ██    ██ ██    ██ ██   ██   ║
║   ██    ██ ██    ██ ███████   ║
║   ██ ▄▄ ██ ██    ██ ██   ██   ║
║    ██████   ██████  ██   ██   ║
║       ▀▀                      ║
║                               ║
║   Q U A N T U M  T A R O T    ║
║                               ║
║      R E T R O  E D I T I O N ║
║                               ║
╚═══════════════════════════════╝
`}
      </Text>

      <Text style={styles.tagline}>
        Old-school ASCII art meets{'\n'}
        quantum divination
      </Text>

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Onboarding')}
      >
        <Text style={styles.buttonText}>
          ✧ BEGIN JOURNEY ✧
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.settingsButton}
        onPress={() => navigation.navigate('Settings')}
      >
        <Text style={styles.settingsText}>⚙ Settings</Text>
      </TouchableOpacity>

      <Text style={styles.footer}>
        No subscriptions. No data collection.{'\n'}
        Just pure tarot.
      </Text>
    </View>
  );
}

function createStyles(theme) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.background,
      justifyContent: 'center',
      alignItems: 'center',
      padding: 20
    },
    logo: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.text,
      textAlign: 'center',
      marginBottom: 20
    },
    tagline: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.textDim,
      textAlign: 'center',
      marginBottom: 40
    },
    button: {
      borderWidth: 2,
      borderColor: theme.border,
      paddingVertical: 15,
      paddingHorizontal: 40,
      marginBottom: 20
    },
    buttonText: {
      fontFamily: 'monospace',
      fontSize: 16,
      color: theme.text,
      textAlign: 'center'
    },
    settingsButton: {
      padding: 10
    },
    settingsText: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.textDim
    },
    footer: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.textDim,
      textAlign: 'center',
      position: 'absolute',
      bottom: 20
    }
  });
}
