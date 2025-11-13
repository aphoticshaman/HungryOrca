import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { getAsciiCard } from '../data/asciiCards';

export default function ReadingScreen({ route, navigation }) {
  // Validate route params
  if (!route || !route.params || !route.params.reading) {
    // Navigate back to safety if params missing
    React.useEffect(() => {
      navigation.navigate('Welcome');
    }, []);
    return null;
  }

  const { reading } = route.params;
  const { theme } = useTheme();
  const styles = createStyles(theme);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>
{`╔═══════════════════════════════╗
║   YOUR READING                ║
╚═══════════════════════════════╝`}
      </Text>

      <Text style={styles.intention}>
        "{reading.userIntention}"
      </Text>

      {reading.cards.map((cardPos, index) => (
        <View key={index} style={styles.cardContainer}>
          <Text style={styles.position}>{cardPos.position}</Text>

          <Text style={styles.asciiCard}>
            {getAsciiCard(cardPos.cardIndex, cardPos.reversed)}
          </Text>

          <Text style={styles.cardName}>
            {cardPos.card.name}
            {cardPos.reversed && ' (Reversed)'}
          </Text>

          <Text style={styles.interpretation}>
            {cardPos.interpretation}
          </Text>

          {index < reading.cards.length - 1 && (
            <View style={styles.divider} />
          )}
        </View>
      ))}

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Welcome')}
      >
        <Text style={styles.buttonText}>RETURN HOME</Text>
      </TouchableOpacity>

      <Text style={styles.signature}>
        Quantum Signature: {reading.positions[0].quantumSignature.slice(0, 16)}...
      </Text>
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
      marginBottom: 20
    },
    intention: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.textDim,
      fontStyle: 'italic',
      textAlign: 'center',
      marginBottom: 30
    },
    cardContainer: {
      marginBottom: 30
    },
    position: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.accent,
      marginBottom: 10,
      textAlign: 'center',
      fontWeight: 'bold'
    },
    asciiCard: {
      fontFamily: 'monospace',
      fontSize: 9,
      color: theme.text,
      textAlign: 'center',
      marginBottom: 15
    },
    cardName: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      textAlign: 'center',
      marginBottom: 10
    },
    interpretation: {
      fontFamily: 'monospace',
      fontSize: 11,
      color: theme.text,
      lineHeight: 18
    },
    divider: {
      height: 1,
      backgroundColor: theme.border,
      marginTop: 20
    },
    button: {
      borderWidth: 2,
      borderColor: theme.border,
      paddingVertical: 15,
      marginTop: 20,
      marginBottom: 10
    },
    buttonText: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.text,
      textAlign: 'center'
    },
    signature: {
      fontFamily: 'monospace',
      fontSize: 8,
      color: theme.textDim,
      textAlign: 'center',
      marginTop: 10,
      marginBottom: 20
    }
  });
}
