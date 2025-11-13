import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { QuantumSpreadEngine } from '../services/quantumEngine';
import { AdaptiveLanguageEngine } from '../services/adaptiveLanguage';
import { getCardByIndex } from '../data/tarotLoader';
import { recordReading, saveReading, getUserProfile } from '../utils/storage';
import { CARD_BACK } from '../data/asciiCards';

export default function CardDrawingScreen({ route, navigation }) {
  const { readingType, profile, intention, spreadType } = route.params;
  const { theme } = useTheme();
  const [status, setStatus] = useState('Collapsing quantum wave function...');
  const fadeAnim = useState(new Animated.Value(0))[0];

  const styles = createStyles(theme);

  useEffect(() => {
    performReading();
  }, []);

  async function performReading() {
    // Animate card back
    Animated.loop(
      Animated.sequence([
        Animated.timing(fadeAnim, { toValue: 1, duration: 1000, useNativeDriver: true }),
        Animated.timing(fadeAnim, { toValue: 0.5, duration: 1000, useNativeDriver: true })
      ])
    ).start();

    // Build communication profile
    const userProfile = await getUserProfile();
    const birthYear = userProfile.birthday ? new Date(userProfile.birthday).getFullYear() : null;
    const commProfile = AdaptiveLanguageEngine.buildCommunicationProfile(profile, birthYear);

    setStatus('Mixing intention with quantum entropy...');
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Perform quantum reading
    const engine = new QuantumSpreadEngine();
    setStatus('Drawing cards from the quantum field...');

    const reading = await engine.performReading(spreadType, intention, readingType);

    setStatus('Interpreting...');
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Enrich with card data and interpretations
    const enrichedReading = {
      ...reading,
      userIntention: intention,
      commProfile,
      cards: reading.positions.map(pos => {
        const card = getCardByIndex(pos.cardIndex);
        const interpretation = AdaptiveLanguageEngine.generateCardInterpretation(
          card,
          pos.position,
          pos.reversed,
          commProfile,
          readingType
        );

        return {
          ...pos,
          card,
          interpretation
        };
      })
    };

    // Save reading
    await saveReading(enrichedReading);
    await recordReading();

    // Navigate to result
    navigation.replace('Reading', { reading: enrichedReading });
  }

  return (
    <View style={styles.container}>
      <Text style={styles.status}>{status}</Text>

      <Animated.View style={{ opacity: fadeAnim }}>
        <Text style={styles.cardBack}>{CARD_BACK}</Text>
      </Animated.View>

      <Text style={styles.subtitle}>
        Genuine quantum randomness{'\n'}
        from your device hardware
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
    status: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      marginBottom: 30,
      textAlign: 'center'
    },
    cardBack: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.text,
      textAlign: 'center'
    },
    subtitle: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.textDim,
      marginTop: 30,
      textAlign: 'center'
    }
  });
}
