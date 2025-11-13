import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Animated, TouchableOpacity, Alert } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { QuantumSpreadEngine } from '../services/quantumEngine';
import { AdaptiveLanguageEngine } from '../services/adaptiveLanguage';
import { LunatiQEngine } from '../services/lunatiQEngine';
import { getCardByIndex } from '../data/tarotLoader';
import { recordReading, saveReading, getUserProfile } from '../utils/storage';
import { CARD_BACK } from '../data/asciiCards';
import CardImage from '../components/CardImage';

export default function CardDrawingScreen({ route, navigation }) {
  // Validate route params
  if (!route || !route.params || !route.params.readingType || !route.params.profile || !route.params.intention || !route.params.spreadType) {
    // Navigate back to safety if params missing
    React.useEffect(() => {
      navigation.navigate('ReadingType');
    }, []);
    return null;
  }

  const { readingType, profile, intention, spreadType } = route.params;
  const { theme } = useTheme();
  const [status, setStatus] = useState('Collapsing quantum wave function...');
  const [error, setError] = useState(null);
  const fadeAnim = useState(new Animated.Value(0))[0];

  const styles = createStyles(theme);

  useEffect(() => {
    performReading();
  }, []);

  async function performReading() {
    try {
      setError(null);

      // Animate card back
      Animated.loop(
        Animated.sequence([
          Animated.timing(fadeAnim, { toValue: 1, duration: 1000, useNativeDriver: true }),
          Animated.timing(fadeAnim, { toValue: 0.5, duration: 1000, useNativeDriver: true })
        ])
      ).start();

      // Build communication profile
      const userProfile = await getUserProfile();
      if (!userProfile) {
        throw new Error('Failed to load user profile');
      }

      const birthYear = userProfile.birthday ? new Date(userProfile.birthday).getFullYear() : null;
      const commProfile = AdaptiveLanguageEngine.buildCommunicationProfile(profile, birthYear);

      if (!commProfile) {
        throw new Error('Failed to build communication profile');
      }

      setStatus('Mixing intention with quantum entropy...');
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Perform quantum reading
      const engine = new QuantumSpreadEngine();
      setStatus('Drawing cards from the quantum field...');

      const reading = await engine.performReading(spreadType, intention, readingType);

      if (!reading || !reading.positions) {
        throw new Error('Invalid reading generated');
      }

      setStatus('Interpreting through multi-modal AGI...');
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Get all cards first for relational analysis
      const cards = reading.positions.map(pos => getCardByIndex(pos.cardIndex));

      // Initialize LunatiQ AGI engine
      const lunatiQ = new LunatiQEngine();

      // Generate AGI-powered interpretations
      const interpretedCards = lunatiQ.generateSpreadInterpretation(
        cards,
        reading.positions,
        profile,
        intention,
        commProfile
      );

      // Enrich with card data and AGI interpretations
      const enrichedReading = {
        ...reading,
        userIntention: intention,
        commProfile,
        cards: interpretedCards.map((interpreted, index) => ({
          ...reading.positions[index],
          card: interpreted.card,
          interpretation: interpreted.interpretation
        }))
      };

      // Validate enriched reading
      if (!enrichedReading.cards || enrichedReading.cards.length === 0) {
        throw new Error('No cards in reading');
      }

      // Save reading
      await saveReading(enrichedReading);
      await recordReading();

      // Navigate to result
      navigation.replace('Reading', { reading: enrichedReading });
    } catch (err) {
      console.error('Reading failed:', err);
      setError(err.message || 'Unknown error occurred');
      setStatus('Quantum disruption detected');
    }
  }

  function handleRetry() {
    setError(null);
    setStatus('Collapsing quantum wave function...');
    performReading();
  }

  function handleGoBack() {
    navigation.goBack();
  }

  return (
    <View style={styles.container}>
      <Text style={styles.status}>{status}</Text>

      {error ? (
        <View style={styles.errorContainer}>
          <Text style={styles.errorTitle}>✧ QUANTUM DISRUPTION ✧</Text>
          <Text style={styles.errorMessage}>{error}</Text>
          <View style={styles.errorButtons}>
            <TouchableOpacity style={styles.retryButton} onPress={handleRetry}>
              <Text style={styles.retryButtonText}>TRY AGAIN</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.backButton} onPress={handleGoBack}>
              <Text style={styles.backButtonText}>GO BACK</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <>
          <Animated.View style={{ opacity: fadeAnim }}>
            <Text style={styles.cardBack}>{CARD_BACK}</Text>
          </Animated.View>

          <Text style={styles.subtitle}>
            Genuine quantum randomness{'\n'}
            from your device hardware
          </Text>
        </>
      )}
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
    },
    errorContainer: {
      alignItems: 'center',
      padding: 20
    },
    errorTitle: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.accent,
      marginBottom: 20,
      textAlign: 'center'
    },
    errorMessage: {
      fontFamily: 'monospace',
      fontSize: 11,
      color: theme.text,
      marginBottom: 30,
      textAlign: 'center',
      lineHeight: 18
    },
    errorButtons: {
      flexDirection: 'column',
      gap: 10,
      width: '100%'
    },
    retryButton: {
      borderWidth: 2,
      borderColor: theme.accent,
      paddingVertical: 12,
      paddingHorizontal: 20,
      marginBottom: 10
    },
    retryButtonText: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.accent,
      textAlign: 'center'
    },
    backButton: {
      borderWidth: 1,
      borderColor: theme.border,
      paddingVertical: 12,
      paddingHorizontal: 20
    },
    backButtonText: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      textAlign: 'center'
    }
  });
}
