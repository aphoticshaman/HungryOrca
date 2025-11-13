import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView, Alert } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { canDrawReading, getTimeUntilNextReading, isPremiumUser } from '../utils/storage';

export default function IntentionScreen({ route, navigation }) {
  // Validate route params
  if (!route || !route.params || !route.params.readingType || !route.params.profile) {
    // Navigate back to safety if params missing
    React.useEffect(() => {
      navigation.navigate('ReadingType');
    }, []);
    return null;
  }

  const { readingType, profile } = route.params;
  const { theme } = useTheme();
  const [intention, setIntention] = useState('');
  const [spreadType, setSpreadType] = useState('three_card');

  const styles = createStyles(theme);

  const SPREADS = [
    { type: 'single_card', name: 'Single Card', desc: 'Quick guidance' },
    { type: 'three_card', name: 'Past-Present-Future', desc: 'Classic 3-card' },
    { type: 'relationship', name: 'Relationship', desc: '6-card deep dive' }
  ];

  async function handleDrawCards() {
    if (!intention.trim()) {
      Alert.alert('Hold on', 'What question do you hold in your heart?');
      return;
    }

    // Check daily limit (free tier)
    const premium = await isPremiumUser();
    if (!premium) {
      const canDraw = await canDrawReading();
      if (!canDraw) {
        const timeLeft = await getTimeUntilNextReading();
        const hoursLeft = Math.ceil(timeLeft / (1000 * 60 * 60));
        Alert.alert(
          'Daily Limit Reached',
          `Free tier: 1 reading per day.\nNext reading in ${hoursLeft} hours.\n\nUpgrade to Premium for unlimited readings!`,
          [
            { text: 'Maybe Later', style: 'cancel' },
            { text: 'Upgrade ($3.99)', onPress: () => {} }  // TODO: IAP
          ]
        );
        return;
      }
    }

    navigation.navigate('CardDrawing', {
      readingType,
      profile,
      intention,
      spreadType
    });
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>
{`╔═══════════════════════════════╗
║   SET YOUR INTENTION          ║
╚═══════════════════════════════╝`}
      </Text>

      <Text style={styles.label}>What question do you hold?</Text>
      <TextInput
        style={styles.textArea}
        value={intention}
        onChangeText={setIntention}
        placeholder="What do I need to know about..."
        placeholderTextColor={theme.textDim}
        multiline
        numberOfLines={4}
      />

      <Text style={styles.label}>Choose your spread:</Text>
      {SPREADS.map((spread) => (
        <TouchableOpacity
          key={spread.type}
          style={[
            styles.spreadOption,
            spreadType === spread.type && styles.spreadSelected
          ]}
          onPress={() => setSpreadType(spread.type)}
        >
          <Text style={styles.spreadName}>{spread.name}</Text>
          <Text style={styles.spreadDesc}>{spread.desc}</Text>
        </TouchableOpacity>
      ))}

      <TouchableOpacity style={styles.button} onPress={handleDrawCards}>
        <Text style={styles.buttonText}>✧ DRAW CARDS ✧</Text>
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
    label: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      marginBottom: 10,
      marginTop: 20
    },
    textArea: {
      fontFamily: 'monospace',
      fontSize: 14,
      color: theme.text,
      borderWidth: 1,
      borderColor: theme.border,
      padding: 12,
      backgroundColor: theme.background,
      minHeight: 100,
      textAlignVertical: 'top'
    },
    spreadOption: {
      borderWidth: 1,
      borderColor: theme.border,
      padding: 15,
      marginBottom: 10
    },
    spreadSelected: {
      borderColor: theme.accent,
      backgroundColor: theme.accent + '20'
    },
    spreadName: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.text,
      marginBottom: 4
    },
    spreadDesc: {
      fontFamily: 'monospace',
      fontSize: 10,
      color: theme.textDim
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
