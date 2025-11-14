/**
 * INTENTION SCREEN - Set reading intention
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, TextInput } from 'react-native';
import { NeonText, LPMUDText, FlickerText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';
import { validateIntention } from '../utils/intentionValidator';

const SPREAD_TYPES = [
  {
    id: 'single_card',
    name: 'SINGLE CARD',
    description: 'Quick guidance snapshot',
    cardCount: 1,
    pattern: 'linear'
  },
  {
    id: 'three_card',
    name: 'PAST-PRESENT-FUTURE',
    description: 'Timeline analysis',
    cardCount: 3,
    pattern: 'linear'
  },
  {
    id: 'daily',
    name: 'DAILY CHECK-IN',
    description: 'Focus | Avoid | Gift',
    cardCount: 3,
    pattern: 'linear'
  },
  {
    id: 'decision',
    name: 'DECISION TREE',
    description: 'Path A vs Path B analysis',
    cardCount: 6,
    pattern: 'tree'
  },
  {
    id: 'relationship',
    name: 'RELATIONSHIP',
    description: 'Deep connection analysis',
    cardCount: 6,
    pattern: 'spatial'
  },
  {
    id: 'celtic_cross',
    name: 'CELTIC CROSS',
    description: 'Comprehensive 10-card spread',
    cardCount: 10,
    pattern: 'spatial'
  }
];

export default function IntentionScreen({ route, navigation }) {
  const { readingType, zodiacSign, birthdate } = route.params;
  const [intention, setIntention] = useState('');
  const [spreadType, setSpreadType] = useState('three_card');
  const [error, setError] = useState('');
  const [validation, setValidation] = useState(null);

  const selectedSpread = SPREAD_TYPES.find(s => s.id === spreadType);

  // Validate intention on change
  useEffect(() => {
    if (intention.trim().length > 0) {
      const result = validateIntention(intention);
      setValidation(result);
    } else {
      setValidation(null);
    }
  }, [intention]);

  const handleContinue = () => {
    if (!intention.trim()) {
      setError('Intention required');
      return;
    }

    if (intention.trim().length < 3) {
      setError('Intention too short');
      return;
    }

    // Validate with 5W+H checker
    const validationResult = validateIntention(intention);

    // BLOCK if validation fails (score < 33%)
    if (!validationResult.valid) {
      setError('AGI REFUSES: Intention lacks context. See feedback below.');
      return;
    }

    // Navigate to card drawing
    navigation.navigate('CardDrawing', {
      readingType,
      zodiacSign,
      birthdate,
      intention: intention.trim(),
      spreadType
    });
  };

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIC${'>'} SET INTENTION$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.headerSubtitle}>
            {readingType.toUpperCase()} | {zodiacSign}
          </NeonText>
        </View>

        {/* Intention input */}
        <View style={styles.inputSection}>
          <LPMUDText style={styles.inputLabel}>
            $HIY$QUESTION:$NOR$
          </LPMUDText>

          <TextInput
            style={styles.textInput}
            value={intention}
            onChangeText={(text) => {
              setIntention(text);
              setError('');
            }}
            placeholder="What guidance do you seek?"
            placeholderTextColor={NEON_COLORS.dimCyan}
            multiline
            numberOfLines={4}
            maxLength={1000}
          />

          {/* 5W+H Validation Feedback */}
          {validation && (
            <View style={[
              styles.validationBox,
              validation.valid ? styles.validationGood : styles.validationPoor
            ]}>
              <View style={styles.validationHeader}>
                <NeonText
                  color={validation.score >= 0.67 ? NEON_COLORS.hiGreen : validation.score >= 0.33 ? NEON_COLORS.hiYellow : NEON_COLORS.hiRed}
                  style={styles.validationScore}
                >
                  5W+H SCORE: {Math.round(validation.score * 100)}%
                </NeonText>
                <NeonText
                  color={NEON_COLORS.dimCyan}
                  style={styles.validationElements}
                >
                  [{validation.present.join(', ').toUpperCase()}]
                </NeonText>
              </View>
              <LPMUDText style={styles.validationFeedback}>
                {validation.score >= 0.67 ? '$HIG$' : validation.score >= 0.33 ? '$HIY$' : '$HIR$'}
                {validation.feedback}
                $NOR$
              </LPMUDText>
            </View>
          )}

          {error && (
            <NeonText color={NEON_COLORS.hiRed} style={styles.errorText}>
              {'>'} {error}
            </NeonText>
          )}

          <NeonText color={NEON_COLORS.dimCyan} style={styles.charCount}>
            {intention.length} / 1000
          </NeonText>
        </View>

        {/* Spread selection */}
        <View style={styles.spreadSection}>
          <LPMUDText style={styles.spreadLabel}>
            $HIY$SPREAD TYPE:$NOR$
          </LPMUDText>

          <View style={styles.spreadList}>
            {SPREAD_TYPES.map((spread) => (
              <TouchableOpacity
                key={spread.id}
                onPress={() => setSpreadType(spread.id)}
                style={[
                  styles.spreadCard,
                  spreadType === spread.id && styles.spreadCardSelected
                ]}
              >
                <View style={styles.spreadHeader}>
                  <LPMUDText style={styles.spreadName}>
                    {spreadType === spread.id ? '$HIC$' : '$NOR$'}
                    {spread.name}
                    $NOR$
                  </LPMUDText>
                  <NeonText
                    color={spreadType === spread.id ? NEON_COLORS.hiCyan : NEON_COLORS.dimCyan}
                    style={styles.cardCount}
                  >
                    [{spread.cardCount} CARDS]
                  </NeonText>
                </View>

                <NeonText
                  color={NEON_COLORS.dimWhite}
                  style={styles.spreadDescription}
                >
                  {spread.description}
                </NeonText>

                {spreadType === spread.id && (
                  <View style={styles.selectedIndicator}>
                    <NeonText color={NEON_COLORS.hiCyan} style={styles.selectedText}>
                      {'[ SELECTED ]'}
                    </NeonText>
                  </View>
                )}
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Continue button */}
        <TouchableOpacity
          onPress={handleContinue}
          style={styles.continueButton}
        >
          <FlickerText
            color={NEON_COLORS.hiGreen}
            style={styles.continueButtonText}
            flickerSpeed={150}
          >
            {'[ DRAW CARDS ]'}
          </FlickerText>
        </TouchableOpacity>

        {/* Back button */}
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        >
          <NeonText color={NEON_COLORS.dimCyan} style={styles.backButtonText}>
            {'[ ← BACK ]'}
          </NeonText>
        </TouchableOpacity>

        <View style={styles.spacer} />
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
    marginBottom: 25,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  headerTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 5,
    lineHeight: 22,
  },
  headerSubtitle: {
    fontSize: 11,
    fontFamily: 'monospace',
  },
  inputSection: {
    marginBottom: 25,
  },
  inputLabel: {
    fontSize: 12,
    fontFamily: 'monospace',
    marginBottom: 10,
    lineHeight: 16,
  },
  textInput: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 12,
    fontSize: 14,
    fontFamily: 'monospace',
    color: NEON_COLORS.hiWhite,
    backgroundColor: '#000000',
    minHeight: 100,
    textAlignVertical: 'top',
  },
  errorText: {
    fontSize: 11,
    fontFamily: 'monospace',
    marginTop: 8,
  },
  charCount: {
    fontSize: 9,
    fontFamily: 'monospace',
    marginTop: 5,
    textAlign: 'right',
  },
  validationBox: {
    marginTop: 12,
    padding: 12,
    borderWidth: 2,
    backgroundColor: '#000000',
  },
  validationGood: {
    borderColor: NEON_COLORS.hiGreen,
  },
  validationPoor: {
    borderColor: NEON_COLORS.hiRed,
  },
  validationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  validationScore: {
    fontSize: 11,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  validationElements: {
    fontSize: 9,
    fontFamily: 'monospace',
  },
  validationFeedback: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  spreadSection: {
    marginBottom: 25,
  },
  spreadLabel: {
    fontSize: 12,
    fontFamily: 'monospace',
    marginBottom: 10,
    lineHeight: 16,
  },
  spreadList: {
    gap: 10,
  },
  spreadCard: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 12,
    backgroundColor: '#000000',
  },
  spreadCardSelected: {
    borderColor: NEON_COLORS.hiCyan,
    borderWidth: 3,
  },
  spreadHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  spreadName: {
    fontSize: 13,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    lineHeight: 16,
  },
  cardCount: {
    fontSize: 10,
    fontFamily: 'monospace',
  },
  spreadDescription: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  selectedIndicator: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: NEON_COLORS.hiCyan,
  },
  selectedText: {
    fontSize: 10,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  continueButton: {
    padding: 18,
    borderWidth: 3,
    borderColor: NEON_COLORS.hiGreen,
    alignItems: 'center',
    marginBottom: 15,
  },
  continueButtonText: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  backButton: {
    padding: 15,
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
  },
  backButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
  spacer: {
    height: 40,
  },
});
