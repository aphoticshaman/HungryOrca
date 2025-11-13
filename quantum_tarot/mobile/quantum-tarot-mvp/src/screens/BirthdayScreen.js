/**
 * BIRTHDAY SCREEN - Enter birthday, show astro sign
 */

import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, TextInput } from 'react-native';
import { NeonText, LPMUDText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

// Calculate zodiac sign from month/day
function getZodiacSign(month, day) {
  const zodiac = [
    { name: 'Capricorn', start: [12, 22], end: [1, 19] },
    { name: 'Aquarius', start: [1, 20], end: [2, 18] },
    { name: 'Pisces', start: [2, 19], end: [3, 20] },
    { name: 'Aries', start: [3, 21], end: [4, 19] },
    { name: 'Taurus', start: [4, 20], end: [5, 20] },
    { name: 'Gemini', start: [5, 21], end: [6, 20] },
    { name: 'Cancer', start: [6, 21], end: [7, 22] },
    { name: 'Leo', start: [7, 23], end: [8, 22] },
    { name: 'Virgo', start: [8, 23], end: [9, 22] },
    { name: 'Libra', start: [9, 23], end: [10, 22] },
    { name: 'Scorpio', start: [10, 23], end: [11, 21] },
    { name: 'Sagittarius', start: [11, 22], end: [12, 21] }
  ];

  for (const sign of zodiac) {
    const [startMonth, startDay] = sign.start;
    const [endMonth, endDay] = sign.end;

    if (
      (month === startMonth && day >= startDay) ||
      (month === endMonth && day <= endDay)
    ) {
      return sign.name;
    }
  }

  return null;
}

export default function BirthdayScreen({ route, navigation }) {
  const { readingType } = route.params;
  const [month, setMonth] = useState('');
  const [day, setDay] = useState('');
  const [year, setYear] = useState('');
  const [zodiacSign, setZodiacSign] = useState(null);
  const [error, setError] = useState('');

  const handleValidate = () => {
    const m = parseInt(month);
    const d = parseInt(day);
    const y = parseInt(year);

    // Validate
    if (!m || !d || !y || m < 1 || m > 12 || d < 1 || d > 31 || y < 1900 || y > 2024) {
      setError('Invalid date');
      return;
    }

    // Calculate sign
    const sign = getZodiacSign(m, d);
    setZodiacSign(sign);
    setError('');
  };

  const handleContinue = () => {
    if (zodiacSign) {
      navigation.navigate('Intention', { readingType, zodiacSign, birthdate: { month, day, year } });
    }
  };

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIC${'>'} BIRTH DATE$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.headerSubtitle}>
            Enter to calculate your sign
          </NeonText>
        </View>

        {/* Input fields */}
        <View style={styles.inputContainer}>
          <View style={styles.inputRow}>
            <View style={styles.inputBox}>
              <NeonText color={NEON_COLORS.dimCyan} style={styles.inputLabel}>
                MONTH (1-12)
              </NeonText>
              <TextInput
                style={styles.input}
                value={month}
                onChangeText={setMonth}
                keyboardType="number-pad"
                maxLength={2}
                placeholder="MM"
                placeholderTextColor={NEON_COLORS.dimCyan}
              />
            </View>

            <View style={styles.inputBox}>
              <NeonText color={NEON_COLORS.dimCyan} style={styles.inputLabel}>
                DAY (1-31)
              </NeonText>
              <TextInput
                style={styles.input}
                value={day}
                onChangeText={setDay}
                keyboardType="number-pad"
                maxLength={2}
                placeholder="DD"
                placeholderTextColor={NEON_COLORS.dimCyan}
              />
            </View>

            <View style={styles.inputBox}>
              <NeonText color={NEON_COLORS.dimCyan} style={styles.inputLabel}>
                YEAR
              </NeonText>
              <TextInput
                style={styles.input}
                value={year}
                onChangeText={setYear}
                keyboardType="number-pad"
                maxLength={4}
                placeholder="YYYY"
                placeholderTextColor={NEON_COLORS.dimCyan}
              />
            </View>
          </View>

          {error && (
            <NeonText color={NEON_COLORS.hiRed} style={styles.errorText}>
              {'>'} {error}
            </NeonText>
          )}

          <TouchableOpacity onPress={handleValidate} style={styles.validateButton}>
            <NeonText color={NEON_COLORS.hiCyan} style={styles.validateButtonText}>
              {'[ CALCULATE SIGN ]'}
            </NeonText>
          </TouchableOpacity>
        </View>

        {/* Show zodiac sign */}
        {zodiacSign && (
          <View style={styles.resultBox}>
            <LPMUDText style={styles.resultText}>
              $HIY${'>'} YOUR SIGN$NOR${'\n'}
              $HIW${zodiacSign.toUpperCase()}$NOR$
            </LPMUDText>

            <TouchableOpacity onPress={handleContinue} style={styles.continueButton}>
              <NeonText color={NEON_COLORS.hiGreen} style={styles.continueButtonText}>
                {'[ CONTINUE ]'}
              </NeonText>
            </TouchableOpacity>
          </View>
        )}

        {/* Back button */}
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.backButtonText}>
            {'[ ← BACK ]'}
          </NeonText>
        </TouchableOpacity>
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
    marginBottom: 30,
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
    fontSize: 12,
    fontFamily: 'monospace',
  },
  inputContainer: {
    marginBottom: 30,
  },
  inputRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },
  inputBox: {
    flex: 1,
  },
  inputLabel: {
    fontSize: 10,
    fontFamily: 'monospace',
    marginBottom: 5,
  },
  input: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 12,
    fontSize: 16,
    fontFamily: 'monospace',
    color: NEON_COLORS.hiCyan,
    backgroundColor: '#000000',
    textAlign: 'center',
  },
  errorText: {
    fontSize: 12,
    fontFamily: 'monospace',
    marginBottom: 15,
  },
  validateButton: {
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    alignItems: 'center',
  },
  validateButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  resultBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.hiYellow,
    padding: 20,
    marginBottom: 30,
    alignItems: 'center',
  },
  resultText: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
    lineHeight: 24,
  },
  continueButton: {
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiGreen,
    alignItems: 'center',
    width: '100%',
  },
  continueButtonText: {
    fontSize: 14,
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
});
