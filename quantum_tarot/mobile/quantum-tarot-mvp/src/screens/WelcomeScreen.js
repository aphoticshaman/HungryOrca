/**
 * WELCOME SCREEN - Cyberpunk terminal entry point with birthday
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, TextInput, Dimensions } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import CyberpunkHeader from '../components/CyberpunkHeader';
import { NeonText, LPMUDText, MatrixRain } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const STORAGE_KEY = '@lunatiq_birthday';

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

export default function WelcomeScreen({ navigation }) {
  const [month, setMonth] = useState('');
  const [day, setDay] = useState('');
  const [year, setYear] = useState('');
  const [zodiacSign, setZodiacSign] = useState(null);
  const [rememberMe, setRememberMe] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadSavedBirthday();
  }, []);

  // Auto-calculate zodiac when date is complete
  useEffect(() => {
    if (month && day && month.length >= 1 && day.length >= 1) {
      const m = parseInt(month);
      const d = parseInt(day);
      if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
        const sign = getZodiacSign(m, d);
        setZodiacSign(sign);
        setError('');
      }
    }
  }, [month, day]);

  async function loadSavedBirthday() {
    try {
      const saved = await AsyncStorage.getItem(STORAGE_KEY);
      if (saved) {
        const { month: m, day: d, year: y } = JSON.parse(saved);
        setMonth(m);
        setDay(d);
        setYear(y);
        const sign = getZodiacSign(parseInt(m), parseInt(d));
        setZodiacSign(sign);
      }
    } catch (error) {
      console.error('Error loading birthday:', error);
    }
  }

  async function saveBirthday() {
    if (rememberMe && month && day && year) {
      try {
        await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ month, day, year }));
      } catch (error) {
        console.error('Error saving birthday:', error);
      }
    }
  }

  async function clearBirthday() {
    try {
      await AsyncStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Error clearing birthday:', error);
    }
  }

  const handleStart = async () => {
    const m = parseInt(month);
    const d = parseInt(day);
    const y = parseInt(year);

    // Validate
    if (!m || !d || !y || m < 1 || m > 12 || d < 1 || d > 31 || y < 1900 || y > 2025) {
      setError('Invalid date - check your input');
      return;
    }

    if (!zodiacSign) {
      setError('Could not calculate zodiac sign');
      return;
    }

    // Save if remember me is checked
    if (rememberMe) {
      await saveBirthday();
    } else {
      await clearBirthday();
    }

    // Navigate to reading type selection
    navigation.navigate('ReadingType', { zodiacSign, birthdate: { month, day, year } });
  };

  return (
    <View style={styles.container}>
      {/* Multicolor Matrix rain background */}
      <View style={StyleSheet.absoluteFill}>
        <MatrixRain width={SCREEN_WIDTH} height={SCREEN_HEIGHT} speed={30} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        {/* Cyberpunk header with wave animation */}
        <CyberpunkHeader showMatrixBg={false} />

        {/* Birthday input section */}
        <View style={styles.birthdaySection}>
          <LPMUDText style={styles.birthdayLabel}>
            $HIY${'>'} BIRTH DATE$NOR$
          </LPMUDText>

          <View style={styles.inputRow}>
            <View style={styles.inputBox}>
              <NeonText color={NEON_COLORS.dimCyan} style={styles.inputLabel}>
                MM
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
                DD
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
                YYYY
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

          {/* Zodiac sign display */}
          {zodiacSign && (
            <View style={styles.zodiacBox}>
              <LPMUDText style={styles.zodiacText}>
                $HIC$SIGN:$NOR$ $HIW${zodiacSign.toUpperCase()}$NOR$
              </LPMUDText>
            </View>
          )}

          {/* Error display */}
          {error && (
            <NeonText color={NEON_COLORS.hiRed} style={styles.errorText}>
              {'>'} {error}
            </NeonText>
          )}

          {/* Remember me checkbox */}
          <TouchableOpacity
            onPress={() => setRememberMe(!rememberMe)}
            style={styles.checkboxRow}
          >
            <View style={styles.checkbox}>
              {rememberMe && (
                <NeonText color={NEON_COLORS.hiCyan} style={styles.checkmark}>
                  ✓
                </NeonText>
              )}
            </View>
            <NeonText color={NEON_COLORS.dimWhite} style={styles.checkboxLabel}>
              Remember my birthday
            </NeonText>
          </TouchableOpacity>
        </View>

        {/* Start button */}
        <TouchableOpacity
          onPress={handleStart}
          style={styles.startButton}
        >
          <NeonText
            color={NEON_COLORS.hiCyan}
            style={styles.startButtonText}
          >
            {'[ START ]'}
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
    flexGrow: 1,
    paddingBottom: 40,
  },
  birthdaySection: {
    marginHorizontal: 20,
    marginTop: 20,
    padding: 20,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
  },
  birthdayLabel: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 15,
    lineHeight: 18,
  },
  inputRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  inputBox: {
    flex: 1,
  },
  inputLabel: {
    fontSize: 9,
    fontFamily: 'monospace',
    marginBottom: 5,
    textAlign: 'center',
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
  zodiacBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.hiYellow,
    padding: 10,
    marginBottom: 15,
    alignItems: 'center',
  },
  zodiacText: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    lineHeight: 18,
  },
  errorText: {
    fontSize: 11,
    fontFamily: 'monospace',
    marginBottom: 15,
  },
  checkboxRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkmark: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  checkboxLabel: {
    fontSize: 12,
    fontFamily: 'monospace',
  },
  startButton: {
    margin: 20,
    padding: 25,
    borderWidth: 3,
    borderColor: NEON_COLORS.hiCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    alignItems: 'center',
  },
  startButtonText: {
    fontSize: 24,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  spacer: {
    height: 20,
  },
});
