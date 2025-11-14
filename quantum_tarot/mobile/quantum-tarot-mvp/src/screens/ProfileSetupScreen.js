/**
 * PROFILE SETUP SCREEN - Name and birthday entry
 */

import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, TextInput, Dimensions } from 'react-native';
import { NeonText, LPMUDText, MatrixRain } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

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

export default function ProfileSetupScreen({ navigation }) {
  const [name, setName] = useState('');
  const [month, setMonth] = useState('');
  const [day, setDay] = useState('');
  const [year, setYear] = useState('');
  const [zodiacSign, setZodiacSign] = useState(null);
  const [error, setError] = useState('');

  // Auto-calculate zodiac sign
  React.useEffect(() => {
    if (month && day) {
      const m = parseInt(month);
      const d = parseInt(day);
      if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
        const sign = getZodiacSign(m, d);
        setZodiacSign(sign);
        setError('');
      }
    }
  }, [month, day]);

  const handleContinue = () => {
    // Validate
    if (!name.trim()) {
      setError('Enter a profile name');
      return;
    }

    if (!month || !day || !year) {
      setError('Enter complete birth date');
      return;
    }

    const m = parseInt(month);
    const d = parseInt(day);
    const y = parseInt(year);

    if (m < 1 || m > 12 || d < 1 || d > 31 || y < 1900 || y > 2025) {
      setError('Invalid date');
      return;
    }

    if (!zodiacSign) {
      setError('Invalid date');
      return;
    }

    // Go to MBTI personality test (40 questions)
    navigation.navigate('MBTITest', {
      userProfile: {
        profileName: name.trim(),
        birthdate: `${year}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`,
        zodiacSign
      }
    });
  };

  return (
    <View style={styles.container}>
      {/* Matrix rain background */}
      <View style={StyleSheet.absoluteFill}>
        <MatrixRain width={SCREEN_WIDTH} height={SCREEN_HEIGHT} speed={30} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIM${'>'} CREATE PROFILE$NOR$
          </LPMUDText>
        </View>

        {/* Name input */}
        <View style={styles.section}>
          <LPMUDText style={styles.label}>
            $HIY${'>'} PROFILE NAME$NOR$
          </LPMUDText>
          <TextInput
            style={styles.nameInput}
            value={name}
            onChangeText={setName}
            placeholder="Enter name"
            placeholderTextColor={NEON_COLORS.dimCyan}
            maxLength={20}
          />
        </View>

        {/* Birthday input */}
        <View style={styles.section}>
          <LPMUDText style={styles.label}>
            $HIY${'>'} BIRTH DATE$NOR$
          </LPMUDText>
          <View style={styles.dateRow}>
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

          {zodiacSign && (
            <View style={styles.zodiacDisplay}>
              <LPMUDText style={styles.zodiacText}>
                $HIM${'>'} $HIW${zodiacSign.toUpperCase()}$NOR$
              </LPMUDText>
            </View>
          )}
        </View>

        {/* Error */}
        {error && (
          <NeonText color={NEON_COLORS.hiRed} style={styles.errorText}>
            {'>'} {error}
          </NeonText>
        )}

        {/* Continue button */}
        <TouchableOpacity onPress={handleContinue} style={styles.continueButton}>
          <LPMUDText style={styles.continueButtonText}>
            $HIC${'[ '} $HIW$CONTINUE TO QUESTIONS$NOR$ $HIC${' ]'}$NOR$
          </LPMUDText>
        </TouchableOpacity>

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
    flexGrow: 1,
    padding: 20,
  },
  header: {
    marginBottom: 30,
    borderBottomWidth: 2,
    borderBottomColor: NEON_COLORS.dimCyan,
    paddingBottom: 15,
  },
  headerTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    lineHeight: 22,
  },
  section: {
    marginBottom: 25,
  },
  label: {
    fontSize: 14,
    fontFamily: 'monospace',
    marginBottom: 10,
    lineHeight: 18,
  },
  nameInput: {
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    padding: 15,
    fontSize: 16,
    fontFamily: 'monospace',
    color: NEON_COLORS.hiWhite,
    backgroundColor: '#000000',
  },
  dateRow: {
    flexDirection: 'row',
    gap: 10,
  },
  inputBox: {
    flex: 1,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 10,
    backgroundColor: '#000000',
  },
  inputLabel: {
    fontSize: 10,
    fontFamily: 'monospace',
    marginBottom: 5,
  },
  input: {
    color: NEON_COLORS.hiCyan,
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    backgroundColor: '#000000',
    textAlign: 'center',
  },
  zodiacDisplay: {
    marginTop: 15,
    padding: 10,
    borderWidth: 1,
    borderColor: NEON_COLORS.hiMagenta,
    backgroundColor: 'rgba(255, 0, 255, 0.1)',
  },
  zodiacText: {
    fontSize: 14,
    fontFamily: 'monospace',
    textAlign: 'center',
    lineHeight: 18,
  },
  errorText: {
    fontSize: 11,
    fontFamily: 'monospace',
    marginBottom: 15,
  },
  continueButton: {
    padding: 18,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    backgroundColor: 'rgba(0, 255, 255, 0.1)',
    marginTop: 20,
  },
  continueButtonText: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
    lineHeight: 20,
  },
  backButton: {
    padding: 15,
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
    marginTop: 15,
  },
  backButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
});
