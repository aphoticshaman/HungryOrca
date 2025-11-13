/**
 * WELCOME SCREEN - Cyberpunk terminal entry point
 */

import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import CyberpunkHeader from '../components/CyberpunkHeader';
import { NeonText, LPMUDText, GlitchText, FlickerText, MorphText, MatrixRain } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

export default function WelcomeScreen({ navigation }) {
  const [showMatrix, setShowMatrix] = useState(false);

  const handleStart = () => {
    setShowMatrix(true);
    setTimeout(() => {
      navigation.navigate('ReadingType');
    }, 1500);
  };

  return (
    <View style={styles.container}>
      {/* Matrix rain effect on start */}
      {showMatrix && (
        <View style={StyleSheet.absoluteFill}>
          <MatrixRain width={400} height={SCREEN_HEIGHT} speed={40} />
        </View>
      )}

      <View style={styles.content}>
        {/* Cyberpunk header */}
        <CyberpunkHeader showMatrixBg={false} />

        {/* Spacer */}
        <View style={styles.spacer} />

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
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  content: {
    flex: 1,
    justifyContent: 'space-between',
  },
  spacer: {
    flex: 1,
  },
  startButton: {
    margin: 20,
    padding: 25,
    borderWidth: 3,
    borderColor: NEON_COLORS.hiCyan,
    backgroundColor: '#000000',
    alignItems: 'center',
  },
  startButtonText: {
    fontSize: 24,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
});
