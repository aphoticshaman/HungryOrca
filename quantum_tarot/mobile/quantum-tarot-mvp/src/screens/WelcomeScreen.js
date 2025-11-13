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
      navigation.navigate('Onboarding');
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

      <ScrollView contentContainerStyle={styles.content}>
        {/* Cyberpunk header */}
        <CyberpunkHeader showMatrixBg={false} />

        {/* Welcome message with LPMUD colors */}
        <View style={styles.messageBox}>
          <LPMUDText style={styles.message}>
            $HIC${'>'} QUANTUM TAROT ${'<'}$NOR${'\n'}
            $HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR${'\n\n'}

            $HIW$Welcome.$NOR${'\n\n'}

            $HIG$Offline AGI interpretation$NOR${'\n'}
            $HIG$Quantum randomization$NOR${'\n'}
            $HIG$Terminal interface$NOR${'\n\n'}

            $HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR${'\n'}
            $HIC$BEGIN$NOR$
          </LPMUDText>
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
    paddingTop: 0,
  },
  messageBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 15,
    marginVertical: 20,
    backgroundColor: '#000000',
  },
  message: {
    fontSize: 13,
    fontFamily: 'monospace',
    lineHeight: 18,
  },
  demoBox: {
    marginVertical: 20,
    alignItems: 'center',
    gap: 15,
  },
  demoText: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  startButton: {
    padding: 20,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiYellow,
    backgroundColor: '#000000',
    alignItems: 'center',
    marginVertical: 20,
  },
  startButtonText: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  featureBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    padding: 15,
    marginTop: 20,
    backgroundColor: '#000000',
  },
  featureTitle: {
    fontSize: 14,
    fontFamily: 'monospace',
    marginBottom: 10,
  },
  featureText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 18,
  },
  versionText: {
    fontSize: 9,
    fontFamily: 'monospace',
    textAlign: 'center',
    marginTop: 20,
    marginBottom: 40,
  },
});
