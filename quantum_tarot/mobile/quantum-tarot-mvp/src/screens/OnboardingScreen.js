/**
 * ONBOARDING SCREEN - Terminal tutorial
 */

import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { NeonText, LPMUDText, FlickerText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

export default function OnboardingScreen({ navigation }) {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: '$HIC$01 | PERSONALITY PROFILING$NOR$',
      content: `$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$

$HIW$First: 10 questions about you.$NOR$

The interpretation adapts to
$HIM$your psychological profile$NOR$.

DBT + CBT + MRT framework.

$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$`
    },
    {
      title: '$HIC$02 | OFFLINE AGI$NOR$',
      content: `$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$

$HIW$LunatIQ Tarot Engine$NOR$

Runs locally on your device.
No cloud. No API calls.

4-layer interpretation system
adapts to your profile.

$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$`
    },
    {
      title: '$HIC$03 | QUANTUM RNG$NOR$',
      content: `$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$

$HIW$Hardware randomization$NOR$

Uses your device's crypto hardware
for true quantum entropy.

Mixed with SHA-256 and your
intention for unique draws.

$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$`
    }
  ];

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(step + 1);
    } else {
      navigation.navigate('Questions');
    }
  };

  const handleSkip = () => {
    navigation.navigate('Questions');
  };

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <NeonText color={NEON_COLORS.hiCyan} style={styles.headerTitle}>
            {'>'} SYSTEM INITIALIZATION
          </NeonText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.headerSubtitle}>
            STEP {step + 1} OF {steps.length}
          </NeonText>
        </View>

        {/* Step content */}
        <View style={styles.stepBox}>
          <LPMUDText style={styles.stepTitle}>
            {steps[step].title}
          </LPMUDText>

          <LPMUDText style={styles.stepContent}>
            {steps[step].content}
          </LPMUDText>
        </View>

        {/* Navigation */}
        <View style={styles.buttonRow}>
          <TouchableOpacity onPress={handleSkip} style={styles.skipButton}>
            <NeonText color={NEON_COLORS.dimCyan} style={styles.skipButtonText}>
              {'[ SKIP ]'}
            </NeonText>
          </TouchableOpacity>

          <TouchableOpacity onPress={handleNext} style={styles.nextButton}>
            <FlickerText
              color={NEON_COLORS.hiCyan}
              style={styles.nextButtonText}
              flickerSpeed={200}
            >
              {step < steps.length - 1 ? '[ NEXT ]' : '[ BEGIN ]'}
            </FlickerText>
          </TouchableOpacity>
        </View>

        {/* Progress indicators */}
        <View style={styles.progressRow}>
          {steps.map((_, i) => (
            <View
              key={i}
              style={[
                styles.progressDot,
                {
                  borderColor: i === step ? NEON_COLORS.hiCyan : NEON_COLORS.dimCyan,
                  backgroundColor: i === step ? NEON_COLORS.cyan : 'transparent',
                }
              ]}
            />
          ))}
        </View>
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
  },
  headerSubtitle: {
    fontSize: 11,
    fontFamily: 'monospace',
  },
  stepBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 20,
    marginBottom: 30,
    backgroundColor: '#000000',
  },
  stepTitle: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 15,
    lineHeight: 20,
  },
  stepContent: {
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 18,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 30,
  },
  skipButton: {
    padding: 15,
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    flex: 0.4,
    alignItems: 'center',
  },
  skipButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
  nextButton: {
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    flex: 0.55,
    alignItems: 'center',
  },
  nextButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  progressRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
  },
  progressDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
  },
});
