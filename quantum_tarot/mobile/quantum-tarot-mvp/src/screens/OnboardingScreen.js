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

$HIW$First, we map your psyche.$NOR$

The cards respond to $HIM$who you are$NOR$,
not just what you ask.

10 questions. No bullshit.
DBT + CBT + MRT framework.

This isn't astrology.
This is $HIG$psychological profiling$NOR$.

$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$`
    },
    {
      title: '$HIC$02 | THE AGI ENGINE$NOR$',
      content: `$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$

$HIW$LunatiQ - Offline AGI$NOR$

No cloud. No API calls.
No tracking. $HIG$Zero bullshit$NOR$.

4-layer interpretation:
  $HIM$→$NOR$ Fuzzy Orchestrator
  $HIM$→$NOR$ 5 Specialized Agents
  $HIM$→$NOR$ Ensemble Blender
  $HIM$→$NOR$ Adaptive Language

It $HIM$learns your voice$NOR$.
It $HIG$evolves with you$NOR$.

$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$`
    },
    {
      title: '$HIC$03 | QUANTUM RANDOMNESS$NOR$',
      content: `$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$

$HIW$Hardware RNG + SHA-256$NOR$

Not Math.random().
Not Date.now().

True quantum entropy from
your device's $HIG$crypto hardware$NOR$.

Mixed with SHA-256 hashing
and your unique intention.

The cards are $HIM$truly random$NOR$.
The interpretation is $HIM$deeply personal$NOR$.

$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$`
    },
    {
      title: '$HIC$04 | TERMINAL AESTHETIC$NOR$',
      content: `$HIY$━━━━━━━━━━━━━━━━━━━━━━━━━━$NOR$

$HIW$No skeuomorphism.$NOR$
$HIW$No pretty pictures.$NOR$
$HIW$No mystical bullshit.$NOR$

Just pure $HIC$terminal hacker vibes$NOR$:
  $HIG$✓$NOR$ LPMUD color codes
  $HIG$✓$NOR$ Matrix rain effects
  $HIG$✓$NOR$ CRT scan lines
  $HIG$✓$NOR$ Neon glows everywhere
  $HIG$✓$NOR$ Glitch animations
  $HIG$✓$NOR$ ASCII art cards

This is our $HIM$moat$NOR$.
No other tarot app does this.

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
