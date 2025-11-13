/**
 * CYBERPUNK HEADER - Glitchy neon title screen
 */

import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { NeonText, LPMUDText, GlitchText, MatrixRain } from './TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function CyberpunkHeader({ showMatrixBg = false }) {
  return (
    <View style={styles.container}>
      {showMatrixBg && (
        <View style={StyleSheet.absoluteFill}>
          <MatrixRain width={SCREEN_WIDTH} height={200} speed={60} />
        </View>
      )}

      <View style={styles.content}>
        {/* Main title with glitch effect */}
        <GlitchText
          style={styles.mainTitle}
          glitchChance={0.03}
          glitchSpeed={200}
        >
          {'   QUANTUM   '}
        </GlitchText>

        {/* Subtitle with LPMUD colors */}
        <LPMUDText style={styles.subtitle}>
          $HIC$╔═══════════════════════╗$NOR${'\n'}
          $HIC$║$HIM$    T A R O T        $HIC$║$NOR${'\n'}
          $HIC$╚═══════════════════════╝$NOR$
        </LPMUDText>

        {/* Tagline */}
        <NeonText
          color={NEON_COLORS.dimCyan}
          style={styles.tagline}
        >
          {'>'} RETRO TERMINAL EDITION {'<'}
        </NeonText>

        {/* Version info */}
        <NeonText
          color={NEON_COLORS.dimYellow}
          style={styles.version}
        >
          v1.0.0 | SDK 54 | OFFLINE AGI
        </NeonText>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    paddingVertical: 30,
    paddingHorizontal: 20,
    backgroundColor: '#000000',
    borderBottomWidth: 2,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  content: {
    alignItems: 'center',
  },
  mainTitle: {
    fontSize: 36,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    color: NEON_COLORS.hiCyan,
    textShadowColor: NEON_COLORS.glowCyan,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 15,
    letterSpacing: 8,
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 15,
  },
  tagline: {
    fontSize: 12,
    fontFamily: 'monospace',
    marginBottom: 10,
  },
  version: {
    fontSize: 9,
    fontFamily: 'monospace',
  },
});
