/**
 * CYBERPUNK HEADER - Glitchy neon title screen
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { NeonText, LPMUDText, GlitchText, MatrixRain } from './TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function CyberpunkHeader({ showMatrixBg = false, compact = false }) {
  const [colorIndex, setColorIndex] = useState(0);

  const colors = [
    NEON_COLORS.hiCyan,
    NEON_COLORS.hiMagenta,
    NEON_COLORS.hiYellow,
    NEON_COLORS.hiGreen,
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setColorIndex((prev) => (prev + 1) % colors.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <LPMUDText style={styles.compactTitle}>
          $HIC$LunatiQ$NOR$ $HIM$TAROT SYSTEM$NOR$
        </LPMUDText>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {showMatrixBg && (
        <View style={StyleSheet.absoluteFill}>
          <MatrixRain width={SCREEN_WIDTH} height={200} speed={60} />
        </View>
      )}

      <View style={styles.content}>
        {/* Main title with cycling colors and outline */}
        <NeonText
          color={colors[colorIndex]}
          style={[
            styles.mainTitle,
            {
              textShadowColor: colors[colorIndex],
            }
          ]}
        >
          LunatiQ
        </NeonText>

        <NeonText
          color={NEON_COLORS.hiMagenta}
          style={styles.subtitle}
        >
          TAROT SYSTEM
        </NeonText>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    paddingVertical: 40,
    paddingHorizontal: 20,
    backgroundColor: '#000000',
    borderBottomWidth: 2,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  content: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  mainTitle: {
    fontSize: 48,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 20,
    letterSpacing: 4,
    marginBottom: 15,
  },
  subtitle: {
    fontSize: 32,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
    letterSpacing: 12,
    textShadowColor: NEON_COLORS.glowMagenta,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 15,
  },
  compactContainer: {
    width: '100%',
    paddingVertical: 15,
    paddingHorizontal: 20,
    backgroundColor: '#000000',
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
  },
  compactTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
