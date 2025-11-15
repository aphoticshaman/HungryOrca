/**
 * SETTINGS SCREEN - App configuration
 */

import React from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Linking } from 'react-native';
import { NeonText, LPMUDText, FlickerText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

export default function SettingsScreen({ navigation }) {
  const handleClearData = () => {
    // TODO: Implement clear data
    alert('Data cleared (placeholder)');
  };

  const handleResetProfile = () => {
    // TODO: Implement reset
    alert('Profile reset (placeholder)');
  };

  const handleOpenGitHub = () => {
    Linking.openURL('https://github.com/aphoticshaman/HungryOrca');
  };

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIC${'>'} SYSTEM CONFIGURATION$NOR$
          </LPMUDText>
        </View>

        {/* Version info */}
        <View style={styles.section}>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.sectionTitle}>
            {'>'} VERSION INFO
          </NeonText>

          <View style={styles.infoBox}>
            <LPMUDText style={styles.infoText}>
              $HIY$APP VERSION:$NOR$ 1.0.0{'\n'}
              $HIY$EXPO SDK:$NOR$ 54{'\n'}
              $HIY$REACT NATIVE:$NOR$ 0.81{'\n'}
              $HIY$REACT:$NOR$ 19.1.0{'\n\n'}

              $HIC$AGI ENGINE:$NOR$ LunatIQ Tarot v1.0{'\n'}
              $HIC$MODE:$NOR$ Offline (No cloud){'\n'}
              $HIC$RNG:$NOR$ Quantum hardware
            </LPMUDText>
          </View>
        </View>

        {/* Data management */}
        <View style={styles.section}>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.sectionTitle}>
            {'>'} DATA MANAGEMENT
          </NeonText>

          <TouchableOpacity onPress={handleResetProfile} style={styles.actionButton}>
            <NeonText color={NEON_COLORS.hiYellow} style={styles.actionButtonText}>
              {'[ RESET PERSONALITY PROFILE ]'}
            </NeonText>
          </TouchableOpacity>

          <TouchableOpacity onPress={handleClearData} style={styles.actionButton}>
            <NeonText color={NEON_COLORS.hiRed} style={styles.actionButtonText}>
              {'[ CLEAR ALL DATA ]'}
            </NeonText>
          </TouchableOpacity>
        </View>

        {/* Aesthetic settings */}
        <View style={styles.section}>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.sectionTitle}>
            {'>'} AESTHETIC
          </NeonText>

          <View style={styles.infoBox}>
            <LPMUDText style={styles.infoText}>
              $HIG$✓$NOR$ Terminal effects: $HIW$ENABLED$NOR${'\n'}
              $HIG$✓$NOR$ Matrix rain: $HIW$ENABLED$NOR${'\n'}
              $HIG$✓$NOR$ CRT scan lines: $HIW$ENABLED$NOR${'\n'}
              $HIG$✓$NOR$ Neon glows: $HIW$MAX$NOR${'\n'}
              $HIG$✓$NOR$ Glitch effects: $HIW$ENABLED$NOR${'\n\n'}

              $HIM$Theme locked:$NOR$ Cyberpunk terminal{'\n'}
              $HIM$No other options.$NOR$ This is the way.
            </LPMUDText>
          </View>
        </View>

        {/* About */}
        <View style={styles.section}>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.sectionTitle}>
            {'>'} ABOUT
          </NeonText>

          <View style={styles.aboutBox}>
            <LPMUDText style={styles.aboutText}>
              $HIY$QUANTUM TAROT$NOR${'\n'}
              $HIC$Retro Terminal Edition$NOR${'\n\n'}

              Built by $HIM$@aphoticshaman$NOR${'\n'}
              For hackers, by a hacker.{'\n\n'}

              No skeuomorphism.{'\n'}
              No pretty pictures.{'\n'}
              No subscriptions.{'\n'}
              No tracking.{'\n'}
              No bullshit.{'\n\n'}

              Just pure $HIG$terminal aesthetics$NOR$ +{'\n'}
              genuine $HIG$AGI interpretation$NOR$.{'\n\n'}

              $HIY$This is our moat.$NOR$
            </LPMUDText>
          </View>

          <TouchableOpacity onPress={handleOpenGitHub} style={styles.linkButton}>
            <FlickerText color={NEON_COLORS.hiCyan} style={styles.linkButtonText}>
              {'[ VIEW ON GITHUB ]'}
            </FlickerText>
          </TouchableOpacity>
        </View>

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
    marginBottom: 30,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  headerTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    lineHeight: 22,
  },
  section: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 14,
    fontFamily: 'monospace',
    marginBottom: 12,
  },
  infoBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    padding: 15,
    backgroundColor: '#000000',
  },
  infoText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 17,
  },
  actionButton: {
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimYellow,
    marginBottom: 10,
    alignItems: 'center',
  },
  actionButtonText: {
    fontSize: 12,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  aboutBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimMagenta,
    padding: 15,
    marginBottom: 15,
    backgroundColor: '#000000',
  },
  aboutText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 17,
  },
  linkButton: {
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    alignItems: 'center',
  },
  linkButtonText: {
    fontSize: 12,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  backButton: {
    padding: 15,
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
    marginTop: 20,
  },
  backButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
  spacer: {
    height: 40,
  },
});
