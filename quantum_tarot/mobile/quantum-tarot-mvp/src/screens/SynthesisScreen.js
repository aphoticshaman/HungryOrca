/**
 * SYNTHESIS SCREEN - Display mega synthesis with encrypted reveal
 *
 * Shows the 600-1500 word mega synthesis generated from:
 * - All cards + positions
 * - MCQ answers (cognitive dissonance, emotional patterns)
 * - MBTI type + communication style
 * - Advanced astrology (Lilith, Chiron, Nodes)
 * - Pop culture quotes
 * - Balanced Wisdom (Middle Way)
 */

import React, { useRef } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';
import CyberpunkHeader from '../components/CyberpunkHeader';
import { NeonText, LPMUDText } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

export default function SynthesisScreen({ route, navigation }) {
  const {
    synthesis,
    cards,
    intention,
    readingType,
    spreadType
  } = route.params || {};

  const scrollViewRef = useRef(null);

  const handleFinish = () => {
    // Navigate back to welcome screen
    navigation.navigate('Welcome');
  };

  const handleSaveReading = async () => {
    // TODO: Implement reading save to AsyncStorage
    console.log('Save reading functionality to be implemented');
  };

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        {/* Compact header */}
        <CyberpunkHeader compact={true} />

        {/* Title section */}
        <View style={styles.titleSection}>
          <LPMUDText style={styles.title}>
            $HIM${'>'} $HIY$SYNTHESIS$NOR$ $HIM${'<'}$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.subtitle}>
            Your personalized reading
          </NeonText>
          {intention && (
            <View style={styles.intentionBox}>
              <LPMUDText style={styles.intentionText}>
                $HIY$INTENTION:$NOR$ {intention}
              </LPMUDText>
            </View>
          )}
        </View>

        {/* Scrollable synthesis */}
        <ScrollView
          ref={scrollViewRef}
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={true}
        >
          <View style={styles.synthesisContainer}>
            <NeonText color={NEON_COLORS.hiWhite} style={styles.synthesisText}>
              {synthesis}
            </NeonText>
          </View>

          {/* Reading metadata */}
          <View style={styles.metadataBox}>
            <LPMUDText style={styles.metadataText}>
              $HIC$READING DETAILS$NOR${'\n'}
              $NOR$Type: {readingType || 'General'}$NOR${'\n'}
              $NOR$Spread: {spreadType || 'Unknown'}$NOR${'\n'}
              $NOR$Cards: {cards?.length || 0}$NOR$
            </LPMUDText>
          </View>
        </ScrollView>

        {/* Footer buttons */}
        <View style={styles.footer}>
          <TouchableOpacity onPress={handleSaveReading} style={styles.saveButton}>
            <LPMUDText style={styles.saveButtonText}>
              $HIY${'[ '} SAVE $NOR${' ]'}$NOR$
            </LPMUDText>
          </TouchableOpacity>

          <TouchableOpacity onPress={handleFinish} style={styles.finishButton}>
            <LPMUDText style={styles.finishButtonText}>
              $HIG${'[ '} COMPLETE $NOR${' ]'}$NOR$
            </LPMUDText>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  safeArea: {
    flex: 1,
  },
  titleSection: {
    padding: 20,
    paddingBottom: 12,
    borderBottomWidth: 2,
    borderBottomColor: NEON_COLORS.dimCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
  },
  title: {
    fontSize: 24,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 5,
    textAlign: 'center',
    lineHeight: 30,
  },
  subtitle: {
    fontSize: 12,
    fontFamily: 'monospace',
    textAlign: 'center',
    marginBottom: 12,
  },
  intentionBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.dimYellow,
    padding: 10,
    marginTop: 10,
    backgroundColor: 'rgba(10, 10, 0, 0.8)',
  },
  intentionText: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 16,
    textAlign: 'center',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  synthesisContainer: {
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    padding: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    marginBottom: 20,
  },
  synthesisText: {
    fontSize: 14,
    lineHeight: 22,
    fontFamily: 'monospace',
  },
  metadataBox: {
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    padding: 15,
    backgroundColor: 'rgba(0, 10, 10, 0.8)',
    marginTop: 10,
  },
  metadataText: {
    fontSize: 10,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  footer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 2,
    borderTopColor: NEON_COLORS.dimCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
  },
  saveButton: {
    flex: 1,
    padding: 16,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimYellow,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    alignItems: 'center',
  },
  saveButtonText: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  finishButton: {
    flex: 1,
    padding: 16,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiGreen,
    backgroundColor: 'rgba(0, 26, 0, 0.8)',
    alignItems: 'center',
  },
  finishButtonText: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
