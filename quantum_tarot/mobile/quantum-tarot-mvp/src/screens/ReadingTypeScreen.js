/**
 * READING TYPE SCREEN - Select reading category
 */

import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { NeonText, LPMUDText, GlitchText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const READING_TYPES = [
  {
    id: 'career',
    name: 'CAREER',
    color: '$HIC$',
    description: 'Work, purpose, ambition, professional path',
    icon: '▲'
  },
  {
    id: 'romance',
    name: 'ROMANCE',
    color: '$HIM$',
    description: 'Love, relationships, attraction, connection',
    icon: '♥'
  },
  {
    id: 'wellness',
    name: 'WELLNESS',
    color: '$HIG$',
    description: 'Health, energy, balance, self-care',
    icon: '+'
  },
  {
    id: 'finance',
    name: 'FINANCE',
    color: '$HIY$',
    description: 'Money, resources, abundance, security',
    icon: '$'
  },
  {
    id: 'personal_growth',
    name: 'PERSONAL GROWTH',
    color: '$HIW$',
    description: 'Self-development, wisdom, transformation',
    icon: '◆'
  },
  {
    id: 'decision',
    name: 'DECISION',
    color: '$HIR$',
    description: 'Crossroads, choices, paths forward',
    icon: '⚡'
  },
  {
    id: 'general',
    name: 'GENERAL',
    color: '$HIC$',
    description: 'Open inquiry, life overview, guidance',
    icon: '◉'
  },
  {
    id: 'shadow_work',
    name: 'SHADOW WORK',
    color: '$HIM$',
    description: 'Deep dive, trauma, hidden aspects, healing',
    icon: '◢'
  }
];

export default function ReadingTypeScreen({ route, navigation }) {
  const { zodiacSign, birthdate } = route.params;
  const [selected, setSelected] = useState(null);

  const handleSelect = (type) => {
    if (selected === type) {
      // Second tap - confirm and navigate
      navigation.navigate('Intention', { readingType: type, zodiacSign, birthdate });
    } else {
      // First tap - select
      setSelected(type);
    }
  };

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <GlitchText style={styles.headerTitle} glitchChance={0.03}>
            {'>'} SELECT READING TYPE
          </GlitchText>
          <NeonText color={NEON_COLORS.dimYellow} style={styles.headerSubtitle}>
            What realm needs illumination?
          </NeonText>
        </View>

        {/* Reading types grid */}
        <View style={styles.grid}>
          {READING_TYPES.map((type) => (
            <TouchableOpacity
              key={type.id}
              onPress={() => handleSelect(type.id)}
              style={[
                styles.typeCard,
                selected === type.id && styles.typeCardSelected
              ]}
            >
              <View style={styles.typeHeader}>
                <LPMUDText style={styles.typeIcon}>
                  {type.color}{type.icon}$NOR$
                </LPMUDText>
                <LPMUDText style={styles.typeName}>
                  {type.color}{type.name}$NOR$
                </LPMUDText>
              </View>

              <NeonText
                color={NEON_COLORS.dimWhite}
                style={styles.typeDescription}
              >
                {type.description}
              </NeonText>

              {selected === type.id && (
                <View style={styles.selectedIndicator}>
                  <NeonText color={NEON_COLORS.hiCyan} style={styles.selectedText}>
                    {'[ TAP AGAIN TO CONFIRM ]'}
                  </NeonText>
                </View>
              )}
            </TouchableOpacity>
          ))}
        </View>

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
    color: NEON_COLORS.hiCyan,
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 12,
    fontFamily: 'monospace',
  },
  grid: {
    gap: 15,
  },
  typeCard: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    padding: 15,
    backgroundColor: '#000000',
  },
  typeCardSelected: {
    borderColor: NEON_COLORS.hiCyan,
    borderWidth: 3,
  },
  typeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 10,
  },
  typeIcon: {
    fontSize: 20,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  typeName: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  typeDescription: {
    fontSize: 11,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  selectedIndicator: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: NEON_COLORS.hiCyan,
  },
  selectedText: {
    fontSize: 11,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  spacer: {
    height: 40,
  },
});
