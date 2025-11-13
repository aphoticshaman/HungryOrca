/**
 * WELCOME SCREEN - Main menu with profile management
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import CyberpunkHeader from '../components/CyberpunkHeader';
import { NeonText, LPMUDText, MatrixRain } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const PROFILES_KEY = '@lunatiq_profiles';
const ACTIVE_PROFILE_KEY = '@lunatiq_active_profile';

export default function WelcomeScreen({ navigation }) {
  const [activeProfile, setActiveProfile] = useState(null);
  const [profiles, setProfiles] = useState([]);

  useEffect(() => {
    loadProfiles();
  }, []);

  // Reload profiles when screen comes into focus
  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      loadProfiles();
    });
    return unsubscribe;
  }, [navigation]);

  async function loadProfiles() {
    try {
      const profilesData = await AsyncStorage.getItem(PROFILES_KEY);
      const activeId = await AsyncStorage.getItem(ACTIVE_PROFILE_KEY);

      if (profilesData) {
        const parsed = JSON.parse(profilesData);
        setProfiles(parsed);

        if (activeId) {
          const active = parsed.find(p => p.id === activeId);
          setActiveProfile(active);
        }
      }
    } catch (error) {
      console.error('Error loading profiles:', error);
    }
  }

  const handleNewReading = () => {
    if (!activeProfile) {
      // No profile selected - prompt to create one
      navigation.navigate('ProfileSetup', { isNewProfile: true });
    } else {
      // Go directly to reading type selection
      navigation.navigate('ReadingType', {
        zodiacSign: activeProfile.zodiacSign,
        birthdate: activeProfile.birthdate,
        personalityProfile: activeProfile.personality
      });
    }
  };

  const handleNewProfile = () => {
    navigation.navigate('ProfileSetup', { isNewProfile: true });
  };

  const handleChooseProfile = () => {
    navigation.navigate('ProfileSelect', { profiles });
  };

  return (
    <View style={styles.container}>
      {/* Matrix rain background */}
      <View style={StyleSheet.absoluteFill}>
        <MatrixRain width={SCREEN_WIDTH} height={SCREEN_HEIGHT} speed={30} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        {/* Animated header */}
        <CyberpunkHeader />

        {/* Main menu buttons */}
        <View style={styles.menuContainer}>
          <TouchableOpacity onPress={handleNewReading} style={styles.menuButton}>
            <LPMUDText style={styles.menuButtonText}>
              $HIC${'[ '} $HIW$NEW READING$NOR$ $HIC${' ]'}$NOR$
            </LPMUDText>
            <NeonText color={NEON_COLORS.dimCyan} style={styles.menuButtonSubtext}>
              {'>'} Start a tarot reading
            </NeonText>
          </TouchableOpacity>

          <TouchableOpacity onPress={handleNewProfile} style={styles.menuButton}>
            <LPMUDText style={styles.menuButtonText}>
              $HIM${'[ '} $HIY$NEW PROFILE$NOR$ $HIM${' ]'}$NOR$
            </LPMUDText>
            <NeonText color={NEON_COLORS.dimYellow} style={styles.menuButtonSubtext}>
              {'>'} Create personality profile
            </NeonText>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={handleChooseProfile}
            style={[styles.menuButton, profiles.length === 0 && styles.menuButtonDisabled]}
            disabled={profiles.length === 0}
          >
            <LPMUDText style={styles.menuButtonText}>
              $HIY${'[ '} $HIW$CHOOSE PROFILE$NOR$ $HIY${' ]'}$NOR$
            </LPMUDText>
            <NeonText
              color={profiles.length === 0 ? NEON_COLORS.dimRed : NEON_COLORS.dimCyan}
              style={styles.menuButtonSubtext}
            >
              {'>'} {profiles.length === 0 ? 'No profiles yet' : `${profiles.length} profile(s) available`}
            </NeonText>
          </TouchableOpacity>
        </View>

        <View style={styles.spacer} />
      </ScrollView>

      {/* Active profile display - bottom */}
      <View style={styles.profileBar}>
        <LPMUDText style={styles.profileText}>
          $HIY$ACTIVE PROFILE:$NOR${' '}
          {activeProfile ? (
            `$HIC$${activeProfile.name}$NOR$ | $HIM$${activeProfile.zodiacSign}$NOR$`
          ) : (
            '$HIR$NONE$NOR$'
          )}
        </LPMUDText>
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
    flexGrow: 1,
    paddingBottom: 40,
  },
  menuContainer: {
    marginHorizontal: 20,
    marginTop: 40,
    gap: 20,
  },
  menuButton: {
    padding: 20,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    backgroundColor: 'rgba(0, 255, 255, 0.05)',
  },
  menuButtonDisabled: {
    borderColor: NEON_COLORS.dimCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    opacity: 0.5,
  },
  menuButtonText: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
    lineHeight: 24,
  },
  menuButtonSubtext: {
    fontSize: 11,
    fontFamily: 'monospace',
    textAlign: 'center',
  },
  spacer: {
    flex: 1,
    minHeight: 40,
  },
  profileBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 15,
    borderTopWidth: 2,
    borderTopColor: NEON_COLORS.dimCyan,
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
  },
  profileText: {
    fontSize: 11,
    fontFamily: 'monospace',
    textAlign: 'center',
    lineHeight: 16,
  },
});
