/**
 * PROFILE SELECT SCREEN - Choose from saved profiles
 */

import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NeonText, LPMUDText, ScanLines } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const PROFILES_KEY = '@lunatiq_profiles';
const ACTIVE_PROFILE_KEY = '@lunatiq_active_profile';

export default function ProfileSelectScreen({ navigation }) {
  const [profiles, setProfiles] = useState([]);
  const [activeProfileId, setActiveProfileId] = useState(null);

  useEffect(() => {
    loadProfiles();
  }, []);

  async function loadProfiles() {
    try {
      const profilesData = await AsyncStorage.getItem(PROFILES_KEY);
      const activeId = await AsyncStorage.getItem(ACTIVE_PROFILE_KEY);

      if (profilesData) {
        setProfiles(JSON.parse(profilesData));
      }

      if (activeId) {
        setActiveProfileId(activeId);
      }
    } catch (error) {
      console.error('Error loading profiles:', error);
    }
  }

  const handleSelectProfile = async (profile) => {
    try {
      await AsyncStorage.setItem(ACTIVE_PROFILE_KEY, profile.id);
      // Navigate back to welcome
      navigation.navigate('Welcome');
    } catch (error) {
      console.error('Error selecting profile:', error);
    }
  };

  const handleDeleteProfile = (profile) => {
    Alert.alert(
      'Delete Profile',
      `Delete ${profile.name}? This cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const updatedProfiles = profiles.filter(p => p.id !== profile.id);
              await AsyncStorage.setItem(PROFILES_KEY, JSON.stringify(updatedProfiles));

              // If deleted active profile, clear it
              if (activeProfileId === profile.id) {
                await AsyncStorage.removeItem(ACTIVE_PROFILE_KEY);
              }

              setProfiles(updatedProfiles);
            } catch (error) {
              console.error('Error deleting profile:', error);
            }
          }
        }
      ]
    );
  };

  return (
    <View style={styles.container}>
      <ScanLines />

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <LPMUDText style={styles.headerTitle}>
            $HIY${'>'} SELECT PROFILE$NOR$
          </LPMUDText>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.subtitle}>
            {profiles.length} profile(s) available
          </NeonText>
        </View>

        {profiles.length === 0 ? (
          <View style={styles.emptyBox}>
            <NeonText color={NEON_COLORS.dimRed} style={styles.emptyText}>
              No profiles found
            </NeonText>
          </View>
        ) : (
          profiles.map((profile) => {
            const isActive = profile.id === activeProfileId;

            return (
              <View key={profile.id} style={styles.profileCard}>
                <TouchableOpacity
                  onPress={() => handleSelectProfile(profile)}
                  style={[
                    styles.profileButton,
                    isActive && styles.profileButtonActive
                  ]}
                >
                  <View style={styles.profileHeader}>
                    <LPMUDText style={styles.profileName}>
                      {isActive ? '$HIC$' : '$HIW$'}{profile.name}$NOR$
                    </LPMUDText>
                    {isActive && (
                      <NeonText color={NEON_COLORS.hiGreen} style={styles.activeBadge}>
                        ACTIVE
                      </NeonText>
                    )}
                  </View>

                  <View style={styles.profileDetails}>
                    <LPMUDText style={styles.profileDetail}>
                      $HIM${'>'} {profile.zodiacSign}$NOR$
                    </LPMUDText>
                    <LPMUDText style={styles.profileDetail}>
                      $HIY${'>'} {profile.birthdate}$NOR$
                    </LPMUDText>
                  </View>
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={() => handleDeleteProfile(profile)}
                  style={styles.deleteButton}
                >
                  <NeonText color={NEON_COLORS.dimRed} style={styles.deleteText}>
                    DELETE
                  </NeonText>
                </TouchableOpacity>
              </View>
            );
          })
        )}

        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        >
          <NeonText color={NEON_COLORS.dimCyan} style={styles.backButtonText}>
            {'[ ← BACK ]'}
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
  },
  header: {
    marginBottom: 30,
    borderBottomWidth: 2,
    borderBottomColor: NEON_COLORS.dimCyan,
    paddingBottom: 15,
  },
  headerTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    marginBottom: 8,
    lineHeight: 22,
  },
  subtitle: {
    fontSize: 11,
    fontFamily: 'monospace',
  },
  emptyBox: {
    padding: 30,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimRed,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 14,
    fontFamily: 'monospace',
  },
  profileCard: {
    marginBottom: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
  },
  profileButton: {
    padding: 15,
    backgroundColor: '#000000',
  },
  profileButtonActive: {
    borderLeftWidth: 4,
    borderLeftColor: NEON_COLORS.hiGreen,
    backgroundColor: 'rgba(0, 255, 0, 0.05)',
  },
  profileHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  profileName: {
    fontSize: 16,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    lineHeight: 20,
  },
  activeBadge: {
    fontSize: 10,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  profileDetails: {
    gap: 5,
  },
  profileDetail: {
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  deleteButton: {
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: NEON_COLORS.dimRed,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 0, 0, 0.05)',
  },
  deleteText: {
    fontSize: 11,
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
});
