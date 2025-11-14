/**
 * READING HISTORY SCREEN
 * View, manage, and reopen past tarot readings
 * Max 20 readings stored locally
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  SafeAreaView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NeonText, LPMUDText } from '../components/TerminalEffects';
import { NEON_COLORS } from '../styles/cyberpunkColors';

const READINGS_KEY = '@lunatiq_saved_readings';
const MAX_READINGS = 20;

export default function ReadingHistoryScreen({ navigation }) {
  const [readings, setReadings] = useState([]);
  const [editingId, setEditingId] = useState(null);
  const [editingName, setEditingName] = useState('');

  useEffect(() => {
    loadReadings();
  }, []);

  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      loadReadings();
    });
    return unsubscribe;
  }, [navigation]);

  async function loadReadings() {
    try {
      const data = await AsyncStorage.getItem(READINGS_KEY);
      if (data) {
        const parsed = JSON.parse(data);
        // Sort by date, most recent first
        parsed.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        setReadings(parsed);
      }
    } catch (error) {
      console.error('Error loading readings:', error);
    }
  }

  async function saveReadings(updatedReadings) {
    try {
      await AsyncStorage.setItem(READINGS_KEY, JSON.stringify(updatedReadings));
      setReadings(updatedReadings);
    } catch (error) {
      console.error('Error saving readings:', error);
    }
  }

  const handleOpenReading = (reading) => {
    navigation.navigate('Synthesis', {
      synthesis: reading.synthesis,
      cards: reading.cards,
      intention: reading.intention,
      readingType: reading.readingType,
      spreadType: reading.spreadType,
      readingId: reading.id,
      readingName: reading.name,
    });
  };

  const handleRename = (reading) => {
    setEditingId(reading.id);
    setEditingName(reading.name);
  };

  const saveRename = async () => {
    const updatedReadings = readings.map(r =>
      r.id === editingId ? { ...r, name: editingName.trim() || r.name } : r
    );
    await saveReadings(updatedReadings);
    setEditingId(null);
    setEditingName('');
  };

  const handleDelete = (reading) => {
    Alert.alert(
      'Delete Reading',
      `Are you sure you want to delete "${reading.name}"?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            const updatedReadings = readings.filter(r => r.id !== reading.id);
            await saveReadings(updatedReadings);
          },
        },
      ]
    );
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <LPMUDText style={styles.backButtonText}>$HIC$← BACK$NOR$</LPMUDText>
          </TouchableOpacity>
          <LPMUDText style={styles.title}>
            $HIG$PAST READINGS$NOR$
          </LPMUDText>
          <View style={styles.backButton} />
        </View>

        {/* Readings count */}
        <View style={styles.countBar}>
          <NeonText color={NEON_COLORS.dimCyan} style={styles.countText}>
            {readings.length} / {MAX_READINGS} readings saved
          </NeonText>
        </View>

        {/* Readings list */}
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          {readings.length === 0 ? (
            <View style={styles.emptyState}>
              <LPMUDText style={styles.emptyText}>
                $HIR$NO SAVED READINGS$NOR${'\n\n'}
                $DIM$Complete a reading to save it here$NOR$
              </LPMUDText>
            </View>
          ) : (
            readings.map((reading) => (
              <View key={reading.id} style={styles.readingCard}>
                {editingId === reading.id ? (
                  <View style={styles.editContainer}>
                    <TextInput
                      style={styles.editInput}
                      value={editingName}
                      onChangeText={setEditingName}
                      onBlur={saveRename}
                      autoFocus
                      maxLength={50}
                      placeholderTextColor={NEON_COLORS.dimCyan}
                    />
                  </View>
                ) : (
                  <TouchableOpacity onPress={() => handleOpenReading(reading)} style={styles.readingContent}>
                    <LPMUDText style={styles.readingName}>
                      $HIC${'>'} {reading.name}$NOR$
                    </LPMUDText>
                    <View style={styles.readingMeta}>
                      <NeonText color={NEON_COLORS.dimYellow} style={styles.metaText}>
                        {reading.readingType || 'General'} | {reading.spreadType || 'Unknown'}
                      </NeonText>
                      <NeonText color={NEON_COLORS.dimCyan} style={styles.metaText}>
                        {formatDate(reading.timestamp)}
                      </NeonText>
                    </View>
                    {reading.intention && (
                      <NeonText color={NEON_COLORS.dimMagenta} style={styles.intention}>
                        "{reading.intention.substring(0, 60)}{reading.intention.length > 60 ? '...' : ''}"
                      </NeonText>
                    )}
                  </TouchableOpacity>
                )}

                <View style={styles.actions}>
                  {editingId !== reading.id && (
                    <>
                      <TouchableOpacity onPress={() => handleRename(reading)} style={styles.actionButton}>
                        <LPMUDText style={styles.actionText}>$HIY$RENAME$NOR$</LPMUDText>
                      </TouchableOpacity>
                      <TouchableOpacity onPress={() => handleDelete(reading)} style={styles.actionButton}>
                        <LPMUDText style={styles.actionText}>$HIR$DELETE$NOR$</LPMUDText>
                      </TouchableOpacity>
                    </>
                  )}
                </View>
              </View>
            ))
          )}
        </ScrollView>
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 2,
    borderBottomColor: NEON_COLORS.dimCyan,
  },
  backButton: {
    width: 80,
  },
  backButtonText: {
    fontSize: 14,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  countBar: {
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: NEON_COLORS.dimCyan,
    alignItems: 'center',
  },
  countText: {
    fontSize: 12,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 24,
  },
  readingCard: {
    borderWidth: 2,
    borderColor: NEON_COLORS.dimCyan,
    backgroundColor: 'rgba(0, 255, 255, 0.05)',
    padding: 16,
    marginBottom: 16,
    borderRadius: 8,
  },
  readingContent: {
    marginBottom: 12,
  },
  readingName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  readingMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  metaText: {
    fontSize: 12,
  },
  intention: {
    fontSize: 11,
    fontStyle: 'italic',
    marginTop: 6,
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: NEON_COLORS.dimCyan,
    borderRadius: 4,
  },
  actionText: {
    fontSize: 12,
  },
  editContainer: {
    marginBottom: 12,
  },
  editInput: {
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    backgroundColor: 'rgba(0, 255, 255, 0.1)',
    padding: 12,
    fontSize: 16,
    fontFamily: 'monospace',
    color: NEON_COLORS.hiCyan,
  },
});
