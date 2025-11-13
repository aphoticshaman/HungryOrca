/**
 * ERROR BOUNDARY - Catch React errors gracefully
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { NEON_COLORS } from '../styles/cyberpunkColors';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.container}>
          <View style={styles.errorBox}>
            <Text style={styles.errorTitle}>
              {'>'} SYSTEM ERROR {'<'}
            </Text>

            <Text style={styles.errorText}>
              Something broke. Check the console.
            </Text>

            <Text style={styles.errorDetails}>
              {this.state.error?.toString()}
            </Text>

            <TouchableOpacity onPress={this.handleReset} style={styles.resetButton}>
              <Text style={styles.resetButtonText}>
                {'[ RESET ]'}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      );
    }

    return this.props.children;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorBox: {
    borderWidth: 2,
    borderColor: NEON_COLORS.hiRed,
    padding: 20,
    width: '100%',
    maxWidth: 400,
  },
  errorTitle: {
    fontSize: 18,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    color: NEON_COLORS.hiRed,
    marginBottom: 15,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 14,
    fontFamily: 'monospace',
    color: NEON_COLORS.dimWhite,
    marginBottom: 15,
    textAlign: 'center',
  },
  errorDetails: {
    fontSize: 11,
    fontFamily: 'monospace',
    color: NEON_COLORS.dimYellow,
    marginBottom: 20,
  },
  resetButton: {
    padding: 15,
    borderWidth: 2,
    borderColor: NEON_COLORS.hiCyan,
    alignItems: 'center',
  },
  resetButtonText: {
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 'bold',
    color: NEON_COLORS.hiCyan,
  },
});
