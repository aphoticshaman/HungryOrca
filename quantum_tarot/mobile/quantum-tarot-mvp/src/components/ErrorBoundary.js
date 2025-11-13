import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

/**
 * Error Boundary - Catches crashes and shows recovery UI
 */
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App crashed:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    // Reset to welcome screen
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.container}>
          <Text style={styles.title}>✧ QUANTUM DISRUPTION ✧</Text>
          <Text style={styles.message}>
            The quantum field collapsed unexpectedly.
          </Text>
          <Text style={styles.error}>
            {this.state.error?.message || 'Unknown error'}
          </Text>
          <TouchableOpacity style={styles.button} onPress={this.handleReset}>
            <Text style={styles.buttonText}>RESTART</Text>
          </TouchableOpacity>
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
    padding: 20
  },
  title: {
    fontFamily: 'monospace',
    fontSize: 16,
    color: '#00FF41',
    marginBottom: 20,
    textAlign: 'center'
  },
  message: {
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#00FF41',
    marginBottom: 30,
    textAlign: 'center'
  },
  error: {
    fontFamily: 'monospace',
    fontSize: 10,
    color: '#FF4444',
    marginBottom: 30,
    textAlign: 'center'
  },
  button: {
    borderWidth: 2,
    borderColor: '#00FF41',
    paddingVertical: 15,
    paddingHorizontal: 40
  },
  buttonText: {
    fontFamily: 'monospace',
    fontSize: 14,
    color: '#00FF41'
  }
});
