import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';
import { ThemeProvider } from './src/context/ThemeContext';
import { ErrorBoundary } from './src/components/ErrorBoundary';
import AdManager from './src/utils/AdManager';

// Screens
import WelcomeScreen from './src/screens/WelcomeScreen';
import OnboardingScreen from './src/screens/OnboardingScreen';
import ProfileSetupScreen from './src/screens/ProfileSetupScreen';
import ProfileSelectScreen from './src/screens/ProfileSelectScreen';
import ReadingTypeScreen from './src/screens/ReadingTypeScreen';
import BirthdayScreen from './src/screens/BirthdayScreen';
import PersonalityQuestionsScreen from './src/screens/PersonalityQuestionsScreen';
import MBTITestScreen from './src/screens/MBTITestScreen';
import IntentionScreen from './src/screens/IntentionScreen';
import CardDrawingScreen from './src/screens/CardDrawingScreen';
import CardInterpretationScreen from './src/screens/CardInterpretationScreen';
import ReadingScreen from './src/screens/ReadingScreen';
import SynthesisScreen from './src/screens/SynthesisScreen';
import ReadingHistoryScreen from './src/screens/ReadingHistoryScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Stack = createStackNavigator();

export default function App() {
  // Initialize AdMob on app startup
  useEffect(() => {
    console.log('[App] Initializing AdManager...');
    AdManager.initialize();
  }, []);

  return (
    <ErrorBoundary>
      <ThemeProvider>
        <StatusBar style="light" />
        <NavigationContainer>
          <Stack.Navigator
            initialRouteName="Welcome"
            screenOptions={{
              headerShown: false,
              cardStyle: { backgroundColor: '#000000' },
              animationEnabled: true,
              gestureEnabled: true
            }}
          >
            <Stack.Screen name="Welcome" component={WelcomeScreen} />
            <Stack.Screen name="Onboarding" component={OnboardingScreen} />
            <Stack.Screen name="ProfileSetup" component={ProfileSetupScreen} />
            <Stack.Screen name="ProfileSelect" component={ProfileSelectScreen} />
            <Stack.Screen name="PersonalityQuestions" component={PersonalityQuestionsScreen} />
            <Stack.Screen name="MBTITest" component={MBTITestScreen} />
            <Stack.Screen name="ReadingType" component={ReadingTypeScreen} />
            <Stack.Screen name="Birthday" component={BirthdayScreen} />
            <Stack.Screen name="Intention" component={IntentionScreen} />
            <Stack.Screen name="CardDrawing" component={CardDrawingScreen} />
            <Stack.Screen name="CardInterpretation" component={CardInterpretationScreen} />
            <Stack.Screen name="Reading" component={ReadingScreen} />
            <Stack.Screen name="Synthesis" component={SynthesisScreen} />
            <Stack.Screen name="ReadingHistory" component={ReadingHistoryScreen} />
            <Stack.Screen name="Settings" component={SettingsScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
