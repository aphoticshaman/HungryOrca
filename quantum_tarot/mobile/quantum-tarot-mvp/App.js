import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';
import { ThemeProvider } from './src/context/ThemeContext';
import { ErrorBoundary } from './src/components/ErrorBoundary';

// Screens
import WelcomeScreen from './src/screens/WelcomeScreen';
import OnboardingScreen from './src/screens/OnboardingScreen';
import ProfileSetupScreen from './src/screens/ProfileSetupScreen';
import ProfileSelectScreen from './src/screens/ProfileSelectScreen';
import ReadingTypeScreen from './src/screens/ReadingTypeScreen';
import BirthdayScreen from './src/screens/BirthdayScreen';
import PersonalityQuestionsScreen from './src/screens/PersonalityQuestionsScreen';
import IntentionScreen from './src/screens/IntentionScreen';
import CardDrawingScreen from './src/screens/CardDrawingScreen';
import ReadingScreen from './src/screens/ReadingScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Stack = createStackNavigator();

export default function App() {
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
            <Stack.Screen name="ReadingType" component={ReadingTypeScreen} />
            <Stack.Screen name="Birthday" component={BirthdayScreen} />
            <Stack.Screen name="Intention" component={IntentionScreen} />
            <Stack.Screen name="CardDrawing" component={CardDrawingScreen} />
            <Stack.Screen name="Reading" component={ReadingScreen} />
            <Stack.Screen name="Settings" component={SettingsScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
