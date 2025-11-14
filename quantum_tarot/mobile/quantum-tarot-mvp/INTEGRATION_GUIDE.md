# INTEGRATION GUIDE: MBTI + MCQ + SYNTHESIS FLOW

This guide shows how to integrate the MBTI test, post-card MCQs, and mega synthesis engine into your app's reading flow.

## Overview

The complete reading flow:

```
User Profile Setup
    ↓
MBTI Test (or Vibe Mode skip)
    ↓
Reading Configuration (intention, type, spread)
    ↓
Card Drawing + MCQs (1-3 questions after each card)
    ↓
Mega Synthesis Generation
    ↓
Display Reading
```

## 1. User Profile Setup

First, ensure user has completed basic profile:

```javascript
// Example user profile structure
const userProfile = {
  name: 'Ryan',
  birthday: '1990-03-15',
  zodiacSign: 'Pisces',
  mbtiType: null, // Will be set by MBTI test
  vibeModeEnabled: false,
  pronouns: 'he/him',
};
```

## 2. MBTI Test Integration

### A. Navigate to MBTI Test Screen

```javascript
// In your onboarding or settings screen
import MBTITestScreen from './screens/MBTITestScreen';

// Navigation
navigation.navigate('MBTITest', {
  userProfile: userProfile,
  onComplete: (updatedProfile) => {
    // Save updated profile with MBTI type
    saveUserProfile(updatedProfile);

    // Continue to reading setup
    navigation.navigate('ReadingSetup');
  },
});
```

### B. Vibe Mode (Skip MBTI)

The MBTI screen has built-in vibe mode checkbox:
- User checks "Vibe Mode" → Warning modal appears
- "Yes, proceed anyway" → `onComplete` called with `mbtiType: null, vibeModeEnabled: true`
- "No, complete my profile" → Returns to MBTI test

### C. Handling Results

```javascript
const handleMBTIComplete = (updatedProfile) => {
  if (updatedProfile.mbtiType) {
    // Full profile with MBTI type
    console.log('MBTI Type:', updatedProfile.mbtiType); // e.g., 'INTJ'
    console.log('Scores:', updatedProfile.mbtiScores); // { EI: -12, SN: -8, TF: 10, JP: 8 }
    console.log('Strengths:', updatedProfile.mbtiStrengths);
  } else if (updatedProfile.vibeModeEnabled) {
    // Vibe mode - less personalized readings
    console.log('User opted for vibe mode (less accurate readings)');
  }

  // Save to AsyncStorage
  await AsyncStorage.setItem('userProfile', JSON.stringify(updatedProfile));
};
```

## 3. Reading Configuration

Before card drawing, collect reading parameters:

```javascript
const readingConfig = {
  intention: 'career transition clarity', // User's stated intention
  readingType: 'career', // career, romance, wellness, general, etc.
  spreadType: 'celtic_cross', // celtic_cross, three_card, etc.
  numCards: 10, // Based on spread type
};
```

## 4. Card Drawing with MCQs

### A. Setup MCQ State

```javascript
import { useState } from 'react';
import { generatePostCardQuestions } from './utils/postCardQuestions';
import MCQModal from './components/MCQModal';

const CardDrawingScreen = () => {
  const [drawnCards, setDrawnCards] = useState([]);
  const [mcqAnswers, setMcqAnswers] = useState([]);
  const [showMCQModal, setShowMCQModal] = useState(false);
  const [currentMCQs, setCurrentMCQs] = useState([]);
  const [currentCardIndex, setCurrentCardIndex] = useState(0);

  // ... rest of component
};
```

### B. After Each Card Draw

```javascript
const handleCardDrawn = (card) => {
  // Add card to drawn cards
  const newDrawnCards = [...drawnCards, card];
  setDrawnCards(newDrawnCards);

  // Generate MCQs for this card
  const mcqs = generatePostCardQuestions(
    card, // { cardIndex, reversed, position, positionMeaning }
    readingConfig.intention, // "career transition clarity"
    readingConfig.readingType, // "career"
    newDrawnCards.length, // Card number (1-10)
    readingConfig.numCards, // Total cards (10)
    mcqAnswers // Previous MCQ answers for context
  );

  setCurrentMCQs(mcqs);
  setCurrentCardIndex(newDrawnCards.length - 1);
  setShowMCQModal(true);
};
```

### C. Render MCQ Modal

```javascript
return (
  <View>
    {/* Your card drawing UI */}

    <MCQModal
      visible={showMCQModal}
      questions={currentMCQs}
      cardName={drawnCards[currentCardIndex]?.name || 'Card'}
      cardNumber={currentCardIndex + 1}
      totalCards={readingConfig.numCards}
      onComplete={(answers) => {
        // Store answers for this card
        const newMCQAnswers = [
          ...mcqAnswers,
          ...answers.map(answer => ({
            cardIndex: currentCardIndex,
            ...answer,
          })),
        ];
        setMcqAnswers(newMCQAnswers);
        setShowMCQModal(false);

        // Continue to next card or finish
        if (drawnCards.length < readingConfig.numCards) {
          // Draw next card
          drawNextCard();
        } else {
          // All cards drawn - generate synthesis
          generateSynthesis();
        }
      }}
      onSkip={() => {
        // User skipped MCQs - continue anyway
        setShowMCQModal(false);

        if (drawnCards.length < readingConfig.numCards) {
          drawNextCard();
        } else {
          generateSynthesis();
        }
      }}
    />
  </View>
);
```

### D. MCQ Question Types by Card Position

The `generatePostCardQuestions` function automatically selects appropriate questions:

- **First card (1)**: Resonance + Aspect + Emotion (3 questions)
- **Middle cards (2-9)**: Confirmation + Situation + Action (1-3 questions, varies)
- **Last card (10)**: Takeaway + Readiness (2 questions)

## 5. Mega Synthesis Generation

After all cards drawn and MCQs answered:

```javascript
import { generateMegaSynthesis } from './utils/megaSynthesisEngine';

const generateSynthesis = async () => {
  const synthesis = await generateMegaSynthesis({
    cards: drawnCards, // Array of { cardIndex, reversed, position, positionMeaning }
    mcqAnswers: mcqAnswers, // All MCQ answers
    userProfile: userProfile, // { name, birthday, zodiacSign, mbtiType, ... }
    intention: readingConfig.intention, // "career transition clarity"
    readingType: readingConfig.readingType, // "career"
    spreadType: readingConfig.spreadType, // "celtic_cross"
  });

  // synthesis is now a 600-1500 word markdown string
  console.log(synthesis);

  // Navigate to synthesis display screen
  navigation.navigate('ReadingSynthesis', { synthesis });
};
```

## 6. Synthesis Display

Display the synthesis with markdown rendering:

```javascript
import Markdown from 'react-native-markdown-display';

const SynthesisScreen = ({ route }) => {
  const { synthesis } = route.params;

  return (
    <ScrollView style={styles.container}>
      <Markdown>
        {synthesis}
      </Markdown>
    </ScrollView>
  );
};
```

## 7. Complete Example Flow

```javascript
// App.js or Navigation setup
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import OnboardingScreen from './screens/OnboardingScreen';
import MBTITestScreen from './screens/MBTITestScreen';
import ReadingSetupScreen from './screens/ReadingSetupScreen';
import CardDrawingScreen from './screens/CardDrawingScreen';
import SynthesisScreen from './screens/SynthesisScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Onboarding">
        <Stack.Screen name="Onboarding" component={OnboardingScreen} />
        <Stack.Screen name="MBTITest" component={MBTITestScreen} />
        <Stack.Screen name="ReadingSetup" component={ReadingSetupScreen} />
        <Stack.Screen name="CardDrawing" component={CardDrawingScreen} />
        <Stack.Screen name="Synthesis" component={SynthesisScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

## 8. Data Flow Diagram

```
┌─────────────────┐
│  User Profile   │
│  (AsyncStorage) │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  MBTI Test OR   │
│  Vibe Mode Skip │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Reading Config  │
│ (intention,     │
│  type, spread)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Card Drawing    │◄──┐
│ Loop (1-10)     │   │
└────────┬────────┘   │
         │            │
         ↓            │
┌─────────────────┐   │
│ MCQ Modal       │   │
│ (1-3 questions) │   │
└────────┬────────┘   │
         │            │
         ↓            │
   More cards? ───────┘
         │ No
         ↓
┌─────────────────┐
│ Generate Mega   │
│ Synthesis       │
│ (600-1500 words)│
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Display Reading │
│ (Markdown)      │
└─────────────────┘
```

## 9. Error Handling

```javascript
const generateSynthesis = async () => {
  try {
    const synthesis = await generateMegaSynthesis({
      cards: drawnCards,
      mcqAnswers: mcqAnswers,
      userProfile: userProfile,
      intention: readingConfig.intention,
      readingType: readingConfig.readingType,
      spreadType: readingConfig.spreadType,
    });

    if (!synthesis || synthesis.length < 100) {
      throw new Error('Synthesis generation failed - output too short');
    }

    navigation.navigate('Synthesis', { synthesis });
  } catch (error) {
    console.error('Synthesis generation error:', error);
    Alert.alert(
      'Error Generating Reading',
      'We encountered an issue creating your synthesis. Please try again.',
      [{ text: 'OK' }]
    );
  }
};
```

## 10. Vibe Mode Handling in Synthesis

When user skips MBTI:

```javascript
const synthesis = await generateMegaSynthesis({
  cards: drawnCards,
  mcqAnswers: mcqAnswers,
  userProfile: {
    ...userProfile,
    mbtiType: null, // No MBTI type
    vibeModeEnabled: true,
  },
  // ... rest of config
});

// Synthesis will:
// - Skip MBTI-specific guidance section
// - Use generic psychological language
// - Still incorporate astrology, MCQ insights, balanced wisdom
```

## 11. Skipped MCQs Handling

If user skips MCQs for some/all cards:

```javascript
// Empty MCQ answers array is fine
const synthesis = await generateMegaSynthesis({
  cards: drawnCards,
  mcqAnswers: [], // Empty - user skipped all MCQs
  // ... rest of config
});

// Synthesis will:
// - Skip MCQ-based insights
// - Skip cognitive dissonance detection
// - Focus on card meanings, astrology, MBTI
// - Still be 600+ words but less personalized
```

## 12. Testing Checklist

- [ ] User can complete MBTI test and see results
- [ ] User can enable vibe mode and skip MBTI with warning
- [ ] MCQ modal appears after each card draw
- [ ] MCQ questions vary by card position (first/middle/last)
- [ ] User can skip individual MCQ sets
- [ ] Synthesis generates successfully with full context
- [ ] Synthesis generates with partial context (vibe mode, skipped MCQs)
- [ ] Synthesis displays correctly with markdown formatting
- [ ] Pop culture quotes appear for each card
- [ ] Balanced wisdom (Middle Way) phrases appear throughout
- [ ] MBTI-specific guidance appears (when MBTI type present)
- [ ] Astrology context (Lilith, Chiron, Nodes) weaves in
- [ ] Action steps match user's readiness level from MCQs

## 13. Performance Optimization

```javascript
// Cache astrological calculations
import AsyncStorage from '@react-native-async-storage/async-storage';

const getCachedAstroContext = async (birthday, zodiacSign) => {
  const cacheKey = `astro_${birthday}_${zodiacSign}`;
  const cached = await AsyncStorage.getItem(cacheKey);

  if (cached) {
    const { data, timestamp } = JSON.parse(cached);
    // Cache valid for 24 hours
    if (Date.now() - timestamp < 24 * 60 * 60 * 1000) {
      return data;
    }
  }

  // Calculate fresh
  const astroContext = getFullAstrologicalContext(birthday, zodiacSign);

  // Cache it
  await AsyncStorage.setItem(
    cacheKey,
    JSON.stringify({ data: astroContext, timestamp: Date.now() })
  );

  return astroContext;
};
```

## 14. Analytics Integration

Track key events:

```javascript
import analytics from '@react-native-firebase/analytics';

// MBTI completion
await analytics().logEvent('mbti_completed', {
  mbti_type: updatedProfile.mbtiType,
});

// Vibe mode selection
await analytics().logEvent('vibe_mode_enabled');

// MCQ completion rate
await analytics().logEvent('mcq_completed', {
  card_number: cardIndex + 1,
  questions_answered: answers.length,
});

// Synthesis generation
await analytics().logEvent('synthesis_generated', {
  reading_type: readingType,
  spread_type: spreadType,
  word_count: synthesis.split(' ').length,
  has_mbti: !!userProfile.mbtiType,
  mcq_count: mcqAnswers.length,
});
```

## Next Steps

1. Implement `ReadingSetupScreen` to collect intention/type/spread
2. Update `CardDrawingScreen` to integrate MCQ modal flow
3. Create `SynthesisScreen` with markdown rendering
4. Add navigation flow between screens
5. Test complete end-to-end reading experience
6. Optimize performance (astro caching, synthesis generation)
7. Add analytics tracking

---

**Questions or issues? Check the component source code or PERSONALIZATION_SYSTEM.md for detailed API documentation.**
