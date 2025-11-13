# Quantum Tarot - Spread Layout Design System

## Philosophy

Different spread types require **different UI paradigms**:

1. **Single/3-card spreads** → Simple linear or triangular layouts
2. **Celtic Cross (10 cards)** → Complex spatial layout with overlapping cards
3. **Goal Progress/Decision spreads** → Timeline or decision-tree layouts
4. **Clairvoyant/Predictive** → Past→Present→Future flow with branching

Each spread needs its position meanings, spatial coordinates, and interaction patterns.

---

## Spread Type Categories

### **LINEAR SPREADS** (Simple Timeline)
- **UI Pattern**: Horizontal scroll or vertical stack
- **Interaction**: Swipe through cards sequentially
- **Examples**: Daily Draw, Past-Present-Future, Goal Progress

### **SPATIAL SPREADS** (2D Positioning)
- **UI Pattern**: Zoomable 2D canvas with positioned cards
- **Interaction**: Tap card to focus, pinch to zoom out
- **Examples**: Celtic Cross, Relationship, Horseshoe

### **DECISION TREE SPREADS** (Branching)
- **UI Pattern**: Vertical flow with branching paths
- **Interaction**: Explore alternate outcomes
- **Examples**: Decision Analysis, Fork in the Road, Clairvoyant Predictive

### **MATRIX SPREADS** (Grid Layout)
- **UI Pattern**: Grid with categorical positions
- **Interaction**: Navigate by category or swipe
- **Examples**: Year Ahead (12 cards), Chakra Spread (7 cards)

---

## Detailed Spread Definitions

### **1. SINGLE CARD DRAW**
**Use cases**: Daily guidance, quick answer, focus point
**Card count**: 1
**Layout**: Centered, large

```
Position Meanings:
- Focus: "What you need to know right now"
```

**UI Design:**
- Full-screen centered card
- Large ASCII art or image
- Interpretation below
- No navigation (single panel)

---

### **2. THREE-CARD: PAST-PRESENT-FUTURE**
**Use cases**: Timeline reading, situation analysis, general guidance
**Card count**: 3
**Layout**: Horizontal row

```
Layout:
┌─────┐   ┌─────┐   ┌─────┐
│PAST │ → │PRES │ → │FUTR │
└─────┘   └─────┘   └─────┘

Position Meanings:
- Past: "What has led to this moment"
- Present: "Current situation and energies"
- Future: "Where things are heading"
```

**UI Design:**
- Horizontal scroll/swipe between 3 panels
- Dots indicator at bottom showing position (• ○ ○)
- Swipe left/right to navigate
- Each panel: Card + position + interpretation
- Arrow hints: "← Past | Future →"

---

### **3. THREE-CARD: GOAL PROGRESS**
**Use cases**: Tracking progress toward a specific goal
**Card count**: 3
**Layout**: Vertical ascending stairs

```
Layout:
         ┌─────┐
         │ GOAL│  ← Where you're heading
         └─────┘
    ┌─────┐
    │PROGR│       ← Current progress
    └─────┘
┌─────┐
│START│           ← Starting point
└─────┘

Position Meanings:
- Starting Point: "Where you began, foundation"
- Current Progress: "Where you are now, momentum"
- Goal Manifestation: "Likely outcome, what manifests"
```

**UI Design:**
- Vertical scroll (scroll up to see goal)
- Visual ascending staircase effect
- Progress indicator on side (33% → 66% → 100%)
- Bottom-to-top reading (start at bottom)

---

### **4. THREE-CARD: DECISION ANALYSIS**
**Use cases**: Weighing options, making choices
**Card count**: 3
**Layout**: Fork design

```
Layout:
      ┌─────┐
      │OPTN1│ ← Path A outcome
      └─────┘
         ╱
┌─────┐╱
│CRUX ├
└─────┘╲
         ╲
      ┌─────┐
      │OPTN2│ ← Path B outcome
      └─────┘

Position Meanings:
- Crux: "The core decision point, what matters most"
- Option A: "Outcome if you choose Path A"
- Option B: "Outcome if you choose Path B"
```

**UI Design:**
- Start with center card (Crux) in focus
- Tap "Explore Option A" button → transition to option A card
- Tap "Explore Option B" button → transition to option B card
- Back button returns to crux
- Visual branching arrows

---

### **5. THREE-CARD: DAILY CHECK-IN**
**Use cases**: Morning guidance, daily reflection
**Card count**: 3
**Layout**: Simple row with specific context

```
Layout:
┌─────┐   ┌─────┐   ┌─────┐
│FOCUS│   │AVOID│   │GIFT │
└─────┘   └─────┘   └─────┘

Position Meanings:
- Focus: "What deserves your attention today"
- Avoid: "What to be cautious of or release"
- Gift: "Hidden opportunity or blessing available"
```

**UI Design:**
- Same horizontal swipe as Past-Present-Future
- Different position labels and colors
- Focus (green accent), Avoid (red accent), Gift (gold accent)

---

### **6. CLAIRVOYANT PREDICTIVE (3-CARD)**
**Use cases**: "If I do X, what happens?" - forecasting a decision
**Card count**: 3
**Layout**: Vertical flow with temporal markers

```
Layout:
┌─────────────┐
│  NOW/CHOICE │ ← The decision you're considering
└─────────────┘
       ↓
┌─────────────┐
│ NEAR FUTURE │ ← Immediate consequences (days/weeks)
└─────────────┘
       ↓
┌─────────────┐
│ FAR OUTCOME │ ← Ultimate result (months/year)
└─────────────┘

Position Meanings:
- Now/Choice: "The energy around your intended action"
- Near Future: "Immediate ripples and reactions"
- Far Outcome: "Where this path ultimately leads"
```

**UI Design:**
- Vertical scroll with temporal flow
- Time markers on left side: "Now" → "2-4 weeks" → "3-6 months"
- Animated arrow transitions between cards
- Warning banner: "Remember: you always have free will"

---

### **7. SIX-CARD: RELATIONSHIP SPREAD**
**Use cases**: Romantic relationships, partnerships, interpersonal dynamics
**Card count**: 6
**Layout**: Two columns with bridge

```
Layout:
┌─────┐     ┌─────┐
│ YOU │ ←→  │THEM │  Row 1: The individuals
└─────┘     └─────┘
       ┌─────┐
       │CNXN │       Row 2: The bond (centered)
       └─────┘
┌─────┐     ┌─────┐
│CHALL│     │ADVC │  Row 3: Challenge & guidance
└─────┘     └─────┘
       ┌─────┐
       │OUTC │       Row 4: Where it's going (centered)
       └─────┘

Position Meanings:
- You: "Your energy, what you bring"
- Them: "Their energy, what they bring"
- Connection: "The bond between you, relationship essence"
- Challenge: "What you're working through together"
- Advice: "How to navigate this connection"
- Outcome: "Where the relationship is heading"
```

**UI Design:**
- Paginated 2D layout (swipe or scroll)
- Row 1: Horizontal swipe between You ↔ Them
- Tap "View Connection" → Navigate to centered Connection card
- Row 3: Horizontal swipe between Challenge ↔ Advice
- Tap "View Outcome" → Navigate to Outcome card
- Mini-map indicator showing current position in spread

---

### **8. TEN-CARD: CELTIC CROSS** (THE BIG ONE)
**Use cases**: Comprehensive life situation, deep exploration, major decisions
**Card count**: 10
**Layout**: Traditional Celtic Cross pattern

```
Layout (Spatial Coordinates):

              [4]              Row -2 (top)
               ↑

    [10]     [5][1][2]    [6]  Row 0 (center)
              ↓
              [3]              Row 2 (bottom)

              ↓

          [7][8][9]            Row 4 (staff)

Detailed positions:
[1] Present/Heart of matter (center)
[2] Crossing/Challenge (overlaps [1], horizontal)
[3] Below/Foundation (below [1])
[4] Above/Conscious goal (above [1])
[5] Past/Recent past (left of [1])
[6] Future/Near future (right of [1])
[7] Self/How you see yourself (bottom staff)
[8] Environment/How others see you (staff)
[9] Hopes & Fears (staff)
[10] Outcome/Final result (top staff)

Position Meanings:
1. Present Situation: "The heart of the matter, current state"
2. Challenge/Crossing: "What opposes or complicates (may help or hinder)"
3. Foundation: "Root cause, subconscious influence, distant past"
4. Recent Past: "Events leading here, what's leaving"
5. Crowning: "Conscious goals, best possible outcome, highest aspiration"
6. Near Future: "Coming soon, next 1-3 months"
7. Self Perception: "How you see yourself in this"
8. External Influences: "How others see you, environmental factors"
9. Hopes & Fears: "Secret wishes and anxieties (often contradictory)"
10. Final Outcome: "Where all energies are leading, ultimate result"
```

**UI Design - MULTI-MODE:**

**Mode 1: Overview (Zoomed Out)**
- Show all 10 cards as small thumbnails in spatial arrangement
- Cross in center (cards 1-6) + Staff on right (cards 7-10)
- Pinch to zoom in on specific card
- Tap card to enter detail view

**Mode 2: Sectional Navigation**
- **Section 1: The Cross (cards 1-6)** - Main situation
  - Swipe through or use 2D pan gesture
  - Highlight relationships: "2 crosses 1", "3 supports 1", etc.

- **Section 2: The Staff (cards 7-10)** - Personal journey
  - Vertical scroll through staff
  - Shows internal → external → outcome progression

**Mode 3: Card Detail View**
- Full screen single card
- Position label + card image + interpretation
- Navigation arrows showing related cards:
  - From card 1: "See Challenge (2) | See Foundation (3) | See Goal (4)"
  - From card 10: "Review Hopes/Fears (9) | View Synthesis"

**Mode 4: Synthesis View**
- Text summary combining all 10 cards
- LunatiQ AGI generates holistic interpretation
- References card relationships:
  - "The 3 of Swords crossing your Ten of Pentacles suggests..."
  - "Your foundation (Death reversed) connects to your hopes (Star) by..."

**Navigation Widget** (always visible):
```
[Cross View] [Staff View] [Card 1] [Card 2] ... [Card 10] [Synthesis]
```

---

### **9. SEVEN-CARD: HORSESHOE SPREAD**
**Use cases**: Exploring multiple facets of a situation
**Card count**: 7
**Layout**: Horseshoe arc

```
Layout:
[1]                     [7]    ← Outcome
  [2]               [6]        ← Influences
    [3]         [5]            ← Obstacles/Advice
        [4]                    ← Present (center)

Position Meanings:
1. Past: "What's behind you"
2. Present: "Where you are now"
3. Hidden Influences: "What you can't see"
4. Obstacles: "What stands in the way"
5. Environment: "External factors"
6. Advice: "What action to take"
7. Outcome: "Where this leads"
```

**UI Design:**
- Curved carousel that follows horseshoe arc
- Swipe along curve
- Current card centered and enlarged
- Other cards visible but dimmed
- Visual arc line connecting all positions

---

### **10. TWELVE-CARD: YEAR AHEAD**
**Use cases**: Annual forecast, birthday readings, new year
**Card count**: 12
**Layout**: Clock face or 4×3 grid

```
Layout Option A (Clock):
         [12][1][2]
      [11]    •    [3]
      [10]    •    [4]
         [9][8][7][6][5]

Layout Option B (Grid):
[Jan] [Feb] [Mar] [Apr]
[May] [Jun] [Jul] [Aug]
[Sep] [Oct] [Nov] [Dec]

Position Meanings:
- Each card: "Themes and energy for [Month]"
```

**UI Design:**
- Grid view showing all 12 cards as small thumbnails
- Tap any month to see full interpretation
- "Current month" highlighted with accent color
- Swipe left/right to navigate months in order
- Option to view quarter summary (Q1: Jan-Mar, etc.)

---

## UI Components Needed

### **1. SpreadLayoutEngine** (New Service)
```javascript
class SpreadLayoutEngine {
  getLayoutForSpread(spreadType) {
    // Returns spatial coordinates, interaction mode, etc.
  }

  getCardPosition(spreadType, cardIndex) {
    // Returns {x, y, z-index, rotation} for 2D positioning
  }

  getNavigationMode(spreadType) {
    // Returns: 'horizontal_swipe' | 'vertical_scroll' | '2d_pan' | 'paginated'
  }
}
```

### **2. CardPanel Component**
- Displays single card in detail
- Shows: position label, card image/ASCII, name, reversed indicator, interpretation
- Navigation hints for related cards
- Quantum signature in corner

### **3. SpreadNavigator Component**
- Mini-map showing current position in spread
- Quick jump to specific cards
- Progress indicator
- Mode switcher (for Celtic Cross: Overview | Sectional | Detail | Synthesis)

### **4. SpatialSpreadView Component**
- 2D canvas for complex spreads (Celtic Cross, Horseshoe)
- Pan/zoom gestures
- Tap to focus card
- Visual connection lines between related positions

### **5. SynthesisView Component**
- LunatiQ-generated holistic interpretation
- References multiple cards in natural language
- Highlight card names → tap to jump to that card
- "Overall theme", "Key challenges", "Guidance summary"

---

## Implementation Priority

**Phase 1: Enhanced 3-Card Layouts** (Quick wins)
- ✅ Past-Present-Future (horizontal swipe)
- ✅ Goal Progress (vertical stairs)
- ✅ Decision Analysis (fork/branch)
- ✅ Daily Check-In (horizontal with colored accents)
- ✅ Clairvoyant Predictive (vertical timeline)

**Phase 2: 6-Card Relationship Spread**
- Multi-row paginated layout
- Row navigation
- Mini-map

**Phase 3: 10-Card Celtic Cross** (Complex)
- Spatial positioning system
- Multi-mode UI (overview, sectional, detail, synthesis)
- Card relationship visualization
- Pan/zoom gestures

**Phase 4: Specialty Spreads**
- Horseshoe (7 cards)
- Year Ahead (12 cards)
- Custom user spreads

---

## Data Structure Updates

### **Spread Definition Format**
```javascript
{
  type: 'celtic_cross',
  name: 'Celtic Cross',
  cardCount: 10,
  category: 'spatial',  // linear | spatial | decision_tree | matrix
  navigationMode: '2d_pan',

  positions: [
    {
      index: 0,
      name: 'Present Situation',
      meaning: 'The heart of the matter, current state',
      coordinates: { x: 0.5, y: 0.5, rotation: 0 },  // Normalized 0-1
      zIndex: 2,
      relatedPositions: [1, 2, 3, 4]  // Card 1 relates to 2,3,4,5
    },
    {
      index: 1,
      name: 'Challenge',
      meaning: 'What opposes or complicates',
      coordinates: { x: 0.5, y: 0.5, rotation: 90 },  // Crosses card 1
      zIndex: 3,  // Higher z-index to appear "on top"
      relatedPositions: [0]
    },
    // ... 8 more positions
  ],

  sections: [
    {
      name: 'The Cross',
      cardIndices: [0, 1, 2, 3, 4, 5],
      description: 'The situation itself'
    },
    {
      name: 'The Staff',
      cardIndices: [6, 7, 8, 9],
      description: 'Your journey through it'
    }
  ],

  viewModes: ['overview', 'sectional', 'detail', 'synthesis']
}
```

---

## Visual Design Language

### **Color Coding by Position Type**
- **Past**: Muted purple/blue (fading)
- **Present**: Bright accent color (active)
- **Future**: Green/cyan (emerging)
- **Challenge**: Red/orange (alert)
- **Advice**: Gold/yellow (wisdom)
- **Outcome**: White/bright (culmination)

### **Typography**
- Position labels: Bold monospace, uppercase
- Card names: Regular monospace
- Interpretations: Smaller monospace, line-wrapped
- Quantum signatures: Tiny monospace, very dim

### **Transitions**
- Card flips: 3D rotation on Y-axis (reveal animation)
- Swipe navigation: Slide with parallax
- Zoom: Smooth scale transform
- Mode switching: Fade + scale

### **Spatial Indicators**
- Dotted lines connecting related cards
- Arrow hints: "← Past | Future →"
- Breadcrumbs: "Cross > Card 1 > Detail"
- Progress: "Card 3 of 10" or "● ○ ○"

---

## Edge Cases & Considerations

1. **Screen size limitations**:
   - Celtic Cross might not fit on small phones in overview mode
   - Solution: Allow zooming, default to sectional view on small screens

2. **Interpretation length**:
   - Long interpretations might make cards too tall
   - Solution: Collapsible text, "Read more" expansion

3. **Reversed cards**:
   - Visual indicator (upside down or marker)
   - Interpretation adjusts accordingly

4. **Saving/Sharing**:
   - Export spread as image (all cards arranged)
   - Save reading to history with layout preserved
   - Share specific cards or full spread

5. **Offline support**:
   - All layouts work offline
   - Interpretations cached

6. **Accessibility**:
   - Screen reader support for position meanings
   - High contrast mode
   - Larger text options

---

## Example: Complete Celtic Cross Flow

**User Journey:**

1. **Intent Screen** → Select "Celtic Cross" spread
2. **Drawing Screen** → "Drawing 10 cards from quantum field..."
3. **Reading Screen - Overview Mode**:
   - See all 10 cards in spatial layout
   - Pinch to zoom to specific area
   - Tap card 1 (Present) to enter detail

4. **Card Detail View**:
   - Full screen: Card 1 image + "PRESENT SITUATION" + interpretation
   - Buttons: [See Challenge →] [See Foundation ↓] [See Past ←]
   - Tap [See Challenge]

5. **Card Detail View** (Card 2):
   - Full screen: Card 2 crossed over Card 1
   - "This card CROSSES your present situation, showing what complicates matters"
   - Navigation: [Back to Present] [See Goal ↑]

6. **Section Navigation**:
   - Bottom nav: [The Cross] [The Staff] [Synthesis]
   - Tap [The Staff]

7. **Staff Section View**:
   - Vertical scroll through cards 7→8→9→10
   - Each card visible in sequence
   - "Your internal journey through this situation"

8. **Synthesis View**:
   - LunatiQ holistic interpretation
   - "Your Queen of Swords in the Present, crossed by the Five of Cups, reveals..."
   - Tap any card name → jump to that card
   - Final guidance paragraph

9. **Return Home** / **Save Reading**

---

## Technical Implementation Notes

### **Component Architecture**
```
ReadingScreen (container)
├─ SpreadLayoutEngine (determines layout)
├─ SpreadNavigator (mini-map, mode switcher)
└─ Spread View (depends on type):
    ├─ LinearSpreadView (horizontal swipe)
    ├─ VerticalSpreadView (vertical scroll)
    ├─ SpatialSpreadView (2D pan/zoom)
    ├─ DecisionTreeView (branching)
    └─ SynthesisView (text summary)
```

### **State Management**
```javascript
const [currentSpread, setCurrentSpread] = useState(reading);
const [viewMode, setViewMode] = useState('overview'); // overview | detail | synthesis
const [focusedCardIndex, setFocusedCardIndex] = useState(null);
const [sectionView, setSectionView] = useState(null); // 'cross' | 'staff' | null
```

### **Gesture Handling**
- React Native PanResponder for 2D pan
- Swipe detection for card navigation
- Pinch gesture for zoom (react-native-gesture-handler)
- Tap to focus, long-press for options

---

## SUMMARY

This design provides:
- ✅ **5 different 3-card spread types** (various use cases)
- ✅ **6-card Relationship spread** (paginated multi-row)
- ✅ **10-card Celtic Cross** (complex spatial with multi-mode UI)
- ✅ **Scalable architecture** for adding more spreads
- ✅ **Interaction patterns** appropriate to each spread type
- ✅ **Synthesis view** powered by LunatiQ AGI
- ✅ **Visual design language** consistent with retro ASCII aesthetic

The key insight: **Don't force all spreads into one UI pattern**. Match the interface to the spread's inherent structure and purpose.
