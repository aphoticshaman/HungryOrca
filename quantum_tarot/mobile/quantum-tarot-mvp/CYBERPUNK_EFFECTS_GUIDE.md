# 🔮⚡ CYBERPUNK TERMINAL EFFECTS GUIDE

## Overview

Full neon cyberpunk hacker aesthetic with Matrix-style terminal effects. Forget traditional tarot aesthetics - we're going **FULL TERMINAL HACKER** with glowing neon text, LPMUD color codes, character morphing, and Matrix rain.

## Color System

### Neon Colors (`src/styles/cyberpunkColors.js`)

```javascript
import { NEON_COLORS } from '../styles/cyberpunkColors';

// Primary neons
NEON_COLORS.cyan        // #00FFFF - Electric cyan
NEON_COLORS.magenta     // #FF00FF - Hot magenta
NEON_COLORS.yellow      // #FFFF00 - Blazing yellow
NEON_COLORS.green       // #00FF00 - Matrix green

// High-intensity (bright as fuck)
NEON_COLORS.hiCyan      // Brightest cyan
NEON_COLORS.hiMagenta   // Brightest magenta
NEON_COLORS.hiYellow    // Brightest yellow
NEON_COLORS.hiWhite     // Pure white glow

// Dimmed (for less emphasis)
NEON_COLORS.dimCyan     // Darker cyan
NEON_COLORS.dimMagenta  // Darker magenta

// Glow colors (for text shadows)
NEON_COLORS.glowCyan    // rgba(0, 255, 255, 0.8)
NEON_COLORS.glowMagenta // rgba(255, 0, 255, 0.8)
```

## Terminal Effects Components

### NeonText - Basic glowing text

```javascript
import { NeonText } from '../components/TerminalEffects';

<NeonText color={NEON_COLORS.cyan}>
  GLOWING CYAN TEXT
</NeonText>

// Custom glow
<NeonText
  color={NEON_COLORS.hiMagenta}
  glowColor={NEON_COLORS.glowMagenta}
>
  CUSTOM GLOW
</NeonText>
```

### LPMUDText - LPMUD color codes

Supports text MUD style color codes like `$HIY$`, `$HIC$`, `$HIM$`, etc.

```javascript
import { LPMUDText } from '../components/TerminalEffects';

<LPMUDText>
  $HIC$CYAN TEXT $HIY$YELLOW TEXT $HIM$MAGENTA$NOR$
</LPMUDText>
```

**Available codes:**
- `$HIY$` - High-intensity yellow
- `$HIW$` - High-intensity white
- `$HIC$` - High-intensity cyan
- `$HIM$` - High-intensity magenta
- `$HIG$` - High-intensity green
- `$HIR$` - High-intensity red
- `$HIB$` - High-intensity blue
- `$YEL$`, `$WHT$`, `$CYN$`, `$MAG$` - Normal intensity
- `$NOR$` or `$RESET$` - Reset to default

### FlickerText - CRT monitor flicker

```javascript
import { FlickerText } from '../components/TerminalEffects';

<FlickerText
  color={NEON_COLORS.cyan}
  flickerSpeed={100}  // milliseconds
>
  FLICKERING TEXT
</FlickerText>
```

### GlitchText - Random character glitching

```javascript
import { GlitchText } from '../components/TerminalEffects';

<GlitchText
  glitchChance={0.1}    // 10% chance per interval
  glitchSpeed={100}     // Check every 100ms
>
  GLITCHY TEXT
</GlitchText>
```

### MorphText - Character-by-character Matrix transformation

```javascript
import { MorphText } from '../components/TerminalEffects';

<MorphText
  morphSpeed={50}       // 50ms per character
  color={NEON_COLORS.hiGreen}
  onMorphComplete={() => console.log('Morph done!')}
>
  TEXT THAT MORPHS IN
</MorphText>
```

### MatrixRain - Falling character columns

```javascript
import { MatrixRain } from '../components/TerminalEffects';

<MatrixRain
  width={300}
  height={400}
  speed={50}  // Update interval in ms
/>
```

### ScanLines - CRT scan line overlay

```javascript
import { ScanLines } from '../components/TerminalEffects';

<View style={StyleSheet.absoluteFill}>
  <ScanLines />
</View>
```

## Complete Components

### CyberpunkCard

Full-featured flippable card with terminal aesthetic:

```javascript
import CyberpunkCard from '../components/CyberpunkCard';

<CyberpunkCard
  cardIndex={0}           // Index in CARD_DATABASE
  reversed={false}        // Is card reversed?
  position="Past"         // Spread position name
  onReveal={() => {}}     // Callback when card revealed
/>
```

**Features:**
- Neon glowing border with pulsing animation
- Glitch effect on card number
- LPMUD color-coded ASCII art
- Flicker button text
- Flip animation (front/back)
- Matrix rain on reveal
- CRT scan lines overlay
- Terminal-style data stream on back

### CyberpunkHeader

Glitchy title screen:

```javascript
import CyberpunkHeader from '../components/CyberpunkHeader';

<CyberpunkHeader showMatrixBg={true} />
```

## Usage Examples

### Card Title with Glitch

```javascript
<GlitchText style={styles.title} glitchChance={0.05}>
  [THE FOOL]
</GlitchText>
```

### Multi-color Data Display

```javascript
<LPMUDText>
  $HIC${'>'} CARD DATA ${'<'}$NOR${'\n'}
  $HIY$━━━━━━━━━━━━━━$NOR${'\n'}
  $HIM$ELEMENT:$NOR$ FIRE{'\n'}
  $HIG$UPRIGHT:$NOR$ courage, passion
</LPMUDText>
```

### Matrix Effect Transition

```javascript
const [showing, setShowing] = useState(false);

{showing && (
  <View style={StyleSheet.absoluteFill}>
    <MatrixRain width={SCREEN_WIDTH} height={600} />
  </View>
)}
```

### Flicker Button

```javascript
<TouchableOpacity onPress={handlePress}>
  <FlickerText color={NEON_COLORS.hiYellow}>
    [ PRESS TO CONTINUE ]
  </FlickerText>
</TouchableOpacity>
```

## Styling Tips

### Monospace Font

Always use monospace for terminal aesthetic:

```javascript
fontFamily: 'monospace'
```

### Neon Glow Shadow

```javascript
textShadowColor: NEON_COLORS.glowCyan,
textShadowOffset: { width: 0, height: 0 },
textShadowRadius: 10,
```

### Terminal Border

```javascript
borderWidth: 2,
borderColor: NEON_COLORS.dimCyan,
backgroundColor: '#000000',
```

### Box Drawing Characters

Use for frames:
```
╔═══╗  ┌───┐  ┏━━━┓
║   ║  │   │  ┃   ┃
╚═══╝  └───┘  ┗━━━┛
```

## Performance Notes

- FlickerText uses setInterval - keep instances limited
- GlitchText uses setInterval - use sparingly (< 5 at once)
- MatrixRain is expensive - use for transitions only
- MorphText auto-stops after completion
- ScanLines uses Animated.loop - one per screen max

## Future Enhancements

- [ ] Audio: Terminal beep sounds on reveal
- [ ] Haptics: Vibration on glitch effects
- [ ] More glitch patterns (horizontal shift, color separation)
- [ ] Terminal typing effect (cursor + character delay)
- [ ] Hologram shimmer effect
- [ ] Data corruption animation
- [ ] Network "loading" effect with fake packets

## Aesthetic Philosophy

**NO SKEUOMORPHISM.** No realistic card images. No gradient backgrounds.

This is **PURE TERMINAL**. Think:
- 1980s hacker movies
- The Matrix (1999)
- Cyberpunk 2077 UI
- Text-based MUDs
- Old CRT monitors
- Neon everything

We're creating a **MOAT** - no other tarot app does this. This aesthetic IS the product differentiation.
