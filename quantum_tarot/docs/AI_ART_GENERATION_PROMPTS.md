# Quantum Tarot - AI Art Generation Prompts

## Overview
This document provides Midjourney/DALL-E prompts to generate all 78 tarot cards across 5 aesthetic styles.

**Total Cards Needed:** 78 cards × 5 styles = 390 images

**Recommended Tools:**
- **Midjourney** (best for artistic quality, $30/mo) - RECOMMENDED
- **DALL-E 3** (good for specific styles, $20 for 115 images)
- **Stable Diffusion** (free, more technical setup)

## General Prompt Structure

```
[Card Name] tarot card, [aesthetic style], [key symbolism], [mood/atmosphere],
professional tarot card illustration, vertical orientation, ornate border,
--ar 2:3 --v 6
```

## 5 Aesthetic Styles

### Style 1: Minimal Modern (Gen Z Analytical)
**Base Prompt Template:**
```
[Card] tarot card, minimal modern style, clean line art, geometric shapes,
gold foil on pure black background, art deco influences, simple elegant design,
professional minimalist illustration, --ar 2:3 --v 6
```

### Style 2: Soft Mystical (Millennial Intuitive)
**Base Prompt Template:**
```
[Card] tarot card, soft mystical watercolor style, dreamy lavender and rose gold palette,
gentle glowing effects, ethereal atmosphere, delicate details, romantic and spiritual,
professional tarot illustration, --ar 2:3 --v 6
```

### Style 3: Bold Authentic (Gen Z Authentic/Raw)
**Base Prompt Template:**
```
[Card] tarot card, bold modern style, high contrast neon colors (electric blue, hot pink, deep purple),
Y2K aesthetic, digital art, sharp edges, contemporary spiritual, edgy design,
professional illustration, --ar 2:3 --v 6
```

### Style 4: Elegant Classic (Traditional)
**Base Prompt Template:**
```
[Card] tarot card, traditional Rider-Waite style, ornate gold borders,
classic tarot imagery, rich jewel tones, art nouveau influences,
sophisticated and timeless, professional traditional illustration, --ar 2:3 --v 6
```

### Style 5: Witchy Earthy (Alternative Spiritual)
**Base Prompt Template:**
```
[Card] tarot card, witchy earthy style, botanical illustrations, natural textures (wood, stone),
terracotta and sage green palette, crystals and moon phases, hand-drawn organic feel,
professional nature-inspired illustration, --ar 2:3 --v 6
```

---

## Complete Card Prompts

### Major Arcana

#### 0 - The Fool
**Symbolism:** Youth stepping off cliff, small dog, sun, white rose, mountains, bag on stick

**Minimal Modern:**
```
The Fool tarot card, minimal modern style, simple line drawing of figure stepping forward,
geometric sun, clean white silhouette on black background, gold accent lines,
art deco simplicity, professional minimalist tarot, --ar 2:3 --v 6
```

**Soft Mystical:**
```
The Fool tarot card, soft watercolor style, innocent figure stepping into clouds,
gentle sunrise in lavender and pink, dreamy atmosphere, white rose glowing softly,
ethereal and hopeful mood, professional spiritual illustration, --ar 2:3 --v 6
```

**Bold Authentic:**
```
The Fool tarot card, bold Y2K style, figure in motion with neon outline (electric blue and hot pink),
digital sunrise, high contrast, contemporary spiritual energy, edgy and fearless,
professional modern tarot, --ar 2:3 --v 6
```

**Elegant Classic:**
```
The Fool tarot card, traditional Rider-Waite style, young traveler at cliff edge,
small white dog, yellow sun overhead, white rose, ornate gold border,
rich traditional colors, classic tarot artistry, --ar 2:3 --v 6
```

**Witchy Earthy:**
```
The Fool tarot card, witchy botanical style, figure among wildflowers,
natural wood textures, sage and terracotta palette, pressed flower aesthetic,
crystals in border, earthy spiritual vibe, --ar 2:3 --v 6
```

---

#### 1 - The Magician
**Symbolism:** Figure with infinity symbol, tools on table (wand, cup, sword, pentacle), red and white roses

**Minimal Modern:**
```
The Magician tarot card, minimal geometric style, figure with infinity symbol above head,
four simple icons (circle, triangle, line, square) representing tools,
gold on black, clean powerful design, --ar 2:3 --v 6
```

**Soft Mystical:**
```
The Magician tarot card, soft watercolor, glowing figure channeling cosmic energy,
delicate rose gold infinity symbol, gentle magical atmosphere,
lavender and pink roses, ethereal power, --ar 2:3 --v 6
```

**Bold Authentic:**
```
The Magician tarot card, bold neon style, figure with electric energy flowing,
infinity symbol in hot pink, tools glowing with cyber magic,
high contrast modern mystic, --ar 2:3 --v 6
```

**Elegant Classic:**
```
The Magician tarot card, traditional style, robed figure with raised wand,
table with four tools, infinity symbol overhead, red and white roses,
ornate gold border, classic Rider-Waite interpretation, --ar 2:3 --v 6
```

**Witchy Earthy:**
```
The Magician tarot card, witchy botanical style, figure surrounded by herbs and crystals,
natural altar with earthly tools, infinity made of vines,
terracotta and green palette, organic magic, --ar 2:3 --v 6
```

---

#### 2 - The High Priestess
**Symbolism:** Seated between pillars (B and J), crescent moon crown, Torah/scroll, pomegranates

**Minimal Modern:**
```
The High Priestess tarot card, minimal style, simple figure seated between two vertical lines,
crescent moon symbol, clean symmetry, silver and black palette,
elegant mystery, --ar 2:3 --v 6
```

**Soft Mystical:**
```
The High Priestess tarot card, soft watercolor, mystical figure bathed in moonlight,
sheer veils, gentle purple and silver tones, dreamy intuitive atmosphere,
pomegranates glowing softly, --ar 2:3 --v 6
```

**Bold Authentic:**
```
The High Priestess tarot card, bold style, powerful figure between neon pillars,
electric blue moon crown, contemporary mystic energy,
mysterious and strong, --ar 2:3 --v 6
```

**Elegant Classic:**
```
The High Priestess tarot card, traditional Rider-Waite style, priestess between B and J pillars,
crescent moon crown, scroll of Torah, blue robes, pomegranate tapestry,
ornate gold frame, --ar 2:3 --v 6
```

**Witchy Earthy:**
```
The High Priestess tarot card, witchy style, figure among night-blooming flowers,
natural wood pillars, moon phases in border, pomegranates and herbs,
deep green and silver, nocturnal magic, --ar 2:3 --v 6
```

---

## Production Strategy

### Phase 1: MVP (Minimal Viable Product)
Generate cards for **ONE aesthetic style only** (recommend Soft Mystical for broadest appeal):
- 22 Major Arcana cards
- 56 Minor Arcana cards (14 per suit)
- **Total: 78 cards**

**Cost Estimate:**
- Midjourney: $30/month (unlimited in relax mode)
- Time: ~2-4 hours for 78 cards (with iterations)

### Phase 2: Full Product
Generate all 5 aesthetic styles:
- **Total: 390 cards**
- **Cost Estimate:** 2-3 months of Midjourney ($60-90)
- **Time:** 1-2 weeks of focused work

---

## Batch Generation Script

Create a file with all prompts and automate with Midjourney Discord bot:

```
/imagine The Fool tarot card, soft watercolor style, innocent figure stepping into clouds...
/imagine The Magician tarot card, soft watercolor, glowing figure channeling cosmic energy...
/imagine The High Priestess tarot card, soft watercolor, mystical figure bathed in moonlight...
[...continue for all 78 cards]
```

## Post-Processing

After generation:
1. **Upscale** all images to high resolution (Midjourney U1-U4 buttons)
2. **Download** in highest quality
3. **Organize** into folders by suit and style:
   ```
   /assets/cards/soft_mystical/
     /major_arcana/
       00_the_fool.png
       01_the_magician.png
     /wands/
       01_ace_of_wands.png
     /cups/
     /swords/
     /pentacles/
   ```
4. **Optimize** for mobile (compress to ~200KB per image)
5. **Add to repo** or CDN

## Quick Reference: Minor Arcana Themes

### Wands (Fire/Action)
- Ace: Single wand sprouting leaves
- 2-10: Progressive action/creativity scenes
- Page: Young enthusiastic figure
- Knight: Figure in motion, passionate energy
- Queen: Confident seated figure with sunflowers
- King: Commanding figure with salamanders

### Cups (Water/Emotion)
- Ace: Chalice overflowing with water
- 2-10: Emotional journey scenes
- Page: Gentle dreamy youth
- Knight: Romantic figure with flowing water
- Queen: Intuitive seated figure by water
- King: Wise figure, calm sea

### Swords (Air/Intellect)
- Ace: Single sword through crown
- 2-10: Mental challenges and conflicts
- Page: Alert vigilant youth
- Knight: Charging figure, storm clouds
- Queen: Seated figure, clear sky
- King: Authoritative figure, clouds

### Pentacles (Earth/Material)
- Ace: Single coin with pentacle symbol
- 2-10: Material world progression
- Page: Studious grounded youth
- Knight: Steady reliable figure in field
- Queen: Abundant garden scene
- King: Prosperous castle scene

---

## Alternative: Commission Custom Art

If AI art doesn't feel right or you want exclusive designs:

**Freelance Platforms:**
- **Fiverr:** $50-200 per card ($3,900-15,600 for 78 cards)
- **Upwork:** $75-300 per card
- **Artstation:** Premium artists, $200-500 per card

**Recommendation:** Use AI for MVP, commission custom art once you have revenue.

---

## Legal Considerations

### AI-Generated Art
- **Midjourney/DALL-E:** You own commercial rights to generated images
- **No copyright issues** as long as you're not copying existing tarot decks
- **Rider-Waite is public domain** (published 1909), so interpretations are fair game

### Best Practice
Include in your Terms of Service:
- "Card artwork generated using AI tools (Midjourney/DALL-E)"
- "Inspired by traditional Rider-Waite-Smith symbolism"
- "Original interpretations and designs"

---

## Next Steps

1. **Sign up for Midjourney** ($30/mo)
2. **Start with Major Arcana** (22 cards, easiest to test)
3. **Pick one aesthetic** (Soft Mystical recommended for MVP)
4. **Generate and iterate** (usually takes 1-3 attempts per card to get it right)
5. **Organize and upload** to your asset pipeline

**Time Estimate:** 3-5 hours for all 22 Major Arcana in one style.

---

**Pro Tip:** Join Midjourney's Discord and look at other tarot decks people have generated. Use `/imagine` with image references (--iw parameter) to maintain consistency across your deck.

Your cards will be unique, professional, and ready for production! 🎨✨
