"""
Quantum Tarot - Adaptive Language & Communication Engine
Delivers readings in personalized voice based on personality profile + demographics

RESEARCH SYNTHESIS (2024-2025):
- Labyrinthos: Clean aesthetic, educational, but too game-like
- Co-Star: Witty/blunt but criticized as harsh/cold/triggering
- Millennials: Prefer soft, calming, serene communication
- Gen Z: Prefer authentic, raw, less perfection
- Both want: Personalized, bite-sized, interactive, modern spiritual content
- Key insight: "How we talk to friends on the couch, in group chats"

OUR APPROACH:
Avoid Co-Star's harshness trap while maintaining authenticity.
Deliver same card meaning in COMPLETELY different voices based on:
1. Personality profile (PRIMARY)
2. Gender identity (SECONDARY)
3. Reading type context
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class CommunicationVoice(Enum):
    """Different communication styles based on personality"""
    ANALYTICAL_GUIDE = "analytical_guide"  # For high analytical, low intuitive
    INTUITIVE_MYSTIC = "intuitive_mystic"  # For high intuitive, low analytical
    SUPPORTIVE_FRIEND = "supportive_friend"  # For high emotional regulation needs
    DIRECT_COACH = "direct_coach"  # For high action orientation
    GENTLE_NURTURER = "gentle_nurturer"  # For sensitivity, past trauma indicators
    WISE_MENTOR = "wise_mentor"  # For structure-oriented, seeks guidance
    PLAYFUL_EXPLORER = "playful_explorer"  # For high openness, low rigidity
    BALANCED_SAGE = "balanced_sage"  # Default middle path


class AestheticProfile(Enum):
    """Visual/textual aesthetic preferences"""
    MINIMAL_MODERN = "minimal_modern"  # Gen Z, clean lines
    SOFT_MYSTICAL = "soft_mystical"  # Millennial, dreamy
    BOLD_AUTHENTIC = "bold_authentic"  # Gen Z, raw truth
    ELEGANT_CLASSIC = "elegant_classic"  # Traditional, timeless
    WITCHY_EARTHY = "witchy_earthy"  # Alternative spiritual


@dataclass
class UserCommunicationProfile:
    """Complete communication preferences for a user"""
    primary_voice: CommunicationVoice
    secondary_voice: Optional[CommunicationVoice] = None  # Blend if needed
    aesthetic: AestheticProfile = AestheticProfile.SOFT_MYSTICAL

    # Delivery preferences
    sentence_length: str = "medium"  # "short", "medium", "long"
    metaphor_density: str = "medium"  # "low", "medium", "high"
    therapeutic_explicitness: str = "subtle"  # "hidden", "subtle", "explicit"
    spiritual_language: str = "moderate"  # "minimal", "moderate", "rich"
    emoji_use: bool = True  # Gen Z/Millennial preference

    # Tone modifiers
    warmth_level: float = 0.7  # 0-1 scale
    directness_level: float = 0.5  # 0-1 scale
    empowerment_vs_comfort: float = 0.5  # 0=comfort, 1=challenge


class AdaptiveLanguageEngine:
    """
    Generates personalized card interpretations based on user profile.
    Same card, different delivery.
    """

    def __init__(self):
        pass

    def determine_voice(self, personality_profile) -> CommunicationVoice:
        """
        Map personality profile to communication voice.
        This is where psychology meets UX.
        """
        # Extract key traits
        analytical = personality_profile.analytical_intuitive < 0.4  # Lower = more analytical
        intuitive = personality_profile.analytical_intuitive > 0.6  # Higher = more intuitive
        high_emotion = personality_profile.emotional_regulation < 0.4  # Needs support
        action_oriented = personality_profile.action_orientation > 0.6
        sensitive = personality_profile.emotional_regulation < 0.3
        structured = personality_profile.structure_flexibility < 0.4
        flexible = personality_profile.structure_flexibility > 0.6

        # Decision tree for voice selection
        if sensitive or high_emotion:
            return CommunicationVoice.GENTLE_NURTURER
        elif analytical and structured:
            return CommunicationVoice.ANALYTICAL_GUIDE
        elif intuitive and flexible:
            return CommunicationVoice.INTUITIVE_MYSTIC
        elif action_oriented and personality_profile.optimism_realism > 0.6:
            return CommunicationVoice.DIRECT_COACH
        elif structured and personality_profile.internal_external_locus > 0.5:
            return CommunicationVoice.WISE_MENTOR
        elif flexible and personality_profile.risk_tolerance > 0.6:
            return CommunicationVoice.PLAYFUL_EXPLORER
        else:
            return CommunicationVoice.BALANCED_SAGE

    def determine_aesthetic(self, personality_profile, birth_year: Optional[int] = None) -> AestheticProfile:
        """Determine visual/textual aesthetic"""
        # Generational defaults
        gen_z = birth_year and birth_year >= 1997
        millennial = birth_year and 1981 <= birth_year < 1997

        if gen_z and personality_profile.analytical_intuitive < 0.4:
            return AestheticProfile.MINIMAL_MODERN
        elif gen_z:
            return AestheticProfile.BOLD_AUTHENTIC
        elif millennial:
            return AestheticProfile.SOFT_MYSTICAL
        elif personality_profile.analytical_intuitive < 0.3:
            return AestheticProfile.MINIMAL_MODERN
        elif personality_profile.structure_flexibility < 0.4:
            return AestheticProfile.ELEGANT_CLASSIC
        else:
            return AestheticProfile.WITCHY_EARTHY

    def build_communication_profile(
        self,
        personality_profile,
        birth_year: Optional[int] = None,
        gender_identity: Optional[str] = None
    ) -> UserCommunicationProfile:
        """
        Build complete communication profile.
        This determines HOW we talk to this specific user.
        """
        voice = self.determine_voice(personality_profile)
        aesthetic = self.determine_aesthetic(personality_profile, birth_year)

        # Adjust delivery parameters based on profile
        profile = UserCommunicationProfile(
            primary_voice=voice,
            aesthetic=aesthetic
        )

        # Fine-tune based on personality traits

        # Sentence length
        if personality_profile.analytical_intuitive < 0.4:
            profile.sentence_length = "medium"  # Analytical wants clarity
        elif personality_profile.analytical_intuitive > 0.6:
            profile.sentence_length = "long"  # Intuitive enjoys flow
        else:
            profile.sentence_length = "medium"

        # Metaphor density
        if personality_profile.analytical_intuitive < 0.3:
            profile.metaphor_density = "low"  # Give it to them straight
        elif personality_profile.analytical_intuitive > 0.7:
            profile.metaphor_density = "high"  # They love symbolic language
        else:
            profile.metaphor_density = "medium"

        # Therapeutic explicitness
        if personality_profile.primary_framework == "CBT":
            profile.therapeutic_explicitness = "subtle"  # They get it
        elif personality_profile.primary_framework == "DBT":
            profile.therapeutic_explicitness = "subtle"  # Name the skills subtly
        else:
            profile.therapeutic_explicitness = "hidden"  # Deep integration

        # Spiritual language
        if personality_profile.analytical_intuitive < 0.3:
            profile.spiritual_language = "minimal"  # Less woo, more practical
        elif personality_profile.analytical_intuitive > 0.7:
            profile.spiritual_language = "rich"  # Full mystical experience
        else:
            profile.spiritual_language = "moderate"

        # Emoji use (generational)
        gen_z_millennial = birth_year and birth_year >= 1981
        profile.emoji_use = gen_z_millennial

        # Tone modifiers
        profile.warmth_level = 0.9 - personality_profile.emotional_regulation  # Lower reg = need more warmth
        profile.directness_level = personality_profile.action_orientation  # Action-oriented want direct
        profile.empowerment_vs_comfort = personality_profile.internal_external_locus  # Internal locus = can handle challenge

        return profile

    def generate_card_interpretation(
        self,
        card,
        position_meaning: str,
        is_reversed: bool,
        comm_profile: UserCommunicationProfile,
        reading_type: str
    ) -> str:
        """
        Generate the actual card interpretation text.
        This is where the magic happens - same card, different delivery.
        """
        # Select base meaning
        if reading_type == "career":
            base_meaning = card.career_interpretation
        elif reading_type == "romance":
            base_meaning = card.romance_interpretation
        elif reading_type == "wellness":
            base_meaning = card.wellness_interpretation
        elif reading_type == "family":
            base_meaning = card.family_interpretation
        elif reading_type == "self_growth":
            base_meaning = card.self_growth_interpretation
        elif reading_type == "school":
            base_meaning = card.school_interpretation
        else:
            base_meaning = card.upright_meaning if not is_reversed else card.reversed_meaning

        # Adapt to communication voice
        adapted = self._adapt_to_voice(
            base_meaning=base_meaning,
            card=card,
            position=position_meaning,
            is_reversed=is_reversed,
            voice=comm_profile.primary_voice,
            comm_profile=comm_profile
        )

        return adapted

    def _adapt_to_voice(
        self,
        base_meaning: str,
        card,
        position: str,
        is_reversed: bool,
        voice: CommunicationVoice,
        comm_profile: UserCommunicationProfile
    ) -> str:
        """
        Transform base meaning to match communication voice.
        This is the secret sauce.
        """

        # Get voice-specific framings
        voice_templates = self._get_voice_templates()
        template = voice_templates[voice]

        # Build the interpretation
        parts = []

        # Opening (position context)
        opening = template["opening"].format(
            position=position,
            card_name=card.name,
            reversed="reversed " if is_reversed else ""
        )
        parts.append(opening)

        # Core meaning (adapted)
        core = self._adapt_sentence_style(base_meaning, comm_profile)
        parts.append(core)

        # Psychological insight (if appropriate explicitness)
        if comm_profile.therapeutic_explicitness != "hidden":
            psych = self._generate_psychological_insight(card, voice, comm_profile)
            if psych:
                parts.append(psych)

        # Growth prompt (if appropriate)
        if comm_profile.empowerment_vs_comfort > 0.5:
            prompt = self._adapt_sentence_style(card.psychology.growth_prompt, comm_profile)
            parts.append(prompt)

        # Closing (voice-specific)
        if template.get("closing"):
            parts.append(template["closing"])

        # Add emoji if profile allows
        if comm_profile.emoji_use and voice in [
            CommunicationVoice.PLAYFUL_EXPLORER,
            CommunicationVoice.SUPPORTIVE_FRIEND,
            CommunicationVoice.GENTLE_NURTURER
        ]:
            parts[-1] = parts[-1] + " " + self._get_card_emoji(card)

        return " ".join(parts)

    def _get_voice_templates(self) -> Dict:
        """
        Voice-specific sentence framings.
        This is how we talk differently to different people.
        """
        return {
            CommunicationVoice.ANALYTICAL_GUIDE: {
                "opening": "In the {position} position, {reversed}{card_name} indicates:",
                "connector": "This suggests that",
                "closing": None
            },
            CommunicationVoice.INTUITIVE_MYSTIC: {
                "opening": "The {reversed}{card_name} appears in your {position}, whispering:",
                "connector": "The universe is showing you that",
                "closing": "Trust what you already know."
            },
            CommunicationVoice.SUPPORTIVE_FRIEND: {
                "opening": "Hey, so {reversed}{card_name} showed up in your {position}, and here's what I'm seeing:",
                "connector": "What this means for you is",
                "closing": "You've got this."
            },
            CommunicationVoice.DIRECT_COACH: {
                "opening": "{reversed}{card_name} in the {position} position. Here's what you need to know:",
                "connector": "Bottom line:",
                "closing": "Now take action."
            },
            CommunicationVoice.GENTLE_NURTURER: {
                "opening": "Sweetie, {reversed}{card_name} has come through in your {position}, gently reminding you:",
                "connector": "This is inviting you to",
                "closing": "Be gentle with yourself through this."
            },
            CommunicationVoice.WISE_MENTOR: {
                "opening": "The {reversed}{card_name} appears in your {position} as a teacher, offering this wisdom:",
                "connector": "What this lesson brings is",
                "closing": "Reflect on this truth."
            },
            CommunicationVoice.PLAYFUL_EXPLORER: {
                "opening": "Ooh, {reversed}{card_name} in your {position}! Here's what's up:",
                "connector": "This is your invitation to",
                "closing": "Have fun with this!"
            },
            CommunicationVoice.BALANCED_SAGE: {
                "opening": "{reversed}{card_name} in the {position} position speaks to:",
                "connector": "This brings both challenge and opportunity:",
                "closing": "Balance is the key."
            }
        }

    def _adapt_sentence_style(self, text: str, comm_profile: UserCommunicationProfile) -> str:
        """Adjust sentence length and style"""
        # For MVP, return as-is
        # In production, would use NLP to restructure
        return text

    def _generate_psychological_insight(
        self,
        card,
        voice: CommunicationVoice,
        comm_profile: UserCommunicationProfile
    ) -> Optional[str]:
        """Generate therapeutic insight appropriate to voice and explicitness"""

        if not card.psychology or not card.psychology.core_metaphor:
            return None

        metaphor = card.psychology.core_metaphor

        if comm_profile.therapeutic_explicitness == "hidden":
            return None
        elif comm_profile.therapeutic_explicitness == "subtle":
            # Integrate without naming the framework
            if voice == CommunicationVoice.SUPPORTIVE_FRIEND:
                return f"Kind of like when {metaphor.lower()}"
            elif voice == CommunicationVoice.ANALYTICAL_GUIDE:
                return f"This pattern mirrors {metaphor.lower()}"
            else:
                return None
        else:  # explicit
            frameworks = ", ".join(card.psychology.dbt_skills + card.psychology.cbt_concepts)
            return f"(Relates to: {frameworks})"

    def _get_card_emoji(self, card) -> str:
        """Get appropriate emoji for card"""
        # Simple mapping - can expand
        emoji_map = {
            "The Fool": "🌟",
            "The Magician": "✨",
            "The High Priestess": "🌙",
            "The Empress": "🌸",
            "The Emperor": "👑",
            "The Sun": "☀️",
            "The Moon": "🌕",
            "The Star": "⭐",
            "The World": "🌍",
            "Ace": "🎯",
        }

        for key, emoji in emoji_map.items():
            if key in card.name:
                return emoji

        # Default by suit
        if card.suit.value == "wands":
            return "🔥"
        elif card.suit.value == "cups":
            return "💧"
        elif card.suit.value == "swords":
            return "⚔️"
        elif card.suit.value == "pentacles":
            return "🌿"

        return "✨"


# ============================================================================
# EXAMPLE USAGE - SAME CARD, DIFFERENT VOICES
# ============================================================================

def demonstrate_adaptive_language():
    """Show how same card is delivered differently"""
    from .complete_deck import get_complete_deck, Suit
    from .personality_profiler import PersonalityProfile

    deck = get_complete_deck()
    three_of_swords = [c for c in deck if c.name == "Three of Swords"][0]

    engine = AdaptiveLanguageEngine()

    # Scenario 1: Analytical, emotionally regulated person
    profile1 = PersonalityProfile(
        user_id="user1",
        reading_type="romance",
        timestamp=0,
        analytical_intuitive=0.2,  # Very analytical
        emotional_regulation=0.7,  # Good regulation
        action_orientation=0.6,
        primary_framework="CBT"
    )
    comm1 = engine.build_communication_profile(profile1, birth_year=1995)
    interp1 = engine.generate_card_interpretation(
        three_of_swords, "Present", False, comm1, "romance"
    )

    # Scenario 2: Intuitive, emotionally sensitive person
    profile2 = PersonalityProfile(
        user_id="user2",
        reading_type="romance",
        timestamp=0,
        analytical_intuitive=0.8,  # Very intuitive
        emotional_regulation=0.2,  # Needs support
        action_orientation=0.3,
        primary_framework="DBT"
    )
    comm2 = engine.build_communication_profile(profile2, birth_year=2000)
    interp2 = engine.generate_card_interpretation(
        three_of_swords, "Present", False, comm2, "romance"
    )

    print("=" * 70)
    print("SAME CARD, DIFFERENT DELIVERY")
    print("Card: Three of Swords (Heartbreak)")
    print("=" * 70)
    print(f"\nAnalytical Person ({comm1.primary_voice.value}):")
    print(f"Voice: {comm1.primary_voice.value}")
    print(f"Aesthetic: {comm1.aesthetic.value}")
    print(interp1)

    print(f"\n\nIntuitive Person ({comm2.primary_voice.value}):")
    print(f"Voice: {comm2.primary_voice.value}")
    print(f"Aesthetic: {comm2.aesthetic.value}")
    print(interp2)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_adaptive_language()
