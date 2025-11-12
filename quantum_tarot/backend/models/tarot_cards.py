"""
Quantum Tarot - Comprehensive Rider-Waite-Smith Deck Database
Integrates traditional meanings with DBT, CBT, and Army MRT frameworks
"""

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass, field


class Suit(Enum):
    MAJOR = "major_arcana"
    WANDS = "wands"
    CUPS = "cups"
    SWORDS = "swords"
    PENTACLES = "pentacles"


class Element(Enum):
    SPIRIT = "spirit"
    FIRE = "fire"
    WATER = "water"
    AIR = "air"
    EARTH = "earth"


class ReadingType(Enum):
    CAREER = "career"
    FAMILY = "family"
    ROMANCE = "romance"
    WELLNESS = "wellness"
    SELF_GROWTH = "self_growth"
    SCHOOL = "school"
    GENERAL = "general"
    SURPRISE = "surprise_me"


@dataclass
class PsychologicalMapping:
    """Maps tarot meanings to therapeutic frameworks"""
    dbt_skills: List[str] = field(default_factory=list)  # Dialectical Behavior Therapy
    cbt_concepts: List[str] = field(default_factory=list)  # Cognitive Behavioral Therapy
    mrt_pillars: List[str] = field(default_factory=list)  # Master Resilience Training
    core_metaphor: str = ""
    growth_prompt: str = ""  # Therapeutic question/prompt


@dataclass
class TarotCard:
    """Complete tarot card definition with multi-dimensional meanings"""
    number: int
    name: str
    suit: Suit
    element: Element

    # Traditional meanings
    upright_keywords: List[str]
    reversed_keywords: List[str]
    upright_meaning: str
    reversed_meaning: str

    # Deep spiritual/psychological layers
    shadow_work: str  # Jungian shadow integration
    soul_lesson: str  # Spiritual growth aspect

    # Context-specific interpretations
    career_interpretation: str
    romance_interpretation: str
    wellness_interpretation: str
    family_interpretation: str
    self_growth_interpretation: str
    school_interpretation: str

    # Psychological framework integration
    psychology: PsychologicalMapping = field(default_factory=PsychologicalMapping)

    # Astrological correspondences
    astro_sign: Optional[str] = None
    astro_planet: Optional[str] = None

    # For quantum reading generation
    quantum_weight: float = 1.0  # Can be modified based on user profile


# ============================================================================
# MAJOR ARCANA - The Fool's Journey (0-21)
# ============================================================================

MAJOR_ARCANA = [
    TarotCard(
        number=0,
        name="The Fool",
        suit=Suit.MAJOR,
        element=Element.SPIRIT,
        upright_keywords=["new beginnings", "innocence", "spontaneity", "free spirit", "leap of faith"],
        reversed_keywords=["recklessness", "taken advantage of", "inconsideration", "naivety"],
        upright_meaning="The Fool represents new beginnings, unlimited potential, and the courage to step into the unknown. It's the soul before incarnation, pure possibility.",
        reversed_meaning="Caution against reckless decisions, naivety, or being taken advantage of. Need for more planning.",
        shadow_work="Examine where fear of judgment prevents authentic self-expression. Where do you play it too safe?",
        soul_lesson="Trust in the journey. Every master was once a beginner. Divine protection accompanies genuine innocence.",
        career_interpretation="A new job, career change, or entrepreneurial venture beckons. Trust your instincts even without all the answers.",
        romance_interpretation="New relationship or fresh start in existing one. Approach with open heart but maintain awareness.",
        wellness_interpretation="Time to try new health approaches. Listen to your body's wisdom, not just external advice.",
        family_interpretation="New family dynamics or the need to break unhealthy patterns. Beginner's mind helps.",
        self_growth_interpretation="You're at a threshold. The old self must be released for the new to emerge. Trust the process.",
        school_interpretation="New subject or approach to learning. Your unique perspective is valuable, not ignorant.",
        psychology=PsychologicalMapping(
            dbt_skills=["Wise Mind", "Willingness", "Non-Judgmental Stance"],
            cbt_concepts=["Cognitive Flexibility", "Growth Mindset", "Challenging Catastrophizing"],
            mrt_pillars=["Self-Awareness", "Mental Agility", "Character Strengths"],
            core_metaphor="Like starting therapy: vulnerable but necessary. The courage to not know everything.",
            growth_prompt="What would you attempt if you knew you couldn't fail? What's one small step toward that?"
        ),
        astro_sign="Uranus",
        astro_planet="Uranus"
    ),

    TarotCard(
        number=1,
        name="The Magician",
        suit=Suit.MAJOR,
        element=Element.SPIRIT,
        upright_keywords=["manifestation", "resourcefulness", "power", "inspired action", "skill"],
        reversed_keywords=["manipulation", "poor planning", "untapped talents", "scattered energy"],
        upright_meaning="You have all the tools you need. The Magician channels divine will into material reality. As above, so below.",
        reversed_meaning="Talents being misused or wasted. Manipulation or being manipulated. Lack of focus dissipates power.",
        shadow_work="Where do you give away your power? Where do you manipulate rather than inspire?",
        soul_lesson="You are a conduit for Source energy. Your will aligned with Divine Will creates miracles.",
        career_interpretation="You have the skills and resources. Now execute. Perfect time for presentations, negotiations, new projects.",
        romance_interpretation="Take initiative. Clear communication of desires manifests the relationship you want.",
        wellness_interpretation="Your mind powerfully affects your body. Visualization, affirmations, and intentional healing work now.",
        family_interpretation="Use your communication skills to mediate or clarify family situations. You can bridge gaps.",
        self_growth_interpretation="Recognize your own power. You're not a victim of circumstances but a creator of reality.",
        school_interpretation="You have the intellectual capacity. Apply focused study techniques and you'll excel.",
        psychology=PsychologicalMapping(
            dbt_skills=["Opposite Action", "Accumulating Positive Emotions", "Building Mastery"],
            cbt_concepts=["Self-Efficacy", "Locus of Control", "Behavioral Activation"],
            mrt_pillars=["Self-Regulation", "Mental Agility", "Optimism"],
            core_metaphor="Like mastering DBT skills: you have the tools, now practice applying them consciously.",
            growth_prompt="What resources do you already possess that you've been overlooking? How can you use them today?"
        ),
        astro_sign="Mercury",
        astro_planet="Mercury"
    ),

    TarotCard(
        number=2,
        name="The High Priestess",
        suit=Suit.MAJOR,
        element=Element.SPIRIT,
        upright_keywords=["intuition", "sacred knowledge", "divine feminine", "subconscious", "inner voice"],
        reversed_keywords=["secrets", "disconnected from intuition", "withdrawal", "repressed feelings"],
        upright_meaning="Trust your inner knowing. The answers lie within, not in external validation. She guards the veil between worlds.",
        reversed_meaning="Ignoring intuition in favor of logic alone. Secrets being kept. Disconnection from inner wisdom.",
        shadow_work="What truths are you avoiding? What does your intuition whisper that your ego refuses to hear?",
        soul_lesson="The universe speaks through symbols, dreams, and synchronicities. Stillness reveals what action obscures.",
        career_interpretation="Trust your gut about workplace dynamics. Information is coming that's not yet public. Wait before acting.",
        romance_interpretation="Pay attention to what's unsaid. Your intuition about this person/situation is correct. Honor your mysteries.",
        wellness_interpretation="Listen to subtle body signals. Explore mind-body connection, meditation, or alternative healing modalities.",
        family_interpretation="Family secrets or unspoken dynamics at play. Your intuition can navigate what words cannot.",
        self_growth_interpretation="Develop your intuitive abilities. Journal, meditate, note synchronicities. Your inner knowing is strengthening.",
        school_interpretation="Trust your understanding even if you can't immediately articulate it. Incubation period before breakthrough.",
        psychology=PsychologicalMapping(
            dbt_skills=["Observe and Describe", "Mindfulness", "Wise Mind"],
            cbt_concepts=["Interoceptive Awareness", "Emotion Identification", "Reflective Functioning"],
            mrt_pillars=["Self-Awareness", "Emotional Regulation", "Realistic Optimism"],
            core_metaphor="Like the pause before responding in therapy: stillness and reflection reveal deeper truths.",
            growth_prompt="What has your intuition been trying to tell you? What would change if you trusted it?"
        ),
        astro_sign="Moon",
        astro_planet="Moon"
    ),

    TarotCard(
        number=3,
        name="The Empress",
        suit=Suit.MAJOR,
        element=Element.SPIRIT,
        upright_keywords=["abundance", "nature", "nurturing", "fertility", "creativity", "sensuality"],
        reversed_keywords=["creative block", "dependence", "smothering", "neglect", "lack"],
        upright_meaning="Divine feminine creative power. Abundance flows naturally. Mother Earth's generosity. Creation through love.",
        reversed_meaning="Creative blocks, self-neglect, or over-giving to depletion. Disconnection from body/nature/pleasure.",
        shadow_work="Where do you deny yourself pleasure or abundance? Where do you smother vs. nurture?",
        soul_lesson="You are worthy of beauty, pleasure, and abundance simply by existing. Creation is your birthright.",
        career_interpretation="Projects come to fruition. Creative ventures favored. Nurture your work and it will grow abundantly.",
        romance_interpretation="Love blossoms. Sensuality and deep connection. Possible pregnancy or birth of new relationship phase.",
        wellness_interpretation="Nourish your body. Connect with nature. Pleasure and self-care are healing, not selfish.",
        family_interpretation="Nurturing family energy. Motherhood themes. Creating abundance and beauty in home environment.",
        self_growth_interpretation="Develop your creative expression. Practice radical self-care. You deserve to flourish.",
        school_interpretation="Learning should be enjoyable. Find beauty in your studies. Creative approaches yield best results.",
        psychology=PsychologicalMapping(
            dbt_skills=["Accumulating Positive Emotions", "Building Mastery", "PLEASE skills (Physical health)"],
            cbt_concepts=["Self-Compassion", "Positive Reinforcement", "Pleasure Principle"],
            mrt_pillars=["Optimism", "Character Strengths (Love, Creativity)", "Relationship Building"],
            core_metaphor="Like building positive experiences in behavioral activation: nurture what you want to grow.",
            growth_prompt="What part of your life is calling for more nourishment? How can you mother yourself today?"
        ),
        astro_sign="Venus",
        astro_planet="Venus"
    ),

    TarotCard(
        number=4,
        name="The Emperor",
        suit=Suit.MAJOR,
        element=Element.SPIRIT,
        upright_keywords=["authority", "structure", "control", "leadership", "stability", "father figure"],
        reversed_keywords=["domination", "rigidity", "lack of discipline", "tyranny", "inflexibility"],
        upright_meaning="Divine masculine power. Structure creates freedom. Wise leadership. Authority tempered by responsibility.",
        reversed_meaning="Authoritarian control, rigidity, or lack of healthy structure. Abuse of power or fear of claiming it.",
        shadow_work="Where do you dominate or submit inappropriately? How do you handle authority?",
        soul_lesson="True power serves others. Structure and discipline are acts of love that create safety for growth.",
        career_interpretation="Leadership opportunities. Create systems and structure. Assert authority appropriately. Build your empire.",
        romance_interpretation="Need for commitment and structure. Traditional partnership. Healthy boundaries create intimacy.",
        wellness_interpretation="Discipline in health routines pays off. Structure your self-care. Consider traditional medicine.",
        family_interpretation="Father figure issues or stepping into provider/protector role. Creating family stability through structure.",
        self_growth_interpretation="Develop your inner authority. Create structures that support your goals. Discipline is self-love.",
        school_interpretation="Structure and discipline in study habits. Respect authority but develop your own expertise.",
        psychology=PsychologicalMapping(
            dbt_skills=["Building Structure", "Setting Boundaries", "Opposite Action (when avoiding responsibility)"],
            cbt_concepts=["Self-Regulation", "Delayed Gratification", "Internal Locus of Control"],
            mrt_pillars=["Self-Regulation", "Character Strengths (Leadership, Integrity)", "Optimism"],
            core_metaphor="Like creating a behavioral plan in therapy: structure and consistency build toward goals.",
            growth_prompt="What area of your life needs more structure? What boundary needs to be set or enforced?"
        ),
        astro_sign="Aries",
        astro_planet="Mars"
    ),

    TarotCard(
        number=5,
        name="The Hierophant",
        suit=Suit.MAJOR,
        element=Element.SPIRIT,
        upright_keywords=["tradition", "conformity", "spiritual wisdom", "institutions", "education", "belief systems"],
        reversed_keywords=["rebellion", "unconventional", "challenging tradition", "restriction", "dogma"],
        upright_meaning="Spiritual tradition and formal education. Learning from established wisdom. Initiation into mysteries through proper channels.",
        reversed_meaning="Questioning dogma, breaking from tradition, or being constrained by outdated beliefs. Need for personal spiritual path.",
        shadow_work="Where do you blindly follow or rebelliously reject? Can you discern useful tradition from limiting dogma?",
        soul_lesson="Truth appears in many forms. Respect lineage while honoring personal gnosis. Balance tradition with innovation.",
        career_interpretation="Formal education, certifications, or working within established systems. Mentorship or becoming a teacher.",
        romance_interpretation="Traditional commitment like marriage. Shared values and beliefs foundation for relationship. Meeting through institutions.",
        wellness_interpretation="Work with established healing modalities. Consider therapy, medical professionals, or time-tested practices.",
        family_interpretation="Family traditions and cultural heritage. Religious or value-based family dynamics. Honoring lineage.",
        self_growth_interpretation="Seek wisdom from teachers and traditions, but filter through your own experience. Create your own philosophy.",
        school_interpretation="Formal education serves you well now. Learn the rules before breaking them. Respect expertise.",
        psychology=PsychologicalMapping(
            dbt_skills=["Dialectics (both/and thinking)", "Interpersonal Effectiveness (respecting authority)"],
            cbt_concepts=["Schema Therapy", "Core Beliefs examination", "Social Learning"],
            mrt_pillars=["Character Strengths (Wisdom, Humility)", "Relationship Building", "Mental Agility"],
            core_metaphor="Like learning evidence-based therapy: respect proven methods while adapting to individual needs.",
            growth_prompt="What traditions or teachings serve you? Which inherited beliefs need questioning?"
        ),
        astro_sign="Taurus",
        astro_planet="Venus"
    ),
]

# Continuing with remaining Major Arcana...
# (This is getting long - I'll create a complete database but showing the structure first)

# ============================================================================
# MINOR ARCANA - WANDS (Fire Element)
# ============================================================================

WANDS_SUIT = [
    TarotCard(
        number=1,
        name="Ace of Wands",
        suit=Suit.WANDS,
        element=Element.FIRE,
        upright_keywords=["inspiration", "creative spark", "new opportunity", "enthusiasm", "potential"],
        reversed_keywords=["delay", "lack of direction", "creative blocks", "missed opportunity"],
        upright_meaning="Pure creative potential. Divine inspiration strikes. A new passionate beginning. The seed of manifestation.",
        reversed_meaning="Creative blocks, false starts, or hesitation preventing action on inspiration. Spark not catching fire.",
        shadow_work="What creative impulses do you dismiss as impractical? Where does fear kill inspiration before it can grow?",
        soul_lesson="Source speaks through inspiration. Act on divine downloads. Your passion is guidance.",
        career_interpretation="New project, job offer, or business idea with great potential. Strike while iron is hot.",
        romance_interpretation="New passionate attraction or renewed spark. Follow your excitement but let it develop.",
        wellness_interpretation="New fitness routine or wellness practice calls to you. Your enthusiasm will fuel success.",
        family_interpretation="New creative project with family or renewed family energy. Shared enthusiasm builds bonds.",
        self_growth_interpretation="A new path for personal development ignites your passion. Follow this thread.",
        school_interpretation="Exciting new subject or approach to learning. Your enthusiasm makes you a quick study.",
        psychology=PsychologicalMapping(
            dbt_skills=["Willingness", "Opposite Action (if avoiding action)", "Building Mastery"],
            cbt_concepts=["Behavioral Activation", "Motivation", "Self-Efficacy"],
            mrt_pillars=["Mental Agility", "Optimism", "Character Strengths (Creativity, Zest)"],
            core_metaphor="Like the first therapy session: full of potential if you commit to the work.",
            growth_prompt="What idea keeps exciting you? What's one action to explore it today?"
        ),
        astro_sign=None,
        astro_planet=None,
        quantum_weight=1.2  # Slightly higher weight for aces - new beginnings
    ),
]

# I'll create the complete database with all 78 cards...
# For now, showing the structure
