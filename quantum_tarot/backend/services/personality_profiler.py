"""
Quantum Tarot - Personality Profiling System
10-question adaptive battery per reading type that goes beyond cold reading
Maps to psychological frameworks (DBT/CBT/MRT) and astrological profiles
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import datetime


class ResponseType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    LIKERT_SCALE = "likert"  # 1-5 scale
    BINARY = "binary"  # Yes/No


@dataclass
class Question:
    """Individual question in personality battery"""
    id: str
    text: str
    response_type: ResponseType
    options: List[str]

    # What this question actually measures (hidden from user)
    measures_trait: str  # e.g., "emotional_regulation", "action_orientation"
    dbt_skill: Optional[str] = None
    cbt_pattern: Optional[str] = None
    mrt_pillar: Optional[str] = None

    # For adaptive questioning
    follow_up_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class PersonalityProfile:
    """Complete psychological profile built from responses"""
    user_id: str
    reading_type: str
    timestamp: float

    # Raw responses
    responses: Dict[str, any] = field(default_factory=dict)

    # Calculated traits (0.0 to 1.0 scales)
    emotional_regulation: float = 0.5
    action_orientation: float = 0.5  # Thinking vs Doing
    internal_external_locus: float = 0.5  # Control belief
    optimism_realism: float = 0.5
    analytical_intuitive: float = 0.5
    risk_tolerance: float = 0.5
    social_orientation: float = 0.5
    structure_flexibility: float = 0.5
    past_future_focus: float = 0.5
    avoidance_approach: float = 0.5  # Coping style

    # Astrological integration
    sun_sign: Optional[str] = None
    moon_sign: Optional[str] = None
    rising_sign: Optional[str] = None

    # Dominant psychological frameworks identified
    primary_framework: Optional[str] = None  # "CBT", "DBT", "MRT"
    intervention_style: Optional[str] = None  # "directive", "exploratory", "supportive"


class QuestionBank:
    """
    Repository of questions for each reading type.
    Questions are designed to seem like normal personality questions
    but actually gather deep psychological data.
    """

    @staticmethod
    def get_career_questions() -> List[Question]:
        """Career reading personality battery"""
        return [
            Question(
                id="career_1",
                text="When facing a difficult work decision, you typically:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Make a quick decision and adjust if needed",
                    "Analyze all options thoroughly before choosing",
                    "Seek advice from trusted colleagues",
                    "Trust your gut feeling"
                ],
                measures_trait="action_orientation",
                cbt_pattern="decision_making_style",
                mrt_pillar="mental_agility"
            ),

            Question(
                id="career_2",
                text="On a scale of 1-5, how much control do you feel over your career path?",
                response_type=ResponseType.LIKERT_SCALE,
                options=["1 - Very little", "2 - Some", "3 - Moderate", "4 - Quite a bit", "5 - Complete control"],
                measures_trait="internal_external_locus",
                cbt_pattern="locus_of_control",
                mrt_pillar="optimism"
            ),

            Question(
                id="career_3",
                text="When you receive critical feedback at work, you:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Feel hurt but try to learn from it",
                    "Immediately start planning improvements",
                    "Question the feedback giver's motives",
                    "Need time alone to process your emotions"
                ],
                measures_trait="emotional_regulation",
                dbt_skill="distress_tolerance",
                cbt_pattern="cognitive_reappraisal"
            ),

            Question(
                id="career_4",
                text="Your ideal work environment has:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Clear structure and defined expectations",
                    "Flexibility to create your own approach",
                    "Mix of both structure and freedom",
                    "Constantly changing challenges"
                ],
                measures_trait="structure_flexibility",
                mrt_pillar="self_regulation"
            ),

            Question(
                id="career_5",
                text="When considering a career change, you think most about:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "What hasn't worked in the past",
                    "Exciting possibilities ahead",
                    "Current skills and resources",
                    "What others expect of you"
                ],
                measures_trait="past_future_focus",
                cbt_pattern="temporal_orientation",
                mrt_pillar="optimism"
            ),

            Question(
                id="career_6",
                text="In team projects, you naturally tend to:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Take charge and organize",
                    "Support others' ideas",
                    "Contribute ideas but let others lead",
                    "Work independently on your part"
                ],
                measures_trait="social_orientation",
                mrt_pillar="relationship_building"
            ),

            Question(
                id="career_7",
                text="True or False: I often worry about making the wrong career move.",
                response_type=ResponseType.BINARY,
                options=["True", "False"],
                measures_trait="risk_tolerance",
                cbt_pattern="catastrophizing",
                dbt_skill="mindfulness"
            ),

            Question(
                id="career_8",
                text="When facing a career challenge, your first instinct is to:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Research and gather information",
                    "Take immediate action",
                    "Talk it through with someone",
                    "Reflect and meditate on it"
                ],
                measures_trait="analytical_intuitive",
                cbt_pattern="coping_style"
            ),

            Question(
                id="career_9",
                text="Rate your agreement: 'Success comes from hard work, not luck.'",
                response_type=ResponseType.LIKERT_SCALE,
                options=["1 - Strongly disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly agree"],
                measures_trait="internal_external_locus",
                cbt_pattern="attribution_style",
                mrt_pillar="character_strengths"
            ),

            Question(
                id="career_10",
                text="When work stress builds up, you cope by:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Pushing through and working harder",
                    "Taking breaks for self-care",
                    "Venting to friends or family",
                    "Avoiding thinking about it"
                ],
                measures_trait="avoidance_approach",
                dbt_skill="distress_tolerance",
                mrt_pillar="self_regulation"
            ),
        ]

    @staticmethod
    def get_romance_questions() -> List[Question]:
        """Romance reading personality battery"""
        return [
            Question(
                id="romance_1",
                text="In relationships, you value most:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Emotional intimacy and deep connection",
                    "Excitement and passion",
                    "Stability and commitment",
                    "Independence within togetherness"
                ],
                measures_trait="social_orientation",
                dbt_skill="interpersonal_effectiveness"
            ),

            Question(
                id="romance_2",
                text="When conflict arises with a partner, you typically:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Address it immediately and directly",
                    "Need space before discussing",
                    "Try to smooth things over quickly",
                    "Analyze what went wrong first"
                ],
                measures_trait="emotional_regulation",
                dbt_skill="emotion_regulation",
                cbt_pattern="conflict_resolution"
            ),

            Question(
                id="romance_3",
                text="Rate: 'I trust my instincts about potential partners.'",
                response_type=ResponseType.LIKERT_SCALE,
                options=["1 - Not at all", "2 - Slightly", "3 - Moderately", "4 - Quite a bit", "5 - Completely"],
                measures_trait="analytical_intuitive",
                cbt_pattern="self_trust"
            ),

            Question(
                id="romance_4",
                text="You feel most loved when your partner:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Tells you verbally",
                    "Shows physical affection",
                    "Does helpful things for you",
                    "Spends quality time with you"
                ],
                measures_trait="social_orientation",
                dbt_skill="interpersonal_effectiveness",
                measures_trait="love_language_primary"
            ),

            Question(
                id="romance_5",
                text="In past relationships, you've struggled most with:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Communication breakdowns",
                    "Trust issues",
                    "Different life goals",
                    "Emotional unavailability (yours or theirs)"
                ],
                measures_trait="past_future_focus",
                cbt_pattern="relationship_patterns",
                dbt_skill="mindfulness"
            ),

            Question(
                id="romance_6",
                text="True or False: I often sacrifice my needs for relationship harmony.",
                response_type=ResponseType.BINARY,
                options=["True", "False"],
                measures_trait="internal_external_locus",
                dbt_skill="interpersonal_effectiveness",
                cbt_pattern="boundaries"
            ),

            Question(
                id="romance_7",
                text="When dating, you prefer to:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Take things slowly and see what develops",
                    "Know quickly if there's potential",
                    "Let the other person set the pace",
                    "Have clear milestones and timeline"
                ],
                measures_trait="action_orientation",
                cbt_pattern="relationship_pacing"
            ),

            Question(
                id="romance_8",
                text="Rate: 'I'm optimistic about finding/maintaining lasting love.'",
                response_type=ResponseType.LIKERT_SCALE,
                options=["1 - Very pessimistic", "2 - Somewhat pessimistic", "3 - Neutral", "4 - Somewhat optimistic", "5 - Very optimistic"],
                measures_trait="optimism_realism",
                mrt_pillar="optimism",
                cbt_pattern="relationship_schemas"
            ),

            Question(
                id="romance_9",
                text="You're most attracted to partners who are:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Emotionally expressive and open",
                    "Intellectual and thought-provoking",
                    "Adventurous and spontaneous",
                    "Steady and reliable"
                ],
                measures_trait="structure_flexibility",
                measures_trait="attraction_pattern"
            ),

            Question(
                id="romance_10",
                text="When single, you:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Actively seek partnership",
                    "Focus on yourself and let love find you",
                    "Feel incomplete without a partner",
                    "Enjoy the freedom and aren't in a rush"
                ],
                measures_trait="avoidance_approach",
                cbt_pattern="relationship_beliefs",
                mrt_pillar="character_strengths"
            ),
        ]

    @staticmethod
    def get_wellness_questions() -> List[Question]:
        """Wellness reading personality battery"""
        return [
            Question(
                id="wellness_1",
                text="When you feel physically unwell, you first:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Research symptoms and possible causes",
                    "Rest and let your body heal naturally",
                    "Seek professional medical advice",
                    "Try alternative/holistic remedies"
                ],
                measures_trait="analytical_intuitive",
                cbt_pattern="health_beliefs"
            ),

            Question(
                id="wellness_2",
                text="Rate: 'My mental health significantly impacts my physical health.'",
                response_type=ResponseType.LIKERT_SCALE,
                options=["1 - Not at all", "2 - Slightly", "3 - Moderately", "4 - Significantly", "5 - Completely"],
                measures_trait="emotional_regulation",
                dbt_skill="mindfulness",
                cbt_pattern="mind_body_connection"
            ),

            Question(
                id="wellness_3",
                text="Your self-care routine is best described as:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Consistent and structured",
                    "Intuitive based on what I need",
                    "Something I struggle to maintain",
                    "Non-existent - I put others first"
                ],
                measures_trait="structure_flexibility",
                dbt_skill="please_skills",
                mrt_pillar="self_regulation"
            ),

            Question(
                id="wellness_4",
                text="When stressed, your body tells you through:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Tension/pain (headaches, back pain, etc.)",
                    "Digestive issues",
                    "Sleep disturbances",
                    "I don't notice physical signs"
                ],
                measures_trait="emotional_regulation",
                dbt_skill="mindfulness",
                cbt_pattern="interoceptive_awareness"
            ),

            Question(
                id="wellness_5",
                text="True or False: I often ignore health warning signs until they become serious.",
                response_type=ResponseType.BINARY,
                options=["True", "False"],
                measures_trait="avoidance_approach",
                cbt_pattern="health_avoidance",
                mrt_pillar="self_awareness"
            ),

            Question(
                id="wellness_6",
                text="Your relationship with exercise is:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "I love it and do it regularly",
                    "I know I should but struggle with consistency",
                    "I prefer gentle movement like yoga or walking",
                    "Physical limitations prevent regular exercise"
                ],
                measures_trait="action_orientation",
                dbt_skill="building_mastery",
                cbt_pattern="behavioral_activation"
            ),

            Question(
                id="wellness_7",
                text="Rate your agreement: 'I'm in tune with what my body needs.'",
                response_type=ResponseType.LIKERT_SCALE,
                options=["1 - Strongly disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly agree"],
                measures_trait="analytical_intuitive",
                dbt_skill="mindfulness",
                cbt_pattern="body_awareness"
            ),

            Question(
                id="wellness_8",
                text="When making health decisions, you most trust:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Medical professionals and research",
                    "Your own intuition and body wisdom",
                    "Alternative practitioners and natural approaches",
                    "Combination of conventional and alternative"
                ],
                measures_trait="internal_external_locus",
                cbt_pattern="health_locus_control"
            ),

            Question(
                id="wellness_9",
                text="Your biggest wellness challenge is:",
                response_type=ResponseType.MULTIPLE_CHOICE,
                options=[
                    "Managing stress and anxiety",
                    "Physical health issues or chronic pain",
                    "Motivation and consistency",
                    "Balancing wellness with other responsibilities"
                ],
                measures_trait="emotional_regulation",
                measures_trait="primary_wellness_barrier"
            ),

            Question(
                id="wellness_10",
                text="Rate: 'I believe I can significantly improve my health through my choices.'",
                response_type=ResponseType.LIKERT_SCALE,
                options=["1 - Not at all", "2 - Slightly", "3 - Moderately", "4 - Quite a bit", "5 - Completely"],
                measures_trait="optimism_realism",
                mrt_pillar="optimism",
                cbt_pattern="self_efficacy"
            ),
        ]

    # Add more question sets for other reading types...
    @staticmethod
    def get_family_questions() -> List[Question]:
        """Family reading questions - will implement"""
        # TODO: Implement remaining reading types
        pass

    @staticmethod
    def get_self_growth_questions() -> List[Question]:
        """Self-growth reading questions - will implement"""
        pass


class PersonalityAnalyzer:
    """
    Analyzes responses to build comprehensive psychological profile.
    Maps responses to trait scores and identifies intervention style.
    """

    def __init__(self):
        self.question_bank = QuestionBank()

    def calculate_profile(
        self,
        reading_type: str,
        responses: Dict[str, any],
        birthday: Optional[str] = None,
        name: Optional[str] = None
    ) -> PersonalityProfile:
        """
        Build PersonalityProfile from question responses.

        Args:
            reading_type: Type of reading (career, romance, etc.)
            responses: Dict mapping question_id to user's answer
            birthday: User's birthday for astrological integration
            name: User's name for personalization

        Returns:
            Complete PersonalityProfile with calculated traits
        """
        profile = PersonalityProfile(
            user_id=name or "anonymous",
            reading_type=reading_type,
            timestamp=datetime.datetime.now().timestamp(),
            responses=responses
        )

        # Get questions for this reading type
        questions = self._get_questions_for_type(reading_type)

        # Calculate each trait from responses
        trait_scores = {
            "emotional_regulation": [],
            "action_orientation": [],
            "internal_external_locus": [],
            "optimism_realism": [],
            "analytical_intuitive": [],
            "risk_tolerance": [],
            "social_orientation": [],
            "structure_flexibility": [],
            "past_future_focus": [],
            "avoidance_approach": []
        }

        # Process each response
        for question in questions:
            if question.id not in responses:
                continue

            response = responses[question.id]
            trait = question.measures_trait

            if trait not in trait_scores:
                continue

            # Score based on response type
            if question.response_type == ResponseType.LIKERT_SCALE:
                # Likert: extract numeric value (1-5) -> normalize to 0-1
                score = self._extract_likert_score(response) / 5.0
                trait_scores[trait].append(score)

            elif question.response_type == ResponseType.MULTIPLE_CHOICE:
                # Map specific answers to trait scores
                score = self._score_multiple_choice(question, response)
                if score is not None:
                    trait_scores[trait].append(score)

            elif question.response_type == ResponseType.BINARY:
                # True/False -> 0 or 1
                score = 1.0 if response.lower() == "true" else 0.0
                # Some traits reverse scoring
                if "worry" in question.text.lower() or "sacrifice" in question.text.lower():
                    score = 1.0 - score  # Reverse score
                trait_scores[trait].append(score)

        # Average scores for each trait
        for trait, scores in trait_scores.items():
            if scores:
                setattr(profile, trait, sum(scores) / len(scores))

        # Identify primary framework and intervention style
        profile.primary_framework = self._identify_framework(profile)
        profile.intervention_style = self._identify_intervention_style(profile)

        # Integrate astrological data if birthday provided
        if birthday:
            profile.sun_sign, profile.moon_sign, profile.rising_sign = \
                self._calculate_astrological_signs(birthday)

        return profile

    def _get_questions_for_type(self, reading_type: str) -> List[Question]:
        """Get question set for reading type"""
        if reading_type == "career":
            return self.question_bank.get_career_questions()
        elif reading_type == "romance":
            return self.question_bank.get_romance_questions()
        elif reading_type == "wellness":
            return self.question_bank.get_wellness_questions()
        # TODO: Add other types
        return []

    def _extract_likert_score(self, response: str) -> int:
        """Extract numeric score from Likert response"""
        # Response format: "1 - Not at all" -> extract "1"
        try:
            return int(response.split("-")[0].strip())
        except:
            return 3  # Default to middle

    def _score_multiple_choice(self, question: Question, response: str) -> Optional[float]:
        """
        Score multiple choice based on what it measures.
        Returns 0.0 to 1.0 score for the trait.
        """
        # This is contextual - different scoring per question
        # For now, use position in options as rough guide
        try:
            index = question.options.index(response)
            # Normalize to 0-1 range
            return index / (len(question.options) - 1)
        except ValueError:
            return None

    def _identify_framework(self, profile: PersonalityProfile) -> str:
        """
        Identify primary therapeutic framework that would resonate.

        DBT: High emotional dysregulation, relationship issues
        CBT: Analytical style, thought-focused
        MRT: Action-oriented, resilience focus
        """
        # Weight different factors
        if profile.emotional_regulation < 0.4 and profile.social_orientation > 0.6:
            return "DBT"  # Emotion regulation + interpersonal focus
        elif profile.analytical_intuitive < 0.4:  # More analytical
            return "CBT"  # Cognitive restructuring
        elif profile.action_orientation > 0.6 and profile.optimism_realism > 0.5:
            return "MRT"  # Action + resilience
        else:
            return "Integrative"  # Mix approaches

    def _identify_intervention_style(self, profile: PersonalityProfile) -> str:
        """Identify how to deliver the reading"""
        if profile.structure_flexibility < 0.4:
            return "directive"  # Clear guidance needed
        elif profile.analytical_intuitive < 0.4:
            return "exploratory"  # Questions, self-discovery
        else:
            return "supportive"  # Validate, encourage

    def _calculate_astrological_signs(self, birthday: str) -> tuple:
        """
        Calculate sun, moon, and rising signs from birthday.

        NOTE: Proper calculation requires birth time and location for moon/rising.
        For MVP, we'll just calculate sun sign from date.
        """
        try:
            birth_date = datetime.datetime.strptime(birthday, "%Y-%m-%d")
            month = birth_date.month
            day = birth_date.day

            # Sun sign calculation
            sun_sign = self._get_sun_sign(month, day)

            # For moon and rising, we'd need birth time and location
            # For now, return None - can add full calculation later
            return sun_sign, None, None
        except:
            return None, None, None

    def _get_sun_sign(self, month: int, day: int) -> str:
        """Calculate sun sign from month and day"""
        if (month == 3 and day >= 21) or (month == 4 and day <= 19):
            return "Aries"
        elif (month == 4 and day >= 20) or (month == 5 and day <= 20):
            return "Taurus"
        elif (month == 5 and day >= 21) or (month == 6 and day <= 20):
            return "Gemini"
        elif (month == 6 and day >= 21) or (month == 7 and day <= 22):
            return "Cancer"
        elif (month == 7 and day >= 23) or (month == 8 and day <= 22):
            return "Leo"
        elif (month == 8 and day >= 23) or (month == 9 and day <= 22):
            return "Virgo"
        elif (month == 9 and day >= 23) or (month == 10 and day <= 22):
            return "Libra"
        elif (month == 10 and day >= 23) or (month == 11 and day <= 21):
            return "Scorpio"
        elif (month == 11 and day >= 22) or (month == 12 and day <= 21):
            return "Sagittarius"
        elif (month == 12 and day >= 22) or (month == 1 and day <= 19):
            return "Capricorn"
        elif (month == 1 and day >= 20) or (month == 2 and day <= 18):
            return "Aquarius"
        else:  # Feb 19 - Mar 20
            return "Pisces"


if __name__ == "__main__":
    # Test the profiler
    analyzer = PersonalityAnalyzer()

    # Sample responses
    sample_responses = {
        "career_1": "Analyze all options thoroughly before choosing",
        "career_2": "4 - Quite a bit",
        "career_3": "Immediately start planning improvements",
        "career_4": "Mix of both structure and freedom",
        "career_5": "Exciting possibilities ahead",
        "career_6": "Take charge and organize",
        "career_7": "False",
        "career_8": "Research and gather information",
        "career_9": "5 - Strongly agree",
        "career_10": "Taking breaks for self-care",
    }

    profile = analyzer.calculate_profile(
        reading_type="career",
        responses=sample_responses,
        birthday="1990-06-15",
        name="Test User"
    )

    print("=== Personality Profile ===\n")
    print(f"User: {profile.user_id}")
    print(f"Reading Type: {profile.reading_type}")
    print(f"Sun Sign: {profile.sun_sign}\n")
    print(f"Primary Framework: {profile.primary_framework}")
    print(f"Intervention Style: {profile.intervention_style}\n")
    print("Trait Scores (0-1 scale):")
    print(f"  Emotional Regulation: {profile.emotional_regulation:.2f}")
    print(f"  Action Orientation: {profile.action_orientation:.2f}")
    print(f"  Internal Locus: {profile.internal_external_locus:.2f}")
    print(f"  Optimism: {profile.optimism_realism:.2f}")
    print(f"  Analytical: {profile.analytical_intuitive:.2f}")
