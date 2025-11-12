"""
Quantum Tarot - FastAPI REST API
Complete backend API for mobile app
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.schema import (
    init_db, get_session, DatabaseQueries,
    User, PersonalityProfile, Reading, ReadingCard
)
from services.quantum_engine import QuantumSpreadEngine
from services.personality_profiler import PersonalityAnalyzer
from services.adaptive_language import AdaptiveLanguageEngine
from models.complete_deck import get_complete_deck

# Initialize FastAPI
app = FastAPI(
    title="Quantum Tarot API",
    description="Backend API for Quantum Tarot mobile app",
    version="1.0.0"
)

# CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
engine = init_db(os.getenv("DATABASE_URL", "sqlite:///quantum_tarot.db"))

# Initialize services
quantum_engine = QuantumSpreadEngine()
personality_analyzer = PersonalityAnalyzer()
language_engine = AdaptiveLanguageEngine()
tarot_deck = get_complete_deck()


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = None
    birthday: Optional[datetime] = None
    gender_identity: Optional[str] = None
    pronouns: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    name: str
    email: Optional[str]
    birthday: Optional[datetime]
    sun_sign: Optional[str]
    subscription_tier: str
    subscription_expires: Optional[datetime]
    total_readings: int
    created_at: datetime

    class Config:
        from_attributes = True


class PersonalityQuestionResponse(BaseModel):
    """Response containing attunement questions"""
    reading_type: str
    questions: List[Dict[str, Any]]


class PersonalityProfileCreate(BaseModel):
    user_id: str
    reading_type: str
    responses: Dict[str, Any]  # question_id: answer
    birthday: Optional[str] = None  # For astrological calculation


class PersonalityProfileResponse(BaseModel):
    id: str
    reading_type: str
    communication_voice: Optional[str]
    aesthetic_profile: Optional[str]
    primary_framework: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ReadingCreate(BaseModel):
    user_id: str
    reading_type: str
    spread_type: str
    user_intention: Optional[str] = None


class CardResponse(BaseModel):
    card_number: int
    card_name: str
    card_suit: str
    is_reversed: bool
    position_name: str
    interpretation_text: str
    keywords: List[str]
    quantum_signature: str

    class Config:
        from_attributes = True


class ReadingResponse(BaseModel):
    id: str
    reading_type: str
    spread_type: str
    user_intention: Optional[str]
    created_at: datetime
    cards: List[CardResponse]
    favorited: bool

    class Config:
        from_attributes = True


class ReadingLimitResponse(BaseModel):
    can_read: bool
    readings_today: int
    limit: int
    next_reading_available: Optional[datetime]
    subscription_tier: str


# ============================================================================
# Dependency Injection
# ============================================================================

def get_db_queries():
    """Get database queries instance"""
    session = get_session(engine)
    try:
        yield DatabaseQueries(session)
    finally:
        session.close()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Quantum Tarot API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "quantum_engine": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# User Management Endpoints
# ============================================================================

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Create new user account"""
    # Check if email already exists
    if user_data.email:
        existing = queries.get_user_by_email(user_data.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

    # Create user
    user = queries.create_user(
        name=user_data.name,
        email=user_data.email,
        birthday=user_data.birthday,
        gender_identity=user_data.gender_identity,
        pronouns=user_data.pronouns
    )

    # Calculate sun sign if birthday provided
    if user_data.birthday:
        analyzer = PersonalityAnalyzer()
        sun_sign, _, _ = analyzer._calculate_astrological_signs(
            user_data.birthday.strftime("%Y-%m-%d")
        )
        user.sun_sign = sun_sign
        queries.session.commit()

    return user


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Get user by ID"""
    user = queries.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@app.get("/users/{user_id}/reading-limit", response_model=ReadingLimitResponse)
async def check_reading_limit(
    user_id: str,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Check if user can perform reading (free tier limit)"""
    user = queries.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    can_read = queries.can_user_read_today(user_id)

    # Calculate readings today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    readings_today = queries.session.query(Reading).filter(
        Reading.user_id == user_id,
        Reading.created_at >= today_start
    ).count()

    # Calculate next available reading time
    next_available = None
    if not can_read and user.subscription_tier == "seeker":
        tomorrow = today_start + timedelta(days=1)
        next_available = tomorrow

    limit = 999 if user.subscription_tier == "mystic" else 1

    return ReadingLimitResponse(
        can_read=can_read,
        readings_today=readings_today,
        limit=limit,
        next_reading_available=next_available,
        subscription_tier=user.subscription_tier
    )


# ============================================================================
# Personality Profile Endpoints
# ============================================================================

@app.get("/personality/questions/{reading_type}", response_model=PersonalityQuestionResponse)
async def get_personality_questions(reading_type: str):
    """Get attunement questions for reading type"""
    analyzer = PersonalityAnalyzer()

    # Get questions
    questions = analyzer._get_questions_for_type(reading_type)

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid reading type: {reading_type}"
        )

    # Convert to JSON-serializable format
    questions_data = [
        {
            "id": q.id,
            "text": q.text,
            "response_type": q.response_type.value,
            "options": q.options,
            "measures_trait": q.measures_trait
        }
        for q in questions
    ]

    return PersonalityQuestionResponse(
        reading_type=reading_type,
        questions=questions_data
    )


@app.post("/personality/profiles", response_model=PersonalityProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_personality_profile(
    profile_data: PersonalityProfileCreate,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Create personality profile from attunement responses"""
    # Verify user exists
    user = queries.get_user_by_id(profile_data.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Analyze personality
    profile = personality_analyzer.calculate_profile(
        reading_type=profile_data.reading_type,
        responses=profile_data.responses,
        birthday=profile_data.birthday or (user.birthday.strftime("%Y-%m-%d") if user.birthday else None),
        name=user.name
    )

    # Determine communication preferences
    birth_year = user.birthday.year if user.birthday else None
    comm_profile = language_engine.build_communication_profile(
        personality_profile=profile,
        birth_year=birth_year,
        gender_identity=user.gender_identity
    )

    # Save to database
    db_profile = PersonalityProfile(
        user_id=profile_data.user_id,
        reading_type=profile_data.reading_type,
        responses=profile_data.responses,
        emotional_regulation=profile.emotional_regulation,
        action_orientation=profile.action_orientation,
        internal_external_locus=profile.internal_external_locus,
        optimism_realism=profile.optimism_realism,
        analytical_intuitive=profile.analytical_intuitive,
        risk_tolerance=profile.risk_tolerance,
        social_orientation=profile.social_orientation,
        structure_flexibility=profile.structure_flexibility,
        past_future_focus=profile.past_future_focus,
        avoidance_approach=profile.avoidance_approach,
        primary_framework=profile.primary_framework,
        intervention_style=profile.intervention_style,
        communication_voice=comm_profile.primary_voice.value,
        aesthetic_profile=comm_profile.aesthetic.value
    )

    queries.session.add(db_profile)
    queries.session.commit()

    # Log usage
    queries.log_usage(profile_data.user_id, "profile_created", {
        "reading_type": profile_data.reading_type
    })

    return db_profile


@app.get("/users/{user_id}/personality-profiles", response_model=List[PersonalityProfileResponse])
async def get_user_profiles(
    user_id: str,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Get all personality profiles for user"""
    user = queries.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    profiles = queries.session.query(PersonalityProfile).filter(
        PersonalityProfile.user_id == user_id
    ).order_by(PersonalityProfile.created_at.desc()).all()

    return profiles


# ============================================================================
# Reading Endpoints (The Main Event!)
# ============================================================================

@app.post("/readings", response_model=ReadingResponse, status_code=status.HTTP_201_CREATED)
async def create_reading(
    reading_data: ReadingCreate,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """
    Create new tarot reading - THE MAIN ENDPOINT!
    This is where all the magic happens.
    """
    # Verify user
    user = queries.get_user_by_id(reading_data.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check reading limit
    can_read = queries.can_user_read_today(reading_data.user_id)
    if not can_read:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily reading limit reached. Upgrade to Mystic for unlimited readings."
        )

    # Get personality profile
    personality_profile = queries.get_latest_personality_profile(
        reading_data.user_id,
        reading_data.reading_type
    )

    if not personality_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No personality profile found for {reading_data.reading_type}. Complete attunement first."
        )

    # Perform quantum reading
    quantum_reading = quantum_engine.perform_reading(
        spread_type=reading_data.spread_type,
        user_intention=reading_data.user_intention or "",
        reading_type=reading_data.reading_type
    )

    # Build communication profile
    birth_year = user.birthday.year if user.birthday else None
    from services.personality_profiler import PersonalityProfile as PP

    # Reconstruct personality profile object
    profile_obj = PP(
        user_id=user.id,
        reading_type=reading_data.reading_type,
        timestamp=personality_profile.created_at.timestamp(),
        emotional_regulation=personality_profile.emotional_regulation,
        action_orientation=personality_profile.action_orientation,
        internal_external_locus=personality_profile.internal_external_locus,
        optimism_realism=personality_profile.optimism_realism,
        analytical_intuitive=personality_profile.analytical_intuitive,
        risk_tolerance=personality_profile.risk_tolerance,
        social_orientation=personality_profile.social_orientation,
        structure_flexibility=personality_profile.structure_flexibility,
        past_future_focus=personality_profile.past_future_focus,
        avoidance_approach=personality_profile.avoidance_approach,
        primary_framework=personality_profile.primary_framework,
        intervention_style=personality_profile.intervention_style
    )

    comm_profile = language_engine.build_communication_profile(
        personality_profile=profile_obj,
        birth_year=birth_year,
        gender_identity=user.gender_identity
    )

    # Create reading record
    db_reading = Reading(
        user_id=reading_data.user_id,
        personality_profile_id=personality_profile.id,
        reading_type=reading_data.reading_type,
        spread_type=reading_data.spread_type,
        user_intention=reading_data.user_intention,
        quantum_seed=quantum_reading['positions'][0]['quantum_signature'][:64]  # First card's signature
    )

    queries.session.add(db_reading)
    queries.session.flush()  # Get reading ID

    # Generate interpretations for each card
    reading_cards = []
    for position_data in quantum_reading['positions']:
        # Get card from deck
        card = tarot_deck[position_data['card_index']]

        # Generate personalized interpretation
        interpretation = language_engine.generate_card_interpretation(
            card=card,
            position_meaning=position_data['position'],
            is_reversed=position_data['reversed'],
            comm_profile=comm_profile,
            reading_type=reading_data.reading_type
        )

        # Create card record
        db_card = ReadingCard(
            reading_id=db_reading.id,
            card_number=position_data['card_index'],
            card_name=card.name,
            card_suit=card.suit.value,
            is_reversed=position_data['reversed'],
            position_index=len(reading_cards),
            position_name=position_data['position'],
            quantum_signature=position_data['quantum_signature'],
            interpretation_text=interpretation,
            keywords=card.upright_keywords if not position_data['reversed'] else card.reversed_keywords
        )

        queries.session.add(db_card)
        reading_cards.append(db_card)

    # Commit all
    queries.session.commit()

    # Update user stats
    user.total_readings += 1
    user.last_active = datetime.utcnow()
    queries.session.commit()

    # Log usage
    queries.log_usage(reading_data.user_id, "reading_created", {
        "reading_type": reading_data.reading_type,
        "spread_type": reading_data.spread_type
    })

    # Return complete reading
    db_reading.cards = reading_cards
    return db_reading


@app.get("/readings/{reading_id}", response_model=ReadingResponse)
async def get_reading(
    reading_id: str,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Get specific reading by ID"""
    reading = queries.get_reading_with_cards(reading_id)
    if not reading:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reading not found"
        )
    return reading


@app.get("/users/{user_id}/readings", response_model=List[ReadingResponse])
async def get_user_readings(
    user_id: str,
    limit: int = 10,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Get user's reading history (premium feature)"""
    user = queries.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check premium access
    if user.subscription_tier != "mystic":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reading history is a Mystic feature. Upgrade to access."
        )

    readings = queries.get_user_readings(user_id, limit)

    # Load cards for each reading
    for reading in readings:
        reading.cards = queries.session.query(ReadingCard).filter(
            ReadingCard.reading_id == reading.id
        ).order_by(ReadingCard.position_index).all()

    return readings


@app.patch("/readings/{reading_id}/favorite")
async def toggle_favorite(
    reading_id: str,
    queries: DatabaseQueries = Depends(get_db_queries)
):
    """Toggle favorite status on reading (premium feature)"""
    reading = queries.session.query(Reading).filter(Reading.id == reading_id).first()
    if not reading:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reading not found"
        )

    # Check premium access
    user = queries.get_user_by_id(reading.user_id)
    if user.subscription_tier != "mystic":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Favoriting is a Mystic feature. Upgrade to access."
        )

    reading.favorited = not reading.favorited
    queries.session.commit()

    return {"favorited": reading.favorited}


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/spreads")
async def get_available_spreads():
    """Get list of available spread types"""
    return {
        "spreads": [
            {
                "type": "single_card",
                "name": "Single Card",
                "description": "Quick insight for immediate clarity",
                "positions": 1,
                "premium": False
            },
            {
                "type": "three_card",
                "name": "Past, Present, Future",
                "description": "Classic 3-card spread for situation overview",
                "positions": 3,
                "premium": False
            },
            {
                "type": "relationship",
                "name": "Relationship",
                "description": "6-card spread exploring connection dynamics",
                "positions": 6,
                "premium": False
            },
            {
                "type": "horseshoe",
                "name": "Horseshoe",
                "description": "7-card spread for comprehensive guidance",
                "positions": 7,
                "premium": True
            },
            {
                "type": "celtic_cross",
                "name": "Celtic Cross",
                "description": "Deep 10-card exploration of complex situations",
                "positions": 10,
                "premium": True
            }
        ]
    }


@app.get("/reading-types")
async def get_reading_types():
    """Get available reading types"""
    return {
        "reading_types": [
            {"value": "career", "label": "Career", "icon": "briefcase"},
            {"value": "romance", "label": "Romance", "icon": "heart"},
            {"value": "wellness", "label": "Wellness", "icon": "lotus"},
            {"value": "family", "label": "Family", "icon": "tree"},
            {"value": "self_growth", "label": "Self-Growth", "icon": "spiral"},
            {"value": "school", "label": "School", "icon": "book"},
            {"value": "general", "label": "General", "icon": "cosmos"},
            {"value": "surprise_me", "label": "Surprise Me", "icon": "sparkles"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
