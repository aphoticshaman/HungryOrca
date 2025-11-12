"""
Quantum Tarot - Database Schema
SQLAlchemy models for PostgreSQL/SQLite
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey,
    create_engine, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()


def generate_uuid():
    """Generate UUID for primary keys"""
    return str(uuid.uuid4())


class User(Base):
    """User account and profile"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=True)  # Optional for free users
    birthday = Column(DateTime, nullable=True)  # For astrological integration
    gender_identity = Column(String(50), nullable=True)
    pronouns = Column(String(50), nullable=True)

    # Subscription
    subscription_tier = Column(String(20), default="seeker")  # seeker, mystic
    subscription_expires = Column(DateTime, nullable=True)

    # Computed astrological data
    sun_sign = Column(String(20), nullable=True)
    moon_sign = Column(String(20), nullable=True)
    rising_sign = Column(String(20), nullable=True)

    # Account metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_readings = Column(Integer, default=0)

    # Relationships
    personality_profiles = relationship("PersonalityProfile", back_populates="user", cascade="all, delete-orphan")
    readings = relationship("Reading", back_populates="user", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_subscription', 'subscription_tier'),
    )


class PersonalityProfile(Base):
    """Personality profile from attunement questions"""
    __tablename__ = "personality_profiles"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    reading_type = Column(String(50), nullable=False)  # career, romance, etc.

    # Raw responses (JSON array of question_id: answer)
    responses = Column(JSON, nullable=False)

    # Calculated trait scores (0.0 to 1.0)
    emotional_regulation = Column(Float, default=0.5)
    action_orientation = Column(Float, default=0.5)
    internal_external_locus = Column(Float, default=0.5)
    optimism_realism = Column(Float, default=0.5)
    analytical_intuitive = Column(Float, default=0.5)
    risk_tolerance = Column(Float, default=0.5)
    social_orientation = Column(Float, default=0.5)
    structure_flexibility = Column(Float, default=0.5)
    past_future_focus = Column(Float, default=0.5)
    avoidance_approach = Column(Float, default=0.5)

    # Derived psychological insights
    primary_framework = Column(String(20), nullable=True)  # DBT, CBT, MRT, Integrative
    intervention_style = Column(String(20), nullable=True)  # directive, exploratory, supportive
    communication_voice = Column(String(50), nullable=True)  # analytical_guide, intuitive_mystic, etc.
    aesthetic_profile = Column(String(50), nullable=True)  # minimal_modern, soft_mystical, etc.

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="personality_profiles")

    # Indexes
    __table_args__ = (
        Index('idx_profile_user', 'user_id'),
        Index('idx_profile_type', 'reading_type'),
    )


class Reading(Base):
    """Individual tarot reading session"""
    __tablename__ = "readings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    personality_profile_id = Column(String(36), ForeignKey("personality_profiles.id"), nullable=True)

    # Reading context
    reading_type = Column(String(50), nullable=False)
    spread_type = Column(String(50), nullable=False)  # three_card, celtic_cross, etc.
    user_intention = Column(Text, nullable=True)  # Their question/focus

    # Quantum metadata
    quantum_seed = Column(String(64), nullable=False)  # Hex encoded quantum entropy
    collapse_timestamp = Column(DateTime, default=datetime.utcnow)

    # Reading metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    favorited = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)  # User's journal notes (premium)

    # Relationships
    user = relationship("User", back_populates="readings")
    cards = relationship("ReadingCard", back_populates="reading", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_reading_user', 'user_id'),
        Index('idx_reading_created', 'created_at'),
        Index('idx_reading_type', 'reading_type'),
    )


class ReadingCard(Base):
    """Individual card in a reading"""
    __tablename__ = "reading_cards"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    reading_id = Column(String(36), ForeignKey("readings.id"), nullable=False)

    # Card identification
    card_number = Column(Integer, nullable=False)  # 0-77
    card_name = Column(String(100), nullable=False)
    card_suit = Column(String(20), nullable=False)
    is_reversed = Column(Boolean, default=False)

    # Position in spread
    position_index = Column(Integer, nullable=False)  # 0-based position in spread
    position_name = Column(String(50), nullable=False)  # "Past", "Present", etc.

    # Quantum signature for provenance
    quantum_signature = Column(String(64), nullable=False)

    # Generated interpretation (adapted to user)
    interpretation_text = Column(Text, nullable=False)
    keywords = Column(JSON, nullable=True)  # Array of keywords for this card

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    reading = relationship("Reading", back_populates="cards")

    # Indexes
    __table_args__ = (
        Index('idx_card_reading', 'reading_id'),
        Index('idx_card_position', 'position_index'),
    )


class UsageLog(Base):
    """Track usage for free tier limits"""
    __tablename__ = "usage_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)

    action = Column(String(50), nullable=False)  # "reading_created", "profile_created", etc.
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Additional context
    metadata = Column(JSON, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_usage_user_time', 'user_id', 'timestamp'),
        Index('idx_usage_action', 'action'),
    )


class Subscription(Base):
    """Subscription and payment tracking"""
    __tablename__ = "subscriptions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)

    # Subscription details
    tier = Column(String(20), nullable=False)  # seeker, mystic
    status = Column(String(20), nullable=False)  # active, cancelled, expired, trial

    # Billing
    amount = Column(Float, nullable=True)
    currency = Column(String(3), default="USD")
    billing_period = Column(String(20), nullable=True)  # monthly, yearly

    # Payment provider integration
    stripe_customer_id = Column(String(100), nullable=True)
    stripe_subscription_id = Column(String(100), nullable=True)

    # Dates
    trial_start = Column(DateTime, nullable=True)
    trial_end = Column(DateTime, nullable=True)
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_sub_user', 'user_id'),
        Index('idx_sub_status', 'status'),
        Index('idx_sub_stripe_customer', 'stripe_customer_id'),
    )


# ============================================================================
# Database utility functions
# ============================================================================

def init_db(database_url: str = "sqlite:///quantum_tarot.db"):
    """Initialize database and create all tables"""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()


def drop_all_tables(engine):
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(engine)


# ============================================================================
# Query helpers
# ============================================================================

class DatabaseQueries:
    """Common database queries"""

    def __init__(self, session):
        self.session = session

    def get_user_by_email(self, email: str):
        """Get user by email"""
        return self.session.query(User).filter(User.email == email).first()

    def get_user_by_id(self, user_id: str):
        """Get user by ID"""
        return self.session.query(User).filter(User.id == user_id).first()

    def can_user_read_today(self, user_id: str) -> bool:
        """Check if free user has readings left today (1/day limit)"""
        user = self.get_user_by_id(user_id)

        if user.subscription_tier == "mystic":
            return True  # Unlimited for premium

        # Check readings today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        readings_today = self.session.query(Reading).filter(
            Reading.user_id == user_id,
            Reading.created_at >= today_start
        ).count()

        return readings_today < 1  # Free tier: 1 per day

    def get_user_readings(self, user_id: str, limit: int = 10):
        """Get user's recent readings"""
        return self.session.query(Reading).filter(
            Reading.user_id == user_id
        ).order_by(Reading.created_at.desc()).limit(limit).all()

    def get_reading_with_cards(self, reading_id: str):
        """Get complete reading with all cards"""
        reading = self.session.query(Reading).filter(Reading.id == reading_id).first()
        if reading:
            # Eagerly load cards
            reading.cards = self.session.query(ReadingCard).filter(
                ReadingCard.reading_id == reading_id
            ).order_by(ReadingCard.position_index).all()
        return reading

    def get_latest_personality_profile(self, user_id: str, reading_type: str):
        """Get most recent personality profile for this reading type"""
        return self.session.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.reading_type == reading_type
        ).order_by(PersonalityProfile.created_at.desc()).first()

    def create_user(self, name: str, email: str = None, birthday: datetime = None,
                   gender_identity: str = None, pronouns: str = None):
        """Create new user"""
        user = User(
            name=name,
            email=email,
            birthday=birthday,
            gender_identity=gender_identity,
            pronouns=pronouns
        )
        self.session.add(user)
        self.session.commit()
        return user

    def log_usage(self, user_id: str, action: str, metadata: dict = None):
        """Log user action"""
        log = UsageLog(
            user_id=user_id,
            action=action,
            metadata=metadata
        )
        self.session.add(log)
        self.session.commit()


if __name__ == "__main__":
    # Test database creation
    print("Creating database schema...")
    engine = init_db("sqlite:///test_quantum_tarot.db")
    print("✓ Database created successfully")

    # Test session
    session = get_session(engine)
    queries = DatabaseQueries(session)

    # Create test user
    print("\nCreating test user...")
    user = queries.create_user(
        name="Test User",
        email="test@example.com",
        birthday=datetime(1990, 6, 15),
        pronouns="she/her"
    )
    print(f"✓ User created: {user.id}")

    # Test retrieval
    retrieved = queries.get_user_by_email("test@example.com")
    print(f"✓ User retrieved: {retrieved.name}")

    # Test reading limit
    can_read = queries.can_user_read_today(user.id)
    print(f"✓ Can user read today: {can_read}")

    session.close()
    print("\n✓ All tests passed!")
