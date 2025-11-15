/**
 * TypeScript type definitions for Quantum Tarot
 */

export type Arcana = 'major' | 'minor';

export type Suit = 'wands' | 'cups' | 'swords' | 'pentacles' | null;

export type Rank = 'ace' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' |
                   'page' | 'knight' | 'queen' | 'king' | null;

export type Element = 'fire' | 'water' | 'air' | 'earth' | 'spirit';

export type Modality = 'cardinal' | 'fixed' | 'mutable' | null;

export type Chakra = 'root' | 'sacral' | 'solar_plexus' | 'heart' | 'throat' |
                     'third_eye' | 'crown';

export interface CardKeywords {
  upright: string[];
  reversed: string[];
}

export interface TarotCard {
  id: number;
  name: string;
  arcana: Arcana;
  suit: Suit;
  rank: Rank;
  number: number;
  element: Element;
  modality: Modality;
  astrology: string;
  numerology: number;
  kabbalah: string;
  symbols: string[];
  archetypes: string[];
  themes: string[];
  keywords: CardKeywords;
  jungian: string;
  chakra: Chakra;
  seasonality: string;
  timeframe: string;
  advice: string;
  shadow?: string;
  shadow_work?: string;
  light: string;
  questions: string[];
  description: string;
}

export type SpreadType =
  | 'single_card'
  | 'three_card'
  | 'goal_progress'
  | 'decision_analysis'
  | 'daily_checkin'
  | 'clairvoyant_predictive'
  | 'relationship'
  | 'celtic_cross'
  | 'horseshoe';

export type ReadingType =
  | 'career'
  | 'romance'
  | 'wellness'
  | 'spiritual'
  | 'decision'
  | 'general'
  | 'shadow_work'
  | 'year_ahead';

export interface SpreadPosition {
  position: string;
  meaning: string;
  x?: number;
  y?: number;
  rotation?: number;
  zIndex?: number;
  relatedPositions?: string[];
  colorAccent?: string;
  section?: string;
}

export interface DrawnCard {
  cardIndex: number;
  reversed: boolean;
  position?: string;
}

export interface QuantumReading {
  cards: DrawnCard[];
  quantumSeed: string;
  timestamp: string;
  spreadType: SpreadType;
  intention: string;
}

export interface UserProfile {
  name: string;
  birthday: string;
  pronouns: string;
  createdAt: number;
}

export type TherapeuticFramework = 'DBT' | 'CBT' | 'MRT' | 'Integrative';

export type InterventionStyle = 'directive' | 'exploratory' | 'supportive';

export type CommunicationVoice =
  | 'analytical_guide'
  | 'intuitive_mystic'
  | 'supportive_friend'
  | 'direct_coach'
  | 'gentle_nurturer'
  | 'wise_mentor'
  | 'playful_explorer'
  | 'balanced_sage';

export interface PersonalityProfile {
  userId: string;
  readingType: ReadingType;
  timestamp: number;
  responses: Record<string, any>;
  // Calculated traits (0.0 - 1.0)
  emotionalRegulation: number;
  actionOrientation: number;
  internalExternalLocus: number;
  optimismRealism: number;
  analyticalIntuitive: number;
  riskTolerance: number;
  socialOrientation: number;
  structureFlexibility: number;
  pastFutureFocus: number;
  avoidanceApproach: number;
  // Derived insights
  primaryFramework: TherapeuticFramework;
  interventionStyle: InterventionStyle;
  primaryVoice: CommunicationVoice;
  sunSign?: string;
}

export interface CommunicationProfile {
  primaryVoice: CommunicationVoice;
  sentenceLength: 'short' | 'medium' | 'long';
  metaphorDensity: 'low' | 'medium' | 'high';
  therapeuticExplicitness: 'hidden' | 'subtle' | 'explicit';
  spiritualLanguage: 'minimal' | 'moderate' | 'rich';
  emojiUse: boolean;
  warmthLevel: number;
  directnessLevel: number;
  empowermentVsComfort: number;
}

export interface Reading {
  spreadType: SpreadType;
  spreadName: string;
  readingType: ReadingType;
  timestamp: number;
  userIntention: string;
  commProfile: CommunicationProfile;
  positions: ReadingPosition[];
}

export interface ReadingPosition {
  position: string;
  meaning: string;
  cardIndex: number;
  reversed: boolean;
  quantumSignature: string;
  card: TarotCard;
  interpretation: string;
}

export interface AppConfig {
  version: 'free' | 'premium';
  name: string;
  slug: string;
  bundleId: string;
  features: {
    dailyReadingLimit: number | null;
    unlimitedReadings: boolean;
    allSpreadTypes: boolean;
    themeSelection: boolean;
    readingHistory: boolean;
    advancedInterpretations: boolean;
    exportReadings: boolean;
    availableSpreads: SpreadType[];
    availableReadingTypes: ReadingType[];
  };
  monetization: {
    upgradePrompt: boolean;
    upgradeUrl: string | null;
    upgradePrice: string | null;
  };
}

export interface FuzzyActivation {
  archetypal: number;
  practical: number;
  psychological: number;
  relational: number;
  mystical: number;
}

export interface AgentOutput {
  agent: string;
  interpretation: string;
  confidence: number;
}

export interface MetaAnalysis {
  dominantElement: Element | null;
  arcanaSignificance: 'fated_spiritual' | 'practical_personal' | 'balanced';
  reversalSignificance: 'heavy_blockage' | 'moderate_resistance' | 'some_shadow' | 'flow_state';
  commonArchetypes: Array<{ archetype: string; count: number }>;
  chakraBreakdown: Record<Chakra, number>;
}
