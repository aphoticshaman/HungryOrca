/**
 * LunatiQ AGI Engine Test Harness
 * ================================
 *
 * Comprehensive testing of all LunatiQ components:
 * 1. Fuzzy logic primitives
 * 2. Multi-modal analyzers
 * 3. Fuzzy orchestrator
 * 4. Interpretation agents
 * 5. Ensemble blender
 * 6. End-to-end integration
 */

import { LunatiQOrchestrator } from '../src/services/fuzzyOrchestrator.js';
import {
  ArchetypalAgent,
  PracticalAgent,
  PsychologicalAgent,
  RelationalAgent,
  MysticalAgent
} from '../src/services/interpretationAgents.js';
import { LunatiQEngine } from '../src/services/lunatiQEngine.js';
import { AdaptiveLanguageEngine } from '../src/services/adaptiveLanguage.js';
import { getCardByIndex } from '../src/data/tarotLoader.js';

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: Fuzzy Logic Primitives
// ═══════════════════════════════════════════════════════════════════════════════

function testFuzzyPrimitives() {
  console.log('\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  TEST 1: Fuzzy Logic Primitives                              ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  const orchestrator = new LunatiQOrchestrator();

  // Test fuzzy variable setup
  console.log('✓ Orchestrator instantiated');
  console.log(`✓ Fuzzy variables created: ${Object.keys(orchestrator).filter(k => k.includes('Intensity') || k.includes('Resonance')).length}`);
  console.log(`✓ Fuzzy rules loaded: ${orchestrator.rules.length}`);

  // Test fuzzification
  const testValue = 0.7;
  const fuzzified = orchestrator.archetypeIntensity.fuzzify(testValue);
  console.log(`\nFuzzification test (input: ${testValue}):`);
  console.log(`  low: ${fuzzified.low.toFixed(3)}`);
  console.log(`  medium: ${fuzzified.medium.toFixed(3)}`);
  console.log(`  high: ${fuzzified.high.toFixed(3)}`);

  const expectedHigh = fuzzified.high > 0.5;
  console.log(`\n${expectedHigh ? '✓' : '✗'} High activation detected correctly`);

  return expectedHigh;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: Multi-Modal Analyzers
// ═══════════════════════════════════════════════════════════════════════════════

function testAnalyzers() {
  console.log('\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  TEST 2: Multi-Modal Analyzers                               ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  // Create sample spread: The Fool, The Magician, The Empress
  const cards = [
    getCardByIndex(0),  // The Fool
    getCardByIndex(1),  // The Magician
    getCardByIndex(3)   // The Empress
  ];

  const positions = [
    { position: 'Past', cardIndex: 0, reversed: false },
    { position: 'Present', cardIndex: 1, reversed: false },
    { position: 'Future', cardIndex: 3, reversed: true }
  ];

  const orchestrator = new LunatiQOrchestrator();

  // Test SymbolicAnalyzer
  console.log('Testing SymbolicAnalyzer...');
  const symbolic = orchestrator.symbolicAnalyzer.analyze(cards);
  console.log(`  archetypeIntensity: ${symbolic.archetypeIntensity.toFixed(3)}`);
  console.log(`  suitDiversity: ${symbolic.suitDiversity.toFixed(3)}`);
  console.log(`  narrativeFlow: ${symbolic.narrativeFlow.toFixed(3)}`);

  const symbolicPass = symbolic.archetypeIntensity === 1.0; // All major arcana
  console.log(`${symbolicPass ? '✓' : '✗'} Archetype intensity correct (expected 1.0 for all major arcana)`);

  // Test RelationalAnalyzer
  console.log('\nTesting RelationalAnalyzer...');
  const relational = orchestrator.relationalAnalyzer.analyze(cards, positions);
  console.log(`  reversalTension: ${relational.reversalTension.toFixed(3)}`);
  console.log(`  positionCoherence: ${relational.positionCoherence.toFixed(3)}`);
  console.log(`  elementalBalance: ${relational.elementalBalance.toFixed(3)}`);

  const relationalPass = relational.reversalTension === 0.7; // Has reversal
  console.log(`${relationalPass ? '✓' : '✗'} Reversal tension correct`);

  // Test EnergeticAnalyzer
  console.log('\nTesting EnergeticAnalyzer...');
  const userProfile = {
    analytical: 0.3,
    mystical: 0.8,
    emotionalRegulation: 0.5
  };
  const intention = "What do I need to know about my career transition?";

  const energetic = orchestrator.energeticAnalyzer.analyze(cards, userProfile, intention);
  console.log(`  emotionalIntensity: ${energetic.emotionalIntensity.toFixed(3)}`);
  console.log(`  userResonance: ${energetic.userResonance.toFixed(3)}`);
  console.log(`  intentionAlignment: ${energetic.intentionAlignment.toFixed(3)}`);

  const energeticPass = energetic.emotionalIntensity > 0; // Should detect Tower
  console.log(`${energeticPass ? '✓' : '✗'} Emotional intensity detected`);

  return symbolicPass && relationalPass && energeticPass;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3: Fuzzy Orchestrator Activation Computation
// ═══════════════════════════════════════════════════════════════════════════════

function testOrchestratorActivations() {
  console.log('\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  TEST 3: Fuzzy Orchestrator Activation Computation           ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  const orchestrator = new LunatiQOrchestrator();

  // High archetype, high emotional intensity spread
  const cards = [
    getCardByIndex(0), // The Fool
    getCardByIndex(1), // The Magician
    getCardByIndex(3)  // The Empress
  ];

  const positions = [
    { position: 'Past', cardIndex: 0, reversed: false },
    { position: 'Present', cardIndex: 1, reversed: false },
    { position: 'Future', cardIndex: 3, reversed: false }
  ];

  const userProfile = {
    analyticalIntuitive: 0.2,
    emotionalRegulation: 0.3,
    actionOrientation: 0.7,
    optimismRealism: 0.5,
    riskTolerance: 0.5,
    structureFlexibility: 0.5,
    internalExternalLocus: 0.6,
    primaryFramework: 'CBT',
    analytical: 0.8,
    mystical: 0.2
  };

  const intention = "Why does everything feel like it's falling apart?";

  const result = orchestrator.computeActivations(cards, positions, userProfile, intention);

  console.log('Activation Levels:');
  console.log(`  archetypal: ${result.activations.archetypal.toFixed(3)}`);
  console.log(`  practical: ${result.activations.practical.toFixed(3)}`);
  console.log(`  psychological: ${result.activations.psychological.toFixed(3)}`);
  console.log(`  relational: ${result.activations.relational.toFixed(3)}`);
  console.log(`  mystical: ${result.activations.mystical.toFixed(3)}`);

  const pass = result.activations.archetypal > 0.7; // High archetype cards should activate archetypal mode
  console.log(`\n${pass ? '✓' : '✗'} Major arcana spread correctly activates archetypal mode`);

  return pass;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 4: Interpretation Agents
// ═══════════════════════════════════════════════════════════════════════════════

function testInterpretationAgents() {
  console.log('\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  TEST 4: Interpretation Agents                                ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  const card = getCardByIndex(2); // The High Priestess
  const position = 'Present';
  const isReversed = false;

  const context = {
    intention: "Should I quit my job?",
    profile: {
      analyticalIntuitive: 0.5,
      emotionalRegulation: 0.6
    },
    cards: [card],
    features: {}
  };

  // Test ArchetypalAgent
  console.log('Testing ArchetypalAgent (high activation)...');
  const archetypalAgent = new ArchetypalAgent();
  const archetypalOutput = archetypalAgent.generateInterpretation(
    card,
    position,
    isReversed,
    0.85,
    context
  );
  console.log(`Output length: ${archetypalOutput ? archetypalOutput.length : 0} chars`);
  console.log(`Contains "Odin": ${archetypalOutput && archetypalOutput.includes('Odin')}`);
  const archetypalPass = archetypalOutput && archetypalOutput.length > 50;
  console.log(`${archetypalPass ? '✓' : '✗'} Archetypal agent generates substantial output`);

  // Test PracticalAgent
  console.log('\nTesting PracticalAgent (high activation)...');
  const practicalAgent = new PracticalAgent();
  const practicalOutput = practicalAgent.generateInterpretation(
    card,
    position,
    isReversed,
    0.80,
    context
  );
  console.log(`Output length: ${practicalOutput ? practicalOutput.length : 0} chars`);
  const practicalPass = practicalOutput && practicalOutput.length > 20;
  console.log(`${practicalPass ? '✓' : '✗'} Practical agent generates output`);

  // Test PsychologicalAgent
  console.log('\nTesting PsychologicalAgent (high activation)...');
  const psychologicalAgent = new PsychologicalAgent();
  const psychologicalOutput = psychologicalAgent.generateInterpretation(
    card,
    position,
    isReversed,
    0.75,
    context
  );
  console.log(`Output length: ${psychologicalOutput ? psychologicalOutput.length : 0} chars`);
  console.log(`Contains "Radical Acceptance": ${psychologicalOutput && psychologicalOutput.includes('Radical Acceptance')}`);
  const psychologicalPass = psychologicalOutput && psychologicalOutput.length > 20;
  console.log(`${psychologicalPass ? '✓' : '✗'} Psychological agent generates output`);

  // Test activation threshold (low activation should return null)
  console.log('\nTesting activation threshold (0.2 - should return null)...');
  const lowOutput = archetypalAgent.generateInterpretation(
    card,
    position,
    isReversed,
    0.2,
    context
  );
  const thresholdPass = lowOutput === null;
  console.log(`${thresholdPass ? '✓' : '✗'} Low activation correctly returns null`);

  return archetypalPass && practicalPass && psychologicalPass && thresholdPass;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 5: Ensemble Blender
// ═══════════════════════════════════════════════════════════════════════════════

function testEnsembleBlender() {
  console.log('\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  TEST 5: Ensemble Blender                                     ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  const engine = new LunatiQEngine();
  const card = getCardByIndex(0); // The Fool

  const agentOutputs = {
    archetypal: "The Fool emerges as the Innocent Wanderer, a primal force.",
    practical: "Apply for that unconventional role this week.",
    psychological: "You're in a growth mindset—embracing uncertainty as possibility.",
    relational: null,
    mystical: "Clear your crown chakra. Burn sage in doorways."
  };

  const activations = {
    archetypal: 0.75,
    practical: 0.85,
    psychological: 0.60,
    relational: 0.25,
    mystical: 0.40
  };

  // Test analytical voice (should prioritize practical + psychological)
  console.log('Testing Analytical Voice blend...');
  const analyticalProfile = {
    primaryVoice: 'analytical_guide',
    emojiUse: false
  };
  const analyticalBlend = engine.blender.blendInterpretations(
    agentOutputs,
    activations,
    analyticalProfile,
    card,
    'Present'
  );
  console.log(`Output: ${analyticalBlend.substring(0, 100)}...`);
  const analyticalPass = analyticalBlend.includes('Apply for') && analyticalBlend.includes('growth mindset');
  console.log(`${analyticalPass ? '✓' : '✗'} Analytical voice prioritizes practical + psychological`);

  // Test intuitive mystic voice (should prioritize archetypal + mystical)
  console.log('\nTesting Intuitive Mystic Voice blend...');
  const mysticProfile = {
    primaryVoice: 'intuitive_mystic',
    emojiUse: true
  };
  const mysticBlend = engine.blender.blendInterpretations(
    agentOutputs,
    activations,
    mysticProfile,
    card,
    'Present'
  );
  console.log(`Output: ${mysticBlend.substring(0, 100)}...`);
  const mysticPass = mysticBlend.includes('Innocent Wanderer') && mysticBlend.includes('chakra');
  console.log(`${mysticPass ? '✓' : '✗'} Mystic voice prioritizes archetypal + mystical`);

  // Test direct coach voice (should use only practical)
  console.log('\nTesting Direct Coach Voice blend...');
  const coachProfile = {
    primaryVoice: 'direct_coach',
    emojiUse: false
  };
  const coachBlend = engine.blender.blendInterpretations(
    agentOutputs,
    activations,
    coachProfile,
    card,
    'Present'
  );
  console.log(`Output: ${coachBlend}`);
  const coachPass = coachBlend.includes('Apply for') && !coachBlend.includes('chakra');
  console.log(`${coachPass ? '✓' : '✗'} Coach voice uses only practical guidance`);

  return analyticalPass && mysticPass && coachPass;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 6: End-to-End Integration
// ═══════════════════════════════════════════════════════════════════════════════

function testEndToEnd() {
  console.log('\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  TEST 6: End-to-End Integration Test                         ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  const engine = new LunatiQEngine();

  // Simulate a 3-card reading
  const cards = [
    getCardByIndex(0), // The Fool
    getCardByIndex(2), // The High Priestess
    getCardByIndex(4)  // The Emperor
  ];

  const positions = [
    { position: 'Past', cardIndex: 0, reversed: false },
    { position: 'Present', cardIndex: 2, reversed: false },
    { position: 'Future', cardIndex: 4, reversed: false }
  ];

  const userProfile = {
    analyticalIntuitive: 0.7,
    emotionalRegulation: 0.5,
    actionOrientation: 0.6,
    optimismRealism: 0.6,
    riskTolerance: 0.7,
    structureFlexibility: 0.6,
    internalExternalLocus: 0.7,
    primaryFramework: 'DBT',
    analytical: 0.3,
    mystical: 0.7
  };

  const intention = "What do I need to know about my spiritual journey?";

  const commProfile = AdaptiveLanguageEngine.buildCommunicationProfile(
    userProfile,
    1995
  );

  console.log(`Communication Profile: ${commProfile.primaryVoice}`);
  console.log(`Generating ${cards.length}-card spread interpretation...`);

  try {
    const interpretedCards = engine.generateSpreadInterpretation(
      cards,
      positions,
      userProfile,
      intention,
      commProfile
    );

    console.log(`\n✓ Generated ${interpretedCards.length} interpretations`);

    interpretedCards.forEach((interpreted, i) => {
      console.log(`\n--- Card ${i + 1}: ${interpreted.card.name} (${interpreted.position}) ---`);
      console.log(`Interpretation length: ${interpreted.interpretation.length} chars`);
      console.log(`First 150 chars: ${interpreted.interpretation.substring(0, 150)}...`);
    });

    const pass = interpretedCards.length === 3 &&
                 interpretedCards.every(c => c.interpretation && c.interpretation.length > 50);

    console.log(`\n${pass ? '✓' : '✗'} All cards have substantial interpretations`);

    return pass;

  } catch (error) {
    console.error('\n✗ End-to-end test FAILED with error:');
    console.error(error);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUN ALL TESTS
// ═══════════════════════════════════════════════════════════════════════════════

export function runAllTests() {
  console.log('\n');
  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║                                                               ║');
  console.log('║          LUNATIQ AGI ENGINE - TEST SUITE                      ║');
  console.log('║                                                               ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝');

  const results = {
    fuzzyPrimitives: false,
    analyzers: false,
    orchestrator: false,
    agents: false,
    blender: false,
    endToEnd: false
  };

  try {
    results.fuzzyPrimitives = testFuzzyPrimitives();
    results.analyzers = testAnalyzers();
    results.orchestrator = testOrchestratorActivations();
    results.agents = testInterpretationAgents();
    results.blender = testEnsembleBlender();
    results.endToEnd = testEndToEnd();
  } catch (error) {
    console.error('\n\n✗ TEST SUITE CRASHED:');
    console.error(error);
  }

  // Summary
  console.log('\n\n╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  TEST SUMMARY                                                 ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝\n');

  const passed = Object.values(results).filter(r => r).length;
  const total = Object.keys(results).length;

  Object.entries(results).forEach(([name, pass]) => {
    const icon = pass ? '✓' : '✗';
    const status = pass ? 'PASS' : 'FAIL';
    console.log(`${icon} ${name.padEnd(20)} ${status}`);
  });

  console.log(`\n${passed}/${total} tests passed (${((passed/total)*100).toFixed(0)}%)`);

  if (passed === total) {
    console.log('\n🎉 ALL TESTS PASSED - LunatiQ AGI is operational!');
  } else {
    console.log('\n⚠️  SOME TESTS FAILED - Debug required');
  }

  return results;
}

// Export for use in React Native
export default runAllTests;
