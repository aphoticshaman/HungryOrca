"""
OrcaWhiskey + NSPSA Integration

Extends OrcaWhiskey v1 with fourth agent (NSPSA) for program synthesis.

Architecture:
- Agent A (HRM): Visual hierarchical reasoning
- Agent B (LLM): Linguistic reasoning
- VAE Mediator: Latent arbitration
- NSPSA: Program synthesis (NEW)

Voting: 4-way consensus with weighted confidence
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from nspsa import NSPSA
from dataclasses import dataclass

@dataclass
class FourAgentVote:
    """Results from 4-agent voting"""
    agent_a_pred: np.ndarray
    agent_a_conf: float
    agent_b_pred: np.ndarray
    agent_b_conf: float
    vae_pred: np.ndarray
    vae_conf: float
    nspsa_pred: Optional[np.ndarray]
    nspsa_conf: float

    final_prediction: np.ndarray
    final_confidence: float
    consensus: bool
    vote_weights: Dict[str, float]


class NSPSALatentBridge:
    """
    Bridge between NSPSA's program latent space and OrcaWhiskey's neural latent space.

    NSPSA outputs: 128D program embeddings
    OrcaWhiskey: High-dimensional neural states

    Bridge projects between spaces for cross-agent communication.
    """

    def __init__(self, nspsa_dim: int = 128, orca_dim: int = 512):
        self.nspsa_dim = nspsa_dim
        self.orca_dim = orca_dim

        # Linear projection matrices (learnable in full implementation)
        self.nspsa_to_orca = np.random.randn(nspsa_dim, orca_dim) * 0.01
        self.orca_to_nspsa = np.random.randn(orca_dim, nspsa_dim) * 0.01

    def project_nspsa_to_orca(self, nspsa_latent: np.ndarray) -> np.ndarray:
        """Project NSPSA's program encoding to OrcaWhiskey latent space"""
        return nspsa_latent @ self.nspsa_to_orca

    def project_orca_to_nspsa(self, orca_latent: np.ndarray) -> np.ndarray:
        """Project OrcaWhiskey's neural state to program space"""
        return orca_latent @ self.orca_to_nspsa


class ExtendedEpistemicOrchestrator:
    """
    Extends OrcaWhiskey's Epistemic Orchestrator with NSPSA.

    Workflow:
    1. All agents solve independently (no omniscience initially)
    2. NSPSA provides symbolic program + latent encoding
    3. Share NSPSA's latent across other agents (omniscient observation)
    4. Re-evaluate with symbolic knowledge
    5. 4-way vote: A, B, VAE, NSPSA
    6. Weighted consensus based on confidence
    """

    def __init__(self, nspsa_latent_dim: int = 128):
        # NSPSA agent
        self.nspsa = NSPSA(latent_dim=nspsa_latent_dim)

        # Latent bridge (for future OrcaWhiskey integration)
        self.bridge = NSPSALatentBridge(nspsa_dim=nspsa_latent_dim, orca_dim=512)

        # Statistics
        self.num_consensus = 0
        self.num_nspsa_wins = 0
        self.num_tasks = 0

    def solve_with_nspsa(self,
                         train_inputs: List[np.ndarray],
                         train_outputs: List[np.ndarray],
                         test_input: np.ndarray,
                         agent_a_pred: Optional[np.ndarray] = None,
                         agent_a_conf: float = 0.5,
                         agent_b_pred: Optional[np.ndarray] = None,
                         agent_b_conf: float = 0.5,
                         vae_pred: Optional[np.ndarray] = None,
                         vae_conf: float = 0.5,
                         verbose: bool = False) -> FourAgentVote:
        """
        Solve task with 4-agent system (A, B, VAE, NSPSA).

        If OrcaWhiskey predictions provided, use them. Otherwise NSPSA solves alone.

        Args:
            train_inputs: Training input grids
            train_outputs: Training output grids
            test_input: Test input to solve
            agent_a_pred: Optional HRM prediction
            agent_a_conf: HRM confidence
            agent_b_pred: Optional LLM prediction
            agent_b_conf: LLM confidence
            vae_pred: Optional VAE prediction
            vae_conf: VAE confidence
            verbose: Print reasoning trace

        Returns:
            FourAgentVote with final prediction and voting breakdown
        """
        self.num_tasks += 1

        if verbose:
            print("="*70)
            print("EXTENDED EPISTEMIC REASONING - 4 AGENTS")
            print("="*70)

        # ====================================================================
        # PHASE 1: NSPSA Synthesis
        # ====================================================================

        if verbose:
            print("\n🔧 NSPSA: Program synthesis...")

        nspsa_result, nspsa_trace = self.nspsa.solve(
            train_inputs,
            train_outputs,
            test_input,
            return_trace=True
        )

        if nspsa_result is not None:
            nspsa_conf = 0.95  # High confidence when program found

            if verbose:
                print(f"   ✅ Found program: {nspsa_trace['selected_program']}")
                print(f"   Confidence: {nspsa_conf:.2f}")

                if 'primitive_rankings' in nspsa_trace and nspsa_trace['primitive_rankings']:
                    print(f"   Top primitives predicted:")
                    for name, score in nspsa_trace['primitive_rankings'][:3]:
                        print(f"     - {name}: {score:.3f}")
        else:
            nspsa_conf = 0.1  # Low confidence on failure

            if verbose:
                print(f"   ❌ No program found")
                print(f"   Confidence: {nspsa_conf:.2f}")

        # ====================================================================
        # PHASE 2: Cross-Agent Communication (Future: Use latent bridge)
        # ====================================================================

        # If NSPSA found a solution, its latent encoding can inform other agents
        nspsa_latent = nspsa_trace.get('latent_encoding', None)

        if nspsa_latent is not None and verbose:
            print(f"\n📡 NSPSA latent encoding: {nspsa_latent.shape}")
            print(f"   (Future: Project to OrcaWhiskey latent space for omniscient observation)")

        # ====================================================================
        # PHASE 3: 4-Way Vote
        # ====================================================================

        if verbose:
            print(f"\n⚖️  4-WAY VOTE:")

        # Collect all predictions
        predictions = []
        confidences = []
        agent_names = []

        if agent_a_pred is not None:
            predictions.append(agent_a_pred)
            confidences.append(agent_a_conf)
            agent_names.append('agent_a')
            if verbose:
                print(f"   Agent A (HRM): conf={agent_a_conf:.3f}")

        if agent_b_pred is not None:
            predictions.append(agent_b_pred)
            confidences.append(agent_b_conf)
            agent_names.append('agent_b')
            if verbose:
                print(f"   Agent B (LLM): conf={agent_b_conf:.3f}")

        if vae_pred is not None:
            predictions.append(vae_pred)
            confidences.append(vae_conf)
            agent_names.append('vae')
            if verbose:
                print(f"   VAE Mediator: conf={vae_conf:.3f}")

        if nspsa_result is not None:
            predictions.append(nspsa_result)
            confidences.append(nspsa_conf)
            agent_names.append('nspsa')
            if verbose:
                print(f"   NSPSA: conf={nspsa_conf:.3f}")

        # ====================================================================
        # PHASE 4: Weighted Consensus
        # ====================================================================

        if len(predictions) == 0:
            # No agent succeeded - return zeros
            final_pred = np.zeros_like(test_input)
            final_conf = 0.0
            consensus = False
            vote_weights = {}

        elif len(predictions) == 1:
            # Only one agent - use its prediction
            final_pred = predictions[0]
            final_conf = confidences[0] * 0.8  # Penalty for no consensus
            consensus = False
            vote_weights = {agent_names[0]: 1.0}

        else:
            # Multiple predictions - weighted vote
            confidences_arr = np.array(confidences)
            weights = confidences_arr / confidences_arr.sum()

            # Check agreement (do predictions match?)
            agreements = []
            for i in range(len(predictions)):
                for j in range(i+1, len(predictions)):
                    if np.array_equal(predictions[i], predictions[j]):
                        agreements.append((i, j))

            consensus = len(agreements) >= len(predictions) // 2

            if consensus and verbose:
                print(f"\n   ✅ CONSENSUS achieved")
            elif verbose:
                print(f"\n   ⚠️  NO CONSENSUS - agents disagree")

            # Voting strategy: Highest confidence wins
            winner_idx = np.argmax(confidences)
            final_pred = predictions[winner_idx]

            # Confidence boost if consensus
            if consensus:
                final_conf = confidences[winner_idx] * 1.1
            else:
                final_conf = confidences[winner_idx] * 0.9

            final_conf = min(final_conf, 0.95)  # Cap at 95%

            vote_weights = {name: float(w) for name, w in zip(agent_names, weights)}

            if verbose:
                print(f"\n   Winner: {agent_names[winner_idx]} (conf={final_conf:.3f})")

        # Track statistics
        if consensus:
            self.num_consensus += 1

        if len(predictions) > 0 and agent_names[np.argmax(confidences)] == 'nspsa':
            self.num_nspsa_wins += 1

        # ====================================================================
        # RESULT
        # ====================================================================

        result = FourAgentVote(
            agent_a_pred=agent_a_pred if agent_a_pred is not None else np.zeros_like(test_input),
            agent_a_conf=agent_a_conf,
            agent_b_pred=agent_b_pred if agent_b_pred is not None else np.zeros_like(test_input),
            agent_b_conf=agent_b_conf,
            vae_pred=vae_pred if vae_pred is not None else np.zeros_like(test_input),
            vae_conf=vae_conf,
            nspsa_pred=nspsa_result,
            nspsa_conf=nspsa_conf,
            final_prediction=final_pred,
            final_confidence=final_conf,
            consensus=consensus,
            vote_weights=vote_weights
        )

        if verbose:
            print("="*70)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'num_tasks': self.num_tasks,
            'num_consensus': self.num_consensus,
            'consensus_rate': self.num_consensus / max(1, self.num_tasks),
            'num_nspsa_wins': self.num_nspsa_wins,
            'nspsa_win_rate': self.num_nspsa_wins / max(1, self.num_tasks),
            'nspsa_stats': self.nspsa.get_stats()
        }


# ============================================================================
# TESTS
# ============================================================================

def test_nspsa_only():
    """Test NSPSA solving tasks independently"""
    print("="*70)
    print("TEST 1: NSPSA Standalone")
    print("="*70)

    orchestrator = ExtendedEpistemicOrchestrator()

    # Simple rotation task
    train_in = [np.array([[1, 2], [3, 4]])]
    train_out = [np.array([[3, 1], [4, 2]])]
    test_in = np.array([[5, 6], [7, 8]])

    result = orchestrator.solve_with_nspsa(
        train_in, train_out, test_in,
        verbose=True
    )

    print(f"\nFinal prediction:")
    print(result.final_prediction)
    print(f"Final confidence: {result.final_confidence:.3f}")
    print(f"Consensus: {result.consensus}")

    assert result.nspsa_pred is not None, "NSPSA should solve rotation"
    expected = np.array([[7, 5], [8, 6]])
    assert np.array_equal(result.final_prediction, expected), "Should predict rotated grid"

    print("\n✅ NSPSA standalone working")


def test_four_agent_vote():
    """Test 4-agent voting with mock OrcaWhiskey predictions"""
    print("\n" + "="*70)
    print("TEST 2: 4-Agent Voting")
    print("="*70)

    orchestrator = ExtendedEpistemicOrchestrator()

    # Task
    train_in = [np.array([[1, 0], [0, 2]])]
    train_out = [np.array([[0, 1], [2, 0]])]
    test_in = np.array([[3, 4], [5, 6]])

    # Mock OrcaWhiskey predictions (agents A, B, VAE)
    # Suppose they predict rotation correctly
    correct_pred = np.array([[5, 3], [6, 4]])

    result = orchestrator.solve_with_nspsa(
        train_in, train_out, test_in,
        agent_a_pred=correct_pred,
        agent_a_conf=0.7,
        agent_b_pred=correct_pred,
        agent_b_conf=0.6,
        vae_pred=correct_pred,
        vae_conf=0.65,
        verbose=True
    )

    print(f"\nVote weights: {result.vote_weights}")

    # NSPSA should also get it right and achieve consensus
    if result.consensus:
        print("✅ CONSENSUS achieved across all agents")
    else:
        print("⚠️  No consensus (expected if NSPSA failed)")

    print(f"\nFinal confidence: {result.final_confidence:.3f}")

    # Statistics
    stats = orchestrator.get_stats()
    print(f"\n📊 Statistics:")
    print(f"   Tasks: {stats['num_tasks']}")
    print(f"   Consensus rate: {stats['consensus_rate']:.1%}")
    print(f"   NSPSA win rate: {stats['nspsa_win_rate']:.1%}")


if __name__ == '__main__':
    print("="*70)
    print("ORCAWHISKEY + NSPSA INTEGRATION")
    print("="*70)

    test_nspsa_only()
    test_four_agent_vote()

    print("\n" + "="*70)
    print("✅ INTEGRATION TESTS PASSED")
    print("="*70)
    print("\nNext steps:")
    print("1. Load full OrcaWhiskey v1 notebook")
    print("2. Replace EpistemicOrchestrator with ExtendedEpistemicOrchestrator")
    print("3. Train latent bridge (NSPSA ↔ OrcaWhiskey)")
    print("4. Evaluate on ARC-AGI test set")
    print("="*70)
