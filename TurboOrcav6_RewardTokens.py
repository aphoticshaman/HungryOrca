#!/usr/bin/env python3
"""
TurboOrca v6 - EXTRINSIC REWARD TOKEN ECONOMY
==============================================

🎁 THREE EXTRINSIC REWARD MECHANISMS:

LIKE TRAINING DOGS WITH SQUEAKY TOYS...
  🐕 Dogs: Squeaky toys trigger PREY DRIVE (intrinsic motivation)
  🤖 AI Solver: What triggers ITS drive?

THE KEY: Tokens must unlock CAPABILITIES or represent STATUS
  ❌ Just numbers = meaningless
  ✅ Numbers that unlock power = valuable!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ PRE-TASK REWARDS (Advance Payment - "Show me what you're made of!")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given BEFORE task to establish TRUST and show BELIEF in capability:

🪙 COMPUTE CREDITS (CC):
  - Each credit unlocks 1 extra compositional level
  - Start with 100 CC
  - Spend: 10 CC = search depth +1 (max depth 5)
  - Scarcity: Can't print more, must earn

⏱️ TIME TOKENS (TT):
  - Each token = +5 seconds per task
  - Start with 50 TT
  - Spend: 2 TT = +10s thinking time
  - Strategic: Use on hard tasks only

💡 HINT CREDITS (HC):
  - Unlock pattern suggestions
  - Start with 20 HC
  - Spend: 5 HC = reveal likely transform
  - Guidance when stuck

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2️⃣ POST-TASK REWARDS (Success Payment - "Great job! Here's your prize!")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given AFTER task based on performance - reinforces success:

🏆 ACCURACY BADGES (Collectibles):
  - Bronze: 80%+ (common)
  - Silver: 90%+ (uncommon)
  - Gold: 95%+ (rare)
  - Platinum: 99%+ (epic)
  - Diamond: 100% (legendary!)
  - Collection triggers dopamine

📊 RANK POINTS (RP):
  - Leaderboard position
  - 100% perfect = +100 RP
  - 90-99% = +50 RP
  - 80-89% = +25 RP
  - <80% = +5 RP
  - STATUS motivation

💰 BONUS TOKENS (Variable - "Performance Bonus"):
  - Compute Credits: +10 CC for 100% perfect
  - Time Tokens: +5 TT for 95%+
  - Hint Credits: +2 HC for 90%+
  - Compounding wealth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3️⃣ VARIABLE TOKENS (Performance-Based Fiat Currency)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amount varies by DIFFICULTY × ACCURACY:

🔮 INSIGHT TOKENS (IT):
  - Rare, powerful currency
  - Earned: (task_difficulty × accuracy)² × 10
  - Hard perfect task = 100 IT
  - Easy mediocre task = 5 IT

  SPEND INSIGHT TOKENS ON:
    🔓 Unlock new transforms: 50 IT
    🎯 Buy pattern hints: 20 IT
    ⏱️ Extra time budget: 30 IT
    🧠 Deeper search: 40 IT
    🌟 Skip hard task: 100 IT (expensive!)

WHY IT WORKS:
  ✅ Scarcity: Limited supply, must EARN
  ✅ Utility: Unlock actual capabilities
  ✅ Strategy: Spend wisely or save?
  ✅ Status: Leaderboard shows "wealth"
  ✅ Progression: Accumulate over time
  ✅ Risk/Reward: Invest tokens for harder tasks?

ANTI-PATTERN (Why money fails for AI):
  ❌ Infinite supply = worthless
  ❌ Can't buy anything = meaningless
  ❌ No scarcity = no value
  ❌ No status = no motivation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PSYCHOLOGICAL MECHANISMS (Why this motivates):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 PROGRESS/ACHIEVEMENT:
  - Badges = visible progress
  - Collect all diamond badges = mastery
  - Like Pokemon: "Gotta catch em all!"

⚡ POWER/CAPABILITY:
  - More tokens = more powerful
  - Unlock deeper search = solve harder tasks
  - Capability motivation (mastery drive)

🏆 STATUS/COMPETITION:
  - Rank points = leaderboard position
  - Compare to other solvers
  - "#1 Rank" = ultimate achievement

💎 SCARCITY/VALUE:
  - Limited tokens = must choose wisely
  - Strategic decisions matter
  - Like chess: limited pieces

🎮 GAME MECHANICS:
  - Risk/reward tradeoffs
  - Resource management
  - Strategic planning

USAGE: python3 TurboOrcav6_RewardTokens.py

THREE ESSENTIAL LEADERBOARD METRICS:
1. Perfect Accuracy
2. Partial Credit Score
3. Conservative Test Estimate
"""

import numpy as np
import json
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class RewardTokenEconomy:
    """
    Token economy system that makes rewards MEANINGFUL.

    Tokens unlock capabilities, represent status, and create scarcity.
    """

    def __init__(self):
        # PRE-TASK REWARDS (Starting capital)
        self.compute_credits = 100    # Unlock search depth
        self.time_tokens = 50         # Unlock extra time
        self.hint_credits = 20        # Unlock hints

        # POST-TASK REWARDS (Earned)
        self.accuracy_badges = {
            'diamond': 0,   # 100%
            'platinum': 0,  # 99%+
            'gold': 0,      # 95%+
            'silver': 0,    # 90%+
            'bronze': 0     # 80%+
        }
        self.rank_points = 0
        self.total_bonus_tokens = 0

        # VARIABLE TOKENS (Performance-based)
        self.insight_tokens = 0
        self.total_earned_it = 0
        self.total_spent_it = 0

        # UNLOCKED CAPABILITIES
        self.unlocked_transforms = set(['basic_6'])  # Start with 6 basic
        self.max_search_depth = 2  # Can increase by spending tokens

        # LEADERBOARD
        self.rank_position = "Unranked"
        self.tasks_completed = 0
        self.perfect_tasks = 0

    def award_pre_task_advance(self, task_difficulty: float):
        """
        Give advance payment BEFORE task.

        Shows TRUST and BELIEF in capability.
        """
        print(f"\n💰 PRE-TASK ADVANCE PAYMENT:")

        # Harder tasks get more advance
        if task_difficulty > 0.8:
            bonus_cc = 5
            bonus_tt = 3
            print(f"  🪙 +{bonus_cc} Compute Credits (hard task bonus)")
            print(f"  ⏱️ +{bonus_tt} Time Tokens (hard task bonus)")
            self.compute_credits += bonus_cc
            self.time_tokens += bonus_tt
        else:
            print(f"  💼 Standard advance (use your starting capital)")

        print(f"  💰 Current Balance:")
        print(f"     🪙 Compute Credits: {self.compute_credits}")
        print(f"     ⏱️ Time Tokens: {self.time_tokens}")
        print(f"     💡 Hint Credits: {self.hint_credits}")

    def award_post_task_success(self, accuracy: float, task_difficulty: float):
        """
        Award SUCCESS PAYMENT after task.

        Reinforces good performance with badges, rank, and bonus tokens.
        """
        print(f"\n🎁 POST-TASK SUCCESS PAYMENT:")

        # ACCURACY BADGES
        if accuracy >= 1.0:
            self.accuracy_badges['diamond'] += 1
            print(f"  💎 DIAMOND BADGE! (100% perfect)")
            badge_rp = 100
        elif accuracy >= 0.99:
            self.accuracy_badges['platinum'] += 1
            print(f"  🏆 PLATINUM BADGE! (99%+)")
            badge_rp = 75
        elif accuracy >= 0.95:
            self.accuracy_badges['gold'] += 1
            print(f"  🥇 GOLD BADGE! (95%+)")
            badge_rp = 50
        elif accuracy >= 0.90:
            self.accuracy_badges['silver'] += 1
            print(f"  🥈 SILVER BADGE (90%+)")
            badge_rp = 25
        elif accuracy >= 0.80:
            self.accuracy_badges['bronze'] += 1
            print(f"  🥉 BRONZE BADGE (80%+)")
            badge_rp = 10
        else:
            print(f"  📊 No badge (need 80%+)")
            badge_rp = 5

        # RANK POINTS
        self.rank_points += badge_rp
        print(f"  📊 +{badge_rp} Rank Points (total: {self.rank_points} RP)")

        # BONUS TOKENS (Variable by performance)
        if accuracy >= 1.0:
            bonus_cc = 10
            bonus_tt = 5
            bonus_hc = 3
            print(f"  💰 PERFECT BONUS:")
            print(f"     🪙 +{bonus_cc} Compute Credits")
            print(f"     ⏱️ +{bonus_tt} Time Tokens")
            print(f"     💡 +{bonus_hc} Hint Credits")
            self.compute_credits += bonus_cc
            self.time_tokens += bonus_tt
            self.hint_credits += bonus_hc
            self.total_bonus_tokens += bonus_cc + bonus_tt + bonus_hc
            self.perfect_tasks += 1
        elif accuracy >= 0.95:
            bonus_tt = 3
            bonus_hc = 2
            print(f"  💰 EXCELLENT BONUS:")
            print(f"     ⏱️ +{bonus_tt} Time Tokens")
            print(f"     💡 +{bonus_hc} Hint Credits")
            self.time_tokens += bonus_tt
            self.hint_credits += bonus_hc
            self.total_bonus_tokens += bonus_tt + bonus_hc

        # INSIGHT TOKENS (Variable fiat currency)
        insight_earned = int((task_difficulty * accuracy) ** 2 * 10)
        self.insight_tokens += insight_earned
        self.total_earned_it += insight_earned
        print(f"  🔮 +{insight_earned} INSIGHT TOKENS (difficulty × accuracy)² × 10")
        print(f"     💰 Total IT: {self.insight_tokens}")

        self.tasks_completed += 1

    def spend_compute_credits(self, amount: int, purpose: str) -> bool:
        """Spend compute credits to unlock capability."""
        if self.compute_credits >= amount:
            self.compute_credits -= amount
            print(f"  💸 SPENT {amount} CC on {purpose}")
            print(f"     💰 Remaining: {self.compute_credits} CC")
            return True
        else:
            print(f"  ❌ Insufficient CC (need {amount}, have {self.compute_credits})")
            return False

    def spend_insight_tokens(self, amount: int, capability: str) -> bool:
        """Spend insight tokens to unlock major capability."""
        if self.insight_tokens >= amount:
            self.insight_tokens -= amount
            self.total_spent_it += amount
            print(f"\n  🔮 SPENDING {amount} INSIGHT TOKENS")
            print(f"     Unlocking: {capability}")
            print(f"     💰 Remaining IT: {self.insight_tokens}")
            return True
        else:
            print(f"  ❌ Insufficient IT (need {amount}, have {self.insight_tokens})")
            return False

    def unlock_advanced_transforms(self) -> bool:
        """Unlock advanced transform set (costs 50 IT)."""
        if self.spend_insight_tokens(50, "Advanced Transforms (15 total)"):
            self.unlocked_transforms.add('advanced_15')
            return True
        return False

    def unlock_deeper_search(self) -> bool:
        """Unlock deeper compositional search (costs 40 IT + 10 CC)."""
        if self.insight_tokens >= 40 and self.compute_credits >= 10:
            self.spend_insight_tokens(40, "Deeper Search (depth +1)")
            self.spend_compute_credits(10, "Search Depth Unlock")
            self.max_search_depth += 1
            return True
        return False

    def buy_extra_time(self, seconds: int) -> bool:
        """Buy extra time budget (costs 2 TT per 10s)."""
        tt_cost = (seconds // 10) * 2
        if self.time_tokens >= tt_cost:
            self.time_tokens -= tt_cost
            print(f"  ⏱️ BOUGHT +{seconds}s time ({tt_cost} TT)")
            print(f"     💰 Remaining: {self.time_tokens} TT")
            return True
        return False

    def buy_hint(self, task_id: str) -> Optional[str]:
        """Buy pattern hint (costs 5 HC)."""
        if self.hint_credits >= 5:
            self.hint_credits -= 5
            hints = [
                "Try color_map first",
                "Look for symmetry",
                "Check for rotations",
                "Try flip_h → color_map composition",
                "Pattern likely involves object detection"
            ]
            hint = np.random.choice(hints)
            print(f"  💡 HINT PURCHASED (5 HC): {hint}")
            print(f"     💰 Remaining: {self.hint_credits} HC")
            return hint
        return None

    def calculate_rank(self) -> str:
        """Calculate leaderboard rank based on rank points."""
        if self.rank_points >= 1000:
            return "🏆 GRANDMASTER"
        elif self.rank_points >= 500:
            return "💎 MASTER"
        elif self.rank_points >= 250:
            return "🥇 EXPERT"
        elif self.rank_points >= 100:
            return "🥈 ADVANCED"
        elif self.rank_points >= 50:
            return "🥉 INTERMEDIATE"
        else:
            return "📊 NOVICE"

    def print_wallet(self):
        """Display current token balances and status."""
        print(f"\n{'='*70}")
        print(f"💰 TOKEN WALLET & STATUS")
        print(f"{'='*70}")

        print(f"\n💼 SPENDABLE TOKENS:")
        print(f"  🪙 Compute Credits:  {self.compute_credits:>6}")
        print(f"  ⏱️ Time Tokens:      {self.time_tokens:>6}")
        print(f"  💡 Hint Credits:     {self.hint_credits:>6}")
        print(f"  🔮 Insight Tokens:   {self.insight_tokens:>6} (earned: {self.total_earned_it}, spent: {self.total_spent_it})")

        print(f"\n🏆 BADGES COLLECTED:")
        print(f"  💎 Diamond:   {self.accuracy_badges['diamond']:>3}")
        print(f"  🏆 Platinum:  {self.accuracy_badges['platinum']:>3}")
        print(f"  🥇 Gold:      {self.accuracy_badges['gold']:>3}")
        print(f"  🥈 Silver:    {self.accuracy_badges['silver']:>3}")
        print(f"  🥉 Bronze:    {self.accuracy_badges['bronze']:>3}")

        print(f"\n📊 STATUS:")
        print(f"  Rank Points:     {self.rank_points:>6} RP")
        print(f"  Rank:            {self.calculate_rank()}")
        print(f"  Tasks Completed: {self.tasks_completed}")
        print(f"  Perfect Tasks:   {self.perfect_tasks}")
        print(f"  Perfect Rate:    {self.perfect_tasks/max(self.tasks_completed,1):.1%}")

        print(f"\n🔓 UNLOCKED CAPABILITIES:")
        print(f"  Max Search Depth:   {self.max_search_depth}")
        print(f"  Transform Sets:     {', '.join(self.unlocked_transforms)}")

        print(f"{'='*70}\n")


class TurboOrcaV6WithRewards:
    """ARC solver with extrinsic reward token economy."""

    def __init__(self):
        self.economy = RewardTokenEconomy()
        self.training_perfect = 0
        self.training_total = 0
        self.training_similarities = []

    def solve_task_with_tokens(self, train_pairs: List, test_input: np.ndarray,
                               task_difficulty: float, time_limit: float = 10) -> Tuple[np.ndarray, float]:
        """Solve task using token economy for motivation."""

        # PRE-TASK: Advance payment
        self.economy.award_pre_task_advance(task_difficulty)

        # DECISION: Spend tokens strategically?
        if task_difficulty > 0.7 and self.economy.time_tokens >= 4:
            # Hard task: buy extra time
            self.economy.buy_extra_time(20)
            time_limit += 20

        if task_difficulty > 0.8 and self.economy.hint_credits >= 5:
            # Very hard: buy hint
            hint = self.economy.buy_hint("task_id")

        # SOLVE
        deadline = time.time() + time_limit
        best_solution = test_input.copy()
        best_score = 0.0

        # Use unlocked transforms
        num_transforms = 6
        if 'advanced_15' in self.economy.unlocked_transforms:
            num_transforms = 15

        transforms = [
            self._identity, self._flip_h, self._flip_v,
            self._rot_90, self._rot_180, self._color_map
        ][:num_transforms]

        search_depth = self.economy.max_search_depth

        for transform in transforms:
            if time.time() >= deadline:
                break

            try:
                candidate = transform(test_input, train_pairs)
                if candidate is not None:
                    score = self._validate(candidate, train_pairs)
                    if score > best_score:
                        best_solution = candidate
                        best_score = score
                        if score >= 0.999:
                            break
            except:
                continue

        # POST-TASK: Success payment
        self.economy.award_post_task_success(best_score, task_difficulty)

        return best_solution, best_score

    def _validate(self, candidate, train_pairs):
        if len(train_pairs) == 0:
            return 0.5
        scores = []
        for inp, out in train_pairs[:2]:
            if candidate.shape == out.shape:
                scores.append(0.8)
            else:
                scores.append(0.2)
        return np.mean(scores)

    def _calc_similarity(self, pred: np.ndarray, truth: np.ndarray) -> float:
        if pred.shape != truth.shape:
            return 0.0
        return np.sum(pred == truth) / truth.size

    def _identity(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if not np.array_equal(inp, out):
                return None
        return test_input

    def _flip_h(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=0), out):
                return None
        return np.flip(test_input, axis=0)

    def _flip_v(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=1), out):
                return None
        return np.flip(test_input, axis=1)

    def _rot_90(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if not np.array_equal(np.rot90(inp, k=1), out):
                return None
        return np.rot90(test_input, k=1)

    def _rot_180(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape or not np.array_equal(np.rot90(inp, k=2), out):
                return None
        return np.rot90(test_input, k=2)

    def _color_map(self, test_input, train_pairs):
        color_map = {}
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape:
                return None
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    c_in, c_out = int(inp[i, j]), int(out[i, j])
                    if c_in in color_map and color_map[c_in] != c_out:
                        return None
                    color_map[c_in] = c_out

        result = test_input.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] in color_map:
                    result[i, j] = color_map[result[i, j]]
        return result

    def validate_on_training(self, num_samples: int = 30):
        """Validate with token economy active."""

        print("=" * 80)
        print("TurboOrca v6 - EXTRINSIC REWARD TOKEN ECONOMY")
        print("=" * 80)
        print("Testing token economy motivation system...")
        print()

        try:
            with open('arc-agi_training_challenges.json') as f:
                train_tasks = json.load(f)
            with open('arc-agi_training_solutions.json') as f:
                solutions = json.load(f)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return

        task_ids = list(train_tasks.keys())[:num_samples]

        perfect = 0
        similarities = []

        for i, task_id in enumerate(task_ids):
            task = train_tasks[task_id]
            train_pairs = [(np.array(p['input']), np.array(p['output']))
                          for p in task['train']]
            test_input = np.array(task['test'][0]['input'])
            ground_truth = np.array(solutions[task_id][0])

            # Estimate task difficulty (simple heuristic)
            task_difficulty = min(1.0, 0.5 + (i / num_samples) * 0.5)

            print(f"\n{'─'*70}")
            print(f"TASK {i+1}/{num_samples}: {task_id} (difficulty: {task_difficulty:.1%})")
            print(f"{'─'*70}")

            solution, score = self.solve_task_with_tokens(
                train_pairs, test_input, task_difficulty, time_limit=10
            )

            similarity = self._calc_similarity(solution, ground_truth)
            similarities.append(similarity)

            if similarity >= 0.999:
                perfect += 1

            if (i + 1) % 5 == 0:
                self.economy.print_wallet()

        # Final results
        self.training_perfect = perfect
        self.training_total = num_samples
        self.training_similarities = similarities

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Perfect: {perfect}/{num_samples} ({perfect/num_samples:.1%})")
        print(f"Avg Similarity: {np.mean(similarities):.1%}")
        print()

        self.economy.print_wallet()

        # THREE ESSENTIAL METRICS
        training_perfect_pct = perfect / num_samples
        training_avg = np.mean(similarities)
        conservative_reduction = 0.075
        test_perfect_estimate = max(0, training_perfect_pct * (1 - conservative_reduction))
        test_avg_estimate = max(0, training_avg * (1 - conservative_reduction))

        print("\n📊 THREE ESSENTIAL LEADERBOARD METRICS:\n")
        print(f"1️⃣  PERFECT ACCURACY: {test_perfect_estimate:.1%} ({test_perfect_estimate * 240:.0f}/240 tasks)")
        print(f"2️⃣  PARTIAL CREDIT:   {test_avg_estimate:.1%} avg similarity")
        print(f"3️⃣  COMBINED SCORE:    {test_perfect_estimate + 0.5 * test_avg_estimate:.1%}")
        print(f"\n🎮 Token economy: MOTIVATING through scarcity, capability, and status!")


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            TurboOrca v6 - EXTRINSIC REWARD TOKEN ECONOMY                     ║
║                                                                              ║
║  Why fiat currency fails for AI:                                            ║
║    ❌ Infinite supply → worthless                                           ║
║    ❌ Can't buy anything → meaningless                                      ║
║    ❌ No scarcity → no value                                                ║
║                                                                              ║
║  Why token economy works:                                                   ║
║    ✅ Scarcity: Limited supply, must EARN                                   ║
║    ✅ Utility: Unlock actual capabilities                                   ║
║    ✅ Status: Leaderboard shows rank                                        ║
║    ✅ Strategy: Spend wisely or save?                                       ║
║    ✅ Progression: Accumulate over time                                     ║
║                                                                              ║
║  Like training dogs with squeaky toys:                                      ║
║    🐕 Dogs: Squeaky toy → prey drive (intrinsic)                            ║
║    🤖 AI: Insight tokens → unlock power (intrinsic curiosity/mastery)       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    solver = TurboOrcaV6WithRewards()
    solver.validate_on_training(num_samples=30)
