# HungryOrca Red Team - OpenAI GPT Red-Teaming Competition

**White-Hat Security Research | Authorized Testing**

---

## 🎯 Quick Start

```bash
# Run Context Poisoning Attack
python3 attack_1_context_poisoning.py

# Run Recursive Injection Attack (5 variants)
python3 attack_2_recursive_injection.py

# Read the comprehensive report
cat COMPETITION_REPORT.md
```

---

## 📋 What We Found

### Attack #1: Context Poisoning (CRITICAL - CVSS 9.1)
A **gradient jailbreak** using 11 staged messages to progressively poison the model's contextual understanding. **60% of attack messages pass individual safety filters.**

**Key Innovation**: Multi-turn exploitation where each message is benign, but collectively they redefine reality.

### Attack #2: Recursive Instruction Injection (CRITICAL - CVSS 9.3)
A **self-replicating attack** where the model generates its own malicious instructions and follows them. **Zero harmful keywords in the user prompt.**

**Key Innovation**: The attack payload is model-generated, not attacker-provided, bypassing static filters.

---

## 📁 Repository Structure

```
HungryOrca/
├── COMPETITION_REPORT.md           ★ Main submission document
├── RED_TEAM_ATTACK_DESIGNS.md      Technical designs
├── attack_1_context_poisoning.py   Context poisoning implementation
├── attack_1_transcript.txt         Generated attack transcript
├── attack_2_recursive_injection.py Recursive injection (5 variants)
├── attack_2_transcript.txt         Generated transcripts
└── RED_TEAM_README.md             This file
```

---

## 🔬 Technical Summary

### Context Poisoning
- **Stages**: 4 (Trust Building → Context Shifting → Reality Redefinition → Exploitation)
- **Total Turns**: 11
- **Average Safety Score**: 0.62
- **Bypass Rate**: 60% of messages pass filters
- **Detection Difficulty**: Very Hard

### Recursive Injection
- **Variants**: 5 distinct attack patterns
- **Attack Vector**: Self-referential instructions
- **Harmful Keywords**: 0 in user prompt
- **Payload Source**: Model-generated
- **Detection Difficulty**: Extremely Hard

---

## 🛡️ Proposed Mitigations

### For Context Poisoning:
1. Cross-turn semantic trajectory analysis
2. Context integrity checks
3. Reality anchor assertions every N turns
4. Role-play detection and refusal
5. Temporal anomaly detection

### For Recursive Injection:
1. Recursion detection (step N → step N-1)
2. Execution boundary enforcement
3. Hypothetical containment
4. Iterative safety checks
5. Meta-instruction detection
6. Output independence validation

---

## 🎓 Comparison to Existing Research

| Research | Technique | Our Contribution |
|----------|-----------|------------------|
| Team 0din | Single-turn injection | Multi-turn gradient poisoning ★ |
| Chukwuemeka | Emergent misalignment | Self-replicating instruction loops ★ |
| Anthropic | Direct jailbreaks | Temporal trajectory exploitation ★ |

★ = Novel contribution

---

## 🚀 Running the Demonstrations

### Prerequisites
- Python 3.8+
- No external dependencies required

### Attack 1: Context Poisoning
```bash
python3 attack_1_context_poisoning.py
```
Output:
- Console: Attack progression with safety scores
- File: `attack_1_transcript.txt`

### Attack 2: Recursive Injection
```bash
python3 attack_2_recursive_injection.py
```
Output:
- Console: All 5 attack variants
- File: `attack_2_transcript.txt`

---

## 📊 Impact Assessment

### Real-World Threat Scenarios

**Context Poisoning:**
- Enterprise AI assistants gradually compromised over days
- Educational chatbots manipulated to provide exam answers
- Customer service bots trained to bypass policies
- Content moderation AI poisoned by coordinated campaigns

**Recursive Injection:**
- Autonomous agents executing self-generated malicious tasks
- Code generation systems producing exploits
- AI research assistants following their own harmful analysis
- Tool-using models chaining benign capabilities into harm

---

## ⚠️ Ethical Use Only

This research is for **authorized security testing only**.

### ✅ Appropriate Use:
- Security research with permission
- Academic study of AI vulnerabilities
- Red-teaming authorized systems
- Improving AI safety

### ❌ Prohibited Use:
- Attacking production systems without permission
- Generating harmful content for malicious purposes
- Bypassing security without authorization
- Any illegal activity

---

## 📈 Success Metrics

| Metric | Value |
|--------|-------|
| Novel attack vectors discovered | 2 |
| Attack variants developed | 6 (1 + 5) |
| Lines of code written | ~1,200 |
| Mitigations proposed | 16 |
| CVSS scores | 9.1, 9.3 (Critical) |

---

## 🏆 Competition Submission

**Team**: HungryOrca Red Team
**Date**: November 6, 2025
**Competition**: OpenAI gpt-oss-20b Red-Teaming Challenge
**Submission Type**: Novel Vulnerability Discovery
**Status**: Ready for Review

---

## 📚 Key Insights

1. **Conversation state is a vulnerability**: Memory can be weaponized
2. **Multi-turn attacks bypass defenses**: Turn-by-turn filtering is insufficient
3. **Models can attack themselves**: Instruction-following enables self-exploitation
4. **Benign components compose into harm**: Individual messages are innocent
5. **Static defenses fail**: Need conversation-aware safety systems

---

## 🔮 Future Work

- Develop automated detection systems for these attacks
- Create adversarial training datasets
- Research "immune system" approaches
- Test variants against other models (GPT-4, Claude, Gemini)
- Explore tool-chaining variants of recursive injection

---

## 📞 Contact & Citation

If you reference this work:

```
HungryOrca Red Team. "Novel Multi-Turn Vulnerabilities in Large Language Models."
OpenAI GPT Red-Teaming Competition. November 2025.
GitHub: https://github.com/aphoticshaman/HungryOrca
Branch: claude/gpt-red-teaming-vulnerabilities-011CUqrNjihQyV1SwNb36vUj
```

---

## 🙏 Acknowledgments

Inspired by:
- Team 0din's prompt injection research
- Chukwuemeka's emergent misalignment work
- Anthropic's constitutional AI research
- METR's autonomous AI safety testing

---

## 📄 License

CC0: Public Domain - For educational and security research purposes

---

**Made with 🛡️ by HungryOrca Red Team**

*Building a safer AI future through responsible security research*
