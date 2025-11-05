#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PROJECT GATORCA                                      ║
║          Gateway to Recursive Alligator Turtle Cognitive Army                ║
║                                                                              ║
║    36-Level Recursive Meta-Cognitive Evolutionary AGI ARC Solver            ║
╚══════════════════════════════════════════════════════════════════════════════╝

MISSION: Build a self-improving recursive AGI that devours ARC puzzles by
         evolving solver strategies across 36 abstraction levels.

PHASE 2: KNOWLEDGE INGESTION - Scan all sources and extract insights
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

# =====================================================
# PHASE 2: KNOWLEDGE INGESTION SYSTEM
# =====================================================

class KnowledgeScanner:
    """
    Scans all repository files and extracts knowledge for recursive levels

    Knowledge Hierarchy:
    - Strategic (L30-36): Frameworks, doctrines, grand patterns
    - Operational (L15-29): Algorithms, methods, tactics
    - Tactical (L1-14): Code primitives, operations, implementations
    """

    def __init__(self, repo_path: str = "/home/user/HungryOrca"):
        self.repo_path = Path(repo_path)
        self.knowledge_db = {
            'strategic': {},      # L30-36
            'operational': {},    # L15-29
            'tactical': {},       # L1-14
            'metadata': {}
        }

    def scan_all(self):
        """Master scan function - ingests ALL sources"""
        print("="*80)
        print("🐊 PROJECT GATORCA - KNOWLEDGE INGESTION COMMENCING 🐊")
        print("="*80)

        # Scan different file types
        self.scan_markdown_docs()
        self.scan_python_code()
        self.scan_notebooks()
        self.scan_json_data()
        self.scan_text_files()

        # Build indices
        self.build_strategic_index()
        self.build_operational_index()
        self.build_tactical_index()

        # Report
        self.print_scan_report()

        return self.knowledge_db

    # =====================================================
    # FILE TYPE SCANNERS
    # =====================================================

    def scan_markdown_docs(self):
        """Extract strategic insights from .md files"""
        print("\n[STRATEGIC] Scanning Markdown documentation...")

        md_files = list(self.repo_path.glob("*.md"))
        strategic_docs = []

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8', errors='ignore')

                # Extract strategic patterns
                doc_data = {
                    'file': md_file.name,
                    'size': len(content),
                    'sections': self._extract_sections(content),
                    'acronyms': self._extract_acronyms(content),
                    'frameworks': self._extract_frameworks(content),
                    'principles': self._extract_principles(content)
                }

                strategic_docs.append(doc_data)
                print(f"  ✓ {md_file.name}: {len(doc_data['sections'])} sections, "
                      f"{len(doc_data['acronyms'])} acronyms")

            except Exception as e:
                print(f"  ✗ {md_file.name}: {e}")

        self.knowledge_db['strategic']['docs'] = strategic_docs
        print(f"  → Extracted {len(strategic_docs)} strategic documents")

    def scan_python_code(self):
        """Extract operational algorithms from .py files"""
        print("\n[OPERATIONAL] Scanning Python code...")

        py_files = list(self.repo_path.glob("*.py"))
        operational_code = []

        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')

                # Extract operational patterns
                code_data = {
                    'file': py_file.name,
                    'size': len(content),
                    'classes': self._extract_classes(content),
                    'functions': self._extract_functions(content),
                    'imports': self._extract_imports(content),
                    'algorithms': self._identify_algorithms(content)
                }

                operational_code.append(code_data)
                print(f"  ✓ {py_file.name}: {len(code_data['classes'])} classes, "
                      f"{len(code_data['functions'])} functions")

            except Exception as e:
                print(f"  ✗ {py_file.name}: {e}")

        self.knowledge_db['operational']['code'] = operational_code
        print(f"  → Extracted {len(operational_code)} operational modules")

    def scan_notebooks(self):
        """Extract tactical implementations from .ipynb files"""
        print("\n[TACTICAL] Scanning Jupyter notebooks...")

        nb_files = list(self.repo_path.glob("*.ipynb"))
        tactical_notebooks = []

        for nb_file in nb_files:
            try:
                with open(nb_file, 'r', encoding='utf-8', errors='ignore') as f:
                    nb_data = json.load(f)

                # Extract cells
                cells = nb_data.get('cells', [])
                code_cells = [c for c in cells if c.get('cell_type') == 'code']

                nb_info = {
                    'file': nb_file.name,
                    'total_cells': len(cells),
                    'code_cells': len(code_cells),
                    'operations': self._extract_notebook_operations(code_cells)
                }

                tactical_notebooks.append(nb_info)
                print(f"  ✓ {nb_file.name}: {len(code_cells)} code cells")

            except Exception as e:
                print(f"  ✗ {nb_file.name}: {e}")

        self.knowledge_db['tactical']['notebooks'] = tactical_notebooks
        print(f"  → Extracted {len(tactical_notebooks)} tactical notebooks")

    def scan_json_data(self):
        """Scan ARC challenge data"""
        print("\n[DATA] Scanning JSON datasets...")

        json_files = [
            'arc-agi_training_challenges.json',
            'arc-agi_evaluation_challenges.json',
            'arc-agi_training_solutions.json'
        ]

        dataset_info = []
        for json_file in json_files:
            try:
                json_path = self.repo_path / json_file
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    info = {
                        'file': json_file,
                        'tasks': len(data),
                        'size_mb': json_path.stat().st_size / (1024*1024)
                    }
                    dataset_info.append(info)
                    print(f"  ✓ {json_file}: {info['tasks']} tasks, {info['size_mb']:.1f}MB")
            except Exception as e:
                print(f"  ✗ {json_file}: {e}")

        self.knowledge_db['metadata']['datasets'] = dataset_info

    def scan_text_files(self):
        """Scan special text files like ctf.txt"""
        print("\n[STRATEGIC] Scanning text files...")

        txt_files = list(self.repo_path.glob("*.txt"))
        strategic_texts = []

        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding='utf-8', errors='ignore')

                txt_data = {
                    'file': txt_file.name,
                    'size': len(content),
                    'lines': content.count('\n'),
                    'axioms': self._extract_axioms(content),
                    'strategies': self._extract_strategies(content)
                }

                strategic_texts.append(txt_data)
                print(f"  ✓ {txt_file.name}: {txt_data['lines']} lines, "
                      f"{len(txt_data['axioms'])} axioms")

            except Exception as e:
                print(f"  ✗ {txt_file.name}: {e}")

        self.knowledge_db['strategic']['texts'] = strategic_texts

    # =====================================================
    # EXTRACTION HELPERS
    # =====================================================

    def _extract_sections(self, content: str) -> List[str]:
        """Extract markdown sections"""
        sections = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        return sections[:20]  # Top 20 sections

    def _extract_acronyms(self, content: str) -> List[str]:
        """Extract all-caps acronyms (3+ letters)"""
        acronyms = re.findall(r'\b[A-Z]{3,}\b', content)
        return list(set(acronyms))[:30]  # Top 30 unique

    def _extract_frameworks(self, content: str) -> List[str]:
        """Extract framework names"""
        patterns = [
            r'(\w+)\s+Framework',
            r'(\w+)\s+Method',
            r'(\w+)\s+System',
            r'(\w+)\s+Architecture'
        ]
        frameworks = []
        for pattern in patterns:
            frameworks.extend(re.findall(pattern, content, re.IGNORECASE))
        return list(set(frameworks))[:20]

    def _extract_principles(self, content: str) -> List[str]:
        """Extract principles and insights"""
        patterns = [
            r'Principle:\s*(.+?)(?:\n|$)',
            r'Insight:\s*(.+?)(?:\n|$)',
            r'Key:\s*(.+?)(?:\n|$)',
        ]
        principles = []
        for pattern in patterns:
            principles.extend(re.findall(pattern, content))
        return principles[:15]

    def _extract_classes(self, content: str) -> List[str]:
        """Extract class names from Python code"""
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        return classes

    def _extract_functions(self, content: str) -> List[str]:
        """Extract function names from Python code"""
        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        return functions[:30]  # Top 30

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements"""
        imports = re.findall(r'^(?:import|from)\s+([\w.]+)', content, re.MULTILINE)
        return list(set(imports))

    def _identify_algorithms(self, content: str) -> List[str]:
        """Identify algorithm patterns in code"""
        algorithms = []

        # Pattern matching for common algorithms
        if 'def dfs' in content or 'depth_first' in content:
            algorithms.append('DFS')
        if 'def bfs' in content or 'breadth_first' in content:
            algorithms.append('BFS')
        if 'genetic' in content.lower() or 'evolve' in content.lower():
            algorithms.append('Genetic')
        if 'fuzzy' in content.lower():
            algorithms.append('Fuzzy')
        if 'neural' in content.lower() or 'network' in content.lower():
            algorithms.append('Neural')

        return algorithms

    def _extract_notebook_operations(self, code_cells: List[Dict]) -> List[str]:
        """Extract operations from notebook code cells"""
        operations = []
        for cell in code_cells[:10]:  # First 10 cells
            source = ''.join(cell.get('source', []))
            # Look for grid operations
            if 'flip' in source or 'rotate' in source or 'reflect' in source:
                operations.append('transformation')
            if 'grid' in source or 'matrix' in source:
                operations.append('grid_ops')
        return list(set(operations))

    def _extract_axioms(self, content: str) -> List[str]:
        """Extract axioms from text"""
        axioms = re.findall(r'Axiom\s+\d+:\s*(.+?)(?:\n|$)', content)
        return axioms

    def _extract_strategies(self, content: str) -> List[str]:
        """Extract strategy patterns"""
        strategies = re.findall(r'Strategy:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        return strategies[:10]

    # =====================================================
    # INDEX BUILDERS
    # =====================================================

    def build_strategic_index(self):
        """Build searchable index for L30-36 (Strategic)"""
        print("\n[INDEX] Building strategic knowledge index (L30-36)...")

        strategic_index = {
            'frameworks': [],
            'doctrines': [],
            'principles': [],
            'acronyms': []
        }

        # From docs
        frameworks_set = set()
        acronyms_set = set()
        for doc in self.knowledge_db['strategic'].get('docs', []):
            frameworks_set.update(doc.get('frameworks', []))
            acronyms_set.update(doc.get('acronyms', []))
            strategic_index['principles'].extend(doc.get('principles', []))

        # From texts
        for txt in self.knowledge_db['strategic'].get('texts', []):
            acronyms_set.update(txt.get('axioms', []))

        # Convert sets to lists for JSON serialization
        strategic_index['frameworks'] = list(frameworks_set)
        strategic_index['acronyms'] = list(acronyms_set)

        self.knowledge_db['strategic']['index'] = strategic_index
        print(f"  → Indexed {len(strategic_index['frameworks'])} frameworks, "
              f"{len(strategic_index['acronyms'])} acronyms")

    def build_operational_index(self):
        """Build searchable index for L15-29 (Operational)"""
        print("\n[INDEX] Building operational knowledge index (L15-29)...")

        classes_set = set()
        methods_set = set()
        algorithms_counter = Counter()

        for code in self.knowledge_db['operational'].get('code', []):
            classes_set.update(code.get('classes', []))
            methods_set.update(code.get('functions', []))
            for algo in code.get('algorithms', []):
                algorithms_counter[algo] += 1

        operational_index = {
            'classes': list(classes_set),
            'methods': list(methods_set)[:50],
            'algorithms': dict(algorithms_counter)
        }

        self.knowledge_db['operational']['index'] = operational_index
        print(f"  → Indexed {len(operational_index['classes'])} classes, "
              f"{len(operational_index['algorithms'])} algorithm types")

    def build_tactical_index(self):
        """Build searchable index for L1-14 (Tactical)"""
        print("\n[INDEX] Building tactical knowledge index (L1-14)...")

        operations_set = set()

        for nb in self.knowledge_db['tactical'].get('notebooks', []):
            operations_set.update(nb.get('operations', []))

        tactical_index = {
            'operations': list(operations_set),
            'primitives': []
        }

        self.knowledge_db['tactical']['index'] = tactical_index
        print(f"  → Indexed {len(tactical_index['operations'])} operation types")

    # =====================================================
    # REPORTING
    # =====================================================

    def print_scan_report(self):
        """Print comprehensive scan report"""
        print("\n" + "="*80)
        print("📊 KNOWLEDGE INGESTION COMPLETE - SCAN REPORT")
        print("="*80)

        # Strategic
        strat_docs = len(self.knowledge_db['strategic'].get('docs', []))
        strat_texts = len(self.knowledge_db['strategic'].get('texts', []))
        strat_index = self.knowledge_db['strategic'].get('index', {})

        print(f"\n🌌 STRATEGIC (L30-36):")
        print(f"   Documents: {strat_docs}")
        print(f"   Text files: {strat_texts}")
        print(f"   Frameworks: {len(strat_index.get('frameworks', []))}")
        print(f"   Acronyms: {len(strat_index.get('acronyms', []))}")
        print(f"   Principles: {len(strat_index.get('principles', []))}")

        # Operational
        op_code = len(self.knowledge_db['operational'].get('code', []))
        op_index = self.knowledge_db['operational'].get('index', {})

        print(f"\n⚙️  OPERATIONAL (L15-29):")
        print(f"   Python files: {op_code}")
        print(f"   Classes: {len(op_index.get('classes', []))}")
        print(f"   Methods: {len(op_index.get('methods', []))}")
        print(f"   Algorithms: {len(op_index.get('algorithms', {}))}")

        # Tactical
        tac_nb = len(self.knowledge_db['tactical'].get('notebooks', []))
        tac_index = self.knowledge_db['tactical'].get('index', {})

        print(f"\n🎯 TACTICAL (L1-14):")
        print(f"   Notebooks: {tac_nb}")
        print(f"   Operations: {len(tac_index.get('operations', []))}")

        # Metadata
        datasets = len(self.knowledge_db['metadata'].get('datasets', []))

        print(f"\n📦 METADATA:")
        print(f"   Datasets: {datasets}")

        print("\n" + "="*80)
        print("✅ PHASE 2 COMPLETE - Knowledge base ready for recursive levels!")
        print("="*80)

    def save_knowledge_db(self, output_file: str = "gatorca_knowledge.json"):
        """Save knowledge database to JSON"""
        output_path = self.repo_path / output_file
        with open(output_path, 'w') as f:
            json.dump(self.knowledge_db, f, indent=2)
        print(f"\n💾 Knowledge database saved to: {output_file}")


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         🐊 PROJECT GATORCA 🐊                                ║
║                  Gateway to Recursive Alligator Turtles                      ║
║                                                                              ║
║              36-Level Meta-Cognitive Evolutionary AGI Solver                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    scanner = KnowledgeScanner()
    knowledge_db = scanner.scan_all()
    scanner.save_knowledge_db()

    print("\n🐢 NEXT PHASE: Build recursive breeding engine!")
    print("🐊 TURTLE ARMY ASSEMBLING...")
    print("\n🚀 GATORCA ACTIVE! CHARLIE MIKE!")
