#!/usr/bin/env python3
"""
PI EVOLUTION - DEEP THINKING MODE

Features:
- Chain-of-thought reasoning (Groq explains WHY)
- Higher-dimensional mathematical thinking
- Complete formula history feedback
- Multiple candidates per iteration
- Meta-analysis of failures
- Extended thinking time
- Cross-domain mathematical concepts

"""

import os
import sys
import json
import math
import time
import random
import mmap
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import re

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.markdown import Markdown
    from groq import Groq
    import numpy as np
except ImportError:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.markdown import Markdown
    from groq import Groq
    import numpy as np

console = Console()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
PI_FILE = Path("pi_digits.txt")

MIN_ACCURACY = 50.0
MAX_ITERATIONS = 30  # More chances
SAMPLE_SIZE = 15000  # Larger sample for better statistics
CANDIDATES_PER_ITERATION = 3  # Test multiple formulas each time

OUTPUT_DIR = Path("./pi_evolution")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# FAST PI READER
# ============================================================================

class FastPiReader:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file_size = filepath.stat().st_size
        self.file_handle = open(filepath, 'r+b')
        self.mmap = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        
        header = self.mmap[:20].decode('utf-8', errors='ignore')
        self.offset = header.index('.') + 1 if '.' in header else (1 if header[0] == '3' else 0)
        self.total_digits = self.file_size - self.offset
        
        console.print(f"[green]✓ Loaded {self.total_digits:,} π digits[/green]\n")
    
    def get_batch_fast(self, positions: List[int]) -> np.ndarray:
        positions_array = np.array(positions, dtype=np.int64)
        results = np.zeros(len(positions), dtype=np.int8)
        
        for i, pos in enumerate(positions_array):
            if 1 <= pos <= self.total_digits:
                byte_pos = self.offset + pos - 1
                char = chr(self.mmap[byte_pos])
                if char.isdigit():
                    results[i] = int(char)
        return results
    
    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file_handle'):
            self.file_handle.close()

# ============================================================================
# FORMULA EVALUATOR
# ============================================================================

def evaluate_formula_fast(formula_code: str, reader: FastPiReader, sample_size: int) -> Tuple[float, List[Dict], int]:
    """Returns: (accuracy, results, error_count)"""
    
    safe_globals = {
        'math': math, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'log': math.log, 'log10': math.log10, 'exp': math.exp, 'sqrt': math.sqrt,
        'abs': abs, 'int': int, 'float': float, 'pow': pow, 'pi': math.pi, 'e': math.e,
        'np': np, 'floor': math.floor, 'ceil': math.ceil,
    }
    
    positions = np.random.randint(1, reader.total_digits, size=sample_size, dtype=np.int64)
    actual_digits = reader.get_batch_fast(positions)
    predicted_digits = np.zeros(len(positions), dtype=np.int8)
    errors = 0
    
    for i, n in enumerate(positions):
        try:
            local_vars = {'n': int(n)}
            exec(f"result = {formula_code}", safe_globals, local_vars)
            predicted_digits[i] = int(local_vars.get('result', 0)) % 10
        except:
            errors += 1
            predicted_digits[i] = 0
    
    correct_mask = (predicted_digits == actual_digits)
    accuracy = (np.sum(correct_mask) / len(correct_mask)) * 100.0
    
    results = []
    for i in range(min(20, len(positions))):
        results.append({
            'position': int(positions[i]),
            'actual': int(actual_digits[i]),
            'predicted': int(predicted_digits[i]),
            'correct': bool(correct_mask[i])
        })
    
    return accuracy, results, errors

# ============================================================================
# DEEP THINKING EVOLVER
# ============================================================================

class DeepThinker:
    """Groq with deep mathematical reasoning"""
    
    def __init__(self, key):
        self.groq = Groq(api_key=key)
        self.iteration = 0
        self.all_formulas = []  # Complete history
        self.all_results = []
        self.best_ever = 0.0
        
    def ask_with_reasoning(self, prompt: str, temp: float = 0.6) -> Dict:
        """Get formula WITH reasoning chain"""
        
        for attempt in range(3):
            try:
                response = self.groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a mathematical genius exploring the deep structure of π.

You think beyond simple 2D circle geometry. You consider:
- Higher-dimensional geometry (hyperspheres, n-dimensional volumes)
- Quantum mechanics (π in wave functions, uncertainty principle)
- Number theory (π in prime distributions, Riemann zeta function)
- Chaos theory (strange attractors, fractal dimensions)
- Information theory (π in entropy, compression)
- Statistical mechanics (π in partition functions)

You provide:
1. Your reasoning (WHY you think this approach might work)
2. The mathematical formula
3. Your confidence level

Be creative but rigorous."""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=2000  # More tokens = more thinking
                )
                
                content = response.choices[0].message.content
                return self._parse_response(content)
                
            except Exception as e:
                if attempt < 2:
                    console.print(f"[yellow]API retry {attempt+1}[/yellow]")
                    time.sleep(2)
        
        return {'reasoning': 'Failed to generate', 'formula': '(n % 10)', 'confidence': 0}
    
    def initial_deep_prompt(self):
        """Initial prompt with deep mathematical context"""
        
        console.print("[bold cyan]═══ DEEP THINKING ITERATION 1 ═══[/bold cyan]\n")
        
        prompt = f"""You are trying to find a mathematical relationship between position n and the nth digit of π.

CONTEXT - What we know about π:

**Geometric Origins:**
- π = circumference/diameter of circle (2D)
- But π appears in sphere volumes (3D), hypersphere volumes (nD)
- Volume of n-dimensional ball: V_n(r) = π^(n/2) / Γ(n/2 + 1) × r^n

**Deeper Appearances:**
- Gaussian integral: ∫_-infinity^infinity e^(-x²) dx = √π
- Riemann zeta: ζ(2) = π²/6 = 1 + 1/4 + 1/9 + ...
- Heisenberg uncertainty: ΔxΔp ≥ ħ/(4π)
- Fourier transforms, complex analysis, everywhere

**What this suggests:**
π might encode information about:
- Spatial relationships in higher dimensions
- Wave interference patterns
- Number-theoretic structures
- Information content of geometric objects

**Your task:**
Generate a mathematical expression f(n) that maps position n → digit (0-9).

Think about:
1. Could n's position encode dimensional information?
2. Might there be modular arithmetic patterns (n mod p for primes)?
3. Could chaotic/fractal iteration reveal structure?
4. Does n's representation in different bases matter?
5. Quantum-like superposition of functions?

Provide:
1. **REASONING**: Why this approach might work (2-3 sentences)
2. **FORMULA**: One-line Python expression using n
3. **CONFIDENCE**: 0-100% how likely this is to beat random (10%)

Available: sin, cos, tan, log, exp, sqrt, abs, int, floor, ceil, pow, pi, e, n

Format:
REASONING: [your reasoning]
FORMULA: [expression]
CONFIDENCE: [0-100]"""

        return self.ask_with_reasoning(prompt, temp=0.5)
    
    def improve_with_context(self):
        """Improve using ALL previous history"""
        
        self.iteration += 1
        
        console.print(f"[bold cyan]═══ DEEP THINKING ITERATION {self.iteration + 1} ═══[/bold cyan]")
        console.print(f"[yellow]Best so far: {self.best_ever:.3f}%[/yellow]\n")
        
        history_summary = self._build_history_summary()
        failure_analysis = self._analyze_failures()
        
        temp = min(0.85, 0.5 + self.iteration * 0.02)
        
        # NOTE: Doubled {{ }} to escape them in f-string
        prompt = f"""You have tried {len(self.all_formulas)} approaches to predict π digits. None beat random chance (10%).

    **COMPLETE HISTORY:**
    {history_summary}

    **FAILURE ANALYSIS:**
    {failure_analysis}

    **DEEP MATHEMATICAL THINKING:**

    π is not random - it's deterministic. The digits MUST be generated by some principle.

    Consider unexplored approaches:

    1. **Higher-dimensional geometry:**
    - Could n represent a point in hyperdimensional space?
    - Volume formulas for n-spheres involve π^(n/2)
    - Might digit encode dimensional projections?

    2. **Number theory:**
    - BBP formula: π = Σ [1/16^k × (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))]
    - This shows π has base-16 structure
    - Try modular arithmetic: (n mod p) for various primes
    - Look at n prime factorization

    3. **Chaos/Fractals:**
    - Logistic map: x_next = r·x_n·(1-x_n)
    - Iterate functions using n as seed
    - Extract digit from chaotic attractor

    4. **Quantum-inspired:**
    - Superposition: combine multiple functions
    - Interference: sin(n·a) + sin(n·b) creates beats

    5. **Information theory:**
    - π digits should maximize entropy
    - Could they encode maximum information under geometric constraint?

    6. **Cross-domain:**
    - Digit sum of n
    - n in different bases
    - Geometric/harmonic means

    **Try something RADICALLY different.**

    Provide:
    REASONING: [Deep explanation]
    FORMULA: [Python expression]
    CONFIDENCE: [0-100]"""

        return self.ask_with_reasoning(prompt, temp)    
    def _build_history_summary(self) -> str:
        """Summarize ALL previous attempts"""
        
        if len(self.all_formulas) <= 10:
            # Show all if few
            return "\n".join([
                f"{i+1}. {f['formula'][:70]} → {f['accuracy']:.2f}% (confidence: {f.get('confidence', '?')}%)"
                for i, f in enumerate(self.all_formulas)
            ])
        else:
            # Show best and recent
            sorted_formulas = sorted(self.all_formulas, key=lambda x: x['accuracy'], reverse=True)
            best_5 = sorted_formulas[:5]
            recent_5 = self.all_formulas[-5:]
            
            summary = "**Top 5 performers:**\n"
            summary += "\n".join([f"  {f['formula'][:70]} → {f['accuracy']:.2f}%" for f in best_5])
            summary += "\n\n**Most recent 5:**\n"
            summary += "\n".join([f"  {f['formula'][:70]} → {f['accuracy']:.2f}%" for f in recent_5])
            
            return summary
    
    def _analyze_failures(self) -> str:
        """Analyze WHY formulas fail"""
        
        if not self.all_results:
            return "No failure data yet."
        
        # Look at what digits are predicted vs actual
        recent_results = self.all_results[-3:] if len(self.all_results) >= 3 else self.all_results
        
        analysis = []
        for result_set in recent_results:
            predicted_dist = {}
            actual_dist = {}
            
            for r in result_set[:20]:
                pred = r.get('predicted', 0)
                actual = r['actual']
                predicted_dist[pred] = predicted_dist.get(pred, 0) + 1
                actual_dist[actual] = actual_dist.get(actual, 0) + 1
            
            analysis.append(f"Predicted distribution: {predicted_dist}")
            analysis.append(f"Actual distribution: {actual_dist}")
        
        return "\n".join(analysis[:6])  # Limit length
    
    def _parse_response(self, content: str) -> Dict:
        """Extract reasoning, formula, confidence from response"""
        
        result = {
            'reasoning': '',
            'formula': '(n % 10)',
            'confidence': 50
        }
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=FORMULA:|$)', content, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        # Extract formula
        formula_match = re.search(r'FORMULA:\s*(.+?)(?=CONFIDENCE:|$)', content, re.DOTALL | re.IGNORECASE)
        if formula_match:
            formula_text = formula_match.group(1).strip()
            # Clean up formula
            formula_text = formula_text.replace('```python', '').replace('```', '').strip()
            lines = [l.strip() for l in formula_text.split('\n') if l.strip() and not l.strip().startswith('#')]
            if lines:
                result['formula'] = lines[0]
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', content, re.IGNORECASE)
        if conf_match:
            result['confidence'] = int(conf_match.group(1))
        
        return result

# ============================================================================
# MAIN DEEP EVOLUTION
# ============================================================================

def deep_evolve():
    console.print(Panel.fit(
        "[bold cyan]π DEEP STRUCTURE SEARCH - NUCLEAR MODE[/bold cyan]\n\n"
        "[yellow]Features:[/yellow]\n"
        "[green]• Chain-of-thought reasoning[/green]\n"
        "[green]• Higher-dimensional thinking[/green]\n"
        "[green]• Complete formula memory[/green]\n"
        "[green]• Multiple candidates per iteration[/green]\n"
        "[green]• Meta-analysis of failures[/green]\n\n"
        "[bold red]WE'RE NOT STOPPING UNTIL WE FIND SOMETHING[/bold red]",
        border_style="cyan"
    ))
    
    if not PI_FILE.exists():
        console.print(f"[red]{PI_FILE} not found[/red]")
        sys.exit(1)
    
    console.print(f"\n[cyan]Sample size: {SAMPLE_SIZE:,} per test[/cyan]")
    console.print(f"[cyan]Candidates per iteration: {CANDIDATES_PER_ITERATION}[/cyan]")
    console.print(f"[cyan]Max iterations: {MAX_ITERATIONS}[/cyan]\n")
    
    reader = FastPiReader(PI_FILE)
    thinker = DeepThinker(GROQ_API_KEY)
    
    console.print("[yellow]═══ DEEP EVOLUTION START ═══[/yellow]\n")
    
    # Initial deep thinking
    response = thinker.initial_deep_prompt()
    
    console.print(Panel(response['reasoning'], title="[cyan]Initial Reasoning[/cyan]", border_style="cyan"))
    console.print(f"[green]Formula:[/green] [yellow]{response['formula']}[/yellow]")
    console.print(f"[blue]Confidence:[/blue] {response['confidence']}%\n")
    
    console.print(f"[cyan]Testing on {SAMPLE_SIZE:,} positions...[/cyan]")
    accuracy, results, errors = evaluate_formula_fast(response['formula'], reader, SAMPLE_SIZE)
    
    if errors > SAMPLE_SIZE * 0.1:
        console.print(f"[red]⚠️  {errors} errors[/red]")
    
    console.print(f"[yellow]Accuracy: {accuracy:.3f}%[/yellow]\n")
    
    thinker.all_formulas.append({
        'iteration': 1,
        'formula': response['formula'],
        'reasoning': response['reasoning'],
        'confidence': response['confidence'],
        'accuracy': accuracy,
        'errors': errors
    })
    thinker.all_results.append(results)
    
    best_formula = response['formula']
    best_accuracy = accuracy
    best_reasoning = response['reasoning']
    thinker.best_ever = accuracy
    
    # Evolution loop with multiple candidates
    for iteration in range(1, MAX_ITERATIONS):
        if best_accuracy >= MIN_ACCURACY:
            console.print(f"[bold green]🎉 TARGET REACHED![/bold green]\n")
            break
        
        console.print(f"[bold yellow]Generating {CANDIDATES_PER_ITERATION} candidates...[/bold yellow]\n")
        
        candidates = []
        
        for candidate_num in range(CANDIDATES_PER_ITERATION):
            response = thinker.improve_with_context()
            
            console.print(f"[dim]Candidate {candidate_num + 1}:[/dim]")
            console.print(Panel(response['reasoning'][:200] + "...", border_style="dim"))
            console.print(f"[green]Formula:[/green] [yellow]{response['formula'][:80]}[/yellow]")
            console.print(f"[blue]AI Confidence:[/blue] {response['confidence']}%\n")
            
            # Test it
            console.print(f"[cyan]Testing...[/cyan]")
            start = time.time()
            accuracy, results, errors = evaluate_formula_fast(response['formula'], reader, SAMPLE_SIZE)
            elapsed = time.time() - start
            
            if errors > SAMPLE_SIZE * 0.1:
                console.print(f"[red]⚠️  {errors} errors[/red]")
            
            console.print(f"[green]Done in {elapsed:.2f}s[/green]")
            console.print(f"[yellow]Accuracy: {accuracy:.3f}%[/yellow]\n")
            
            candidates.append({
                'formula': response['formula'],
                'reasoning': response['reasoning'],
                'confidence': response['confidence'],
                'accuracy': accuracy,
                'errors': errors,
                'results': results
            })
            
            thinker.all_formulas.append({
                'iteration': iteration + 1,
                'candidate': candidate_num + 1,
                'formula': response['formula'],
                'reasoning': response['reasoning'],
                'confidence': response['confidence'],
                'accuracy': accuracy,
                'errors': errors
            })
            thinker.all_results.append(results)
            
            time.sleep(0.5)  # Brief pause between candidates
        
        # Pick best candidate
        best_candidate = max(candidates, key=lambda x: x['accuracy'])
        
        console.print(f"[bold cyan]Best candidate this iteration:[/bold cyan]")
        console.print(f"[yellow]Accuracy: {best_candidate['accuracy']:.3f}%[/yellow]")
        console.print(f"[green]Formula: {best_candidate['formula'][:80]}[/green]\n")
        
        if best_candidate['accuracy'] > best_accuracy:
            improvement = best_candidate['accuracy'] - best_accuracy
            best_accuracy = best_candidate['accuracy']
            best_formula = best_candidate['formula']
            best_reasoning = best_candidate['reasoning']
            thinker.best_ever = max(thinker.best_ever, best_accuracy)
            
            console.print(f"[bold green]⬆⬆⬆ NEW RECORD! +{improvement:.3f}% ⬆⬆⬆[/bold green]\n")
            console.print(Panel(best_reasoning, title="[green]Winning Reasoning[/green]", border_style="green"))
            console.print()
        else:
            console.print(f"[dim]No improvement. Best remains: {best_accuracy:.3f}%[/dim]\n")
    
    # Final test
    console.print("\n[bold yellow]═══ FINAL VALIDATION ═══[/bold yellow]\n")
    console.print(f"[cyan]Best formula found:[/cyan]\n[yellow]{best_formula}[/yellow]\n")
    console.print(Panel(best_reasoning, title="[cyan]Reasoning Behind Best Formula[/cyan]", border_style="cyan"))
    
    console.print(f"\n[cyan]Final test on {SAMPLE_SIZE * 2:,} positions...[/cyan]\n")
    final_acc, final_results, final_errors = evaluate_formula_fast(best_formula, reader, SAMPLE_SIZE * 2)
    
    console.print(f"[bold green]FINAL ACCURACY: {final_acc:.3f}%[/bold green]\n")
    
    # Display comprehensive results
    table = Table(title=f"Evolution History ({len(thinker.all_formulas)} total attempts)")
    table.add_column("Iter", width=6)
    table.add_column("Formula", style="yellow")
    table.add_column("AI Conf", width=8)
    table.add_column("Accuracy", width=10)
    
    for f in thinker.all_formulas[:30]:  # Show first 30
        formula_short = f['formula'][:50] + "..." if len(f['formula']) > 50 else f['formula']
        acc = f['accuracy']
        conf = f.get('confidence', 0)
        
        iter_str = f"{f['iteration']}"
        if 'candidate' in f:
            iter_str += f".{f['candidate']}"
        
        color = "bold green" if acc >= MIN_ACCURACY else "green" if acc > 12 else "yellow" if acc > 10 else "red"
        
        table.add_row(
            iter_str,
            formula_short,
            f"{conf}%",
            f"[{color}]{acc:.2f}%[/{color}]"
        )
    
    console.print(table)
    console.print()
    
    # Verdict
    if final_acc >= MIN_ACCURACY:
        verdict = (
            f"[bold green]{'='*70}[/bold green]\n"
            f"[bold green]🎉🎉🎉 BREAKTHROUGH ACHIEVED! 🎉🎉🎉[/bold green]\n"
            f"[bold green]{'='*70}[/bold green]\n\n"
            f"[bold white]FINAL ACCURACY: {final_acc:.3f}%[/bold white]\n\n"
            f"[cyan]WINNING FORMULA:[/cyan]\n[yellow]{best_formula}[/yellow]\n\n"
            f"[cyan]REASONING:[/cyan]\n{best_reasoning}\n\n"
            f"[green]After {len(thinker.all_formulas)} attempts with deep reasoning,[/green]\n"
            f"[green]we found a formula that beats random chance![/green]\n\n"
            f"[bold yellow]⚠️  REQUIRES RIGOROUS PEER REVIEW ⚠️[/bold yellow]"
        )
    elif final_acc > 12:
        verdict = (
            f"[yellow]INTERESTING SIGNAL DETECTED[/yellow]\n\n"
            f"Accuracy: {final_acc:.3f}% (above random 10%)\n\n"
            f"Formula: {best_formula}\n\n"
            f"Reasoning: {best_reasoning[:200]}...\n\n"
            f"Not conclusive, but suggests possible weak structure."
        )
    else:
        verdict = (
            f"[red]NO PREDICTIVE PATTERN FOUND[/red]\n\n"
            f"Best: {final_acc:.3f}%\n\n"
            f"After {len(thinker.all_formulas)} deep thinking attempts,\n"
            f"including higher-dimensional reasoning and complete history feedback,\n"
            f"no formula significantly beats random chance.\n\n"
            f"[yellow]This strongly suggests π digits are genuinely unpredictable\n"
            f"by mathematical formulas accessible to current AI.[/yellow]\n\n"
            f"[dim]π remains undefeated.[/dim]"
        )
    
    console.print(Panel(verdict, border_style="white", padding=1))
    
    # Save comprehensive results
    report = {
        'timestamp': datetime.now().isoformat(),
        'final_accuracy': final_acc,
        'best_formula': best_formula,
        'best_reasoning': best_reasoning,
        'total_attempts': len(thinker.all_formulas),
        'iterations': MAX_ITERATIONS,
        'sample_size': SAMPLE_SIZE,
        'candidates_per_iteration': CANDIDATES_PER_ITERATION,
        'all_formulas': thinker.all_formulas,
        'top_10': sorted(thinker.all_formulas, key=lambda x: x['accuracy'], reverse=True)[:10]
    }
    
    output_file = OUTPUT_DIR / f"deep_thinking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    console.print(f"\n[dim]💾 Complete report saved to {output_file}[/dim]")
    
    del reader

# ============================================================================
# MAIN
# ============================================================================

def main():
    console.print("""
[bold cyan]
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   π DEEP STRUCTURE SEARCH                                ║
║   NUCLEAR MODE - MAXIMUM THINKING DEPTH                  ║
║                                                          ║
║   "If a pattern exists, we WILL find it."                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
[/bold cyan]
""")
    
    deep_evolve()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Evolution interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()