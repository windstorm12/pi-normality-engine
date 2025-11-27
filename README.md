## Project Structure

*   **`normality_proof.py`**: A comprehensive engine testing π against 8 distinct mathematical frameworks (Ergodic Gauss maps, Weyl's Equidistribution, Scale Invariance, etc.).
*   **`convergent_errors.py`**: Tests the "Convergent Error Theory," analyzing if the errors of π's rational approximations are uniformly distributed.
*   **`experimental_harness.py`**: A research workbench that generates digits on-the-fly to run spectral tests (Gowers $U^2$), Martingale betting strategies, and resonant-base correlations (Hex vs Decimal).
*   **`compression_scan.py`**: Scans digit sequences using compression algorithms (zlib, lzma, bz2) to detect low-entropy anomalies.
*   **`pi_evolution.py`**: An AI-powered script that uses the Groq API to iteratively "evolve" mathematical formulas attempting to predict the nth digit of π.

## Mathematical Concepts Used

*   **Gowers $U^2$ Norm:** Used in the experimental harness to test for additive structure in the digits (higher-order Fourier analysis).
*   **Martingale Difference:** A betting test (Kelly criterion) to determine if a gambler could make money betting against the uniform distribution of digits.
*   **Resonant Base Correlation:** Investigating if patterns in Base-16 (Hex) π leak into Base-10 (Decimal) π at specific logarithmic intervals.
*   **Continued Fractions:** Analyzing the distribution of partial quotients (Gauss-Kuzmin statistics).
*   **Ergodic Theory:** Testing invariance under the Gauss map $T(x) = \{1/x\}$.
