### Evaluation of Pulse Level Quantum Fourier Models

Quantum Fourier Models (QFMs) are implemented at the pulse level using Qiskit Dynamics to explore how variations in pulse parameters ϕ affect their partial Fourier series representation.

## Context

In the realm of quantum machine learning, quantum models can generally be described as a partial Fourier series, where the data encoding gates influence the accessible frequencies, while their parameters, often denoted as $\theta$, determine the amplitudes and phases of these frequencies [[1]](https://arxiv.org/abs/2008.08605). These quantum models will henceforth be referred to as quantum fourier models (QFMs).
 
On hardware platforms such as superconducting circuits or trapped ions, the execution of quantum algorithms relies on precisely shaped and timed electromagnetic pulses. Finer control over the qubit's time evolution is achieved at the pulse level by varying pulse parameters, denoted as $\phi$.

The goal of this project is twofold:

First, to implement QFMs at the pulse level. This involves defining the underlying Hamiltonians and tuning pulse parameters $\phi$ to realize the desired data encoding circuits. The Qiskit Dynamics module, which provides a high degree of control over pulse definitions, is used for this implementation [[2]](https://joss.theoj.org/papers/10.21105/joss.05853).

Secondly, to evaluate the impact of varying pulse-level parameters $\phi$ on the resulting partial Fourier series representation of the quantum model's output, while keeping the abstract gate-level parameters $\theta$ constant.


[1] https://arxiv.org/abs/2008.08605 (The effect of data encoding on the expressive power of variational quantum machine learning models, Schuld 2020)

[2] https://joss.theoj.org/papers/10.21105/joss.05853 (Qiskit Dynamics: A Python package for simulating the time dynamics of quantum systems)

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/cirKITers/pulse-fourier.git
    cd pulse-fourier
    ```

2. Create a virtual environment and activate it:
    - On Windows:
      ```bash
      python -m venv .venv
      .\.venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```

3. Install the project in editable mode:
    ```bash
    pip install -e .
    ```


