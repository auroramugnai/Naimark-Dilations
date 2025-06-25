"""
This script sets up and solves an SDP using PICOS to test whether a 
quantum ensemble of (n_elements=3) states in a (dim=3)-dimensional Hilbert space can reproduce a target 
classical guessing probability p_classical=0.75 under a fixed binary projective measurement, 
with M0 = |0⟩⟨0| and M1 = |1⟩⟨1| + |2⟩⟨2|.

Optimization variables:
- φ₀, φ₁, φ₂: quantum states (density matrices), constrained to be positive semidefinite 
  with trace equal to their associated probability.
- p₀, p₁, p₂: probabilities associated with each state, constrained to be non-negative and sum to 1.
- ρ_S: the average state, defined as the convex combination of the φ_λ, constrained to be 
  a valid density matrix.

The program maximizes the guessing probability 
p_guess = ∑_λ max{Tr[φ_λ M0], Tr[φ_λ M1]} 
under the constraint that this total equals the classical value (p_classical=0.75). 

Outputs:
- The optimal value of the guessing probability (should be 0.75).
- The optimal average state ρ_S achieving this value under the constraints.
"""

import numpy as np
import picos as pic
from sys import exit


def setNumericalPrecisionForSolver(problem):
    problem.options["rel_ipm_opt_tol"]=10**-9
    problem.options["rel_prim_fsb_tol"]=10**-9
    problem.options["rel_dual_fsb_tol"]=10**-9
    problem.options["max_footprints"]=None

# === Define |0⟩, |1⟩, |2⟩ in the 3D Hilbert space ===
ket_0 = np.array([1, 0, 0])
ket_1 = np.array([0, 1, 0])
ket_2 = np.array([0, 0, 1])

# === Projectors |i⟩⟨i| ===
proj_0 = np.outer(ket_0, ket_0)  # |0><0|
proj_1 = np.outer(ket_1, ket_1)  # |1><1|
proj_2 = np.outer(ket_2, ket_2)  # |2><2|


# === Measurement operators ===
M0 = proj_0                     # First measurement: dyad[0]
M1 = proj_1 + proj_2            # Second measurement: dyad[1] + dyad[2]
MS = np.array([M0, M1])         # Stack them as a NumPy array


# === Set up the semidefinite program ===
problem = pic.Problem()

dim = 3                  # Dimension of the Hilbert space
n_elements = 3           # Number of states
p_classical = 0.75       # Target classical guessing probability


# === Define probability variables for each state ===
probabilities = [pic.RealVariable(f'p_{lamda}', lower=0) for lamda in range(n_elements)]

# Probabilities must be non-negative and sum to 1.
for p in probabilities:
    problem.add_constraint(p >= 0)
problem.add_constraint(sum(probabilities)==1)


# === Define quantum states (density matrices) and constraints ===
states = [pic.HermitianVariable(f'phi_{lamda}', (dim,dim)) for lamda in range(n_elements)]

# Each state must be positive semidefinite and have trace equal to its probability.
for i,p in enumerate(probabilities):
    problem.add_constraint(states[i] >> 0)
    problem.add_constraint(pic.trace(states[i]) == p)


# === Define average state (rho) and constraints ===
rhos = pic.HermitianVariable('rhos', (dim,dim))

problem.add_constraint(rhos >> 0)                  # Positive semidefinite
problem.add_constraint(pic.trace(rhos) == 1)       # Trace-1 
problem.add_constraint(sum(states) == rhos)        # Average over states


# === Set p_guess equal to classical guessing probability ===
# Since PICOS does not support equality constraints involving non-affine expressions 
# such as pic.max(...), we cannot directly write p_guess as a sum of maxima.
# Conceptually, for each state we want to take the larger value between 
# trace(state * M0) and trace(state * M1).
p_guess = 0
for state in states:
    t0 = pic.trace(state * MS[0])
    t1 = pic.trace(state * MS[1])
    if (t0 >= t1):
        p_guess += t0
    else:
        p_guess += t1

problem.add_constraint(p_guess == p_classical) # Comment this and fix the rhos found by the sdp 
                                               # to check that p_guess == p_classical.


# === Objective: Maximize P guess ===
problem.set_objective('max', p_guess)


# === Solve the problem ===
# prob.set_objective('max', obj)
setNumericalPrecisionForSolver(problem)
problem.solve(solver="mosek", verbosity=0)


# === Output results ===
print("\nOptimal objective value:", problem.value)

print("\nProbabilities:")
for i, p in enumerate(probabilities):
    print(f"  p_{i} = {p.value:.6f}")

print("\nQuantum states:")
for i, phi in enumerate(states):
    print(f"  phi_{i} =\n{phi.value}\n")

print("Average state rho =\n", rhos.value)

# This is what you should get:
# 
# rho = np.array([
#                 [0.75, 0, 0],
#                 [0, 0.124, 0],
#                 [0, 0, 0.126]
# ])
