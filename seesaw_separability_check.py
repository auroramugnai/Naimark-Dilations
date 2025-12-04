"""
This script builds an entangled basis for the bipartite system S+A, constructs the corresponding
entangled density matrix, and runs a seesaw optimization algorithm to test whether there exists
a separable rho_SA that reproduces both the same reduced state on S and the same measurement on S.

Note on types: 
    In the main code, all matrices are kept as numpy.ndarray, and all functions take numpy arrays as input.
    Within the functions, inputs are automatically converted to qutip.Qobj when quantum operations (such as partial trace, 
    tensor, or dagger) are needed,  and converted back to numpy.ndarray afterward -- 
    except when PICOS objects are required for SDP optimization.
"""
import matplotlib.pyplot as plt
import numpy as np
import picos as pic
from picos.modeling.problem import Problem
import qutip as qt
from qutip import Qobj
from tqdm import tqdm
from typing import List, Tuple, Union, Any

# ----------------------------------------------------------------------
# Configure numerical precision for PICOS solvers
# ----------------------------------------------------------------------
def setNumericalPrecisionForSolver(problem: Problem) -> None:
    """
    Set high numerical-precision options for a PICOS Problem object.
    Parameters
    ----------
        problem : Problem
            A PICOS Problem instance whose `options` dict will be modified in-place.
    Returns
    -------
        None
    """
    problem.options["rel_ipm_opt_tol"] = 1e-14
    problem.options["rel_prim_fsb_tol"] = 1e-14
    problem.options["rel_dual_fsb_tol"] = 1e-14
    problem.options["max_footprints"] = None
    
################################################################################################
################################################################################################
def buildEntangledBasis(theta: float, verbosity: int = 0) -> List[Qobj]:
    """Build an entangled basis states for the bipartite system S+A.
    Parameters
    ----------
        theta : float
            The complex phase parameter appearing in the linear combination that defines
            the entangled basis states.
        verbosity : int, optional
            If 1, the function prints information.
    Returns
    -------
        basis : List[Qobj]
            A list containing 4 `qutip.Qobj` kets (each of shape 4x1) representing
            the entangled basis states.
    """
    eta=[1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)]
    phi=[np.pi/4,7*np.pi/4,3*np.pi/4,5*np.pi/4]
    mPos=[np.sqrt((1+eta[b])/2)*np.exp(-1j*phi[b]/2)*qt.basis(2,0)+np.sqrt((1-eta[b])/2)*np.exp(1j*phi[b]/2)*qt.basis(2,1) 
          for b in range(4)]
    mNeg=[np.sqrt((1-eta[b])/2)*np.exp(-1j*phi[b]/2)*qt.basis(2,0)-np.sqrt((1+eta[b])/2)*np.exp(1j*phi[b]/2)*qt.basis(2,1) 
          for b in range(4)]
    
    basis = [(np.sqrt(3)+np.exp(1j*theta))/(2*np.sqrt(2))*qt.tensor(mPos[b],mNeg[b])+
            (np.sqrt(3)-np.exp(1j*theta))/(2*np.sqrt(2))*qt.tensor(mNeg[b],mPos[b])
            for b in range(4)]
    
    #### Print the basis. ####
    if verbosity:
        for i, psi in enumerate(basis):
            print(f"--- State Ψ_{i} ---")
            vec = psi.full().flatten()  # 4x1 -> 1D array
            for j, amp in enumerate(vec):
                # j=0 -> |00>, j=1 -> |01>, j=2 -> |10>, j=3 -> |11>
                ket_label = f"|{j//2}{j%2}>"
                magnitude = np.abs(amp)
                phase = np.angle(amp)
                print(f"{ket_label}: amplitude={magnitude:.3f}, phase={phase:.3f} rad")
            print()

    return basis

################################################################################################
################################################################################################
def check_SA_entanglement(rhoSA: Qobj) -> None:
    """Check if 'rhoSA' is entangled by computing the Von Neumann entropies of the subsystems.
    Parameters:
    ----------
    rhoSA : Qobj
        The density matrix (4x4) of the combined system S+A.
    Raises:
    ------
    AssertionError
        If the entropies of S and A are not equal or if they are zero.
    """
    # Partial traces
    rhoS = rhoSA.ptrace(0)
    rhoA = rhoSA.ptrace(1)

    # Von Neumann entropies
    S_S = qt.entropy_vn(rhoS)
    S_A = qt.entropy_vn(rhoA)
    
    assert np.isclose(S_S, S_A, atol=1e-8, rtol=1e-8), "!!! Entropies of S and A are not equal."
    assert S_S > 0, "!!! Entropy is zero, state is not entangled."
    assert S_A > 0, "!!! Entropy is zero, state is not entangled."

################################################################################################
################################################################################################
def find_entangled_rhoSA(basis: List[Qobj]) -> Tuple[Qobj, np.ndarray]:   
    """Look for a probability distribution 'probabilities' that gives an entangled 'rhoSA'. 
       To do this, randomly generate probability distributions until you find 
       one that gives a Negative Partial Transpose (NPT) state.
    Parameters:
    ----------
        basis : List[Qobj]
            The list of entangled basis states Ψ_i.
    Returns:
    -------
        rhoSA : Qobj
            The entangled density matrix of the combined system S+A.
        probabilities : np.ndarray
            The probability distribution (length 4) used to construct rhoSA.  
    """
    while True:
        #### Generate a probability distribution ####
        probabilities = np.random.rand(4)
        probabilities /= probabilities.sum()
        assert np.isclose(probabilities.sum(), 1, atol=1e-8, rtol=1e-8), "Probabilities do not sum to 1."

        #### Built rhoSA from 'basis' and 'probabilities'. ####
        rhoSA = sum([p * (psi * psi.dag()) for psi, p in zip(basis, probabilities)])
        
        #### Check PPT criterion. ####
        rhoSA_PT = qt.partial_transpose(rhoSA, [0,1])
        
        eigvals = rhoSA_PT.eigenenergies()
        if np.any(eigvals < -1e-12):
            # Entangled rhoSA found.
            break

    return rhoSA, probabilities

################################################################################################
################################################################################################
def find_sep_states_max_p_guess(rhoS: np.ndarray, 
                                PiSA_list: List[np.ndarray], 
                                verbosity: int = 1          ) -> Tuple[float, List[np.ndarray]]:
    """Find the set of states {rhoSA_i} that maximizes the guessing probability
       given a fixed reduced state rhoS and measurement PiSA.
    Parameters:
    ----------
        rhoS : np.ndarray
            The reduced density matrix (2x2) of system S.
        PiSA : List[np.ndarray]
            A list of `dim` measurement operators on S+A represented as 4x4 numpy arrays.
        verbosity : int, optional
    Returns:
    -------
        p_guess : float
            The optimal (maximized) guessing probability.
        rhoSA_list : List[np.ndarray]
            The list of optimal states rhoSA_i.
    """
    dim = 4 # Dimension of the total system S+A
    problem = pic.Problem()
    

    # -------------------- Variables -------------------- #
    rhoSA_list = [pic.HermitianVariable(f"|Ψ_{i}XΨ_{i}|", shape=(dim, dim)) for i in range(dim)]
    rhoSA = pic.sum(rhoSA_list)


    # ------------------- Constraints ------------------- #
    # Positivity and normalization.
    problem.add_list_of_constraints([r >> 0 for r in rhoSA_list])
    problem.add_constraint(pic.trace(rhoSA) == 1)

    # Tr_A(rhoSA) = rhoS.
    problem.add_constraint(pic.partial_trace(rhoSA, 1) == rhoS)

    # rhoSA must be separable (<-> PPT)
    problem.add_constraint(sum([rhoSA_i.partial_transpose(1) for rhoSA_i in rhoSA_list]) >> 0)


    # -------------------- Objective -------------------- #
    # maximize p_guess = Σ Tr(Pi ρ_i)
    objective = pic.Constant(0) 
    for rho_i, PiSA_i in zip(rhoSA_list, PiSA_list):
        objective += pic.trace(rho_i * np.array(PiSA_i))

    problem.set_objective("max", objective)
    # print("\n---- Optimization on rhoSA ----\n", problem)
    setNumericalPrecisionForSolver(problem)
    problem.solve(solver="mosek", verbosity=0)

    return objective.value.real, [rhoSA_i.value_as_matrix for rhoSA_i in rhoSA_list]

################################################################################################
################################################################################################
def find_meas_max_p_guess(M_S: List[np.ndarray],
                          rhoSA_list: List[np.ndarray],
                          verbosity: int = 1                ) -> Tuple[float, List[np.ndarray]]:
    """Find the set of measurement operators {PiSA_i} that maximizes the guessing probability
       given a fixed set of states rhoSA_i.
    Parameters:
    ----------
        M_S : List[ np.ndarray]
            A list of `dim` reduced measurement operators on system S. Elements can be
            provided as `qutip.Qobj` or as 2x2 numpy arrays.
        rhoSA_list : List[np.ndarray]
            A list of the states rhoSA_i as numpy matrices (4x4), used
            to compute the ancilla reduced state rhoA = sum_i Tr_S[rhoSA_i].
        verbosity : int, optional
    Returns:
    -------
        p_guess : float
            The optimal guessing probability.
        PiSA_list : List[np.ndarray]
            The list of optimal POVM elements PiSA_i (4x4) returned by PICOS.
    """
    dim = 4 # Dimension of the total system S+A
    problem = pic.Problem()


    # -------------------- Variables  --------------------#
    PiSA_list = [pic.HermitianVariable(f"PiSA[{i}]", shape=(4, 4)) for i in range(dim)]
    

    # ------------------- Constraints ------------------- #
    # Positivity and completeness.
    problem.add_list_of_constraints([PiSA_i >> 0 for PiSA_i in PiSA_list])
    problem.add_constraint(sum(PiSA_list) == np.eye(dim))

    # Impose as a constraint TrA[PiSA_i * (1⊗rhoA)] = MS_i
    I2 = pic.Constant(np.eye(2))

    rhoSA_pic = [pic.Constant(r) for r in rhoSA_list]
    rhoA = [pic.partial_trace(r, 0) for r in rhoSA_pic]
    rhoA = pic.sum(rhoA)

    for i, MS_i in enumerate(M_S):  
        expr = pic.partial_trace(PiSA_list[i] * (I2 @ rhoA), 1)
        problem.add_constraint(expr == MS_i)


    # -------------------- Objective -------------------- #
    # maximize p_guess = Σ Tr(Pi ρ_i)
    objective = pic.Constant(0) 
    for rho_i, PiSA_i in zip(rhoSA_list, PiSA_list):
        objective += pic.trace(rho_i * PiSA_i)

    problem.set_objective("max", objective)
    # print("\n---- Optimization on PiSA ----\n", problem)
    setNumericalPrecisionForSolver(problem)
    problem.solve(solver="mosek", verbosity=0)

    return objective.value.real, [PiSA_i.value_as_matrix for PiSA_i in PiSA_list]


################################################################################################
#############################            MAIN              #####################################
################################################################################################
if __name__ == "__main__":

    # -------------------- State -------------------- #
    # Build a basis |psi_i⟩_SA.
    basis = buildEntangledBasis(theta=0, verbosity=0)

    # From the basis, build an entangled rhoSA.
    rhoSA, probabilities = find_entangled_rhoSA(basis)
    assert np.isclose(sum(probabilities), 1), "!!! Probabilities do not sum to 1."

    rhoSA_list = [p * (psi * psi.dag()).full() for psi, p in zip(basis, probabilities)]
    assert np.isclose(sum([np.trace(rhoSA_i) for rhoSA_i in rhoSA_list]), 1), "!!! rhoSA doesn't have trace 1."

    # Check entanglement of rhoSA.
    check_SA_entanglement(rhoSA)

    # rhoS = Tr_A(rhoSA) and rhoA = Tr_S(rhoSA).
    rhoS = rhoSA.ptrace(0).full()
    rhoA = rhoSA.ptrace(1).full()


    # ----------------- Measurement ----------------- #
    # PiSA_i = |Ψ_i⟩⟨Ψ_i|.
    PiSA = [psi * psi.dag() for psi in basis]
    PiSA_list = [PiSA_i.full() for PiSA_i in PiSA]

    # M_S_i = TrA[PiSA_i * (1⊗rhoA)]
    I2 = qt.qeye(2)
    x = qt.tensor(I2, qt.Qobj(rhoA))
    M_S = [((PiSA_i * x).ptrace(0)).full() for PiSA_i in PiSA]


    # ------------- Seesaw optimization ------------- #
    precision = 1e-7
    attempts = 10
    diffs = []
    new_p_guess = 0
    for _ in tqdm(range(attempts)):
        
        old_p_guess, PiSA_list = find_meas_max_p_guess(M_S, rhoSA_list, verbosity=1) 
        PiSA_list = [np.array(P) for P in PiSA_list]  # convert to numpy again
        diffs.append(new_p_guess - old_p_guess) # Collect data for the plot.
        
        new_p_guess, rhoSA_list = find_sep_states_max_p_guess(rhoS, PiSA_list, verbosity=1)
        rhoSA_list = [np.array(r) for r in rhoSA_list]  # convert to numpy again
        diffs.append(new_p_guess - old_p_guess) # Collect data for the plot.

        if abs(new_p_guess - old_p_guess) < precision:
            print(f"Converged guessing probability: {new_p_guess}")
            break

    print(f"Final guessing probability after {len(diffs)} iterations: {new_p_guess}, with precision {abs(new_p_guess - old_p_guess)}.")


    # --------------- Convergence plot  --------------- #
    plt.figure()
    plt.plot(diffs, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Difference in Guessing Probability')
    plt.show()  

