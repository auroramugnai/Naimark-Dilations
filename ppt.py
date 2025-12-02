import numpy as np
import picos as pic
from picos.modeling.problem import Problem
import qutip as qt
from qutip import Qobj

def buildEntangledBasis(theta, verbosity=0):
    """Build the entangled basis states Ψ_i for SA as described in the paper "Quantifying...".
    Parameters:
    ----------
        theta : float
            The angle parameter in the basis states.
        verbosity : int
            Level of verbosity for printing the basis states.
    Returns:
    -------
        basis : list of Qobj
            The list of entangled basis states Ψ_i.
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
def check_SA_entanglement(rho_SA):
    """Check if 'rho_SA' is entangled by computing the Von Neumann entropies of the subsystems.
    
    Parameters:
    ----------
    rho_SA : Qobj
        The density matrix of the combined system S+A.

    Raises:
    ------
    AssertionError
        If the entropies of S and A are not equal or if they are zero.
    """
    # Partial traces
    rho_S = rho_SA.ptrace(0)
    rho_A = rho_SA.ptrace(1)

    # Von Neumann entropies
    S_S = qt.entropy_vn(rho_S)
    S_A = qt.entropy_vn(rho_A)
    
    assert np.isclose(S_S, S_A, atol=1e-8, rtol=1e-8), "!!! Entropies of S and A are not equal."
    assert S_S > 0, "!!! Entropy is zero, state is not entangled."
    assert S_A > 0, "!!! Entropy is zero, state is not entangled."

################################################################################################
################################################################################################
def find_entangled_rho_SA(basis):
    """Look for a probability distribution 'probabilities' that gives an entangled 'rho_SA'. 
       To do this, randomly generate probability distributions until you find 
       one that gives a Negative Partial Transpose (NPT) state.
    Parameters:
    ----------
        basis : list of Qobj
            The list of entangled basis states Ψ_i.
    Returns:
    -------
        rho_SA : Qobj
            The entangled density matrix of the combined system S+A.
        probabilities : np.ndarray
            The probability distribution used to construct rho_SA.  
    """
    while True:
        #### Generate a probability distribution ####
        probabilities = np.random.rand(4)
        probabilities /= probabilities.sum()
        assert np.isclose(probabilities.sum(), 1, atol=1e-8, rtol=1e-8), "Probabilities do not sum to 1."

        #### Built rho_SA from 'basis' and 'probabilities'. ####
        rho_SA = sum([p * (psi * psi.dag()) for psi, p in zip(basis, probabilities)])
        
        #### Check PPT criterion. ####
        rho_SA_PT = qt.partial_transpose(rho_SA, [0,1])
        
        eigvals = rho_SA_PT.eigenenergies()
        if np.any(eigvals < -1e-12):
            print("\n*** Entangled rho_SA found.")
            break

    return rho_SA, probabilities

################################################################################################
################################################################################################
def find_sep_states_max_p_guess(rhoS, PiSA_list, verbosity=1):
    """Find the set of states {rhoSA_i} that maximizes the guessing probability
       given a fixed reduced state rho_S and measurement PiSA.
    Parameters:
    ----------
        rhoS : np.ndarray
            The reduced density matrix of system S.
        PiSA : list of Qobj
            The measurement operators on the combined system S+A.
    Returns:
    -------
        p_guess : float
            The optimal guessing probability.
        rhoSA_list : list of np.ndarray
            The list of optimal states rho_SA_i.
    """
    dim = 4 # Dimension of the total system S+A
    problem = pic.Problem()
    

    ###  --- Variables --- ###
    rhoSA_list = [pic.HermitianVariable(f"|Ψ_{i}XΨ_{i}|", shape=(dim, dim)) for i in range(dim)]
    rhoSA = pic.sum(rhoSA_list)


    ###  --- Constraints --- ###
    # Positivity and normalization.
    problem.add_list_of_constraints([r >> 0 for r in rhoSA_list])
    problem.add_constraint(pic.trace(rhoSA) == 1)

    # Tr_A(rho_SA) = rho_S.
    problem.add_constraint(pic.partial_trace(rhoSA, 1) == rhoS)

    # rhoSA must be separable: the partial transpose must be a positive matrix.
    problem.add_constraint(sum([rhoSA_i.partial_transpose(1) for rhoSA_i in rhoSA_list]) >> 0)


    ### --- Objective --- ###
    # maximize p_guess = Σ Tr(Pi ρ_i)
    objective = pic.Constant(0) 
    for rho_i, PiSA_i in zip(rhoSA_list, PiSA_list):
        objective += pic.trace(rho_i @ PiSA_i)

    problem.set_objective("max", objective)
    print("\n---- Optimization on rhoSA ----\n", problem)
    problem.solve(solver="mosek", verbosity=0)

    return objective.value.real, [rhoSA_i.value_as_matrix for rhoSA_i in rhoSA_list]

################################################################################################
################################################################################################
def find_meas_max_p_guess(M_S, rhoSA_list, verbosity=1):
    """Find the set of measurement operators {PiSA_i} that maximizes the guessing probability
       given a fixed set of states rho_SA_i.
    Parameters:
    ----------
        M_S : list of np.ndarray
            The reduced measurement operators on system S.
        rhoSA_list : list of np.ndarray
            The list of states rho_SA_i.
    Returns:
    -------
        p_guess : float
            The optimal guessing probability.
        PiSA_list : list of np.ndarray
            The list of optimal measurement operators PiSA_i.
    """
    dim = 4 # Dimension of the total system S+A
    problem = pic.Problem()


    ###  --- Variables --- ###
    PiSA_list = [pic.HermitianVariable(f"PiSA[{i}]", shape=(4, 4)) for i in range(dim)]
    

    ###  --- Constraints --- ###
    # Positivity and completeness.
    problem.add_list_of_constraints([PiSA_i >> 0 for PiSA_i in PiSA_list])
    problem.add_constraint(sum(PiSA_list) == np.eye(dim))

    # Impose as a constraint TrA[PiSA_i * (1⊗rhoA)] = MS_i
    I2 = pic.Constant(np.eye(2))
    for i, MS_i in enumerate(M_S):  

        # rhoA = TrS[rhoSA]
        rhoSA_pic = [pic.Constant(r) for r in rhoSA_list]
        rhoA = [pic.partial_trace(r, 0) for r in rhoSA_pic]
        rhoA = pic.sum(rhoA)
        print('type(rhoA):', type(rhoA))
        # TrA[PiSA_i * (1⊗rhoA)]
        expr = pic.partial_trace(PiSA_list[i] * (I2 @ rhoA), 1)
        problem.add_constraint(expr == MS_i)


    ### --- Objective --- ###
    # maximize p_guess = Σ Tr(Pi ρ_i)
    objective = pic.Constant(0) 
    for rho_i, PiSA_i in zip(rhoSA_list, PiSA_list):
        objective += pic.trace(rho_i @ PiSA_i)
    
    problem.set_objective("max", objective)
    print("\n---- Optimization on PiSA ----\n", problem)
    problem.solve(solver="mosek", verbosity=1)

    return objective.value.real, [PiSA_i.value_as_matrix for PiSA_i in PiSA_list]


################################################################################################
#############################            MAIN              #####################################
################################################################################################
if __name__ == "__main__":
    ### Build a basis |Ψ_i⟩ of SA.
    basis = buildEntangledBasis(theta=0, verbosity=1)

    ### From the basis, build an entangled rho_SA.
    # rho_SA, probabilities = find_entangled_rho_SA(basis)
    # rho_SA_list = [p * (psi * psi.dag()).full() for psi, p in zip(basis, probabilities)]

    ### Or use uniform probabilities ( like in the paper ).
    probabilities = [1/4]*4
    rhoSA_list = [p * (psi * psi.dag()).full() for psi, p in zip(basis, probabilities)]
    rho_SA = sum([p * (psi * psi.dag()) for psi, p in zip(basis, probabilities)])
        
    ### Check entanglement of rho_SA.
    check_SA_entanglement(rho_SA)

    ### rhoS = Tr_A(rho_SA).
    # rhoS = qt.ptrace(rho_SA, 0)
    rhoS = rho_SA.ptrace(0).full()
    # print("rhoSA:\n", rho_SA, "\nrhoS:\n", rhoS, "\ntype(rhoS):", type(rhoS), "type(rho_SA):", type(rho_SA))
    
    ### PiSA_i = |Ψ_i⟩⟨Ψ_i|.
    PiSA = [psi * psi.dag() for psi in basis]
    PiSA_list = [Pi.full() for Pi in PiSA]

    ### M_S_i = Tr_A(PiSA_i).
    M_S = [Pi.ptrace(0).full() for Pi in PiSA]
    
    ### Seesaw optimization ###
    precision = 1e-7
    while True:
        
        old_p_guess, PiSA_list = find_meas_max_p_guess(M_S, rhoSA_list, verbosity=1)
        print(f"Current guessing probability: {old_p_guess}") 
        

        new_p_guess, rhoSA_list = find_sep_states_max_p_guess(rhoS, PiSA_list, verbosity=1)
        print(f"New guessing probability: {new_p_guess}\n")
        # except Exception as e:
        #     x = "\n" + "!"*20 + "\n"
        #     print(x, "An error occurred during optimization:", str(e), x)
        #     break

        if new_p_guess - old_p_guess < precision:
            print(f"Converged guessing probability: {new_p_guess}")
            break
