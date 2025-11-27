import numpy as np
import qutip as qt
from qutip import Qobj

def buildEntangledBasis(theta):
    eta=[1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)]
    phi=[np.pi/4,7*np.pi/4,3*np.pi/4,5*np.pi/4]
    mPos=[np.sqrt((1+eta[b])/2)*np.exp(-1j*phi[b]/2)*qt.basis(2,0)+np.sqrt((1-eta[b])/2)*np.exp(1j*phi[b]/2)*qt.basis(2,1) 
          for b in range(4)]
    mNeg=[np.sqrt((1-eta[b])/2)*np.exp(-1j*phi[b]/2)*qt.basis(2,0)-np.sqrt((1+eta[b])/2)*np.exp(1j*phi[b]/2)*qt.basis(2,1) 
          for b in range(4)]
    return [(np.sqrt(3)+np.exp(1j*theta))/(2*np.sqrt(2))*qt.tensor(mPos[b],mNeg[b])+
               (np.sqrt(3)-np.exp(1j*theta))/(2*np.sqrt(2))*qt.tensor(mNeg[b],mPos[b])
               for b in range(4)]

basis = buildEntangledBasis(theta=0)

#### Print the basis. ####
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


#### Look for a probability distribution that gives an entangled rho_SA. ####
#### To do this, randomly generate probability distributions until you   ####
#### find one that gives a Negative Partial Transpose (NPT) state.       ####
keep_going = True
while keep_going:
    print("----------------------------------------")
    #### Generate a probability distribution ####
    probabilities = np.random.rand(4)
    probabilities /= probabilities.sum()
    assert np.isclose(probabilities.sum(), 1, atol=1e-8, rtol=1e-8), "Probabilities do not sum to 1."

    #### Built rho_SA from 'basis' and 'probabilities'. ####
    rho_SA = sum(p * (psi * psi.dag()) for psi, p in zip(basis, probabilities))
    
    #### Check PPT criterion. ####
    rho_SA_PT = qt.partial_transpose(rho_SA, [0,1])
    
    eigvals = rho_SA_PT.eigenenergies()
    if np.any(eigvals < -1e-12):
        print("The state is entangled (NPT).")
        keep_going = False
    else:
        print("The state is PPT (may be separable).")

print("\nEigenvalues of partial transpose:\n", eigvals)
print("\nrho_SA:\n", rho_SA)


#### Check that the state is entangled by computing the Von Neumann ####
#### entropies of the subsystems. If S(rho_S) = S(rho_A) > 0, then  ####
#### the state is entangled.                                        ####

# Partial traces
print(type(rho_SA))
rho_S = rho_SA.ptrace(0)
rho_A = rho_SA.ptrace(1)

# Von Neumann entropies
S_S = qt.entropy_vn(rho_S)
S_A = qt.entropy_vn(rho_A)
print(f"Entropy of S: {S_S:.3f}")
print(f"Entropy of A: {S_A:.3f}")


"""
Ψ_i = (√3 + e^{iθ})/(2√2) * |mPos_i⟩⊗|mNeg_i⟩ + (√3 - e^{iθ})/(2√2) * |mNeg_i⟩⊗|mPos_i⟩
Because Ψ_i is a sum of |something⟩⊗|other⟩ and |other⟩⊗|something⟩,
The reduced density matrix of S will have the same eigenvalues as the reduced density matrix of A.
In simple words: whatever S “sees”, A sees the same.

Think of a Bell state:

psi+ = 1/sqrt2*(∣01⟩+∣10⟩)
Swap the qubits → nothing changes.
Tracing out the second qubit gives the same reduced density matrix as tracing out the first.
This is exactly what your basis states are doing — just more general amplitudes.
"""