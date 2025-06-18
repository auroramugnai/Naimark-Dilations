import numpy as np
import picos as pic
import qutip as qt
from picos.modeling.problem import Problem
from tqdm import tqdm

num_attempt = 3

# Define identity matrices
I2 = qt.qeye([2])
I4 = qt.qeye([2,2])

for _ in tqdm(range(num_attempt)):

    # Generate a random density matrix of 2 qubits.
    # This a state on the system S+M.
    rhoSM = qt.rand_dm([2, 2])

    # Deduce the density matrix on M.
    rhoM = qt.ptrace(rhoSM, 1)

    # Generate a random pure state (ket) on 2 qubits.
    PiSM0_ket = qt.rand_ket([2, 2])

    # Create the projector out of the pure state.
    # This is a Naimark projector on the system S+M.
    PiSM0 = qt.ket2dm(PiSM0_ket)
    
    # Find the other Naimark projector using completeness.
    PiSM1 = I4 - PiSM0
    
    # Find the measurements on S associated to the Naimark projectors on S+M.
    MS0 = qt.ptrace(PiSM0 * (qt.tensor(I2, rhoM)), 0)
    MS1 = I2 - MS0
    
    # Condiser a convex decomposition M(a, lambda) of {MS0, MS1} = {MSa},
    # where lambda labels the decomposition element and a is the outcome.
    #
    # 1) Impose that it is a convex decomposition of our POVM:
    #    i)  p(lambda=0) * M(a, 0) + p(lambda=1) * M(a, 1) = MSa.
    #    ii) Or M~(a, 0) + M~(a, 1) = MSa.
    #        We will impose M00 + M01 = MS0.
    #                       M10 + M11 = MS1.
    #
    # 2) Impose that for every fixed lambda the decomposition is a POVM:
    #    i)  M(0, lambda) + M(1, lambda) = I2.                       
    #    ii) Or M~(0, lambda) + M~(1, lambda) =  p(lambda) * I2.
    #        We will impose M00 + M10 = p0 *I2.
    #        (The other one with lambda=1 is not necessary)
    #
    # 3) Impose that every element of the decomposition is >> 0.
    #
    # 4) Test.
    #    If {MS0, MS1} is extremal, M00 should not be equal to MS0:
    #           MS0 - M00 >> epsilon*I2
    #
    # -> To sum up, 1 eq. constraint, 5 ineq. constraints and 2 variables:
    #
    #     M00 + M01 = MS0.   
    #     M10 + M11 = MS1.        --> 
    #     M00 + M10 = p0 *I2.          M00 + M10 = p0 *I2.             [2)]
    #     M00, M01, M11, M10 >> 0      M00, MS0-M00, MS1-M10, M10 >> 0 [1) + 3)]
    #     MS0 - M00 >> epsilon*I2      MS0 - M00 >> epsilon*I2         [4)]
    #
    #     Test fails   -> no decomposition found for {MS0, MS1} -> there's hope it's extremal.
    #     Test succeds -> decomposition found for {MS0, MS1} -> it's NOT extremal.

    problem = Problem()
    
    # Turn I2, MS0, MS1  to a pic constant.
    I2_pic = pic.Constant("I2", np.eye(2))
    MS0_pic = pic.Constant("MS0", MS0.full())
    MS1_pic = pic.Constant("MS1", MS1.full())

    # Setup the variables of the problem.
    M00 = pic.HermitianVariable('M00', 2)
    M10 = pic.HermitianVariable('M10', 2)
    p0 = pic.RealVariable('p0', lower=0, upper=1)
    
    # Constraint 1) and 3).
    problem.add_constraint(M00 >> 0)
    problem.add_constraint(M10 >> 0)
    problem.add_constraint(MS0_pic - M00 >> 0) # M01>>0 + Constraint 1)
    problem.add_constraint(MS1_pic - M10 >> 0) # M11>>0 + Constraint 1)   

    # Constraint 2).
    problem.add_constraint(M00 + M10 == p0*I2_pic)

    # Constraint 4). Test. 
    epsilon = pic.Constant('eps', 1e-15)  # small constant
    problem.add_constraint(MS0_pic - M00 - epsilon*I2_pic >> 0)
    
    # Trivial objective (as it is a feasibility problem).
    obj = pic.Constant('1', 1)
    problem.set_objective("max", obj)
        
    try:
        problem.solve()
    except:
        print("No decomposition found!")
        print("rhoSM = \n", rhoSM)
        print("PiSM0 = \n", PiSM0)






