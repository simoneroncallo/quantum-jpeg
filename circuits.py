import numpy as np
from itertools import product
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT

def circuit_builder(states, n0, n2):
    """ Create a list of n-qubit quantum circuits "qcs", each initialized 
    using the array "states". For each circuit
        1. Perform QFT
        2. Apply IQFT on the first (n0 - n2)//2 qubits
        3. Discard (n0 - n2)//2 qubits from the middle of the register, then 
            perform a measurement of the remaining qubits and store the 
            results in a classical register of n2 bits 
    Return "qcs". """
    
    ntilde = (n0 - n2)//2 # Total number of qubits removed aat each Step
    n1 =  n0 - ntilde # Number of qubits kept, before IQFT is applied
    
    qcs = []
    for idx in range(states.shape[0]):
        q = QuantumRegister(n0)
        c = ClassicalRegister(n2)
        qc = QuantumCircuit(q,c)
        
        qc.initialize(states[idx], q)    
        
        qc.h(q) # Optional, improves the reconstruction
        
        # Apply QFT on the full circuit
        qc.compose(QFT(num_qubits=n0, approximation_degree=0, do_swaps=True, \
                       inverse=False, insert_barriers=True, name='QFT'), \
                       inplace = True)
        
        qc.barrier()
        
        # Apply IQFT on the first n1 qubits (Rule 1)
        qc.compose(QFT(num_qubits=n1, approximation_degree=0, do_swaps=True, \
                       inverse=True, insert_barriers=True, name='IQFT'), \
                       qubits = q[0:n1], inplace = True)
        
        creg_idx = 0
        for idx in range(n1):  
            if n0//2 - ntilde <= idx <= n0//2 - 1: 
                continue # Qubit discarded from the measurement (Rule 2)
            qc.h(q[idx]) # Optional, improves the reconstruction
            qc.measure(q[idx],c[creg_idx])
            creg_idx += 1
        qcs.append(qc)
        
    return qcs

def reconstruction(qcs, n2, shots, norm):
    """ Simulate "qcs" with a given number of "shots". 
    Return an array "out_freq" with shape (S, 2**n2), describing the output 
    frequencies of each circuit and eventually rescaled using the corresponding
    component of "norm". """
    
    out_freq = np.zeros((len(qcs), 2**n2)) # Shape (S, 2**n2)
    
    for idx in range(len(qcs)):
        simulator = AerSimulator()
        qcs[idx] = transpile(qcs[idx], simulator)
        result = simulator.run(qcs[idx], shots = shots).result()
        
        counts = result.get_counts(qcs[idx]) # Counts at the output of qcs[idx]
        tot = sum(counts.values(), 0.0)
        prob = {key: val/tot for key, val in counts.items()} # Frequencies
        
        out = np.zeros(2**n2) # Shape (1, 2**n2)
        
        # Generate all the possible n2 qubits configurations
        cfgs = list(product(('0','1'), repeat = n2))
        cfgs = [''.join(cfg) for cfg in cfgs] 
        for i in range(2**n2):
            out[i] = prob.get(cfgs[i], 0)
            
        out_freq[idx,:] = out[:]*norm[idx]
    
    return out_freq