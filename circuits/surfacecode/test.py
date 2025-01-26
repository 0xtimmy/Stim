import stim
import numpy as np

circuit = stim.Circuit((f:=open("./test_circuit.stim")).read())

sampler = circuit.compile_sampler()
samples = sampler.sample(shots=100)

samples = samples.reshape(100, 3, 5)

print(f"True: {np.sum(samples, axis=0)}")