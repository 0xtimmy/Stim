import stim

circuit = stim.Circuit((f:=open("./basicz.stim")).read())

SHOTS=100

sampler = circuit.compile_sampler(seed=0)
samples = sampler.sample(SHOTS)

print("samples=",samples)

print(f"sample flipped {100*sum(o[0] != o[1] for o in samples)/SHOTS}% of the time")
print("flips at: ", [i for i,o in enumerate(samples) if o[0] != o[1]])