import stim

circuit = stim.Circuit((f:=open("./basic.stim")).read())

SHOTS=100

sampler = circuit.compile_sampler(seed=0)
samples = sampler.sample(SHOTS)

print(f"oversable flipped {100*sum(o[0] != o[1] for o in samples)/SHOTS}% of the time")
print("flips at: ", [i for i,o in enumerate(samples) if o[0] != o[1]])

sampler = circuit.compile_detector_sampler(seed=0)
detection_events, observable_flips = sampler.sample(SHOTS, separate_observables=True)

print(f"oversable flipped {100*sum(o[0] != o[1] for o in observable_flips)/SHOTS}% of the time")
print("flips at: ", [i for i,o in enumerate(observable_flips) if o[0] != o[1]])