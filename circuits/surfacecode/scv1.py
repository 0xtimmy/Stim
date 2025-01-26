import stimcircuits
import pymatching
import stim
import numpy as np

rounds = 12
noise = 0.001

smp_width = 8 # distance 3
det_width = 8

# smp_width = 24 # distance 5
# det_width = 24

# smp_width = 48 # distance 7
# det_width = 48

# z_circuit = stimcircuits.generate_circuit(
#     "surface_code:rotated_memory_z",
#     rounds=rounds,
#     distance=5,
#     before_round_data_depolarization=noise)

z_circuit = stim.Circuit((f:=open("./cut_circuit.stim")).read())

with open("cut_circuit.svg", 'w') as f:
    f.write(z_circuit.diagram("detslice-with-ops-svg").__str__())
    f.close()

# with open("cut_circuit.stim", 'w') as f:
#     f.write(z_circuit.__str__())
#     f.close()

# x_circuit = stimcircuits.generate_circuit("surface_code:rotated_memory_x", rounds=2, distance=3)

# with open("x_circuit.svg", 'w') as f:
#     f.write(x_circuit.diagram("detslice-with-ops-svg").__str__())
#     f.close()

# with open("x_circuit.stim", 'w') as f:
#     f.write(x_circuit.__str__())
#     f.close()

sampler = z_circuit.compile_sampler()
sample = sampler.sample(shots=100)

# print("sample: ", sample.shape)
# for i in range(rounds):
#     print("".join(["!" if v else "_" for v in sample[0][i*smp_width:(i+1)*smp_width]]))
# print("other measurements=", sample[0][rounds*smp_width:])

detector = z_circuit.compile_detector_sampler()
detections, observables = detector.sample(shots=100, separate_observables=True)

# print("detections: ", detections.shape)
# for i in range(rounds):
#     print("".join(["!" if v else "_" for v in detections[0][i*det_width:(i+1)*det_width]]))
# print("observables: ", observables.shape, " = ", observables)

detector_error_model = z_circuit.detector_error_model(decompose_errors=True)
matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
predictions = matcher.decode_batch(detections)
# print("predictions: ", predictions.shape, " = ", predictions)

# print("correct execution? ", np.equal(observables, predictions))

print(sample.shape)

t = np.array([[1,1],[0,0]])

correct = np.equal(observables, predictions)
print(f"Correct: {np.sum(correct)}%")
print(f"True: {np.sum(observables)}%")

# print(f"Samples: {sample[:,-9:].sum(axis=0)}")
# print(f"Samples: {np.equal(sample[:,-9:], sample[:,:9]).sum(axis=0)}")
# print(sample[-1,:9+8])
# print(sample[-1,-9-8:])

# stbl = sample[:,9:-9]
# stbl = stbl.reshape(100, -1, 8)
# print(f"Stabilizers: {stbl.shape}")
# print(stbl.sum(axis=0))