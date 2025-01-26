import stim
import numpy as np
import pymatching
import matplotlib.pyplot as plt
import sinter
from typing import List

# print the version
print(stim.__version__)

# create a circuit
circuit = stim.Circuit()

# initialize the bell states
circuit.append("H", [0])
circuit.append("CNOT", [0, 1])

# measure both quibits
circuit.append("M", [0, 1])

# print the circuit diagram
print(circuit.diagram())

# compile a sampler
sampler = circuit.compile_sampler()

# print sample results
print(sampler.sample(shots=10))

# Now we can add detector annotations
circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
print(repr(circuit))

# now we can sample from the detector
sampler = circuit.compile_detector_sampler()
print("detector w/0 noise")
print(sampler.sample(shots=5))

# Run a detector circuit with errors
with open("./intro0.stim") as f:
    circuit = stim.Circuit(f.read())
    f.close()
    sampler = circuit.compile_detector_sampler()
    print("detector w/ noise")
    print(sampler.sample(shots=10))

    # how often the detector fires on average
    print(f"detector fraction = {np.sum(sampler.sample(shots=10**6)) / 10**6}")

# Running a repetition code
circuit = stim.Circuit.generated(
    "repetition_code:memory",
    rounds=25,
    distance=9,
    before_round_data_depolarization=0.04,
    before_measure_flip_probability=0.01)

print(repr(circuit))

# sample measurements
sampler = circuit.compile_sampler()
one_sample = sampler.sample(shots=1)[0]
for k in range(0, len(one_sample), 8):
    timeslice = one_sample[k:k+8]
    print("".join("1" if e else "_" for e in timeslice))

# sample detectors
detector_sampler = circuit.compile_detector_sampler()
one_sample = detector_sampler.sample(shots=1)[0]
for k in range(0, len(one_sample), 8):
    timeslice = one_sample[k:k+8]
    print("".join("!" if e else "_" for e in timeslice))

# using error models
dem = circuit.detector_error_model()
print(repr(dem))

# count logical errors
def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    print("detections: ", detection_events.shape)
    print(detection_events)
    print("observables: ", observable_flips.shape)
    print(observable_flips)
    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)
    print("predictions: ", predictions.shape)
    print(predictions)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors

# use it on the repetition code
print(f"using before_round_data_depolarization=0.03")
circuit = stim.Circuit.generated("repetition_code:memory", rounds=100, distance=9, before_round_data_depolarization=0.03)
num_shots = 1
num_logical_errors = count_logical_errors(circuit, num_shots)
print("there were", num_logical_errors, "wrong predictions (logical errors) out of", num_shots, "shots")

# increasing the noise increases the number of logical errors
print(f"using before_round_data_depolarization=0.13")
circuit = stim.Circuit.generated(
    "repetition_code:memory",
    rounds=100,
    distance=9,
    before_round_data_depolarization=0.13,
    before_measure_flip_probability=0.01)
num_shots = 1
num_logical_errors = count_logical_errors(circuit, num_shots)
print("there were", num_logical_errors, "wrong predictions (logical errors) out of", num_shots, "shots")

# estimate the threshold
# num_shots = 10_000
# for d in [3, 5, 7]:
#     xs = []
#     ys = []
#     for noise in [0.01, 0.05, 0.075, 0.1, 0.5, 0.75]:
#         circuit = stim.Circuit.generated(
#             "repetition_code:memory",
#             rounds=d * 3,
#             distance=d,
#             before_round_data_depolarization=noise)
#         num_errors_sampled = count_logical_errors(circuit, num_shots)
#         xs.append(noise)
#         ys.append(num_errors_sampled / num_shots)
#     plt.plot(xs, ys, label="d=" + str(d))
# plt.loglog()
# plt.xlabel("physical error rate")
# plt.ylabel("logical error rate per shot")
# plt.legend()
# plt.show()

# for a surface code
num_shots = 1
for d in [3, 5, 7]:
    xs = []
    ys = []
    for noise in [0.008, 0.009, 0.01, 0.011, 0.012]:
        surface_code_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d*3,
            distance=d,
            after_clifford_depolarization=noise,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
            before_round_data_depolarization=noise)
        num_errors_sampled = count_logical_errors(circuit, num_shots)
        xs.append(noise)
        ys.append(num_errors_sampled / num_shots)
    plt.plot(xs, ys, label="d=" + str(d))
plt.loglog()
plt.xlabel("physical error rate")
plt.ylabel("logical error rate per shot")
plt.legend()
plt.show()