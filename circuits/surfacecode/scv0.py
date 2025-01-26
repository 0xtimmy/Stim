from generator import Renderer
import stim
import pymatching

circuit_file = "circuit.stim"

SHOTS = 100

# render the circuit
r = Renderer(3, 0.008)
r.render(circuit_file)

circuit = stim.Circuit((f:=open(circuit_file)).read())
f.close()

exit()

sampler = circuit.compile_detector_sampler()
detection_events, observable_flips = sampler.sample(SHOTS, separate_observables=True)

print(detection_events.shape)

# detector_error_model = circuit.detector_error_model(decompose_errors=True)
# matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
# predictions = matcher.decode_batch(detection_events)

print("observable_flips: ", observable_flips.shape)
print(f"oversable flipped {100*sum(o[0] != o[1] for o in observable_flips)/SHOTS}% of the time")
print(observable_flips[:10])
# print("predictions: ", predictions.shape)
# print(f"predictions flipped {100*sum(o[0] != o[1] for o in predictions)/SHOTS}% of the time")
# print(predictions)

with open("circuit.svg", 'w') as f:
    f.write(circuit.diagram("detslice-with-ops-svg").__str__())
    f.close()


r.render_detections(detection_events)

sampler = circuit.compile_sampler()
samples = sampler.sample(1)

print(samples[0])
