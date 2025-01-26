from generator import Renderer
import stim
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 4})

r = Renderer(3, 0.000)

r.id()
r.id()
r.mz()
r.id()
r.x()
r.id()
r.mz()
r.id()
r.x()
r.id()
r.mz()
r.id()
r.x()
r.id()
r.x()
r.id()
r.mz()

samples, detections, observables = r.run()

r.render_full_circuit(samples[0], detections[0], observables[0], filename="scz.png")