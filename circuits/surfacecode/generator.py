from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import stim

DET_FALSE = "#5CB338"
DET_TRUE = "#ECE852"

PRED_ERROR = "#FFC145"
TRUE_ERROR = "#FB4141"

DATA_UP = "#DA498D"
DATA_DOWN = "#69247C"

STABILIZER_UP = "#00CCDD"
STABILIZER_DOWN = "#6439FF"

def hex_to_rgb(hex_value):
    return [int(hex_value[i:i+2], 16) / 255.0 for i in (1, 3, 5)]

def stbclr(x): return hex_to_rgb(STABILIZER_UP if x else STABILIZER_DOWN)
def datclr(x): return hex_to_rgb(DATA_UP if x else DATA_DOWN)
def detclr(x): return hex_to_rgb(DET_TRUE if x else DET_FALSE)

class Renderer():

    runseed = 0

    def __init__(self, d: int, noise: float):
        assert d % 2 != 0, "code distance must be odd"
        self.dim = d*2-1
        self.num_qubits = self.dim*self.dim
        self.cycles = 0
        self.observable_id = 0
        self.noise = noise
        self.x_errors = np.zeros((1, self.dim,self.dim))
        self.z_errors = np.zeros((1, self.dim,self.dim))

        self.measurement_record: List[int] = []

        self.logical_measurements: List[List[int]] = []
        self.sc_measurements: List[List[int]] = []

        self.prog: List[Instr] = []
        self.rendered = False
        self.cached_prog = ""

        self.ops: List[Tuple[Instr, int, int]] = []

    def render(self, filename: str):
        with open(filename, "w") as f:
            # write qubit coords
            prg = []
            prg = self._render_coords(prg)
            prg = self._render_initial_reset(prg)
            prg = self._render_TICK(prg)

            # Render program
            for instr in self.prog:
                self.ops.append(instr)
                prg = instr.render(self, prg)
                prg = self._render_stabilizer_cycle(prg)

            f.write("\n".join(prg))
            f.close()

            self.rendered = True
            self.cached_prog = filename

    def run(self, shots=1):
        if not self.rendered: self.render("out.stim")
        circuit = stim.Circuit((f:=open(self.cached_prog)).read())

        with open("circuit.svg", 'w') as f:
            f.write(circuit.diagram("detslice-with-ops-svg").__str__())
            f.close()

        sampler = circuit.compile_sampler(seed=Renderer.runseed)
        detector = circuit.compile_detector_sampler(seed=Renderer.runseed)
        Renderer.runseed += 1

        samples = sampler.sample(shots)
        detection_events, observable_flips = detector.sample(shots, separate_observables=True)

        return samples, detection_events, observable_flips 

    def tableau(self):
        simulator = stim.TableauSimulator(seed=Renderer.runseed)

        if not self.rendered: self.render("out.stim")
        circuit = stim.Circuit((f:=open(self.cached_prog)).read())

        simulator.do(circuit)

        print("Tableau:")
        print(simulator.current_inverse_tableau())

        return simulator
    
    # Logical Operations
    def lop(self, op):
        self.prog.append(op)
    
    def id(self): self.prog.append(Identity())
    def x(self): self.prog.append(X())
    def z(self): self.prog.append(Z())
    def mz(self): self.prog.append(MZ())
    def mx(self): self.prog.append(MX())

    # Rendering Graphics

    def render_full_circuit(self, samples: np.ndarray, detections: np.ndarray, observables: np.ndarray, filename:Optional[str]=None):
        print("samples.shape=",samples.shape)
        print("detections.shape=",detections.shape)
        print("observables.shape=",observables.shape)

        T = self.cycles
        sample_img = np.ones((T,self.dim,self.dim,3)) * 255
        sc_measurements = self.sc_measurements[:]
        logical_measurements = self.logical_measurements[:]
        logical_outcomes = []
        for t in range(T):
            if len(logical_measurements) > 0 and logical_measurements[0][0] == t:
                logical_outcomes.append((t, sum(samples[m] for m in logical_measurements[0][1])%2))
                for m in logical_measurements.pop(0)[1]:
                    x,y = self._get_coords(self.measurement_record[-m-1])
                    sample_img[t,x,y] = datclr(samples[m])
            print(sc_measurements[0])
            for _ in ["mz", "mx"]:
                for m in sc_measurements.pop(0):
                    x,y = self._get_coords(self.measurement_record[-m-1])
                    sample_img[t,x,y] = stbclr(samples[m])
        
        det_img = np.ones((T,self.dim,self.dim,3)) * 255
        i = 0
        for t in range(T):
            for mz in [True, False]:
                for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "MZ" and mz:
                            det_img[t,x,y] = detclr(detections[i])
                            i += 1
                        if self._get_qtype(x,y) == "MX" and not mz:
                            det_img[t,x,y] = detclr(detections[i])
                            i += 1

        fig, axes = plt.subplots(2, T, figsize=(T*3,2*3), constrained_layout=True)

        for t in range(T):
            ax = axes[0][t]
            for x in range(self.dim):
                for y in range(self.dim):
                    if self._get_qtype(x,y) == "MZ": ax.annotate("Z", (x,y), textcoords="offset points", xytext=(0, 0), ha='center')
                    if self._get_qtype(x,y) == "MX": ax.annotate("X", (x,y), textcoords="offset points", xytext=(0, 0), ha='center')
            ax.imshow(sample_img[t], aspect='equal', interpolation='nearest')
            if isinstance(self.ops[t], MZ) or isinstance(self.ops[t], MX):
                lo = logical_outcomes.pop(0)
                assert lo[0] == t, "logical outcome does not match timestep"
                ax.set_title(f"Samples at tick #{t}\nMeasurement = {lo[1]}")
            else: ax.set_title(f"Samples at tick #{t}\nApplying: {self.ops[t].__repr__()}")
            ax.axis('off')

            ax = axes[1][t]
            for x in range(self.dim):
                for y in range(self.dim):
                    if self._get_qtype(x,y) == "MZ": ax.annotate("Z", (x,y), textcoords="offset points", xytext=(0, 0), ha='center')
                    if self._get_qtype(x,y) == "MX": ax.annotate("X", (x,y), textcoords="offset points", xytext=(0, 0), ha='center')
            ax.imshow(det_img[t], aspect='equal', interpolation='nearest')
            ax.set_title(f"Detections at tick #{t}")
            ax.axis('off')
        if type(filename) is str: plt.savefig(filename)
        else: plt.show()
    
    def render_detections(self, detections: np.ndarray):
        S, THW = detections.shape
        if S > 5: S = 1
        T = self.cycles

        img = np.zeros((S,T,self.dim,self.dim,3))

        for s in range(S):
            i = 0
            for t in range(T):
                for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "DATA":
                            img[s,t,x,y] = [255, 255, 255]
                        if self._get_qtype(x,y) == "MZ":
                            img[s,t,x,y] = [100,0,0] if detections[s,i] else [0,100,0]
                            i += 1
                        if self._get_qtype(x,y) == "MX":
                            img[s,t,x,y] = [0,100,0]
        
        fig, axes = plt.subplots(S, T, figsize=(S*3, T*3), constrained_layout=True)
        if S == 1:
            axes = [axes]

        for s in range(S):
            for t in range(T):
                ax = axes[s][t]
                for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "MZ": ax.annotate("Z", (x,y), textcoords="offset points", xytext=(0, 0), ha='center')
                        if self._get_qtype(x,y) == "MX": ax.annotate("X", (x,y), textcoords="offset points", xytext=(0, 0), ha='center')
                ax.imshow(img[s,t], aspect='equal')
                ax.set_title(f"Sample #{s}, T={t}")
                ax.axis('off')
        plt.show()

    def render_samples(self, detections: np.ndarray):

        S, THW = detections.shape
        T = self.cycles
        cmap = ListedColormap(["white", "red", "green"])
        
        measurements = np.zeros((S,T,self.dim,self.dim))
        for s in range(S):
            i = 0
            for t in range(T):
                for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "MZ":
                            measurements[s,t,x,y] = 2 if detections[s,i] else 1
                            i += 1
                for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "MX":
                            measurements[s,t,x,y] = 2 if detections[s,i] else 1
                            i += 1

        fig, axes = plt.subplots(S, T, figsize=(S * 3, T*3), constrained_layout=True)

        # Ensure axes is always iterable
        if S == 1:
            axes = [axes]

        for s in range(S):
            for t in range(T):
                ax = axes[s][t]
                ax.imshow(measurements[s,t], cmap=cmap, aspect='equal')
                ax.set_title(f"Sample #{s}, T={t}")
                ax.axis('off')
        plt.show()

    def render_flips(self, detections: np.ndarray, observables: np.ndarray):
        S, THW = detections.shape
        T = self.cycles
        # cmap = ListedColormap(["white", "red", "green", "yellow", "orange", "purple"])
        
        measurements = np.zeros((S,T,self.dim,self.dim))
        for s in range(S):
            i = 0
            for t in range(T):
                for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "MZ":
                            measurements[s,t,x,y] = 2 if detections[s,i] else 1
                            i += 1
                for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "MX":
                            measurements[s,t,x,y] = 2 if detections[s,i] else 1
                            i += 1
        
        flips = np.ones((S,T-1,self.dim,self.dim,3)) * 255
        for s in range(S):
            for t in range(T-1):
                 for x in range(self.dim):
                    for y in range(self.dim):
                        if self._get_qtype(x,y) == "MX" or self._get_qtype(x,y) == "MZ":
                            flips[s,t,x,y] = [0,255,0] if measurements[s,t,x,y] == measurements[s,t+1,x,y] else [255,0,0]
                        if self._get_qtype(x,y) == "DATA" and self.x_errors[t+1,x,y] == 1: flips[s,t,x,y] = [0,100,100]
                        if self._get_qtype(x,y) == "DATA" and self.z_errors[t+1,x,y] == 1: flips[s,t,x,y] = [100,100,0]
                        if self._get_qtype(x,y) == "DATA" and self.z_errors[t+1,x,y] == 1 and self.x_errors[t,x,y] == 1: flips[s,t,x,y] = [100,100,100]

        fig, axes = plt.subplots(S, T, figsize=(S * 3, T*3), constrained_layout=True)

        # Ensure axes is always iterable
        if S == 1:
            axes = [axes]

        for s in range(S):
            for t in range(T-1):
                ax = axes[s][t]
                ax.imshow(flips[s,t], aspect='equal')
                ax.set_title(f"Sample #{s}, T={t}")
                ax.axis('off')
        plt.show()

    def _render_z_readout(self, prg:List[str]) -> str:
        observables = []
        for x in range(self.dim):
            for y in range(self.dim):
                if self._get_qtype(x,y) == "DATA":
                    prg = self._render_M(x,y,prg)
                if self._get_qtype(x,y) == "MX":
                    prg = self._render_M(x,y,prg)
                observables.append((x,y))
        prg.append(f"OBSERVABLE_INCLUDE({self.observable_id}) " + " ".join([f"rec[-{self.measurement_record.index(self._get_qid(x,y))+1}]" for x,y in observables]))
        self.observable_id += 1
        return prg
    def _render_x_readout(self, prg:List[str]) -> str:
        observables = []
        for x in range(self.dim):
            for y in range(self.dim):
                if self._get_qtype(x,y) == "DATA":
                    prg = self._render_MX(x,y,prg)
                    observables.append((x,y))
        prg.append(f"OBSERVABLE_INCLUDE({self.observable_id}) " + " ".join([f"rec[-{self.measurement_record.index(self._get_qid(x,y))+1}]" for x,y in observables]))
        self.observable_id += 1
        return prg
    def _render_qubit_coord(self, x:int, y:int) -> str:
         return f"QUBIT_COORDS({x}, {y}) {self._get_qid(x,y)}"
    def _render_TICK(self, prg:List[str]) -> List[str]:
        # prg.append(f"X_ERROR({self.noise}) " + " ".join([f"{self._get_qid(x,y)}" for x in range(self.dim) for y in range(self.dim)]))
        # prg.append(f"Z_ERROR({self.noise}) " + " ".join([f"{self._get_qid(x,y)}" for x in range(self.dim) for y in range(self.dim)]))
        # prg = self._rand_error(prg)
        prg.append("TICK")
        return prg
    def _render_M(self, x:int, y:int, prg: List[str]) -> List[str]:
        prg.append(f"M {(qid:=self._get_qid(x,y))}")
        self.measurement_record.insert(0, qid)
        return prg
    def _render_MX(self, x:int, y:int, prg: List[str]) -> List[str]:
        prg.append(f"MX {(qid:=self._get_qid(x,y))}")
        self.measurement_record.insert(0, qid)
        return prg
    def _render_MZ(self, x:int, y:int, prg: List[str]) -> List[str]:
        prg.append(f"MZ {(qid:=self._get_qid(x,y))}")
        self.measurement_record.insert(0, qid)
        return prg
    def _render_MRZ(self, x:int, y:int, prg: List[str]) -> List[str]: 
        prg.append(f"MRZ {(qid:=self._get_qid(x,y))}")
        self.measurement_record.insert(0, qid)
        return prg
    def _render_MRX(self, x:int, y:int, prg:List[str]) -> List[str]:
        prg.append(f"MRX {(qid:=self._get_qid(x,y))}")
        self.measurement_record.insert(0, qid)
        return prg
    def _render_COORD_SHIFT(self, prg: List[str]) -> List[str]:
        prg.append("SHIFT_COORDS(0, 0, 1)")
        return prg
    def _render_DETECTOR(self, x: int, y:int, prg: List[str]) -> List[str]:
        det = f"DETECTOR({x}, {y}, 0) "
        det += f"rec[-{(last_idx:=self.measurement_record.index(self._get_qid(x,y))+1)}] " 
        if self._get_qid(x,y) in self.measurement_record[last_idx:]: det += f"rec[-{self.measurement_record.index(self._get_qid(x,y), last_idx)+1}]"
        prg.append(det)
        return prg
    def _render_R(self, x:int, y:int, prg: List[str]) -> str:
        prg.append(f"R {self._get_qid(x,y)}")
        return prg
    def _render_RX(self, x:int, y:int, prg: List[str]) -> str:
        prg.append(f"RX {self._get_qid(x,y)}")
        return prg
    def _render_H(self, x:int, y:int, prg:List[str]) -> str:
        prg.append(f"H {self._get_qid(x,y)}")
        return prg
    def _render_CX(self, ctrlx:int, ctrly:int, tarx:int, tary:int, prg:List[str]):
        prg.append(f"CX {self._get_qid(ctrlx, ctrly)} {self._get_qid(tarx,tary)}")
        return prg 
    
    def _render_coords(self, prg:List[str]=[]) -> List[str]:
        prg.append("# Qubit Coordinates")
        for x in range(self.dim):
                for y in range(self.dim):
                    qtype = {"DATA": "data qubit", "MZ": "measure z qubit", "MX": "measure x qubit"}[self._get_qtype(x,y)]
                    prg.append(f"{self._render_qubit_coord(x,y)} # {qtype}")
        return prg
    
    def _render_stabilizer_cycle_depr(self, prg:List[str]=[]) -> List[str]:
        prg.append("# " + "-"*78)
        prg.append(f"# Starting Stabilizer Cycle {self.cycles}")
        prg.append("# " + "-"*78)
        # prg = self._rand_error(prg)
        prg = self._render_z_cycle(prg)
        prg = self._render_COORD_SHIFT(prg)
        prg = self._render_x_cycle(prg)
        prg = self._render_COORD_SHIFT(prg)
        self.cycles += 1
        self.x_errors = np.concatenate((self.x_errors, np.zeros((1, self.dim, self.dim))), axis=0)
        self.z_errors = np.concatenate((self.z_errors, np.zeros((1, self.dim, self.dim))), axis=0)
        return prg
    
    def _render_stabilizer_cycle(self, prg:List[str]=[]) -> List[str]:
        prg.append("# " + "-"*78)
        prg.append(f"# Starting Stabilizer Cycle {self.cycles}")
        prg.append("# " + "-"*78)
        self.cycles += 1
        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                #if qt == "MZ": identity
                if qt == "MX": prg = self._render_R(x,y,prg)
        prg = self._render_TICK(prg)

        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                if qt == "MZ": prg = self._render_R(x,y,prg)
                if qt == "MX": prg = self._render_H(x,y,prg)
        prg = self._render_TICK(prg)

        # Check top qubit
        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                if qt == "MZ" and y != 0:
                    prg = self._render_CX(x,y-1,x,y,prg)
                if qt == "MX" and y != 0:
                    prg = self._render_CX(x,y,x,y-1,prg)
        prg = self._render_TICK(prg)

        # Check left qubit
        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                if qt == "MZ" and x != 0:
                       prg = self._render_CX(x-1,y,x,y,prg)
                if qt == "MX" and x != 0:
                       prg = self._render_CX(x,y,x-1,y,prg)
        prg = self._render_TICK(prg)

        # Check right qubit
        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                if qt == "MZ" and x+1 < self.dim:
                       prg = self._render_CX(x+1,y,x,y,prg)
                if qt == "MX" and x+1 < self.dim:
                       prg = self._render_CX(x,y,x+1,y,prg)
        prg = self._render_TICK(prg)

        # Check bottom qubit
        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                if qt == "MZ" and y+1 < self.dim:
                       prg = self._render_CX(x,y+1,x,y,prg)
                if qt == "MX" and y+1 < self.dim:
                       prg = self._render_CX(x,y,x,y+1,prg)
        prg = self._render_TICK(prg)

        zsyndrome = []
        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                if qt == "MZ":
                    zsyndrome.append(len(self.measurement_record))
                    prg=self._render_MRZ(x,y,prg)
                    prg = self._render_DETECTOR(x,y,prg)
                if qt == "MX": prg = self._render_H(x,y,prg)
        self.sc_measurements.append(zsyndrome)
        prg = self._render_TICK(prg)

        xsyndrome = []
        for x in range(self.dim):
            for y in range(self.dim):
                qt = self._get_qtype(x,y)
                # if qt == "MZ": identity
                if qt == "MX":
                    xsyndrome.append(len(self.measurement_record))
                    prg=self._render_MRX(x,y,prg)
                    prg = self._render_DETECTOR(x,y,prg)        
        self.sc_measurements.append(xsyndrome)
        prg = self._render_TICK(prg)
        return prg

    def _get_qid(self, x: int, y: int): return x+y*self.dim
    def _get_coords(self, id: int): return (id%self.dim, id//self.dim)
    def _get_qtype(self, x: int, y: int): return {0: "DATA", 1: {0: "MZ", 1: "MX"}[y%2]}[(x+self.dim*y)%2]

    def _render_initial_reset(self, prg:List[str]) -> List[str]:
        # Inital Reset
        prg.append("# Initial Reset")
        for x in range(self.dim):
            for y in range(self.dim):
                prg = self._render_R(x,y,prg)
        return prg
    
    def _rand_error(self,prg:List[str]) -> List[str]:
        for x in range(self.dim):
            for y in range(self.dim):
                if self._get_qtype(x,y) == "DATA":
                    if np.random.rand() < self.noise:
                        prg.append(f"X_ERROR(1.0) {self._get_qid(x,y)}")
                        self.x_errors[self.cycles,x,y] = 1
                    if np.random.rand() < self.noise:
                        prg.append(f"Z_ERROR(1.0) {self._get_qid(x,y)}")
                        self.z_errors[self.cycles,x,y] = 1
        return prg

    def _render_z_cycle(self, prg:List[str]) -> List[str]:
        prg.append("# Measure Z Cycle")
        prg.append("# Measure the top qubit")
        # measure the above qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MZ" and y != 0:
                       prg = self._render_CX(x,y-1,x,y,prg)
        prg = self._render_TICK(prg)
        prg.append("# Measure the right qubit")
        # measure the right qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MZ" and x != 0:
                       prg = self._render_CX(x-1,y,x,y,prg)
        prg = self._render_TICK(prg)
        prg.append("# Measure the left qubit")
        # measure the left qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MZ" and x+1 < self.dim:
                       prg = self._render_CX(x+1,y,x,y,prg)
        prg = self._render_TICK(prg)
        prg.append("# Measure the bottom qubit")
        # measure the bottom qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MZ" and y+1 < self.dim:
                       prg = self._render_CX(x,y+1,x,y,prg)
        prg = self._render_TICK(prg)
        syndrome = []
        prg.append("# Perform the measurement")
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MZ":
                       syndrome.append(len(self.measurement_record))
                       prg=self._render_MRZ(x,y,prg)
        self.sc_measurements.append(syndrome)
        for x in range(self.dim):
             for y in range(self.dim):
                if self._get_qtype(x,y) == "MZ":
                    prg = self._render_DETECTOR(x,y,prg)
        prg = self._render_TICK(prg)
        return prg
    
    def _render_x_cycle(self, prg:List[str]) -> List[str]:
        prg.append("# Measure X Cycle")
        prg.append("# Initialize to ground state")
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX":
                       prg = self._render_R(x,y, prg)
        prg = self._render_TICK(prg)
        prg.append("# Put the measure bit into the measure-X basis")
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX":
                       prg = self._render_H(x,y,prg)
        prg = self._render_TICK(prg)
        prg.append("# Measure the top qubit")
        # measure the above qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX" and y != 0:
                       prg = self._render_CX(x,y,x,y-1,prg)
        prg = self._render_TICK(prg)
        prg.append("# Measure the right qubit")
        # measure the right qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX" and x != 0:
                       prg = self._render_CX(x,y,x-1,y,prg)
        prg = self._render_TICK(prg)
        prg.append("# Measure the bottom qubit")
        # measure the bottom qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX" and x+1 < self.dim:
                       prg = self._render_CX(x,y,x+1,y,prg)
        prg = self._render_TICK(prg)
        prg.append("# Measure the left qubit")
        # measure the left qubit
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX" and y+1 < self.dim:
                       prg = self._render_CX(x,y,x,y+1,prg)
        prg = self._render_TICK(prg)
        
        prg.append("# Put the measure bit out of the measure-X basis")
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX":
                       prg = self._render_H(x,y, prg)
        prg = self._render_TICK(prg)
        syndrome = []
        prg.append("# Perform the measurement")
        for x in range(self.dim):
             for y in range(self.dim):
                  if self._get_qtype(x,y) == "MX":
                       syndrome.append(len(self.measurement_record))
                       prg=self._render_MRX(x,y,prg)
        self.sc_measurements.append(syndrome)
        for x in range(self.dim):
             for y in range(self.dim):
                 if self._get_qtype(x,y) == "MX":
                    prg = self._render_DETECTOR(x,y,prg)
        prg = self._render_TICK(prg)
        return prg
    
class Instr():

    type = "inline"

    def __init__(self, code:List[str]=[]):
        self.code = code

    def render(self, renderer:Renderer, prg:List[str]):
        prg += self.code
        return prg
    
    def __repr__(self): return self.type

class Identity(Instr):
    type = "tick"
    def __init__(self):
        self.code = ["TICK"]

    def render(self, renderer:Renderer, prg:List[str]):
        prg.append("TICK")
        return prg

class Z(Instr):
    type = "Zl"
    def __init__(self):
        self.code = []

    def render(self, renderer:Renderer, prg:List[str]):
        lx = renderer.dim//2
        for y in range(0, renderer.dim, 2):
            prg.append(f"Z {renderer._get_qid(lx,y)}")
        prg.append("TICK")
        return prg

class X(Instr):
    type = "Xl"
    def __init__(self):
        self.code = []

    def render(self, renderer:Renderer, prg:List[str]):
        ly = renderer.dim//2
        for x in range(0, renderer.dim, 2):
            prg.append(f"X {renderer._get_qid(x,ly)}")
        prg.append("TICK")
        return prg
    
class MX(Instr):
    type = "MXl"
    def __init__(self):
        self.code = []

    def render(self, renderer:Renderer, prg:List[str]):
        observables = []
        parity_of = []
        for x in range(renderer.dim):
            for y in range(renderer.dim):
                if renderer._get_qtype(x,y) == "DATA":
                    parity_of.append(len(renderer.measurement_record))
                    prg = renderer._render_MX(x,y,prg)
                    observables.append((x,y))
        prg.append(f"OBSERVABLE_INCLUDE({renderer.observable_id}) " + " ".join([f"rec[-{renderer.measurement_record.index(renderer._get_qid(x,y))+1}]" for x,y in observables]))
        prg.append("TICK")
        renderer.observable_id += 1
        renderer.logical_measurements.append((renderer.cycles, parity_of))
        return prg
    
class MZ(Instr):
    type = "MZl"
    def __init__(self):
        self.code = []

    def render(self, renderer:Renderer, prg:List[str]):
        observables = []
        parity_of = []
        for x in range(renderer.dim):
            for y in range(renderer.dim):
                if renderer._get_qtype(x,y) == "DATA":
                    parity_of.append(len(renderer.measurement_record))
                    prg = renderer._render_MZ(x,y,prg)
                    observables.append((x,y))
        prg.append(f"OBSERVABLE_INCLUDE({renderer.observable_id}) " + " ".join([f"rec[-{renderer.measurement_record.index(renderer._get_qid(x,y))+1}]" for x,y in observables]))
        prg.append("TICK")
        renderer.observable_id += 1
        renderer.logical_measurements.append((renderer.cycles, parity_of))
        return prg
