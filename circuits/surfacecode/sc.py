import stim
import sinter

def main():
    circuit = stim.Circuit((f:=open("./memory.stim")).read())
    f.close()

    # d = 3
    # noise = 0.1
    # circuit = stim.Circuit.generated(
    #         "surface_code:rotated_memory_z",
    #         rounds=d * 3,
    #         distance=d,
    #         after_clifford_depolarization=noise,
    #         after_reset_flip_probability=noise,
    #         before_measure_flip_probability=noise,
    #         before_round_data_depolarization=noise,
    #     )
    # print(circuit)
    with open("circuit.svg", 'w') as f:
        f.write(circuit.diagram("detslice-with-ops-svg").__str__())
        f.close()

if __name__ == "__main__":
    main()