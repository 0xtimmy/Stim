#ifndef SIM_FRAME_H
#define SIM_FRAME_H

#include <random>

#include "circuit.h"
#include "pauli_string.h"

struct PauliFrameSimCollapse {
    SparsePauliString destabilizer;
};

struct PauliFrameSimMeasurement {
    size_t target_qubit;
    bool invert;
};

struct PauliFrameSimCycle {
    std::vector<Operation> step1_unitary;
    std::vector<PauliFrameSimCollapse> step2_collapse;
    std::vector<PauliFrameSimMeasurement> step3_measure;
    std::vector<size_t> step4_reset;
};

struct SimFrame {
    size_t num_qubits = 0;
    size_t num_measurements = 0;
    std::vector<PauliFrameSimCycle> cycles;

    static SimFrame recorded_from_tableau_sim(const std::vector<Operation> &operations);
    void sample(aligned_bits256& out, std::mt19937 &rng);
    std::string str() const;
};

std::ostream &operator<<(std::ostream &out, const SimFrame &ps);


#endif