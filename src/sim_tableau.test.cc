#include "gtest/gtest.h"
#include "sim_tableau.h"

TEST(SimTableau, identity) {
    auto s = SimTableau(1);
    ASSERT_EQ(s.measure({0})[0], false);
}

TEST(SimTableau, bit_flip) {
    auto s = SimTableau(1);
    s.H({0});
    s.SQRT_Z({0});
    s.SQRT_Z({0});
    s.H({0});
    ASSERT_EQ(s.measure({0})[0], true);
    s.X({0});
    ASSERT_EQ(s.measure({0})[0], false);
}

TEST(SimTableau, identity2) {
    auto s = SimTableau(2);
    ASSERT_EQ(s.measure({0})[0], false);
    ASSERT_EQ(s.measure({1})[0], false);
}

TEST(SimTableau, bit_flip_2) {
    auto s = SimTableau(2);
    s.H({0});
    s.SQRT_Z({0});
    s.SQRT_Z({0});
    s.H({0});
    ASSERT_EQ(s.measure({0})[0], true);
    ASSERT_EQ(s.measure({1})[0], false);
}

TEST(SimTableau, epr) {
    auto s = SimTableau(2);
    s.H({0});
    s.CX({0, 1});
    ASSERT_EQ(s.is_deterministic(0), false);
    ASSERT_EQ(s.is_deterministic(1), false);
    auto v1 = s.measure({0})[0];
    ASSERT_EQ(s.is_deterministic(0), true);
    ASSERT_EQ(s.is_deterministic(1), true);
    auto v2 = s.measure({1})[0];
    ASSERT_EQ(v1, v2);
}

TEST(SimTableau, big_determinism) {
    auto s = SimTableau(1000);
    s.H({0});
    ASSERT_FALSE(s.is_deterministic(0));
    for (size_t k = 1; k < 1000; k++) {
        ASSERT_TRUE(s.is_deterministic(k));
    }
}

TEST(SimTableau, phase_kickback_consume_s_state) {
    for (size_t k = 0; k < 8; k++) {
        auto s = SimTableau(2);
        s.H({1});
        s.SQRT_Z({1});
        s.H({0});
        s.CX({0, 1});
        ASSERT_EQ(s.is_deterministic(1), false);
        bool v1 = s.measure({1})[0];
        if (v1) {
            s.SQRT_Z({0});
            s.SQRT_Z({0});
        }
        s.SQRT_Z({0});
        s.H({0});
        ASSERT_EQ(s.is_deterministic(0), true);
        ASSERT_EQ(s.measure({0})[0], true);
    }
}

TEST(SimTableau, phase_kickback_preserve_s_state) {
    auto s = SimTableau(2);

    // Prepare S state.
    s.H({1});
    s.SQRT_Z({1});

    // Prepare test input.
    s.H({0});

    // Kickback.
    s.CX({0, 1});
    s.H({1});
    s.CX({0, 1});
    s.H({1});

    // Check.
    s.SQRT_Z({0});
    s.H({0});
    ASSERT_EQ(s.is_deterministic(0), true);
    ASSERT_EQ(s.measure({0})[0], true);
    s.SQRT_Z({1});
    s.H({1});
    ASSERT_EQ(s.is_deterministic(1), true);
    ASSERT_EQ(s.measure({1})[0], true);
}

TEST(SimTableau, kickback_vs_stabilizer) {
    auto sim = SimTableau(3);
    sim.H({2});
    sim.CX({2, 0});
    sim.CX({2, 1});
    sim.SQRT_Z({0});
    sim.SQRT_Z({1});
    sim.H({0});
    sim.H({1});
    sim.H({2});
    ASSERT_EQ(sim.inv_state.str(),
              "+-xz-xz-xz-\n"
              "| +- +- ++\n"
              "| ZY __ _X\n"
              "| __ ZY _X\n"
              "| XX XX XZ");
}

TEST(SimTableau, s_state_distillation_low_depth) {
    for (size_t reps = 0; reps < 10; reps++) {
        auto sim = SimTableau(9);

        std::vector<std::vector<uint8_t>> stabilizers = {
                {0, 1, 2, 3},
                {0, 1, 4, 5},
                {0, 2, 4, 6},
                {1, 2, 4, 7},
        };
        std::vector<std::unordered_map<std::string, std::vector<uint8_t>>> checks{
                {{"s", {0}}, {"q", stabilizers[0]}},
                {{"s", {1}}, {"q", stabilizers[1]}},
                {{"s", {2}}, {"q", stabilizers[2]}},
        };

        std::vector<bool> stabilizer_measurements;
        size_t anc = 8;
        for (const auto &stabilizer : stabilizers) {
            sim.H({anc});
            for (const auto &k : stabilizer) {
                sim.CX({anc, k});
            }
            sim.H({anc});
            ASSERT_EQ(sim.is_deterministic(anc), false);
            bool v = sim.measure({anc})[0];
            if (v) {
                sim.X({anc});
            }
            stabilizer_measurements.push_back(v);
        }

        std::vector<bool> qubit_measurements;
        for (size_t k = 0; k < 7; k++) {
            sim.SQRT_Z({k});
            sim.H({k});
            qubit_measurements.push_back(sim.measure({k})[0]);
        }

        bool sum = false;
        for (auto e : stabilizer_measurements) {
            sum ^= e;
        }
        for (auto e : qubit_measurements) {
            sum ^= e;
        }
        if (sum) {
            sim.Z({7});
        }

        sim.SQRT_Z({7});
        sim.H({7});
        ASSERT_EQ(sim.is_deterministic(7), true);
        ASSERT_EQ(sim.measure({7})[0], false);

        for (const auto &c : checks) {
            bool r = false;
            for (auto k : c.at("s")) {
                r ^= stabilizer_measurements[k];
            }
            for (auto k : c.at("q")) {
                r ^= qubit_measurements[k];
            }
            ASSERT_EQ(r, false);
        }
    }
}

TEST(SimTableau, s_state_distillation_low_space) {
    for (size_t rep = 0; rep < 10; rep++) {
        auto sim = SimTableau(5);

        std::vector<std::vector<uint8_t>> phasors = {
                {0,},
                {1,},
                {2,},
                {0, 1, 2},
                {0, 1, 3},
                {0, 2, 3},
                {1, 2, 3},
        };

        size_t anc = 4;
        for (const auto &phasor : phasors) {
            sim.H({anc});
            for (const auto &k : phasor) {
                sim.CX({anc, k});
            }
            sim.H({anc});
            sim.SQRT_Z({anc});
            sim.H({anc});
            ASSERT_EQ(sim.is_deterministic(anc), false);
            bool v = sim.measure({anc})[0];
            if (v) {
                for (const auto &k : phasor) {
                    sim.X({k});
                }
                sim.X({anc});
            }
        }

        for (size_t k = 0; k < 3; k++) {
            ASSERT_EQ(sim.is_deterministic(k), true);
            ASSERT_EQ(sim.measure({k})[0], false);
        }
        sim.SQRT_Z({3});
        sim.H({3});
        ASSERT_EQ(sim.is_deterministic(3), true);
        ASSERT_EQ(sim.measure({3})[0], true);
    }
}

TEST(SimTableau, single_qubit_gates_consistent_with_tableau_data) {
    auto t = Tableau::random(10);
    SimTableau sim(10);
    SimTableau sim2(10);
    sim.inv_state = t;
    sim2.inv_state = t;
    for (const auto &kv : SIM_TABLEAU_GATE_FUNC_DATA) {
        const auto &name = kv.first;
        if (name == "M" || name == "R") {
            continue;
        }
        const auto &action = kv.second;
        const auto &inverse_op_tableau = GATE_TABLEAUS.at(GATE_INVERSE_NAMES.at(name));
        if (inverse_op_tableau.num_qubits == 2) {
            action(sim, {7, 4});
            sim2.tableau_op(name, {7, 4});
            t.inplace_scatter_prepend(inverse_op_tableau, {7, 4});
        } else {
            action(sim, {5});
            sim2.tableau_op(name, {5});
            t.inplace_scatter_prepend(inverse_op_tableau, {5});
        }
        ASSERT_EQ(sim.inv_state, t) << name;
        ASSERT_EQ(sim.inv_state, sim2.inv_state) << name;
    }
}

TEST(SimTableau, simulate) {
    auto results = SimTableau::simulate(Circuit::from_text(
            "H 0\n"
            "CNOT 0 1\n"
            "M 0\n"
            "M 1\n"
            "M 2\n"));
    ASSERT_EQ(results[0], results[1]);
    ASSERT_EQ(results[2], false);
}

TEST(SimTableau, simulate_reset) {
    auto results = SimTableau::simulate(Circuit::from_text(
            "X 0\n"
            "M 0\n"
            "R 0\n"
            "M 0\n"
            "R 0\n"
            "M 0\n"));
    ASSERT_EQ(results[0], true);
    ASSERT_EQ(results[1], false);
    ASSERT_EQ(results[2], false);
}

TEST(SimTableau, to_vector_sim) {
    SimTableau sim_tab(2);
    SimVector sim_vec(2);
    ASSERT_TRUE(sim_tab.to_vector_sim().approximate_equals(sim_vec, true));

    sim_tab.X({0});
    sim_vec.apply("X", 0);
    ASSERT_TRUE(sim_tab.to_vector_sim().approximate_equals(sim_vec, true));

    sim_tab.H({0});
    sim_vec.apply("H", 0);
    ASSERT_TRUE(sim_tab.to_vector_sim().approximate_equals(sim_vec, true));

    sim_tab.SQRT_Z({0});
    sim_vec.apply("SQRT_Z", 0);
    ASSERT_TRUE(sim_tab.to_vector_sim().approximate_equals(sim_vec, true));

    sim_tab.CX({0, 1});
    sim_vec.apply("CX", 0, 1);
    ASSERT_TRUE(sim_tab.to_vector_sim().approximate_equals(sim_vec, true));

    sim_tab.inv_state = Tableau::random(10);
    sim_vec = sim_tab.to_vector_sim();
    ASSERT_TRUE(sim_tab.to_vector_sim().approximate_equals(sim_vec, true));

    sim_tab.tableau_op("XCX", {4, 7});
    sim_vec.apply("XCX", 4, 7);
    ASSERT_TRUE(sim_tab.to_vector_sim().approximate_equals(sim_vec, true));
}

bool vec_sim_corroborates_measurement_process(const SimTableau &sim, std::vector<size_t> measurement_targets) {
    SimTableau sim_tab = sim;
    auto vec_sim = sim_tab.to_vector_sim();
    auto results = sim_tab.measure(measurement_targets);
    PauliStringVal buf(sim_tab.inv_state.num_qubits);
    auto p = buf.ptr();
    for (size_t k = 0; k < measurement_targets.size(); k++) {
        p.z_ref[measurement_targets[k]] = true;
        p.bit_ptr_sign.set(results[k]);
        float f = vec_sim.project(p);
        if (fabsf(f - 0.5) > 1e-4 && fabsf(f - 1) > 1e-4) {
            return false;
        }
        p.z_ref[measurement_targets[k]] = false;
    }
    return true;
}

TEST(SimTableau, measurement_vs_vector_sim) {
    for (size_t k = 0; k < 10; k++) {
        SimTableau sim_tab(2);
        sim_tab.inv_state = Tableau::random(2);
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {0}));
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {1}));
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {0, 1}));
    }
    for (size_t k = 0; k < 10; k++) {
        SimTableau sim_tab(4);
        sim_tab.inv_state = Tableau::random(4);
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {0, 1}));
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {2, 1}));
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {0, 1, 2, 3}));
    }
    {
        SimTableau sim_tab(12);
        sim_tab.inv_state = Tableau::random(12);
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {0, 1, 2, 3}));
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {0, 10, 11}));
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {11, 5, 7}));
        ASSERT_TRUE(vec_sim_corroborates_measurement_process(sim_tab, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    }
}

TEST(SimTableau, extended_collapse) {
    SimTableau sim(3);
    sim.H({0});
    sim.CX({0, 1, 0, 2});
    auto r = sim.inspected_collapse({1});
    ASSERT_EQ(r[0].str(), "+X0*X1*X2");
}
