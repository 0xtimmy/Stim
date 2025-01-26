import stim
import sinter
import os
import matplotlib.pyplot as plt
from typing import List

surface_code_tasks = [
    sinter.Task(
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d * 3,
            distance=d,
            after_clifford_depolarization=noise,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
            before_round_data_depolarization=noise,
        ),
        json_metadata={'d': d, 'r': d * 3, 'p': noise},
    )
    for d in [3, 5, 7]
    for noise in [0.008, 0.009, 0.01, 0.011, 0.012]
]

def main():
    # Collect the samples (takes a few minutes).
    collected_surface_code_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=surface_code_tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=5_000,
        print_progress=True,
    )

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_surface_code_stats,
        x_func=lambda stat: stat.json_metadata['p'],
        group_func=lambda stat: stat.json_metadata['d'],
        failure_units_per_shot_func=lambda stat: stat.json_metadata['r'],
    )
    ax.set_ylim(5e-3, 5e-2)
    ax.set_xlim(0.008, 0.012)
    ax.loglog()
    ax.set_title("Surface Code Error Rates per Round under Circuit Noise")
    ax.set_xlabel("Phyical Error Rate")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.legend()
    fig.set_dpi(120)  # Show it bigger
    fig.savefig("out.png")

if __name__ == '__main__':
    main()

