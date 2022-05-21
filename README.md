# quantum-trajectories
C++ class to dissipative quantum trajectories using ITensor MPS. Time evolution and stochastic quantum jumps are evaluated using local gates (utils/TGate.h).

"DissipativePhaseTransition/" contains the source code to generate the trajectory states in dissipative cavity array, subject to two types of dissipation. The dissipation is monitored (quantum jumps collected) and the generation of entanglement in the trajectory states is investigated. We report a phase transition of averaged entanglement (see https://doi.org/10.1103/PhysRevLett.126.123604)

The two other folders are subsequent projects to study the jump statistics in different models. 1) TFI_decay: transverse-field Ising model with spin decay dissipation, 2) Spin_decay_mix: when dissipative jumps are mixed with unitary. These projects have been investigated more carefully later (see monitoring-entanglement).
