# D3REAM

　　D3REAM is a $\textit{de novo}$ inverse materials design (DNID) approach that fully automates the materials design for target physical properties, without the need to provide atomic composition, chemical stoichiometry, and crystal structure in advance.

## System requirements

- Python >= 3.9
- scikit-opt == 0.6.6
- hyperopt == 0.2.7
- optuna == 3.3.0
- megnet == 1.3.2
- tensorflow == 2.9.3
- ase == 3.22.1
- m3gnet == 0.2.4
- pymatgen == 2023.10.11

## Getting started

We provide the examples of D3REAM for the inverse design materials with target properties. The examples show at the path `D3REAM/example`.

- `0.input_file` shows the input parameters of D3REAM.
- `1.UPot-BO` shows the examples of D3REAM for the inverse design materials with high cohesive energy and bulk modulus by using the UPot-BO method.
- `2.UPot-EES` shows the examples of D3REAM for the inverse design materials with high cohesive energy and thermal expansion by using the UPot-EES method.

## The details of input parameters of D3REAM

```python
[BASE]

# The chemical formula of the compound, element symbol + count, i.e., Ca4 S4, Cs1 Pb1 I3
# compound = Ca4 S4

# [2] / [1-10] / [1-10, 15] / [1-10, 15-20] / [1, 5-10, 15, 16]
atom_element = [1,3-9,11-17,19-22,29-35,37-40,47-53,55-56,81-83] [1,3-9,11-17,19-22,29-35,37-40,47-53,55-56,81-83]

# [2] / [1-10] / [1-10, 15] / [1-10, 15-20] / [1, 5-10, 15, 16]
atom_count = [1-5] [1-5]

# use or nor search children cell
use_children_cell = True

# limit the max atomic distance
#   1) min_atomic_dist_limit = 0, no limit;
#   2) min_atomic_dist_limit < 0, relative distance, dist_ab < (radii_a+radii_b)*abs(min_atomic_dist_limit);
#   3) min_atomic_dist_limit > 0, absolute distance (unit: Angstrom), dist_ab < min_atomic_dist_limit
min_atomic_dist_limit = -0.7

# [min_V, max_V], limit cell volume size; if `volume_limit = [0, 0 ]`, no limit
volume_limit = [0, 0]

# limit the max vacuum size (unit: Angstrom); if `max_vacuum_limit = 0`, no limit
max_vacuum_limit = 5.0

# Output path, use to save the results.
output_path = .

[CALCULATOR]

# megnet, m3gnet, vasp
calculator = m3gnet

# The GN model file path, it is better to use absolute path.
calculator_path = F:\d3ream\calculators\m3gnet\origin_model\EFS2021

# relax or not
use_calculator_relax = True

# keep symmetry or not, when relax structure
use_keep_symmetry = True

# symmetry precicion
symprec = 0.001

# Load model and predict using GPU
use_gpu = False

[OPTIMIZER]

# Search algorithm: 1) 'rand' (Random Search); 2) 'tpe' (Bayesian Optimization);
#                   3) 'pso' (Particle Swarm Optimization);
#                   4) 'etpe'
#                   5) `tpe2` (Bayesian Optimization)
algorithm = tpe2

# The count of initial random points, only valid when the algorithm is tpe
n_init = 200

# The maximum steps of program runs
max_step = 5000

# Specify the random seed, -1 is None
rand_seed = 100

# TODO support future
use_resume = False

# only support `tpe2`
n_mpi = 1

# Database URL, only n_mpi>=2
storage = ''

[LATTICE]

# [2] / [1-10] / [1-10, 15] / [1-10, 15-20] / [1, 5-10, 15, 16]
space_group = [1-230]

# Generate WyckPos site: 1) 1 -> Generate `max_wyck_pos_count` WyckPos combinations before optimization;
#                        2) 2 -> Generate all WyckPos combinations after optimization (not recommended when the number of atoms > 15);
#                        3) 3 -> Generate WyckPos by optimization algorithms with random site (['a', 'a', 'b', 'b', [rand], ...]);
#                        4) 4 -> Generate WyckPos by optimization algorithms strictly (['a', 'a', 'b', 'b']);
wyck_pos_gen = 3

# The maximum count of WyckPos combinations, only valid when `wyck_pos_gen = 1`
max_wyck_pos_count = 200000

# use or nor flexible WyckPos site (# TODO delete)
# use_flexible_site = True
# Lattice a,b,c (unit: Angstrom):
# [2] / [1-10] / [1-10, 15] / [1-10, 15-20] / [1, 5-10, 15, 16]
lattice_a = [2-30]
lattice_b = [2-30]
lattice_c = [2-30]

# Lattice alpha,beta,gamma (unit: degree):
# [2] / [1-10] / [1-10, 15] / [1-10, 15-20] / [1, 5-10, 15, 16]
lattice_alpha = [20-160]
lattice_beta = [20-160]
lattice_gamma = [20-160]

# float
# lattice_precision = 0.1
```

## Cite

　　If you use D3REAM for research, please consider citing our paper.
