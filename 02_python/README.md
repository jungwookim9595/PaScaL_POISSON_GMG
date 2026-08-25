# PaScaL_POISSON_GMG — Python (GPU)

Python interface to the device-resident CUDA geometric-multigrid Poisson solver in `00_C/src/gpu`.

The solver library is reused as-is; nothing under `00_C` is modified. Its sources are recompiled
here with `-fPIC` into `lib/libgmg_capi.so`, together with the thin C shim `src/gmg_capi.cu`, and
driven from Python through `ctypes` with CuPy device arrays.

## Layout

```
02_python/
├── Makefile              build lib/libgmg_capi.so from 00_C/src/gpu + src/gmg_capi.cu
├── src/
│   ├── gmg_capi.cu       C shim: opaque handles, accessors, library globals
│   └── pascal_gmg.py     ctypes interface: Topology, Grid, MultigridPlan
├── example/
│   └── poisson3d_gmg.py  validation and benchmark against the analytic solution
├── run/PARA_INPUT.inp    problem definition, 1 process; the default when the
│                         example is run without a path argument
├── environment.yml       conda environment
└── setup_env.sh          toolchain + conda env + mpi4py, all in one
```

Job scripts (`*.slurm`) and alternative input files are not tracked — they are site- and run-specific.
`.gitignore` keeps them out, so a working copy may hold more than the list above.

Compared with the Fortran-based wrappers of `PaScaL_TDMA` and `PaScaL_POISSON_FFT`, two things are
different and both make this side smaller:

- The solver is already C, so the shim performs no ABI translation. It exists only to hide the
  structs behind opaque handles, to enforce the create/destroy pairings of the reference driver,
  and to set the library globals (`myrank`, `nprocs`, `np_dim[]`, `period[]`).
- The solver builds its own MPI cartesian topology inside `mpi_topology_create()`. The FFT wrapper
  had to reproduce the topology, the block decomposition and 32 MPI derived datatypes in Python;
  here Python supplies the process counts and periodicity flags and nothing else.

The example is new. In `00_C` there is no separate example directory: `src/gpu/poisson_gpu.cu`
holds `main()` together with the input-file parser, the right-hand-side kernel, the RMS reduction
and the timing loop. That file is excluded from the shared library, and `example/poisson3d_gmg.py`
takes over its role.

## Dependencies

- NVIDIA HPC SDK (nvcc, and its bundled HPC-X OpenMPI)
- Python, NumPy, CuPy
- mpi4py, **built from source against the same MPI** the library is linked with

`setup_env.sh` arranges all of it: it loads the toolchain, pins nvcc to CUDA 12.9 (the module puts a
CUDA 13 `nvcc` first in `PATH`, which would not match the CuPy wheel), creates and activates the
conda environment, and builds mpi4py from source.

Creating the conda environment is the one step that gives trouble on Neuron. conda 23.3.1 fetching
the ~435 MB conda-forge repodata **hangs**: on two separate attempts it sat for 40 and 31 minutes
with four sockets in `CLOSE-WAIT`, its threads parked in `futex_wait`, having used 1 and 2.4 seconds
of CPU respectively. The CDN drops the HTTPS connection and conda never notices. Waiting does not
help — the process is deadlocked, not slow.

The call is therefore wrapped in a timeout (`GMG_CONDA_TIMEOUT`, default 1800 s, `SIGKILL` 30 s
after `SIGTERM` since a futex-parked conda ignores the latter), and on failure it prints the ways
out. The reliable one here is `--offline`, which resolves python and pip from the local package
cache and never touches conda-forge; the pip wheels still come from PyPI, which works fine:

```bash
conda env create --offline -f environment.yml -p <prefix>   # verified to work on Neuron
ENV_NAME=pascal-poisson-gpu source setup_env.sh             # reuse the FFT env (same dependencies)
GMG_CONDA_TIMEOUT=3600 source setup_env.sh                  # only if it is genuinely slow
```

To tell "slow" from "hung": `ps -o etime,%cpu -p $(pgrep -f 'conda-env create')`. A few minutes at
~0 % CPU means hung.

After the environment exists, re-running `setup_env.sh` picks it up and continues.

## Build and run

```bash
source 02_python/setup_env.sh     # must be sourced, not executed
cd 02_python
make                              # -> lib/libgmg_capi.so
mpirun -n 1 python example/poisson3d_gmg.py            # uses run/PARA_INPUT.inp
```

`make GPU_CC=80` restricts the build to A100 and roughly halves the compile time; the default
`GPU_CC=70 80` covers both the V100 and A100 partitions.

The example takes the path of an input file, defaulting to `run/PARA_INPUT.inp`. That file fixes the
process grid through `npx`/`npy`/`npz`, and the run aborts unless their product equals the number of
ranks — so a different rank count needs its own copy:

```bash
sed 's/^npx 1$/npx 2/' run/PARA_INPUT.inp > run/PARA_INPUT_np2.inp
mpirun -n 2 python example/poisson3d_gmg.py run/PARA_INPUT_np2.inp
```

On a batch system, wrap those two lines in a job script of your own; `setup_env.sh` is the only
setup it needs, and `make` is safe to call from it since it is a no-op when the library is current.

## Interface

```python
import pascal_gmg as gmg

gmg.bind_gpu()                                     # before any CuPy allocation
topo = gmg.Topology(nproc=(2, 1, 1))               # 3D cartesian process grid
grid = gmg.Grid(topo, n=(128, 128, 128),
                origin=(-0.5,) * 3, length=(2.0,) * 3)
plan = gmg.MultigridPlan(grid, nlevel=6, ncycle=100)

grid.b[grid.interior] = rhs                        # CuPy view of the solver's own buffer
plan.solve(maxiter=10, tol=1e-10, omega=1.6)
plan.sync()
solution = grid.x[grid.interior]

plan.destroy(); grid.destroy(); topo.destroy()
```

Fields are `(nx+2, ny+2, nz+2)` float64 in **C order** — the solver indexes
`i*(ny+2)*(nz+2) + j*(nz+2) + k`. The Fortran-based wrappers of the sibling projects use `order='F'`
instead; this one does not.

No data is copied across the boundary. `geometry_subdomain_create_gpu` allocates `x`, `b` and `r` on
the device, and `Grid` wraps those addresses as CuPy arrays through `cupy.cuda.UnownedMemory`, so
`grid.b[...] = ...` is a kernel writing straight into the solver's own buffer.

### Boundary conditions

The halo slots at index 0 and n+1 of a non-periodic direction sit exactly **on** the domain faces
(`xg[0] == ox`, and the gap to the first cell centre is half a cell). No kernel writes them, and
the halo exchange skips them because the neighbour is `MPI_PROC_NULL`. Whatever is stored there is
therefore a Dirichlet boundary value, and the zero left by the allocator makes it homogeneous.

## Test problem

`example/poisson3d_gmg.py` solves for

```
p_exact = prod_d sin(k_d (x_d - o_d)),        k_d = m_d pi / l_d
rhs     = lambda_discrete * p_exact
lambda_discrete = -sum_d 4 sin(k_d h_d / 2)^2 / h_d^2
```

which vanishes on all six faces and so matches the boundary condition above. The coefficient is the
eigenvalue of the **discrete** Laplacian. That matters at the boundary rows, where the half-cell
spacing turns the stencil into `(2u_0 - 3u_1 + u_2)/h^2`: with `u_0 = 0` that row demands
`u_2 = (2cos(kh) + 1) u_1`, and `2cos(kh)sin(kh/2) = sin(3kh/2) - sin(kh/2)` supplies exactly that,
so the sine is an exact eigenvector of the assembled matrix including its boundary rows. A converged
solve must reproduce it to the level the residual tolerance allows.

The continuous coefficient `-k^2` — which the C driver uses — is off by 2.0e-4 relative at 128³, and
that is not a rounding detail: measured on the same grid and solver settings, it leaves

| right-hand side coefficient | RMS error at 128³ |
|---|---|
| discrete, `-sum 4 sin(k h/2)^2 / h^2` | 5.5e-13 |
| continuous, `-k^2` | 7.1e-05 |

eight orders apart, so a real failure would be indistinguishable from the discretisation error if
the continuous form were used. The C driver also prints the squared error without taking its square
root, so its `RMS` line is a mean squared error and is not directly comparable with the RMS here.

The construction needs a uniform grid; the example refuses `ax`/`ay`/`az` other than 1.0.

The default mode numbers `m_d = 2` reproduce, for the reference domain (origin −0.5, length 2), the
same field `cos(pi x) cos(pi y) cos(pi z)` that the C driver uses.

## Verified results

A100, `nvhpc/25.11_cuda12`, CUDA 12.9:

| grid | levels | ranks | process grid | aggregation | RMS error | mean solve |
|---|---|---|---|---|---|---|
| 128³ | 6 | 1 | 1×1×1 | 0 (none) | 5.5058586453670508e-13 | 90 ms |
| 128³ | 6 | 2 | 2×1×1 | 0 (none) | 5.5058586453670508e-13 | 159 ms |
| 128³ | 6 | 2 | 2×1×1 | 1 (single, level 3) | 5.5058586453670508e-13 | 114 ms |
| 256³ | 7 | 1 | 1×1×1 | 0 (none) | 7.2595179698144940e-13 | 423 ms |
| 256³ | 7 | 2 | 2×1×1 | 0 (none) | 7.2595179698144940e-13 | 347 ms |

At each grid size the ranks agree to the last digit, which is what a correct decomposition should
give on this problem. The 128³ two-rank case is slower than one rank — the halo exchanges and the
coarse-level collectives outweigh the halved work at that size; at 256³ the split pays off.

Three further checks, all at 64³ over np = 1 and 2:

- Four consecutive solves through the same plan give bit-identical RMS, so the plan carries no
  state from one solve into the next — what a CFD time-stepping loop needs.
- A fifth solve started from the previous solution instead of from zero converges to
  **1.87e-16**, i.e. round-off. That is an independent confirmation of the eigenvector argument
  above: the exact solution of the assembled discrete system really is the analytic field, and the
  ~7e-13 in the table is the residual tolerance, not a discretisation error.
- `solve(sol=..., rhs=...)` with CuPy arrays the caller allocated gives bit-identical results to
  the default, which uses the solver's own `grid.x` and `grid.b`.

## Known issues in the reference sources

These live in `00_C` and are left untouched; they are recorded here because they affect anyone
reading or extending this interface.

- **`aggregation_method = 2` (adaptive) aborts.** Every configuration tried — np = 1 and 2,
  `number_of_levels` 2 through 6, `aggregation_level` 0 through 4 — dies with
  `CUDA error multigrid_gpu.cu:1551: invalid configuration argument`, a `restriction_kernel` launch
  whose grid dimension is zero. The prebuilt `00_C/run/test_poisson` fails identically on the same
  input, so this is a defect in the solver and not in the wrapper. It matches the comment in
  `poisson_gpu.cu` (`2 : adaptive aggregation (not implemented yet)`). Methods 0 and 1 are verified
  above. `00_C/run/PARA_INPUT.inp` ships with method 2 selected.
- **`geometry.h` names two argument groups in the wrong order.** The declaration of
  `geometry_domain_create_gpu` reads `(..., double lx, ly, lz, double ox, oy, oz, ...)` while the
  definition in `geometry_gpu.cu` reads `(..., double ox, oy, oz, double lx, ly, lz, ...)`. All six
  are `double`, so the mismatch is in the names only and every existing call binds correctly — but
  a binding written from the header alone would swap origin and length. `src/gmg_capi.cu` follows
  the definition and says so.
- **`mpi_topology_destroy()` frees only `mpi_world_cart`**, leaking the three 1D sub-communicators
  and `comm_boundary`. Build the topology once per process; `Topology` refuses a second one.
- **`00_C/Makefile`'s `gpu` target loads a module that does not exist** (`nvhpc/25.11_`; the real
  names are `nvhpc/25.11_cuda12` / `_cuda13`). `module load` returns success without loading
  anything, so that build silently depends on whatever is already in `PATH`. This directory does
  not use that target.
