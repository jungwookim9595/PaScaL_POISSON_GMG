#!/usr/bin/env python3
"""Validation and benchmark of the wrapper library against the analytic solution.

Solves the 3D Poisson equation on the GPU through pascal_gmg, on the problem described by a
PARA_INPUT.inp file, and reports the RMS error against the exact solution.  All arrays are
device-resident: the solver allocates them, this example fills them through CuPy views of the very
same memory, and nothing is copied across the language boundary.

Test problem
------------
The solver's boundary treatment is homogeneous Dirichlet on the six domain faces.  The coordinate
slots at index 0 and n+1 sit exactly on the faces (xg[0] = ox, and the gap to the first cell centre
is half a cell), no kernel writes them, and the halo exchange skips them because the neighbour is
MPI_PROC_NULL -- so the zero the allocator leaves there *is* the boundary condition.

The exact solution is therefore a product of sines that vanish on all six faces:

    p_exact = sin(kx (x-ox)) sin(ky (y-oy)) sin(kz (z-oz)),   kd = md * pi / ld
    rhs     = lambda_discrete * p_exact
    lambda_discrete = -sum_d 4 sin(kd hd / 2)^2 / hd^2,       hd = ld / nd

The coefficient is the eigenvalue of the *discrete* Laplacian, not of the continuous one.  That
matters here more than it looks: the discrete operator is not uniform at the boundary, where the
half-cell spacing makes the stencil (2 u_0 - 3 u_1 + u_2)/h^2 instead of (u_0 - 2 u_1 + u_2)/h^2.
The sine survives that row anyway -- with u_0 = 0 the row demands u_2 = (2 cos(k h) + 1) u_1, and
2 cos(kh) sin(kh/2) = sin(3kh/2) - sin(kh/2) gives exactly that -- so p_exact is an exact
eigenvector of the assembled matrix including its boundary rows, and a converged solve must
reproduce it to the level the residual tolerance allows.  Measured at 128^3 with the settings in
PARA_INPUT.inp, the discrete coefficient leaves an RMS of 5.5e-13 and the continuous one -k^2
leaves 7.1e-05 -- eight orders apart, so with the continuous form a real failure would be
indistinguishable from discretisation error.

The default mode numbers are md = 2, which for the reference domain (origin -0.5, length 2) gives
sin(2 pi (x+0.5) / 2) = cos(pi x): the same field the C driver 00_C/src/gpu/poisson_gpu.cu uses.
Only the right-hand side coefficient differs from that driver, which uses the continuous -3 pi^2 --
and which also prints the squared error without taking its square root, so its "RMS" line is a mean
squared error and is not directly comparable with the RMS printed here.

This construction assumes a uniform grid; the example refuses a stretched one (ax/ay/az != 1),
because the eigenvector argument above does not survive variable spacing.

Usage (the path may be omitted, in which case run/PARA_INPUT.inp is used):
    mpirun -n 1 python example/poisson3d_gmg.py
    mpirun -n 1 python example/poisson3d_gmg.py run/PARA_INPUT.inp

The input file fixes the process grid through npx/npy/npz and the run aborts unless their product
equals the number of ranks, so another rank count needs its own copy of the file:
    sed 's/^npx 1$/npx 2/' run/PARA_INPUT.inp > run/PARA_INPUT_np2.inp
    mpirun -n 2 python example/poisson3d_gmg.py run/PARA_INPUT_np2.inp

The input file is the same key-value format the C driver reads, so the C binary accepts these files
unchanged.  Keys the C driver does not know are ignored by it and used here:

    mode_x/mode_y/mode_z   mode number per direction of the exact solution   (default 2)
    rms_tolerance          pass threshold for the RMS error                  (default 1e-8)
    bench_warmup           untimed solves before timing                      (default 2)
    bench_iters            timed solves                                      (default 5)

Rank 0 prints one machine-readable line for a benchmark driver:

    BENCH,<np>,<npx>x<npy>x<npz>,<nx>,<ny>,<nz>,<levels>,<iters>,
          <setup_s>,<solve_s>,<rms>,<OK|BAD>
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pascal_gmg as gmg                                        # noqa: E402

import cupy as cp                                               # noqa: E402
import numpy as np                                              # noqa: E402
from mpi4py import MPI                                          # noqa: E402


# --- input file -------------------------------------------------------------
# Defaults for every key, so a short file still runs.  Types come from the default value: a key
# whose default is an int is parsed as an int.  That is what keeps npx from silently becoming 2.0
# and reaching the c_int argument as a float.
DEFAULTS = {
    "nx": 128, "ny": 128, "nz": 128,
    "ox": -0.5, "oy": -0.5, "oz": -0.5,
    "lx": 2.0, "ly": 2.0, "lz": 2.0,
    "ax": 1.0, "ay": 1.0, "az": 1.0,
    "npx": 1, "npy": 1, "npz": 1,
    "maxiteration": 10, "tolerance": 1.0e-10,
    "number_of_vcycles": 100, "number_of_levels": 6,
    "aggregation_method": 0, "aggregation_level": 0,
    "omega_sor": 1.6,
    # Keys below are used by this example only; the C driver ignores them.
    "mode_x": 2, "mode_y": 2, "mode_z": 2,
    "rms_tolerance": 1.0e-8,
    "bench_warmup": 2, "bench_iters": 5,
}


def read_input(path):
    """Parse the PARA_INPUT.inp key-value format: one 'key value' pair per line, '#' comments."""
    cfg = dict(DEFAULTS)
    with open(path) as fp:
        for lineno, line in enumerate(fp, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'key value', got {line!r}")
            key, value = parts
            if key not in cfg:
                continue                        # forward compatible: unknown keys are not an error
            cfg[key] = int(value) if isinstance(DEFAULTS[key], int) else float(value)
    return cfg


comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

input_file = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "run", "PARA_INPUT.inp")
cfg = read_input(input_file)

n = (cfg["nx"], cfg["ny"], cfg["nz"])
origin = (cfg["ox"], cfg["oy"], cfg["oz"])
length = (cfg["lx"], cfg["ly"], cfg["lz"])
stretch = (cfg["ax"], cfg["ay"], cfg["az"])
nproc = (cfg["npx"], cfg["npy"], cfg["npz"])
mode = (cfg["mode_x"], cfg["mode_y"], cfg["mode_z"])

if nproc[0] * nproc[1] * nproc[2] != size:
    if rank == 0:
        print(f"[ERROR] {input_file}: npx*npy*npz = {nproc[0] * nproc[1] * nproc[2]} "
              f"but this job has {size} ranks", file=sys.stderr)
    sys.exit(1)
if any(a != 1.0 for a in stretch):
    if rank == 0:
        print(f"[ERROR] {input_file}: this validation needs a uniform grid (ax=ay=az=1.0), "
              f"got {stretch}.  The exact solution is an eigenvector of the discrete operator "
              f"only for constant spacing.", file=sys.stderr)
    sys.exit(1)

# --- set up -----------------------------------------------------------------
dev = gmg.bind_gpu()                            # before any CuPy allocation

comm.Barrier()
t0 = MPI.Wtime()
topo = gmg.Topology(nproc=nproc, periodic=(False, False, False))
grid = gmg.Grid(topo, n=n, origin=origin, length=length, stretch=stretch)
plan = gmg.MultigridPlan(grid, nlevel=cfg["number_of_levels"], ncycle=cfg["number_of_vcycles"],
                         aggr_method=cfg["aggregation_method"], aggr_level=cfg["aggregation_level"])
plan.sync()
setup_local = MPI.Wtime() - t0
setup_global = comm.allreduce(setup_local, op=MPI.MAX)

if rank == 0:
    print(f"input {input_file}")
    print(f"grid {n[0]}x{n[1]}x{n[2]}   origin {origin}   length {length}   nproc {nproc}")
    print(f"levels {cfg['number_of_levels']}   vcycles {cfg['number_of_vcycles']}   "
          f"maxiter {cfg['maxiteration']}   tol {cfg['tolerance']:g}   omega {cfg['omega_sor']}   "
          f"aggregation {cfg['aggregation_method']}/{cfg['aggregation_level']}")
    print(f"field shape (nx+2,ny+2,nz+2) = {grid.shape}")
print(f"  rank {rank} cart {topo.ranks}: GPU {dev}  i {grid.ista}..{grid.iend}  "
      f"j {grid.jsta}..{grid.jend}  k {grid.ksta}..{grid.kend}", flush=True)

# --- test problem -----------------------------------------------------------
h = tuple(length[d] / n[d] for d in range(3))
kappa = tuple(mode[d] * math.pi / length[d] for d in range(3))
lam = -sum(4.0 * math.sin(0.5 * kappa[d] * h[d]) ** 2 / h[d] ** 2 for d in range(3))
lam_continuous = -sum(k * k for k in kappa)
if rank == 0:
    print(f"modes {mode}   discrete eigenvalue {lam:.12f}   "
          f"continuous {lam_continuous:.12f}   relative gap "
          f"{abs(lam - lam_continuous) / abs(lam_continuous):.3e}")

# Coordinates are already on the device; the exact field is built there too.
xg, yg, zg = grid.coords()
p_exact = (cp.sin(kappa[0] * (xg - origin[0])) *
           cp.sin(kappa[1] * (yg - origin[1])) *
           cp.sin(kappa[2] * (zg - origin[2])))


def reset():
    """Restore the initial state: zero solution, right-hand side on the interior cells only.

    The zero written into the halo layer of x is what imposes the Dirichlet boundary value on a
    rank that owns a domain face; on an interior face it is overwritten by the halo exchange.  b is
    zeroed outside the interior for the same reason the C driver does it -- no kernel reads it
    there, but leaving stale values would be misleading.
    """
    grid.x[...] = 0.0
    grid.b[...] = 0.0
    grid.b[grid.interior] = lam * p_exact[grid.interior]


# --- correctness ------------------------------------------------------------
reset()
comm.Barrier()
t0 = MPI.Wtime()
plan.solve(maxiter=cfg["maxiteration"], tol=cfg["tolerance"], omega=cfg["omega_sor"])
plan.sync()
solve_first = MPI.Wtime() - t0

diff = grid.x[grid.interior] - p_exact[grid.interior]
err_global = comm.allreduce(float(cp.sum(diff * diff)), op=MPI.SUM)
ncell = n[0] * n[1] * n[2]
rms = math.sqrt(err_global / ncell)

# --- timing -----------------------------------------------------------------
for _ in range(cfg["bench_warmup"]):
    reset()
    plan.solve(maxiter=cfg["maxiteration"], tol=cfg["tolerance"], omega=cfg["omega_sor"])
plan.sync()
comm.Barrier()

times = np.empty(cfg["bench_iters"])
for it in range(cfg["bench_iters"]):
    # Rebuilding the initial state is outside the timer, as it is in the C driver's loop.
    reset()
    plan.sync()
    comm.Barrier()
    t0 = MPI.Wtime()
    plan.solve(maxiter=cfg["maxiteration"], tol=cfg["tolerance"], omega=cfg["omega_sor"])
    plan.sync()
    times[it] = MPI.Wtime() - t0
solve_global = comm.allreduce(float(times.mean()), op=MPI.MAX)

plan.destroy()
grid.destroy()
topo.destroy()

# --- report -----------------------------------------------------------------
ok = rms < cfg["rms_tolerance"]
if rank == 0:
    print()
    print(f"RMS error against analytic solution : {rms:.16e}")
    print(f"pass threshold (rms_tolerance)      : {cfg['rms_tolerance']:.3e}")
    print(f"setup time (rank-max)               : {setup_global * 1e3:.3f} ms")
    print(f"first solve                         : {solve_first * 1e3:.3f} ms")
    print(f"mean solve time (rank-max, {cfg['bench_iters']} runs) : {solve_global * 1e3:.3f} ms")
    print()
    print(f"BENCH,{size},{nproc[0]}x{nproc[1]}x{nproc[2]},{n[0]},{n[1]},{n[2]},"
          f"{cfg['number_of_levels']},{cfg['bench_iters']},"
          f"{setup_global:.6e},{solve_global:.6e},{rms:.6e},{'OK' if ok else 'BAD'}")
    print("PASS" if ok else "FAIL")

sys.exit(0 if ok else 1)
