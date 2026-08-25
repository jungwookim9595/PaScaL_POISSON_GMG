"""Python interface to the device-resident CUDA geometric-multigrid Poisson solver of PaScaL_POISSON_GMG.

Loads libgmg_capi.so (ctypes) and exposes the solver through three classes that follow the call
sequence of the C reference driver (00_C/src/gpu/poisson_gpu.cu):

    Topology        the 3D cartesian process grid.  The solver builds it itself from MPI_COMM_WORLD,
                    so unlike the FFT wrapper this side only supplies the process counts and the
                    periodicity flags.  One per process.
    Grid            the global domain plus this rank's block of it, and the device fields that live
                    on that block.
    MultigridPlan   the Poisson operator and the level hierarchy: create, solve, destroy.

Nothing but device addresses crosses the language boundary.  The solver allocates its own fields
(geometry_subdomain_create_gpu cudaMallocs x, b and r), and Grid hands them back as CuPy arrays
that alias the very same memory, so filling a right-hand side from Python is a kernel on data the
solver already owns -- there is no copy in either direction.

Array convention: dtype float64, C order (the solver indexes i*(ny+2)*(nz+2) + j*(nz+2) + k).

    x, b, r     (nx+2, ny+2, nz+2)     one halo layer on each side; interior is [1:nx+1, 1:ny+1, 1:nz+1]
    xg, yg, zg  (nx+2,) (ny+2,) (nz+2) coordinates

Index 0 and n+1 of a coordinate vector are not ghost cell centres: they hold the two domain faces
themselves, so the distance from index 0 to index 1 is half a cell.  In a non-periodic direction the
matching slots of x are never written -- no kernel touches them and the halo exchange skips a
MPI_PROC_NULL neighbour -- so whatever is stored there is a Dirichlet boundary value, and the zero
left by the allocator makes the boundary condition homogeneous.

Typical use (one MPI rank per GPU):

    import pascal_gmg as gmg

    gmg.bind_gpu()                                  # before any CuPy allocation
    topo = gmg.Topology(nproc=(2, 1, 1))
    grid = gmg.Grid(topo, n=(128, 128, 128), origin=(-0.5,) * 3, length=(2.0,) * 3)
    plan = gmg.MultigridPlan(grid, nlevel=6, ncycle=100)

    grid.b[grid.interior] = ...                     # right-hand side, interior cells only
    plan.solve(maxiter=10, tol=1e-10, omega=1.6)
    plan.sync()
    ...                                             # read grid.x
    plan.destroy(); grid.destroy(); topo.destroy()

MPI is not passed in and cannot be: the solver calls MPI_COMM_WORLD directly.  Python and this
library must therefore be built against the same MPI, which setup_env.sh arranges by building
mpi4py from source against the toolchain's OpenMPI.
"""
import ctypes
import os

# mpi4py comes first so that MPI_Init has run by the time anything here calls into the library:
# gmg_topology_create needs a live MPI_COMM_WORLD, and the solver reads myrank/nprocs from it.
# (The FFT wrapper has the opposite constraint -- its shared library must load *before* mpi4py or
# its OpenACC runtime enumerates no GPU.  This solver is pure CUDA with no OpenACC regions, so it
# has no such runtime to lose, and the natural order applies.)
from mpi4py import MPI

import cupy as cp
import numpy as np

_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib", "libgmg_capi.so")
_lib = ctypes.CDLL(_SO)

# argtypes are not decoration.  Without them ctypes passes every Python int as a 64-bit value while
# the shim reads 32-bit int, and arguments arrive shifted; and it would pass a device address
# through a C int, truncating the pointer.
_INT_P = ctypes.POINTER(ctypes.c_int)

_lib.gmg_mpi_initialized.restype = ctypes.c_int
_lib.gmg_mpi_initialized.argtypes = []
_lib.gmg_world_info.restype = ctypes.c_int
_lib.gmg_world_info.argtypes = [_INT_P] * 2

_lib.gmg_topology_create.restype = ctypes.c_int
_lib.gmg_topology_create.argtypes = [ctypes.c_int] * 6
_lib.gmg_topology_destroy.restype = None
_lib.gmg_topology_destroy.argtypes = []
_lib.gmg_topology_ranks.restype = None
_lib.gmg_topology_ranks.argtypes = [_INT_P] * 3

_lib.gmg_device_count.restype = ctypes.c_int
_lib.gmg_device_count.argtypes = []
_lib.gmg_set_device.restype = ctypes.c_int
_lib.gmg_set_device.argtypes = [ctypes.c_int]
_lib.gmg_get_device.restype = ctypes.c_int
_lib.gmg_get_device.argtypes = []
_lib.gmg_device_sync.restype = ctypes.c_int
_lib.gmg_device_sync.argtypes = []

_lib.gmg_domain_create.restype = ctypes.c_void_p
_lib.gmg_domain_create.argtypes = [ctypes.c_int] * 3 + [ctypes.c_double] * 9
_lib.gmg_domain_destroy.restype = None
_lib.gmg_domain_destroy.argtypes = [ctypes.c_void_p]

_lib.gmg_subdomain_create.restype = ctypes.c_void_p
_lib.gmg_subdomain_create.argtypes = [ctypes.c_void_p]
_lib.gmg_subdomain_destroy.restype = None
_lib.gmg_subdomain_destroy.argtypes = [ctypes.c_void_p]
_lib.gmg_subdomain_dims.restype = None
_lib.gmg_subdomain_dims.argtypes = [ctypes.c_void_p] + [_INT_P] * 3
_lib.gmg_subdomain_range.restype = None
_lib.gmg_subdomain_range.argtypes = [ctypes.c_void_p] + [_INT_P] * 6
_lib.gmg_subdomain_boundary.restype = None
_lib.gmg_subdomain_boundary.argtypes = [ctypes.c_void_p] + [_INT_P] * 6
# restype c_void_p, not the default c_int: a device address truncated to 32 bits is a wild pointer.
_lib.gmg_subdomain_field.restype = ctypes.c_void_p
_lib.gmg_subdomain_field.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.gmg_subdomain_coord.restype = ctypes.c_void_p
_lib.gmg_subdomain_coord.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.gmg_matrix_create.restype = ctypes.c_void_p
_lib.gmg_matrix_create.argtypes = [ctypes.c_void_p]
_lib.gmg_matrix_destroy.restype = None
_lib.gmg_matrix_destroy.argtypes = [ctypes.c_void_p]

_lib.gmg_multigrid_create.restype = ctypes.c_int
_lib.gmg_multigrid_create.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
_lib.gmg_multigrid_solve.restype = ctypes.c_int
_lib.gmg_multigrid_solve.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_double,
                                                             ctypes.c_double]
_lib.gmg_multigrid_destroy.restype = None
_lib.gmg_multigrid_destroy.argtypes = []

_FIELD_X, _FIELD_B, _FIELD_R = 0, 1, 2

__all__ = ["device_count", "bind_gpu", "Topology", "Grid", "MultigridPlan"]


# ---------------------------------------------------------------------------
# Device binding
# ---------------------------------------------------------------------------
def device_count():
    """Number of visible CUDA devices."""
    return cp.cuda.runtime.getDeviceCount()


def bind_gpu(comm=MPI.COMM_WORLD, device=None):
    """Bind this rank to one GPU.  Call before allocating any CuPy array or creating a Grid.

    The device is set on both runtimes.  nvcc links the CUDA runtime into libgmg_capi.so
    statically, so the library carries its own copy of the runtime's current-device state; a CuPy
    device selection alone would leave the solver allocating on device 0 while the CuPy views point
    at another GPU.  Setting both keeps the two in step.

    device defaults to the node-local rank, as the C driver's `myrank % ngpus` does for a single
    node.  Node-local is the more robust form: world rank modulo device count collides as soon as
    the job spans more than one node.
    """
    if device is None:
        local = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
        device = local.rank % max(1, device_count())
        local.Free()
    cp.cuda.Device(device).use()
    status = _lib.gmg_set_device(device)
    if status != 0:
        raise RuntimeError(f"cudaSetDevice({device}) failed (status {status})")
    return device


def _dev_view(ptr, shape, owner):
    """Wrap a device address owned by the library as a CuPy array, without copying.

    UnownedMemory is what marks the buffer as foreign: a plain cupy allocation would have Python's
    garbage collector call cudaFree on it, and geometry_subdomain_destroy_gpu would then free the
    same pointer a second time.  owner is kept as a reference so the Grid cannot be collected while
    a view into its memory is still alive.
    """
    nbytes = int(np.prod(shape)) * np.dtype(np.float64).itemsize
    mem = cp.cuda.UnownedMemory(int(ptr), nbytes, owner)
    return cp.ndarray(shape, dtype=cp.float64, memptr=cp.cuda.MemoryPointer(mem, 0))


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
class Topology:
    """The 3D cartesian process grid, MPI_Cart_create'd inside the library.

    nproc     (npx, npy, npz); the product must equal comm.size
    periodic  periodicity per direction.  False leaves the halo slot at each end of that direction
              untouched by the exchange, which is what turns the value stored there into a Dirichlet
              boundary condition.

    There is one topology per process: the library keeps the communicators in globals
    (comm_1d_x/y/z, mpi_world_cart, comm_boundary).  Creating a second one before destroying the
    first is refused.  Note also that the library's mpi_topology_destroy() frees only
    mpi_world_cart, so a create/destroy loop leaks the three 1D sub-communicators -- build the
    topology once and keep it for the life of the process.
    """

    def __init__(self, nproc, periodic=(False, False, False), comm=MPI.COMM_WORLD):
        self.nproc = tuple(int(v) for v in nproc)
        self.periodic = tuple(bool(v) for v in periodic)
        self.comm = comm

        if len(self.nproc) != 3 or len(self.periodic) != 3:
            raise ValueError("nproc and periodic must each have three entries")

        status = _lib.gmg_topology_create(self.nproc[0], self.nproc[1], self.nproc[2],
                                          int(self.periodic[0]), int(self.periodic[1]),
                                          int(self.periodic[2]))
        if status == -1:
            raise RuntimeError("MPI is not initialised")
        if status == -2:
            raise RuntimeError("a Topology already exists in this process; destroy it first")
        if status == -3:
            raise ValueError(f"nproc product {self.nproc[0] * self.nproc[1] * self.nproc[2]} "
                             f"!= comm size {comm.size}")
        if status != 0:
            raise RuntimeError(f"gmg_topology_create failed (status {status})")
        self._live = True

        rank, size = ctypes.c_int(0), ctypes.c_int(0)
        _lib.gmg_world_info(ctypes.byref(rank), ctypes.byref(size))
        self.rank, self.size = rank.value, size.value
        self._grids = 0                 # live Grid objects built on this topology

    @property
    def ranks(self):
        """(rx, ry, rz): this process's position in the cartesian grid."""
        rx, ry, rz = (ctypes.c_int(0) for _ in range(3))
        _lib.gmg_topology_ranks(ctypes.byref(rx), ctypes.byref(ry), ctypes.byref(rz))
        return rx.value, ry.value, rz.value

    def destroy(self):
        # The halo exchange of every Grid runs on comm_1d_x/y/z, which this call
        # invalidates.  Refusing here turns a use-after-free deep inside the solver into a
        # message at the point where the ordering actually went wrong.
        if self._grids:
            raise RuntimeError(f"{self._grids} Grid(s) still alive on this topology; "
                               f"destroy them first")
        if getattr(self, "_live", False):
            _lib.gmg_topology_destroy()
            self._live = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
class Grid:
    """The global domain, this rank's block of it, and the device fields on that block.

    n        (nx, ny, nz) global cell counts
    origin   (ox, oy, oz) lower corner of the domain
    length   (lx, ly, lz) domain extent, so the domain is [ox, ox+lx] x ...
    stretch  (ax, ay, az) mesh stretching ratio per direction; 1.0 gives a uniform grid

    Attributes nx/ny/nz are this rank's interior cell counts, ista..kend its global index range
    (1-based, inclusive), and x/b/r the device fields as CuPy views of shape (nx+2, ny+2, nz+2).
    """

    def __init__(self, topology, n, origin, length, stretch=(1.0, 1.0, 1.0)):
        if not getattr(topology, "_live", False):
            raise RuntimeError("topology has been destroyed")
        self.topology = topology
        self.n = tuple(int(v) for v in n)
        self.origin = tuple(float(v) for v in origin)
        self.length = tuple(float(v) for v in length)
        self.stretch = tuple(float(v) for v in stretch)

        self._domain = _lib.gmg_domain_create(self.n[0], self.n[1], self.n[2],
                                              self.origin[0], self.origin[1], self.origin[2],
                                              self.length[0], self.length[1], self.length[2],
                                              self.stretch[0], self.stretch[1], self.stretch[2])
        if not self._domain:
            raise RuntimeError("gmg_domain_create failed")

        self._subdomain = _lib.gmg_subdomain_create(self._domain)
        if not self._subdomain:
            _lib.gmg_domain_destroy(self._domain)
            self._domain = None
            raise RuntimeError("gmg_subdomain_create failed")

        nx, ny, nz = (ctypes.c_int(0) for _ in range(3))
        _lib.gmg_subdomain_dims(self._subdomain, ctypes.byref(nx), ctypes.byref(ny),
                                ctypes.byref(nz))
        self.nx, self.ny, self.nz = nx.value, ny.value, nz.value

        rng = [ctypes.c_int(0) for _ in range(6)]
        _lib.gmg_subdomain_range(self._subdomain, *(ctypes.byref(v) for v in rng))
        (self.ista, self.iend, self.jsta, self.jend, self.ksta, self.kend) = (v.value for v in rng)

        self.x = self._field(_FIELD_X)
        self.b = self._field(_FIELD_B)
        self.r = self._field(_FIELD_R)
        self.xg = self._coord(0, self.nx + 2)
        self.yg = self._coord(1, self.ny + 2)
        self.zg = self._coord(2, self.nz + 2)

        self._plan = None               # live MultigridPlan built on this grid
        topology._grids += 1

    # -- internals ---------------------------------------------------------
    def _field(self, which):
        ptr = _lib.gmg_subdomain_field(self._subdomain, which)
        if not ptr:
            raise RuntimeError(f"gmg_subdomain_field({which}) returned NULL")
        return _dev_view(ptr, self.shape, self)

    def _coord(self, which, length):
        ptr = _lib.gmg_subdomain_coord(self._subdomain, which)
        if not ptr:
            raise RuntimeError(f"gmg_subdomain_coord({which}) returned NULL")
        return _dev_view(ptr, (length,), self)

    # -- shape helpers -----------------------------------------------------
    @property
    def shape(self):
        """Full field shape, halo layer included."""
        return (self.nx + 2, self.ny + 2, self.nz + 2)

    @property
    def interior(self):
        """Index tuple selecting the interior cells: grid.b[grid.interior] = rhs."""
        return (slice(1, self.nx + 1), slice(1, self.ny + 1), slice(1, self.nz + 1))

    @property
    def is_boundary(self):
        """((x0, x1), (y0, y1), (z0, z1)): whether each face is a domain boundary, not a rank interface."""
        flags = [ctypes.c_int(0) for _ in range(6)]
        _lib.gmg_subdomain_boundary(self._subdomain, *(ctypes.byref(v) for v in flags))
        v = [bool(f.value) for f in flags]
        return ((v[0], v[1]), (v[2], v[3]), (v[4], v[5]))

    def coords(self):
        """(xg, yg, zg) reshaped to broadcast against a full (nx+2, ny+2, nz+2) field."""
        return (self.xg[:, None, None], self.yg[None, :, None], self.zg[None, None, :])

    # -- lifetime ----------------------------------------------------------
    def destroy(self):
        """Release the device fields.  Destroy any MultigridPlan built on this grid first."""
        if getattr(self, "_plan", None) is not None and self._plan._live:
            raise RuntimeError("a MultigridPlan is still alive on this grid; destroy it first "
                               "(its solve() would then read freed device memory)")
        if getattr(self, "_subdomain", None) is None and getattr(self, "_domain", None) is None:
            return                                          # already destroyed
        # Drop the views before the memory goes away, so a stray reference cannot outlive it.
        self.x = self.b = self.r = None
        self.xg = self.yg = self.zg = None
        if getattr(self, "_subdomain", None):
            _lib.gmg_subdomain_destroy(self._subdomain)
            self._subdomain = None
        if getattr(self, "_domain", None):
            _lib.gmg_domain_destroy(self._domain)
            self._domain = None
        self.topology._grids -= 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


# ---------------------------------------------------------------------------
# Multigrid
# ---------------------------------------------------------------------------
class MultigridPlan:
    """The Poisson operator and the coarse-level hierarchy.

    nlevel       number of grid levels including the finest
    ncycle       V-cycles per solve() call
    aggr_method  0 = no aggregation, 1 = single, 2 = adaptive.  Aggregation gathers the coarse
                 levels onto fewer processes, so that coarsening can continue past the point where
                 a level would have fewer cells than ranks.
    aggr_level   level at which that gathering happens; must be 0 when aggr_method is 0

    multigrid_gpu.cu keeps the hierarchy in file-scope statics, so there is one plan per process and
    a second create is refused.
    """

    def __init__(self, grid, nlevel, ncycle, aggr_method=0, aggr_level=0):
        if getattr(grid, "_subdomain", None) is None:
            raise RuntimeError("grid has been destroyed")
        if aggr_method not in (0, 1, 2):
            raise ValueError(f"aggr_method must be 0, 1 or 2, got {aggr_method}")
        if aggr_method == 0 and aggr_level != 0:
            raise ValueError("aggr_level must be 0 when aggr_method is 0")
        self.grid = grid
        self.nlevel = int(nlevel)
        self.ncycle = int(ncycle)
        self.aggr_method = int(aggr_method)
        self.aggr_level = int(aggr_level)

        self._matrix = _lib.gmg_matrix_create(grid._subdomain)
        if not self._matrix:
            raise RuntimeError("gmg_matrix_create failed")

        status = _lib.gmg_multigrid_create(grid._subdomain, self.nlevel, self.ncycle,
                                           self.aggr_method, self.aggr_level)
        if status != 0:
            _lib.gmg_matrix_destroy(self._matrix)
            self._matrix = None
            raise RuntimeError("gmg_multigrid_create failed "
                               "(a MultigridPlan already exists in this process?)")
        self._live = True
        grid._plan = self

    def solve(self, maxiter, tol, omega, sol=None, rhs=None):
        """Run the V-cycles.  sol is updated in place.

        sol and rhs default to the grid's own x and b.  Passing others is allowed -- the C solver
        takes plain pointers -- but they must be float64, C-contiguous and of shape grid.shape.

        Asynchronous: the call returns once the kernels are queued.  Call sync() before reading the
        result back to the host or before timing.
        """
        if not self._live:
            raise RuntimeError("plan already destroyed")
        sol = self.grid.x if sol is None else sol
        rhs = self.grid.b if rhs is None else rhs
        status = _lib.gmg_multigrid_solve(_devptr(sol, self.grid.shape, "sol"), self._matrix,
                                          _devptr(rhs, self.grid.shape, "rhs"),
                                          self.grid._subdomain,
                                          int(maxiter), float(tol), float(omega))
        if status != 0:
            raise RuntimeError(f"gmg_multigrid_solve failed (status {status})")

    def sync(self):
        """Block until the GPU is done; needed before timing or reading results back."""
        status = _lib.gmg_device_sync()
        if status != 0:
            raise RuntimeError(f"cudaDeviceSynchronize failed (status {status})")

    def destroy(self):
        if getattr(self, "_live", False):
            _lib.gmg_multigrid_destroy()
            self._live = False
        if getattr(self, "_matrix", None):
            _lib.gmg_matrix_destroy(self._matrix)
            self._matrix = None
        if getattr(self.grid, "_plan", None) is self:
            self.grid._plan = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


def _devptr(a, shape, name):
    """Validate a CuPy array against the grid and return its device address."""
    if a.dtype != cp.float64:
        raise TypeError(f"{name} must be float64, got {a.dtype}")
    if a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} != expected {shape}")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous; the solver indexes "
                         f"i*(ny+2)*(nz+2) + j*(nz+2) + k")
    return ctypes.c_void_p(a.data.ptr)
