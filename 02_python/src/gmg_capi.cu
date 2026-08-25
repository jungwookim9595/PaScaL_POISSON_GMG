/*======================================================================================================================
 * @file        gmg_capi.cu
 * @brief       C-callable shim around the CUDA geometric-multigrid Poisson solver of PaScaL_POISSON_GMG.
 * @details     The solver in 00_C/src/gpu is already C, so unlike the Fortran-based wrappers of
 *              PaScaL_TDMA and PaScaL_POISSON_FFT this file performs no ABI translation.  What it
 *              does provide is the part a ctypes caller cannot express safely:
 *
 *              - Opaque handles.  domain, subdomain and matrix_poisson are plain structs, but
 *                subdomain alone carries 13 MPI_Datatype fields.  Mirroring that layout in
 *                ctypes.Structure would hard-code field offsets and padding chosen by nvcc and by
 *                the MPI implementation, and it would break silently when either changes.  The
 *                structs are therefore allocated here, handed out as void*, and read back through
 *                the accessors below.
 *              - Call-order guarantees.  The reference driver (00_C/src/gpu/poisson_gpu.cu) pairs
 *                geometry_subdomain_create_gpu with geometry_subdomain_ddt_create, and multigrid
 *                teardown must precede subdomain teardown.  Those pairings are enforced here
 *                instead of being restated in Python.
 *              - The library's global state.  myrank, nprocs, np_dim[] and period[] live in
 *                mpi_topology.c and must be set before mpi_topology_create().
 *
 *              What deliberately does NOT cross this boundary is the field data.  Every device
 *              buffer is allocated by the library (geometry_subdomain_create_gpu cudaMallocs x, b
 *              and r); Python receives the raw device addresses and wraps them as CuPy views, so
 *              the data never leaves the GPU and is never copied.
 *
 *              MPI is not passed in.  The solver calls MPI_COMM_WORLD directly and builds its own
 *              cartesian topology, so Python only has to make sure MPI is initialised first (by
 *              importing mpi4py) and that this library was linked against the same MPI.
 *
 * @author
 *              - Jungwoo Kim (yasandy@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
 *
 * @date        August 2026
 * @version     1.0
 * @par         License
 *              This project is released under the terms of the MIT License (see LICENSE file).
 *====================================================================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <cuda_runtime.h>

#include "geometry.h"
#include "matrix.h"
#include "multigrid.h"
#include "mpi_topology.h"

/*----------------------------------------------------------------------------------------------------------------
 * Field and coordinate selectors used by gmg_subdomain_field / gmg_subdomain_coord.
 *--------------------------------------------------------------------------------------------------------------*/
#define GMG_FIELD_X   0
#define GMG_FIELD_B   1
#define GMG_FIELD_R   2

#define GMG_COORD_X   0
#define GMG_COORD_Y   1
#define GMG_COORD_Z   2

/*----------------------------------------------------------------------------------------------------------------
 * Process-wide state.
 *
 * mpi_topology_create() writes the library globals comm_1d_x/y/z, mpi_world_cart and comm_boundary,
 * so there is exactly one topology per process.  mpi_topology_destroy() frees mpi_world_cart only --
 * the three 1D sub-communicators and comm_boundary are leaked by the library -- hence the guard:
 * a create/destroy loop would exhaust the MPI communicator table.  Likewise multigrid_gpu.cu keeps
 * its level hierarchy in file-scope statics, so only one multigrid hierarchy can exist at a time.
 *--------------------------------------------------------------------------------------------------------------*/
static int topology_live  = 0;
static int multigrid_live = 0;

extern "C" {

/*======================================================================================================================
 * MPI and topology
 *====================================================================================================================*/

/**
 * @brief   Whether MPI_Init has already been called (by mpi4py, normally).
 */
int gmg_mpi_initialized(void)
{
    int flag = 0;
    MPI_Initialized(&flag);
    return flag;
}

/**
 * @brief   Publish MPI_COMM_WORLD's rank and size into the library globals myrank and nprocs.
 * @details The reference driver does this in main() right after MPI_Init.  Several library
 *          routines print or branch on myrank, and multigrid_solve_vcycle_gpu falls back to it
 *          when MPI is not initialised, so it must be set before anything else runs.
 * @return  0 on success, -1 if MPI is not initialised.
 */
int gmg_world_info(int *rank, int *size)
{
    if (!gmg_mpi_initialized()) return -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (rank) *rank = myrank;
    if (size) *size = nprocs;
    return 0;
}

/**
 * @brief   Build the 3D cartesian topology and the boundary communicator.
 * @param   npx,npy,npz     Process counts per direction; the product must equal MPI_COMM_WORLD's size.
 * @param   px,py,pz        Periodicity per direction, 0 = non-periodic.  A non-periodic direction
 *                          leaves the halo slot at index 0 / n+1 untouched by the halo exchange,
 *                          which is what imposes the Dirichlet boundary value stored there.
 * @return  0 on success, -1 if MPI is not initialised, -2 if a topology already exists,
 *          -3 if npx*npy*npz does not match the world size.
 */
int gmg_topology_create(int npx, int npy, int npz, int px, int py, int pz)
{
    int size = 0;

    if (!gmg_mpi_initialized()) return -1;
    if (topology_live)          return -2;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (npx * npy * npz != size) return -3;

    np_dim[0] = npx; np_dim[1] = npy; np_dim[2] = npz;
    period[0] = px;  period[1] = py;  period[2] = pz;

    mpi_topology_create();
    mpi_boundary_create();
    topology_live = 1;
    return 0;
}

void gmg_topology_destroy(void)
{
    if (!topology_live) return;
    mpi_topology_destroy();
    topology_live = 0;
}

/**
 * @brief   Rank of this process within each of the three 1D communicators.
 * @details Reported so the example can label its output by cartesian position rather than by world
 *          rank; MPI_Cart_create is called with reorder = 0, but the two still differ whenever the
 *          process grid is not a plain z-split.
 */
void gmg_topology_ranks(int *rx, int *ry, int *rz)
{
    if (rx) *rx = comm_1d_x.myrank;
    if (ry) *ry = comm_1d_y.myrank;
    if (rz) *rz = comm_1d_z.myrank;
}

/*======================================================================================================================
 * Device
 *====================================================================================================================*/

int gmg_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

int gmg_set_device(int device)
{
    return (int)cudaSetDevice(device);
}

int gmg_get_device(void)
{
    int d = -1;
    if (cudaGetDevice(&d) != cudaSuccess) return -1;
    return d;
}

/**
 * @brief   Block until every queued kernel has finished.
 * @return  The cudaError_t as an int; 0 is cudaSuccess.
 */
int gmg_device_sync(void)
{
    return (int)cudaDeviceSynchronize();
}

/*======================================================================================================================
 * Global domain
 *====================================================================================================================*/

/**
 * @brief   Allocate and build the global grid.
 * @details Argument order follows the *definition* in 00_C/src/gpu/geometry_gpu.cu, which is
 *              (gdm, nx, ny, nz, ox, oy, oz, lx, ly, lz, ax, ay, az, period)
 *          The declaration in 00_C/include/geometry.h names the same six doubles in the opposite
 *          order (lx,ly,lz then ox,oy,oz).  Every parameter is a double, so the mismatch is in the
 *          names only and the call below binds correctly -- the reference driver passes them this
 *          way too -- but anyone reading the header alone would swap origin and length.
 *
 * @param   ax,ay,az    Mesh stretching ratio per direction; 1.0 gives a uniform grid.
 * @return  Opaque handle, or NULL on allocation failure.
 */
void *gmg_domain_create(int nx, int ny, int nz,
                        double ox, double oy, double oz,
                        double lx, double ly, double lz,
                        double ax, double ay, double az)
{
    domain *gdm = (domain *)calloc(1, sizeof(domain));
    if (!gdm) return NULL;

    geometry_domain_create_gpu(gdm, nx, ny, nz, ox, oy, oz, lx, ly, lz, ax, ay, az, period);
    return (void *)gdm;
}

void gmg_domain_destroy(void *handle)
{
    domain *gdm = (domain *)handle;
    if (!gdm) return;
    geometry_domain_destroy_gpu(gdm);
    free(gdm);
}

/*======================================================================================================================
 * Subdomain
 *====================================================================================================================*/

/**
 * @brief   Carve this rank's block out of the global domain and commit its MPI derived datatypes.
 * @details geometry_subdomain_create_gpu allocates the three device fields x, b and r, each of
 *          shape (nx+2, ny+2, nz+2) in C order, and zeroes them.  The zero it leaves in the slots
 *          at index 0 and nx+1 of a non-periodic direction is the boundary value: those slots sit
 *          exactly on the domain face (xg[0] = ox), no kernel writes them, and the halo exchange
 *          skips them because the neighbour is MPI_PROC_NULL.
 *
 *          ddt_create is folded in because the reference driver always calls the two as a pair and
 *          multigrid_create_gpu needs the datatypes.
 *
 * @return  Opaque handle, or NULL on allocation failure.
 */
void *gmg_subdomain_create(void *domain_handle)
{
    domain *gdm = (domain *)domain_handle;
    subdomain *sdm;

    if (!gdm) return NULL;

    sdm = (subdomain *)calloc(1, sizeof(subdomain));
    if (!sdm) return NULL;

    geometry_subdomain_create_gpu(sdm, gdm);
    geometry_subdomain_ddt_create(sdm);
    return (void *)sdm;
}

void gmg_subdomain_destroy(void *handle)
{
    subdomain *sdm = (subdomain *)handle;
    if (!sdm) return;
    geometry_subdomain_ddt_destroy(sdm);
    geometry_subdomain_destroy_gpu(sdm);
    free(sdm);
}

/**
 * @brief   Interior cell counts of this rank's block (halo layer excluded).
 */
void gmg_subdomain_dims(void *handle, int *nx, int *ny, int *nz)
{
    subdomain *sdm = (subdomain *)handle;
    if (!sdm) return;
    if (nx) *nx = sdm->nx;
    if (ny) *ny = sdm->ny;
    if (nz) *nz = sdm->nz;
}

/**
 * @brief   Global index range owned by this rank, 1-based and inclusive on both ends.
 */
void gmg_subdomain_range(void *handle, int *ista, int *iend,
                                       int *jsta, int *jend,
                                       int *ksta, int *kend)
{
    subdomain *sdm = (subdomain *)handle;
    if (!sdm) return;
    if (ista) *ista = sdm->ista;  if (iend) *iend = sdm->iend;
    if (jsta) *jsta = sdm->jsta;  if (jend) *jend = sdm->jend;
    if (ksta) *ksta = sdm->ksta;  if (kend) *kend = sdm->kend;
}

/**
 * @brief   Whether each face of this block is a physical domain boundary rather than a rank interface.
 */
void gmg_subdomain_boundary(void *handle, int *x0, int *x1, int *y0, int *y1, int *z0, int *z1)
{
    subdomain *sdm = (subdomain *)handle;
    if (!sdm) return;
    if (x0) *x0 = sdm->is_x0_boundary;  if (x1) *x1 = sdm->is_x1_boundary;
    if (y0) *y0 = sdm->is_y0_boundary;  if (y1) *y1 = sdm->is_y1_boundary;
    if (z0) *z0 = sdm->is_z0_boundary;  if (z1) *z1 = sdm->is_z1_boundary;
}

/**
 * @brief   Device address of a field: 0 = x (solution), 1 = b (right-hand side), 2 = r (residual).
 * @details Shape (nx+2, ny+2, nz+2), C order, double precision.  The library owns the allocation;
 *          the caller must not free it.
 */
void *gmg_subdomain_field(void *handle, int which)
{
    subdomain *sdm = (subdomain *)handle;
    if (!sdm) return NULL;
    switch (which) {
        case GMG_FIELD_X: return (void *)sdm->x;
        case GMG_FIELD_B: return (void *)sdm->b;
        case GMG_FIELD_R: return (void *)sdm->r;
        default:          return NULL;
    }
}

/**
 * @brief   Device address of a coordinate vector: 0 = xg, 1 = yg, 2 = zg.
 * @details Length nx+2 (resp. ny+2, nz+2).  Index 1..n are cell centres; index 0 and n+1 hold the
 *          two domain faces themselves, not ghost centres, so the spacing between index 0 and 1 is
 *          half a cell.
 */
void *gmg_subdomain_coord(void *handle, int which)
{
    subdomain *sdm = (subdomain *)handle;
    if (!sdm) return NULL;
    switch (which) {
        case GMG_COORD_X: return (void *)sdm->xg;
        case GMG_COORD_Y: return (void *)sdm->yg;
        case GMG_COORD_Z: return (void *)sdm->zg;
        default:          return NULL;
    }
}

/*======================================================================================================================
 * Poisson matrix
 *====================================================================================================================*/

/**
 * @brief   Build the 7-point Poisson operator for this subdomain.
 * @return  Opaque handle, or NULL on allocation failure.
 */
void *gmg_matrix_create(void *subdomain_handle)
{
    subdomain *sdm = (subdomain *)subdomain_handle;
    matrix_poisson *a;

    if (!sdm) return NULL;

    a = (matrix_poisson *)calloc(1, sizeof(matrix_poisson));
    if (!a) return NULL;

    matrix_poisson_create_gpu(a, sdm);
    return (void *)a;
}

void gmg_matrix_destroy(void *handle)
{
    matrix_poisson *a = (matrix_poisson *)handle;
    if (!a) return;
    matrix_poisson_destroy_gpu(a);
    free(a);
}

/*======================================================================================================================
 * Multigrid
 *====================================================================================================================*/

/**
 * @brief   Build the level hierarchy.
 * @param   nlevel      Number of grid levels including the finest.
 * @param   ncycle      Number of V-cycles per solve.
 * @param   aggr_method 0 = no aggregation, 1 = single, 2 = adaptive.
 * @param   aggr_level  Level at which processes are aggregated; must be 0 when aggr_method is 0.
 * @return  0 on success, -1 if a hierarchy already exists.
 */
int gmg_multigrid_create(void *subdomain_handle, int nlevel, int ncycle,
                         int aggr_method, int aggr_level)
{
    subdomain *sdm = (subdomain *)subdomain_handle;
    if (!sdm) return -1;
    if (multigrid_live) return -1;

    multigrid_create_gpu(sdm, nlevel, ncycle, aggr_method, aggr_level);
    multigrid_live = 1;
    return 0;
}

/**
 * @brief   Run the V-cycle solver.
 * @param   sol         Device address of the solution field, updated in place.
 * @param   rhs         Device address of the right-hand side.
 * @details sol and rhs are passed as addresses rather than taken from the subdomain so that a
 *          caller can drive the solver with its own device buffers, exactly as the C signature
 *          allows.  Both must have shape (nx+2, ny+2, nz+2) in C order.
 * @return  0 on success, -1 if no hierarchy exists.
 */
int gmg_multigrid_solve(void *sol, void *matrix_handle, void *rhs, void *subdomain_handle,
                        int maxiteration, double tolerance, double omega_sor)
{
    if (!multigrid_live) return -1;
    if (!sol || !rhs || !matrix_handle || !subdomain_handle) return -1;

    multigrid_solve_vcycle_gpu((double *)sol,
                               (matrix_poisson *)matrix_handle,
                               (double *)rhs,
                               (subdomain *)subdomain_handle,
                               maxiteration, tolerance, omega_sor);
    return 0;
}

void gmg_multigrid_destroy(void)
{
    if (!multigrid_live) return;
    multigrid_destroy_gpu();
    multigrid_live = 0;
}

} /* extern "C" */
