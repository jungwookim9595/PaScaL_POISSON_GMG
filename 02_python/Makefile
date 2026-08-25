# =============================================================================
# PaScaL_POISSON_GMG - Python C-API wrapper library
#
#   Target: lib/libgmg_capi.so
#           Shared library bundling the CUDA multigrid solver of 00_C/src/gpu
#           with the shim in src/gmg_capi.cu, called from Python via ctypes
#           with CuPy device arrays.
#
#   The 00_C tree is reused as-is and never modified.  Its own build produces an
#   executable, not a relocatable library, so the sources are recompiled here
#   with -Xcompiler -fPIC.  Everything from src/gpu is included except
#   poisson_gpu.cu, which holds main() and the example problem -- that role is
#   taken over by example/poisson3d_gmg.py.  Dropping it costs nothing: the
#   lv_gdm_coarsest_* globals it defines are shadowed by file-scope statics of
#   the same name in multigrid_gpu.cu, so no other object refers to them.
#   timer.c is left out here as well, matching 00_C/src/gpu/Makefile.
#
#   Usage:
#     source setup_env.sh          # toolchain + conda env; do this first
#     make                         # build lib/libgmg_capi.so
#     make GPU_CC=80               # build for A100 only (faster to compile)
#     make clean
# =============================================================================
GMG_HOME ?= $(CURDIR)/../00_C

SRCDIR  = $(GMG_HOME)/src/gpu
INCDIR  = $(GMG_HOME)/include
OBJDIR  = ./obj
LIBDIR  = ./lib
TARGET  = $(LIBDIR)/libgmg_capi.so

NVCC ?= nvcc

# Target GPU architectures.  The architecture of the GPU that runs the code must
# be in this list; Neuron has both V100 (70) and A100 (80) partitions, and a
# binary built for only one of them falls back to PTX JIT on the other.
GPU_CC   ?= 70 80
ARCHFLAGS = $(foreach cc,$(GPU_CC),-gencode arch=compute_$(cc),code=sm_$(cc))

# -ccbin mpicxx makes nvcc use the MPI compiler wrapper as its host compiler, so
# mpi.h and libmpi resolve without spelling out the HPC-X paths.  It also pins
# the .so to the same MPI that mpi4py is built against in setup_env.sh; if the
# two diverge, the solver's MPI_COMM_WORLD is not the one Python initialised.
HOSTBIN ?= mpicxx

OPT ?= -O3

# Warnings from the reference sources that are noise here, not defects:
# 177/550 unused-and-set-but-unused variables, 780 unreachable code.
SUPPRESS = -Wno-deprecated-gpu-targets \
           -diag-suppress=177 -diag-suppress=550 -diag-suppress=780 \
           -Xcompiler=--diag_suppress=declared_but_not_referenced \
           -Xcompiler=--diag_suppress=set_but_not_used \
           -Xcompiler=--diag_suppress=hidden_by_old_for_init

NVCCFLAGS = $(OPT) $(ARCHFLAGS) -ccbin $(HOSTBIN) -Xcompiler -fPIC \
            $(SUPPRESS) -I$(INCDIR)

# Solver sources from 00_C (poisson_gpu.cu excluded: it is the example driver).
LIB_CU = multigrid_gpu.cu geometry_gpu.cu matrix_gpu.cu \
         rbgs_poisson_matrix_gpu.cu poisson_matrix_operator_gpu.cu
LIB_C  = mpi_topology.c para_range.c

# Every object includes several of these, and they are few; depending on all of
# them costs a full rebuild after a header edit and nothing otherwise.
HEADERS = $(wildcard $(INCDIR)/*.h)

OBJS = $(addprefix $(OBJDIR)/,$(LIB_CU:.cu=.o)) \
       $(addprefix $(OBJDIR)/,$(LIB_C:.c=.o)) \
       $(OBJDIR)/gmg_capi.o

all: $(TARGET)

$(TARGET): $(OBJS) | $(LIBDIR)
	$(NVCC) $(NVCCFLAGS) -shared $(OBJS) -o $@

# Solver sources (00_C/src/gpu).  The .c files are handed to nvcc exactly as
# 00_C/src/gpu/Makefile does, so their symbols keep the linkage the solver's
# extern "C" headers declare.
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADERS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Shim source (02_python/src)
$(OBJDIR)/%.o: src/%.cu $(HEADERS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR) $(LIBDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(LIBDIR)

.PHONY: all clean
