#!/usr/bin/env bash
# =============================================================================
# PaScaL_POISSON_GMG - Python(GPU) 환경 설정
#
#   사용법:  source 02_python/setup_env.sh
#   ★ 반드시 'source' (module load / conda activate 가 현재 셸에 남아야 함)
#
#   하는 일 (모두 idempotent — 여러 번 source 해도 안전):
#     1) NVHPC 25.11 (CUDA 12.9) + 번들 HPC-X OpenMPI 모듈 로드
#     2) conda env 없으면 environment.yml 로 생성 (python+pip 만 conda, 나머지는 pip wheel)
#     3) conda env 활성화
#     4) mpi4py 없으면 '모듈 MPI' 에 맞춰 소스 빌드 (conda 로 깔지 않음)
#     5) 검증 출력
#
#   왜 00_C 가 쓰는 gcc + mpi/openmpi-4.1.8 조합이 아니라 NVHPC 인가:
#   nvcc 와 MPI 를 한 모듈에서 함께 얻을 수 있어 헤더·라이브러리 경로를 손으로
#   맞출 필요가 없고, 무엇보다 lib/libgmg_capi.so 와 mpi4py 가 같은 libmpi 를
#   쓰게 만들기 쉽다.  솔버는 파이썬에서 커뮤니케이터를 받지 않고
#   MPI_COMM_WORLD 를 직접 부르므로, 둘이 갈라지면 솔버의 집합통신이 파이썬이
#   초기화하지 않은 world 위에서 돌게 된다.
#
#   conda 는 홈이 아니라 /scratch 를 쓴다. 홈은 용량보다 파일 개수 쿼터가 먼저
#   걸리고 conda env 하나가 파일 2만 개 수준이라 홈에는 들어가지 않는다.
#   패키지 캐시·env·pip 캐시·CuPy JIT 캐시를 모두 /scratch 로 돌린다.
#
#   조정 (환경변수로 덮어쓰기 가능):
#     ENV_NAME=... CONDA_SH=... GMG_CONDA_ROOT=... source setup_env.sh
#
#   PaScaL_POISSON_FFT 의 pascal-poisson-gpu env 와 의존성이 같으므로
#   그대로 재사용할 수 있다:
#     ENV_NAME=pascal-poisson-gpu source setup_env.sh
# =============================================================================

# conda env 생성이 실패했을 때의 탈출구. 1,2 번은 이 사이트에서 실제로 확인했다.
# 멈춘 것과 그냥 느린 것을 구분하려면 conda 프로세스의 CPU 사용량을 보면 된다:
#   ps -o etime,%cpu -p $(pgrep -f 'conda-env create')
# 몇 분째 %cpu 가 0 에 가까우면 repodata 를 받다가 죽은 것이고 더 기다려도 소용없다.
_gmg_env_hint() {
    local env_prefix="$1"
    cat >&2 <<EOF
       빠져나갈 길:
       1) 네트워크 없이 만든다 — conda-forge repodata 를 건드리지 않으므로 이 사이트에서
          가장 확실하다. python/pip 이 캐시에 있어야 하고, pip wheel 은 PyPI 에서 받는다:
            conda env create --offline -f environment.yml -p ${env_prefix}
       2) PaScaL_POISSON_FFT 의 env 를 그대로 재사용 — 의존성이 동일하다:
            ENV_NAME=pascal-poisson-gpu source setup_env.sh
       3) 정말 느린 것뿐이라면(위 %cpu 확인) 더 기다린다:
            GMG_CONDA_TIMEOUT=3600 source setup_env.sh
EOF
}

_gmg_setup() {
    local ENV_NAME="${ENV_NAME:-pascal-gmg-gpu}"
    local CONDA_ROOT="${GMG_CONDA_ROOT:-/scratch/${USER}/conda}"
    local here yml cbase t sh env_prefix

    here="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
    yml="${here}/environment.yml"
    [ -f "$yml" ] || { echo "[ERR] environment.yml 없음: $yml" >&2; return 1; }

    # ---- 1) 툴체인 ----
    if ! command -v module >/dev/null 2>&1; then
        source /apps/Modules/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null
    fi
    command -v module >/dev/null 2>&1 || {
        echo "[ERR] 'module' 명령을 찾을 수 없습니다." >&2; return 1; }

    echo "==> 툴체인 로드: nvhpc/25.11_cuda12"
    module purge >/dev/null 2>&1
    module load nvhpc/25.11_cuda12 >/dev/null 2>&1 || {
        echo "[ERR] nvhpc/25.11_cuda12 로드 실패" >&2; return 1; }

    : "${NVHPC_ROOT:?nvhpc 모듈이 NVHPC_ROOT 를 설정하지 않았습니다}"
    export CUDA_VERSION=12.9
    export CUDA_HOME="${NVHPC_ROOT}/cuda/${CUDA_VERSION}"
    export CUDA_PATH="${CUDA_HOME}"
    export NVHPC_MPI_ROOT="${HPCX_MPI_DIR}"

    # nvcc 를 CUDA 12.9 쪽으로 고정한다. 모듈이 PATH 에 넣는 compilers/bin/nvcc 는
    # CUDA 13.0 이고, environment.yml 은 CuPy 와 NVRTC 를 CUDA 12 로 묶어 두었다.
    # 둘이 갈라지면 .so 와 CuPy 가 서로 다른 CUDA 위에서 돌게 되므로 여기서 못 박는다.
    # command -v 로 비교하므로 두 번 source 해도 PATH 가 늘어나지 않는다.
    if [ -x "${CUDA_HOME}/bin/nvcc" ] && [ "$(command -v nvcc)" != "${CUDA_HOME}/bin/nvcc" ]; then
        export PATH="${CUDA_HOME}/bin:${PATH}"
        hash -r
    fi

    echo "==> 툴체인:"
    for t in nvcc mpicxx mpicc; do
        if command -v "$t" >/dev/null 2>&1; then printf '    %-8s %s\n' "$t" "$(command -v "$t")"
        else echo "    [WARN] $t 가 PATH 에 없음"; fi
    done
    if command -v nvcc >/dev/null 2>&1; then
        printf '    %-8s %s\n' "CUDA" "$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    fi

    # ---- 2,3) conda env ----
    if ! command -v conda >/dev/null 2>&1; then
        for sh in "${CONDA_SH:-}" \
                  /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh \
                  /apps/applications/Miniconda/24.11.1/etc/profile.d/conda.sh; do
            [ -n "$sh" ] && [ -f "$sh" ] && { source "$sh"; break; }
        done
    fi
    command -v conda >/dev/null 2>&1 || { echo "[ERR] conda 없음 (CONDA_SH 로 지정하세요)" >&2; return 1; }
    cbase="$(conda info --base 2>/dev/null)"
    [ -f "${cbase}/etc/profile.d/conda.sh" ] && source "${cbase}/etc/profile.d/conda.sh"

    export CONDA_PKGS_DIRS="${CONDA_ROOT}/pkgs"
    export CONDA_ENVS_DIRS="${CONDA_ROOT}/envs"
    export PIP_CACHE_DIR="${CONDA_ROOT}/pip-cache"
    # CuPy 는 실행 중 JIT 커널을 ~/.cupy/kernel_cache 에 쓴다. 홈 파일 쿼터가
    # 차면 'Disk quota exceeded' 로 죽으므로 캐시도 /scratch 로 돌린다.
    export CUPY_CACHE_DIR="${CONDA_ROOT}/cupy-cache"
    mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$PIP_CACHE_DIR" "$CUPY_CACHE_DIR" \
        || { echo "[ERR] conda 작업 디렉터리 생성 실패: $CONDA_ROOT" >&2; return 1; }
    env_prefix="${CONDA_ENVS_DIRS}/${ENV_NAME}"

    if [ -d "${env_prefix}/conda-meta" ]; then
        echo "==> conda env 존재 → 생성 건너뜀: ${env_prefix}"
    else
        echo "==> conda env 생성: ${env_prefix}"
        echo "    (python+pip 만 conda, 나머지는 pip wheel — 수 분 걸릴 수 있음)"
        # timeout 을 씌우는 이유: conda 23.3.1 은 conda-forge CDN 이 HTTPS 연결을
        # 끊으면 'Collecting package metadata' 에서 futex 대기로 영영 멈춘다
        # (소켓은 CLOSE-WAIT, CPU 는 0%). 무한정 매달리는 대신 실패로 끝내고
        # 빠져나갈 길을 알려준다.
        # -k 30: 그 futex 대기 상태에서는 SIGTERM 을 안 받으므로 30 초 뒤 SIGKILL 을 보낸다.
        timeout -k 30 "${GMG_CONDA_TIMEOUT:-1800}" conda env create -f "$yml" -p "$env_prefix"
        case $? in
            0) ;;
            124) echo "[ERR] conda env 생성이 ${GMG_CONDA_TIMEOUT:-1800}s 안에 끝나지 않았습니다." >&2
                 _gmg_env_hint "$env_prefix"; return 1 ;;
            *)   echo "[ERR] conda env 생성 실패" >&2
                 _gmg_env_hint "$env_prefix"; return 1 ;;
        esac
    fi

    echo "==> conda activate ${env_prefix}"
    conda activate "$env_prefix" || { echo "[ERR] conda activate 실패" >&2; return 1; }

    # ---- 4) mpi4py : 모듈 MPI 에 맞춰 소스 빌드 ----
    if python -c "import mpi4py" 2>/dev/null; then
        echo "==> mpi4py 설치됨 → 건너뜀"
        echo "    (MPI 모듈을 바꿨다면: pip uninstall -y mpi4py 후 이 스크립트를 다시 source)"
    else
        command -v mpicc >/dev/null 2>&1 || { echo "[ERR] mpicc 없음 — mpi4py 빌드 불가" >&2; return 1; }
        # HPC-X mpicc 의 기본 백엔드는 nvc 라서 gcc 전용 플래그(-fwrapv 등)를 거부한다.
        # OMPI_CC=gcc 로 백엔드만 gcc 로 바꾸고, libmpi 는 그대로 HPC-X 를 링크한다.
        #
        # --no-cache-dir 가 핵심이다. --no-binary 는 '내려받은 휠을 쓰지 말라'는 뜻일 뿐이라,
        # pip 이 예전에 직접 빌드해 캐시에 넣어둔 휠은 그대로 재사용한다. 그 휠이 다른 MPI 에
        # 링크돼 있으면 여기서 소스 빌드를 하는 이유가 통째로 사라진다 — 파이썬이 초기화한
        # MPI_COMM_WORLD 와 libgmg_capi.so 가 부르는 MPI_COMM_WORLD 가 달라진다.
        echo "==> mpi4py 소스 빌드 (mpicc 백엔드=gcc, libmpi=HPC-X, 캐시 무시)"
        OMPI_CC="${OMPI_CC:-gcc}" MPICC="$(command -v mpicc)" \
            pip install --no-cache-dir --no-binary=mpi4py mpi4py \
            || { echo "[ERR] mpi4py 빌드 실패" >&2; return 1; }
    fi

    # ---- 5) 검증 ----
    echo "==> 검증:"
    nvidia-smi -L 2>/dev/null | sed 's/^/    /'
    GMG_PY_DIR="$here" python - <<'PY'
import os, sys

def show(n, f):
    try: print("    %-8s %s" % (n, f()))
    except Exception as e: print("    %-8s ERR: %s" % (n, e))

print("    python   %s" % sys.version.split()[0])
show("numpy",  lambda: __import__("numpy").__version__)
show("cupy",   lambda: __import__("cupy").__version__)

def _mpi():
    # 솔버가 MPI_COMM_WORLD 를 직접 부르므로 파이썬과 .so 가 같은 libmpi 여야 한다.
    from mpi4py import MPI
    return MPI.Get_library_version().strip().splitlines()[0]
show("mpi4py", _mpi)

so = os.path.join(os.environ["GMG_PY_DIR"], "lib", "libgmg_capi.so")
if not os.path.exists(so):
    print("    capi.so  아직 없음 -> 'make' 로 빌드하세요")
else:
    import ctypes
    lib = ctypes.CDLL(so)
    lib.gmg_device_count.restype = ctypes.c_int
    print("    capi.so  로드 OK, gmg_device_count()=%d" % lib.gmg_device_count())
    # .so 와 파이썬이 같은 libmpi 를 잡았는지 /proc 의 매핑으로 확인한다.
    # 두 경로가 갈라지면 솔버의 MPI_Cart_create 가 파이썬이 초기화하지 않은
    # world 위에서 돌기 때문에, 여기서 잡아주는 편이 낫다.
    try:
        maps = open("/proc/self/maps").read()
        libs = sorted({l.split()[-1] for l in maps.splitlines()
                       if "libmpi.so" in l.split()[-1]})
        print("    libmpi   %s" % (libs[0] if len(libs) == 1 else
                                   "충돌! " + " / ".join(libs)))
    except OSError:
        pass
PY

    echo
    echo "==> 환경 준비 완료."
    echo "    - 매 세션:  source ${here}/setup_env.sh"
    echo "    - 빌드:     make                       (lib/libgmg_capi.so)"
    echo "    - 실행:     mpirun -n 1 python example/poisson3d_gmg.py PARA_INPUT.inp"
    return 0
}

# 실행(exec) 시 경고 후 진행, source 시 그대로 현재 셸에 반영
if [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "[!] 'source 02_python/setup_env.sh' 로 실행하세요 (module/conda 활성화가 셸에 남도록)."
    _gmg_setup; exit $?
else
    _gmg_setup
fi
