"""Run a CUDA kernel without building an executable: NVRTC compiles the kernel source in memory to
a cubin for this GPU, and the NVIDIA driver (nvcuda) loads and launches it.

Why: Windows 11 Smart App Control blocks a freshly compiled, unsigned .exe or .dll ("An Application
Control policy has blocked this file"), so an nvcc-built test program cannot run there. NVRTC
(nvrtc64_*.dll) and the driver are NVIDIA-signed, and the kernel never becomes a file - the same path
PyTorch and CuPy use for runtime kernels. Only the standard library is needed (ctypes).

    python cuda_jit.py [--cuda-path DIR]      self-test: adds two arrays on the GPU and checks them

The Neuro CUDA checks use `Gpu` the same way: compile(source) -> kernel, then launch it on buffers.
"""
from __future__ import annotations

import argparse
import array
import ctypes
import glob
import os
import sys

CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class CudaJitError(RuntimeError):
    pass


def _cuda_bin_dirs(cuda_path: str | None) -> list[str]:
    roots = [cuda_path, os.environ.get("CUDA_PATH"),
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "Neuro", "cuda", "12.9")]
    dirs = []
    for root in roots:
        if root:
            for sub in ("bin", os.path.join("bin", "x64"), "lib64", os.path.join("targets", "x86_64-linux", "lib")):
                path = os.path.join(root, sub)
                if os.path.isdir(path) and path not in dirs:
                    dirs.append(path)
    return dirs


def _load_nvrtc(cuda_path: str | None) -> ctypes.CDLL:
    if os.name == "nt":
        for folder in _cuda_bin_dirs(cuda_path):
            # nvrtc64_120_0.dll needs nvrtc-builtins64_129.dll from the same folder.
            names = [n for n in glob.glob(os.path.join(folder, "nvrtc64_*.dll"))
                     if "builtins" not in n and ".alt." not in n]
            if names:
                os.add_dll_directory(folder)
                # NVRTC loads nvrtc-builtins itself with the plain search order, which ignores
                # add_dll_directory: the folder must be on PATH as well.
                if folder.lower() not in os.environ.get("PATH", "").lower().split(os.pathsep):
                    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
                return ctypes.CDLL(sorted(names)[-1])
        raise CudaJitError("NVRTC (nvrtc64_*.dll) was not found. Run platforms\\install-cuda.cmd first.")
    for folder in _cuda_bin_dirs(cuda_path):
        names = sorted(glob.glob(os.path.join(folder, "libnvrtc.so*")))
        if names:
            return ctypes.CDLL(names[-1])
    return ctypes.CDLL("libnvrtc.so")


def _load_driver() -> ctypes.CDLL:
    try:
        return ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    except OSError as exc:
        raise CudaJitError(f"The NVIDIA driver library could not be loaded ({exc}). Is an NVIDIA driver installed?") from None


class Nvrtc:
    def __init__(self, cuda_path: str | None = None):
        self.lib = _load_nvrtc(cuda_path)
        lib = self.lib
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
                                           ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        for name in ("nvrtcGetCUBINSize", "nvrtcGetProgramLogSize"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        for name in ("nvrtcGetCUBIN", "nvrtcGetProgramLog"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        major, minor = ctypes.c_int(), ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        self.version = (major.value, minor.value)

    def _check(self, code: int, what: str) -> None:
        if code != 0:
            raise CudaJitError(f"{what}: {self.lib.nvrtcGetErrorString(code).decode()}")

    def compile(self, source: str, arch: str, options: list[str] | None = None, name: str = "kernel.cu") -> bytes:
        """Source -> cubin (machine code) for one GPU architecture such as "sm_86". A cubin, not PTX:
        it does not depend on the driver's PTX version."""
        prog = ctypes.c_void_p()
        self._check(self.lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None),
                    "nvrtcCreateProgram")
        try:
            opts = [f"--gpu-architecture={arch}", "--std=c++17", *(options or [])]
            c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
            code = self.lib.nvrtcCompileProgram(prog, len(opts), c_opts)
            if code != 0:
                size = ctypes.c_size_t()
                self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                self.lib.nvrtcGetProgramLog(prog, log)
                raise CudaJitError("compile error:\n" + log.value.decode(errors="replace").strip())
            size = ctypes.c_size_t()
            self._check(self.lib.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "nvrtcGetCUBINSize")
            cubin = ctypes.create_string_buffer(size.value)
            self._check(self.lib.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
            return cubin.raw
        finally:
            self.lib.nvrtcDestroyProgram(ctypes.byref(prog))


class Gpu:
    """One GPU through the driver API: memory, modules and launches on its primary context."""

    def __init__(self, ordinal: int = 0, cuda_path: str | None = None):
        self.cu = _load_driver()
        cu = self.cu
        cu.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        cu.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cu.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cu.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        cu.cuDevicePrimaryCtxRelease_v2.argtypes = [ctypes.c_int]
        cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cu.cuModuleUnload.argtypes = [ctypes.c_void_p]
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._check(cu.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(cu.cuDeviceGet(ctypes.byref(self.device), ordinal), "cuDeviceGet")
        name = ctypes.create_string_buffer(256)
        self._check(cu.cuDeviceGetName(name, 256, self.device), "cuDeviceGetName")
        self.name = name.value.decode(errors="replace")
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(cu.cuDeviceGetAttribute(ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.device), "cc major")
        self._check(cu.cuDeviceGetAttribute(ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.device), "cc minor")
        self.arch = f"sm_{major.value}{minor.value}"
        self.ctx = ctypes.c_void_p()
        self._check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.device), "cuDevicePrimaryCtxRetain")
        self._check(cu.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        self.nvrtc = Nvrtc(cuda_path)
        self._modules: list[ctypes.c_void_p] = []
        self._buffers: list[int] = []

    def _check(self, code: int, what: str) -> None:
        if code != 0:
            text = ctypes.c_char_p()
            self.cu.cuGetErrorString(code, ctypes.byref(text))
            raise CudaJitError(f"{what}: {(text.value or b'error %d' % code).decode()}")

    def kernel(self, source: str, name: str, options: list[str] | None = None) -> ctypes.c_void_p:
        """Compile `source` for this GPU and return the extern "C" __global__ function `name`."""
        module = ctypes.c_void_p()
        self._check(self.cu.cuModuleLoadData(ctypes.byref(module), self.nvrtc.compile(source, self.arch, options)),
                    "cuModuleLoadData")
        self._modules.append(module)
        func = ctypes.c_void_p()
        self._check(self.cu.cuModuleGetFunction(ctypes.byref(func), module, name.encode()),
                    f"kernel {name} (declare it extern \"C\")")
        return func

    def upload(self, data: array.array) -> int:
        nbytes = data.itemsize * len(data)
        ptr = ctypes.c_uint64()
        self._check(self.cu.cuMemAlloc_v2(ctypes.byref(ptr), max(1, nbytes)), "cuMemAlloc")
        self._buffers.append(ptr.value)
        if nbytes:
            address, _ = data.buffer_info()
            self._check(self.cu.cuMemcpyHtoD_v2(ptr.value, address, nbytes), "cuMemcpyHtoD")
        return ptr.value

    def download(self, ptr: int, out: array.array) -> array.array:
        address, _ = out.buffer_info()
        self._check(self.cu.cuMemcpyDtoH_v2(address, ptr, out.itemsize * len(out)), "cuMemcpyDtoH")
        return out

    def launch(self, func: ctypes.c_void_p, grid: tuple[int, int, int], block: tuple[int, int, int],
               args: list[ctypes._SimpleCData], shared: int = 0) -> None:
        """Args as ctypes values: c_uint64(device pointer), c_int(n), c_float(x)..."""
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        self._check(self.cu.cuLaunchKernel(func, *grid, *block, shared, None, params, None), "cuLaunchKernel")
        self._check(self.cu.cuCtxSynchronize(), "kernel run")

    def close(self) -> None:
        for ptr in self._buffers:
            self.cu.cuMemFree_v2(ptr)
        for module in self._modules:
            self.cu.cuModuleUnload(module)
        self._buffers, self._modules = [], []
        self.cu.cuDevicePrimaryCtxRelease_v2(self.device)


SELF_TEST = r"""
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
"""


def self_test(cuda_path: str | None = None) -> str:
    gpu = Gpu(cuda_path=cuda_path)
    try:
        n = 1 << 20
        a = array.array("f", (float(i) for i in range(n)))
        b = array.array("f", [2.0]) * n
        func = gpu.kernel(SELF_TEST, "add")
        da, db = gpu.upload(a), gpu.upload(b)
        dc = gpu.upload(array.array("f", [0.0]) * n)
        gpu.launch(func, ((n + 255) // 256, 1, 1), (256, 1, 1),
                   [ctypes.c_uint64(da), ctypes.c_uint64(db), ctypes.c_uint64(dc), ctypes.c_int(n)])
        c = gpu.download(dc, array.array("f", [0.0]) * n)
        bad = next((i for i in range(n) if c[i] != a[i] + 2.0), None)
        if bad is not None:
            raise CudaJitError(f"wrong result at {bad}: {c[bad]} != {a[bad] + 2.0}")
        major, minor = gpu.nvrtc.version
        return f"CUDA JIT OK: {gpu.name} ({gpu.arch}), NVRTC {major}.{minor}, {n} numbers added on the GPU"
    finally:
        gpu.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuda-path", help="CUDA toolkit folder (default: CUDA_PATH, then %%LOCALAPPDATA%%\\Neuro\\cuda\\12.9)")
    parser.add_argument("--compile-only", metavar="ARCH",
                        help="only compile the self-test kernel with NVRTC for ARCH (e.g. sm_70); no GPU is used")
    args = parser.parse_args()
    try:
        if args.compile_only:
            nvrtc = Nvrtc(args.cuda_path)
            cubin = nvrtc.compile(SELF_TEST, args.compile_only)
            print(f"NVRTC {nvrtc.version[0]}.{nvrtc.version[1]} compiled the self-test kernel for "
                  f"{args.compile_only}: {len(cubin)} bytes of cubin")
        else:
            print(self_test(args.cuda_path))
    except CudaJitError as exc:
        print(f"CUDA JIT FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
