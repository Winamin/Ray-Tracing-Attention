"""Bounded GPU sampling and thread-safe JSONL records for the resident CLI."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import csv
import ctypes
import json
import os
import subprocess
import threading
import time


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLog:
    def __init__(self, path=None):
        self.path = Path(path).resolve() if path else None
        self.lock = threading.Lock()
        self.stream = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("a", encoding="utf8")

    def write(self, event, **fields):
        if self.stream:
            record = {"time": timestamp(), "event": event, **fields}
            with self.lock:
                self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self.stream.flush()

    def close(self):
        with self.lock:
            if self.stream:
                self.stream.close()
                self.stream = None


class GpuMonitor:
    FIELDS = ("index", "name", "memory.total", "memory.used", "utilization.gpu",
              "temperature.gpu", "power.draw", "power.limit", "clocks.sm", "clocks.mem", "pstate")
    KEYS = ("index", "name", "memory_total_mib", "memory_used_mib", "gpu_utilization_percent",
            "temperature_c", "power_w", "power_limit_w", "sm_clock_mhz", "memory_clock_mhz", "pstate")

    def __init__(self, interval=.2, log=None):
        self.interval = interval
        self.log = log
        self.samples = deque(maxlen=2048)
        self.latest = {"available": False, "gpus": [], "reason": "尚未采样"}
        self.phase, self.tokens = "startup", 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.rt_samples = deque(maxlen=512)
        self.nvml = None
        try:
            self.nvml = NvmlSensors()
        except (OSError, RuntimeError, AttributeError):
            pass

    def sample(self):
        try:
            if self.nvml:
                gpus = self.nvml.sample()
            else:
                options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(self.FIELDS),
                                         "--format=csv,noheader,nounits"], capture_output=True,
                                        text=True, timeout=3, **options)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or "nvidia-smi 查询失败")
                gpus = []
                for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                    if len(row) != len(self.KEYS):
                        continue
                    gpu = dict(zip(self.KEYS, (v.strip() for v in row)))
                    for key in self.KEYS:
                        if key not in ("name", "pstate"):
                            try:
                                gpu[key] = float(gpu[key])
                            except ValueError:
                                gpu[key] = None
                    gpus.append(gpu)
            report = {"available": bool(gpus), "gpus": gpus, "source": "NVML" if self.nvml else "nvidia-smi"}
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            report = {"available": False, "gpus": [], "reason": str(error)}
        with self.lock:
            report.update(time=timestamp(), monotonic=time.perf_counter(), phase=self.phase, tokens=self.tokens)
            self.latest = report
            self.samples.append(report)
        if self.log:
            self.log.write("gpu_sample", **report)
        return report

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def progress(self, phase, tokens=0):
        with self.lock:
            self.phase, self.tokens = phase, tokens

    def token(self, result, tokens):
        with self.lock:
            self.phase, self.tokens = 'decode', tokens
            self.rt_samples.append({'at': time.perf_counter(), 'tokens': len(result['ids']),
                'gpu_ms': result['gpu_ms'], 'rt_gpu_ms': result.get('rt_query_gpu_ms'),
                'cache_hit': result['reply_cache_hit']})

    def live(self):
        now = time.perf_counter()
        with self.lock:
            rows = [s for s in self.rt_samples if s['at'] >= now-1.]
            phase, tokens = self.phase, self.tokens
        gpu_ms = sum(s['gpu_ms'] for s in rows)
        rt_ms = sum(s['rt_gpu_ms'] or 0 for s in rows)
        measured = any(s['rt_gpu_ms'] is not None for s in rows)
        count = sum(s['tokens'] for s in rows)
        return {'phase': phase, 'tokens': tokens, 'tokens_per_second': count,
                'rt_query_ms_per_token': rt_ms/count if measured and count else None,
                'rt_query_gpu_share_percent': rt_ms/gpu_ms*100 if measured and gpu_ms else None,
                'rt_core_utilization_percent': None, 'window_seconds': 1.}

    def since(self, start):
        with self.lock:
            samples = [s for s in self.samples if s["monotonic"] >= start]
        peaks = {}
        for sample in samples:
            for gpu in sample["gpus"]:
                index = str(gpu["index"])
                peak = peaks.setdefault(index, {"name": gpu["name"], "memory_peak_mib": None,
                                               "gpu_utilization_peak_percent": None})
                for source, dest in (("memory_used_mib", "memory_peak_mib"),
                                     ("gpu_utilization_percent", "gpu_utilization_peak_percent")):
                    if gpu[source] is not None:
                        peak[dest] = max(peak[dest] or 0., gpu[source])
        return {"sample_count": len(samples), "gpus": peaks}

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self._run, name="rt-gpu-monitor", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=4)
        if self.nvml:
            self.nvml.close()


class NvmlSensors:
    """Direct driver reads; no process launch in the sampling loop."""
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    class Utilization(ctypes.Structure):
        _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

    def __init__(self):
        self.dll = ctypes.CDLL('nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
        if self.dll.nvmlInit_v2() != 0:
            raise RuntimeError('NVML 初始化失败')
        count = ctypes.c_uint()
        if self.dll.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            self.close()
            raise RuntimeError('NVML 设备枚举失败')
        self.devices = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if self.dll.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == 0:
                name = ctypes.create_string_buffer(96)
                self.dll.nvmlDeviceGetName(handle, name, len(name))
                self.devices.append((index, handle, name.value.decode('utf8', errors='replace')))

    def value(self, name, handle, *args):
        value = ctypes.c_uint()
        method = getattr(self.dll, name, None)
        if method and method(handle, *args, ctypes.byref(value)) == 0:
            return float(value.value)
        return None

    def sample(self):
        gpus = []
        for index, handle, name in self.devices:
            mem, util = self.Memory(), self.Utilization()
            has_mem = self.dll.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) == 0
            has_util = self.dll.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)) == 0
            power = self.value('nvmlDeviceGetPowerUsage', handle)
            limit = self.value('nvmlDeviceGetEnforcedPowerLimit', handle)
            state = self.value('nvmlDeviceGetPerformanceState', handle)
            gpus.append({'index': index, 'name': name,
                'memory_total_mib': mem.total/1048576 if has_mem else None,
                'memory_used_mib': mem.used/1048576 if has_mem else None,
                'gpu_utilization_percent': util.gpu if has_util else None,
                'temperature_c': self.value('nvmlDeviceGetTemperature', handle, 0),
                'power_w': power/1000 if power is not None else None,
                'power_limit_w': limit/1000 if limit is not None else None,
                'sm_clock_mhz': self.value('nvmlDeviceGetClockInfo', handle, 1),
                'memory_clock_mhz': self.value('nvmlDeviceGetClockInfo', handle, 2),
                'pstate': f'P{int(state)}' if state is not None else '—'})
        return gpus

    def close(self):
        self.dll.nvmlShutdown()
