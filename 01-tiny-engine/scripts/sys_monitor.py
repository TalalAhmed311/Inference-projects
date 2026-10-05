"""Coarse system resource sampler for multi-request experiments.

Samples (process-wide / host-wide — not per-weight tensors):
  - CPU % and host RAM
  - GPU util % and VRAM (nvidia-smi)
  - Disk read/write bytes (cumulative counters → rate between samples)
  - Optional process RSS of a watched PID (the server)

Disk I/O is whole-device / process aggregate — useful for spotting weight/cache
page-ins or HF cache reads, not individual parameter moves.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore


@dataclass
class Sample:
    t: float
    cpu_pct: float | None = None
    ram_used_gib: float | None = None
    ram_total_gib: float | None = None
    ram_pct: float | None = None
    gpu_util_pct: float | None = None
    gpu_mem_used_mib: float | None = None
    gpu_mem_total_mib: float | None = None
    disk_read_mib_s: float | None = None
    disk_write_mib_s: float | None = None
    proc_rss_gib: float | None = None
    proc_cpu_pct: float | None = None


@dataclass
class MonitorSummary:
    n_samples: int = 0
    cpu_pct_mean: float | None = None
    cpu_pct_peak: float | None = None
    ram_used_peak_gib: float | None = None
    ram_pct_peak: float | None = None
    gpu_util_mean_pct: float | None = None
    gpu_util_peak_pct: float | None = None
    gpu_mem_peak_mib: float | None = None
    disk_read_total_mib: float | None = None
    disk_write_total_mib: float | None = None
    disk_read_peak_mib_s: float | None = None
    disk_write_peak_mib_s: float | None = None
    proc_rss_peak_gib: float | None = None


def _gpu_once() -> tuple[float | None, float | None, float | None]:
    if not shutil.which("nvidia-smi"):
        return None, None, None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        utils, used, total = [], [], []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            utils.append(float(parts[0]))
            used.append(float(parts[1]))
            total.append(float(parts[2]))
        if not utils:
            return None, None, None
        return sum(utils) / len(utils), sum(used), sum(total)
    except Exception:  # noqa: BLE001
        return None, None, None


class SystemMonitor:
    def __init__(self, interval: float = 0.5, pid: int | None = None):
        self.interval = interval
        self.pid = pid
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_disk = None
        self._prev_t: float | None = None
        self._disk_read0 = None
        self._disk_write0 = None
        self._proc = psutil.Process(pid) if psutil and pid else None
        if self._proc:
            try:
                self._proc.cpu_percent(None)  # prime
            except Exception:  # noqa: BLE001
                self._proc = None

    def _disk_counters(self):
        if not psutil:
            return None
        try:
            c = psutil.disk_io_counters()
            if c is None:
                return None
            return c.read_bytes, c.write_bytes
        except Exception:  # noqa: BLE001
            return None

    def _sample(self) -> Sample:
        now = time.time()
        s = Sample(t=now)
        if psutil:
            s.cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            s.ram_used_gib = vm.used / 2**30
            s.ram_total_gib = vm.total / 2**30
            s.ram_pct = vm.percent
            disk = self._disk_counters()
            if disk is not None:
                r, w = disk
                if self._disk_read0 is None:
                    self._disk_read0, self._disk_write0 = r, w
                if self._prev_disk is not None and self._prev_t is not None:
                    dt = max(now - self._prev_t, 1e-6)
                    s.disk_read_mib_s = (r - self._prev_disk[0]) / 2**20 / dt
                    s.disk_write_mib_s = (w - self._prev_disk[1]) / 2**20 / dt
                self._prev_disk = (r, w)
                self._prev_t = now
            if self._proc is not None:
                try:
                    s.proc_rss_gib = self._proc.memory_info().rss / 2**30
                    s.proc_cpu_pct = self._proc.cpu_percent(None)
                except Exception:  # noqa: BLE001
                    pass
        gu, gused, gtot = _gpu_once()
        s.gpu_util_pct, s.gpu_mem_used_mib, s.gpu_mem_total_mib = gu, gused, gtot
        return s

    def _loop(self):
        if psutil:
            psutil.cpu_percent(None)
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sys-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[Sample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples

    def window(self, t0: float, t1: float) -> list[Sample]:
        return [s for s in self.samples if t0 <= s.t <= t1]

    def summarize(self, samples: list[Sample] | None = None) -> MonitorSummary:
        xs = samples if samples is not None else self.samples
        out = MonitorSummary(n_samples=len(xs))
        if not xs:
            return out

        def col(name):
            return [getattr(s, name) for s in xs if getattr(s, name) is not None]

        def mean(vs):
            return sum(vs) / len(vs) if vs else None

        def peak(vs):
            return max(vs) if vs else None

        out.cpu_pct_mean = mean(col("cpu_pct"))
        out.cpu_pct_peak = peak(col("cpu_pct"))
        out.ram_used_peak_gib = peak(col("ram_used_gib"))
        out.ram_pct_peak = peak(col("ram_pct"))
        out.gpu_util_mean_pct = mean(col("gpu_util_pct"))
        out.gpu_util_peak_pct = peak(col("gpu_util_pct"))
        out.gpu_mem_peak_mib = peak(col("gpu_mem_used_mib"))
        out.disk_read_peak_mib_s = peak(col("disk_read_mib_s"))
        out.disk_write_peak_mib_s = peak(col("disk_write_mib_s"))
        out.proc_rss_peak_gib = peak(col("proc_rss_gib"))
        # approximate totals from rates × interval
        reads = col("disk_read_mib_s")
        writes = col("disk_write_mib_s")
        if len(xs) >= 2:
            dur = xs[-1].t - xs[0].t
            if dur > 0 and reads:
                out.disk_read_total_mib = mean(reads) * dur
            if dur > 0 and writes:
                out.disk_write_total_mib = mean(writes) * dur
        return out

    def dump_jsonl(self, path: Path, samples: list[Sample] | None = None):
        xs = samples if samples is not None else self.samples
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for s in xs:
                f.write(json.dumps(asdict(s)) + "\n")
