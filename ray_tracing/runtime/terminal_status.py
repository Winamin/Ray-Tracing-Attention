"""A single reserved terminal footer, updated independently of token output."""
import ctypes
import os
import shutil
import sys
import threading
import unicodedata


def fit(text, width):
    pieces, used = [], 0
    for char in text:
        used += 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used > width:
            break
        pieces.append(char)
    return ''.join(pieces)


def formatted(value, suffix='', digits=1):
    return '—' if value is None else f'{value:.{digits}f}{suffix}'


class TerminalStatus:
    def __init__(self, monitor, lock):
        self.monitor, self.lock = monitor, lock
        self.stop = threading.Event()
        self.thread = None
        self.height = 0
        self.rendered = None

    def start(self):
        if not sys.stdout.isatty() or (os.name != 'nt' and os.environ.get('TERM') == 'dumb'):
            return
        if os.name == 'nt':
            kernel = ctypes.windll.kernel32
            kernel.GetStdHandle.restype = ctypes.c_void_p
            handle = kernel.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if not kernel.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)):
                return
            if not kernel.SetConsoleMode(ctypes.c_void_p(handle), mode.value | 4):
                return
        width, height = shutil.get_terminal_size()
        if height < 6:
            return
        self.height = height
        with self.lock:
            sys.stdout.write(f'\033[2J\033[H\033[1;{height-1}r')
            sys.stdout.flush()
        self.draw()
        self.thread = threading.Thread(target=self._run, name='rt-status-footer', daemon=True)
        self.thread.start()

    def draw(self):
        if not self.height:
            return
        live = self.monitor.live()
        snapshot = self.monitor.snapshot()
        gpu = snapshot['gpus'][0] if snapshot['gpus'] else {}
        phase = {'startup': '加载', 'prefill': '上下文', 'decode': '生成', 'idle': '就绪'}.get(live['phase'], live['phase'])
        text = (f"{phase} · RT查询 {formatted(live['rt_query_ms_per_token'], 'ms', 3)} "
                f"占时 {formatted(live['rt_query_gpu_share_percent'], '%')} · "
                f"GPU {formatted(gpu.get('gpu_utilization_percent'), '%', 0)} · "
                f"{live['tokens_per_second']} t/s · {formatted(gpu.get('memory_used_mib'), 'MiB', 0)}")
        width, height = shutil.get_terminal_size()
        if height < 6:
            return
        with self.lock:
            resized = height != self.height
            self.height = height
            rendered = (width, height, text)
            if rendered == self.rendered:
                return
            self.rendered = rendered
            margin = f'\033[1;{height-1}r' if resized else ''
            sys.stdout.write(f'\0337{margin}\033[{height};1H\033[2K\033[2m{fit(text, width-1)}\033[0m\0338')
            sys.stdout.flush()

    def _run(self):
        while not self.stop.wait(.2):
            self.draw()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.height:
            with self.lock:
                sys.stdout.write(f'\0337\033[r\033[{self.height};1H\033[2K\0338')
                sys.stdout.flush()
