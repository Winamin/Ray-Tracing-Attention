"""Independent RT geometry arithmetic worker; no language-model assets required."""
from pathlib import Path
import os
import struct
import subprocess
import tempfile

import numpy as np


class GeometryRuntime:
    """Debug scalar arithmetic, MLP and single-head attention through RT distances.

    Exponentials use a compiled piecewise-linear surface. Shader integer logic
    encodes operands and connects ray queries;
    floating arithmetic and reductions are evaluated by triangle intersections.
    """

    OPERATIONS = {
        'add': 0, 'multiply': 1, 'divide': 2, 'sum': 3,
        'exp': 4, 'softmax': 5, 'matmul': 6, 'swiglu': 7,
    }

    def __init__(self):
        from ..compiler.build import build
        from ..compiler.spec import ModelSpec

        # The existing build system supplies the worker and its SPIR-V. This
        # small specification is only a build configuration, not a loaded model.
        spec = ModelSpec('llama', 256, 256, 32, ('full',), 1, 1, 256, 256,
                         attention_gate=False, qk_norm=False)
        worker = build(spec)
        self.stderr = tempfile.TemporaryFile()
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        self.process = subprocess.Popen(
            [str(worker), '--geometry-worker'], cwd=worker.parent,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, **options)
        try:
            magic, version, segments, operators = struct.unpack('<4I', self._read(16))
            if magic != 0x52544741 or version != 1:
                raise RuntimeError('Geometry arithmetic protocol mismatch')
            self.metadata = {
                'worker': str(Path(worker).resolve()), 'protocol_version': version,
                'operators': operators, 'exponential_segments': segments,
                'exponential': 'piecewise_linear_rt_exp2',
                'arithmetic': 'triangle_intersection_distance',
                'operand_encoding': 'shader_integer_ieee754',
                'softmax_reference_selection': 'rt_closest_hit',
                'dot_product': 'two_coordinate_plane_fragments',
                'pair_planes': 65536, 'pair_templates': 64, 'pair_instances': 1024,
                'model_loaded': False,
            }
        except BaseException:
            self.close()
            raise

    def _read(self, count):
        parts = []
        while count:
            part = self.process.stdout.read(count)
            if not part:
                self.stderr.seek(0)
                message = self.stderr.read().decode('utf8', errors='replace')
                raise RuntimeError(message or 'Geometry worker closed its output')
            parts.append(part)
            count -= len(part)
        return b''.join(parts)

    @staticmethod
    def _array(value):
        result = np.ascontiguousarray(value, dtype='<f4')
        if not result.size or not np.isfinite(result).all():
            raise ValueError('Geometry operands must be nonempty and finite')
        return result

    @staticmethod
    def _matrix(value):
        result = GeometryRuntime._array(value)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2:
            raise ValueError('Expected a vector or matrix')
        return result

    def _request(self, operation, rows, mode, dimensions, arrays):
        header = struct.pack('<3I', operation, rows, mode)
        shape = struct.pack('<' + 'I' * len(dimensions), *dimensions)
        self.process.stdin.write(header + shape)
        for value in arrays:
            self.process.stdin.write(np.ascontiguousarray(value, dtype='<f4').tobytes())
        self.process.stdin.flush()
        status, count = struct.unpack('<2I', self._read(8))
        payload = self._read(count)
        if status:
            raise RuntimeError('Geometry arithmetic request failed')
        magic, segments, builds, output_count = struct.unpack_from('<4I', payload)
        if magic != 0x52474131 or len(payload) != 56 + output_count * 4:
            raise RuntimeError('Geometry arithmetic response mismatch')
        rays, resident_bytes = struct.unpack_from('<2Q', payload, 16)
        setup_ms, gpu_ms, wall_ms = struct.unpack_from('<3d', payload, 32)
        return {
            'values': np.frombuffer(payload, '<f4', count=output_count, offset=56).copy(),
            'rays': rays, 'space_builds': builds, 'blas_count': 5, 'tlas_count': 1,
            'instance_count': 1028, 'pair_planes': 65536,
            'space_bytes': resident_bytes, 'exponential_segments': segments,
            'setup_wall_ms': setup_ms, 'gpu_ms': gpu_ms, 'request_wall_ms': setup_ms + wall_ms,
            'exponential': 'piecewise_linear_rt_exp2',
        }

    def arithmetic(self, operation, left, right=None):
        """Run scalar/row operators; matmul takes weights as [output, input]."""
        if operation not in self.OPERATIONS:
            raise ValueError('Unknown geometry arithmetic operator')
        mode = self.OPERATIONS[operation]
        left = self._array(left)
        if operation in ('sum', 'softmax'):
            matrix = self._matrix(left)
            rows, columns = matrix.shape
            result_shape = (() if left.ndim == 1 else (rows,)) if operation == 'sum' else left.shape
            result = self._request(16, rows, mode, (columns, 1), [matrix])
        elif operation == 'matmul':
            matrix = self._matrix(left)
            weights = self._matrix(right)
            if matrix.shape[1] != weights.shape[1]:
                raise ValueError('Input and weight column counts differ')
            rows, columns = matrix.shape
            result_shape = (rows, len(weights))
            result = self._request(16, rows, mode, (columns, len(weights)), [matrix, weights])
        else:
            arrays = [left]
            if operation != 'exp':
                right = self._array(right)
                if left.shape != right.shape:
                    raise ValueError('Scalar operator shapes differ')
                if operation == 'divide' and np.any(right == 0):
                    raise ValueError('Geometry divisor must be nonzero')
                arrays.append(right)
            result_shape = left.shape
            result = self._request(16, left.size, mode, (1, 1), arrays)
        result['values'] = result['values'].reshape(result_shape)
        return result

    def mlp(self, inputs, gate_weights, up_weights, down_weights):
        """Return W_down[SwiGLU(W_gate x, W_up x)] without adding a residual."""
        inputs = self._matrix(inputs)
        gate = self._matrix(gate_weights)
        up = self._matrix(up_weights)
        down = self._matrix(down_weights)
        rows, hidden = inputs.shape
        middle = len(gate)
        if gate.shape != up.shape or gate.shape[1] != hidden or down.shape[1] != middle:
            raise ValueError('MLP projection shapes are incompatible')
        result = self._request(17, rows, 0, (hidden, middle, len(down)), [inputs, gate, up, down])
        raw = result.pop('values')
        boundary = rows * middle
        result['activation'] = raw[:boundary].reshape(rows, middle)
        result['output'] = raw[boundary:].reshape(rows, len(down))
        return result

    def attention(self, queries, keys, values):
        """Single-head scaled QK, geometric softmax and PV; history is preselected.

        All supplied keys participate. This debug interface applies no causal
        mask, RoPE, GQA expansion or output gate; callers supply those operands.
        """
        queries = self._matrix(queries)
        keys = self._matrix(keys)
        values = self._matrix(values)
        if queries.shape[1] != keys.shape[1] or len(keys) != len(values):
            raise ValueError('Attention operand shapes are incompatible')
        rows, dimension = queries.shape
        history, value_width = values.shape
        result = self._request(18, rows, 0, (dimension, history, value_width),
                               [queries, keys, np.ascontiguousarray(values.T)])
        raw = result.pop('values')
        boundary = rows * history
        result['weights'] = raw[:boundary].reshape(rows, history)
        result['output'] = raw[boundary:].reshape(rows, value_width)
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(struct.pack('<I', 0))
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
