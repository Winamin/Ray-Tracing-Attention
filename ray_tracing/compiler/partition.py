"""Disjoint coordinate blocks and their exact inverse, shared with native RT."""
from dataclasses import dataclass
from pathlib import Path
import json
import struct
import numpy as np

BLOCK_WIDTH = 256
PAIR_WIDTH = 2
MAGIC = b'RTPART01'


@dataclass(frozen=True)
class CoordinatePartition:
    dimension: int
    block_width: int = BLOCK_WIDTH

    def __post_init__(self):
        if self.dimension <= 0 or self.dimension % self.block_width:
            raise ValueError('Coordinate blocks must cover an aligned positive dimension')
        if self.block_width != BLOCK_WIDTH:
            raise ValueError('Native coefficient blocks contain 256 coordinates')

    @property
    def blocks(self):
        return self.dimension // self.block_width

    @property
    def pairs(self):
        return self.dimension // PAIR_WIDTH

    def split(self, values):
        values = np.asarray(values)
        if values.shape[-1] != self.dimension:
            raise ValueError('The complete coordinate vector is required')
        return values.reshape(*values.shape[:-1], self.blocks, self.block_width)

    def restore(self, blocks):
        blocks = np.asarray(blocks)
        if blocks.shape[-2:] != (self.blocks, self.block_width):
            raise ValueError('Every coordinate block must be present in order')
        return blocks.reshape(*blocks.shape[:-2], self.dimension)

    def packets(self, values):
        values = np.asarray(values)
        if values.dtype != np.dtype('float32'):
            raise ValueError('State packets preserve the original float32 bit patterns')
        return self.split(values).view(np.uint32).reshape(
            *values.shape[:-1], self.blocks, self.block_width // PAIR_WIDTH, PAIR_WIDTH)

    def from_packets(self, packets):
        packets = np.asarray(packets)
        if packets.dtype != np.dtype('uint32') or packets.shape[-3:] != (
                self.blocks, self.block_width // PAIR_WIDTH, PAIR_WIDTH):
            raise ValueError('Every two-coordinate packet and its block identity are required')
        return packets.reshape(*packets.shape[:-3], self.dimension).view(np.float32)

    def descriptor(self, vocabulary):
        return {'version': 1, 'transform': 'ordered_disjoint_selectors',
                'dimension': self.dimension, 'block_width': self.block_width,
                'blocks': self.blocks, 'pair_width': PAIR_WIDTH, 'pairs': self.pairs,
                'state_codec': 'original_float32_bits',
                'weight_codec': 'original_compressed_coefficients',
                'geometry': 'block_plane_aabbs',
                'instance_count': 1, 'primitive_count': vocabulary * self.blocks,
                'identity': ['token_id', 'block_id', 'pair_id'],
                'file': 'coordinate_partition.bin'}

    def write(self, directory):
        path = Path(directory) / 'coordinate_partition.bin'
        records = np.array([(b, b * self.block_width, self.block_width,
                             b * self.block_width // PAIR_WIDTH)
                            for b in range(self.blocks)], dtype='<u4')
        contents = MAGIC + struct.pack('<4I', self.dimension, self.block_width,
                                       self.blocks, PAIR_WIDTH) + records.tobytes()
        if not path.exists() or path.read_bytes() != contents:
            pending = path.with_suffix('.partial')
            pending.write_bytes(contents)
            pending.replace(path)
        return path

    @classmethod
    def read(cls, path):
        data = Path(path).read_bytes()
        if len(data) < 24 or data[:8] != MAGIC:
            raise ValueError('Invalid coordinate partition asset')
        dimension, width, blocks, pair = struct.unpack_from('<4I', data, 8)
        result = cls(dimension, width)
        expected = np.array([(b, b * width, width, b * width // PAIR_WIDTH)
                             for b in range(result.blocks)], dtype='<u4')
        if blocks != result.blocks or pair != PAIR_WIDTH or data[24:] != expected.tobytes():
            raise ValueError('Coordinate selectors overlap, omit a block or change its identity')
        return result


def ensure_partition(directory):
    """Compile the missing indexing asset from the saved partition."""
    directory = Path(directory)
    metadata_path = directory / 'model.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf8'))
    partition = CoordinatePartition(int(metadata['hidden_size']))
    partition.write(directory)
    descriptor = partition.descriptor(int(metadata['vocabulary']))
    if metadata.get('coordinate_partition') != descriptor:
        metadata['coordinate_partition'] = descriptor
        pending = metadata_path.with_suffix('.partial')
        pending.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
        pending.replace(metadata_path)
    return partition
