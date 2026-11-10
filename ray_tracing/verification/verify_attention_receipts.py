"""Verify complete RT attention receipts against an independent FP64 oracle.

The diagnostic consumes rotated Q and compact K/V. The oracle decodes the
production Q4_K/Q6_K history packets with gguf as its attention input.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import numpy as np

from ..compiler.spec import asset_spec
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import LOCAL, default_model


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    light = np.exp(-np.abs(value))
    return np.where(value >= 0, 1 / (1 + light), light / (1 + light))


def error_metrics(actual, expected, tolerance=3e-6):
    """Compare maximum error with an absolute floor and reference scale."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return {'passed': False, 'actual_shape': list(actual.shape),
                'expected_shape': list(expected.shape)}
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {'passed': False, 'finite': False}
    delta = actual - expected
    maximum = float(np.max(np.abs(delta), initial=0))
    peak = float(np.max(np.abs(expected), initial=0))
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.
    ref_rms = float(np.sqrt(np.mean(expected * expected))) if expected.size else 0.
    limit = tolerance * max(1., peak)
    return {'passed': maximum <= limit, 'finite': True, 'max_abs': maximum,
            'rms': rms, 'relative_rms': rms / max(ref_rms, 1e-30),
            'max_error_over_peak': maximum / max(peak, 1e-30),
            'allowed_max_abs': limit}


def decode_history(packets, spec):
    """Independent decoder; each returned row follows input history time order."""
    from gguf import GGMLQuantizationType, dequantize
    packets = np.asarray(packets, dtype=np.uint8)
    if packets.ndim != 2 or packets.shape[1] != spec.history_bytes:
        raise AssertionError(('History packet shape', packets.shape, spec.history_bytes))
    blocks = (spec.kv_width + 255) // 256
    boundary = blocks * 144
    keys = dequantize(np.ascontiguousarray(packets[:, :boundary]),
                      GGMLQuantizationType.Q4_K)
    values = dequantize(np.ascontiguousarray(packets[:, boundary:boundary + blocks * 210]),
                        GGMLQuantizationType.Q6_K)
    shape = (len(packets), spec.kv_heads, spec.attention_head_dim)
    keys = keys.reshape(len(packets), blocks * 256)[:, :spec.kv_width].reshape(shape)
    values = values.reshape(len(packets), blocks * 256)[:, :spec.kv_width].reshape(shape)
    assert np.isfinite(keys).all() and np.isfinite(values).all(), 'Nonfinite history decode'
    return keys.astype(np.float64), values.astype(np.float64)


def shader_provenance(worker):
    module = Path(worker).with_name('resident_attention_receipt.spv')
    assert module.is_file(), 'Compiled attention receipt module is missing'
    validator = os.environ.get('RT_SPIRV_VAL')
    disassembler = (str(Path(validator).with_name('spirv-dis.exe')) if validator else
                    shutil.which('spirv-dis') or str(LOCAL / 'tools/glslang/mingw64/bin/spirv-dis.exe'))
    result = subprocess.run([disassembler, str(module)], capture_output=True,
                            text=True, check=True)
    assert 'OpRayQueryInitializeKHR' in result.stdout and 'OpRayQueryProceedKHR' in result.stdout
    return {'module': str(module), 'sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
            'ray_query_initialize': True, 'ray_query_proceed': True,
            'cooperative_matrix_instructions': 'OpCooperativeMatrix' in result.stdout}


def full_attention(query, keys, values, gate, query_start, history_start, capacity,
                   gated=True):
    """Dense FP64 reference with explicit GQA mapping and absolute causal mask."""
    query = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    rows, heads, dimension = query.shape
    assert keys.shape == values.shape and keys.shape[2] == dimension
    assert heads % keys.shape[1] == 0
    mapping = np.arange(heads) // (heads // keys.shape[1])
    expanded_key = keys[:, mapping, :]
    expanded_value = values[:, mapping, :]
    scores = np.einsum('rhd,nhd->rhn', query, expanded_key,
                       dtype=np.float64, optimize=True) / np.sqrt(float(dimension))
    ages = history_start + np.arange(len(keys), dtype=np.int64)
    times = query_start + np.arange(rows, dtype=np.int64)
    visible = ((ages[None, :] <= times[:, None]) &
               (times[:, None] - ages[None, :] < capacity))
    assert visible.any(axis=1).all(), 'Each query needs at least one visible history item'
    masked = np.where(visible[:, None, :], scores, -np.inf)
    maximum = masked.max(axis=2)
    energy = np.exp(masked - maximum[:, :, None])
    reference = energy.sum(axis=2, dtype=np.float64)
    numerator = np.einsum('rhn,nhd->rhd', energy, expanded_value,
                          dtype=np.float64, optimize=True)
    output = numerator / reference[:, :, None]
    if gated:
        output *= sigmoid(gate)
    return {'output': output, 'maximum': maximum, 'reference': reference,
            'scores': scores, 'visible': visible, 'values': expanded_value,
            'times': times}


def receipt_reference(dense, block, history_start):
    """Construct expected groups from time identities, independently of GPU order."""
    rows, heads, _ = dense['scores'].shape
    records = []
    for row in range(rows):
        available = int(np.count_nonzero(dense['visible'][row]))
        time = int(dense['times'][row])
        for group in range((available + block - 1) // block):
            count = min(block, available - group * block)
            # GPU coverage bit k identifies time - (group*block+k).
            indices = time - history_start - group * block - np.arange(count)
            assert np.all(indices >= 0) and np.all(dense['visible'][row, indices])
            for head in range(heads):
                scores = dense['scores'][row, head, indices]
                maximum = float(scores.max())
                energy = np.exp(scores - maximum)
                content = dense['values'][indices, head, :]
                numerator = np.einsum('n,nd->d', energy, content, dtype=np.float64)
                records.append({'row': row, 'head': head, 'group': group,
                                'count': count, 'maximum': maximum,
                                'reference': float(energy.sum()), 'numerator': numerator,
                                'content': content, 'scores': scores})
    return records


def check_receipts(raw, expected, block, heads, rows, dimension, gate, gated=True, compact=False):
    """Check every represented slot, then independently merge returned GPU summaries."""
    stride = dimension + 8 + block // 32
    raw = np.asarray(raw, dtype=np.uint32)
    assert raw.shape == (len(expected), stride), ('Receipt shape', raw.shape, len(expected), stride)
    assert np.all((raw[:, 7] & 3) == 3), ('Missing verified/finite flags', raw[:, 7].tolist())
    assert np.all((raw[:, 7] & ~np.uint32(15)) == 0), ('Unknown receipt flags', raw[:, 7].tolist())
    numbers = raw.view(np.float32)
    wanted_max = np.array([e['maximum'] for e in expected])
    wanted_z = np.array([e['reference'] for e in expected])
    wanted_u = np.array([e['numerator'] for e in expected])
    actual_u = numbers[:, 8 + block // 32:]
    count_sum = exp_sum = value_group_sum = weight_term_sum = rays = 0
    for record, e in zip(raw, expected):
        count = int(record[2])
        assert count == e['count'], (e['row'], e['head'], e['group'], count, e['count'])
        wanted_mask = np.zeros(block // 32, dtype=np.uint32)
        for word in range(block // 32):
            bits = min(32, max(0, count - word * 32))
            wanted_mask[word] = np.uint32((1 << bits) - 1 if bits else 0)
        np.testing.assert_array_equal(record[8:8 + block // 32], wanted_mask,
                                      err_msg=f"coverage row={e['row']} head={e['head']} group={e['group']}")
        # Compact uses a complete-block maximum; the old dense producer updates
        # its maximum every four positions. Both retain parallel history rays
        # with sixteen positions per ray.
        maximum_exp_evaluations = count if compact else count + (count + 3) // 4 - 1
        assert 1 <= int(record[3]) <= maximum_exp_evaluations, ('Actual exp count', record[3], count)
        assert 1 <= int(record[4]) <= count, ('V vector group count', record[4], count)
        assert int(record[5]) == count, ('Weight contribution coverage', record[5], count)
        assert bool(int(record[7]) & 8) == (int(record[4]) < count), 'V grouping flag mismatch'
        if e['head']:
            assert int(record[6]) == 0, 'Heads must share the RT traversal'
        else:
            assert int(record[6]) == 2 + (count + 15) // 16, ('Lens/history RT initializations', record[6], count)
        count_sum += count
        exp_sum += int(record[3])
        value_group_sum += int(record[4])
        weight_term_sum += int(record[5])
        rays += int(record[6])
    maximum = np.full((rows, heads), -np.inf, dtype=np.float64)
    for i, e in enumerate(expected):
        maximum[e['row'], e['head']] = max(maximum[e['row'], e['head']], float(numbers[i, 0]))
    denominator = np.zeros((rows, heads), dtype=np.float64)
    numerator = np.zeros((rows, heads, dimension), dtype=np.float64)
    for i, e in enumerate(expected):
        row, head = e['row'], e['head']
        scale = np.exp(float(numbers[i, 0]) - maximum[row, head])
        denominator[row, head] += float(numbers[i, 1]) * scale
        numerator[row, head] += actual_u[i].astype(np.float64) * scale
    assert np.isfinite(denominator).all() and np.all(denominator > 0)
    merged = numerator / denominator[:, :, None]
    if gated:
        merged *= sigmoid(gate)
    checks = {'maximum': error_metrics(numbers[:, 0], wanted_max),
              'normalizer': error_metrics(numbers[:, 1], wanted_z),
              'weighted_value': error_metrics(actual_u, wanted_u)}
    stats = {'records': len(raw), 'represented_head_positions': count_sum,
             'record_fraction': len(raw) / max(count_sum, 1), 'receipt_bytes': raw.nbytes,
             'exp_evaluations': exp_sum, 'value_groups': value_group_sum,
             'weight_terms': weight_term_sum, 'shared_rt_initializations': rays,
             'complete_coverage': True, 'maximum': maximum, 'normalizer': denominator}
    return checks, stats, merged


def cases(spec, block):
    """Cases target skipped contributions, grouping boundaries and head identities."""
    rng = np.random.default_rng(20261110)
    heads, kv, dimension = spec.attention_heads, spec.kv_heads, spec.attention_head_dim

    def sample(name, count, rows=1, history_start=0):
        return {'name': name, 'query': rng.normal(0, .25, (rows, heads, dimension)).astype(np.float32),
                'keys': rng.normal(0, .25, (count, kv, dimension)).astype(np.float32),
                'values': rng.normal(0, .5, (count, kv, dimension)).astype(np.float32),
                'gate': rng.normal(0, .5, (rows, heads, dimension)).astype(np.float32),
                'history_start': history_start, 'query_start': history_start + count - rows}

    boundaries = sorted({1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                         block - 1, block, block + 1, 2 * block + 1})
    for count in boundaries:
        yield sample(f'boundary_{count}', count, min(count, 7))
    c = sample('zero_q_uniform', 33, 8)
    c['query'].fill(0)
    yield c
    c = sample('subnormal_and_signed_zero_q', 33)
    c['query'].fill(np.nextafter(np.float32(0), np.float32(1)))
    c['query'][:, :, 1::4] = np.float32(-0.)
    c['query'][:, :, 2::4] = np.float32(0.)
    c['query'][:, :, 3::4] *= np.float32(-1.)
    yield c
    c = sample('tiny_q_with_nontrivial_scores', 65, 17)
    c['query'].fill(0)
    c['query'][:, :, 0] = np.float32(1e-6)
    c['query'][:, :, 1] = np.float32(.25)
    c['keys'].fill(0)
    c['keys'][:, :, 0] = np.linspace(-1e6, 1e6, len(c['keys']), dtype=np.float32)[:, None]
    c['values'][:, :, 0] = np.linspace(-1., 1., len(c['values']), dtype=np.float32)[:, None]
    yield c
    c = sample('identical_q_distinct_keys', block + 1, 7)
    c['query'].fill(np.float32(.01))
    yield c
    c = sample('identical_scores_distinct_values', 2 * block + 1)
    c['query'].fill(0)
    c['query'][:, :, 0] = .25
    c['keys'].fill(0)
    c['keys'][:, :, 0] = 4.
    c['expect_one_exp_per_group'] = True
    yield c
    c = sample('identical_values_distinct_scores', block + 17, 8)
    c['values'][:] = c['values'][0]
    c['expect_one_value_run_per_group'] = True
    yield c
    c = sample('value_runs_cross_16_and_block', 2 * block + 17)
    run_values = rng.normal(0, .5, ((len(c['values']) + 16) // 17, kv, dimension)).astype(np.float32)
    c['values'][:] = run_values[np.arange(len(c['values'])) // 17]
    c['check_value_run_count'] = True
    yield c
    c = sample('large_signed_value_cancellation', block + 1)
    c['query'].fill(0)
    c['values'][:] = np.where(np.arange(len(c['values']))[:, None, None] % 2, -1024., 1024.)
    c['values'][-1] = .25
    yield c
    c = sample('extreme_scores_and_gates', 129, 8)
    c['query'].fill(0)
    c['query'][:, :, 0] = 4096.
    c['keys'].fill(0)
    c['keys'][:, :, 0] = np.linspace(-16., 16., len(c['keys']), dtype=np.float32)[:, None]
    c['gate'][:] = np.where(np.arange(dimension) % 2, -80., 80.)
    yield c
    c = sample('gqa_head_identity', 33, 32)
    for head in range(kv):
        c['keys'][:, head] += np.float32(head / 4.)
        c['values'][:, head] += np.float32(head * 2.)
    yield c
    c = sample('nonzero_start_and_ring_slots', 2 * block + 1, 32,
               history_start=3 * (1 << (2 * block).bit_length()) + 13)
    yield c
    c = sample('early_queries_exclude_future', block + 33, 17, history_start=19)
    c['query_start'] = c['history_start'] + 7
    c['values'][32:] += np.float32(1e4)
    yield c


def compact_cases(spec, block):
    rng=np.random.default_rng(20261111)
    dimension=spec.attention_head_dim
    for name in ('identical_packets','one_value_column','two_value_columns','one_key_column',
                 'equivalence_splits_at_block','uniform_complete_contributions',
                 'nonconsecutive_packets','cancelled_key_group'):
        count=2*block+17;rows=8
        c=dict(name='compact_'+name,query=rng.normal(0,.25,(rows,spec.attention_heads,dimension)).astype('f4'),
               keys=rng.normal(0,.25,(count,spec.kv_heads,dimension)).astype('f4'),
               values=rng.normal(0,.5,(count,spec.kv_heads,dimension)).astype('f4'),
               gate=rng.normal(0,.5,(rows,spec.attention_heads,dimension)).astype('f4'),
               query_start=count-rows,history_start=0)
        line=np.linspace(-1.,1.,count,dtype=np.float32)[:,None,None]
        if name=='identical_packets':
            c['keys'][:]=c['keys'][0];c['values'][:]=c['values'][0];c['expect_token_compaction']=True
        elif name in ('one_value_column','two_value_columns','equivalence_splits_at_block'):
            c['values'][:]=line
            if name=='one_value_column':c['expect_value_dimensions']=1
            elif name=='two_value_columns':
                c['values'][:]*=np.where(np.arange(dimension)%2,-1.,1.).astype('f4');c['expect_value_dimensions']=2
            else:c['values'][:block,:,1]*=-1
        elif name=='one_key_column':
            c['keys'][:]=line*.5;c['expect_key_dimensions']=1
        elif name=='nonconsecutive_packets':
            identity=(np.arange(count)*7)%5
            c['keys'][:]=c['keys'][:5].copy()[identity]
            c['values'][:]=c['values'][:5].copy()[identity]
            c['expect_max_unique_packets']=5
        elif name=='cancelled_key_group':
            c['keys'][:]=line*.5;c['expect_key_dimensions']=1
            c['query'][:]=np.where(np.arange(dimension)%2,-.25,.25).astype('f4')
            c['query'][:,:,0]+=np.float32(1e-6)
        else:
            c['query'].fill(0);c['keys'].fill(0);c['values'].fill(.5)
            c['expect_token_compaction']=True;c['expect_value_dimensions']=1
        yield c

    # Every sampled history position has identical V columns, but a position
    # between the samples differs. Full verification must split the columns.
    # The differing coefficient also sits outside the sampled packet words.
    q=np.zeros((1,spec.attention_heads,dimension),dtype='f4')
    k=np.zeros((block,spec.kv_heads,dimension),dtype='f4')
    v=np.full_like(k,.5)
    v[block//4,:,min(120,dimension-1)]=-.5
    yield dict(name='compact_sampled_hash_collision',query=q,keys=k,values=v,gate=q,
               query_start=block-1,history_start=0,expect_value_dimensions=2,
               expect_max_unique_packets=2,expect_packet_hash_collision=True)


def cpu_selftest():
    """Check group merge identities against a separate full softmax expression."""
    rng = np.random.default_rng(619)
    count = 257
    q = rng.normal(0, .25, (3, 4, 32))
    k = rng.normal(0, .25, (count, 2, 32))
    v = rng.normal(0, .5, (count, 2, 32))
    gate = np.zeros_like(q)
    dense = full_attention(q, k, v, gate, 254, 0, 512, False)
    for block in (128, 256, 512):
        groups = receipt_reference(dense, block, 0)
        maximum = np.full((3, 4), -np.inf)
        for group in groups:
            at = (group['row'], group['head'])
            maximum[at] = max(maximum[at], group['maximum'])
        z = np.zeros((3, 4))
        u = np.zeros((3, 4, 32))
        for group in groups:
            at = (group['row'], group['head'])
            scale = np.exp(group['maximum'] - maximum[at])
            z[at] += group['reference'] * scale
            u[at] += group['numerator'] * scale
        np.testing.assert_allclose(u / z[:, :, None], dense['output'], atol=2e-15, rtol=2e-14)
    constant = np.broadcast_to(np.arange(32)[None, None, :], v.shape).copy()
    result = full_attention(q, k, constant, gate, 254, 0, 512, False)
    np.testing.assert_allclose(result['output'], np.broadcast_to(np.arange(32), q.shape),
                               atol=2e-13, rtol=2e-14)
    return {'group_merge_FP64': True, 'constant_value_identity': True,
            'groups': (128, 256, 512), 'gpu_executed': False}


def pinned_worker(worker, spec):
    if worker is None:
        return nullcontext()
    worker = Path(worker).resolve()
    if not worker.is_file() or (worker.parent / 'complete').read_text(encoding='utf8').strip() != spec.fingerprint:
        raise ValueError('Pinned worker does not match the selected model specification')

    def build(requested, output=None):
        if requested.fingerprint != spec.fingerprint or output is not None:
            raise ValueError('Pinned verification worker cannot serve another build specification')
        return worker
    return patch('ray_tracing.compiler.build.build', side_effect=build)


def verify(asset, block=256, layer=None, worker=None, output=None):
    asset = Path(asset).resolve()
    directory = asset.parent if asset.is_file() else asset
    spec = asset_spec(directory)
    if not spec.full_layers:
        raise ValueError('Attention verification requires a full attention layer')
    layer = spec.full_layers[0] if layer is None else layer
    if layer not in spec.full_layers:
        raise ValueError('Selected layer is not a full attention layer')
    suite = list(cases(spec, block))+list(compact_cases(spec,block))
    largest = max(len(case['keys']) for case in suite)
    capacity = max(32, 1 << (largest - 1).bit_length())
    report = {'asset': str(directory), 'block': block, 'layer': layer,
              'capacity': capacity, 'seed': 20261110, 'model_spec': spec.to_dict(),
              'oracle': 'Independent gguf packet decode and dense FP64 full attention',
              'FP32_bitwise_equivalence_claimed': False,
              'cpu_selftest': cpu_selftest(), 'cases': [], 'failures': [], 'status': 'running'}

    def save_progress():
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding='utf8')

    with pinned_worker(worker, spec), ResidentRuntime(directory / 'optical_model.bin', capacity=capacity, cache=False,
                         prefill_batch=32, attention_backend='grouped_rt',
                         attention_block=block) as runtime:
        report['metadata'] = runtime.metadata
        worker = Path(runtime.metadata['worker'])
        report['worker'] = str(worker)
        report['worker_sha256'] = hashlib.sha256(worker.read_bytes()).hexdigest()
        report['shader_provenance'] = shader_provenance(worker)
        for case in suite:
            report['running_case'] = {'name': case['name'], 'history': len(case['keys']),
                                      'rows': len(case['query']), 'query_start': case['query_start'],
                                      'history_start': case['history_start']}
            save_progress()
            result = runtime.attention(case['query'], case['keys'], case['values'], case['gate'],
                                       layer=layer, query_start=case['query_start'],
                                       history_start=case['history_start'])
            key, value = decode_history(result['history_packets'], spec)
            reference = full_attention(case['query'], key, value, case['gate'],
                                       case['query_start'], case['history_start'], capacity,
                                       spec.attention_gate)
            groups = receipt_reference(reference, block, case['history_start'])
            checks, stats, merged = check_receipts(result['receipts'], groups, block,
                                                  spec.attention_heads, len(case['query']),
                                                  spec.attention_head_dim, case['gate'],
                                                  spec.attention_gate,compact=result['receipt_format']=='compact_columns')
            checks['output_FP64'] = error_metrics(result['output'], reference['output'])
            checks['output_from_GPU_summaries'] = error_metrics(result['output'], merged)
            checks['global_maximum'] = error_metrics(result['physical'][:, :, 0] - 20.,
                                                      reference['maximum'])
            checks['global_normalizer'] = error_metrics(result['physical'][:, :, 1],
                                                         reference['reference'])
            checks['merged_maximum'] = error_metrics(result['physical'][:, :, 0] - 20.,
                                                      stats.pop('maximum'))
            checks['merged_normalizer'] = error_metrics(result['physical'][:, :, 1],
                                                         stats.pop('normalizer'))
            visible = int(reference['visible'].sum())
            assert stats['represented_head_positions'] == visible * spec.attention_heads
            if result['receipt_format']=='compact_columns':
                raw=result['compact_receipts'];coverage=block//32;dh=spec.attention_head_dim
                mapping_offset=16+coverage;axes=np.arange(dh,dtype=np.uint32)
                maps=(raw[:,mapping_offset+axes//4]>>((axes%4)*8))&255
                for i,group in enumerate(groups):
                    content_bits=group['content'].astype('f4').view('u4')
                    np.testing.assert_array_equal(content_bits,content_bits[:,maps[i]],err_msg='Compressed V column is not exactly equal')
                    indices=group['row']
                    end=case['query_start']-case['history_start']+indices-group['group']*block
                    selected=result['history_packets'][end-np.arange(group['count'])]
                    unique=len(np.unique(selected,axis=0))
                    assert int(raw[i,10+coverage])==unique, 'Full packet multiplicity lost'
                    if case.get('expect_packet_hash_collision'):
                        pairs=selected.shape[1]//8
                        sampled=selected.reshape(len(selected),pairs,8)[:,np.array([0,(pairs-1)//2,pairs-1])]
                        assert unique==2 and len(np.unique(sampled.reshape(len(selected),-1),axis=0))==1
                if case.get('expect_token_compaction'):assert np.all(raw[:,10+coverage]==1)
                if case.get('expect_max_unique_packets'):assert np.all(raw[:,10+coverage]<=case['expect_max_unique_packets'])
                assert np.all(raw[:,3]<=raw[:,10+coverage]), 'Exp evaluated for a duplicate packet'
                if case.get('expect_value_dimensions'):assert np.all(raw[:,9+coverage]==case['expect_value_dimensions'])
                if case.get('expect_key_dimensions'):assert np.all(raw[:,8+coverage]==case['expect_key_dimensions'])
                work=result['shader_work']
                assert work['qk_products']<=visible*spec.attention_heads*dh
                assert work['value_products']<=visible*spec.attention_heads*dh
            if case.get('expect_one_exp_per_group'):
                assert np.all(result['receipts'][:, 3] == 1), 'Equal score run crossed a tile without sharing exp'
            if case.get('expect_one_value_run_per_group'):
                assert np.all(result['receipts'][:, 4] == 1), 'Equal V vector run crossed a tile'
            if case.get('check_value_run_count'):
                for record, group in zip(result['receipts'], groups):
                    bits = group['content'].astype(np.float32).view(np.uint32)
                    runs = 1 + int(np.count_nonzero(np.any(bits[1:] != bits[:-1], axis=1)))
                    assert int(record[4]) == runs, ('V run coverage', int(record[4]), runs)
            record = {'name': case['name'], 'history': len(case['keys']), 'rows': len(case['query']),
                      'query_start': case['query_start'], 'history_start': case['history_start'],
                      'checks': checks, 'coverage': stats,
                      'legacy_available': bool(result.get('legacy_available', True)),
                      'gpu_ms': float(result['gpu_ms']),
                      'history_packet_sha256': hashlib.sha256(result['history_packets'].tobytes()).hexdigest()}
            if result['receipt_format']=='compact_columns':
                record['shader_work']=result['shader_work'];record['actual_receipt_written_bytes']=result['receipt_written_bytes']
            if record['legacy_available']:
                record.update(legacy_output_FP64=error_metrics(result['legacy_output'], reference['output']),
                              new_vs_legacy_output=error_metrics(result['output'], result['legacy_output']),
                              legacy_maximum_FP64=error_metrics(result['legacy_physical'][:, :, 0] - 20., reference['maximum']),
                              legacy_normalizer_FP64=error_metrics(result['legacy_physical'][:, :, 1], reference['reference']),
                              legacy_gpu_ms=float(result['legacy_gpu_ms']))
            record['passed'] = all(check['passed'] for check in checks.values())
            report['cases'].append(record)
            if not record['passed']:
                report['failures'].append({'case': case['name'], 'checks': [key for key, check in checks.items() if not check['passed']]})
            save_progress()
        report['resources'] = runtime.resources()
    report['passed'] = not report['failures']
    report['check_count'] = sum(len(case['checks']) for case in report['cases'])
    report['status'] = 'complete'
    report.pop('running_case', None)
    save_progress()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=default_model())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--block', type=int, choices=(128, 256, 512), default=256)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--worker', type=Path, help='Pin every block configuration to one validated native build')
    parser.add_argument('--cpu-selftest', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = cpu_selftest() if args.cpu_selftest else verify(args.asset, args.block, args.layer, args.worker, args.output)
    except Exception as error:
        if args.output and args.output.is_file():
            failed = json.loads(args.output.read_text(encoding='utf8'))
            failed.update(status='failed', passed=False, error_type=type(error).__name__, error=str(error))
            args.output.write_text(json.dumps(failed, indent=2), encoding='utf8')
        raise
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    if args.cpu_selftest:
        print(json.dumps(report))
        return
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']),
                      'checks': report['check_count'], 'failures': report['failures']}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
