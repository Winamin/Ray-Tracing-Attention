"""Ray Tracing Attention CLI: resident Vulkan chat, profiling, GPU telemetry and session logs."""
from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import statistics
import sys
import threading
import time
import uuid

from tokenizers.decoders import DecodeStream

if __package__:
    from .prompt_encoder import LocalPromptEncoder
    from .settings import default_model, LOCAL
    from .resident_runtime import ResidentRuntime
    from .telemetry import EventLog, GpuMonitor, timestamp
    from .terminal_status import TerminalStatus
else:
    from prompt_encoder import LocalPromptEncoder
    from settings import default_model, LOCAL
    from resident_runtime import ResidentRuntime
    from telemetry import EventLog, GpuMonitor, timestamp
    from terminal_status import TerminalStatus

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_OUTPUT = default_model()
DEFAULT_MODEL = ACTIVE_OUTPUT
MAX_TOKENS = 65536
STAGES = {"projection_geometry": "投影 / 几何", "bvh_update": "BVH 增量更新",
          "lens_history_trace": "透镜 / 历史 RT 查询", "coordinate_restore": "高维坐标恢复",
          "detector": "探测器", "output_mlp": "输出 / Ray Tracing MLP", "vocabulary": "词表总计",
          "vocabulary_rt": "  词表 RT 查询", "vocabulary_coefficients": "  词表系数探测",
          "vocabulary_reduce": "  词表归约"}
HELP = """直接输入文字开始聊天；/paste 输入多行，以 /end 结束。
/fx                     展开最近一轮性能和当前设备详情
/status                 GPU、RT、速度、窗口与缓存状态
/memory                 设备分配、历史树、光路包、工作区与缓存
/rt                     当前 RT 配置和历史独立射线基准
/profile on|off          分阶段 GPU 计时（on 时执行计算路径）
/tokens N  /batch N      输出预算 / 每次 GPU 提交的 token 数
/retry                  用相同前缀重新回答
/continue [N]           从当前 GPU 终态继续生成
/bench [N]              重放最近请求，逐轮统计普通 / 缓存速度
/clear                  清空对话
/system 文本            设置系统提示并开始新对话
/save [路径] /load 路径  保存 / 载入会话（JSON）
/log                    显示本次日志路径
/help /quit             命令帮助 / 退出"""


def rate(tokens, milliseconds):
    return tokens * 1000 / milliseconds if milliseconds > 0 else None


def number(value, suffix="", digits=2):
    return "—" if value is None else f"{value:,.{digits}f}{suffix}"


def mib(value):
    return number(value / 1048576, " MiB")


class Console:
    def __init__(self, args):
        self.args = args
        self.machine = args.json or args.jsonl
        self.interactive = not self.machine and sys.stdin.isatty()
        self.lock = threading.RLock()
        self.footer = None

    def start_status(self, monitor):
        if self.interactive and not self.args.no_live:
            self.footer = TerminalStatus(monitor, self.lock)
            self.footer.start()

    def close(self):
        if self.footer:
            self.footer.close()

    def welcome(self):
        if self.interactive:
            with self.lock:
                print("\n  RT Chat\n  /fx 查看性能 · /help 帮助\n", flush=True)

    def begin_reply(self):
        if self.interactive:
            with self.lock:
                print("\nRT › ", end="", flush=True)

    def panel(self, title, rows):
        if self.machine:
            return
        with self.lock:
            print(f"\n── {title} ──", file=sys.stderr)
            for row in rows:
                print(row, file=sys.stderr)
            sys.stderr.flush()

    def event(self, event, **data):
        if self.args.jsonl:
            with self.lock:
                print(json.dumps({"event": event, **data}, ensure_ascii=False, allow_nan=False), flush=True)

    def chunk(self, text, ids):
        self.event("tokens", text=text, ids=ids)
        if not self.machine and not self.args.no_stream:
            with self.lock:
                print(text, end="", flush=True)
        if self.footer:
            self.footer.draw()

    def reply(self, report):
        if self.args.json:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        elif self.args.jsonl:
            self.event("reply", **report)
        else:
            with self.lock:
                if self.args.no_stream:
                    print(report["text"], end="", flush=True)
                spacing = "" if report["text"].endswith("\n\n") else "\n" if report["text"].endswith("\n") else "\n\n"
                print(spacing, end="", flush=True)

    def performance(self, report):
        if self.machine:
            return
        speed = report["speed"]
        rows = [f"{report['tokens']} token · {report['finish_reason']} · "
                    f"普通 GPU {number(speed['compute_gpu_tps'], ' t/s')} · "
                    f"缓存 GPU {number(speed['cache_gpu_tps'], ' t/s')}",
                    f"生成 {number(speed['generation_wall_tps'], ' t/s')} · "
                    f"请求端到端 {number(speed['request_tps'], ' t/s')} · "
                    f"首批 {number(report['first_batch_ms'], ' ms')} · 首文字 {number(report['first_text_ms'], ' ms')}",
                    f"Prefill GPU {number(report['prefill'].get('prefill_gpu_ms'), ' ms')} · "
                    f"生成 GPU {report['generation_gpu_ms']:.3f} ms · 请求 {report['request_wall_ms']:.2f} ms",
                    f"窗口 {report['input_used_tokens']}/{report['capacity']} · 截断 {report['input_truncated_tokens']} · "
                    f"前缀复用 {report['prefill'].get('cached_prefix_tokens', 0)} token · "
                    f"回复缓存 {report['cached_tokens']} token / {report['reply_cache_hits']} 批次",
                    f"RT 回执：MLP {report['rt']['mlp_feature_hits']:,}/{report['rt']['mlp_possible_features']:,} · "
                    f"词表 {report['rt']['vocabulary_hits']:,} · 缓存树命中 {report['rt']['tree_cache_hits']:,}",
                    f"提交 / 等待 {report['submissions']}/{report['waits']} · "
                    f"设备分配 {mib(report['resources']['device_allocation_bytes'])} · "
                    f"本轮增量 {mib(report['device_growth_bytes'])}"]
        if report.get('rt_query_gpu_ms') is not None:
            rows.append(f"RT 查询 GPU {report['rt_query_gpu_ms']:.3f} ms · 查询占时 "
                        f"{report['rt_query_gpu_ms']/report['generation_gpu_ms']*100:.2f}%")
        if 'mlp_route_cached_rows' in report['rt']:
            rows.append(f"MLP 层求交结果：缓存复用 {report['rt']['mlp_route_cached_rows']:,} 次 · "
                        f"新计算 {report['rt']['mlp_route_rt_rows']:,} 次")
        if (cache := report.get('cache')) and cache['capacity']:
            rows.append(f"光路缓存 {cache['nodes']:,}/{cache['capacity']:,} 节点 · 命中后每 token 一条查询光线")
            if cache['state_recovery_tokens'] or cache['checkpoint_restores']:
                rows.append(f"缓存状态恢复 {cache['state_recovery_gpu_ms']:.3f} ms · "
                            f"检查点 {cache['checkpoint_restores']} 次 · {cache['state_recovery_tokens']} token")
        if 'history_cache_boundaries' in report['resources']:
            history = report['prefill']
            rows.append(f"共享历史 BVH {report['resources']['history_cache_boundaries']} 个边界 · "
                        f"接入 {history.get('history_shared_blocks', 0)} 个历史块 · "
                        f"新增计算 {history.get('prefill_computed_tokens', 0)} token · "
                        f"查询 {history.get('history_lookup_rays', 0)} 条光线")
            rows.append(f"共享边界状态 {mib(report['resources']['history_boundary_state_bytes'])} · "
                        f"共享光路块 {mib(report['resources']['history_shared_block_bytes'])}")
        self.panel("本轮性能", rows)
        if report["profile_gpu_ms"]:
            self.panel("GPU 阶段计时（词表子项包含在词表总计内）", [
                f"{label:18} {value:9.3f} ms  {value / report['generation_gpu_ms'] * 100:5.1f}%"
                for key, label in STAGES.items() if (value := report["profile_gpu_ms"].get(key)) is not None
                and report["generation_gpu_ms"] > 0])


class ChatSession:
    def __init__(self, native, encoder, args, console, monitor, log):
        self.native, self.encoder, self.args = native, encoder, args
        self.console, self.monitor, self.log = console, monitor, log
        self.messages = [{"role": "system", "content": args.system}] if args.system else []
        self.last = self.last_context = None
        self.can_continue = False
        self.coordinate_pairs_total = 0
        self.loaded_at = timestamp()
        self.resources = native.resources()
        self.initial_resources = dict(self.resources)
        self.stats = {"requests": 0, "tokens": 0, "cached_tokens": 0, "generation_gpu_ms": 0.,
                      "compute_tokens": 0, "compute_gpu_ms": 0., "cache_gpu_ms": 0.}

    def generation_results(self, maximum):
        remaining = maximum
        per_token = not self.args.no_stream and not self.args.json
        while remaining:
            count = min(self.args.batch, remaining)
            if per_token:
                iterator = iter(self.native.stream_generate(count, profile=self.args.profile, telemetry=True))
            else:
                iterator = (self.native.generate(count, profile=self.args.profile, telemetry=True) for _ in range(1))
            while True:
                started = time.perf_counter()
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                result['host_receive_ms'] = (time.perf_counter()-started)*1000
                yield result
                remaining -= len(result['ids'])
                if not result['control'][2] or not len(result['ids']):
                    return

    def generate(self, context=None, maximum=None):
        maximum = maximum or self.args.new_tokens
        if context is None and not self.can_continue:
            raise ValueError("当前回复已经结束；输入新消息或使用 /retry。")
        self.console.begin_reply()
        start = time.perf_counter()
        before = self.native.resources()
        growth_start = len(self.native.growth_events)
        prefill = {}
        input_tokens = self.last["input_tokens"] if self.last else 0
        used = min(input_tokens, self.native.capacity)
        if context is not None:
            self.can_continue = False
            self.monitor.progress("prefill")
            encoded, checkpoint_boundaries = self.encoder.encode_with_boundaries(context)
            input_tokens = len(encoded)
            prefill = self.native.prefill(encoded, self.encoder.stop_ids,profile=self.args.profile,
                                         checkpoint_boundaries=checkpoint_boundaries)
            used = min(input_tokens, self.native.capacity)
            self.coordinate_pairs_total = int(prefill['control'][7])
        self.log.write("request_start", input_tokens=input_tokens, input_used_tokens=used,
                       budget=maximum, continuation=context is None, profile=self.args.profile)
        self.console.event("request_start", input_tokens=input_tokens, input_used_tokens=used,
                           input_truncated_tokens=input_tokens-used, profile=self.args.profile)
        ids, fragments, profiles = [], [], {}
        gpu_ms = wait_ms = compute_ms = cache_ms = 0.
        cached_tokens = cache_hits = submissions = waits = host_bytes = 0
        cache_state_ms = 0.
        cache_state_tokens = checkpoint_restores = 0
        rt = dict(mlp_feature_hits=0, mlp_possible_features=0, vocabulary_hits=0,
                  tree_cache_hits=0, tree_cache_queries=0, context_coordinate_decoded_pairs=0,
                  cache_candidates=0, cache_rejected_candidates=0,
                  mlp_route_rt_rows=0, mlp_route_cached_rows=0)
        first_batch = first_text = None
        rt_query_ms = 0.
        rt_query_measured = False
        rt_query_parts = {}
        generation_start = time.perf_counter()
        decoder = DecodeStream(skip_special_tokens=True)
        result = None
        try:
            self.monitor.progress('decode')
            for result in self.generation_results(maximum):
                self.monitor.progress("decode", len(ids))
                received_at = time.perf_counter()
                batch_ids = result["ids"].tolist()
                ids.extend(batch_ids)
                gpu_ms += result["gpu_ms"]
                wait_ms += result["submit_wait_ms"]
                submissions += result["generation_submissions"]
                waits += result["generation_waits"]
                host_bytes += result["intermediate_host_bytes"]
                hits = result.get('cached_tokens', len(batch_ids) if result['reply_cache_hit'] else 0)
                cache_hits += int(hits > 0)
                cached_tokens += hits
                cache_ms += result.get('cache_gpu_ms', result['gpu_ms'] if result['reply_cache_hit'] else 0.)
                compute_ms += result.get('compute_gpu_ms', result['gpu_ms'] if not result['reply_cache_hit'] else 0.)
                cache_state_ms += result.get('cache_state_gpu_ms', 0.)
                cache_state_tokens += result.get('cache_state_tokens', 0)
                checkpoint_restores += int(result.get('cache_checkpoint_restored', False))
                for key in rt:
                    if key != 'context_coordinate_decoded_pairs':
                        rt[key] += result.get(key, 0)
                total_pairs = result['context_coordinate_decoded_pairs']
                if not result['reply_cache_hit']:
                    rt['context_coordinate_decoded_pairs'] += max(0, total_pairs-self.coordinate_pairs_total)
                self.coordinate_pairs_total = total_pairs
                for key, value in result.get("profile_gpu_ms", {}).items():
                    profiles[key] = profiles.get(key, 0.) + value
                if 'rt_query_gpu_ms' in result:
                    rt_query_measured = True
                    rt_query_ms += result['rt_query_gpu_ms']
                    for key, value in result['rt_query_kind_gpu_ms'].items():
                        rt_query_parts[key] = rt_query_parts.get(key, 0.) + value
                self.monitor.token(result, len(ids))
                if batch_ids and first_batch is None:
                    first_batch = (received_at-start)*1000
                text = "".join(piece for token in batch_ids
                               if (piece := decoder.step(self.encoder.tokenizer, token)) is not None)
                fragments.append(text)
                if text and first_text is None:
                    first_text = (time.perf_counter()-start)*1000
                self.console.chunk(text, batch_ids)
                self.log.write("batch", tokens=len(batch_ids), total_tokens=len(ids), gpu_ms=result["gpu_ms"],
                               host_ms=result['host_receive_ms'], cache_hit=result["reply_cache_hit"],
                               cached_tokens=hits,cache_nodes=result.get('cache_nodes'),
                               cache_gpu_ms=result.get('cache_gpu_ms'),
                               cache_state_gpu_ms=result.get('cache_state_gpu_ms'),
                               cache_checkpoint_restored=result.get('cache_checkpoint_restored'),
                               rt_query_gpu_ms=result.get('rt_query_gpu_ms'),
                               rt_query_kind_gpu_ms=result.get('rt_query_kind_gpu_ms', {}),
                               rt={key: result.get(key, 0) for key in rt}, profile_gpu_ms=result.get("profile_gpu_ms", {}))
                self.monitor.progress("decode", len(ids))
                if not result["control"][2] or not batch_ids:
                    break
        except BaseException:
            self.can_continue = False
            partial = self.encoder.tokenizer.decode(ids, skip_special_tokens=True)
            self.log.write("request_interrupted", tokens=len(ids), text=partial)
            if partial:
                if context is not None:
                    self.messages = [dict(m) for m in context] + [{"role": "assistant", "content": partial}]
                elif self.messages and self.messages[-1]['role'] == 'assistant':
                    self.messages[-1]['content'] += partial
            raise
        history_output = self.native.finish_history()
        generation_ms = (time.perf_counter()-generation_start)*1000
        request_ms = (time.perf_counter()-start)*1000
        full_text = self.encoder.tokenizer.decode(ids, skip_special_tokens=True)
        streamed = "".join(fragments)
        if full_text.startswith(streamed) and len(full_text) > len(streamed):
            self.console.chunk(full_text[len(streamed):], [])
        self.resources = self.native.resources()
        self.monitor.progress("idle", len(ids))
        self.can_continue = bool(result is not None and result["control"][2] and len(ids))
        rt['mlp_feature_coverage'] = (rt['mlp_feature_hits']/rt['mlp_possible_features']
                                      if rt['mlp_possible_features'] else None)
        report = {"text": full_text, "generated_ids": ids, "tokens": len(ids), "capacity": self.native.capacity,
                  "input_tokens": input_tokens, "input_used_tokens": used, "input_truncated_tokens": input_tokens-used,
                  "continuation": context is None, "finish_reason": "budget" if self.can_continue else "stop",
                  "first_batch_ms": first_batch, "first_text_ms": first_text, "prefill": {
                      key: value for key, value in prefill.items() if key != "control"},
                  "generation_gpu_ms": gpu_ms, "generation_submit_wait_ms": wait_ms,
                  "generation_wall_ms": generation_ms, "request_wall_ms": request_ms,
                  "cached_tokens": cached_tokens, "reply_cache_hits": cache_hits,
                  "compute_tokens": len(ids)-cached_tokens, "compute_gpu_ms": compute_ms, "cache_gpu_ms": cache_ms,
                  "speed": {"compute_gpu_tps": rate(len(ids)-cached_tokens, compute_ms),
                            "cache_gpu_tps": rate(cached_tokens, cache_ms), "aggregate_gpu_tps": rate(len(ids), gpu_ms),
                            "generation_wall_tps": rate(len(ids), generation_ms), "request_tps": rate(len(ids), request_ms)},
                  "rt": rt, "profile_gpu_ms": profiles, "profile_enabled": self.args.profile,
                  "rt_query_gpu_ms": rt_query_ms if rt_query_measured else None,
                  "rt_query_kind_gpu_ms": rt_query_parts,
                  "stream_mode": 'per_token' if not self.args.no_stream and not self.args.json else 'batch',
                  "submissions": submissions, "waits": waits, "intermediate_host_bytes": host_bytes,
                  "absolute_position": result["absolute_position"] if result else 0,
                  "resources": self.resources,
                  "device_growth_bytes": self.resources["device_allocation_bytes"]-before["device_allocation_bytes"],
                  "gpu": self.monitor.snapshot(), "sampled_peaks": self.monitor.since(start),
                  "metadata": self.native.metadata}
        report['tree_growth'] = self.native.growth_events[growth_start:]
        report['history_output'] = history_output
        report['cache'] = {'nodes': result.get('cache_nodes', 0) if result else 0,
                           'capacity': result.get('cache_capacity', 0) if result else 0,
                           'state_recovery_gpu_ms': cache_state_ms,
                           'state_recovery_tokens': cache_state_tokens,
                           'checkpoint_restores': checkpoint_restores}
        self.last = report
        self.stats["requests"] += 1
        for key in ("tokens", "cached_tokens", "generation_gpu_ms", "compute_tokens", "compute_gpu_ms", "cache_gpu_ms"):
            self.stats[key] += report[key]
        self.log.write("reply", **report)
        self.console.reply(report)
        return report

    def ask(self, text):
        self.messages.append({"role": "user", "content": text})
        self.last_context = [dict(m) for m in self.messages]
        report = self.generate(self.last_context)
        self.messages.append({"role": "assistant", "content": report["text"]})
        return report

    def retry(self):
        if self.last_context is None:
            raise ValueError("先输入一条消息。")
        report = self.generate(self.last_context)
        self.messages = [dict(m) for m in self.last_context] + [{"role": "assistant", "content": report["text"]}]
        return report

    def continue_reply(self, count=None):
        report = self.generate(maximum=count)
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"] += report["text"]
        return report

    def save(self, path):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "created_at": self.loaded_at, "saved_at": timestamp(),
                "messages": self.messages, "settings": {"capacity": self.native.capacity,
                "batch": self.args.batch, "new_tokens": self.args.new_tokens, "profile": self.args.profile},
                "statistics": self.stats, "last_reply": self.last}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        temporary.replace(path)
        self.log.write("session_saved", path=str(path))
        return str(path)

    def load(self, path):
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
        messages = data.get("messages")
        if not isinstance(messages, list) or any(not isinstance(m, dict) or
                m.get("role") not in ("system", "user", "assistant") or
                not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("会话需要 messages 列表，每条包含 role 和 content。")
        self.messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        self.last_context = None
        self.can_continue = False
        self.last = None
        self.log.write("session_loaded", path=str(Path(path).resolve()), messages=len(messages))

    def status(self, memory=False):
        self.resources = self.native.resources()
        data = {"worker_pid": self.native.process.pid, "worker_alive": self.native.process.poll() is None,
                "backend": "resident Vulkan", "metadata": self.native.metadata, "gpu": self.monitor.snapshot(),
                "resources": self.resources, "statistics": self.stats, "last_reply": self.last}
        self.log.write("status", **{key: val for key, val in data.items() if key != "last_reply"})
        if self.console.machine:
            if self.args.jsonl:
                self.console.event("status", **data)
            else:
                print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)
            return data
        rows = [f"Vulkan worker PID {data['worker_pid']} · "
                f"{self.native.metadata['depth']} 层 · {self.native.metadata['routing']} · "
                f"{'Tree 槽位' if self.native.growable else '窗口'} {self.native.capacity} · 批次 {self.args.batch} · "
                f"回复缓存 {'开' if self.native.metadata['state_cache'] else '关'} · "
                f"阶段计时 {'开' if self.args.profile else '关'}"]
        for gpu in data["gpu"]["gpus"]:
            rows.extend([gpu["name"], f"显存 {number(gpu['memory_used_mib'], ' MiB', 0)} / "
                         f"{number(gpu['memory_total_mib'], ' MiB', 0)} · GPU 利用率 "
                         f"{number(gpu['gpu_utilization_percent'], '%', 0)} · "
                         f"{number(gpu['temperature_c'], '°C', 0)} · {number(gpu['power_w'], ' W')}",
                         f"SM {number(gpu['sm_clock_mhz'], ' MHz', 0)} · "
                         f"显存时钟 {number(gpu['memory_clock_mhz'], ' MHz', 0)} · {gpu['pstate']}"])
        if not data["gpu"]["available"]:
            rows.append("GPU 传感器：" + data["gpu"].get("reason", "无数据"))
        r = self.resources
        rows.extend([f"RT 状态：{'硬件查询 + 命中回执' if self.native.metadata['routing'] == 'rt_full' else self.native.metadata['routing']}",
                     f"Prefill 并行位置 {self.native.metadata['prefill_batch']} · 权重共享 {self.native.metadata['matrix_token_tile']} token",
                     f"历史 BVH {r['history_blas_count']} BLAS / {r['history_tlas_count']} TLAS · "
                     f"词表查询 {r['word_rays']:,} 条 · 每块词 / 特征 {r['word_tile']}/{r['feature_tile']}",
                     f"累计 {self.stats['requests']} 次请求 / {self.stats['tokens']} token · "
                     f"普通 GPU {number(rate(self.stats['compute_tokens'], self.stats['compute_gpu_ms']), ' t/s')} · "
                     f"缓存 GPU {number(rate(self.stats['cached_tokens'], self.stats['cache_gpu_ms']), ' t/s')}"])
        self.console.panel("Ray Tracing Attention 状态", rows)
        if self.last:
            last = self.last
            self.console.panel("最近请求", [f"{last['tokens']} token · "
                f"生成 {number(last['speed']['generation_wall_tps'], ' t/s')} · "
                f"请求 {number(last['speed']['request_tps'], ' t/s')} · "
                f"首文字 {number(last['first_text_ms'], ' ms')}",
                f"MLP 命中覆盖 {number((last['rt']['mlp_feature_coverage'] or 0)*100, '%')} · "
                f"词表回执 {last['rt']['vocabulary_hits']:,} · "
                f"缓存 {last['cached_tokens']} token"])
        if memory:
            fields = {"device_allocation_bytes": "原生设备分配", "vulkan_allocation_bytes": "Vulkan 总分配",
                      "packed_coefficient_bytes": "固定传输系数", "history_path_packet_bytes": "历史光路包",
                      "history_scene_bytes": "历史 BVH 场景", "word_scene_bytes": "词表场景",
                      "optical_field_bytes": "递推光场", "cache_snapshot_bytes": "缓存高维状态行",
                      "reply_tree_bytes": "缓存树", "hit_receipt_bytes": "历史命中回执",
                      "feature_receipt_bytes": "特征回执 / 工作区", "upload_staging_bytes": "上传暂存",
                      "host_readback_bytes": "主机回读", "debug_audit_bytes": "诊断快照"}
            self.console.panel("显存明细（各项按原生接口分类）", [
                f"{name:18} {mib(r[key])}" for key, name in fields.items()] + [
                f"设备分配相对加载增长 {mib(r['device_allocation_bytes']-self.initial_resources['device_allocation_bytes'])}",
                f"固定工作区 {r['workspace_rows']} 行 · 设备内存池 {r['device_memory_pools']}"])
        return data

    def rt_status(self):
        data = {"metadata": self.native.metadata, "resources": self.native.resources(),
                "latest_profile_gpu_ms": self.last["profile_gpu_ms"] if self.last else {}}
        if self.console.machine:
            if self.args.jsonl:
                self.console.event("rt_status", **data)
            else:
                print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)
        else:
            self.status()
            if data['latest_profile_gpu_ms']:
                self.console.panel("最近请求 RT 阶段", [
                    f"{STAGES[key]} {value:.3f} ms" for key, value in data['latest_profile_gpu_ms'].items()
                    if key in ('lens_history_trace', 'vocabulary_rt', 'bvh_update')])
        return data

    def benchmark(self, count=3):
        if self.last_context is None:
            raise ValueError("先输入一条消息，再运行 /bench。")
        original_ids = self.last['generated_ids'] if self.last else None
        reports = [self.retry() for _ in range(count)]
        data = {"runs": count, "token_outputs_equal": all(
            r["generated_ids"] == original_ids for r in reports),
            "median_request_tps": statistics.median(r["speed"]["request_tps"] for r in reports),
            "compute_runs": sum(r["compute_tokens"] > 0 for r in reports),
            "cached_runs": sum(r["cached_tokens"] > 0 for r in reports)}
        self.log.write("benchmark", **data)
        if self.args.jsonl:
            self.console.event("benchmark", **data)
        elif self.args.json:
            print(json.dumps({"benchmark": data}, ensure_ascii=False), flush=True)
        else:
            self.console.panel("重放统计", [f"{count} 轮 · 输出一致 {data['token_outputs_equal']} · "
                f"计算 {data['compute_runs']} 轮 / 缓存 {data['cached_runs']} 轮 · "
                f"请求中位数 {data['median_request_tps']:.2f} t/s"])

    def command(self, text, default_session):
        command, _, arg = text.strip().partition(" ")
        arg = arg.strip()
        if command in ("/quit", "/exit"):
            return False
        if command == "/fx":
            if self.last:
                self.console.performance(self.last)
            self.status(memory=True)
        elif command == "/help":
            self.console.panel("命令", HELP.splitlines())
        elif command in ("/status", "/memory"):
            self.status(command == "/memory")
        elif command == "/rt":
            self.rt_status()
        elif command in ("/clear", "/system"):
            self.messages = [{"role": "system", "content": arg}] if command == "/system" and arg else []
            self.can_continue = False
            self.last = self.last_context = None
            self.console.panel("对话已重置", [])
        elif command in ("/tokens", "/batch"):
            value = int(arg)
            if not 1 <= value <= (64 if command == "/batch" else MAX_TOKENS):
                raise ValueError(f"/batch 范围 1..64；/tokens 范围 1..{MAX_TOKENS}。")
            setattr(self.args, "batch" if command == "/batch" else "new_tokens", value)
            self.console.panel("生成设置", [f"预算 {self.args.new_tokens} · 批次 {self.args.batch}"])
        elif command == "/profile":
            if arg not in ("on", "off"):
                raise ValueError("用法：/profile on|off")
            self.args.profile = arg == "on"
            self.console.panel("阶段计时", ["已开启，执行计算路径" if self.args.profile else "已关闭，启用缓存路径"])
        elif command == "/retry":
            self.retry()
        elif command == "/continue":
            self.continue_reply(positive_int(arg) if arg else None)
        elif command == "/bench":
            self.benchmark(bounded_int(arg, 1, 100) if arg else 3)
        elif command == "/save":
            path = self.save(arg.strip('"') if arg else default_session)
            self.console.panel("会话已保存", [path])
            self.console.event("session_saved", path=path)
        elif command == "/load":
            if not arg:
                raise ValueError("用法：/load 路径")
            self.load(arg.strip('"'))
            self.console.panel("会话已载入", [f"{len(self.messages)} 条消息"])
        elif command == "/log":
            self.console.panel("日志", [str(self.log.path) if self.log.path else "已关闭日志"])
            self.console.event("log", path=str(self.log.path) if self.log.path else None)
        elif command == "/paste":
            lines = []
            while (line := input()) != "/end":
                lines.append(line)
            if lines:
                self.ask("\n".join(lines))
        else:
            raise ValueError(f"未知命令 {command}，输入 /help 查看命令。")
        return True


def bounded_int(value, low, high):
    value = int(value)
    if not low <= value <= high:
        raise argparse.ArgumentTypeError(f"范围 {low}..{high}")
    return value


def positive_int(value):
    return bounded_int(value, 1, MAX_TOKENS)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--asset", type=Path, default=ACTIVE_OUTPUT / "optical_model.bin")
    p.add_argument("--model", type=Path, help="分词器目录；默认使用 RT 资产目录中的分词器")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="单次提示；- 从 stdin 读取全文")
    source.add_argument("--prompt-file", type=Path, help="读取 UTF-8 提示文件")
    p.add_argument("--system", default="", help="系统提示")
    p.add_argument("--new-tokens", "--max-tokens", type=positive_int, default=MAX_TOKENS)
    p.add_argument("--batch", type=lambda v: bounded_int(v, 1, 64), default=32, help="每次原生请求预算；流式模式逐 token 推送")
    p.add_argument("--capacity", type=lambda v: bounded_int(v, 1, 1048576), default=512,
                   help="历史 Tree 初始槽位；按需扩容并保留完整历史")
    p.add_argument("--prefill-batch", type=int, choices=(32,64,128,256,512,1024), default=512,
                   help="Prefill 并行位置数；按绝对位置批量写入和查询 Tree")
    p.add_argument("--prefill-matrix-rows", type=int, choices=(4,8,16), default=8,
                   help="共享一组量化权重的 token 数")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile", action="store_true", help="启用阶段 GPU 时间戳；执行计算路径")
    p.set_defaults(routing="rt_full",aperture=.08)
    p.add_argument("--word-tile", type=int, choices=(8, 16, 32, 64, 128, 256), default=32)
    p.add_argument("--feature-tile", type=int, choices=(8, 16, 32, 64, 128, 256), default=8)
    p.add_argument("--readout-unroll", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--readout-words", type=int, choices=(4, 8), default=4)
    p.add_argument("--no-stream", action="store_true", help="生成完成后一次输出")
    p.add_argument("--no-live", action="store_true", help="隐藏底部实时状态行")
    output = p.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="每次回复输出一个 JSON 对象")
    output.add_argument("--jsonl", action="store_true", help="输出结构化流式事件")
    p.add_argument("--status-only", action="store_true", help="加载后输出状态和显存明细，再退出")
    p.add_argument("--benchmark", type=lambda v: bounded_int(v, 1, 100), default=0, help="重放提示的次数")
    p.add_argument("--load", type=Path, help="载入会话")
    p.add_argument("--session", type=Path, help="退出时保存会话的路径")
    p.add_argument("--log", type=Path, help="JSONL 日志路径")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--sample-interval", type=float, default=.2, help="GPU 传感器采样间隔，秒（至少 0.2）")
    return p


def main(argv=None):
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf8")
    p = parser()
    args = p.parse_args(argv)
    if not math.isfinite(args.sample_interval) or args.sample_interval < .2 or not 0 < args.aperture <= 2:
        p.error("采样间隔至少 0.2 秒；aperture 范围 (0,2]。")
    if args.benchmark and args.prompt is None and args.prompt_file is None:
        p.error("--benchmark 需要 --prompt 或 --prompt-file。")
    directory = LOCAL / "sessions" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    log = EventLog(None if args.no_log else args.log or directory / "events.jsonl")
    session_path = args.session or (log.path.parent / "session.json" if log.path else None)
    console = Console(args)
    monitor = GpuMonitor(args.sample_interval, log)
    native = session = None
    code = 0
    start = time.perf_counter()
    try:
        if not args.asset.is_file():
            raise FileNotFoundError(f"RT 资产不存在：{args.asset}")
        encoder_dir = args.model or (args.asset.parent if (args.asset.parent / "tokenizer.json").exists() else DEFAULT_MODEL)
        encoder = LocalPromptEncoder(encoder_dir)
        monitor.start()
        console.start_status(monitor)
        native = ResidentRuntime(args.asset, capacity=args.capacity, cache=args.cache, routing=args.routing,
                                 aperture=args.aperture, word_tile=args.word_tile, feature_tile=args.feature_tile,
                                 readout_unroll=args.readout_unroll, readout_words=args.readout_words,
                                 prefill_batch=args.prefill_batch,matrix_rows=args.prefill_matrix_rows)
        session = ChatSession(native, encoder, args, console, monitor, log)
        monitor.progress("idle")
        monitor.sample()
        load_ms = (time.perf_counter()-start)*1000
        log.write("runtime_loaded", load_wall_ms=load_ms, metadata=native.metadata,
                  resources=session.resources, gpu=monitor.snapshot())
        console.event("ready", load_wall_ms=load_ms, metadata=native.metadata, resources=session.resources,
                      log_path=str(log.path) if log.path else None)
        if args.load:
            session.load(args.load)
        if args.status_only:
            session.status(memory=True)
        elif args.prompt is not None or args.prompt_file:
            prompt = (args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else
                      sys.stdin.read() if args.prompt == "-" else args.prompt)
            session.ask(prompt)
            if args.benchmark:
                session.benchmark(args.benchmark)
        else:
            console.welcome()
            while True:
                try:
                    text = input("你 › " if console.interactive else "")
                except EOFError:
                    break
                if not text.strip():
                    continue
                try:
                    if text.startswith("/"):
                        if not session.command(text, session_path or Path("rt_session.json")):
                            break
                    else:
                        session.ask(text)
                except (ValueError, OSError, argparse.ArgumentTypeError) as error:
                    log.write("command_error", message=str(error))
                    if console.machine:
                        print(json.dumps({"event": "error", "message": str(error)}, ensure_ascii=False), flush=True)
                    else:
                        console.panel("输入提示", [str(error)])
    except KeyboardInterrupt:
        code = 130
        log.write("interrupted")
        console.panel("已停止", ["正在保存会话并关闭 Vulkan worker。"])
    except (OSError, RuntimeError, ValueError) as error:
        code = 1
        log.write("error", message=str(error))
        if console.machine:
            print(json.dumps({"event": "error", "message": str(error)}, ensure_ascii=False), flush=True)
        else:
            console.panel("运行错误", [str(error)])
    finally:
        console.close()
        if session and session_path:
            try:
                session.save(session_path)
            except OSError as error:
                print(f"会话保存失败：{error}", file=sys.stderr)
                code = code or 1
        if native:
            native.close()
        monitor.close()
        log.write("closed", exit_code=code, session_path=str(session_path) if session_path else None)
        log.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
