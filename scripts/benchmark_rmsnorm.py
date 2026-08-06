"""Interleaved CUDA Graph and Python-dispatch timing for RMSNorm."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
from gpu_ops import rmsnorm
from reference.rmsnorm import rmsnorm_reference
from scripts.check_env import collect_environment


def capture(fn, x, weight, iterations):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            fn(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            output = fn(x, weight)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output


def run(args):
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required.')
    path = Path(args.output)
    if path.exists():
        raise SystemExit('Choose a new output filename to preserve earlier measurements.')
    torch.manual_seed(20261015)
    rng = random.Random(20261015)
    methods = {'pytorch_eager': rmsnorm_reference, 'cuda_shared': rmsnorm}
    if args.include_warp:
        from gpu_ops import rmsnorm_warp
        methods['cuda_warp'] = rmsnorm_warp
    if args.include_specialized:
        from gpu_ops import rmsnorm_cached, rmsnorm_rows, rmsnorm_vector, rmsnorm_reload, rmsnorm_vector128
        methods.update(cuda_cached=rmsnorm_cached, cuda_rows=rmsnorm_rows, cuda_vector=rmsnorm_vector,
                       cuda_reload=rmsnorm_reload, cuda_vector128=rmsnorm_vector128)
    if args.include_compiled:
        methods['pytorch_compiled'] = torch.compile(rmsnorm_reference, fullgraph=True, dynamic=False)
    if args.methods:
        unknown = set(args.methods) - methods.keys()
        if unknown:
            raise SystemExit(f"Unknown methods: {unknown}")
        methods = {name: methods[name] for name in args.methods}
    root = Path(__file__).resolve().parents[1]
    sources = ['csrc/kernels.cu', 'csrc/bindings.cpp', 'gpu_ops/ops.py', 'gpu_ops/extension.py',
               'reference/rmsnorm.py', 'scripts/benchmark_rmsnorm.py', 'csrc/rmsnorm_specialized.cu']
    payload = {'environment': collect_environment(), 'seed': 20261015,
               'iterations_per_replay': args.iterations, 'repeats': args.repeats,
               'measurement': 'CUDA event timing of a captured repeated sequence; compilation and Python launch excluded',
               'wall_measurement': 'Synchronized ordinary Python calls with output allocation; three interleaved loops',
               'cache_regime': 'Repeated same input/weight buffers; warm-cache microbenchmark, not streaming DRAM bandwidth',
               'source_hashes': {s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in sources},
               'cases': []}
    for rows in args.rows:
        x = torch.randn(rows, args.width, device='cuda', dtype=getattr(torch, args.dtype))
        weight = torch.randn(args.width, device='cuda', dtype=x.dtype)
        expected = rmsnorm_reference(x, weight)
        tolerance = {'rtol': 2e-2, 'atol': 2e-2} if args.dtype == 'bfloat16' else {'rtol': 1e-5, 'atol': 2e-6}
        case = {'rows': rows, 'width': args.width, 'dtype': args.dtype, 'methods': {}, 'execution_orders': []}
        graphs = {}
        for name, fn in methods.items():
            start = time.perf_counter()
            actual = fn(x, weight)
            torch.cuda.synchronize()
            first_ms = (time.perf_counter() - start) * 1000
            torch.testing.assert_close(actual, expected, **tolerance)
            graphs[name] = capture(fn, x, weight, args.iterations)
            case['methods'][name] = {'first_call_ms': first_ms, 'graph_us_samples': [], 'wall_us_samples': [],
                                     'max_abs_error': (actual.float() - expected.float()).abs().max().item()}
        for _ in range(args.repeats):
            order = list(methods)
            rng.shuffle(order)
            case['execution_orders'].append(order)
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name][0].replay()
                end.record()
                end.synchronize()
                case['methods'][name]['graph_us_samples'].append(start.elapsed_time(end) * 1000 / args.iterations)
        for _ in range(3):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    actual = methods[name](x, weight)
                torch.cuda.synchronize()
                case['methods'][name]['wall_us_samples'].append((time.perf_counter() - start) * 1e6 / args.iterations)
        for entry in case['methods'].values():
            entry['graph_us_median'] = statistics.median(entry['graph_us_samples'])
            entry['wall_us_median'] = statistics.median(entry['wall_us_samples'])
        payload['cases'].append(case)
        print('Rows', rows, {name: round(e['graph_us_median'], 3) for name, e in case['methods'].items()}, flush=True)
        del graphs
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', nargs='+', type=int, default=[1, 8, 32, 128, 512, 2048])
    parser.add_argument('--width', type=int, default=1536)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--include-warp', action='store_true')
    parser.add_argument('--include-compiled', action='store_true')
    parser.add_argument('--include-specialized', action='store_true')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 2 or any(r < 1 for r in args.rows):
        parser.error('Use positive row counts/iterations and at least two repeats.')
    with torch.inference_mode():
        run(args)
