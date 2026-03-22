// SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkDotNet.Attributes;
using GnuBgNet;
using GnuBgNet.Encoding;
using GnuBgNet.Evaluation;
using GnuBgNet.Optimized;

namespace GnuBgNet.Benchmarks;

/// <summary>
/// Benchmarks comparing original EvalCache vs InlineHashEvalCache
/// (inline MurmurHash3 without stackalloc intermediary).
/// </summary>
[MemoryDiagnoser]
public class InlineHashCacheBenchmarks
{
    private const int CacheLog2 = 16;
    private EvalCache _original = null!;
    private InlineHashEvalCache _inlineHash = null!;
    private PositionKey[] _keys = null!;
    private float[][] _values = null!;

    [Params(100, 1000)]
    public int Operations { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _original = new EvalCache(CacheLog2);
        _inlineHash = new InlineHashEvalCache(CacheLog2);

        var rng = new System.Random(42);
        _keys = new PositionKey[Operations];
        _values = new float[Operations][];

        for (int i = 0; i < Operations; i++)
        {
            var board = new Board();
            int remaining = 15;
            for (int j = 0; j < 24 && remaining > 0; j++)
            {
                int n = rng.Next(0, Math.Min(remaining + 1, 6));
                board.Player[j] = (uint)n;
                remaining -= n;
            }
            remaining = 15;
            for (int j = 0; j < 24 && remaining > 0; j++)
            {
                int n = rng.Next(0, Math.Min(remaining + 1, 6));
                board.Opponent[j] = (uint)n;
                remaining -= n;
            }
            _keys[i] = PositionId.ToKey(board);
            _values[i] = [rng.NextSingle(), rng.NextSingle(), rng.NextSingle(),
                          rng.NextSingle(), rng.NextSingle()];
        }
    }

    [IterationSetup]
    public void IterationSetup()
    {
        _original.Flush();
        _inlineHash.Flush();
    }

    [Benchmark(Baseline = true)]
    public void Original_AddAndLookup()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        for (int i = 0; i < Operations; i++)
        {
            uint l = _original.Lookup(_keys[i], 0, output);
            if (l != EvalCache.CacheHit)
                _original.Add(_keys[i], 0, _values[i], l);
        }
        for (int i = 0; i < Operations; i++)
            _original.Lookup(_keys[i], 0, output);
    }

    [Benchmark]
    public void InlineHash_AddAndLookup()
    {
        Span<float> output = stackalloc float[Constants.NumOutputs];
        for (int i = 0; i < Operations; i++)
        {
            uint l = _inlineHash.Lookup(_keys[i], 0, output);
            if (l != EvalCache.CacheHit)
                _inlineHash.Add(_keys[i], 0, _values[i], l);
        }
        for (int i = 0; i < Operations; i++)
            _inlineHash.Lookup(_keys[i], 0, output);
    }
}
