// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.Evaluation;

/// <summary>
/// Interface for evaluation result caching.
/// Allows plugging in alternative cache strategies (LRU, concurrent, distributed, no-op, etc.).
/// A return value of <see cref="EvalCache.CacheHit"/> from Lookup indicates a cache hit.
/// </summary>
public interface IEvalCache
{
    uint Lookup(PositionKey key, int evalContext, Span<float> output);
    void Add(PositionKey key, int evalContext, ReadOnlySpan<float> output, uint l);
    void Flush();
}
