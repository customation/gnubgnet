// SPDX-License-Identifier: GPL-3.0-or-later
// Thread-safe evaluation cache with striped locking and optimized hashing.

using System.Runtime.CompilerServices;
using GnuBgNet.Evaluation;

namespace GnuBgNet.Optimized;

/// <summary>
/// Drop-in replacement for <see cref="EvalCache"/> with:
/// - Thread-safe access via striped SpinLocks (safe for parallel rollout)
/// - Inline hash computation (no stackalloc intermediary)
/// - Same 2-way set-associative structure for cache-friendliness
/// </summary>
public sealed class ConcurrentEvalCache : IEvalCache
{
    private struct CacheEntry
    {
        public PositionKey Key;
        public int EvalContext;
        public float Ar0, Ar1, Ar2, Ar3, Ar4, Ar5;
    }

    private struct CacheNode
    {
        public CacheEntry Primary;
        public CacheEntry Secondary;
    }

    private const int StripeBits = 8;
    private const int StripeCount = 1 << StripeBits;
    private const int StripeMask = StripeCount - 1;

    private readonly CacheNode[] _entries;
    private readonly uint _hashMask;
    private readonly object[] _locks;

    public ConcurrentEvalCache(int sizeLog2)
    {
        uint size = 1u << sizeLog2;
        _hashMask = (size >> 1) - 1;
        _entries = new CacheNode[size / 2];
        _locks = new object[StripeCount];
        for (int i = 0; i < StripeCount; i++)
            _locks[i] = new object();
        Flush();
    }

    public void Flush()
    {
        for (int i = 0; i < _entries.Length; i++)
        {
            _entries[i].Primary.Key = new PositionKey { D0 = uint.MaxValue };
            _entries[i].Secondary.Key = new PositionKey { D0 = uint.MaxValue };
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint Lookup(PositionKey key, int evalContext, Span<float> output)
    {
        uint l = GetHashKeyInline(key, evalContext);
        int stripe = (int)(l & StripeMask);

        lock (_locks[stripe])
        {
            ref CacheNode node = ref _entries[l];

            if (node.Primary.Key.Equals(key) && node.Primary.EvalContext == evalContext)
            {
                CopyOut(ref node.Primary, output);
                return EvalCache.CacheHit;
            }

            if (node.Secondary.Key.Equals(key) && node.Secondary.EvalContext == evalContext)
            {
                (node.Primary, node.Secondary) = (node.Secondary, node.Primary);
                CopyOut(ref node.Primary, output);
                return EvalCache.CacheHit;
            }
        }

        return l;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(PositionKey key, int evalContext, ReadOnlySpan<float> output, uint l)
    {
        int stripe = (int)(l & StripeMask);

        lock (_locks[stripe])
        {
            ref CacheNode node = ref _entries[l];
            node.Secondary = node.Primary;

            node.Primary.Key = key;
            node.Primary.EvalContext = evalContext;
            node.Primary.Ar0 = output[0];
            node.Primary.Ar1 = output[1];
            node.Primary.Ar2 = output[2];
            node.Primary.Ar3 = output[3];
            node.Primary.Ar4 = output[4];
            node.Primary.Ar5 = output.Length > 5 ? output[5] : 0f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CopyOut(ref CacheEntry entry, Span<float> output)
    {
        output[0] = entry.Ar0;
        output[1] = entry.Ar1;
        output[2] = entry.Ar2;
        output[3] = entry.Ar3;
        output[4] = entry.Ar4;
        if (output.Length > 5)
            output[5] = entry.Ar5;
    }

    /// <summary>
    /// Inline MurmurHash3 — operates directly on PositionKey fields
    /// without allocating a stackalloc span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private uint GetHashKeyInline(PositionKey key, int evalContext)
    {
        uint hash = (uint)evalContext;

        hash *= 0xcc9e2d51;
        hash = (hash << 15) | (hash >> 17);
        hash *= 0x1b873593;
        hash = (hash << 13) | (hash >> 19);
        hash = hash * 5 + 0xe6546b64;

        hash = MixKey(hash, key.D0);
        hash = MixKey(hash, key.D1);
        hash = MixKey(hash, key.D2);
        hash = MixKey(hash, key.D3);
        hash = MixKey(hash, key.D4);
        hash = MixKey(hash, key.D5);
        hash = MixKey(hash, key.D6);

        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;

        return hash & _hashMask;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint MixKey(uint hash, uint k)
    {
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        hash ^= k;
        hash = (hash << 13) | (hash >> 19);
        hash = hash * 5 + 0xe6546b64;
        return hash;
    }
}
