// Copyright (C) 1997-2000 Gary Wong <gtw@gnu.org>
// Copyright (C) 2002-2022 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from lib/cache.c

using System.Runtime.CompilerServices;

namespace GnuBgNet.Evaluation;

/// <summary>
/// 2-way set-associative evaluation cache with MurmurHash3.
/// Port of evalCache from lib/cache.c.
/// </summary>
public sealed class EvalCache : IEvalCache
{
    public const uint CacheHit = uint.MaxValue;

    private struct CacheEntry
    {
        public PositionKey Key;
        public int EvalContext;
        public float Ar0, Ar1, Ar2, Ar3, Ar4, Ar5; // 5 outputs + cubeful
    }

    private struct CacheNode
    {
        public CacheEntry Primary;
        public CacheEntry Secondary;
    }

    private readonly CacheNode[] _entries;
    private readonly uint _hashMask;

    public EvalCache(int sizeLog2)
    {
        uint size = 1u << sizeLog2;
        _hashMask = (size >> 1) - 1;
        _entries = new CacheNode[size / 2];
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

    /// <summary>
    /// Look up a position in the cache.
    /// Returns CacheHit if found (output filled), or the hash bucket index on miss.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint Lookup(PositionKey key, int evalContext, Span<float> output)
    {
        uint l = GetHashKey(key, evalContext);

        ref CacheNode node = ref _entries[l];

        if (node.Primary.Key.Equals(key) && node.Primary.EvalContext == evalContext)
        {
            CopyOut(ref node.Primary, output);
            return CacheHit;
        }

        if (node.Secondary.Key.Equals(key) && node.Secondary.EvalContext == evalContext)
        {
            // Promote hot entry
            (node.Primary, node.Secondary) = (node.Secondary, node.Primary);
            CopyOut(ref node.Primary, output);
            return CacheHit;
        }

        return l;
    }

    /// <summary>
    /// Add an entry to the cache at bucket l.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(PositionKey key, int evalContext, ReadOnlySpan<float> output, uint l)
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
    /// MurmurHash3-based hash function.
    /// Port of GetHashKey() from lib/cache.c.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private uint GetHashKey(PositionKey key, int evalContext)
    {
        uint hash = (uint)evalContext;

        hash *= 0xcc9e2d51;
        hash = (hash << 15) | (hash >> 17);
        hash *= 0x1b873593;
        hash = (hash << 13) | (hash >> 19);
        hash = hash * 5 + 0xe6546b64;

        Span<uint> data = stackalloc uint[7];
        data[0] = key.D0; data[1] = key.D1; data[2] = key.D2;
        data[3] = key.D3; data[4] = key.D4; data[5] = key.D5; data[6] = key.D6;

        for (int i = 0; i < 7; i++)
        {
            uint k = data[i];
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;
            hash ^= k;
            hash = (hash << 13) | (hash >> 19);
            hash = hash * 5 + 0xe6546b64;
        }

        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;

        return hash & _hashMask;
    }

    /// <summary>
    /// Compute the eval context key for caching (simplified, money game).
    /// </summary>
    public static int ComputeEvalKey(int nPlies, bool cubeful, bool usePrune)
    {
        return ComputeEvalKey(nPlies, cubeful, usePrune, null, false);
    }

    /// <summary>
    /// Compute the eval context key for caching with full match/cube context.
    /// Port of EvalKey() from eval.c.
    ///
    /// Bit layout:
    ///   00-03: nPlies
    ///   04:    fCubeful
    ///   05:    fMove
    ///   06:    fUsePrune
    ///   07-12: away score[move]
    ///   13-18: away score[!move]
    ///   19-22: log2(nCube)
    ///   23-24: fCubeOwner (0=move, 1=opp, 2=centered)
    ///   25:    fCrawford
    ///   26:    fJacoby
    ///   27:    fBeavers
    /// </summary>
    public static int ComputeEvalKey(int nPlies, bool cubeful, bool usePrune,
        CubeInfo? ci, bool fCubefulEquity)
    {
        int key = nPlies | ((cubeful ? 1 : 0) << 4);

        if (ci.HasValue)
            key |= (ci.Value.Move << 5);

        if (nPlies > 0)
            key ^= (usePrune ? 1 : 0) << 6;

        if (ci.HasValue && (nPlies > 0 || fCubefulEquity))
        {
            var c = ci.Value;
            if (c.MatchTo > 0)
            {
                key ^=
                    ((c.MatchTo - c.GetScore(c.Move) - 1) << 7) ^
                    ((c.MatchTo - c.GetScore(1 - c.Move) - 1) << 13) ^
                    (LogCube(c.Cube) << 19) ^
                    ((c.CubeOwner < 0 ? 2 : (c.CubeOwner == c.Move ? 1 : 0)) << 23) ^
                    ((c.Crawford ? 1 : 0) << 25);
            }
            else if (cubeful || fCubefulEquity)
            {
                key ^=
                    ((c.CubeOwner < 0 ? 2 : (c.CubeOwner == c.Move ? 1 : 0)) << 23) ^
                    ((c.Jacoby ? 1 : 0) << 26) ^
                    ((c.Beavers ? 1 : 0) << 27);
            }

            if (fCubefulEquity)
                key ^= unchecked((int)0x6a47b47e);
        }

        return key;
    }

    /// <summary>
    /// Returns floor(log2(n)). Port of LogCube() from eval.c.
    /// </summary>
    private static int LogCube(int n)
    {
        int r = 0;
        while (n > 1) { n >>= 1; r++; }
        return r;
    }
}
