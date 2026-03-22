// SPDX-License-Identifier: GPL-3.0-or-later
// Mersenne Twister PRNG (MT19937) for deterministic rollout dice generation.

namespace GnuBgNet.Random;

/// <summary>
/// Mersenne Twister MT19937 pseudo-random number generator.
/// Deterministic given a seed, used for reproducible rollouts.
/// </summary>
public sealed class MersenneTwister : IDiceGenerator
{
    private const int N = 624;
    private const int M = 397;
    private const uint MatrixA = 0x9908b0dfU;
    private const uint UpperMask = 0x80000000U;
    private const uint LowerMask = 0x7fffffffU;

    private readonly uint[] _mt = new uint[N];
    private int _mti = N + 1;

    public MersenneTwister(uint seed)
    {
        Init(seed);
    }

    /// <summary>
    /// Re-initialize the generator with a new seed. Resets all internal state.
    /// </summary>
    public void Init(uint seed)
    {
        _mt[0] = seed;
        for (_mti = 1; _mti < N; _mti++)
        {
            _mt[_mti] = 1812433253U * (_mt[_mti - 1] ^ (_mt[_mti - 1] >> 30)) + (uint)_mti;
        }
    }

    public uint NextUInt32()
    {
        uint y;

        if (_mti >= N)
        {
            int kk;
            for (kk = 0; kk < N - M; kk++)
            {
                y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                _mt[kk] = _mt[kk + M] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            }
            for (; kk < N - 1; kk++)
            {
                y = (_mt[kk] & UpperMask) | (_mt[kk + 1] & LowerMask);
                _mt[kk] = _mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            }
            y = (_mt[N - 1] & UpperMask) | (_mt[0] & LowerMask);
            _mt[N - 1] = _mt[M - 1] ^ (y >> 1) ^ ((y & 1U) != 0 ? MatrixA : 0U);
            _mti = 0;
        }

        y = _mt[_mti++];

        // Tempering
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;

        return y;
    }

    // gnubg uses rejection sampling: 2^32 / 6 = 715827882, exp232_l = 715827882 * 6 = 4294967292
    // Values >= exp232_l are rejected to eliminate modulo bias.
    private const uint Exp232Q = 715827882U;
    private const uint Exp232L = 4294967292U; // 715827882 * 6

    /// <summary>
    /// Generate a random die value (1-6).
    /// Uses rejection sampling matching gnubg's RollDice() from dice.c.
    /// </summary>
    public int NextDie()
    {
        uint r;
        do { r = NextUInt32(); } while (r >= Exp232L);
        return 1 + (int)(r / Exp232Q);
    }

    /// <summary>
    /// Generate a dice roll (two dice, d0 >= d1).
    /// </summary>
    public (int D0, int D1) NextDiceRoll()
    {
        int d0 = NextDie();
        int d1 = NextDie();
        return d0 >= d1 ? (d0, d1) : (d1, d0);
    }
}
