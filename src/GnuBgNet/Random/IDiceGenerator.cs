// SPDX-License-Identifier: GPL-3.0-or-later

namespace GnuBgNet.Random;

/// <summary>
/// Interface for dice rolling.
/// Allows plugging in alternative RNGs (System.Random, crypto RNG, deterministic sequences, etc.).
/// </summary>
public interface IDiceGenerator
{
    (int D0, int D1) NextDiceRoll();
}
