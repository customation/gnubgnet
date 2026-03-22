// SPDX-License-Identifier: GPL-3.0-or-later

using GnuBgNet;
using GnuBgNet.NeuralNet;

namespace GnuBgNet.Benchmarks;

/// <summary>
/// Shared helpers for locating data files and creating test boards.
/// </summary>
internal static class BenchmarkSetup
{
    internal static string FindDataDir()
    {
        string[] candidates =
        [
            Environment.GetEnvironmentVariable("GNUBG_DATA_DIR") ?? "",
            @"C:\git\github\customation\gnubg",
            @"C:\git\github\customation\gnubgnet\data",
        ];

        foreach (var d in candidates)
        {
            if (!string.IsNullOrEmpty(d) && File.Exists(Path.Combine(d, "gnubg.wd")))
                return d;
        }

        throw new FileNotFoundException(
            "Could not find gnubg.wd. Set GNUBG_DATA_DIR environment variable.");
    }

    internal static NetworkSet LoadNetworks()
        => NetworkSet.LoadBinary(Path.Combine(FindDataDir(), "gnubg.wd"));

    internal static Board CreateOpeningBoard() => Board.Opening();

    /// <summary>
    /// Mid-game contact position with checkers scattered and some on bar.
    /// </summary>
    internal static Board CreateContactBoard()
    {
        var b = new Board();
        // Player: spread across the board with one on bar
        b.Player[0] = 2; b.Player[3] = 3; b.Player[7] = 2;
        b.Player[11] = 3; b.Player[16] = 2; b.Player[20] = 1;
        b.Player[23] = 1; b.Player[24] = 1; // 1 on bar
        // Opponent: also scattered
        b.Opponent[1] = 2; b.Opponent[4] = 3; b.Opponent[8] = 2;
        b.Opponent[12] = 3; b.Opponent[17] = 2; b.Opponent[21] = 2;
        b.Opponent[23] = 1;
        return b;
    }

    /// <summary>
    /// Late-game bearing off position (race).
    /// </summary>
    internal static Board CreateRaceBoard()
    {
        var b = new Board();
        // All checkers in home board
        b.Player[0] = 3; b.Player[1] = 3; b.Player[2] = 3;
        b.Player[3] = 3; b.Player[4] = 2; b.Player[5] = 1;
        b.Opponent[0] = 3; b.Opponent[1] = 3; b.Opponent[2] = 3;
        b.Opponent[3] = 3; b.Opponent[4] = 2; b.Opponent[5] = 1;
        return b;
    }
}
