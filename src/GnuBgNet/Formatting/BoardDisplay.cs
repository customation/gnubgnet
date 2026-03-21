// Copyright (C) 1999-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2020 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of DrawBoard / DrawBoardStd / DrawBoardCls / FIBSBoard from drawboard.c

using System.Text;
using GnuBgNet.Encoding;

namespace GnuBgNet.Formatting;

/// <summary>
/// ASCII board display for backgammon positions.
/// Port of DrawBoard, DrawBoardStd, DrawBoardCls and FIBSBoard from gnubg drawboard.c.
/// </summary>
public static class BoardDisplay
{
    // Characters for stacks of 5+ checkers: index = count, 0-4 = space, 5 = letter, 6-15 = digits/letters.
    private static readonly char[] AchX = "     X6789ABCDEF".ToCharArray();
    private static readonly char[] AchO = "     O6789ABCDEF".ToCharArray();

    /// <summary>
    /// Draw an ASCII board. Delegates to standard (counter-clockwise) or clockwise layout.
    /// </summary>
    /// <param name="board">Board position. Player = on-roll (X), Opponent = O.</param>
    /// <param name="fRoll">true if Player (X) is on roll; false if Opponent (O) is on roll.</param>
    /// <param name="annotations">7 annotation strings (indices 0-6) displayed on the right side. Null entries are skipped.</param>
    /// <param name="matchId">Optional match ID string to display.</param>
    /// <param name="nChequers">Number of chequers per side (default 15).</param>
    /// <param name="clockwise">If true, use clockwise layout; otherwise standard counter-clockwise.</param>
    /// <returns>The ASCII board as a string.</returns>
    public static string DrawBoard(Board board, bool fRoll, string?[]? annotations = null,
        string? matchId = null, int nChequers = 15, bool clockwise = false)
    {
        var asz = new string?[7];
        if (annotations != null)
        {
            for (int i = 0; i < 7 && i < annotations.Length; i++)
                asz[i] = annotations[i];
        }

        return clockwise
            ? DrawBoardCls(board, fRoll, asz, matchId, nChequers)
            : DrawBoardStd(board, fRoll, asz, matchId, nChequers);
    }

    /// <summary>
    /// Standard counter-clockwise layout.
    /// Top: points 13-24 (left to right), Bottom: points 12-1 (left to right).
    /// </summary>
    public static string DrawBoardStd(Board board, bool fRoll, string?[]? annotations = null,
        string? matchId = null, int nChequers = 15)
    {
        var asz = NormalizeAnnotations(annotations);
        var sb = new StringBuilder(1024);

        // anBoard[0] = Opponent, anBoard[1] = Player (X)
        uint[] opp = board.Opponent; // anBoard[0]
        uint[] plr = board.Player;   // anBoard[1]

        int cOffO = nChequers, cOffX = nChequers;
        for (int x = 0; x < 25; x++)
        {
            cOffO -= (int)opp[x];
            cOffX -= (int)plr[x];
        }

        // Position ID line
        string posId = ComputePositionId(board, fRoll);
        sb.Append($" GNU Backgammon  Position ID: {posId}\n");

        // Match ID line
        if (!string.IsNullOrEmpty(matchId))
        {
            sb.Append($"                 Match ID   : {matchId}\n");
        }

        // Top border with point labels
        sb.Append(fRoll
            ? " +13-14-15-16-17-18------19-20-21-22-23-24-+     "
            : " +12-11-10--9--8--7-------6--5--4--3--2--1-+     ");
        if (asz[0] != null) sb.Append(asz[0]);
        sb.Append('\n');

        // Top half: 4 rows of checkers (rows 0-3)
        for (int y = 0; y < 4; y++)
        {
            sb.Append(" |");

            // Points 12-17 (left quadrant of top half)
            for (int x = 12; x < 18; x++)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("| ");
            sb.Append(opp[24] > y ? 'O' : ' '); // opponent bar
            sb.Append(" |");

            // Points 18-23 (right quadrant of top half)
            for (int x = 18; x < 24; x++)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("| ");

            // Off-board O checkers (3 columns, 5 per column)
            for (int x = 0; x < 3; x++)
                sb.Append(cOffO > 5 * x + y ? 'O' : ' ');

            sb.Append(' ');

            if (y < 2 && asz[y + 1] != null)
                sb.Append(asz[y + 1]);
            sb.Append('\n');
        }

        // 5th row (digit/letter row) for top half
        sb.Append(" |");

        for (int x = 12; x < 18; x++)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("| ");
        sb.Append(AchO[opp[24]]); // opponent bar digit
        sb.Append(" |");

        for (int x = 18; x < 24; x++)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("| ");

        for (int x = 0; x < 3; x++)
            sb.Append(cOffO > 5 * x + 4 ? 'O' : ' ');

        sb.Append('\n');

        // BAR row
        sb.Append(fRoll ? 'v' : '^');
        sb.Append("|                  |BAR|                  |     ");
        if (asz[3] != null) sb.Append(asz[3]);
        sb.Append('\n');

        // 5th row (digit/letter row) for bottom half
        sb.Append(" |");

        for (int x = 11; x >= 6; x--)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("| ");
        sb.Append(AchX[plr[24]]); // player bar digit
        sb.Append(" |");

        for (int x = 5; x >= 0; x--)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("| ");

        for (int x = 0; x < 3; x++)
            sb.Append(cOffX > 5 * x + 4 ? 'X' : ' ');

        sb.Append('\n');

        // Bottom half: 4 rows of checkers (rows 3-0, rendered top to bottom)
        for (int y = 3; y >= 0; y--)
        {
            sb.Append(" |");

            // Points 11-6 (left quadrant of bottom half)
            for (int x = 11; x >= 6; x--)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("| ");
            sb.Append(plr[24] > y ? 'X' : ' '); // player bar
            sb.Append(" |");

            // Points 5-0 (right quadrant of bottom half)
            for (int x = 5; x >= 0; x--)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("| ");

            // Off-board X checkers
            for (int x = 0; x < 3; x++)
                sb.Append(cOffX > 5 * x + y ? 'X' : ' ');

            sb.Append(' ');

            if (y < 2 && asz[5 - y] != null)
                sb.Append(asz[5 - y]);
            sb.Append('\n');
        }

        // Bottom border with point labels
        sb.Append(fRoll
            ? " +12-11-10--9--8--7-------6--5--4--3--2--1-+     "
            : " +13-14-15-16-17-18------19-20-21-22-23-24-+     ");
        if (asz[6] != null) sb.Append(asz[6]);
        sb.Append('\n');

        return sb.ToString();
    }

    /// <summary>
    /// Clockwise layout.
    /// Top: points 24-13 (left to right), Bottom: points 1-12 (left to right).
    /// Off-board chequers are on the left side.
    /// </summary>
    public static string DrawBoardCls(Board board, bool fRoll, string?[]? annotations = null,
        string? matchId = null, int nChequers = 15)
    {
        var asz = NormalizeAnnotations(annotations);
        var sb = new StringBuilder(1024);

        uint[] opp = board.Opponent; // anBoard[0]
        uint[] plr = board.Player;   // anBoard[1]

        int cOffO = nChequers, cOffX = nChequers;
        for (int x = 0; x < 25; x++)
        {
            cOffO -= (int)opp[x];
            cOffX -= (int)plr[x];
        }

        // Position ID line (right-aligned "GNU Backgammon")
        string posId = ComputePositionId(board, fRoll);
        sb.Append($"  GNU Backgammon  Position ID: {posId}\n");

        // Match ID line
        if (!string.IsNullOrEmpty(matchId))
        {
            sb.Append($"                    Match ID   : {matchId}\n");
        }

        // Top border with point labels
        sb.Append(fRoll
            ? "    +24-23-22-21-20-19------18-17-16-15-14-13-+  "
            : "    +-1--2--3--4--5--6-------7--8--9-10-11-12-+  ");
        if (asz[0] != null) sb.Append(asz[0]);
        sb.Append('\n');

        // Top half: 4 rows (rows 0-3)
        for (int y = 0; y < 4; y++)
        {
            // Off-board O checkers on left (3 columns, right to left: col 2,1,0)
            for (int x = 2; x >= 0; x--)
                sb.Append(cOffO > 5 * x + y ? 'O' : ' ');

            sb.Append(" |");

            // Points 23-18 (left quadrant)
            for (int x = 23; x >= 18; x--)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("| ");
            sb.Append(opp[24] > y ? 'O' : ' '); // opponent bar
            sb.Append(" |");

            // Points 17-12 (right quadrant)
            for (int x = 17; x >= 12; x--)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("|  ");
            if (y < 2 && asz[y + 1] != null)
                sb.Append(asz[y + 1]);
            sb.Append('\n');
        }

        // 5th row (digit/letter row) for top half
        for (int x = 2; x >= 0; x--)
            sb.Append(cOffO > 5 * x + 4 ? 'O' : ' ');

        sb.Append(" |");

        for (int x = 23; x >= 18; x--)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("| ");
        sb.Append(AchO[opp[24]]); // opponent bar digit
        sb.Append(" |");

        for (int x = 17; x >= 12; x--)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("|  ");
        sb.Append('\n');

        // BAR row
        sb.Append("    |                  |BAR|                  |");
        sb.Append(fRoll ? 'v' : '^');
        sb.Append(' ');
        if (asz[3] != null) sb.Append(asz[3]);
        sb.Append('\n');

        // 5th row (digit/letter row) for bottom half
        for (int x = 2; x >= 0; x--)
            sb.Append(cOffX > 5 * x + 4 ? 'X' : ' ');

        sb.Append(" |");

        for (int x = 0; x < 6; x++)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("| ");
        sb.Append(AchX[plr[24]]); // player bar digit
        sb.Append(" |");

        for (int x = 6; x < 12; x++)
        {
            sb.Append(' ');
            sb.Append(plr[x] != 0 ? AchX[plr[x]] : AchO[opp[23 - x]]);
            sb.Append(' ');
        }

        sb.Append("|  ");
        sb.Append('\n');

        // Bottom half: 4 rows (rows 3-0)
        for (int y = 3; y >= 0; y--)
        {
            for (int x = 2; x >= 0; x--)
                sb.Append(cOffX > 5 * x + y ? 'X' : ' ');

            sb.Append(" |");

            for (int x = 0; x < 6; x++)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("| ");
            sb.Append(plr[24] > y ? 'X' : ' '); // player bar
            sb.Append(" |");

            for (int x = 6; x < 12; x++)
            {
                sb.Append(' ');
                sb.Append(plr[x] > y ? 'X' : opp[23 - x] > y ? 'O' : ' ');
                sb.Append(' ');
            }

            sb.Append("|  ");
            if (y < 2 && asz[5 - y] != null)
                sb.Append(asz[5 - y]);
            sb.Append('\n');
        }

        // Bottom border with point labels
        sb.Append(fRoll
            ? "    +-1--2--3--4--5--6-------7--8--9-10-11-12-+   "
            : "    +24-23-22-21-20-19------18-17-16-15-14-13-+  ");
        if (asz[6] != null) sb.Append(asz[6]);
        sb.Append('\n');

        return sb.ToString();
    }

    /// <summary>
    /// Generate a FIBS board string.
    /// See http://www.fibs.com/fibs_interface.html#board_state
    /// </summary>
    /// <param name="board">Board position.</param>
    /// <param name="fRoll">true if Player (X) is on roll.</param>
    /// <param name="playerName">Name of the player on roll.</param>
    /// <param name="opponentName">Name of the opponent.</param>
    /// <param name="matchLength">Match length (0 = unlimited).</param>
    /// <param name="playerScore">Player's score.</param>
    /// <param name="opponentScore">Opponent's score.</param>
    /// <param name="die0">First die value (0 if not rolled).</param>
    /// <param name="die1">Second die value (0 if not rolled).</param>
    /// <param name="cubeValue">Current cube value.</param>
    /// <param name="cubeOwner">Cube owner: 0 = opponent, 1 = player, -1 = centered.</param>
    /// <param name="crawford">true if this is the Crawford game.</param>
    /// <param name="nChequers">Number of chequers per side (default 15).</param>
    /// <param name="doubled">Doubling state: 0 = none, positive = player doubled, negative = opponent doubled.</param>
    /// <param name="turn">Whose turn: 1 = player, -1 = opponent.</param>
    /// <param name="postCrawford">true if post-Crawford.</param>
    /// <returns>FIBS board string.</returns>
    public static string FIBSBoard(Board board, bool fRoll, string playerName, string opponentName,
        int matchLength, int playerScore, int opponentScore,
        int die0, int die1, int cubeValue, int cubeOwner, bool crawford,
        int nChequers = 15, int doubled = 0, int turn = 1, bool postCrawford = false)
    {
        var sb = new StringBuilder(256);

        uint[] opp = board.Opponent; // anBoard[0]
        uint[] plr = board.Player;   // anBoard[1]

        // Names and match length/score — colons in names replaced with underscores
        sb.Append("board:");
        sb.Append(playerName.Replace(':', '_'));
        sb.Append(':');
        sb.Append(opponentName.Replace(':', '_'));
        sb.AppendFormat(":{0}:{1}:{2}:", matchLength, playerScore, opponentScore);

        // Opponent on bar (negative)
        sb.AppendFormat("{0}:", -(int)opp[24]);

        // 24 board points: from FIBS perspective, iterate i = 0..23
        // opp point = anBoard[0][23-i], plr point = anBoard[1][i]
        // If opponent has checkers: output negative; else output player count
        for (int i = 0; i < 24; i++)
        {
            int oppCount = (int)opp[23 - i];
            sb.AppendFormat("{0}:", oppCount > 0 ? -oppCount : (int)plr[i]);
        }

        // Player on bar
        sb.AppendFormat("{0}:", plr[24]);

        // Whose turn
        sb.Append(fRoll ? "1:" : "-1:");

        // Calculate off-board counts
        int anOff0 = nChequers, anOff1 = nChequers;
        for (int i = 0; i < 25; i++)
        {
            anOff0 -= (int)opp[i];
            anOff1 -= (int)plr[i];
        }

        // Remaining fields:
        // die0:die1:die0:die1:cube:canDouble(opp):canDouble(plr):doubled:color:direction:
        // unused:bar:off1:off0:onHome1:onHome0:onBar1:onBar0:nCanMove:fPostCrawford:fNonCrawford
        int fTurn = turn;
        int nCube = fTurn < 0 ? 1 : cubeValue;
        int canDoubleOpp = (fTurn < 0 || cubeOwner != 0) ? 1 : 0;
        int canDoublePlr = (fTurn < 0 || cubeOwner != 1) ? 1 : 0;
        int fDoubled = doubled != 0 ? (fRoll ? -1 : 1) : 0;

        sb.AppendFormat("{0}:{1}:{2}:{3}:{4}:{5}:{6}:{7}:1:-1:0:25:{8}:{9}:0:0:0:0:{10}:{11}",
            die0, die1, die0, die1,
            nCube, canDoubleOpp, canDoublePlr, fDoubled,
            anOff1, anOff0,
            postCrawford ? 1 : 0,
            crawford ? 0 : 1);

        return sb.ToString();
    }

    private static string?[] NormalizeAnnotations(string?[]? annotations)
    {
        var asz = new string?[7];
        if (annotations != null)
        {
            for (int i = 0; i < 7 && i < annotations.Length; i++)
                asz[i] = annotations[i];
        }
        return asz;
    }

    private static string ComputePositionId(Board board, bool fRoll)
    {
        if (fRoll)
        {
            return PositionId.Encode(board);
        }
        else
        {
            // When opponent is on roll, swap sides for position ID
            var swapped = board.Swapped();
            return PositionId.Encode(swapped);
        }
    }
}
