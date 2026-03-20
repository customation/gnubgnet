// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of FormatMove / FormatMovePlain / ParseMove from drawboard.c

using System.Text;

namespace GnuBgNet.Formatting;

/// <summary>
/// Format and parse backgammon moves in human-readable notation.
/// Port of FormatMove/FormatMovePlain/ParseMove from drawboard.c.
/// </summary>
public static class MoveFormatter
{
    /// <summary>
    /// Format a move with hit markers (*) and duplicate counts ((n)).
    /// Port of FormatMove() from drawboard.c.
    /// </summary>
    public static string FormatMove(Board board, int[] anMove)
    {
        return FormatMoveInternal(board, anMove, markHits: true);
    }

    /// <summary>
    /// Format a move in plain notation (no hit markers).
    /// Port of FormatMovePlain() from drawboard.c.
    /// </summary>
    public static string FormatMovePlain(int[] anMove)
    {
        var sb = new StringBuilder(32);
        for (int i = 0; i < 8 && anMove[i] >= 0; i += 2)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(FormatPoint(anMove[i] + 1));
            sb.Append('/');
            sb.Append(FormatPoint(anMove[i + 1] + 1));
        }
        return sb.ToString();
    }

    /// <summary>
    /// Parse a human-readable move string into the internal int[8] format (0-indexed).
    /// Port of ParseMove() from drawboard.c.
    /// Returns the number of sub-moves (1-4), or -1 on parse error.
    /// </summary>
    public static int ParseMove(string input, int[] an)
    {
        for (int i = 0; i < 8; i++) an[i] = -1;

        int[] anUser = new int[17];
        int c = 0;
        int fl = 0; // bit flags for '/' positions
        int pos = 0;

        while (pos < input.Length)
        {
            char ch = input[pos];

            if (char.IsWhiteSpace(ch))
            {
                pos++;
                continue;
            }

            if (char.IsDigit(ch))
            {
                if (c >= 8) return -1;
                int num = 0;
                while (pos < input.Length && char.IsDigit(input[pos]))
                {
                    num = num * 10 + (input[pos] - '0');
                    pos++;
                }
                if (num < 0 || num > 25) return -1;
                anUser[c++] = num;
                continue;
            }

            switch (ch)
            {
                case 'o' or 'O' or '-':
                    if (c >= 8) return -1;
                    anUser[c++] = 0;
                    if (ch != '-' && pos + 1 < input.Length && (input[pos + 1] == 'f' || input[pos + 1] == 'F'))
                    {
                        pos++;
                        if (pos + 1 < input.Length && (input[pos + 1] == 'f' || input[pos + 1] == 'F'))
                            pos++;
                    }
                    break;

                case 'b' or 'B':
                    if (c >= 8) return -1;
                    anUser[c++] = 25;
                    if (pos + 1 < input.Length && (input[pos + 1] == 'a' || input[pos + 1] == 'A'))
                    {
                        pos++;
                        if (pos + 1 < input.Length && (input[pos + 1] == 'r' || input[pos + 1] == 'R'))
                            pos++;
                    }
                    break;

                case '/':
                    if (c == 0 || (fl & (1 << c)) != 0) return -1;
                    fl |= 1 << c;
                    break;

                case '*' or ',' or ')':
                    break; // ignored

                case '(':
                {
                    pos++;
                    int n = 0;
                    while (pos < input.Length && char.IsDigit(input[pos]))
                    {
                        n = n * 10 + (input[pos] - '0');
                        pos++;
                    }
                    n--; // repeat count is n-1 additional copies
                    if (n < 1) return -1;
                    if (c < 2) return -1;
                    if ((fl & (1 << c)) != 0) return -1;

                    int iBegin;
                    for (iBegin = c - 1; iBegin >= 0; iBegin--)
                        if ((fl & (1 << iBegin)) == 0) break;
                    if (iBegin < 0) return -1;

                    int iEnd = c;
                    if (c + (iEnd - iBegin) * n > 8) return -1;

                    for (int rep = 0; rep < n; rep++)
                    {
                        for (int j = iBegin; j < iEnd; j++)
                        {
                            if ((fl & (1 << j)) != 0)
                                fl |= 1 << c;
                            anUser[c++] = anUser[j];
                        }
                    }
                    continue; // skip pos++ at end
                }

                default:
                    return -1;
            }

            pos++;
        }

        if ((fl & (1 << c)) != 0) return -1; // trailing /

        // Convert user points to internal move pairs
        int idx = 0;
        for (int j = 0; j < c; j++)
        {
            if (idx >= 8) return -1;

            if ((idx & 1) != 0 && anUser[j] == 25) return -1; // move from off
            if ((idx & 1) == 0 && anUser[j] == 0) return -1;  // move to bar

            an[idx] = anUser[j];

            if ((idx & 1) != 0 && j + 1 < c && (fl & (1 << (j + 1))) != 0)
            {
                // Combined move: destination is also next source
                if (idx >= 7) return -1;
                if (an[idx] == 0 || an[idx] == 25) return -1;
                an[++idx] = anUser[j];
            }

            idx++;
        }

        if ((idx & 1) != 0) return -1; // incomplete last move
        if (idx < 8) an[idx] = -1;

        // Convert from 1-indexed to 0-indexed
        for (int i = 0; i < 8 && an[i] >= 0; i++)
        {
            if (an[i] == 25)
                an[i] = 24; // bar
            else if (an[i] == 0)
                an[i] = -1; // off (bear off destination = -1 in internal format)
            else
                an[i]--; // 1-indexed → 0-indexed
        }

        CanonicalMoveOrder(an);
        return idx >> 1;
    }

    /// <summary>
    /// Format a point number as a string (1-based).
    /// 0 = "off", 25 = "bar", else numeric.
    /// </summary>
    private static string FormatPoint(int n)
    {
        if (n == 0) return "off";
        if (n == 25) return "bar";
        return n.ToString();
    }

    /// <summary>
    /// Internal move formatting with optional hit markers.
    /// Port of FormatMove() from drawboard.c.
    /// </summary>
    private static string FormatMoveInternal(Board board, int[] anMove, bool markHits)
    {
        // Convert to 1-based and create sub-move arrays
        int[][] moves = new int[4][];
        int[] moveLengths = new int[4];
        bool[] active = new bool[4];
        int nMoves = 0;

        for (int i = 0; i < 4 && anMove[i << 1] >= 0; i++)
        {
            moves[i] = new int[4];
            moves[i][0] = anMove[i << 1] + 1;       // source (1-based)
            moves[i][1] = anMove[(i << 1) | 1] + 1;  // dest (1-based)
            moveLengths[i] = 2;
            active[i] = true;
            nMoves++;
        }

        for (int i = nMoves; i < 4; i++)
        {
            moves[i] = new int[4];
            active[i] = false;
        }

        // Sort by source point descending
        for (int i = 0; i < nMoves - 1; i++)
        {
            for (int j = i + 1; j < nMoves; j++)
            {
                if (moves[j][0] > moves[i][0])
                {
                    (moves[i], moves[j]) = (moves[j], moves[i]);
                    (moveLengths[i], moveLengths[j]) = (moveLengths[j], moveLengths[i]);
                }
            }
        }

        // Combine chained moves (destination = next source)
        if (markHits)
        {
            for (int i = 0; i < nMoves; i++)
            {
                if (!active[i]) continue;
                for (int j = i + 1; j < nMoves; j++)
                {
                    if (!active[j]) continue;
                    int destI = moves[i][moveLengths[i] - 1];
                    if (destI == moves[j][0])
                    {
                        // Check if there's a hit at intermediate point
                        if (destI >= 1 && destI <= 24 && board.Opponent[24 - destI] > 0)
                        {
                            // Hit: record intermediate
                            moves[i][moveLengths[i]] = moves[j][moveLengths[j] - 1];
                            moveLengths[i]++;
                        }
                        else
                        {
                            // No hit: elide intermediate
                            moves[i][moveLengths[i] - 1] = moves[j][moveLengths[j] - 1];
                        }
                        active[j] = false;
                    }
                }
            }
        }

        // Compact active moves
        var compacted = new List<(int[] Move, int Length)>();
        for (int i = 0; i < nMoves; i++)
        {
            if (active[i])
                compacted.Add((moves[i], moveLengths[i]));
        }

        // Count duplicates
        int[] counts = new int[compacted.Count];
        bool[] dup = new bool[compacted.Count];
        for (int i = 0; i < compacted.Count; i++) counts[i] = 1;

        for (int i = 0; i < compacted.Count; i++)
        {
            if (dup[i]) continue;
            for (int j = i + 1; j < compacted.Count; j++)
            {
                if (dup[j]) continue;
                if (compacted[i].Length == compacted[j].Length)
                {
                    bool same = true;
                    for (int k = 0; k < compacted[i].Length; k++)
                    {
                        if (compacted[i].Move[k] != compacted[j].Move[k])
                        {
                            same = false;
                            break;
                        }
                    }
                    if (same) { counts[i]++; dup[j] = true; }
                }
            }
        }

        // Build output string
        var sb = new StringBuilder(32);
        int hitFlags = 0;
        bool first = true;

        for (int i = 0; i < compacted.Count; i++)
        {
            if (dup[i]) continue;
            if (!first) sb.Append(' ');
            first = false;

            var (move, len) = compacted[i];

            // Source
            sb.Append(FormatPoint(move[0]));

            // Intermediate points (hits)
            for (int j = 1; j < len - 1; j++)
            {
                sb.Append('/');
                sb.Append(FormatPoint(move[j]));
                if (markHits)
                {
                    sb.Append('*');
                    hitFlags |= 1 << move[j];
                }
            }

            // Destination
            sb.Append('/');
            sb.Append(FormatPoint(move[len - 1]));

            // Hit marker at destination
            if (markHits && move[len - 1] >= 1 && move[len - 1] <= 24
                && board.Opponent[24 - move[len - 1]] > 0
                && (hitFlags & (1 << move[len - 1])) == 0)
            {
                sb.Append('*');
                hitFlags |= 1 << move[len - 1];
            }

            // Duplicate count
            if (counts[i] > 1)
            {
                sb.Append('(');
                sb.Append(counts[i]);
                sb.Append(')');
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Sort sub-moves into canonical order (highest source first).
    /// Port of CanonicalMoveOrder() from drawboard.c.
    /// </summary>
    private static void CanonicalMoveOrder(int[] an)
    {
        // Count sub-moves
        int n = 0;
        for (int i = 0; i < 8 && an[i] >= 0; i += 2) n++;

        // Bubble sort by source descending, then dest descending
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                int si = i * 2, sj = j * 2;
                if (an[sj] > an[si] || (an[sj] == an[si] && an[sj + 1] > an[si + 1]))
                {
                    (an[si], an[sj]) = (an[sj], an[si]);
                    (an[si + 1], an[sj + 1]) = (an[sj + 1], an[si + 1]);
                }
            }
        }
    }
}
