// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of parse_mat_file(), parse_mat_move(), parse_mat_submove() from gnubgapi.c

namespace GnuBgNet.Formatting;

/// <summary>
/// Parser for Jellyfish .mat match files.
/// Port of parse_mat_file() from gnubgapi.c lines 1106-1242.
/// </summary>
public static class MatParser
{
    /// <summary>
    /// Parse a Jellyfish .mat file and extract turns.
    /// Each turn has a player index (0 or 1), dice, and from/to pairs.
    /// </summary>
    public static List<GameTurn> ParseFile(string matPath)
    {
        var turns = new List<GameTurn>();
        bool inGame = false;

        foreach (string rawLine in File.ReadLines(matPath))
        {
            string line = rawLine.TrimEnd('\r', '\n');
            if (line.Length == 0 || line[0] == ';') continue;

            // Detect "Game N" header
            ReadOnlySpan<char> trimmed = line.AsSpan().TrimStart();
            if (trimmed.StartsWith("Game "))
            {
                inGame = true;
                continue;
            }

            // Detect match score header (e.g. " 0 point match")
            if (trimmed.EndsWith("point match"))
                continue;

            // Detect "Wins N point" lines
            if (trimmed.Contains("Wins ", StringComparison.Ordinal))
            {
                inGame = false;
                continue;
            }

            if (!inGame) continue;

            // Parse move line: "  N) d1d2: move1   d1d2: move2"
            int pos = 0;
            SkipWhitespace(line, ref pos);

            // Check for "N)" prefix
            if (pos >= line.Length || line[pos] < '0' || line[pos] > '9') continue;
            while (pos < line.Length && line[pos] >= '0' && line[pos] <= '9') pos++;
            if (pos >= line.Length || line[pos] != ')') continue;
            pos++; // skip ')'
            SkipWhitespace(line, ref pos);

            // Parse up to two half-moves
            for (int half = 0; half < 2; half++)
            {
                SkipWhitespace(line, ref pos);
                if (pos >= line.Length) break;

                // Read dice: two digits
                if (pos + 1 >= line.Length) break;
                if (line[pos] < '1' || line[pos] > '6' || line[pos + 1] < '1' || line[pos + 1] > '6')
                    break;

                int die1 = line[pos] - '0';
                int die2 = line[pos + 1] - '0';
                pos += 2;

                // Skip ": "
                if (pos < line.Length && line[pos] == ':') pos++;
                SkipWhitespace(line, ref pos);

                // Determine move text extent
                string moveText;
                if (half == 0)
                {
                    // Look for next dice pattern (digit-digit-colon after whitespace)
                    int end = FindNextDicePattern(line, pos);
                    if (end >= 0)
                    {
                        moveText = line[pos..end].TrimEnd();
                        pos = end;
                    }
                    else
                    {
                        moveText = line[pos..].TrimEnd();
                        pos = line.Length;
                    }
                }
                else
                {
                    moveText = line[pos..].TrimEnd();
                    pos = line.Length;
                }

                int[] anMove = [-1, -1, -1, -1, -1, -1, -1, -1];
                if (!moveText.Contains("Cannot", StringComparison.Ordinal) && moveText.Length > 0)
                    ParseMove(moveText, anMove);

                turns.Add(new GameTurn
                {
                    PositionId = string.Empty, // will be filled during analysis
                    Die1 = die1,
                    Die2 = die2,
                    PlayedMove = anMove,
                    Player = half,
                });
            }
        }

        return turns;
    }

    /// <summary>
    /// Parse a full move string like "13/7 6/3" or "bar/22 13/7(2)".
    /// Port of parse_mat_move() from gnubgapi.c lines 1058-1099.
    /// </summary>
    public static int ParseMove(string moveStr, int[] anMove)
    {
        for (int i = 0; i < 8; i++) anMove[i] = -1;

        int idx = 0;
        int pos = 0;

        while (pos < moveStr.Length && idx < 8)
        {
            SkipWhitespace(moveStr, ref pos);
            if (pos >= moveStr.Length) break;

            // Read one submove token (until space, '(' or end)
            // Also stop at '*' after the token
            int start = pos;
            while (pos < moveStr.Length && moveStr[pos] != ' ' && moveStr[pos] != '\t'
                   && moveStr[pos] != '(')
                pos++;

            string tok = moveStr[start..pos];
            // Strip trailing '*' (hit marker)
            tok = tok.TrimEnd('*');

            if (tok.Length == 0) break;

            if (!ParseSubMove(tok, out int from, out int to))
                break;

            // Check for "(N)" repetition
            int reps = 1;
            if (pos < moveStr.Length && moveStr[pos] == '(')
            {
                pos++; // skip '('
                int repStart = pos;
                while (pos < moveStr.Length && moveStr[pos] >= '0' && moveStr[pos] <= '9') pos++;
                if (repStart < pos)
                    reps = int.Parse(moveStr[repStart..pos]);
                if (pos < moveStr.Length && moveStr[pos] == ')') pos++;
                if (reps < 1 || reps > 4) reps = 1;
            }

            for (int r = 0; r < reps && idx < 8; r++)
            {
                anMove[idx++] = from;
                anMove[idx++] = to;
            }
        }

        return idx / 2;
    }

    /// <summary>
    /// Parse a single submove token (e.g. "13/7", "bar/22", "6/off").
    /// Port of parse_mat_submove() from gnubgapi.c lines 1027-1052.
    /// </summary>
    public static bool ParseSubMove(string tok, out int from, out int to)
    {
        from = -1;
        to = -1;

        // Handle "bar/N"
        if (tok.StartsWith("bar/", StringComparison.OrdinalIgnoreCase))
        {
            from = 24; // bar
            if (int.TryParse(tok.AsSpan(4), out int dest))
            {
                to = dest - 1; // 1-indexed → 0-indexed
                return to >= 0 && to < 24;
            }
            return false;
        }

        int slashIdx = tok.IndexOf('/');
        if (slashIdx < 0) return false;

        // Parse source (1-indexed → 0-indexed)
        // 25 = bar (becomes index 24)
        if (!int.TryParse(tok.AsSpan(0, slashIdx), out int src))
            return false;
        from = src - 1; // 1-indexed → 0-indexed; 25 → 24 (bar)
        if (from < 0 || from > 24) return false;

        string destStr = tok[(slashIdx + 1)..];

        if (destStr.Equals("off", StringComparison.OrdinalIgnoreCase))
        {
            to = -1; // bear off
            return true;
        }

        if (int.TryParse(destStr, out int d))
        {
            to = d - 1; // 1-indexed → 0-indexed; 0 → -1 (bear off)
            if (to == -1) return true; // "N/0" = bear off
            return to >= 0 && to < 24;
        }

        return false;
    }

    private static void SkipWhitespace(string s, ref int pos)
    {
        while (pos < s.Length && (s[pos] == ' ' || s[pos] == '\t'))
            pos++;
    }

    /// <summary>
    /// Find the start position of the next "D1D2:" dice pattern after whitespace.
    /// Returns -1 if not found.
    /// </summary>
    private static int FindNextDicePattern(string line, int startAfter)
    {
        for (int q = startAfter; q + 2 < line.Length; q++)
        {
            if ((q == startAfter || line[q - 1] == ' ' || line[q - 1] == '\t') &&
                line[q] >= '1' && line[q] <= '6' &&
                line[q + 1] >= '1' && line[q + 1] <= '6' &&
                line[q + 2] == ':')
            {
                return q;
            }
        }
        return -1;
    }
}
