// Copyright (C) 2002-2003 Joern Thyssen <jthyssen@dk.ibm.com>
// Copyright (C) 2004-2013 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from matchid.c

namespace GnuBgNet.Encoding;

/// <summary>
/// Decoded match state from a match ID.
/// </summary>
public sealed class MatchInfo
{
    public int Cube { get; set; } = 1;
    public int CubeOwner { get; set; } = -1;
    public int Move { get; set; }
    public bool Crawford { get; set; }
    public GameState GameState { get; set; }
    public int Turn { get; set; }
    public bool Doubled { get; set; }
    public int Resigned { get; set; }
    public int Die1 { get; set; }
    public int Die2 { get; set; }
    public int MatchTo { get; set; }
    public int Score0 { get; set; }
    public int Score1 { get; set; }
    public bool Jacoby { get; set; }
}

/// <summary>
/// Encodes and decodes GNU Backgammon match ID strings (12 characters).
/// </summary>
public static class MatchId
{
    private const string Base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// <summary>
    /// Encode match state to a 12-character match ID string.
    /// Port of MatchID() from matchid.c.
    /// </summary>
    public static string Encode(
        int die1, int die2,
        int turn, int resigned, bool doubled,
        int move, int cubeOwner, bool crawford,
        int matchTo, int score0, int score1,
        int cube, bool jacoby, GameState gs)
    {
        Span<byte> key = stackalloc byte[9];
        key.Clear();

        SetBits(key, 0, 4, LogCube(cube));
        SetBits(key, 4, 2, cubeOwner & 0x3);
        SetBits(key, 6, 1, move);
        SetBits(key, 7, 1, crawford ? 1 : 0);
        SetBits(key, 8, 3, (int)gs);
        SetBits(key, 11, 1, turn);
        SetBits(key, 12, 1, doubled ? 1 : 0);
        SetBits(key, 13, 2, resigned);

        // Dice: higher die first
        if (die1 >= die2)
        {
            SetBits(key, 15, 3, die1 & 0x7);
            SetBits(key, 18, 3, die2 & 0x7);
        }
        else
        {
            SetBits(key, 15, 3, die2 & 0x7);
            SetBits(key, 18, 3, die1 & 0x7);
        }

        SetBits(key, 21, 15, matchTo & 0x7FFF);
        SetBits(key, 36, 15, score0 & 0x7FFF);
        SetBits(key, 51, 15, score1 & 0x7FFF);
        SetBits(key, 66, 1, jacoby ? 0 : 1); // Note: inverted in encoding

        return KeyToBase64(key);
    }

    /// <summary>
    /// Decode a 12-character match ID string to match state.
    /// Port of MatchFromID() from matchid.c.
    /// Returns null if the ID is invalid.
    /// </summary>
    public static MatchInfo? Decode(string matchId)
    {
        if (string.IsNullOrEmpty(matchId))
            return null;

        // Base64 decode into 9 bytes
        Span<byte> decoded = stackalloc byte[Constants.MatchIdLength + 1];
        decoded.Clear();

        int len = Math.Min(matchId.Length, Constants.MatchIdLength);
        for (int i = 0; i < len; i++)
            decoded[i] = DecodeBase64Char(matchId[i]);

        Span<byte> key = stackalloc byte[9];
        key.Clear();

        ReadOnlySpan<byte> pch = decoded;
        int idx = 0;
        for (int i = 0; i < 3; i++)
        {
            key[idx++] = (byte)((pch[0] << 2) | (pch[1] >> 4));
            key[idx++] = (byte)((pch[1] << 4) | (pch[2] >> 2));
            key[idx++] = (byte)((pch[2] << 6) | pch[3]);
            pch = pch[4..];
        }

        // Extract fields
        var info = new MatchInfo();

        int cubeLog = GetBits(key, 0, 4);
        info.Cube = 1 << cubeLog;

        info.CubeOwner = GetBits(key, 4, 2);
        if (info.CubeOwner != 0 && info.CubeOwner != 1)
            info.CubeOwner = -1;

        info.Move = GetBits(key, 6, 1);
        info.Crawford = GetBits(key, 7, 1) != 0;
        info.GameState = (GameState)GetBits(key, 8, 3);
        info.Turn = GetBits(key, 11, 1);
        info.Doubled = GetBits(key, 12, 1) != 0;
        info.Resigned = GetBits(key, 13, 2);
        info.Die1 = GetBits(key, 15, 3);
        info.Die2 = GetBits(key, 18, 3);
        info.MatchTo = GetBits(key, 21, 15);
        info.Score0 = GetBits(key, 36, 15);
        info.Score1 = GetBits(key, 51, 15);
        int jacobyInv = GetBits(key, 66, 1);
        info.Jacoby = jacobyInv == 0; // Inverted in encoding

        // Validation
        if (info.Die1 < 0 || info.Die1 > 6) return null;
        if (info.Die2 < 0 || info.Die2 > 6) return null;
        if (info.MatchTo < 0 || info.MatchTo > Constants.MaxScore) return null;

        if (info.MatchTo > 0)
        {
            if (info.Score0 < 0 || info.Score0 > info.MatchTo) return null;
            if (info.Score1 < 0 || info.Score1 > info.MatchTo) return null;
        }
        else
        {
            if (info.Crawford) return null; // No Crawford in money play
        }

        return info;
    }

    /// <summary>Compute floor(log2(n)).</summary>
    public static int LogCube(int n)
    {
        int i = 0;
        while ((n >>= 1) != 0)
            i++;
        return i;
    }

    #region Internal helpers

    private static void SetBits(Span<byte> key, int bitPos, int nBits, int value)
    {
        for (int i = 0; i < nBits; i++)
        {
            int byteIdx = (bitPos + i) / 8;
            int bitIdx = (bitPos + i) % 8;

            if ((value & (1 << i)) != 0)
                key[byteIdx] |= (byte)(1 << bitIdx);
            else
                key[byteIdx] &= (byte)~(1 << bitIdx);
        }
    }

    private static int GetBits(ReadOnlySpan<byte> key, int bitPos, int nBits)
    {
        int result = 0;
        for (int i = 0; i < nBits; i++)
        {
            int byteIdx = (bitPos + i) / 8;
            int bitIdx = (bitPos + i) % 8;

            if ((key[byteIdx] & (1 << bitIdx)) != 0)
                result |= 1 << i;
        }
        return result;
    }

    private static string KeyToBase64(ReadOnlySpan<byte> key)
    {
        Span<char> result = stackalloc char[Constants.MatchIdLength];
        int pchIdx = 0;
        int puchIdx = 0;

        for (int i = 0; i < 3; i++)
        {
            result[pchIdx++] = Base64Chars[key[puchIdx] >> 2];
            result[pchIdx++] = Base64Chars[((key[puchIdx] & 0x03) << 4) | (key[puchIdx + 1] >> 4)];
            result[pchIdx++] = Base64Chars[((key[puchIdx + 1] & 0x0F) << 2) | (key[puchIdx + 2] >> 6)];
            result[pchIdx++] = Base64Chars[key[puchIdx + 2] & 0x3F];
            puchIdx += 3;
        }

        return new string(result);
    }

    private static byte DecodeBase64Char(char ch)
    {
        if (ch >= 'A' && ch <= 'Z') return (byte)(ch - 'A');
        if (ch >= 'a' && ch <= 'z') return (byte)(ch - 'a' + 26);
        if (ch >= '0' && ch <= '9') return (byte)(ch - '0' + 52);
        if (ch == '+') return 62;
        if (ch == '/') return 63;
        return 255;
    }

    #endregion
}
