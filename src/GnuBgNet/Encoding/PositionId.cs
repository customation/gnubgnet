// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2014 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from positionid.c

namespace GnuBgNet.Encoding;

/// <summary>
/// Encodes and decodes GNU Backgammon position IDs and position keys.
/// </summary>
public static class PositionId
{
    private const string Base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// <summary>
    /// Compute the fast position key (7 × uint32, 4-bit nibbles) from a board.
    /// Port of PositionKey() from positionid.c.
    /// </summary>
    public static PositionKey ToKey(Board board)
    {
        var key = new PositionKey();
        // Player = anBoard[1], Opponent = anBoard[0]
        // anpBoard[0..2] = anBoard[1][0..23] packed 8 per uint
        // anpBoard[3..5] = anBoard[0][0..23] packed 8 per uint
        // anpBoard[6] = anBoard[0][24] | (anBoard[1][24] << 4)

        key.D0 = Pack8(board.Player, 0);
        key.D1 = Pack8(board.Player, 8);
        key.D2 = Pack8(board.Player, 16);
        key.D3 = Pack8(board.Opponent, 0);
        key.D4 = Pack8(board.Opponent, 8);
        key.D5 = Pack8(board.Opponent, 16);
        key.D6 = board.Opponent[24] | (board.Player[24] << 4);

        return key;
    }

    /// <summary>
    /// Reconstruct a board from a fast position key.
    /// Port of PositionFromKey() from positionid.c.
    /// </summary>
    public static Board FromKey(PositionKey key)
    {
        var board = new Board();

        Unpack8(key.D0, ref board.Player, 0);
        Unpack8(key.D1, ref board.Player, 8);
        Unpack8(key.D2, ref board.Player, 16);
        Unpack8(key.D3, ref board.Opponent, 0);
        Unpack8(key.D4, ref board.Opponent, 8);
        Unpack8(key.D5, ref board.Opponent, 16);
        board.Opponent[24] = key.D6 & 0x0Fu;
        board.Player[24] = (key.D6 >> 4) & 0x0Fu;

        return board;
    }

    /// <summary>
    /// Reconstruct a board from key with sides swapped.
    /// Port of PositionFromKeySwapped() from positionid.c.
    /// </summary>
    public static Board FromKeySwapped(PositionKey key)
    {
        var board = new Board();
        FromKeySwappedInto(key, ref board);
        return board;
    }

    /// <summary>
    /// Reconstruct a board from key into an existing board (zero-alloc).
    /// </summary>
    public static void FromKeyInto(PositionKey key, ref Board board)
    {
        board.Player.AsSpan().Clear();
        board.Opponent.AsSpan().Clear();
        Unpack8(key.D0, ref board.Player, 0);
        Unpack8(key.D1, ref board.Player, 8);
        Unpack8(key.D2, ref board.Player, 16);
        Unpack8(key.D3, ref board.Opponent, 0);
        Unpack8(key.D4, ref board.Opponent, 8);
        Unpack8(key.D5, ref board.Opponent, 16);
        board.Opponent[24] = key.D6 & 0x0Fu;
        board.Player[24] = (key.D6 >> 4) & 0x0Fu;
    }

    /// <summary>
    /// Reconstruct a board from key with sides swapped into an existing board (zero-alloc).
    /// </summary>
    public static void FromKeySwappedInto(PositionKey key, ref Board board)
    {
        board.Player.AsSpan().Clear();
        board.Opponent.AsSpan().Clear();
        Unpack8(key.D0, ref board.Opponent, 0);
        Unpack8(key.D1, ref board.Opponent, 8);
        Unpack8(key.D2, ref board.Opponent, 16);
        Unpack8(key.D3, ref board.Player, 0);
        Unpack8(key.D4, ref board.Player, 8);
        Unpack8(key.D5, ref board.Player, 16);
        board.Player[24] = key.D6 & 0x0Fu;
        board.Opponent[24] = (key.D6 >> 4) & 0x0Fu;
    }

    /// <summary>
    /// Encode a board to the 14-character position ID string.
    /// Port of PositionID() from positionid.c.
    /// </summary>
    public static string Encode(Board board)
    {
        // Step 1: build the old position key (variable-length bit packing)
        Span<byte> oldKey = stackalloc byte[10];
        oldKey.Clear();
        OldPositionKey(board, oldKey);

        // Step 2: base64-encode the 10 bytes into 14 chars
        return OldKeyToBase64(oldKey);
    }

    /// <summary>
    /// Encode a position key to the 14-character position ID string.
    /// Port of PositionIDFromKey() from positionid.c.
    /// </summary>
    public static string EncodeFromKey(PositionKey key)
    {
        var board = FromKey(key);
        return Encode(board);
    }

    /// <summary>
    /// Decode a 14-character position ID string to a board.
    /// Port of PositionFromID() from positionid.c.
    /// Returns null if the position is invalid.
    /// </summary>
    public static Board? Decode(string positionId)
    {
        if (string.IsNullOrEmpty(positionId))
            return null;

        // Step 1: base64-decode into 10 bytes (old key)
        Span<byte> decoded = stackalloc byte[Constants.PositionIdLength + 1];
        decoded.Clear();

        int len = Math.Min(positionId.Length, Constants.PositionIdLength);
        for (int i = 0; i < len; i++)
            decoded[i] = DecodeBase64Char(positionId[i]);

        Span<byte> oldKey = stackalloc byte[10];
        oldKey.Clear();

        ReadOnlySpan<byte> pch = decoded;
        int puchIdx = 0;
        for (int i = 0; i < 3; i++)
        {
            oldKey[puchIdx++] = (byte)((pch[0] << 2) | (pch[1] >> 4));
            oldKey[puchIdx++] = (byte)((pch[1] << 4) | (pch[2] >> 2));
            oldKey[puchIdx++] = (byte)((pch[2] << 6) | pch[3]);
            pch = pch[4..];
        }
        oldKey[puchIdx] = (byte)((pch[0] << 2) | (pch[1] >> 4));

        // Step 2: decode old key into board
        var board = new Board();
        OldPositionFromKey(ref board, oldKey);

        // Step 3: validate
        return CheckPosition(board) ? board : null;
    }

    /// <summary>
    /// Check that a board position is legal.
    /// Port of CheckPosition() from positionid.c.
    /// </summary>
    public static bool CheckPosition(Board board)
    {
        uint ac0 = 0, ac1 = 0;
        for (int i = 0; i < Constants.NumPoints; i++)
        {
            ac0 += board.Opponent[i];
            ac1 += board.Player[i];
            if (ac0 > Constants.NumCheckers || ac1 > Constants.NumCheckers)
                return false;
        }

        // Check for both players having checkers on the same point
        for (int i = 0; i < 24; i++)
        {
            if (board.Opponent[i] > 0 && board.Player[23 - i] > 0)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Compute bearoff position ID (combinatorial enumeration).
    /// Port of PositionBearoff() from positionid.c.
    /// </summary>
    public static uint PositionBearoff(ReadOnlySpan<uint> anBoard, int nPoints, int nChequers)
    {
        uint j = (uint)(nPoints - 1);
        for (int i = 0; i < nPoints; i++)
            j += anBoard[i];

        uint fBits = 1u << (int)j;

        for (int i = 0; i < nPoints - 1; i++)
        {
            j -= anBoard[i] + 1;
            fBits |= 1u << (int)j;
        }

        return PositionF(fBits, (uint)(nChequers + nPoints), (uint)nPoints);
    }

    /// <summary>
    /// Decode a bearoff position ID back to board.
    /// Port of PositionFromBearoff() from positionid.c.
    /// </summary>
    public static void PositionFromBearoff(Span<uint> anBoard, uint usID, int nPoints, int nChequers)
    {
        uint fBits = PositionInv(usID, (uint)(nChequers + nPoints), (uint)nPoints);

        for (int i = 0; i < nPoints; i++)
            anBoard[i] = 0;

        int j = nPoints - 1;
        for (int i = 0; i < nChequers + nPoints; i++)
        {
            if ((fBits & (1u << i)) != 0)
            {
                if (j == 0)
                    break;
                j--;
            }
            else
            {
                anBoard[j]++;
            }
        }
    }

    /// <summary>
    /// Combination function C(n, r) using precomputed table.
    /// Port of Combination() from positionid.c.
    /// </summary>
    public static uint Combination(uint n, uint r)
    {
        if (n == 0 || r == 0 || n > MaxN || r > MaxR)
            return 0;
        return CombinationTable[n - 1, r - 1];
    }

    #region Internal helpers

    private const int MaxN = 40;
    private const int MaxR = 25;

    private static readonly uint[,] CombinationTable = InitCombinationTable();

    private static uint[,] InitCombinationTable()
    {
        var table = new uint[MaxN, MaxR];

        for (int i = 0; i < MaxN; i++)
            table[i, 0] = (uint)(i + 1);

        for (int j = 1; j < MaxR; j++)
            table[0, j] = 0;

        for (int i = 1; i < MaxN; i++)
            for (int j = 1; j < MaxR; j++)
                table[i, j] = table[i - 1, j - 1] + table[i - 1, j];

        return table;
    }

    internal static uint PositionF(uint fBits, uint n, uint r)
    {
        if (n == r)
            return 0;

        if ((fBits & (1u << (int)(n - 1))) != 0)
            return Combination(n - 1, r) + PositionF(fBits, n - 1, r - 1);
        else
            return PositionF(fBits, n - 1, r);
    }

    private static uint PositionInv(uint nID, uint n, uint r)
    {
        if (r == 0)
            return 0;
        if (n == r)
            return (1u << (int)n) - 1;

        uint nC = Combination(n - 1, r);
        if (nID >= nC)
            return (1u << (int)(n - 1)) | PositionInv(nID - nC, n - 1, r - 1);
        else
            return PositionInv(nID, n - 1, r);
    }

    private static uint Pack8(in BoardSide arr, int offset)
    {
        return arr[offset]
            | (arr[offset + 1] << 4)
            | (arr[offset + 2] << 8)
            | (arr[offset + 3] << 12)
            | (arr[offset + 4] << 16)
            | (arr[offset + 5] << 20)
            | (arr[offset + 6] << 24)
            | (arr[offset + 7] << 28);
    }

    private static void Unpack8(uint packed, ref BoardSide arr, int offset)
    {
        arr[offset] = packed & 0x0Fu;
        arr[offset + 1] = (packed >> 4) & 0x0Fu;
        arr[offset + 2] = (packed >> 8) & 0x0Fu;
        arr[offset + 3] = (packed >> 12) & 0x0Fu;
        arr[offset + 4] = (packed >> 16) & 0x0Fu;
        arr[offset + 5] = (packed >> 20) & 0x0Fu;
        arr[offset + 6] = (packed >> 24) & 0x0Fu;
        arr[offset + 7] = (packed >> 28) & 0x0Fu;
    }

    /// <summary>
    /// Build the old-format position key (variable-length bit packing).
    /// Port of oldPositionKey() from positionid.c.
    /// anBoard[0] (Opponent) is encoded first, then anBoard[1] (Player).
    /// </summary>
    private static void OldPositionKey(Board board, Span<byte> oldKey)
    {
        int iBit = 0;

        // Encode both sides: first Opponent (anBoard[0]), then Player (anBoard[1])
        for (int side = 0; side < 2; side++)
        {
            ReadOnlySpan<uint> b = side == 0 ? board.Opponent.AsReadOnlySpan() : board.Player.AsReadOnlySpan();
            for (int j = 0; j < Constants.NumPoints; j++)
            {
                uint nc = b[j];
                if (nc > 0)
                {
                    AddBits(oldKey, iBit, nc);
                    iBit += (int)nc + 1;
                }
                else
                {
                    iBit++;
                }
            }
        }
    }

    /// <summary>
    /// Set nBits consecutive 1-bits starting at bitPos.
    /// Port of addBits() from positionid.c.
    /// </summary>
    private static void AddBits(Span<byte> key, int bitPos, uint nBits)
    {
        int k = bitPos / 8;
        int r = bitPos & 7;
        uint b = ((1u << (int)nBits) - 1) << r;

        key[k] |= (byte)b;

        if (k < 8)
        {
            key[k + 1] |= (byte)(b >> 8);
            key[k + 2] |= (byte)(b >> 16);
        }
        else if (k == 8)
        {
            key[k + 1] |= (byte)(b >> 8);
        }
    }

    /// <summary>
    /// Decode old-format position key back to a board.
    /// Port of oldPositionFromKey() from positionid.c.
    /// </summary>
    private static void OldPositionFromKey(ref Board board, ReadOnlySpan<byte> oldKey)
    {
        board.Opponent.AsSpan().Clear();
        board.Player.AsSpan().Clear();

        int side = 0; // 0 = Opponent, 1 = Player
        int point = 0;

        for (int byteIdx = 0; byteIdx < 10; byteIdx++)
        {
            byte cur = oldKey[byteIdx];
            for (int bitIdx = 0; bitIdx < 8; bitIdx++)
            {
                if ((cur & 1) != 0)
                {
                    if (side >= 2 || point >= Constants.NumPoints)
                        return;

                    if (side == 0)
                        board.Opponent[point]++;
                    else
                        board.Player[point]++;
                }
                else
                {
                    if (++point == Constants.NumPoints)
                    {
                        side++;
                        point = 0;
                    }
                }
                cur >>= 1;
            }
        }
    }

    /// <summary>
    /// Convert 10 bytes of old key to 14-character base64 position ID.
    /// Port of oldPositionIDFromKey() from positionid.c.
    /// </summary>
    private static string OldKeyToBase64(ReadOnlySpan<byte> oldKey)
    {
        Span<char> result = stackalloc char[Constants.PositionIdLength];
        int pchIdx = 0;
        int puchIdx = 0;

        for (int i = 0; i < 3; i++)
        {
            result[pchIdx++] = Base64Chars[oldKey[puchIdx] >> 2];
            result[pchIdx++] = Base64Chars[((oldKey[puchIdx] & 0x03) << 4) | (oldKey[puchIdx + 1] >> 4)];
            result[pchIdx++] = Base64Chars[((oldKey[puchIdx + 1] & 0x0F) << 2) | (oldKey[puchIdx + 2] >> 6)];
            result[pchIdx++] = Base64Chars[oldKey[puchIdx + 2] & 0x3F];
            puchIdx += 3;
        }

        result[pchIdx++] = Base64Chars[oldKey[puchIdx] >> 2];
        result[pchIdx++] = Base64Chars[(oldKey[puchIdx] & 0x03) << 4];

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
