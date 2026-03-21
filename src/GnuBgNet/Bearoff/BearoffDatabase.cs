// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from bearoff.c + bearoff.h

using GnuBgNet.Encoding;

namespace GnuBgNet.Bearoff;

public enum BearoffType
{
    TwoSided,
    OneSided,
    Hypergammon,
}

/// <summary>
/// Loads and queries gnubg bearoff databases (.bd files).
/// Port of bearoffcontext from bearoff.c.
/// </summary>
public sealed class BearoffDatabase
{
    private const int HeaderSize = 40;

    public BearoffType Type { get; }
    public int Points { get; }
    public int Chequers { get; }
    public bool Cubeful { get; }    // Two-sided: includes cubeful equities
    public bool Gammon { get; }     // One-sided: includes gammon probabilities
    public bool Compressed { get; } // One-sided: compressed storage
    public bool NormalDist { get; } // One-sided: normal distribution approx
    public uint NumPositions { get; }

    private readonly byte[] _data;

    private BearoffDatabase(byte[] data, BearoffType type, int points, int chequers,
                            bool cubeful, bool gammon, bool compressed, bool normalDist)
    {
        _data = data;
        Type = type;
        Points = points;
        Chequers = chequers;
        Cubeful = cubeful;
        Gammon = gammon;
        Compressed = compressed;
        NormalDist = normalDist;
        NumPositions = PositionId.Combination((uint)(points + chequers), (uint)points);
    }

    /// <summary>
    /// Load a bearoff database from a .bd file.
    /// </summary>
    public static BearoffDatabase Load(string path)
    {
        var data = File.ReadAllBytes(path);
        return Load(data);
    }

    /// <summary>
    /// Load a bearoff database from raw bytes.
    /// </summary>
    public static BearoffDatabase Load(byte[] data)
    {
        if (data.Length < HeaderSize)
            throw new InvalidDataException("Bearoff database too small for header");

        // Validate magic "gnubg"
        if (data[0] != 'g' || data[1] != 'n' || data[2] != 'u' || data[3] != 'b' || data[4] != 'g')
            throw new InvalidDataException("Invalid bearoff database: missing 'gnubg' magic");

        // Type at offset 6-7 (or 6 for 'H')
        char t0 = (char)data[6];
        char t1 = (char)data[7];
        BearoffType type;
        int points, chequers;
        bool cubeful = false, gammon = false, compressed = false, normalDist = false;

        if (t0 == 'O' && t1 == 'S')
        {
            type = BearoffType.OneSided;
            points = ParseAsciiInt(data, 9, 2);
            chequers = ParseAsciiInt(data, 12, 2);
            gammon = data[15] == '1';
            compressed = data[17] == '1';
            normalDist = data[19] == '1';
        }
        else if (t0 == 'T' && t1 == 'S')
        {
            type = BearoffType.TwoSided;
            points = ParseAsciiInt(data, 9, 2);
            chequers = ParseAsciiInt(data, 12, 2);
            cubeful = data[15] == '1';
        }
        else if (t0 == 'H')
        {
            // Hypergammon: header format "gnubg-H<n>" where n = number of chequers (1, 2, or 3)
            type = BearoffType.Hypergammon;
            points = 25;
            chequers = t1 - '0';
            if (chequers < 1 || chequers > 3)
                throw new InvalidDataException($"Invalid hypergammon chequer count: {chequers}");
        }
        else
        {
            throw new InvalidDataException($"Unknown bearoff type: {t0}{t1}");
        }

        return new BearoffDatabase(data, type, points, chequers, cubeful, gammon, compressed, normalDist);
    }

    /// <summary>
    /// Evaluate a bearoff position, producing 5-element output probabilities.
    /// Port of BearoffEval from bearoff.c.
    /// </summary>
    public void Evaluate(Board board, Span<float> output)
    {
        switch (Type)
        {
            case BearoffType.TwoSided:
                EvaluateTwoSided(board, output);
                break;
            case BearoffType.OneSided:
                EvaluateOneSided(board, output);
                break;
            case BearoffType.Hypergammon:
                EvaluateHypergammon(board, output);
                break;
        }
    }

    /// <summary>
    /// Get cubeful equities for a two-sided position.
    /// Returns 4 values: [cubeless, cube-owned, cube-centered, cube-opponent].
    /// </summary>
    public void GetCubefulEquities(Board board, Span<float> equities)
    {
        if (Type != BearoffType.TwoSided || !Cubeful)
            throw new InvalidOperationException("Cubeful equities only available for two-sided cubeful databases");

        uint nUs = PositionId.PositionBearoff(board.Player, Points, Chequers);
        uint nThem = PositionId.PositionBearoff(board.Opponent, Points, Chequers);
        uint iPos = nUs * NumPositions + nThem;

        int offset = HeaderSize + (int)(iPos * 4) * 2; // 4 shorts per position

        for (int i = 0; i < 4; i++)
        {
            ushort raw = BitConverter.ToUInt16(_data, offset + i * 2);
            equities[i] = raw / 32767.5f - 1.0f;
        }
    }

    /// <summary>
    /// Get the raw probability distribution for a one-sided position.
    /// Returns P(bear off in exactly i rolls) for i=0..31.
    /// </summary>
    public void GetDistribution(uint posId, Span<float> probs)
    {
        if (Type != BearoffType.OneSided)
            throw new InvalidOperationException("Distribution only available for one-sided databases");

        Span<ushort> rawProbs = stackalloc ushort[32];
        ReadOneSidedRaw(posId, rawProbs, gammon: false);

        for (int i = 0; i < 32; i++)
            probs[i] = rawProbs[i] / 65535.0f;
    }

    /// <summary>
    /// Get the raw gammon probability distribution for a one-sided position.
    /// </summary>
    public void GetGammonDistribution(uint posId, Span<float> probs)
    {
        if (Type != BearoffType.OneSided || !Gammon)
            throw new InvalidOperationException("Gammon distribution only available for one-sided gammon databases");

        Span<ushort> rawProbs = stackalloc ushort[32];
        ReadOneSidedRaw(posId, rawProbs, gammon: true);

        for (int i = 0; i < 32; i++)
            probs[i] = rawProbs[i] / 65535.0f;
    }

    // ---- Hypergammon evaluation ----

    /// <summary>
    /// Evaluate a hypergammon position from the bearoff database.
    /// Port of BearoffEvalHypergammon from bearoff.c.
    /// </summary>
    private void EvaluateHypergammon(Board board, Span<float> output)
    {
        uint nUs = PositionId.PositionBearoff(board.Player, Points, Chequers);
        uint nThem = PositionId.PositionBearoff(board.Opponent, Points, Chequers);
        uint n = PositionId.Combination((uint)(Points + Chequers), (uint)Points);
        uint iPos = nUs * n + nThem;

        ReadHypergammon(iPos, output, default);
    }

    /// <summary>
    /// Evaluate a hypergammon position returning both probabilities and equities.
    /// Port of BearoffHyper from bearoff.c.
    /// </summary>
    public void EvaluateHypergammonFull(Board board, Span<float> output, Span<float> equities)
    {
        uint nUs = PositionId.PositionBearoff(board.Player, Points, Chequers);
        uint nThem = PositionId.PositionBearoff(board.Opponent, Points, Chequers);
        uint n = PositionId.Combination((uint)(Points + Chequers), (uint)Points);
        uint iPos = nUs * n + nThem;

        ReadHypergammon(iPos, output, equities);
    }

    /// <summary>
    /// Read hypergammon data: 28 bytes per position.
    /// 5 probability values x 3 bytes + 4 equity values x 3 bytes + 1 padding byte.
    /// Port of ReadHypergammon from bearoff.c.
    /// </summary>
    private void ReadHypergammon(uint iPos, Span<float> output, Span<float> equities)
    {
        const int BytesPerPosition = 28;
        int offset = HeaderSize + (int)(iPos * BytesPerPosition);

        if (output.Length >= Constants.NumOutputs)
        {
            for (int i = 0; i < Constants.NumOutputs; i++)
            {
                uint us = (uint)(_data[offset + 3 * i]
                    | (_data[offset + 3 * i + 1] << 8)
                    | (_data[offset + 3 * i + 2] << 16));
                output[i] = us / 16777215.0f;
            }
        }

        if (equities.Length >= 4)
        {
            for (int i = 0; i < 4; i++)
            {
                uint us = (uint)(_data[offset + 15 + 3 * i]
                    | (_data[offset + 15 + 3 * i + 1] << 8)
                    | (_data[offset + 15 + 3 * i + 2] << 16));
                equities[i] = (us / 16777215.0f - 0.5f) * 6.0f;
            }
        }
    }

    // ---- Two-sided evaluation ----

    private void EvaluateTwoSided(Board board, Span<float> output)
    {
        uint nUs = PositionId.PositionBearoff(board.Player, Points, Chequers);
        uint nThem = PositionId.PositionBearoff(board.Opponent, Points, Chequers);
        uint iPos = nUs * NumPositions + nThem;

        int bytesPerPos = Cubeful ? 8 : 2; // 4 shorts vs 1 short
        int offset = HeaderSize + (int)(iPos * (uint)bytesPerPos);

        ushort raw = BitConverter.ToUInt16(_data, offset);
        float equity = raw / 32767.5f - 1.0f;

        // Convert equity to output probabilities
        // equity = P(win) - P(lose) = 2*P(win) - 1 (for money/cubeless)
        output[Constants.OutputWin] = (equity + 1.0f) / 2.0f;
        output[Constants.OutputWinGammon] = 0.0f;
        output[Constants.OutputWinBackgammon] = 0.0f;
        output[Constants.OutputLoseGammon] = 0.0f;
        output[Constants.OutputLoseBackgammon] = 0.0f;
    }

    // ---- One-sided evaluation ----

    private void EvaluateOneSided(Board board, Span<float> output)
    {
        uint nUs = PositionId.PositionBearoff(board.Player, Points, Chequers);
        uint nThem = PositionId.PositionBearoff(board.Opponent, Points, Chequers);

        // Get probability distributions for both sides
        Span<float> probUs = stackalloc float[32];
        Span<float> probThem = stackalloc float[32];
        GetDistribution(nUs, probUs);
        GetDistribution(nThem, probThem);

        // Compute win probability:
        // P(win) = sum over all i: P_us(i) * P_them_cumulative(i-1)
        // where P_them_cumulative(j) = sum_{k>j} P_them(k) = prob opponent hasn't finished by roll j
        float pWin = 0.0f;
        float pThemStillOn = 1.0f; // cumulative: prob opponent hasn't finished

        for (int i = 1; i < 32; i++)
        {
            // We finish on roll i, opponent hasn't finished on roll i-1 or later
            // But opponent also rolls on roll i (simultaneous concept: we move first)
            // In gnubg: player to move finishes on roll i, opponent has rolls 1..i-1
            pWin += probUs[i] * pThemStillOn;
            pThemStillOn -= probThem[i];
        }

        // Gammon probabilities — only compute when all 15 checkers still on board
        float pWinGammon = 0.0f;
        float pLoseGammon = 0.0f;

        int anOn0 = 0, anOn1 = 0;
        for (int i = 0; i < 25; i++)
        {
            anOn0 += (int)board.Opponent[i];
            anOn1 += (int)board.Player[i];
        }

        if ((anOn0 == 15 || anOn1 == 15) && Gammon)
        {
            Span<float> gammonUs = stackalloc float[32];
            Span<float> gammonThem = stackalloc float[32];
            GetGammonDistribution(nUs, gammonUs);
            GetGammonDistribution(nThem, gammonThem);

            if (anOn0 == 15)
            {
                // Opponent has all checkers: can win gammon
                float pThemGammonStillOn = 1.0f;
                for (int i = 1; i < 32; i++)
                {
                    pWinGammon += probUs[i] * pThemGammonStillOn;
                    pThemGammonStillOn -= gammonThem[i];
                }
            }

            if (anOn1 == 15)
            {
                // Player has all checkers: can lose gammon
                // Subtract before accumulating for j > i semantics
                float pUsGammonStillOn = 1.0f;
                for (int i = 1; i < 32; i++)
                {
                    pUsGammonStillOn -= gammonUs[i];
                    pLoseGammon += probThem[i] * pUsGammonStillOn;
                }
            }
        }

        output[Constants.OutputWin] = pWin;
        output[Constants.OutputWinGammon] = pWinGammon;
        output[Constants.OutputWinBackgammon] = 0.0f;
        output[Constants.OutputLoseGammon] = pLoseGammon;
        output[Constants.OutputLoseBackgammon] = 0.0f;
    }

    // ---- Raw data reading ----

    private void ReadOneSidedRaw(uint posId, Span<ushort> probs, bool gammon)
    {
        for (int i = 0; i < 32; i++)
            probs[i] = 0;

        if (NormalDist)
        {
            // Normal distribution mode: 16 bytes per position
            ReadOneSidedND(posId, probs, gammon);
            return;
        }

        if (Compressed)
        {
            ReadOneSidedCompressed(posId, probs, gammon);
            return;
        }

        // Uncompressed: 64 bytes per position (128 if gammon)
        int bytesPerPos = Gammon ? 128 : 64;
        int baseOffset = HeaderSize + (int)posId * bytesPerPos;
        int dataOffset = gammon ? baseOffset + 64 : baseOffset;

        for (int i = 0; i < 32; i++)
        {
            probs[i] = BitConverter.ToUInt16(_data, dataOffset + i * 2);
        }
    }

    private void ReadOneSidedCompressed(uint posId, Span<ushort> probs, bool gammon)
    {
        int indexEntrySize = Gammon ? 8 : 6;
        int indexOffset = HeaderSize + (int)posId * indexEntrySize;

        uint iOffset = BitConverter.ToUInt32(_data, indexOffset);
        byte nz = _data[indexOffset + 4];
        byte ioff = _data[indexOffset + 5];

        if (gammon && Gammon)
        {
            nz = _data[indexOffset + 6];
            ioff = _data[indexOffset + 7];
        }

        // Data starts after all index entries
        int dataStart = HeaderSize + (int)NumPositions * indexEntrySize;
        int dataOffset = dataStart + (int)iOffset * 2;

        if (gammon && Gammon)
        {
            // For gammon data, the offset is cumulative from the same iOffset base
            // but we need the gammon-specific offset
            // Re-read the non-gammon nz to skip past it
            byte nzNormal = _data[indexOffset + 4];
            dataOffset = dataStart + ((int)iOffset + nzNormal) * 2;
        }

        for (int i = 0; i < nz && (ioff + i) < 32; i++)
        {
            probs[ioff + i] = BitConverter.ToUInt16(_data, dataOffset + i * 2);
        }
    }

    private void ReadOneSidedND(uint posId, Span<ushort> probs, bool gammon)
    {
        // 16 bytes per position: mu, sigma, mu_gammon, sigma_gammon (each float)
        int baseOffset = HeaderSize + (int)posId * 16;
        int floatOffset = gammon ? baseOffset + 8 : baseOffset;

        float mu = BitConverter.ToSingle(_data, floatOffset);
        float sigma = BitConverter.ToSingle(_data, floatOffset + 4);

        // Reconstruct distribution from normal distribution
        if (sigma <= 0.0f)
        {
            // Degenerate: all probability at mean
            int iMu = Math.Clamp((int)MathF.Round(mu), 0, 31);
            probs[iMu] = 65535;
            return;
        }

        float total = 0.0f;
        Span<float> tempProbs = stackalloc float[32];
        for (int i = 0; i < 32; i++)
        {
            float z = (i - mu) / sigma;
            float p = MathF.Exp(-0.5f * z * z) / (sigma * MathF.Sqrt(2.0f * MathF.PI));
            tempProbs[i] = p;
            total += p;
        }

        // Normalize and convert to ushort
        if (total > 0.0f)
        {
            for (int i = 0; i < 32; i++)
                probs[i] = (ushort)(tempProbs[i] / total * 65535.0f + 0.5f);
        }
    }

    private static int ParseAsciiInt(byte[] data, int offset, int maxLen)
    {
        int result = 0;
        for (int i = 0; i < maxLen; i++)
        {
            byte b = data[offset + i];
            if (b >= (byte)'0' && b <= (byte)'9')
                result = result * 10 + (b - '0');
        }
        return result;
    }
}
