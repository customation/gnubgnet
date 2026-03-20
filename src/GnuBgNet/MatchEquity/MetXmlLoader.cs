// Copyright (C) 1998-2002 Gary Wong <gtw@gnu.org>
// Copyright (C) 2000-2019 the AUTHORS
// SPDX-License-Identifier: GPL-3.0-or-later
// Port of readMET / met_parser from matchequity.c

using System.Xml;

namespace GnuBgNet.MatchEquity;

/// <summary>
/// Loads match equity tables from gnubg's XML MET file format.
/// Port of readMET() from matchequity.c.
/// Supports "explicit" type (pre-computed values) and falls back to
/// Zadeh computation for "zadeh"/"mec" types.
/// </summary>
public static class MetXmlLoader
{
    /// <summary>
    /// Load a match equity table from an XML file.
    /// </summary>
    public static MatchEquityTable LoadFromFile(string path)
    {
        var xml = File.ReadAllText(path);
        return LoadFromXml(xml);
    }

    /// <summary>
    /// Load a match equity table from XML string content.
    /// </summary>
    public static MatchEquityTable LoadFromXml(string xml)
    {
        var met = new MatchEquityTable();
        var doc = new XmlDocument();
        doc.XmlResolver = null; // Skip external DTD resolution
        using var reader = new XmlTextReader(new StringReader(xml))
        {
            DtdProcessing = DtdProcessing.Ignore
        };
        doc.Load(reader);

        var root = doc.DocumentElement;
        if (root == null || root.Name != "met")
            throw new InvalidDataException("Invalid MET XML: root element must be 'met'");

        // Parse pre-Crawford table
        var preCrawford = root.SelectSingleNode("pre-crawford-table");
        if (preCrawford != null)
        {
            string tableType = preCrawford.Attributes?["type"]?.Value ?? "explicit";
            if (tableType == "explicit")
                ParseExplicitPreCrawford(preCrawford, met);
            // For "zadeh" or "mec" types, we'd use the computed default
            // which is already the Zadeh model in MatchEquityTable.ComputeDefault()
        }

        // Parse post-Crawford table(s)
        foreach (XmlNode node in root.SelectNodes("post-crawford-table")!)
        {
            string tableType = node.Attributes?["type"]?.Value ?? "explicit";
            string player = node.Attributes?["player"]?.Value ?? "both";

            if (tableType == "explicit")
                ParseExplicitPostCrawford(node, met, player);
        }

        return met;
    }

    private static void ParseExplicitPreCrawford(XmlNode node, MatchEquityTable met)
    {
        int row = 0;
        foreach (XmlNode rowNode in node.SelectNodes("row")!)
        {
            if (row >= Constants.MaxScore) break;
            int col = 0;
            foreach (XmlNode meNode in rowNode.SelectNodes("me")!)
            {
                if (col >= Constants.MaxScore) break;
                if (float.TryParse(meNode.InnerText.Trim(),
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out float value))
                {
                    met.Met[row, col] = value;
                }
                col++;
            }
            row++;
        }
    }

    private static void ParseExplicitPostCrawford(XmlNode node, MatchEquityTable met, string player)
    {
        int side0 = 0, side1 = 1;
        bool bothSides = player == "both";

        if (player == "0") { side0 = 0; side1 = -1; }
        else if (player == "1") { side0 = -1; side1 = 1; }

        var rows = node.SelectNodes("row");
        if (rows == null || rows.Count == 0) return;

        // For "both" player, single row applies to both sides
        // For specific player, row applies to that side only
        int rowIdx = 0;
        foreach (XmlNode rowNode in rows)
        {
            int col = 0;
            foreach (XmlNode meNode in rowNode.SelectNodes("me")!)
            {
                if (col >= Constants.MaxScore) break;
                if (float.TryParse(meNode.InnerText.Trim(),
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out float value))
                {
                    if (bothSides)
                    {
                        met.PostCrawford[0, col] = value;
                        met.PostCrawford[1, col] = value;
                    }
                    else if (rowIdx == 0 && side0 >= 0)
                    {
                        met.PostCrawford[side0, col] = value;
                    }
                    else if (rowIdx == 1 && side1 >= 0)
                    {
                        met.PostCrawford[side1, col] = value;
                    }
                }
                col++;
            }
            rowIdx++;
        }
    }
}
