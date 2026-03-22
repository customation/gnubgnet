// SPDX-License-Identifier: GPL-3.0-or-later
// Builds ONNX model graphs from GnuBgNet neural network weights.

using GnuBgNet.NeuralNet;

namespace GnuBgNet.Gpu.Onnx;

/// <summary>
/// Constructs an ONNX model from a <see cref="NeuralNetwork"/>.
/// The model represents a 2-layer feedforward network:
///   Input → Gemm(W_hidden, B_hidden) → Mul(β_h) → Sigmoid → Gemm(W_output, B_output) → Mul(β_o) → Sigmoid → Output
/// </summary>
internal static class OnnxModelBuilder
{
    /// <summary>
    /// Build an ONNX model and serialize to bytes.
    /// </summary>
    public static byte[] Build(NeuralNetwork network)
    {
        var model = new ModelProto
        {
            IrVersion = 8,
            OpsetImport = { new OperatorSetIdProto { Version = 13 } },
            Graph = BuildGraph(network),
        };

        using var ms = new MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, model);
        return ms.ToArray();
    }

    private static GraphProto BuildGraph(NeuralNetwork nn)
    {
        var graph = new GraphProto { Name = "gnubg_network" };

        // --- Input ---
        graph.Input.Add(MakeValueInfo("input", -1, nn.InputCount)); // -1 = dynamic batch

        // --- Initializers (weights & biases as constant tensors) ---
        // HiddenWeight: [InputCount, HiddenCount] — row i = weights from input i to all hidden
        graph.Initializer.Add(MakeTensor("hidden_weight", nn.HiddenWeight,
            [nn.InputCount, nn.HiddenCount]));
        graph.Initializer.Add(MakeTensor("hidden_bias", nn.HiddenThreshold,
            [nn.HiddenCount]));
        graph.Initializer.Add(MakeTensor("output_weight", nn.OutputWeight,
            [nn.HiddenCount, nn.OutputCount]));
        graph.Initializer.Add(MakeTensor("output_bias", nn.OutputThreshold,
            [nn.OutputCount]));
        graph.Initializer.Add(MakeScalar("beta_hidden", nn.BetaHidden));
        graph.Initializer.Add(MakeScalar("beta_output", nn.BetaOutput));

        // --- Nodes ---
        // 1. hidden_linear = input @ hidden_weight + hidden_bias
        graph.Node.Add(MakeGemm("gemm_hidden", "input", "hidden_weight", "hidden_bias", "hidden_linear"));

        // 2. hidden_scaled = hidden_linear * beta_hidden
        graph.Node.Add(MakeBinaryOp("mul_hidden", "Mul", "hidden_linear", "beta_hidden", "hidden_scaled"));

        // 3. hidden_act = sigmoid(hidden_scaled)
        graph.Node.Add(MakeUnaryOp("sigmoid_hidden", "Sigmoid", "hidden_scaled", "hidden_act"));

        // 4. output_linear = hidden_act @ output_weight + output_bias
        graph.Node.Add(MakeGemm("gemm_output", "hidden_act", "output_weight", "output_bias", "output_linear"));

        // 5. output_scaled = output_linear * beta_output
        graph.Node.Add(MakeBinaryOp("mul_output", "Mul", "output_linear", "beta_output", "output_scaled"));

        // 6. output = sigmoid(output_scaled)
        graph.Node.Add(MakeUnaryOp("sigmoid_output", "Sigmoid", "output_scaled", "output"));

        // --- Output ---
        graph.Output.Add(MakeValueInfo("output", -1, nn.OutputCount));

        return graph;
    }

    private static ValueInfoProto MakeValueInfo(string name, int batchDim, int featureDim)
    {
        var shape = new TensorShapeProto();
        if (batchDim < 0)
            shape.Dim.Add(new DimensionProto { DimParam = "batch" });
        else
            shape.Dim.Add(new DimensionProto { DimValue = batchDim });
        shape.Dim.Add(new DimensionProto { DimValue = featureDim });

        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                TensorType = new TensorTypeProto { ElemType = 1, Shape = shape },
            },
        };
    }

    private static TensorProto MakeTensor(string name, float[] data, long[] dims)
    {
        var t = new TensorProto { Name = name, DataType = 1 };
        t.Dims.AddRange(dims);
        t.FloatData.AddRange(data);
        return t;
    }

    private static TensorProto MakeScalar(string name, float value)
    {
        return new TensorProto
        {
            Name = name,
            DataType = 1,
            FloatData = { value },
            // Scalar: no dims (rank-0 tensor)
        };
    }

    private static NodeProto MakeGemm(string name, string a, string b, string c, string output)
    {
        return new NodeProto
        {
            Name = name,
            OpType = "Gemm",
            Input = { a, b, c },
            Output = { output },
            Attribute =
            {
                new AttributeProto { Name = "alpha", F = 1.0f, Type = 1 },
                new AttributeProto { Name = "beta", F = 1.0f, Type = 1 },
                new AttributeProto { Name = "transB", I = 0, Type = 2 },
            },
        };
    }

    private static NodeProto MakeBinaryOp(string name, string op, string a, string b, string output)
    {
        return new NodeProto
        {
            Name = name,
            OpType = op,
            Input = { a, b },
            Output = { output },
        };
    }

    private static NodeProto MakeUnaryOp(string name, string op, string input, string output)
    {
        return new NodeProto
        {
            Name = name,
            OpType = op,
            Input = { input },
            Output = { output },
        };
    }
}
