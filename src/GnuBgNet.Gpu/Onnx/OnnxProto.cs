// SPDX-License-Identifier: GPL-3.0-or-later
// Minimal ONNX protobuf types for model construction.
// Field numbers match the ONNX IR spec (onnx.proto3).

using ProtoBuf;

namespace GnuBgNet.Gpu.Onnx;

[ProtoContract]
internal class ModelProto
{
    [ProtoMember(1)] public long IrVersion { get; set; } = 8;
    [ProtoMember(7)] public GraphProto? Graph { get; set; }
    [ProtoMember(8)] public List<OperatorSetIdProto> OpsetImport { get; set; } = [];
}

[ProtoContract]
internal class OperatorSetIdProto
{
    [ProtoMember(2)] public long Version { get; set; }
}

[ProtoContract]
internal class GraphProto
{
    [ProtoMember(1)] public List<NodeProto> Node { get; set; } = [];
    [ProtoMember(2)] public string Name { get; set; } = "";
    [ProtoMember(5)] public List<TensorProto> Initializer { get; set; } = [];
    [ProtoMember(11)] public List<ValueInfoProto> Input { get; set; } = [];
    [ProtoMember(12)] public List<ValueInfoProto> Output { get; set; } = [];
}

[ProtoContract]
internal class NodeProto
{
    [ProtoMember(1)] public List<string> Input { get; set; } = [];
    [ProtoMember(2)] public List<string> Output { get; set; } = [];
    [ProtoMember(3)] public string Name { get; set; } = "";
    [ProtoMember(4)] public string OpType { get; set; } = "";
    [ProtoMember(5)] public List<AttributeProto> Attribute { get; set; } = [];
}

[ProtoContract]
internal class AttributeProto
{
    [ProtoMember(1)] public string Name { get; set; } = "";
    [ProtoMember(4)] public float F { get; set; }
    [ProtoMember(3)] public long I { get; set; }
    [ProtoMember(20)] public int Type { get; set; } // 1=FLOAT, 2=INT
}

[ProtoContract]
internal class TensorProto
{
    [ProtoMember(1, IsPacked = true)] public List<long> Dims { get; set; } = [];
    [ProtoMember(2)] public int DataType { get; set; } = 1; // FLOAT
    [ProtoMember(4, IsPacked = true)] public List<float> FloatData { get; set; } = [];
    [ProtoMember(8)] public string Name { get; set; } = "";
}

[ProtoContract]
internal class ValueInfoProto
{
    [ProtoMember(1)] public string Name { get; set; } = "";
    [ProtoMember(2)] public TypeProto? Type { get; set; }
}

[ProtoContract]
internal class TypeProto
{
    [ProtoMember(1)] public TensorTypeProto? TensorType { get; set; }
}

[ProtoContract]
internal class TensorTypeProto
{
    [ProtoMember(1)] public int ElemType { get; set; } = 1; // FLOAT
    [ProtoMember(2)] public TensorShapeProto? Shape { get; set; }
}

[ProtoContract]
internal class TensorShapeProto
{
    [ProtoMember(1)] public List<DimensionProto> Dim { get; set; } = [];
}

[ProtoContract]
internal class DimensionProto
{
    [ProtoMember(1)] public long DimValue { get; set; }
    [ProtoMember(2)] public string? DimParam { get; set; }
}
