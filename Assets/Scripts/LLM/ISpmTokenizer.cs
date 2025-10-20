using System;

public interface ISpmTokenizer
{
    int BosId { get; }
    int EosId { get; }
    int UnkId { get; }
    int PadId { get; }
    int[] Encode(string text, bool addBos = false, bool addEos = false);
    string Decode(ReadOnlySpan<int> ids);
}