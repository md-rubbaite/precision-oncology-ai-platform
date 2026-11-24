from pathlib import Path

import pytest
from conftest import HEADER, maf

from oncoai.maf import Filters, MafError, Mutation, patient_id, read_maf

FULL = HEADER + "\tChromosome\tStart_Position\tHGVSp_Short\tt_ref_count\tt_alt_count\tCONTEXT"


def row(
    sample="S1",
    gene="TP53",
    vclass="Missense_Mutation",
    vtype="SNP",
    ref="C",
    alt="T",
    chrom="17",
    pos="100",
    prot="p.R1H",
    tr="20",
    ta="10",
    ctx="AAAAACAAAAA",
):
    return "\t".join([gene, sample, vclass, vtype, ref, alt, chrom, pos, prot, tr, ta, ctx])


def test_reads_minimal_file_and_skips_comments():
    text = "#version 2.4\n" + maf("TP53\tS1\tMissense_Mutation\tSNP\tC\tT", "KRAS\tS2\tSilent\tSNP\tG\tA")
    data = read_maf(text)
    assert data.qc.rows == 2 and data.qc.kept == 2 and data.qc.samples == 2
    assert {m.gene for m in data.mutations} == {"TP53", "KRAS"}


def test_missing_required_column_is_fatal():
    with pytest.raises(MafError, match="Tumor_Seq_Allele2"):
        read_maf("Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\tVariant_Type\tReference_Allele\nA\tB\tC\tD\tE\n")


def test_empty_input_has_no_header():
    with pytest.raises(MafError, match="no header"):
        read_maf("#only a comment\n")


def test_warnings_when_context_and_counts_missing():
    data = read_maf(maf("TP53\tS1\tMissense_Mutation\tSNP\tC\tT"))
    assert any("CONTEXT" in w for w in data.qc.warnings)
    assert any("read counts" in w for w in data.qc.warnings)


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("TP53\t\tMissense_Mutation\tSNP\tC\tT", "empty sample id"),
        ("TP53\tjohn doe@example.com\tMissense_Mutation\tSNP\tC\tT", "sample id with unexpected characters (use pseudonymous ids)"),
        ("\tS1\tMissense_Mutation\tSNP\tC\tT", "empty gene symbol"),
        ("TP53\tS1\tMissense_Mutation\tXYZ\tC\tT", "unknown Variant_Type"),
        ("TP53\tS1\tMissense_Mutation\tSNP\tC\tZ", "invalid allele"),
        ("TP53\tS1\tMissense_Mutation\tSNP\tC\tC", "reference equals alternate allele"),
    ],
)
def test_rejections_are_counted_with_reason_and_line(line, reason):
    data = read_maf(maf(line, "KRAS\tS2\tSilent\tSNP\tG\tA"))
    assert data.qc.rejected[reason] == 1
    assert data.qc.examples[reason] == 2
    assert data.qc.kept == 1


def test_duplicate_rows_are_dropped():
    data = read_maf("\n".join([FULL, row(), row(), row(pos="200")]))
    assert data.qc.kept == 2 and data.qc.rejected["duplicate row"] == 1


def test_read_filters():
    rows = [row(ta="1", tr="99", pos="1"), row(ta="5", tr="95", pos="2"), row(ta="30", tr="70", pos="3")]
    text = "\n".join([FULL, *rows])
    assert read_maf(text, Filters(min_alt_reads=3)).qc.kept == 2
    assert read_maf(text, Filters(min_vaf=0.1)).qc.kept == 1
    assert read_maf(text).qc.kept == 3


def test_unknown_class_is_kept_with_warning_and_not_counted_as_protein_changing():
    data = read_maf(maf("TP53\tS1\tMystery_Class\tSNP\tC\tT"))
    assert data.qc.kept == 1
    assert not data.mutations[0].nonsynonymous
    assert any("Mystery_Class" in w for w in data.qc.warnings)


def test_max_rows_guards_untrusted_input():
    text = maf(*[f"G{i}\tS1\tSilent\tSNP\tC\tT" for i in range(5)])
    with pytest.raises(MafError, match="more than 3"):
        read_maf(text, max_rows=3)


def test_crlf_short_rows_and_lowercase_alleles():
    text = "\r\n".join([FULL, "TP53\tS1\tMissense_Mutation\tsnp\tc\tt"]) + "\r\n"
    m = read_maf(text).mutations[0]
    assert (m.vtype, m.ref, m.alt, m.context, m.t_alt) == ("SNP", "C", "T", "", None)


def test_mutation_fields_and_vaf():
    m = read_maf("\n".join([FULL, row(chrom="chr17")])).mutations[0]
    assert m.chrom == "17" and m.pos == 100 and m.protein == "p.R1H"
    assert m.vaf == pytest.approx(10 / 30)
    assert m.is_snv and m.nonsynonymous


def test_bad_context_is_ignored():
    m = read_maf("\n".join([FULL, row(ctx="AAXAA")])).mutations[0]
    assert m.context == ""


def test_vaf_none_without_counts():
    assert Mutation("S", "G", "Silent", "SNP", "C", "T").vaf is None
    assert Mutation("S", "G", "Silent", "SNP", "C", "T", t_ref=0, t_alt=0).vaf is None


def test_reads_from_path(tmp_path: Path):
    path = tmp_path / "x.maf"
    path.write_text(maf("TP53\tS1\tMissense_Mutation\tSNP\tC\tT"), encoding="utf-8")
    assert read_maf(path).qc.kept == 1


def test_by_sample_groups_rows():
    data = read_maf(maf("A\tS1\tSilent\tSNP\tC\tT", "B\tS1\tSilent\tSNP\tC\tA", "C\tS2\tSilent\tSNP\tC\tG"))
    groups = data.by_sample()
    assert sorted(groups) == ["S1", "S2"] and len(groups["S1"]) == 2


def test_patient_id():
    assert patient_id("TCGA-02-2466-01") == "TCGA-02-2466"
    assert patient_id("SAMPLE-7") == "SAMPLE-7"


def test_negative_and_text_counts_are_treated_as_missing():
    data = read_maf("\n".join([FULL, row(tr="-5", ta="NA")]))
    assert data.mutations[0].t_ref is None and data.mutations[0].t_alt is None
