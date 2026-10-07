"""Behavioral tests for the external benchmark conversion and scoring helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asq_phi import align, parse  # noqa: E402
from common import windows  # noqa: E402
from labels import project_label  # noqa: E402
from score import age_characters, merge_names  # noqa: E402
from score_aligned import bare_year_characters, convention_flags, gold_values, miss_shapes  # noqa: E402
import common  # noqa: E402
import n2c2_2014  # noqa: E402


class AsqPhiConversionTest(unittest.TestCase):
    def test_parse_reads_queries_and_hard_negatives(self):
        records = parse('===QUERY===\nSaw Anna S. today?\n===PHI_TAGS===\n'
                        '{"identifier_type": "NAME", "value": "Anna S."}\n\n'
                        '===QUERY===\nNo identifiers here.\n===PHI_TAGS===\n')
        self.assertEqual(records, [("Saw Anna S. today?", [{"identifier_type": "NAME", "value": "Anna S."}]),
                                   ("No identifiers here.", [])])

    def test_every_whole_token_occurrence_is_gold(self):
        query = "MRN 4455 confirmed; recheck MRN 4455 not 44556."
        spans, records = align(query, [{"identifier_type": "MEDICAL_RECORD_NUMBER", "value": "4455"}])
        self.assertEqual([query[s["start"]:s["end"]] for s in spans], ["4455", "4455"])
        self.assertEqual(records[0]["occurrences"], 2)
        self.assertTrue(all(s["label"] == "MRN" for s in spans))

    def test_apostrophe_normalisation_keeps_original_offsets(self):
        query = "Seen at St. Vincent’s on May 3."
        spans, records = align(query, [{"identifier_type": "GEOGRAPHIC_LOCATION", "value": "St. Vincent's"}])
        self.assertEqual(records[0]["method"], "apostrophe")
        self.assertEqual(query[spans[0]["start"]:spans[0]["end"]], "St. Vincent’s")

    def test_nested_value_is_absorbed_by_longer_tag(self):
        query = "Admitted Feb 21, 2023 for review."
        spans, _ = align(query, [{"identifier_type": "DATE", "value": "Feb 21, 2023"},
                                 {"identifier_type": "DATE", "value": "2023"}])
        self.assertEqual([s["text"] for s in spans], ["Feb 21, 2023"])

    def test_unaligned_value_is_reported(self):
        _, records = align("Nothing to see.", [{"identifier_type": "NAME", "value": "Zed Q."}])
        self.assertEqual(records[0]["method"], "unaligned")

    def test_undeclared_source_label_is_rejected(self):
        with self.assertRaises(KeyError):
            align("x", [{"identifier_type": "FREE_TEXT", "value": "x"}])


class ScoringHelpersTest(unittest.TestCase):
    def test_windows_cover_every_character(self):
        text = " ".join(f"word{i}" for i in range(400))
        covered = set()
        for offset, excerpt in windows(text, 300, 50):
            self.assertEqual(text[offset:offset + len(excerpt)], excerpt)
            covered.update(range(offset, offset + len(excerpt)))
        self.assertEqual(covered, set(range(len(text))))

    def test_adjacent_names_merge_but_other_labels_do_not(self):
        text = "Anna S. on 2023 05"
        spans = [{"start": 0, "end": 4, "label": "NAME"}, {"start": 5, "end": 7, "label": "NAME"},
                 {"start": 11, "end": 15, "label": "DATE"}, {"start": 16, "end": 18, "label": "DATE"}]
        merged = merge_names(spans, text)
        self.assertEqual([(s["start"], s["end"]) for s in merged], [(0, 7), (11, 15), (16, 18)])

    def test_names_separated_by_text_stay_separate(self):
        text = "Anna and Bob"
        spans = [{"start": 0, "end": 4, "label": "NAME"}, {"start": 9, "end": 12, "label": "NAME"}]
        self.assertEqual(len(merge_names(spans, text)), 2)

    def test_age_adjustment_covers_ages_below_ninety_only(self):
        text = "A 34-year-old and a 70yo M and a 92-year-old."
        chars = age_characters(text)
        self.assertIn(text.index("34"), chars)
        self.assertIn(text.index("70yo"), chars)
        self.assertNotIn(text.index("92"), chars)

    def test_label_mapping(self):
        self.assertEqual(project_label("meddeid", "Contactdetails", "a@b.org"), "EMAIL")
        self.assertEqual(project_label("meddeid", "Contactdetails", "(555) 010-2000"), "PHONE")
        self.assertIsNone(project_label("meddeid", "Profession", "nurse"))
        self.assertEqual(project_label("gliner-facility", "hospital or medical facility", "Mt. Sinai"), "FACILITY")
        with self.assertRaises(KeyError):
            project_label("gliner-trained", "hospital or medical facility", "Mt. Sinai")


class PolicyAlignedNeutralTest(unittest.TestCase):
    def kept(self, text, value, label):
        start = text.index(value)
        values = gold_values(text, [{"start": start, "end": start + len(value), "label": label}])
        return "".join(text[i] for i in sorted(values[0][1])) if values else None

    def test_honorific_and_credentials_are_neutral(self):
        self.assertEqual(self.kept("Seen by Dr. Anna Smith, MD today.", "Dr. Anna Smith, MD", "NAME"), "AnnaSmith")
        self.assertEqual(self.kept("Validated by: Dr Elys Tate, MBBS", "Dr Elys Tate, MBBS", "NAME"), "ElysTate")
        self.assertEqual(self.kept("Signed, Mae Ong, MBChB, MRCP", "Mae Ong, MBChB, MRCP", "NAME"), "MaeOng")

    def test_spelled_out_professional_titles_are_neutral(self):
        self.assertEqual(self.kept("Validated by Doctor Geraert.", "Doctor Geraert", "NAME"), "Geraert")
        self.assertEqual(self.kept("by Prof. Dr. Iepermans today", "Prof. Dr. Iepermans", "NAME"), "Iepermans")
        self.assertEqual(self.kept("Radiologist Jan Aerschot, Doctor", "Jan Aerschot, Doctor", "NAME"), "JanAerschot")
        self.assertEqual(self.kept("Mr De Meyer received it", "De Meyer", "NAME"), "DeMeyer")

    def test_age_units_are_neutral_and_young_ages_count(self):
        self.assertEqual(self.kept("The patient is 23 years old.", "23 years", "AGE"), "23")
        self.assertEqual(self.kept("A 65-year-old man", "65-year-old", "AGE"), "65")
        self.assertEqual(self.kept("aged sixty-five years", "sixty-five years", "AGE"), "sixtyfive")

    def test_facility_generic_words_are_neutral(self):
        self.assertEqual(self.kept("seen at the burn centre at Middelheim.", "burn centre at Middelheim", "FACILITY"),
                         "Middelheim")
        self.assertEqual(self.kept("admitted to Leuven University Hospital.", "Leuven University Hospital", "FACILITY"),
                         "LeuvenUniversityHospital")

    def test_country_or_province_alone_is_neutral_but_city_is_not(self):
        self.assertIsNone(self.kept("moved from the province of Limburg", "province of Limburg", "LOCATION"))
        self.assertIsNone(self.kept("born in Belgium", "Belgium", "LOCATION"))
        self.assertEqual(self.kept("holiday home in Nice", "Nice", "LOCATION"), "Nice")
        self.assertEqual(self.kept("lives in Antwerp", "Antwerp", "LOCATION"), "Antwerp")

    def test_identifier_field_label_and_hash_are_neutral(self):
        self.assertEqual(self.kept("(Patient ID: ABCD1234)", "Patient ID: ABCD1234", "ID"), "ABCD1234")
        self.assertEqual(self.kept("plan #DB-2345678.", "#DB-2345678", "ID"), "DB2345678")

    def test_relative_dates_are_neutral_but_month_names_stay(self):
        self.assertIsNone(self.kept("seen last week at clinic", "last week", "DATE"))
        self.assertEqual(self.kept("seen last December", "last December", "DATE"), "December")
        self.assertEqual(self.kept("on May 5th, 2021", "May 5th, 2021", "DATE"), "May5th2021")

    def test_location_keeps_proper_names_only(self):
        self.assertEqual(self.kept("our New York clinic", "New York clinic", "LOCATION"), "NewYork")
        self.assertEqual(self.kept("Mayo Clinic in Rochester, MN", "Mayo Clinic in Rochester, MN", "LOCATION"),
                         "MayoClinicRochesterMN")
        self.assertIsNone(self.kept("the county hospital", "county hospital", "LOCATION"))

    def test_bare_years_only(self):
        text = "Dx in 2021; seen 12/11/1958 and 2020-03-04."
        self.assertEqual("".join(text[i] for i in sorted(bare_year_characters(text))), "2021")


SAMPLE_N2C2 = (b'<?xml version="1.0" encoding="UTF-8" ?>\n<deIdi2b2><TEXT><![CDATA[Record date: 2091-04-02\n'
               b'Lorna Vexley, 64, a welder, seen by Dr. Quill at Ombrey General in Tollund, Ohio.\n]]></TEXT><TAGS>'
               b'<DATE id="P0" start="13" end="23" text="2091-04-02" TYPE="DATE" comment="" />'
               b'<NAME id="P1" start="24" end="36" text="Lorna Vexley" TYPE="PATIENT" comment="" />'
               b'<AGE id="P2" start="38" end="40" text="64" TYPE="AGE" comment="" />'
               b'<PROFESSION id="P3" start="44" end="50" text="welder" TYPE="PROFESSION" comment="" />'
               b'<NAME id="P4" start="64" end="69" text="Quill" TYPE="DOCTOR" comment="" />'
               b'<LOCATION id="P5" start="0" end="0" text="Ombrey General" TYPE="HOSPITAL" comment="" />'
               b'<LOCATION id="P6" start="91" end="98" text="Tollund" TYPE="CITY" comment="" />'
               b'<LOCATION id="P7" start="100" end="104" text="Ohio" TYPE="STATE" comment="" />'
               b'</TAGS></deIdi2b2>')


class N2c2ConversionTest(unittest.TestCase):
    def test_restricted_paths_stay_outside_git(self):
        import os
        import tempfile
        old = os.environ.get("DEID_RESTRICTED_DIR")
        try:
            os.environ["DEID_RESTRICTED_DIR"] = str(common.ROOT / "tmp_restricted")
            with self.assertRaises(ValueError):
                common.restricted_root()
            with tempfile.TemporaryDirectory() as outside:
                os.environ["DEID_RESTRICTED_DIR"] = outside
                self.assertTrue(str(common.corpus_path("n2c2_2014_test")).startswith(str(Path(outside).resolve())))
                self.assertEqual(common.corpus_path("asq_phi").parent, common.CORPORA)
        finally:
            if old is None:
                os.environ.pop("DEID_RESTRICTED_DIR", None)
            else:
                os.environ["DEID_RESTRICTED_DIR"] = old

    def test_converter_maps_types_aligns_offsets_and_drops_profession(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "100-01.xml"
            path.write_bytes(SAMPLE_N2C2)
            rows, report = n2c2_2014.convert_files([path])
        row = rows[0]
        got = [(row["text"][s["start"]:s["end"]], s["label"], s["source_label"], s["in_policy"]) for s in row["spans"]]
        self.assertEqual(got, [
            ("2091-04-02", "DATE", "DATE", True),
            ("Lorna Vexley", "NAME", "PATIENT", True),
            ("64", "AGE", "AGE", True),
            ("Quill", "NAME", "DOCTOR", True),
            ("Ombrey General", "FACILITY", "HOSPITAL", True),
            ("Tollund", "LOCATION", "CITY", True),
            ("Ohio", "LOCATION", "STATE", False),
        ])
        self.assertEqual(report["alignment_methods"], {"exact": 7, "nearest": 1})
        self.assertEqual(len(row["source_tags"]), 8)

    def test_policy_aligned_view_neutralises_out_of_policy_spans_and_keeps_all_ages(self):
        text = "Lorna Vexley, 64, from Ohio; aged 93."
        spans = [{"start": 0, "end": 12, "label": "NAME"},
                 {"start": 14, "end": 16, "label": "AGE"},
                 {"start": 23, "end": 27, "label": "LOCATION", "in_policy": False},
                 {"start": 34, "end": 36, "label": "AGE"}]
        kept = [label for label, _ in gold_values(text, spans)]
        self.assertEqual(kept, ["NAME", "AGE", "AGE"])

    def test_unknown_subtype_is_an_error(self):
        import tempfile
        bad = SAMPLE_N2C2.replace(b'TYPE="CITY"', b'TYPE="PLANET"')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.xml"
            path.write_bytes(bad)
            with self.assertRaises(ValueError):
                n2c2_2014.convert_files([path])


class MissShapeTest(unittest.TestCase):
    def shapes(self, text, value, covered):
        start = text.index(value)
        chars = {i for i in range(start, start + len(value)) if text[i].isalnum()}
        cov = text.index(covered)
        missing = chars - set(range(cov, cov + len(covered)))
        return miss_shapes(text, chars, missing)

    def test_trailing_credential_is_a_caps_abbreviation(self):
        self.assertEqual(self.shapes("by Elys Tate, MBBS today", "Elys Tate, MBBS", "Elys Tate"),
                         ["trailing:caps_abbreviation"])

    def test_names_in_an_all_caps_note_are_not_abbreviations(self):
        self.assertEqual(self.shapes("SEEN BY ELYS TATE TODAY", "ELYS TATE", "ELYS"), ["trailing:other"])

    def test_leading_title_and_lowercase_words(self):
        self.assertEqual(self.shapes("seen by Sister Mae Ong", "Sister Mae Ong", "Mae Ong"),
                         ["leading:title_like_word"])
        self.assertEqual(self.shapes("our Chicago clinic", "Chicago clinic", "Chicago"),
                         ["trailing:lowercase_words"])

    def test_flags_need_most_systems_to_share_a_recurring_shape(self):
        def result(n):
            return {"misses": {"NAME": {"partial_shapes": {"trailing:caps_abbreviation": n}}}}
        self.assertEqual(convention_flags([result(27), result(27), result(0)]),
                         [("NAME", "trailing:caps_abbreviation")])
        self.assertEqual(convention_flags([result(27), result(2), result(0)]), [])


if __name__ == "__main__":
    unittest.main()
