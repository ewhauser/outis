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
from score_aligned import bare_year_characters, gold_values  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
