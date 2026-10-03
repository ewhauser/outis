"""Recorded timing wording checks using invented fields and no saved artifacts."""

from copy import deepcopy
import unittest

from clinical_deid.report_html import _timing


COMPLETE_NOTE = (
    "Cached timing composition: neural inference was not performed in this evaluation. "
    "The recorded total and throughput combine cached neural timing with this rules pass; "
    "they are not measurements of one end-to-end run."
)
COMPLETE = {
    "neural_inference_performed": False,
    "neural_seconds": 10.0,
    "rules_seconds": 2.0,
    "total_seconds": 12.0,
    "documents_per_second": 4 / 12,
}
FIELDS = ("neural_seconds", "rules_seconds", "total_seconds", "documents_per_second")


class TimingWordingTests(unittest.TestCase):
    def test_no_timings_do_not_claim_cached_composition(self):
        rendered = _timing({"timing": {"neural_inference_performed": False}})
        self.assertIn("Neural inference was not performed", rendered)
        self.assertIn("No usable timing measurements were recorded", rendered)
        self.assertIn("total time and throughput are unavailable", rendered)
        self.assertNotIn("Cached timing composition", rendered)
        self.assertNotIn("combine cached neural timing", rendered)
        self.assertNotIn("stage-composition", rendered)
        self.assertIn("<strong>Unavailable</strong>", rendered)

    def test_explicit_nulls_are_unavailable(self):
        rendered = _timing({"timing": {"neural_inference_performed": False, **dict.fromkeys(FIELDS)}})
        self.assertIn("No usable timing measurements were recorded", rendered)
        self.assertNotIn("Timing information is incomplete", rendered)

    def test_invalid_values_do_not_count_as_available(self):
        values = dict(zip(FIELDS, (True, -1.0, float("nan"), "100")))
        rendered = _timing({"timing": {"neural_inference_performed": False, **values}})
        self.assertIn("No usable timing measurements were recorded", rendered)
        self.assertNotIn("Cached timing composition", rendered)

    def test_every_single_recorded_field_is_partial(self):
        for field in FIELDS:
            for value in (0, 2.0):
                with self.subTest(field=field, value=value):
                    rendered = _timing({"timing": {"neural_inference_performed": False, field: value}})
                    self.assertIn("Timing information is incomplete", rendered)
                    self.assertIn("Any recorded neural timing is reused", rendered)
                    self.assertNotIn("No usable timing measurements were recorded", rendered)
                    self.assertNotIn("combine cached neural timing", rendered)

    def test_each_missing_field_prevents_complete_composition_claim(self):
        for field in FIELDS:
            timing = {**COMPLETE}
            timing.pop(field)
            with self.subTest(field=field):
                rendered = _timing({"timing": timing})
                self.assertIn("Timing information is incomplete", rendered)
                self.assertNotIn("combine cached neural timing", rendered)

    def test_complete_cached_timings_preserve_existing_wording_and_display(self):
        rendered = _timing({"timing": COMPLETE})
        self.assertIn(COMPLETE_NOTE, rendered)
        self.assertIn("<strong>0.333</strong>", rendered)
        self.assertIn("10.000 s", rendered)
        self.assertIn("2.000 s", rendered)
        self.assertIn("12.000 s", rendered)
        self.assertIn("stage-composition", rendered)

    def test_zero_total_without_rate_is_partial_and_no_bar_is_invented(self):
        timing = {**COMPLETE, "neural_seconds": 0.0, "rules_seconds": 0.0, "total_seconds": 0.0, "documents_per_second": None}
        rendered = _timing({"timing": timing})
        self.assertIn("Timing information is incomplete", rendered)
        self.assertIn("<strong>Unavailable</strong>", rendered)
        self.assertNotIn("stage-composition", rendered)

    def test_inference_true_and_unknown_wording_are_preserved(self):
        rendered = _timing({"timing": {**COMPLETE, "neural_inference_performed": True}})
        self.assertIn("Neural inference was performed for this evaluation.", rendered)
        self.assertNotIn("Cached timing composition", rendered)
        for timing in ({}, {"neural_inference_performed": None}):
            self.assertIn("Whether neural inference was performed is unavailable.", _timing({"timing": timing}))

    def test_other_elapsed_fields_are_not_timing_evidence_and_input_is_unchanged(self):
        run = {"elapsed_seconds": 9.0, "timing": {"neural_inference_performed": False, "device": "cpu", "elapsed_seconds": 9.0}}
        before = deepcopy(run)
        rendered = _timing(run)
        self.assertIn("No usable timing measurements were recorded", rendered)
        self.assertNotIn("9.000", rendered)
        self.assertEqual(run, before)

