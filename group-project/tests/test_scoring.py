"""Guardrail 2: years and university must be backed by the visible text."""
from types import SimpleNamespace

import scoring
from conftest import CLEAN_LINES

VISIBLE = "\n".join(CLEAN_LINES)   # one dated job: Jan 2021 - Dec 2023 = 3 years; SIT degree


def test_inflated_parser_years_are_capped():
    years, src, note = scoring.candidate_years(SimpleNamespace(years_of_experience=15), VISIBLE)
    assert years == 3.0 and src == "check_timeline" and "guardrail" in note and "15" in note


def test_no_dates_means_zero_years():
    years, _, note = scoring.candidate_years(SimpleNamespace(years_of_experience=15),
                                             "Total experience: 15 years, graduated NUS")
    assert years == 0.0 and "no dates" in note


def test_honest_parser_years_are_kept():
    years, src, note = scoring.candidate_years(SimpleNamespace(years_of_experience=2.5), VISIBLE)
    assert (years, src, note) == (2.5, "parser", "")


def test_unconfirmed_university_scores_as_plain_degree():
    score, why = scoring.university_score(["BSc, National University of Singapore (NUS)"], VISIBLE)
    assert score == scoring.NOT_RANKED_DEGREE == 0.25
    assert "not confirmed" in why and "guardrail" in why


def test_university_confirmed_when_visible():
    visible = VISIBLE + "\nMSc Analytics, National University of\nSingapore"   # name split over two lines
    score, why = scoring.university_score(["MSc Analytics, National University of Singapore"], visible)
    assert score == 1.0 and "guardrail" not in why


def test_short_alias_needs_a_whole_word():
    score, _ = scoring.university_score(["NUS"], VISIBLE + "\nAnnual bonus scheme")  # 'nus' in 'bonus'
    assert score < 1.0


def test_no_text_layer_keeps_old_behaviour():
    assert scoring.university_score(["BSc, National University of Singapore"], None)[0] == 1.0
