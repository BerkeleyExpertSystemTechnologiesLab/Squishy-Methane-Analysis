"""Reporting must preserve numerical and sampling limitations, even large gains."""

import importlib.util
from pathlib import Path


def reporting():
    path=Path(__file__).resolve().parents[1]/'scripts/summarize-scan-value-diagnosis.py'
    spec=importlib.util.spec_from_file_location('diagnosis_reporting',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_unconverged_refinement_cannot_certify_large_observed_gain():
    r=reporting();stat=r.paired([1.,1.01,.99,1.],.01)
    check={'within_material_tolerance':True,'contrast_change':.001,'finest_converged':False}
    assert r.classify_contrast(stat,[0,1],.01,check)=='numerically_unresolved'
    check['finest_converged']=True
    assert r.classify_contrast(stat,[0,1],.01,check)=='supported_in_tested_contrast'
    assert r.classify_contrast(stat,[0,1],.01,check,True)=='numerically_unresolved'


def test_identical_empirical_winners_do_not_prove_family_equivalence():
    r=reporting();stat=r.paired([0.]*8,.01)
    assert r.classify_contrast(stat,[2,2],.01,None)=='no_failure_demonstrated_by_empirical_winners'
    assert r.classify_contrast({'status':'not_assessed'},[2,2],.01,None)=='not_assessed'


def test_numerical_margin_is_separate_from_noise_interval():
    r=reporting();stat=r.paired([.012]*8,.01)
    check={'within_material_tolerance':True,'contrast_change':.005}
    assert r.classify_contrast(stat,[0,1],.01,check)=='sampling_or_numerically_unresolved'
