from __future__ import annotations

from optisample.artifacts.documents.metrics import metrics_document


def test_metrics_document_objective_sums_note_contributions() -> None:
    doc = metrics_document("ungrouped", "piano", 44_100, plan_objective=1.5, notes=[])
    assert doc.objective == 0.0  # no notes → no contribution
    assert doc.plan_objective == 1.5  # the plan objective is carried through verbatim
    assert doc.notes == []
