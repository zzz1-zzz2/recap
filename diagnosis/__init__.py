"""ReCAP Diagnosis — failure event analysis, evidence alignment, hypothesis construction."""
from .failure_event import FailureEvent, FailureCluster, cluster_failures, reasoner_v2_cluster
from .hypothesis import DiagnosisHypothesis, HypothesisStatus, EvidenceReference, from_subtyped_diagnosis
from .alignment import EvidenceAlignment, SubtypedDiagnosis, align_evidence, classify_cluster
