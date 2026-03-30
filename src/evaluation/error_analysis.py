"""
Substitution / deletion / insertion breakdown and per-domain WER analysis.

Uses jiwer's process_words() to get alignment-level error counts, then
aggregates them per domain to identify which categories the model struggles with.
"""

from collections import defaultdict
from typing import Optional

from jiwer import process_words, wer


class ErrorAnalyser:
    """
    Compute detailed error statistics from a list of references and hypotheses.

    Args:
        references   : list of ground-truth transcripts
        hypotheses   : list of model predictions
        domains      : optional list of domain labels (same length as refs)
    """

    def __init__(
        self,
        references: list[str],
        hypotheses: list[str],
        domains: Optional[list[str]] = None,
    ):
        self.references  = references
        self.hypotheses  = hypotheses
        self.domains     = domains or ["unknown"] * len(references)
        self._processed  = process_words(references, hypotheses)

    def error_breakdown(self) -> dict:
        """
        Return substitution / deletion / insertion counts and rates.
        """
        p = self._processed
        total_ref_words = sum(len(r.split()) for r in self.references)

        return {
            "substitutions":     p.substitutions,
            "deletions":         p.deletions,
            "insertions":        p.insertions,
            "hits":              p.hits,
            "substitution_rate": round(p.substitutions / max(total_ref_words, 1), 4),
            "deletion_rate":     round(p.deletions     / max(total_ref_words, 1), 4),
            "insertion_rate":    round(p.insertions    / max(total_ref_words, 1), 4),
        }

    def per_domain_wer(self) -> dict[str, dict]:
        """
        Compute WER separately for each domain label.

        Returns a dict mapping domain → {wer, n_samples} sorted by WER descending.
        """
        domain_refs  = defaultdict(list)
        domain_hyps  = defaultdict(list)

        for ref, hyp, domain in zip(self.references, self.hypotheses, self.domains):
            domain_refs[domain].append(ref)
            domain_hyps[domain].append(hyp)

        per_domain = {}
        for domain in domain_refs:
            refs = domain_refs[domain]
            hyps = domain_hyps[domain]
            try:
                domain_wer = round(wer(refs, hyps), 4)
            except Exception:
                domain_wer = None
            per_domain[domain] = {
                "wer":       domain_wer,
                "n_samples": len(refs),
            }

        # Sort by WER descending — worst domains first
        return dict(
            sorted(
                per_domain.items(),
                key=lambda x: x[1]["wer"] or 0,
                reverse=True,
            )
        )

    def worst_examples(self, n: int = 10) -> list[dict]:
        """
        Return the N examples with the highest individual WER.
        Useful for qualitative error analysis.
        """
        examples = []
        for ref, hyp, domain in zip(self.references, self.hypotheses, self.domains):
            try:
                sample_wer = wer([ref], [hyp])
            except Exception:
                sample_wer = 1.0
            examples.append({
                "reference":  ref,
                "hypothesis": hyp,
                "domain":     domain,
                "wer":        round(sample_wer, 4),
            })

        return sorted(examples, key=lambda x: x["wer"], reverse=True)[:n]

    def print_report(self) -> None:
        """Print a formatted evaluation report to stdout."""
        from jiwer import wer as compute_wer, cer as compute_cer
        overall_wer = compute_wer(self.references, self.hypotheses)
        overall_cer = compute_cer(self.references, self.hypotheses)
        breakdown   = self.error_breakdown()
        per_domain  = self.per_domain_wer()

        print(f"\n{'='*60}")
        print(f"  Evaluation Report")
        print(f"{'='*60}")
        print(f"  Samples    : {len(self.references)}")
        print(f"  WER        : {overall_wer:.4f}  ({overall_wer*100:.2f}%)")
        print(f"  CER        : {overall_cer:.4f}  ({overall_cer*100:.2f}%)")
        print(f"\n  Error breakdown:")
        print(f"    Substitutions : {breakdown['substitutions']:>6}  ({breakdown['substitution_rate']*100:.2f}%)")
        print(f"    Deletions     : {breakdown['deletions']:>6}  ({breakdown['deletion_rate']*100:.2f}%)")
        print(f"    Insertions    : {breakdown['insertions']:>6}  ({breakdown['insertion_rate']*100:.2f}%)")
        print(f"\n  Per-domain WER (worst first):")
        for domain, stats in list(per_domain.items())[:15]:
            wer_str = f"{stats['wer']*100:.2f}%" if stats["wer"] is not None else "N/A"
            print(f"    {domain:<35} {wer_str:>8}  (n={stats['n_samples']})")

        print(f"\n  Worst examples:")
        for i, ex in enumerate(self.worst_examples(5), 1):
            print(f"\n  [{i}] WER={ex['wer']:.3f}  domain={ex['domain']}")
            print(f"       REF : {ex['reference'][:100]}")
            print(f"       HYP : {ex['hypothesis'][:100]}")
        print()