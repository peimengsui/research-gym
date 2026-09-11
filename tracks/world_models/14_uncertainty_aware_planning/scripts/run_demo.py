"""Compare nominal, total-spread, and epistemic-aware WAM planning."""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    decompose_predictive_uncertainty,
    nominal_plan_scores,
    risk_adjusted_plan_scores,
    select_uncertainty_aware_plan,
)
from provided import (  # noqa: E402
    ProvidedWAMEnsemble,
    make_uncertainty_planning_problem,
)


def main() -> None:
    problem = make_uncertainty_planning_problem()
    prediction = ProvidedWAMEnsemble().predict(problem.action_chunks)
    estimate = decompose_predictive_uncertainty(prediction)
    nominal = nominal_plan_scores(
        problem.action_chunks,
        estimate.mean_futures,
        problem.goal_latent,
    )
    risk_coefficient = 1.0
    epistemic_adjusted = risk_adjusted_plan_scores(
        nominal,
        estimate.epistemic,
        risk_coefficient,
    )
    total_spread_adjusted = nominal - risk_coefficient * (
        estimate.aleatoric + estimate.epistemic
    )

    nominal_plan = select_uncertainty_aware_plan(
        problem,
        estimate,
        nominal,
        nominal,
    )
    total_spread_plan = select_uncertainty_aware_plan(
        problem,
        estimate,
        nominal,
        total_spread_adjusted,
    )
    epistemic_plan = select_uncertainty_aware_plan(
        problem,
        estimate,
        nominal,
        epistemic_adjusted,
    )

    print(f"candidate names:       {list(problem.candidate_names)}")
    print(f"nominal scores:        {nominal.tolist()}")
    print(f"aleatoric uncertainty: {estimate.aleatoric.tolist()}")
    print(f"epistemic uncertainty: {estimate.epistemic.tolist()}")
    print(f"total-spread scores:   {total_spread_adjusted.tolist()}")
    print(f"epistemic-risk scores: {epistemic_adjusted.tolist()}")
    print(f"nominal choice:        {nominal_plan.name}")
    print(f"total-spread choice:   {total_spread_plan.name}")
    print(f"epistemic-aware choice:{epistemic_plan.name}")
    print("Agreed multimodality stays valid; model disagreement changes the plan.")


if __name__ == "__main__":
    main()
