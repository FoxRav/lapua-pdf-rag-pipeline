"""Evaluate pipeline: data reconciliation and retrieval metrics."""

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io import get_output_dir, read_json, read_parquet, write_json


def reconcile_data(line_items_df: pd.DataFrame) -> dict[str, Any]:
    """Perform data reconciliation checks."""
    report: dict[str, Any] = {
        "checks": [],
        "passed": 0,
        "failed": 0,
    }
    
    if line_items_df.empty:
        report["error"] = "No line items to reconcile"
        return report
    
    # Check 1: Operating margin = Operating revenue - Operating expenses
    income_df = line_items_df[line_items_df["statement"] == "income_statement"]
    
    if not income_df.empty:
        operating_revenue = income_df[
            income_df["line_item_key"].str.contains("operating_revenue", case=False, na=False)
        ]["value_eur"].sum()
        operating_expenses = income_df[
            income_df["line_item_key"].str.contains("operating_expenses", case=False, na=False)
        ]["value_eur"].sum()
        operating_margin = income_df[
            income_df["line_item_key"].str.contains("operating_margin", case=False, na=False)
        ]["value_eur"].sum()
        
        if operating_revenue != 0 and operating_expenses != 0:
            calculated_margin = operating_revenue - operating_expenses
            diff = abs(operating_margin - calculated_margin)
            tolerance = abs(calculated_margin) * 0.01  # 1% tolerance
            
            check_passed = bool(diff <= tolerance)
            report["checks"].append(
                {
                    "check": "operating_margin_reconciliation",
                    "passed": check_passed,
                    "expected": float(calculated_margin),
                    "actual": float(operating_margin),
                    "difference": float(diff),
                    "tolerance": float(tolerance),
                }
            )
            
            if check_passed:
                report["passed"] += 1
            else:
                report["failed"] += 1
    
    # Check 2: Annual margin / Depreciation ratio
    if not income_df.empty:
        annual_margin = income_df[
            income_df["line_item_key"].str.contains("annual_margin", case=False, na=False)
        ]["value_eur"].sum()
        depreciation = income_df[
            income_df["line_item_key"].str.contains("depreciation", case=False, na=False)
        ]["value_eur"].sum()
        
        if depreciation != 0:
            ratio = annual_margin / abs(depreciation)
            report["checks"].append(
                {
                    "check": "annual_margin_depreciation_ratio",
                    "ratio": float(ratio),
                    "annual_margin": float(annual_margin),
                    "depreciation": float(depreciation),
                }
            )
    
    # Check 3: Cash flow reconciliation (if available)
    cash_flow_df = line_items_df[line_items_df["statement"] == "cash_flow"]
    balance_sheet_df = line_items_df[line_items_df["statement"] == "balance_sheet"]
    
    if not cash_flow_df.empty and not balance_sheet_df.empty:
        cash_change_cf = cash_flow_df[
            cash_flow_df["line_item_key"].str.contains("cash_change", case=False, na=False)
        ]["value_eur"].sum()
        cash_change_bs = balance_sheet_df[
            balance_sheet_df["line_item_key"].str.contains("cash_change", case=False, na=False)
        ]["value_eur"].sum()
        
        if cash_change_cf != 0 and cash_change_bs != 0:
            diff = abs(cash_change_cf - cash_change_bs)
            tolerance = abs(cash_change_cf) * 0.01
            
            check_passed = bool(diff <= tolerance)
            report["checks"].append(
                {
                    "check": "cash_change_reconciliation",
                    "passed": check_passed,
                    "cash_flow_value": float(cash_change_cf),
                    "balance_sheet_value": float(cash_change_bs),
                    "difference": float(diff),
                    "tolerance": float(tolerance),
                }
            )
            
            if check_passed:
                report["passed"] += 1
            else:
                report["failed"] += 1
    
    return report


def retrieval_eval(year: int) -> dict[str, Any]:
    """Perform retrieval evaluation (placeholder for now)."""
    # This would require a question set and expected evidence
    # For now, return a placeholder structure
    
    return {
        "status": "not_implemented",
        "note": "Retrieval eval requires question set and expected evidence mapping",
        "metrics": {
            "hit@5": None,
            "mrr": None,
        },
    }


def main(year: int) -> None:
    """Main eval function."""
    output_dir = get_output_dir(year)
    line_items_path = output_dir / "line_items_long.csv"
    
    if not line_items_path.exists():
        print("Line items not found. Run extract first.")
        return
    
    print(f"Evaluating year {year}...")
    
    # Load line items
    line_items_df = pd.read_csv(line_items_path)
    
    # Reconciliation checks
    print("Performing reconciliation checks...")
    reconcile_report = reconcile_data(line_items_df)
    write_json(reconcile_report, output_dir / "reconcile_report.json")
    
    print(f"  OK: Reconciliation: {reconcile_report['passed']} passed, {reconcile_report['failed']} failed")
    for check in reconcile_report["checks"]:
        status = "OK" if check.get("passed", False) else "FAIL"
        print(f"    {status} {check['check']}")
    
    # Retrieval eval (placeholder)
    print("\nPerforming retrieval evaluation...")
    retrieval_report = retrieval_eval(year)
    write_json(retrieval_report, output_dir / "retrieval_eval.json")
    
    print("  WARNING: Retrieval eval not fully implemented (requires question set)")
    
    # Combined report
    combined_report = {
        "year": year,
        "reconciliation": reconcile_report,
        "retrieval": retrieval_report,
    }
    write_json(combined_report, output_dir / "eval_report.json")
    
    print(f"\nEval complete. Reports saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.05_eval YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    main(year)

