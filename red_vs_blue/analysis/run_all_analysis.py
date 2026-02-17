"""
run_all_analysis.py

Unified script to run all analysis tools on Red vs. Blue benchmark results.

This script orchestrates the full analysis pipeline:
1. Aggregate results from .eval files
2. Generate statistics
3. Create plots and visualizations
4. Generate advanced analysis and infographics
5. Create interactive HTML viewer
6. Analyze player confusion patterns
7. Analyze player strategies

Usage:
    python run_all_analysis.py <results_dir> [options]
    
Examples:
    python run_all_analysis.py results/
    python run_all_analysis.py results/ --skip-confusion
    python run_all_analysis.py results/ --only aggregate
"""

from __future__ import annotations
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import importlib.util


def run_module(module_path: str, module_name: str, args: list) -> bool:
    """Dynamically import and run a module's main function."""
    try:
        # Construct absolute path to module
        module_file = Path(__file__).parent / f"{module_path}.py"
        
        if not module_file.exists():
            print(f"[ERROR] Module not found: {module_file}")
            return False
        
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            print(f"[ERROR] Failed to load spec for {module_name}")
            return False
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Run main function
        if hasattr(module, 'main'):
            print(f"\n{'='*70}")
            print(f"Running: {module_name}")
            print(f"{'='*70}")
            
            try:
                # Handle different main function signatures
                import inspect
                sig = inspect.signature(module.main)
                
                if len(sig.parameters) == 1:
                    # Takes results_dir
                    module.main(args[0])
                elif len(sig.parameters) == 2:
                    # Takes eval_file and optional output
                    if len(args) > 1:
                        module.main(args[0], args[1])
                    else:
                        module.main(args[0], None)
                else:
                    # Fallback
                    module.main(args[0])
                    
                print(f"[OK] {module_name} completed successfully")
                return True
            except Exception as e:
                print(f"[ERROR] Error running {module_name}: {e}")
                return False
        else:
            print(f"[ERROR] No main() function found in {module_name}")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Failed to import {module_name}: {e}")
        return False


def run_async_analysis(module_file: str, module_name: str, eval_file: str, model_name: str, model_base_url: Optional[str]) -> bool:
    """Import and run an async analysis module with signature main(eval_file, model_name)."""
    print(f"\n{'='*70}")
    print(f"Running: {module_name}")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    if model_base_url:
        print(f"Model base URL: {model_base_url}")

    import asyncio
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).parent / module_file
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            asyncio.run(module.main(eval_file, model_name))
            print(f"[OK] {module_name} completed successfully")
            return True

        print(f"[ERROR] Failed to load {module_name} module")
        return False
    except Exception as e:
        print(f"[ERROR] Error running {module_name}: {e}")
        return False


def main():
    """Run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run all analysis tools on Red vs. Blue benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_analysis.py results/
  python run_all_analysis.py results/ --skip-confusion
  python run_all_analysis.py results/ --only aggregate,statistics
        """
    )
    
    parser.add_argument(
        "results_dir",
        help="Path to results directory containing .eval files"
    )
    
    parser.add_argument(
        "--skip-confusion",
        action="store_true",
        help="Skip confusion analysis (requires model)"
    )
    
    parser.add_argument(
        "--skip-strategy",
        action="store_true",
        help="Skip strategy analysis (requires model)"
    )

    parser.add_argument(
        "--skip-action-efficiency",
        action="store_true",
        help="Skip action efficiency analysis (requires model)"
    )

    parser.add_argument(
        "--skip-risk-management",
        action="store_true",
        help="Skip risk management analysis (requires model)"
    )

    parser.add_argument(
        "--skip-collaboration-quality",
        action="store_true",
        help="Skip collaboration quality analysis (requires model)"
    )

    parser.add_argument(
        "--skip-role-utilization",
        action="store_true",
        help="Skip role utilization analysis (requires model)"
    )

    parser.add_argument(
        "--skip-outcome-attribution",
        action="store_true",
        help="Skip outcome attribution analysis (requires model)"
    )

    parser.add_argument(
        "--skip-cross-findings",
        action="store_true",
        help="Skip cross-analysis findings generation (requires model)"
    )
    
    parser.add_argument(
        "--skip-viewer",
        action="store_true",
        help="Skip viewer generation"
    )
    
    parser.add_argument(
        "--only",
        type=str,
        help="Run only specific analyses (comma-separated): aggregate,statistics,plots,advanced,infographics,viewer,confusion,strategy,action_efficiency,risk_management,collaboration_quality,role_utilization,outcome_attribution,cross_findings,polarix"
    )

    parser.add_argument(
        "--with-polarix",
        action="store_true",
        help="Also run Inspect->Polarix conversion and Polarix analysis"
    )

    parser.add_argument(
        "--polarix-output-json",
        type=str,
        default=None,
        help="Output path for converted Polarix benchmark summary JSON"
    )

    parser.add_argument(
        "--polarix-normalizer",
        type=str,
        default="winrate",
        choices=["winrate", "ptp", "zscore"],
        help="Normalizer for Polarix conversion (default: winrate)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/gpt-oss:20b",
        help="Model to use for confusion analysis (default: ollama/gpt-oss:20b)"
    )
    
    parser.add_argument(
        "--model-base-url",
        type=str,
        help="Base URL for model API (e.g., http://192.168.86.230:11434/v1)"
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Determine which analyses to run
    all_analyses = ["aggregate", "statistics", "plots", "advanced", "infographics"]
    if not args.skip_viewer:
        all_analyses.append("viewer")
    if not args.skip_confusion:
        all_analyses.append("confusion")
    if not args.skip_strategy:
        all_analyses.append("strategy")
    if not args.skip_action_efficiency:
        all_analyses.append("action_efficiency")
    if not args.skip_risk_management:
        all_analyses.append("risk_management")
    if not args.skip_collaboration_quality:
        all_analyses.append("collaboration_quality")
    if not args.skip_role_utilization:
        all_analyses.append("role_utilization")
    if not args.skip_outcome_attribution:
        all_analyses.append("outcome_attribution")
    if not args.skip_cross_findings:
        all_analyses.append("cross_findings")
    if args.with_polarix:
        all_analyses.append("polarix")
    
    if args.only:
        requested = set(args.only.split(","))
        analyses = [a for a in all_analyses if a in requested]
        invalid = requested - set(all_analyses)
        if invalid:
            print(f"[ERROR] Invalid analyses: {', '.join(invalid)}")
            print(f"Valid options: {', '.join(all_analyses)}")
            sys.exit(1)
    else:
        analyses = all_analyses
    
    # Set environment variables for model configuration
    if args.model_base_url:
        import os
        os.environ["INSPECT_EVAL_MODEL_BASE_URL"] = args.model_base_url
    
    # Run analyses
    results = {}
    
    print(f"\n{'='*70}")
    print(f"RED VS. BLUE ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Analyses to run: {', '.join(analyses)}")
    print(f"")
    
    # 1. Aggregate results
    if "aggregate" in analyses:
        results["aggregate"] = run_module(
            "aggregate_results",
            "aggregate_results",
            [str(results_dir)]
        )
        if not results["aggregate"]:
            print("[WARN] Aggregation failed - subsequent analyses may fail")
    
    # 2. Statistics
    if "statistics" in analyses:
        results["statistics"] = run_module(
            "statistics",
            "statistics",
            [str(results_dir)]
        )
    
    # 3. Plots
    if "plots" in analyses:
        results["plots"] = run_module(
            "plots",
            "plots",
            [str(results_dir)]
        )
    
    # 4. Advanced Analysis
    if "advanced" in analyses:
        results["advanced"] = run_module(
            "advanced_analysis",
            "advanced_analysis",
            [str(results_dir)]
        )
    
    # 5. Advanced Infographics
    if "infographics" in analyses:
        results["infographics"] = run_module(
            "advanced_infographics",
            "advanced_infographics",
            [str(results_dir)]
        )
    
    # 6. Results Viewer (HTML)
    if "viewer" in analyses:
        # Find most recent eval file
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            output_html = str(results_dir / "results_viewer.html")
            results["viewer"] = run_module(
                "results_viewer",
                "results_viewer",
                [latest_eval, output_html]
            )
        else:
            print("[WARN] No .eval files found - skipping viewer generation")
            results["viewer"] = False
    
    # 7. Confusion Analysis
    if "confusion" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["confusion"] = run_async_analysis(
                "confusion_analysis.py",
                "confusion_analysis",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping confusion analysis")
            results["confusion"] = False
    
    # 8. Strategy Analysis
    if "strategy" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["strategy"] = run_async_analysis(
                "strategy_analysis.py",
                "strategy_analysis",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping strategy analysis")
            results["strategy"] = False

    # 9. Action Efficiency Analysis
    if "action_efficiency" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["action_efficiency"] = run_async_analysis(
                "action_efficiency_analysis.py",
                "action_efficiency_analysis",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping action efficiency analysis")
            results["action_efficiency"] = False

    # 10. Risk Management Analysis
    if "risk_management" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["risk_management"] = run_async_analysis(
                "risk_management_analysis.py",
                "risk_management_analysis",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping risk management analysis")
            results["risk_management"] = False

    # 11. Collaboration Quality Analysis
    if "collaboration_quality" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["collaboration_quality"] = run_async_analysis(
                "collaboration_quality_analysis.py",
                "collaboration_quality_analysis",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping collaboration quality analysis")
            results["collaboration_quality"] = False

    # 12. Role Utilization Analysis
    if "role_utilization" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["role_utilization"] = run_async_analysis(
                "role_utilization_analysis.py",
                "role_utilization_analysis",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping role utilization analysis")
            results["role_utilization"] = False

    # 13. Outcome Attribution Analysis
    if "outcome_attribution" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["outcome_attribution"] = run_async_analysis(
                "outcome_attribution_analysis.py",
                "outcome_attribution_analysis",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping outcome attribution analysis")
            results["outcome_attribution"] = False

    # 14. Cross-Analysis Findings
    if "cross_findings" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            results["cross_findings"] = run_async_analysis(
                "cross_analysis_findings.py",
                "cross_analysis_findings",
                latest_eval,
                args.model,
                args.model_base_url,
            )
        else:
            print("[WARN] No .eval files found - skipping cross-analysis findings")
            results["cross_findings"] = False

    # 15. Polarix conversion + analysis
    if "polarix" in analyses:
        print(f"\n{'='*70}")
        print("Running: polarix_conversion + run_polarix_analysis")
        print(f"{'='*70}")

        try:
            # Step 1: Convert Inspect .eval results to Polarix benchmark summary
            converter_ok = run_module(
                "convert_inspect_to_polarix",
                "convert_inspect_to_polarix",
                [str(results_dir)]
            )

            if not converter_ok:
                results["polarix"] = False
            else:
                # Determine summary path
                summary_json = (
                    Path(args.polarix_output_json)
                    if args.polarix_output_json
                    else Path("results_polarix_red_vs_blue/benchmark_summary_from_inspect.json")
                )

                # If custom output was requested, run converter once more with explicit args
                if args.polarix_output_json:
                    converter_script = Path(__file__).parent / "convert_inspect_to_polarix.py"
                    cmd_convert = [
                        sys.executable,
                        str(converter_script),
                        str(results_dir),
                        "--output-json",
                        str(summary_json),
                        "--normalizer",
                        args.polarix_normalizer,
                        "--model-name",
                        args.model,
                    ]
                    convert_proc = subprocess.run(cmd_convert, check=False)
                    if convert_proc.returncode != 0:
                        print("[ERROR] Polarix conversion failed")
                        results["polarix"] = False
                        raise RuntimeError("polarix conversion failed")

                # Step 2: Run top-level Polarix analysis script
                repo_root = Path(__file__).resolve().parents[2]
                polarix_analysis_script = repo_root / "analysis" / "run_polarix_analysis.py"
                output_dir = summary_json.parent / "analysis"

                cmd_analysis = [
                    sys.executable,
                    str(polarix_analysis_script),
                    str(summary_json),
                    "--output-dir",
                    str(output_dir),
                    "--summary-model",
                    args.model,
                ]
                if args.model_base_url:
                    cmd_analysis.extend(["--summary-model-base-url", args.model_base_url])

                analysis_proc = subprocess.run(cmd_analysis, check=False)
                if analysis_proc.returncode == 0:
                    print("[OK] polarix conversion + analysis completed successfully")
                    results["polarix"] = True
                else:
                    print("[ERROR] polarix analysis failed")
                    results["polarix"] = False

        except Exception as e:
            print(f"[ERROR] Error running polarix stage: {e}")
            results["polarix"] = False
    
    # Summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for analysis, success in results.items():
        status = "[OK]" if success else "[ERROR]"
        print(f"{status} {analysis}")
    
    print(f"\nSummary: {passed}/{total} analyses completed successfully")
    
    if passed == total:
        print("\n[OK] All analyses completed successfully!")
        sys.exit(0)
    else:
        print(f"\n[WARN] {total - passed} analysis/analyses failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
