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
from pathlib import Path
from typing import Optional
import importlib.util


def run_module(module_path: str, module_name: str, args: list) -> bool:
    """Dynamically import and run a module's main function."""
    try:
        # Construct absolute path to module
        module_file = Path(__file__).parent / f"{module_path}.py"
        
        if not module_file.exists():
            print(f"❌ Module not found: {module_file}")
            return False
        
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            print(f"❌ Failed to load spec for {module_name}")
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
                    
                print(f"✓ {module_name} completed successfully")
                return True
            except Exception as e:
                print(f"❌ Error running {module_name}: {e}")
                return False
        else:
            print(f"❌ No main() function found in {module_name}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
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
        "--skip-viewer",
        action="store_true",
        help="Skip interactive HTML viewer generation"
    )
    
    parser.add_argument(
        "--only",
        type=str,
        help="Run only specific analyses (comma-separated): aggregate,statistics,plots,advanced,infographics,viewer,confusion"
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
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Determine which analyses to run
    all_analyses = ["aggregate", "statistics", "plots", "advanced", "infographics"]
    if not args.skip_viewer:
        all_analyses.append("viewer")
    if not args.skip_confusion:
        all_analyses.append("confusion")
    
    if args.only:
        requested = set(args.only.split(","))
        analyses = [a for a in all_analyses if a in requested]
        invalid = requested - set(all_analyses)
        if invalid:
            print(f"❌ Invalid analyses: {', '.join(invalid)}")
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
            print("⚠ Aggregation failed - subsequent analyses may fail")
    
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
            print("⚠ No .eval files found - skipping viewer generation")
            results["viewer"] = False
    
    # 7. Confusion Analysis
    if "confusion" in analyses:
        eval_files = sorted(results_dir.glob("*.eval"), reverse=True)
        if eval_files:
            latest_eval = str(eval_files[0])
            # We need to handle confusion analysis specially due to async
            print(f"\n{'='*70}")
            print(f"Running: confusion_analysis")
            print(f"{'='*70}")
            print(f"Model: {args.model}")
            if args.model_base_url:
                print(f"Model base URL: {args.model_base_url}")
            
            import asyncio
            try:
                # Import and run confusion analysis
                spec = importlib.util.spec_from_file_location(
                    "confusion_analysis",
                    Path(__file__).parent / "confusion_analysis.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["confusion_analysis"] = module
                    spec.loader.exec_module(module)
                    
                    asyncio.run(module.main(latest_eval, args.model))
                    print(f"✓ confusion_analysis completed successfully")
                    results["confusion"] = True
                else:
                    print(f"❌ Failed to load confusion_analysis module")
                    results["confusion"] = False
            except Exception as e:
                print(f"❌ Error running confusion_analysis: {e}")
                results["confusion"] = False
        else:
            print("⚠ No .eval files found - skipping confusion analysis")
            results["confusion"] = False
    
    # Summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for analysis, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {analysis}")
    
    print(f"\nSummary: {passed}/{total} analyses completed successfully")
    
    if passed == total:
        print("\n✓ All analyses completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠ {total - passed} analysis/analyses failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
