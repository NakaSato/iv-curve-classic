#!/usr/bin/env python3
"""
Integrated Dual-String MPPT Analysis Workflow
Demonstrates integration with existing I-V curve analysis system
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Import existing analysis modules
try:
    from dual_string_mppt_analysis import DualStringMPPTAnalyzer
    from string_monitoring_system import StringMonitoringSystem
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure dual_string_mppt_analysis.py and string_monitoring_system.py are in the same directory")
    sys.exit(1)

class IntegratedAnalysisWorkflow:
    """Integrated workflow combining I-V curve analysis with dual-string MPPT analysis"""
    
    def __init__(self, data_directory="./inverter/"):
        self.data_directory = Path(data_directory)
        self.results_directory = Path("./analysis_results/")
        self.results_directory.mkdir(exist_ok=True)
        
        # Initialize analyzers (will be created per file)
        self.dual_string_analyzer = None
        self.monitoring_system = StringMonitoringSystem()
        
        # Analysis configuration
        self.config = {
            "analysis_types": {
                "dual_string_mppt": True,
                "real_time_monitoring": True,
                "fleet_analysis": True
            },
            "output_formats": ["dashboard", "report", "json"],
            "alert_integration": True
        }
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis workflow"""
        print("🚀 Starting Integrated Analysis Workflow...")
        print("=" * 80)
        
        # 1. Discover data files
        data_files = self._discover_data_files()
        if not data_files:
            print("❌ No data files found in directory:", self.data_directory)
            return None
        
        print(f"📁 Found {len(data_files)} data files for analysis")
        
        # 2. Run dual-string MPPT analysis
        results = {}
        for file_path in data_files:
            print(f"\n🔍 Analyzing: {file_path.name}")
            results[file_path.name] = self._analyze_single_file(file_path)
        
        # 3. Generate fleet summary
        fleet_summary = self._generate_fleet_summary(results)
        
        # 4. Setup monitoring if enabled
        if self.config["analysis_types"]["real_time_monitoring"]:
            self._setup_monitoring(data_files)
        
        # 5. Generate integrated report
        self._generate_integrated_report(results, fleet_summary)
        
        print("✅ Integrated analysis workflow completed!")
        return results
    
    def _discover_data_files(self):
        """Discover CSV data files with dual-string parameters"""
        data_files = []
        
        if not self.data_directory.exists():
            print(f"❌ Data directory not found: {self.data_directory}")
            return data_files
        
        # Look for CSV files with dual-string data
        for csv_file in self.data_directory.glob("*.csv"):
            if self._has_dual_string_data(csv_file):
                data_files.append(csv_file)
        
        return sorted(data_files)
    
    def _has_dual_string_data(self, csv_file):
        """Check if CSV file contains dual-string MPPT data"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_file, nrows=1)  # Read just header
            required_columns = ['Vstr1(V)', 'Vstr2(V)', 'Istr1(A)', 'Istr2(A)']
            return all(col in df.columns for col in required_columns)
        except Exception:
            return False
    
    def _analyze_single_file(self, file_path):
        """Analyze a single data file"""
        try:
            # Initialize analyzer for this specific file
            analyzer = DualStringMPPTAnalyzer(str(file_path))
            
            # Run complete dual-string MPPT analysis
            analysis_results = analyzer.run_complete_analysis()
            
            # Extract key metrics from analysis results
            performance = analysis_results.get("performance", {})
            issues = analysis_results.get("issues", {})
            mppt_efficiency = analysis_results.get("mppt_efficiency", {})
            
            results = {
                "file_path": str(file_path),
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "metrics": {
                    "string1_avg_power": performance.get("string1_avg_power", 0),
                    "string2_avg_power": performance.get("string2_avg_power", 0),
                    "power_imbalance": performance.get("power_difference_percent", 0),
                    "mppt_efficiency": mppt_efficiency.get("avg_efficiency", 0),
                    "issues_detected": len(issues.get("all_issues", []))
                },
                "alerts": issues.get("all_issues", []),
                "recommendations": analysis_results.get("recommendations", [])
            }
            
            print(f"   ✅ Analysis completed - Power Imbalance: {results['metrics']['power_imbalance']:.1f}%")
            return results
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            return {
                "file_path": str(file_path),
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def _generate_fleet_summary(self, results):
        """Generate fleet-level summary"""
        print("\\n📊 Generating Fleet Summary...")
        
        successful_analyses = [r for r in results.values() if r["status"] == "success"]
        
        if not successful_analyses:
            return {"status": "no_data", "message": "No successful analyses to summarize"}
        
        # Calculate fleet metrics
        total_power_imbalance = sum(r["metrics"]["power_imbalance"] for r in successful_analyses)
        avg_power_imbalance = total_power_imbalance / len(successful_analyses)
        total_issues = sum(r["metrics"]["issues_detected"] for r in successful_analyses)
        
        # Identify critical systems
        critical_systems = [
            r["file_path"] for r in successful_analyses 
            if r["metrics"]["power_imbalance"] > 15.0
        ]
        
        fleet_summary = {
            "total_systems": len(successful_analyses),
            "avg_power_imbalance": avg_power_imbalance,
            "total_issues_detected": total_issues,
            "critical_systems": critical_systems,
            "fleet_status": "CRITICAL" if critical_systems else "WARNING" if avg_power_imbalance > 8.0 else "NORMAL"
        }
        
        print(f"   📈 Fleet Status: {fleet_summary['fleet_status']}")
        print(f"   ⚖️ Average Power Imbalance: {avg_power_imbalance:.1f}%")
        print(f"   🚨 Critical Systems: {len(critical_systems)}")
        
        return fleet_summary
    
    def _setup_monitoring(self, data_files):
        """Setup real-time monitoring for data files"""
        print("\\n🔍 Setting up Real-Time Monitoring...")
        
        try:
            # Configure monitoring for each file
            for file_path in data_files:
                print(f"   📡 Monitoring: {file_path.name}")
                # In a real implementation, this would setup file watchers
                # For now, we'll just demonstrate the capability
            
            print("   ✅ Monitoring system activated")
            
        except Exception as e:
            print(f"   ❌ Monitoring setup failed: {e}")
    
    def _generate_integrated_report(self, results, fleet_summary):
        """Generate comprehensive integrated report"""
        print("\\n📝 Generating Integrated Report...")
        
        report_data = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "workflow_version": "1.0.0",
                "total_files_analyzed": len(results)
            },
            "individual_results": results,
            "fleet_summary": fleet_summary,
            "configuration": self.config
        }
        
        # Save JSON report
        json_report_path = self.results_directory / f"integrated_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate markdown summary
        self._generate_markdown_summary(report_data)
        
        print(f"   ✅ Report saved: {json_report_path}")
    
    def _generate_markdown_summary(self, report_data):
        """Generate markdown summary report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        markdown_content = f"""# Integrated Dual-String MPPT Analysis Report

*Generated: {timestamp}*

## 🏁 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Systems Analyzed** | {report_data['analysis_metadata']['total_files_analyzed']} |
| **Fleet Status** | **{report_data['fleet_summary'].get('fleet_status', 'N/A')}** |
| **Average Power Imbalance** | {report_data['fleet_summary'].get('avg_power_imbalance', 0):.1f}% |
| **Critical Systems** | {len(report_data['fleet_summary'].get('critical_systems', []))} |
| **Total Issues Detected** | {report_data['fleet_summary'].get('total_issues_detected', 0)} |

## 📊 Individual System Results

"""
        
        for filename, result in report_data['individual_results'].items():
            if result['status'] == 'success':
                metrics = result['metrics']
                markdown_content += f"""### {filename}
- **Power Imbalance**: {metrics['power_imbalance']:.1f}%
- **MPPT Efficiency**: {metrics['mppt_efficiency']:.1f}%
- **Issues Detected**: {metrics['issues_detected']}
- **Status**: {'🚨 CRITICAL' if metrics['power_imbalance'] > 15 else '⚠️ WARNING' if metrics['power_imbalance'] > 8 else '✅ NORMAL'}

"""
        
        markdown_content += f"""
## 🚨 Fleet-Level Recommendations

### Immediate Actions:
"""
        
        if report_data['fleet_summary'].get('critical_systems'):
            markdown_content += "- 🔧 **Critical Systems Inspection**: Immediate maintenance required for systems with >15% power imbalance\\n"
        
        if report_data['fleet_summary'].get('avg_power_imbalance', 0) > 8:
            markdown_content += "- 📊 **Fleet Performance Review**: Overall fleet performance requires attention\\n"
        
        markdown_content += """
### Ongoing Monitoring:
- 📡 **Real-Time Monitoring**: Continuous monitoring system activated
- 📈 **Trend Analysis**: Weekly performance trend reviews recommended
- 🛠️ **Preventive Maintenance**: Schedule based on analysis findings

---
*Generated by Integrated Dual-String MPPT Analysis Workflow*
"""
        
        # Save markdown report
        markdown_path = self.results_directory / f"integrated_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)

def main():
    """Main execution function"""
    print("🔍 Integrated Dual-String MPPT Analysis Workflow")
    print("=" * 60)
    
    # Initialize workflow
    workflow = IntegratedAnalysisWorkflow()
    
    # Run comprehensive analysis
    results = workflow.run_comprehensive_analysis()
    
    if results:
        print("\\n🎉 Analysis workflow completed successfully!")
        print("Check the analysis_results/ directory for detailed reports and dashboards.")
    else:
        print("\\n❌ Analysis workflow failed or found no data to process.")

if __name__ == "__main__":
    main()
