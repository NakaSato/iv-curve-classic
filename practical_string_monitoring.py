#!/usr/bin/env python3
"""
Practical Dual-String I-V Curve Monitoring Application
Real-world demonstration of comprehensive string analysis and monitoring
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the enhanced string I-V analyzer
from enhanced_string_iv_analyzer import EnhancedStringIVAnalyzer

class StringMonitoringApplication:
    """
    Practical application for dual-string I-V curve monitoring
    """
    
    def __init__(self, data_directory: str = "inverter"):
        """Initialize the monitoring application."""
        self.data_directory = Path(data_directory)
        self.monitoring_history = []
        
    def analyze_single_system(self, data_file: str) -> dict:
        """
        Analyze a single dual-string system
        
        Args:
            data_file: Path to the CSV data file
            
        Returns:
            Analysis results dictionary
        """
        print(f"\n🔍 Analyzing System: {Path(data_file).name}")
        print("-" * 50)
        
        try:
            # Initialize analyzer
            analyzer = EnhancedStringIVAnalyzer(data_file)
            
            # Run complete analysis
            results = analyzer.run_complete_string_analysis()
            
            # Extract key metrics for monitoring
            monitoring_data = {
                'timestamp': datetime.now().isoformat(),
                'data_file': Path(data_file).name,
                'data_points': results['summary']['total_data_points'],
                'string1': {
                    'avg_power': results['string1_analysis']['performance']['average_power'],
                    'max_power': results['string1_analysis']['performance']['max_power'],
                    'efficiency': results['string1_analysis']['efficiency']['average_efficiency'],
                    'fill_factor': results['string1_analysis']['iv_curve']['characteristics']['fill_factor'],
                    'mpp_voltage': results['string1_analysis']['iv_curve']['mpp']['voltage'],
                    'mpp_current': results['string1_analysis']['iv_curve']['mpp']['current'],
                    'quality_score': results['string1_analysis']['iv_curve']['quality_metrics']['overall_quality_score']
                },
                'string2': {
                    'avg_power': results['string2_analysis']['performance']['average_power'],
                    'max_power': results['string2_analysis']['performance']['max_power'],
                    'efficiency': results['string2_analysis']['efficiency']['average_efficiency'],
                    'fill_factor': results['string2_analysis']['iv_curve']['characteristics']['fill_factor'],
                    'mpp_voltage': results['string2_analysis']['iv_curve']['mpp']['voltage'],
                    'mpp_current': results['string2_analysis']['iv_curve']['mpp']['current'],
                    'quality_score': results['string2_analysis']['iv_curve']['quality_metrics']['overall_quality_score']
                },
                'power_imbalance_pct': results['summary']['power_imbalance_percent'],
                'files_generated': results['files_generated']
            }
            
            # Add to monitoring history
            self.monitoring_history.append(monitoring_data)
            
            # Display results
            self._display_analysis_summary(monitoring_data)
            
            return monitoring_data
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return None
    
    def _display_analysis_summary(self, data: dict):
        """Display formatted analysis summary."""
        
        print(f"📊 Data Points: {data['data_points']}")
        print()
        
        # String performance summary
        s1 = data['string1']
        s2 = data['string2']
        
        print("🔌 STRING PERFORMANCE SUMMARY:")
        print(f"   String 1: {s1['avg_power']:.1f}W avg | {s1['efficiency']:.1f}% eff | FF: {s1['fill_factor']:.3f}")
        print(f"   String 2: {s2['avg_power']:.1f}W avg | {s2['efficiency']:.1f}% eff | FF: {s2['fill_factor']:.3f}")
        print()
        
        # Performance indicators
        power_imbalance = abs(data['power_imbalance_pct'])
        efficiency_diff = abs(s1['efficiency'] - s2['efficiency'])
        avg_quality = (s1['quality_score'] + s2['quality_score']) / 2
        
        print("🎯 PERFORMANCE INDICATORS:")
        print(f"   Power Imbalance: {power_imbalance:.1f}% {'✅' if power_imbalance < 5 else '⚠️' if power_imbalance < 10 else '🚨'}")
        print(f"   Efficiency Difference: {efficiency_diff:.1f}% {'✅' if efficiency_diff < 5 else '⚠️'}")
        print(f"   Average Quality Score: {avg_quality:.1f}/100 {'✅' if avg_quality > 70 else '⚠️' if avg_quality > 50 else '🚨'}")
        print()
        
        # Issue detection
        issues = self._detect_issues(data)
        if issues:
            print("⚠️ ISSUES DETECTED:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("✅ No critical issues detected")
        
        print()
        print(f"📊 Dashboard: {data['files_generated']['dashboard']}")
        print(f"📝 Report: {data['files_generated']['report']}")
    
    def _detect_issues(self, data: dict) -> list:
        """Detect potential issues from analysis data."""
        issues = []
        
        s1 = data['string1']
        s2 = data['string2']
        
        # Power imbalance check
        if abs(data['power_imbalance_pct']) > 10:
            issues.append("Critical power imbalance - immediate inspection required")
        elif abs(data['power_imbalance_pct']) > 5:
            issues.append("Moderate power imbalance - schedule inspection")
        
        # Efficiency check
        if s1['efficiency'] < 60 or s2['efficiency'] < 60:
            issues.append("Low efficiency detected - cleaning/maintenance needed")
        
        # Fill factor check
        if s1['fill_factor'] < 0.7 or s2['fill_factor'] < 0.7:
            issues.append("Low fill factor - check for series resistance issues")
        
        # Quality check
        avg_quality = (s1['quality_score'] + s2['quality_score']) / 2
        if avg_quality < 50:
            issues.append("Poor data quality - verify monitoring equipment")
        
        return issues
    
    def batch_analysis(self, file_pattern: str = "*.csv") -> dict:
        """
        Perform batch analysis on multiple systems
        
        Args:
            file_pattern: Pattern to match files for analysis
            
        Returns:
            Summary of batch analysis results
        """
        print("🔄 BATCH ANALYSIS MODE")
        print("=" * 60)
        
        if not self.data_directory.exists():
            print(f"❌ Data directory not found: {self.data_directory}")
            return {}
        
        # Find data files
        data_files = list(self.data_directory.glob(file_pattern))
        print(f"📁 Found {len(data_files)} data files")
        
        if not data_files:
            print("❌ No data files found for analysis")
            return {}
        
        # Analyze each file
        batch_results = []
        successful_analyses = 0
        
        for data_file in data_files:
            print(f"\n📊 Processing: {data_file.name}")
            
            try:
                result = self.analyze_single_system(str(data_file))
                if result:
                    batch_results.append(result)
                    successful_analyses += 1
                    print("✅ Analysis completed successfully")
                else:
                    print("❌ Analysis failed")
            except Exception as e:
                print(f"❌ Error processing {data_file.name}: {str(e)}")
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(batch_results)
        
        print(f"\n📈 BATCH ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Files Processed: {len(data_files)}")
        print(f"Successful Analyses: {successful_analyses}")
        print(f"Failed Analyses: {len(data_files) - successful_analyses}")
        
        if batch_results:
            self._display_batch_statistics(batch_summary)
        
        return batch_summary
    
    def _generate_batch_summary(self, results: list) -> dict:
        """Generate summary statistics from batch analysis."""
        if not results:
            return {}
        
        # Aggregate statistics
        total_systems = len(results)
        
        # Power statistics
        s1_powers = [r['string1']['avg_power'] for r in results]
        s2_powers = [r['string2']['avg_power'] for r in results]
        power_imbalances = [abs(r['power_imbalance_pct']) for r in results]
        
        # Efficiency statistics
        s1_efficiencies = [r['string1']['efficiency'] for r in results]
        s2_efficiencies = [r['string2']['efficiency'] for r in results]
        
        # Quality statistics
        quality_scores = [(r['string1']['quality_score'] + r['string2']['quality_score'])/2 for r in results]
        
        # Issue detection
        systems_with_issues = sum(1 for r in results if len(self._detect_issues(r)) > 0)
        
        return {
            'total_systems': total_systems,
            'systems_with_issues': systems_with_issues,
            'power_statistics': {
                'string1_avg': sum(s1_powers) / len(s1_powers),
                'string2_avg': sum(s2_powers) / len(s2_powers),
                'max_imbalance': max(power_imbalances),
                'avg_imbalance': sum(power_imbalances) / len(power_imbalances)
            },
            'efficiency_statistics': {
                'string1_avg': sum(s1_efficiencies) / len(s1_efficiencies),
                'string2_avg': sum(s2_efficiencies) / len(s2_efficiencies),
                'min_efficiency': min(min(s1_efficiencies), min(s2_efficiencies)),
                'max_efficiency': max(max(s1_efficiencies), max(s2_efficiencies))
            },
            'quality_statistics': {
                'avg_quality': sum(quality_scores) / len(quality_scores),
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores)
            }
        }
    
    def _display_batch_statistics(self, summary: dict):
        """Display batch analysis statistics."""
        
        power_stats = summary['power_statistics']
        eff_stats = summary['efficiency_statistics']
        quality_stats = summary['quality_statistics']
        
        print(f"\n⚡ POWER ANALYSIS:")
        print(f"   String 1 Average: {power_stats['string1_avg']:.1f}W")
        print(f"   String 2 Average: {power_stats['string2_avg']:.1f}W")
        print(f"   Maximum Imbalance: {power_stats['max_imbalance']:.1f}%")
        print(f"   Average Imbalance: {power_stats['avg_imbalance']:.1f}%")
        
        print(f"\n⚙️ EFFICIENCY ANALYSIS:")
        print(f"   String 1 Average: {eff_stats['string1_avg']:.1f}%")
        print(f"   String 2 Average: {eff_stats['string2_avg']:.1f}%")
        print(f"   Efficiency Range: {eff_stats['min_efficiency']:.1f}% - {eff_stats['max_efficiency']:.1f}%")
        
        print(f"\n🔍 QUALITY ANALYSIS:")
        print(f"   Average Quality: {quality_stats['avg_quality']:.1f}/100")
        print(f"   Quality Range: {quality_stats['min_quality']:.1f} - {quality_stats['max_quality']:.1f}")
        
        print(f"\n🚨 ISSUE SUMMARY:")
        print(f"   Systems with Issues: {summary['systems_with_issues']}/{summary['total_systems']}")
        issue_rate = (summary['systems_with_issues'] / summary['total_systems']) * 100
        print(f"   Issue Rate: {issue_rate:.1f}%")
    
    def save_monitoring_history(self, filename: str = None):
        """Save monitoring history to JSON file."""
        if not self.monitoring_history:
            print("📝 No monitoring history to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'string_monitoring_history_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.monitoring_history, f, indent=2, default=str)
        
        print(f"💾 Monitoring history saved: {filename}")
        return filename

def main():
    """Main application demonstration."""
    
    print("🚀 DUAL-STRING I-V CURVE MONITORING APPLICATION")
    print("=" * 70)
    print("This application demonstrates practical monitoring of dual-string")
    print("MPPT systems with comprehensive analysis and issue detection.")
    print()
    
    # Initialize monitoring application
    monitor = StringMonitoringApplication()
    
    # Single system analysis demonstration
    print("🔍 SINGLE SYSTEM ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    single_file = "inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    if Path(single_file).exists():
        result = monitor.analyze_single_system(single_file)
        if result:
            print("\n✅ Single system analysis completed successfully!")
    else:
        print(f"❌ Test file not found: {single_file}")
    
    # Batch analysis demonstration
    print("\n\n🔄 BATCH ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    batch_summary = monitor.batch_analysis()
    
    if batch_summary:
        print("\n✅ Batch analysis completed successfully!")
        
        # Save monitoring history
        history_file = monitor.save_monitoring_history()
        
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 30)
        print(f"• Analysis sessions completed: {len(monitor.monitoring_history)}")
        print(f"• Monitoring history saved: {history_file}")
        print(f"• Dashboards and reports generated for each system")
        
    print(f"\n🎯 NEXT STEPS:")
    print("-" * 20)
    print("1. Review generated dashboards for visual analysis")
    print("2. Check detailed reports for technical specifications")  
    print("3. Address any identified issues based on recommendations")
    print("4. Set up regular monitoring schedule using this system")
    print("5. Track performance trends over time")
    
    print(f"\n🎉 Monitoring application demonstration complete!")

if __name__ == "__main__":
    main()
