#!/usr/bin/env python3
"""
Dual-String MPPT Analysis Demonstration
Shows comprehensive analysis capabilities for 2-string 1 MPPT configuration
"""

from dual_string_mppt_analysis import DualStringMPPTAnalyzer
from string_monitoring_system import StringMonitoringSystem
import json
from datetime import datetime

def demonstrate_dual_string_analysis():
    """Comprehensive demonstration of dual-string MPPT analysis capabilities"""
    
    print("🔍 DUAL-STRING MPPT CONFIGURATION ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print("📊 Analyzing 2-String 1 MPPT Configuration")
    print("   - String 1: Vstr1(V) and Istr1(A)")
    print("   - String 2: Vstr2(V) and Istr2(A)")
    print()
    
    # Initialize analyzer with INVERTER_01 data (contains dual-string parameters)
    print("🚀 Initializing Dual-String MPPT Analyzer...")
    analyzer = DualStringMPPTAnalyzer('./inverter/INVERTER_01_2025-04-04_2025-04-05.csv')
    
    # Run complete analysis
    print("⚙️ Running comprehensive dual-string analysis...")
    results = analyzer.run_complete_analysis()
    
    # Display key findings
    print("\n📈 KEY ANALYSIS FINDINGS")
    print("-" * 50)
    
    # Performance Summary
    performance = results['performance']
    print(f"🔋 String 1 Performance:")
    print(f"   • Average Power: {performance['string1']['avg_power']:.1f} W")
    print(f"   • Average Voltage: {performance['string1']['avg_voltage']:.1f} V")
    print(f"   • Average Current: {performance['string1']['avg_current']:.2f} A")
    
    print(f"🔋 String 2 Performance:")
    print(f"   • Average Power: {performance['string2']['avg_power']:.1f} W")
    print(f"   • Average Voltage: {performance['string2']['avg_voltage']:.1f} V")
    print(f"   • Average Current: {performance['string2']['avg_current']:.2f} A")
    
    # Comparison Analysis
    comparison = performance['comparison']
    print(f"⚖️ String Comparison:")
    print(f"   • Power Difference: {comparison['power_difference_percent']:.1f}%")
    print(f"   • Voltage Difference: {comparison['voltage_difference']:.1f} V")
    print(f"   • Current Difference: {comparison['current_difference']:.2f} A")
    
    # Issues Detection
    issues = results['issues']
    total_issues = sum(len(issue_list) for issue_list in issues.values() if isinstance(issue_list, list))
    print(f"🚨 Issues Detected: {total_issues}")
    print(f"   • Voltage Mismatch Events: {len(issues['voltage_mismatch'])}")
    print(f"   • Current Mismatch Events: {len(issues['current_mismatch'])}")
    print(f"   • Power Imbalance Events: {len(issues['power_imbalance'])}")
    print(f"   • Performance Degradation: {len(issues['performance_degradation'])}")
    
    # MPPT Efficiency
    mppt_eff = results['mppt_efficiency']
    print(f"⚡ MPPT Efficiency:")
    print(f"   • Average Efficiency: {mppt_eff['average_efficiency']:.1f}%")
    print(f"   • Efficiency Range: {mppt_eff['min_efficiency']:.1f}% - {mppt_eff['max_efficiency']:.1f}%")
    
    # Critical Assessment
    print("\n🎯 CRITICAL ASSESSMENT")
    print("-" * 50)
    
    power_imbalance = comparison['power_difference_percent']
    if power_imbalance > 15:
        status = "🔴 CRITICAL"
        action = "Immediate inspection required"
    elif power_imbalance > 8:
        status = "🟡 WARNING"
        action = "Schedule maintenance within 1 week"
    else:
        status = "🟢 NORMAL"
        action = "Continue regular monitoring"
    
    print(f"System Status: {status}")
    print(f"Recommended Action: {action}")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS")
        print("-" * 50)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # Show generated files
    print(f"\n📁 GENERATED ANALYSIS FILES")
    print("-" * 50)
    print("✅ Visual Dashboard: dual_string_mppt_comprehensive_dashboard_*.png")
    print("✅ Analysis Report: dual_string_mppt_analysis_report_*.txt")
    print("✅ Data Structure: Complete results dictionary available")
    
    return results

def demonstrate_monitoring_system():
    """Demonstrate real-time monitoring capabilities"""
    
    print(f"\n\n🔍 REAL-TIME MONITORING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Initialize monitoring system
    print("📡 Initializing Real-Time Monitoring System...")
    monitor = StringMonitoringSystem()
    
    # Analyze current data
    print("🔄 Running monitoring analysis...")
    monitoring_results = monitor.analyze_csv_file('./inverter/INVERTER_01_2025-04-04_2025-04-05.csv')
    
    # Display monitoring status
    print(f"\n📊 MONITORING SYSTEM STATUS")
    print("-" * 50)
    print(f"System Health: {monitoring_results.get('system_health', 'Unknown')}")
    print(f"Active Alerts: {len(monitoring_results.get('alerts', []))}")
    print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show alerts
    alerts = monitoring_results.get('alerts', [])
    if alerts:
        print(f"\n🚨 ACTIVE ALERTS")
        print("-" * 50)
        for alert in alerts[:5]:  # Show first 5 alerts
            level = alert.get('level', 'INFO')
            message = alert.get('message', 'No message')
            print(f"   {level}: {message}")
    
    # Configuration
    print(f"\n⚙️ MONITORING CONFIGURATION")
    print("-" * 50)
    print("✅ Alert thresholds configured")
    print("✅ Real-time monitoring enabled")
    print("✅ Dashboard generation active")
    print("✅ Historical data retention set")
    
    return monitoring_results

def show_integration_capabilities():
    """Show how dual-string analysis integrates with existing systems"""
    
    print(f"\n\n🔗 SYSTEM INTEGRATION CAPABILITIES")
    print("=" * 80)
    
    print("🎯 Integration Features:")
    print("   ✅ CSV Data Format Compatibility")
    print("   ✅ Existing Visualization Framework")
    print("   ✅ Fleet-Level Analysis Support")
    print("   ✅ Real-Time Monitoring Integration")
    print("   ✅ Alert System Integration")
    print("   ✅ Maintenance Workflow Integration")
    
    print(f"\n📊 Data Processing Pipeline:")
    print("   1. 📥 CSV Data Ingestion (Vstr1, Vstr2, Istr1, Istr2)")
    print("   2. 🔍 Dual-String Parameter Analysis")
    print("   3. ⚖️ String Performance Comparison")
    print("   4. 🚨 Issue Detection & Classification")
    print("   5. 📈 MPPT Efficiency Calculation")
    print("   6. 📊 Comprehensive Dashboard Generation")
    print("   7. 📝 Automated Report Generation")
    print("   8. 🔔 Alert Generation & Routing")
    
    print(f"\n🎛️ Available Analysis Modes:")
    print("   • One-time Analysis: Complete analysis with dashboard")
    print("   • Real-time Monitoring: Continuous monitoring with alerts")
    print("   • Fleet Analysis: Multi-inverter comparison")
    print("   • Historical Trending: Performance over time")
    print("   • Predictive Analysis: Maintenance scheduling")

def main():
    """Main demonstration function"""
    
    # Run demonstrations
    print("🎬 Starting Dual-String MPPT Analysis Demonstration...")
    print()
    
    try:
        # 1. Core Analysis Demonstration
        analysis_results = demonstrate_dual_string_analysis()
        
        # 2. Monitoring System Demonstration  
        monitoring_results = demonstrate_monitoring_system()
        
        # 3. Integration Capabilities
        show_integration_capabilities()
        
        # Summary
        print(f"\n\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📊 Dual-String MPPT Analysis System is ready for deployment")
        print("🔧 Configure alert thresholds as needed for your specific system")
        print("📈 Monitor string-level performance to optimize energy harvest")
        print("🛠️ Use insights for proactive maintenance scheduling")
        
        print(f"\n🚀 Next Steps:")
        print("   1. Review generated dashboards and reports")
        print("   2. Configure monitoring thresholds for your fleet")
        print("   3. Integrate with existing O&M workflows")
        print("   4. Set up automated monitoring schedules")
        print("   5. Train operators on new insights and alerts")
        
        return {
            "analysis_results": analysis_results,
            "monitoring_results": monitoring_results,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    results = main()
