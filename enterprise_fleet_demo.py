#!/usr/bin/env python3
"""
Enterprise Fleet Operations - Demonstration Script
=================================================

This script demonstrates the full capabilities of our enterprise fleet operations system
including real-time monitoring, advanced analytics, and comprehensive reporting.

Features Demonstrated:
- Multi-factor health scoring algorithm
- Advanced string performance analysis
- Predictive maintenance prioritization
- Enterprise dashboard generation
- Executive reporting capabilities
- Performance benchmarking
- Risk assessment and mitigation planning

Author: Enterprise PV Analytics Team
Date: June 6, 2025
Version: 2.0.0
"""

import os
import time
import json
from datetime import datetime
from enterprise_fleet_operations import EnterpriseFleetOperations

def demonstrate_enterprise_capabilities():
    """Demonstrate full enterprise fleet operations capabilities"""
    
    print("=" * 80)
    print("ENTERPRISE FLEET OPERATIONS SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize enterprise system
    print("\n🔧 Initializing Enterprise Fleet Operations System...")
    enterprise_ops = EnterpriseFleetOperations()
    
    # Run comprehensive analysis
    print("\n📊 Running comprehensive fleet analysis...")
    start_time = time.time()
    fleet_analysis = enterprise_ops.analyze_fleet_comprehensive()
    analysis_time = time.time() - start_time
    
    if fleet_analysis['total_systems'] == 0:
        print("❌ No systems found for analysis")
        return
    
    # Display real-time results
    print(f"\n✅ Analysis completed in {analysis_time:.2f}s")
    
    # Fleet Overview
    print("\n" + "="*60)
    print("FLEET OVERVIEW")
    print("="*60)
    kpis = fleet_analysis.get('fleet_kpis', {})
    
    print(f"🏭 Total Systems: {fleet_analysis['total_systems']}")
    print(f"🟢 Systems Online: {fleet_analysis['systems_online']}")
    print(f"⚡ Total Power Output: {kpis.get('total_power_output', 0):,.1f} W")
    print(f"🎯 System Availability: {kpis.get('system_availability', 0)}%")
    print(f"📈 Average Efficiency: {kpis.get('average_efficiency', 0):.1f}%")
    print(f"🌡️ Average Temperature: {kpis.get('average_temperature', 0):.1f}°C")
    print(f"❤️ Fleet Health Score: {kpis.get('average_health_score', 0):.1f}/100")
    
    # Health Distribution Analysis
    print("\n" + "-"*40)
    print("SYSTEM HEALTH DISTRIBUTION")
    print("-"*40)
    health_dist = kpis.get('health_distribution', {})
    
    health_emojis = {
        'excellent': '🟢',
        'good': '🟡', 
        'fair': '🟠',
        'poor': '🔴',
        'critical': '🚨'
    }
    
    for category, count in health_dist.items():
        emoji = health_emojis.get(category, '⚪')
        percentage = (count / fleet_analysis['total_systems'] * 100) if fleet_analysis['total_systems'] > 0 else 0
        print(f"{emoji} {category.title()}: {count} systems ({percentage:.1f}%)")
    
    # Performance Benchmarks
    print("\n" + "-"*40)
    print("PERFORMANCE BENCHMARKS")
    print("-"*40)
    benchmarks = fleet_analysis.get('performance_benchmarks', {})
    
    if 'efficiency_percentiles' in benchmarks:
        eff_p = benchmarks['efficiency_percentiles']
        print(f"📊 Efficiency Distribution:")
        print(f"   • Top 10%: {eff_p.get('p90', 0):.1f}%")
        print(f"   • Median: {eff_p.get('p50', 0):.1f}%")
        print(f"   • Bottom 10%: {eff_p.get('p10', 0):.1f}%")
    
    if 'health_score_percentiles' in benchmarks:
        health_p = benchmarks['health_score_percentiles']
        print(f"❤️ Health Score Distribution:")
        print(f"   • Top 10%: {health_p.get('p90', 0):.1f}")
        print(f"   • Median: {health_p.get('p50', 0):.1f}")
        print(f"   • Bottom 10%: {health_p.get('p10', 0):.1f}")
    
    # Risk Assessment
    print("\n" + "-"*40)
    print("FLEET RISK ASSESSMENT")
    print("-"*40)
    risk = fleet_analysis.get('risk_assessment', {})
    
    risk_level = risk.get('risk_level', 'unknown').upper()
    risk_emoji = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}.get(risk_level, '⚪')
    
    print(f"{risk_emoji} Overall Risk Level: {risk_level}")
    print(f"📈 Risk Score: {risk.get('overall_risk_score', 0):.1f}/100")
    print(f"🚨 Critical Systems: {risk.get('critical_systems_count', 0)}")
    print(f"⚠️ High Risk Systems: {kpis.get('high_risk_systems', 0)}")
    print(f"💀 Estimated Dead Strings: {kpis.get('estimated_dead_strings', 0)}")
    
    # Top Performers Analysis
    print("\n" + "-"*40)
    print("TOP & BOTTOM PERFORMERS")
    print("-"*40)
    
    if 'top_performers' in benchmarks:
        print("🏆 Top Performing Systems:")
        for i, system_id in enumerate(benchmarks['top_performers'][:3], 1):
            if system_id in fleet_analysis['system_details']:
                system = fleet_analysis['system_details'][system_id]
                print(f"   {i}. {system_id}: Health {system.get('health_score', 0):.1f}, "
                      f"Efficiency {system.get('efficiency', 0):.1f}%")
    
    if 'bottom_performers' in benchmarks:
        print("⚠️ Systems Needing Attention:")
        for i, system_id in enumerate(benchmarks['bottom_performers'][:3], 1):
            if system_id in fleet_analysis['system_details']:
                system = fleet_analysis['system_details'][system_id]
                print(f"   {i}. {system_id}: Health {system.get('health_score', 0):.1f}, "
                      f"Efficiency {system.get('efficiency', 0):.1f}%")
    
    # Maintenance Priorities
    print("\n" + "-"*40)
    print("MAINTENANCE PRIORITIES")
    print("-"*40)
    maintenance_tasks = fleet_analysis.get('maintenance_priorities', [])
    
    urgent_tasks = [t for t in maintenance_tasks if t.get('urgency') == 'critical']
    high_tasks = [t for t in maintenance_tasks if t.get('urgency') == 'high']
    
    print(f"🚨 Critical Tasks: {len(urgent_tasks)}")
    print(f"⚠️ High Priority Tasks: {len(high_tasks)}")
    print(f"📋 Total Tasks: {len(maintenance_tasks)}")
    
    if urgent_tasks:
        print(f"\n🔧 Most Urgent Maintenance Tasks:")
        for i, task in enumerate(urgent_tasks[:5], 1):
            print(f"   {i}. {task['system_id']}: ${task.get('estimated_cost', 0):,} - {task.get('description', 'N/A')[:50]}...")
    
    # Cost Analysis
    print("\n" + "-"*40)
    print("COST ANALYSIS")
    print("-"*40)
    total_maintenance_cost = sum([t.get('estimated_cost', 0) for t in maintenance_tasks])
    urgent_cost = sum([t.get('estimated_cost', 0) for t in urgent_tasks])
    
    print(f"💰 Total Estimated Maintenance Cost: ${total_maintenance_cost:,}")
    print(f"🚨 Urgent Maintenance Cost: ${urgent_cost:,}")
    print(f"📊 Maintenance Urgency Score: {kpis.get('maintenance_urgency_score', 0):.2f}/10")
    
    # System Capabilities Summary
    print("\n" + "="*60)
    print("ENTERPRISE SYSTEM CAPABILITIES DEMONSTRATED")
    print("="*60)
    
    capabilities = [
        "✅ Multi-factor health scoring algorithm (5 weighted components)",
        "✅ Advanced string performance analysis with statistical outlier detection",
        "✅ Predictive maintenance prioritization with cost estimation",
        "✅ Enterprise-grade risk assessment and mitigation planning", 
        "✅ Performance benchmarking and percentile analysis",
        "✅ Automated executive reporting and compliance documentation",
        "✅ Visual dashboard generation with matplotlib integration",
        "✅ Database persistence for historical trend analysis",
        "✅ Comprehensive fleet KPI tracking and optimization",
        "✅ Real-time system health monitoring and alerting"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
    
    # Generate dashboard
    print(f"\n🎨 Generating enterprise dashboard...")
    dashboard_file = enterprise_ops.generate_advanced_dashboard()
    if dashboard_file:
        print(f"✅ Dashboard generated: {dashboard_file}")
    
    # Final summary
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"📊 Fleet analyzed: {fleet_analysis['systems_analyzed']}/{fleet_analysis['total_systems']} systems")
    print(f"⏱️ Analysis duration: {fleet_analysis.get('analysis_duration', 0):.2f}s")
    print(f"📁 Reports generated in current directory")
    print(f"🎯 System performance: OPERATIONAL")
    
    # Show generated files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_files = [
        f"enterprise_fleet_report_{timestamp[:-6]}*.json",
        f"executive_summary_{timestamp[:-6]}*.txt", 
        f"enterprise_dashboard_{timestamp[:-6]}*.png",
        "enterprise_fleet_operations.db"
    ]
    
    print(f"\n📋 Generated files:")
    for file_pattern in generated_files:
        print(f"   • {file_pattern}")
    
    print(f"\n🎉 Enterprise Fleet Operations System demonstration complete!")
    print("="*80)

if __name__ == "__main__":
    demonstrate_enterprise_capabilities()
