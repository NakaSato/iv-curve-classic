#!/usr/bin/env python3
"""
Test Multi-Inverter Fleet Analysis with Real Data Cleaning
Demonstrates comprehensive fleet management capabilities
"""

from production_dual_string_system import ProductionDualStringSystem
import asyncio
import glob
from pathlib import Path

async def test_multi_inverter_fleet():
    """Test comprehensive fleet analysis"""
    print("🚀 MULTI-INVERTER FLEET ANALYSIS TEST")
    print("=" * 80)
    
    system = ProductionDualStringSystem()
    
    # Get all available inverter files
    inverter_files = glob.glob('./inverter/INVERTER_*.csv')
    print(f'🔍 Found {len(inverter_files)} inverter data files')
    
    results = {}
    
    # Analyze first 3 inverters with real data cleaning
    for i, file_path in enumerate(inverter_files[:3], 1):
        inverter_id = f'INVERTER_{i:02d}'
        print(f'\n🔄 Analyzing {inverter_id} with real data cleaning...')
        
        try:
            result = await system.analyze_system(inverter_id, file_path, 'real_inverter_cleaning')
            results[inverter_id] = result
            
            if result['status'] == 'success':
                summary = result['results']['summary']
                print(f'✅ {inverter_id}: Quality={summary["data_quality_score"]:.1f}%, '
                      f'Efficiency={summary["efficiency"]:.1f}%, '
                      f'Power={summary["total_power"]/1000:.1f}kW')
            else:
                print(f'❌ {inverter_id}: Failed - {result.get("error", "Unknown")}')
        except Exception as e:
            print(f'❌ {inverter_id}: Exception - {str(e)}')
            continue
    
    # Generate fleet status
    print(f'\n📊 FLEET ANALYSIS SUMMARY')
    print("-" * 50)
    fleet_status = system.get_fleet_status()
    print(f'🏭 Total Systems: {fleet_status["total_systems"]}')
    print(f'⚖️ Average Health: {fleet_status["average_health"]:.1f}%')
    print(f'⚡ Average Efficiency: {fleet_status["average_efficiency"]:.1f}%')
    print(f'🚨 Active Alerts: {fleet_status["total_alerts"]}')
    
    # System status breakdown
    print(f'\n📈 SYSTEM STATUS DISTRIBUTION')
    print("-" * 50)
    for status, count in fleet_status["systems_by_status"].items():
        print(f'  {status}: {count} systems')
    
    # Individual system details
    print(f'\n🔍 INDIVIDUAL SYSTEM DETAILS')
    print("-" * 50)
    for system_id, system_data in fleet_status.get('systems', {}).items():
        print(f'System {system_id}:')
        print(f'  • Health: {system_data["health_score"]:.1f}%')
        print(f'  • Risk: {system_data["failure_risk"]:.1f}%')
        print(f'  • Power: {system_data["power_output"]:.1f}W')
        print(f'  • Status: {system_data["maintenance_status"]}')
    
    # Generate fleet dashboard and report
    print(f'\n🎨 GENERATING FLEET OUTPUTS')
    print("-" * 50)
    
    dashboard_file = system.create_fleet_dashboard()
    if dashboard_file:
        print(f'📊 Fleet dashboard: {dashboard_file}')
    
    report_file = system.generate_fleet_report()
    if report_file:
        print(f'📄 Fleet report: {report_file}')
    
    print(f'\n🎉 FLEET ANALYSIS COMPLETE!')
    print(f'✅ Successfully analyzed {len([r for r in results.values() if r.get("status") == "success"])} systems')
    print(f'❌ Failed analyses: {len([r for r in results.values() if r.get("status") != "success"])}')

if __name__ == "__main__":
    asyncio.run(test_multi_inverter_fleet())
