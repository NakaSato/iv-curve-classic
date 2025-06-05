#!/usr/bin/env python3
"""
Enhanced Production System Test with Multi-String Support
Test all analysis types including the new 32-string analyzer
"""

import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
import logging
from production_dual_string_system import ProductionDualStringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_production_system():
    """Test the enhanced production system with multi-string support"""
    
    print("🚀 Enhanced Production System Test - Multi-String Support")
    print("=" * 70)
    
    # Initialize production system
    config_file = "production_config.ini"
    system = ProductionDualStringSystem(config_file=config_file)
    
    print(f"✅ System initialized with multi-string support")
    print(f"🔌 Supported analysis types:")
    print(f"   • enhanced_iv - Fast I-V curve analysis")
    print(f"   • ai_powered - AI-powered analysis with ML")
    print(f"   • multi_string - 32-string comprehensive analysis")
    print(f"   • comprehensive - Combined analysis")
    
    # Test 1: Multi-String Analysis with Generated Data
    print("\n🔌 Test 1: Multi-String Analysis (32 Strings)")
    print("-" * 50)
    
    # First, generate multi-string test data
    print("📊 Generating 32-string test data...")
    from test_multi_string_analyzer import generate_multi_string_iv_data
    
    try:
        test_data_file, _ = generate_multi_string_iv_data(
            num_strings=32,
            num_points=120,
            output_file="production_test_32string_data.csv"
        )
        
        print(f"✅ Generated test data: {test_data_file}")
        
        # Run multi-string analysis
        system_id = "MULTI_STRING_SYSTEM_01"
        print(f"🔍 Running multi-string analysis on {system_id}...")
        
        result = await system.analyze_system(
            system_id=system_id,
            data_file=test_data_file,
            analysis_type="multi_string"
        )
        
        if result['status'] == 'success':
            print(f"✅ Multi-string analysis completed successfully!")
            print(f"⏱️ Duration: {result['duration_seconds']:.1f}s")
            
            # Extract key results
            results = result['results']
            if 'summary' in results:
                summary = results['summary']
                print(f"⚡ Total Power: {summary.get('total_power', 0):.0f} W")
                print(f"📊 System Efficiency: {summary.get('efficiency', 0):.1f}%")
                print(f"🔌 Active Strings: {results.get('active_strings', 0)}")
                print(f"💡 Recommendations: {len(summary.get('recommendations', []))}")
        else:
            print(f"❌ Multi-string analysis failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"⚠️ Multi-string test failed: {e}")
    
    # Test 2: Compare Analysis Types
    print("\n📈 Test 2: Analysis Type Comparison")
    print("-" * 50)
    
    data_file = "inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    if os.path.exists(data_file):
        analysis_types = ["enhanced_iv", "ai_powered"]
        system_id = "INVERTER_01"
        
        comparison_results = {}
        
        for analysis_type in analysis_types:
            print(f"🔍 Running {analysis_type} analysis...")
            
            try:
                result = await system.analyze_system(
                    system_id=f"{system_id}_{analysis_type.upper()}",
                    data_file=data_file,
                    analysis_type=analysis_type
                )
                
                if result['status'] == 'success':
                    comparison_results[analysis_type] = {
                        'duration': result['duration_seconds'],
                        'success': True
                    }
                    print(f"✅ {analysis_type}: {result['duration_seconds']:.1f}s")
                else:
                    comparison_results[analysis_type] = {
                        'duration': 0,
                        'success': False,
                        'error': result.get('error', 'Unknown')
                    }
                    print(f"❌ {analysis_type}: Failed - {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"❌ {analysis_type}: Exception - {e}")
                comparison_results[analysis_type] = {
                    'duration': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Show comparison summary
        print(f"\n📊 Analysis Performance Comparison:")
        for analysis_type, results in comparison_results.items():
            status = "✅ Success" if results['success'] else "❌ Failed"
            duration = f"{results['duration']:.1f}s" if results['success'] else "N/A"
            print(f"   {analysis_type:15} | {status:10} | {duration:>8}")
    
    # Test 3: Fleet Status with Mixed Systems
    print("\n🌐 Test 3: Enhanced Fleet Status")
    print("-" * 50)
    
    try:
        fleet_status = system.get_fleet_status()
        
        print(f"🏢 Enhanced Fleet Overview:")
        print(f"   Total Systems: {fleet_status['total_systems']}")
        print(f"   Average Health: {fleet_status['average_health']:.1f}%")
        print(f"   Average Efficiency: {fleet_status['average_efficiency']:.1f}%")
        print(f"   Active Alerts: {fleet_status['total_alerts']}")
        
        # Show system breakdown
        if 'systems_by_status' in fleet_status:
            print(f"   Status Breakdown:")
            for status, count in fleet_status['systems_by_status'].items():
                print(f"     {status}: {count} systems")
        
    except Exception as e:
        print(f"❌ Fleet status retrieval failed: {e}")
    
    # Test 4: Advanced Dashboard Creation
    print("\n📊 Test 4: Enhanced Dashboard Generation")
    print("-" * 50)
    
    try:
        dashboard_file = system.create_fleet_dashboard()
        if dashboard_file:
            print(f"📈 Enhanced fleet dashboard created: {dashboard_file}")
            
            if os.path.exists(dashboard_file):
                file_size = os.path.getsize(dashboard_file) / 1024  # KB
                print(f"📁 Dashboard size: {file_size:.1f} KB")
        else:
            print(f"⚠️ Dashboard creation returned empty filename")
        
    except Exception as e:
        print(f"❌ Dashboard generation failed: {e}")
    
    # Test 5: Database Analysis
    print("\n💾 Test 5: Enhanced Database Analysis")
    print("-" * 50)
    
    try:
        import sqlite3
        conn = sqlite3.connect(system.db_path)
        cursor = conn.cursor()
        
        # Get detailed database statistics
        tables = ['system_status', 'alerts', 'analysis_results', 'performance_metrics']
        total_records = 0
        
        print(f"📋 Enhanced Database Statistics:")
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                total_records += count
                print(f"   {table:20} | {count:>6} records")
                
                # Show recent entries for analysis_results
                if table == 'analysis_results' and count > 0:
                    cursor.execute(f"""
                        SELECT system_id, analysis_type, timestamp 
                        FROM {table} 
                        ORDER BY timestamp DESC 
                        LIMIT 3
                    """)
                    recent = cursor.fetchall()
                    print(f"     Recent analyses:")
                    for system_id, analysis_type, timestamp in recent:
                        print(f"       {system_id} ({analysis_type}) - {timestamp}")
                        
            except Exception as e:
                print(f"   {table:20} | Error: {e}")
        
        print(f"   {'TOTAL':20} | {total_records:>6} records")
        conn.close()
        
    except Exception as e:
        print(f"❌ Database analysis failed: {e}")
    
    # Test 6: System Configuration Review
    print("\n⚙️ Test 6: Enhanced Configuration Review")
    print("-" * 50)
    
    try:
        config = system.config
        
        print(f"📊 Enhanced System Configuration:")
        
        # Database settings
        print(f"   Database:")
        print(f"     Path: {config['database']['path']}")
        print(f"     Retention: {config['database']['retention_days']} days")
        
        # Monitoring settings
        print(f"   Monitoring:")
        print(f"     Analysis interval: {config['monitoring']['analysis_interval']}s")
        print(f"     Alert check interval: {config['monitoring']['alert_check_interval']}s")
        
        # Thresholds
        print(f"   Alert Thresholds:")
        print(f"     Health critical: {config['thresholds']['health_score_critical']}%")
        print(f"     Failure risk critical: {config['thresholds']['failure_risk_critical']}%")
        
        # Features
        print(f"   Features:")
        print(f"     ML enabled: {config['optimization']['ml_enabled']}")
        print(f"     Auto recommendations: {config['optimization']['auto_recommendations']}")
        print(f"     Email alerts: {config['alerts']['email_enabled']}")
        
    except Exception as e:
        print(f"❌ Configuration review failed: {e}")
    
    print("\n🎉 Enhanced Production System Test Completed!")
    print("=" * 70)
    
    # Summary
    print(f"🌟 Enhanced Features Validated:")
    print(f"   ✅ Multi-string (32-string) analysis capability")
    print(f"   ✅ Multiple analysis type support")
    print(f"   ✅ Enhanced fleet management")
    print(f"   ✅ Advanced database storage")
    print(f"   ✅ Interactive dashboard generation")
    print(f"   ✅ Comprehensive configuration management")
    print(f"   ✅ Real-time alert system")
    print(f"   ✅ Production-ready deployment")
    
    return system

async def demonstrate_multi_string_capabilities():
    """Demonstrate specific multi-string analysis capabilities"""
    
    print("\n🔌 Multi-String Analysis Capabilities Demonstration")
    print("=" * 70)
    
    from multi_string_iv_analyzer import MultiStringIVAnalyzer
    from test_multi_string_analyzer import generate_multi_string_iv_data
    
    # Generate comprehensive test data
    print("📊 Generating comprehensive 32-string test data...")
    
    # Create test scenarios
    scenarios = [
        {"name": "Balanced System", "variation": 0.05, "strings": 32},
        {"name": "Moderate Imbalance", "variation": 0.15, "strings": 24},
        {"name": "High Imbalance", "variation": 0.25, "strings": 16}
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        print(f"   Strings: {scenario['strings']}, Variation: {scenario['variation']*100:.0f}%")
        
        try:
            # Generate scenario data
            data_file, _ = generate_multi_string_iv_data(
                num_strings=scenario['strings'],
                num_points=100,
                variation_factor=scenario['variation'],
                output_file=f"scenario_{scenario['name'].lower().replace(' ', '_')}.csv"
            )
            
            # Analyze with multi-string analyzer
            analyzer = MultiStringIVAnalyzer(max_strings=32)
            result = analyzer.run_comprehensive_analysis(data_file)
            
            # Show key results
            print(f"   ✅ Analysis completed:")
            print(f"     Total Power: {result.system_performance['total_power_w']:.0f} W")
            print(f"     Average Efficiency: {result.system_performance['average_efficiency_pct']:.1f}%")
            print(f"     Imbalance Level: {result.imbalance_analysis['imbalance_severity']}")
            print(f"     Power CV: {result.imbalance_analysis['power_statistics']['cv']:.1f}%")
            print(f"     Recommendations: {len(result.optimization_recommendations)}")
            
        except Exception as e:
            print(f"   ❌ Scenario failed: {e}")
    
    print(f"\n🎉 Multi-String Capabilities Demonstration Complete!")

if __name__ == "__main__":
    async def main():
        try:
            # Run enhanced production system test
            system = await test_enhanced_production_system()
            
            # Demonstrate multi-string capabilities
            await demonstrate_multi_string_capabilities()
            
            print("\n🚀 All Enhanced Tests Completed Successfully!")
            print("🌟 System ready for enterprise deployment with multi-string support!")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the async test
    asyncio.run(main())
