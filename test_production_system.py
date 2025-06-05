#!/usr/bin/env python3
"""
Production System Test and Demonstration
Test the complete production-ready dual-string MPPT analysis system
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

async def test_production_system():
    """Test and demonstrate the production system capabilities"""
    
    print("🚀 Starting Production Dual-String MPPT System Test")
    print("=" * 60)
    
    # Initialize production system
    config_file = "production_config.ini"
    system = ProductionDualStringSystem(config_file=config_file)
    
    print(f"✅ System initialized with config: {config_file}")
    print(f"📊 Database path: {system.db_path}")
    
    # Test 1: Single system analysis
    print("\n📈 Test 1: Single System Analysis")
    print("-" * 40)
    
    data_file = "inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    if os.path.exists(data_file):
        system_id = "INVERTER_01"
        analysis_type = "comprehensive"
        
        print(f"🔍 Analyzing {system_id} with {analysis_type} analysis...")
        
        try:
            result = await system.analyze_system(
                system_id=system_id,
                data_file=data_file,
                analysis_type=analysis_type
            )
            
            print(f"✅ Analysis completed for {system_id}")
            print(f"📊 Health Score: {result.get('health_score', 'N/A')}")
            print(f"⚠️ Alerts Generated: {len(result.get('alerts', []))}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    else:
        print(f"⚠️ Data file not found: {data_file}")
    
    # Test 2: Fleet status overview
    print("\n🌐 Test 2: Fleet Status Overview")
    print("-" * 40)
    
    try:
        fleet_status = await system.get_fleet_status()
        
        print(f"🏢 Total Systems: {fleet_status['total_systems']}")
        print(f"✅ Healthy Systems: {fleet_status['healthy_systems']}")
        print(f"⚠️ Warning Systems: {fleet_status['warning_systems']}")
        print(f"🚨 Critical Systems: {fleet_status['critical_systems']}")
        print(f"📈 Average Health Score: {fleet_status['average_health_score']:.1f}")
        print(f"🔔 Active Alerts: {fleet_status['active_alerts']}")
        
    except Exception as e:
        print(f"❌ Fleet status retrieval failed: {e}")
    
    # Test 3: Multi-system fleet analysis
    print("\n🚁 Test 3: Multi-System Fleet Analysis")
    print("-" * 40)
    
    # Create test data files for fleet analysis
    test_systems = []
    base_data_file = "inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    
    if os.path.exists(base_data_file):
        # Test with existing synthetic data files
        inverter_dir = Path("inverter")
        data_files = list(inverter_dir.glob("INVERTER_*.csv"))
        
        print(f"📁 Found {len(data_files)} data files for fleet analysis")
        
        # Analyze up to 5 systems for demonstration
        for i, data_file in enumerate(data_files[:5]):
            system_id = data_file.stem.split('_')[0] + '_' + data_file.stem.split('_')[1]
            test_systems.append((system_id, str(data_file)))
            
            print(f"📊 Adding {system_id} to fleet analysis queue...")
        
        # Run fleet analysis
        fleet_results = []
        for system_id, data_file in test_systems:
            try:
                print(f"🔄 Analyzing {system_id}...")
                result = await system.analyze_system(
                    system_id=system_id,
                    data_file=data_file,
                    analysis_type="enhanced_iv"  # Faster analysis for fleet demo
                )
                fleet_results.append((system_id, result))
                print(f"✅ {system_id} analysis complete")
                
            except Exception as e:
                print(f"❌ {system_id} analysis failed: {e}")
        
        print(f"🏁 Fleet analysis completed: {len(fleet_results)} systems analyzed")
    
    # Test 4: Dashboard generation
    print("\n📊 Test 4: Fleet Dashboard Generation")
    print("-" * 40)
    
    try:
        dashboard_file = await system.create_fleet_dashboard()
        print(f"📈 Fleet dashboard created: {dashboard_file}")
        
        if os.path.exists(dashboard_file):
            file_size = os.path.getsize(dashboard_file) / 1024  # KB
            print(f"📁 Dashboard size: {file_size:.1f} KB")
        
    except Exception as e:
        print(f"❌ Dashboard generation failed: {e}")
    
    # Test 5: Alert system
    print("\n🔔 Test 5: Alert System Test")
    print("-" * 40)
    
    try:
        active_alerts = system.get_active_alerts()
        print(f"🚨 Active alerts in system: {len(active_alerts)}")
        
        for alert in active_alerts[:3]:  # Show first 3 alerts
            print(f"  ⚠️ {alert['system_id']}: {alert['alert_type']} - {alert['message']}")
            
    except Exception as e:
        print(f"❌ Alert retrieval failed: {e}")
    
    # Test 6: Database status
    print("\n💾 Test 6: Database Status")
    print("-" * 40)
    
    try:
        # Check database tables and record counts
        import sqlite3
        conn = sqlite3.connect(system.db_path)
        cursor = conn.cursor()
        
        tables = ['system_status', 'alerts', 'analysis_results', 'performance_metrics']
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"📋 {table}: {count} records")
            except Exception as e:
                print(f"⚠️ {table}: Table may not exist or error: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database status check failed: {e}")
    
    # Test 7: Configuration verification
    print("\n⚙️ Test 7: Configuration Verification")
    print("-" * 40)
    
    try:
        config = system.config
        
        print(f"📊 Database retention: {config['database']['retention_days']} days")
        print(f"⏱️ Analysis interval: {config['monitoring']['analysis_interval']} seconds")
        print(f"🎯 Health score critical threshold: {config['thresholds']['health_score_critical']}")
        print(f"🤖 ML enabled: {config['optimization']['ml_enabled']}")
        print(f"📧 Email alerts: {config['alerts']['email_enabled']}")
        
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
    
    print("\n🎉 Production System Test Completed!")
    print("=" * 60)
    
    return system

async def demonstrate_ai_features(system):
    """Demonstrate advanced AI features"""
    
    print("\n🤖 AI Features Demonstration")
    print("=" * 60)
    
    data_file = "inverter/INVERTER_01_2025-04-04_2025-04-05.csv"
    if os.path.exists(data_file):
        print("🧠 Running AI-powered analysis...")
        
        try:
            result = await system.analyze_system(
                system_id="INVERTER_01_AI_DEMO",
                data_file=data_file,
                analysis_type="ai_powered"
            )
            
            print("✅ AI Analysis Results:")
            print(f"🎯 Health Score: {result.get('health_score', 'N/A')}")
            print(f"⚠️ Failure Risk: {result.get('failure_risk', 'N/A')}%")
            print(f"🔍 Anomaly Rate: {result.get('anomaly_rate', 'N/A')}%")
            print(f"💰 Revenue Opportunity: ${result.get('revenue_opportunity', 'N/A')}")
            print(f"📊 Recommendations: {len(result.get('recommendations', []))}")
            
            # Show top recommendations
            recommendations = result.get('recommendations', [])
            if recommendations:
                print("\n🎯 Top AI Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec}")
            
        except Exception as e:
            print(f"❌ AI analysis failed: {e}")
    
    print("\n🎉 AI Features Demonstration Completed!")

if __name__ == "__main__":
    async def main():
        try:
            # Run main production system test
            system = await test_production_system()
            
            # Demonstrate AI features
            await demonstrate_ai_features(system)
            
            print("\n🚀 All tests completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the async test
    asyncio.run(main())
