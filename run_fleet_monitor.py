#!/usr/bin/env python3
"""
Fleet Monitor Runner
===================

Enhanced runner for the automated fleet scheduler with improved monitoring display
and configuration options.

Usage:
    python run_fleet_monitor.py [duration_minutes]

Author: Advanced PV Analytics System
Date: June 6, 2025
"""

import sys
import time
import json
from datetime import datetime, timedelta
from automated_fleet_scheduler import AutomatedFleetScheduler, create_sample_config
import os

def display_fleet_status(scheduler):
    """Display detailed fleet status in a formatted way."""
    fleet_health = scheduler.get_fleet_health()
    
    if fleet_health.get('total_systems', 0) == 0:
        print("⚠️  No systems found or insufficient data")
        return
    
    print(f"\n{'='*80}")
    print(f"🔄 REAL-TIME FLEET STATUS - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    
    print(f"📊 Fleet Overview:")
    print(f"   Total Systems: {fleet_health['total_systems']}")
    print(f"   Average Health: {fleet_health['average_health_score']:.1f} ± {fleet_health['health_score_std']:.1f}")
    print(f"   Average Efficiency: {fleet_health['average_efficiency']:.1f}% ± {fleet_health['efficiency_std']:.1f}%")
    
    print(f"\n🚨 System Status:")
    print(f"   🔴 Critical: {fleet_health['systems_critical']}")
    print(f"   🟡 Warning: {fleet_health['systems_warning']}")
    print(f"   🟢 Healthy: {fleet_health['systems_healthy']}")
    print(f"   ⚠️  With Faults: {fleet_health['systems_with_faults']}")
    
    # Display individual system details
    print(f"\n📋 Individual System Status:")
    print(f"{'System ID':<12} {'Health':<8} {'Efficiency':<12} {'Power (W)':<12} {'Temp (°C)':<10} {'Trend':<10}")
    print(f"{'-'*70}")
    
    for system_id, status in scheduler.system_statuses.items():
        health_icon = "🔴" if status.health_score < 50 else "🟡" if status.health_score < 70 else "🟢"
        trend_icon = "📈" if status.performance_trend == "improving" else "📉" if status.performance_trend == "declining" else "➡️"
        
        print(f"{system_id:<12} {health_icon} {status.health_score:>5.1f} {status.efficiency:>10.1f}% {status.power_output:>10.0f} {status.temperature:>8.1f} {trend_icon} {status.performance_trend:<8}")

def run_fleet_monitor(duration_minutes=5):
    """Run the fleet monitor for specified duration."""
    print("🚀 AUTOMATED FLEET SCHEDULER - REAL-TIME MONITORING")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duration: {duration_minutes} minutes")
    print("=" * 80)
    
    # Create sample config if it doesn't exist
    if not os.path.exists('scheduler_config.json'):
        print("📝 Creating sample configuration...")
        create_sample_config()
    
    # Initialize scheduler
    print("🔧 Initializing fleet scheduler...")
    scheduler = AutomatedFleetScheduler()
    
    try:
        # Start monitoring
        print("🔄 Starting fleet monitoring...")
        scheduler.start_monitoring()
        
        # Display instructions
        print(f"\n📊 Monitor running for {duration_minutes} minutes")
        print("💡 Check the following files for detailed analysis:")
        print("   • fleet_summary_*.json - Real-time fleet summaries")
        print("   • fleet_scheduler.log - Detailed system logs")
        print("   • fleet_monitoring.db - SQLite database with alerts and tasks")
        print("\n⚡ Press Ctrl+C to stop monitoring early")
        print("=" * 80)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        display_interval = 15  # Update display every 15 seconds
        last_display = datetime.now() - timedelta(seconds=display_interval)
        
        while datetime.now() < end_time:
            current_time = datetime.now()
            
            # Update display every interval
            if (current_time - last_display).seconds >= display_interval:
                display_fleet_status(scheduler)
                last_display = current_time
                
                # Show remaining time
                remaining = end_time - current_time
                remaining_seconds = int(remaining.total_seconds())
                print(f"\n⏱️  Remaining: {remaining_seconds//60:02d}:{remaining_seconds%60:02d}")
                print("=" * 80)
            
            time.sleep(1)
        
        print("\n✅ Monitoring completed successfully!")
        
        # Final summary
        print("\n📈 FINAL FLEET HEALTH REPORT")
        print("=" * 80)
        display_fleet_status(scheduler)
        
        # Show generated files
        print(f"\n📁 Generated Files:")
        fleet_summaries = [f for f in os.listdir('.') if f.startswith('fleet_summary_') and f.endswith('.json')]
        if fleet_summaries:
            latest_summary = max(fleet_summaries)
            print(f"   • Latest Summary: {latest_summary}")
        
        if os.path.exists('fleet_scheduler.log'):
            print(f"   • Log File: fleet_scheduler.log")
        
        if os.path.exists('fleet_monitoring.db'):
            print(f"   • Database: fleet_monitoring.db")
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error in monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scheduler.stop_monitoring()
        print("\n🔚 Fleet scheduler stopped")
        print("=" * 80)

def main():
    """Main function."""
    # Get duration from command line argument
    duration = 5  # Default 5 minutes
    
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            print("❌ Invalid duration. Using default 5 minutes.")
    
    run_fleet_monitor(duration)

if __name__ == "__main__":
    main()
