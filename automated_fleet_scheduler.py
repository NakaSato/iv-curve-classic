#!/usr/bin/env python3
"""
Automated Fleet Scheduler and Real-Time Monitoring System
=========================================================

Comprehensive automated scheduling system for photovoltaic inverter fleet management
with real-time monitoring, predictive maintenance, and intelligent alert management.

Features:
- Continuous real-time monitoring loops
- Automated maintenance scheduling
- Intelligent alert prioritization
- Predictive fault detection
- Performance trend analysis
- Email and webhook notifications
- Resource optimization
- Compliance reporting

Author: Advanced PV Analytics System
Date: June 6, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import json
import os
import time
import threading
import queue
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
# import smtplib
# from email.mime.text import MimeText
# from email.mime.multipart import MimeMultipart
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fleet_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels for prioritization."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5

class MaintenanceType(Enum):
    """Types of maintenance tasks."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    ROUTINE = "routine"
    PREDICTIVE = "predictive"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Alert:
    """Comprehensive alert data structure."""
    id: str
    timestamp: datetime
    system_id: str
    severity: AlertSeverity
    category: str
    message: str
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False
    cooldown_until: Optional[datetime] = None

@dataclass
class MaintenanceTask:
    """Maintenance task data structure."""
    id: str
    system_id: str
    task_type: MaintenanceType
    priority: int
    description: str
    scheduled_date: datetime
    estimated_duration: timedelta
    status: TaskStatus
    assigned_technician: Optional[str] = None
    prerequisites: List[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    notes: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.prerequisites is None:
            self.prerequisites = []
        if self.notes is None:
            self.notes = []

@dataclass
class SystemStatus:
    """Real-time system status tracking."""
    system_id: str
    last_update: datetime
    health_score: float
    efficiency: float
    power_output: float
    temperature: float
    fault_codes: List[str]
    string_status: Dict[str, Any]
    performance_trend: str
    next_maintenance: Optional[datetime]

class FleetDatabase:
    """SQLite database manager for fleet data persistence."""
    
    def __init__(self, db_path: str = "fleet_monitoring.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    system_id TEXT,
                    severity INTEGER,
                    category TEXT,
                    message TEXT,
                    data TEXT,
                    acknowledged BOOLEAN,
                    resolved BOOLEAN,
                    escalated BOOLEAN,
                    cooldown_until TEXT
                )
            """)
            
            # Maintenance tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS maintenance_tasks (
                    id TEXT PRIMARY KEY,
                    system_id TEXT,
                    task_type TEXT,
                    priority INTEGER,
                    description TEXT,
                    scheduled_date TEXT,
                    estimated_duration TEXT,
                    status TEXT,
                    assigned_technician TEXT,
                    prerequisites TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    notes TEXT
                )
            """)
            
            # System status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_status (
                    system_id TEXT PRIMARY KEY,
                    last_update TEXT,
                    health_score REAL,
                    efficiency REAL,
                    power_output REAL,
                    temperature REAL,
                    fault_codes TEXT,
                    string_status TEXT,
                    performance_trend TEXT,
                    next_maintenance TEXT
                )
            """)
            
            # Performance history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT,
                    timestamp TEXT,
                    efficiency REAL,
                    power_output REAL,
                    temperature REAL,
                    health_score REAL
                )
            """)
            
            conn.commit()
    
    def save_alert(self, alert: Alert):
        """Save alert to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.timestamp.isoformat(),
                alert.system_id,
                alert.severity.value,
                alert.category,
                alert.message,
                json.dumps(alert.data),
                alert.acknowledged,
                alert.resolved,
                alert.escalated,
                alert.cooldown_until.isoformat() if alert.cooldown_until else None
            ))
            conn.commit()
    
    def save_maintenance_task(self, task: MaintenanceTask):
        """Save maintenance task to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO maintenance_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id,
                task.system_id,
                task.task_type.value,
                task.priority,
                task.description,
                task.scheduled_date.isoformat(),
                str(task.estimated_duration),
                task.status.value,
                task.assigned_technician,
                json.dumps(task.prerequisites),
                task.created_at.isoformat(),
                task.completed_at.isoformat() if task.completed_at else None,
                json.dumps(task.notes)
            ))
            conn.commit()
    
    def get_pending_tasks(self) -> List[MaintenanceTask]:
        """Get all pending maintenance tasks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM maintenance_tasks WHERE status = ? ORDER BY priority, scheduled_date
            """, (TaskStatus.PENDING.value,))
            
            tasks = []
            for row in cursor.fetchall():
                # Parse duration from string format
                duration_str = row[6]
                try:
                    if ':' in duration_str:
                        hours, minutes, seconds = duration_str.split(':')
                        duration = timedelta(hours=int(hours), minutes=int(minutes), seconds=float(seconds))
                    else:
                        duration = timedelta(seconds=float(duration_str))
                except:
                    duration = timedelta(hours=2)  # Default
                
                task = MaintenanceTask(
                    id=row[0],
                    system_id=row[1],
                    task_type=MaintenanceType(row[2]),
                    priority=row[3],
                    description=row[4],
                    scheduled_date=datetime.fromisoformat(row[5]),
                    estimated_duration=duration,
                    status=TaskStatus(row[7]),
                    assigned_technician=row[8],
                    prerequisites=json.loads(row[9]) if row[9] else [],
                    created_at=datetime.fromisoformat(row[10]),
                    completed_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    notes=json.loads(row[12]) if row[12] else []
                )
                tasks.append(task)
            return tasks

class NotificationManager:
    """Handles webhook notifications (simplified version)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_config = config.get('webhooks', {})
    
    def send_email_alert(self, alert: Alert, recipients: List[str]):
        """Email functionality disabled - logging only."""
        logger.info(f"Email alert would be sent for {alert.id}: {alert.message}")
    
    def send_webhook(self, alert: Alert, webhook_url: str):
        """Send webhook notification for alert."""
        try:
            if not self.webhook_config.get('enabled', False) or not REQUESTS_AVAILABLE:
                logger.info(f"Webhook would be sent for {alert.id}: {alert.message}")
                return
            
            payload = {
                'alert_id': alert.id,
                'system_id': alert.system_id,
                'severity': alert.severity.name,
                'category': alert.category,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            import requests
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=self.webhook_config.get('timeout', 30),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook sent successfully for {alert.id}")
            else:
                logger.warning(f"Webhook failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

class PredictiveAnalytics:
    """Advanced predictive analytics for maintenance scheduling."""
    
    def __init__(self):
        self.performance_history = {}
        self.fault_patterns = {}
    
    def analyze_degradation_trend(self, system_id: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze system performance degradation trends."""
        try:
            if len(historical_data) < 10:
                return {'trend': 'insufficient_data', 'prediction': None}
            
            # Calculate rolling efficiency trend
            historical_data['efficiency_ma'] = historical_data['efficiency'].rolling(window=5).mean()
            
            # Linear regression for trend analysis
            x = np.arange(len(historical_data))
            efficiency_trend = np.polyfit(x, historical_data['efficiency'].fillna(0), 1)
            
            # Predict future performance
            future_days = 30
            future_x = np.arange(len(historical_data), len(historical_data) + future_days)
            predicted_efficiency = np.polyval(efficiency_trend, future_x)
            
            # Determine trend severity
            slope = efficiency_trend[0]
            if slope < -0.01:  # >1% decline per period
                trend_status = 'severe_decline'
                urgency = 'high'
            elif slope < -0.005:  # >0.5% decline per period
                trend_status = 'moderate_decline'
                urgency = 'medium'
            elif slope < 0:
                trend_status = 'minor_decline'
                urgency = 'low'
            else:
                trend_status = 'stable_or_improving'
                urgency = 'none'
            
            return {
                'trend': trend_status,
                'slope': slope,
                'urgency': urgency,
                'predicted_efficiency_30d': predicted_efficiency[-1],
                'current_efficiency': historical_data['efficiency'].iloc[-1],
                'maintenance_recommendation': self._get_maintenance_recommendation(trend_status, urgency)
            }
            
        except Exception as e:
            logger.error(f"Error in degradation analysis: {e}")
            return {'trend': 'analysis_error', 'prediction': None}
    
    def _get_maintenance_recommendation(self, trend_status: str, urgency: str) -> Dict[str, Any]:
        """Generate maintenance recommendations based on trend analysis."""
        recommendations = {
            'severe_decline': {
                'action': 'immediate_inspection',
                'priority': 1,
                'tasks': ['string_analysis', 'inverter_diagnostics', 'thermal_inspection'],
                'schedule_within': timedelta(days=1)
            },
            'moderate_decline': {
                'action': 'scheduled_maintenance',
                'priority': 2,
                'tasks': ['performance_check', 'connection_inspection', 'cooling_system_check'],
                'schedule_within': timedelta(days=7)
            },
            'minor_decline': {
                'action': 'monitoring_increase',
                'priority': 3,
                'tasks': ['data_validation', 'trend_monitoring'],
                'schedule_within': timedelta(days=30)
            },
            'stable_or_improving': {
                'action': 'routine_maintenance',
                'priority': 4,
                'tasks': ['routine_check'],
                'schedule_within': timedelta(days=90)
            }
        }
        
        return recommendations.get(trend_status, recommendations['stable_or_improving'])

class AutomatedFleetScheduler:
    """Main automated fleet scheduling and monitoring system."""
    
    def __init__(self, config_path: str = "scheduler_config.json"):
        self.config = self._load_config(config_path)
        self.database = FleetDatabase()
        self.notification_manager = NotificationManager(self.config)
        self.predictive_analytics = PredictiveAnalytics()
        
        # System state
        self.system_statuses = {}
        self.alert_queue = queue.PriorityQueue()
        self.maintenance_queue = queue.PriorityQueue()
        self.running = False
        self.monitoring_thread = None
        self.scheduler_thread = None
        
        # Alert cooldowns to prevent spam
        self.alert_cooldowns = {}
        
        logger.info("Automated Fleet Scheduler initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'monitoring': {
                'interval_seconds': 300,  # 5 minutes
                'data_retention_days': 365,
                'alert_cooldown_minutes': 60
            },
            'thresholds': {
                'efficiency_min': 85.0,
                'temperature_max': 65.0,
                'health_score_min': 70.0,
                'string_power_variance_max': 20.0
            },
            'email': {
                'enabled': False,
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'use_tls': True,
                'recipients': []
            },
            'webhooks': {
                'enabled': False,
                'urls': [],
                'timeout': 30
            },
            'maintenance': {
                'auto_schedule': True,
                'default_duration_hours': 2,
                'business_hours_only': True,
                'advance_notice_days': 3
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config file, using defaults: {e}")
        
        return default_config
    
    def start_monitoring(self):
        """Start the automated monitoring and scheduling system."""
        if self.running:
            logger.warning("Monitoring system is already running")
            return
        
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Automated fleet monitoring and scheduling started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Automated fleet monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while self.running:
            try:
                self._perform_fleet_scan()
                time.sleep(self.config['monitoring']['interval_seconds'])
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _scheduler_loop(self):
        """Main scheduler loop for maintenance tasks."""
        logger.info("Starting scheduler loop")
        
        while self.running:
            try:
                self._process_maintenance_queue()
                self._schedule_predictive_maintenance()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _perform_fleet_scan(self):
        """Perform comprehensive fleet scan and analysis."""
        try:
            # Get list of cleaned inverter files
            csv_files = [f for f in os.listdir('.') if f.startswith('cleaned_INVERTER_') and f.endswith('.csv')]
            
            if not csv_files:
                logger.warning("No inverter data files found")
                return
            
            logger.info(f"Scanning {len(csv_files)} inverter systems")
            
            for csv_file in csv_files:
                try:
                    # Extract system ID from filename like "cleaned_INVERTER_01_2025-04-04_2025-04-05_20250606_065857.csv"
                    parts = csv_file.split('_')
                    if len(parts) >= 3:
                        system_id = f"{parts[1]}_{parts[2]}"  # INVERTER_01, INVERTER_02, etc.
                    else:
                        system_id = parts[1] if len(parts) >= 2 else csv_file
                    
                    self._analyze_system(system_id, csv_file)
                except Exception as e:
                    logger.error(f"Error analyzing {csv_file}: {e}")
            
            # Generate fleet-wide analysis
            self._generate_fleet_summary()
            
        except Exception as e:
            logger.error(f"Error in fleet scan: {e}")
    
    def _analyze_system(self, system_id: str, csv_file: str):
        """Analyze individual inverter system."""
        try:
            df = pd.read_csv(csv_file)
            
            if df.empty:
                return
            
            # Filter for active power generation (ignore standby/nighttime data)
            df_active = df[df['Pac(W)'] > 1000].copy()  # Only analyze when generating significant power
            
            if df_active.empty:
                logger.warning(f"No active power generation data found for {system_id}")
                return
            
            # Calculate efficiency if missing or empty
            if 'Conversion_Efficiency(%)' not in df_active.columns or df_active['Conversion_Efficiency(%)'].isna().all():
                # Calculate efficiency as (AC Power / DC Power) * 100
                df_active['Conversion_Efficiency(%)'] = (
                    df_active['Pac(W)'] / df_active['Total_String_Power(W)'] * 100
                ).fillna(0)
            
            # Calculate key metrics from active generation period
            latest_data = df_active.iloc[-1].to_dict() if len(df_active) > 0 else {}
            
            efficiency = latest_data.get('Conversion_Efficiency(%)', 0)
            # Handle NaN efficiency
            if pd.isna(efficiency):
                efficiency = 0
            
            # Use available temperature columns only
            temp_values = []
            for temp_col in ['INVTemp(℃)', 'AMTemp1(℃)', 'AMTemp2(℃)', 'BTTemp(℃)', 'OUTTemp(℃)']:
                if temp_col in latest_data:
                    temp_val = latest_data.get(temp_col, 0)
                    if not pd.isna(temp_val):
                        temp_values.append(temp_val)
            
            temperature = max(temp_values) if temp_values else 25.0
            
            power_output = latest_data.get('Pac(W)', 0)
            
            # Calculate health score
            health_score = self._calculate_health_score(latest_data, df_active)
            
            # Analyze string performance
            string_analysis = self._analyze_strings(latest_data)
            
            # Check for fault codes
            fault_codes = []
            for col in ['FaultCode', 'WarnCode', 'PidFaultCode']:
                if col in latest_data and pd.notna(latest_data[col]) and latest_data[col] != 0:
                    fault_codes.append(f"{col}:{latest_data[col]}")
            
            # Update system status
            status = SystemStatus(
                system_id=system_id,
                last_update=datetime.now(),
                health_score=health_score,
                efficiency=efficiency,
                power_output=power_output,
                temperature=temperature,
                fault_codes=fault_codes,
                string_status=string_analysis,
                performance_trend=self._calculate_trend(df_active),
                next_maintenance=self._get_next_maintenance_date(system_id)
            )
            
            self.system_statuses[system_id] = status
            
            # Check for alerts
            self._check_system_alerts(system_id, status, latest_data)
            
            # Perform predictive analysis
            self._perform_predictive_analysis(system_id, df_active)
            
        except Exception as e:
            logger.error(f"Error analyzing system {system_id}: {e}")
    
    def _calculate_health_score(self, latest_data: Dict[str, Any], df: pd.DataFrame) -> float:
        """Calculate comprehensive system health score."""
        try:
            score = 100.0
            
            # Efficiency factor (40% weight)
            efficiency = latest_data.get('Conversion_Efficiency(%)', 0)
            # Handle NaN or invalid efficiency values
            if pd.isna(efficiency) or efficiency <= 0:
                efficiency = 0
            if efficiency < 90:
                score -= (90 - efficiency) * 2
            
            # Temperature factor (20% weight)
            temp_values = []
            for temp_col in ['INVTemp(℃)', 'AMTemp1(℃)', 'AMTemp2(℃)', 'BTTemp(℃)', 'OUTTemp(℃)']:
                if temp_col in latest_data:
                    temp_val = latest_data.get(temp_col, 0)
                    if not pd.isna(temp_val):
                        temp_values.append(temp_val)
            
            max_temp = max(temp_values) if temp_values else 25.0
            if max_temp > 60:
                score -= (max_temp - 60) * 1.5
            
            # Fault codes factor (30% weight)
            fault_penalty = 0
            for col in ['FaultCode', 'WarnCode', 'PidFaultCode']:
                if col in latest_data and pd.notna(latest_data[col]) and latest_data[col] != 0:
                    fault_penalty += 10
            score -= fault_penalty
            
            # String performance factor (10% weight)
            string_scores = []
            for i in range(1, 33):
                power_col = f'Pstr{i}(W)'
                if power_col in latest_data and pd.notna(latest_data[power_col]):
                    string_scores.append(latest_data[power_col])
            
            if string_scores:
                avg_power = np.mean(string_scores)
                std_power = np.std(string_scores)
                if avg_power > 0:
                    cv = (std_power / avg_power) * 100
                    if cv > 20:  # High coefficient of variation
                        score -= (cv - 20) * 0.5
            
            return float(max(0, min(100, score)))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50.0  # Default neutral score
    
    def _analyze_strings(self, latest_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual string performance."""
        try:
            string_powers = []
            active_strings = 0
            
            for i in range(1, 33):
                power_col = f'Pstr{i}(W)'
                if power_col in latest_data and pd.notna(latest_data[power_col]):
                    power = latest_data[power_col]
                    string_powers.append(power)
                    if power > 10:  # Consider active if > 10W
                        active_strings += 1
            
            if not string_powers:
                return {'active_strings': 0, 'total_power': 0, 'analysis': 'no_data'}
            
            total_power = sum(string_powers)
            avg_power = np.mean(string_powers)
            std_power = np.std(string_powers)
            cv = (std_power / avg_power * 100) if avg_power > 0 else 0
            
            # Identify underperforming strings
            underperforming = []
            if avg_power > 0:
                threshold = avg_power * 0.8  # 20% below average
                for i, power in enumerate(string_powers, 1):
                    if power < threshold and power > 0:
                        underperforming.append(f"String{i}")
            
            return {
                'active_strings': active_strings,
                'total_strings': len(string_powers),
                'total_power': total_power,
                'average_power': avg_power,
                'power_std': std_power,
                'coefficient_variation': cv,
                'underperforming_strings': underperforming,
                'analysis': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in string analysis: {e}")
            return {'analysis': 'error'}
    
    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """Calculate performance trend over time."""
        try:
            if len(df) < 5:
                return 'insufficient_data'
            
            # Use last 10 points for trend
            recent_data = df.tail(10)
            
            if 'Conversion_Efficiency(%)' in recent_data.columns:
                efficiency_values = recent_data['Conversion_Efficiency(%)'].dropna()
                if len(efficiency_values) >= 3:
                    x = np.arange(len(efficiency_values))
                    slope = np.polyfit(x, efficiency_values, 1)[0]
                    
                    if slope > 0.1:
                        return 'improving'
                    elif slope < -0.1:
                        return 'declining'
                    else:
                        return 'stable'
            
            return 'stable'
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 'unknown'
    
    def _get_next_maintenance_date(self, system_id: str) -> Optional[datetime]:
        """Get next scheduled maintenance date for system."""
        try:
            tasks = self.database.get_pending_tasks()
            system_tasks = [t for t in tasks if t.system_id == system_id]
            
            if system_tasks:
                return min(task.scheduled_date for task in system_tasks)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next maintenance date: {e}")
            return None
    
    def _check_system_alerts(self, system_id: str, status: SystemStatus, latest_data: Dict):
        """Check system for alert conditions."""
        try:
            thresholds = self.config['thresholds']
            alerts_generated = []
            
            # Efficiency alert
            if status.efficiency < thresholds['efficiency_min']:
                alert = self._create_alert(
                    system_id, AlertSeverity.HIGH, 'efficiency',
                    f"Low efficiency: {status.efficiency:.1f}%",
                    {'efficiency': status.efficiency, 'threshold': thresholds['efficiency_min']}
                )
                alerts_generated.append(alert)
            
            # Temperature alert
            if status.temperature > thresholds['temperature_max']:
                alert = self._create_alert(
                    system_id, AlertSeverity.HIGH, 'temperature',
                    f"High temperature: {status.temperature:.1f}°C",
                    {'temperature': status.temperature, 'threshold': thresholds['temperature_max']}
                )
                alerts_generated.append(alert)
            
            # Health score alert
            if status.health_score < thresholds['health_score_min']:
                severity = AlertSeverity.CRITICAL if status.health_score < 50 else AlertSeverity.HIGH
                alert = self._create_alert(
                    system_id, severity, 'health',
                    f"Low health score: {status.health_score:.1f}",
                    {'health_score': status.health_score, 'threshold': thresholds['health_score_min']}
                )
                alerts_generated.append(alert)
            
            # Fault code alerts
            if status.fault_codes:
                alert = self._create_alert(
                    system_id, AlertSeverity.CRITICAL, 'fault',
                    f"Fault codes detected: {', '.join(status.fault_codes)}",
                    {'fault_codes': status.fault_codes}
                )
                alerts_generated.append(alert)
            
            # String performance alerts
            string_status = status.string_status
            if isinstance(string_status, dict) and 'coefficient_variation' in string_status:
                cv = string_status['coefficient_variation']
                if cv > thresholds['string_power_variance_max']:
                    alert = self._create_alert(
                        system_id, AlertSeverity.MEDIUM, 'string_imbalance',
                        f"High string power variance: {cv:.1f}%",
                        {
                            'coefficient_variation': cv,
                            'threshold': thresholds['string_power_variance_max'],
                            'underperforming_strings': string_status.get('underperforming_strings', [])
                        }
                    )
                    alerts_generated.append(alert)
            
            # Process and queue alerts
            for alert in alerts_generated:
                if self._should_send_alert(alert):
                    self._queue_alert(alert)
            
        except Exception as e:
            logger.error(f"Error checking alerts for {system_id}: {e}")
    
    def _create_alert(self, system_id: str, severity: AlertSeverity, category: str, 
                     message: str, data: Dict[str, Any]) -> Alert:
        """Create new alert object."""
        alert_id = f"{system_id}_{category}_{int(datetime.now().timestamp())}"
        
        return Alert(
            id=alert_id,
            timestamp=datetime.now(),
            system_id=system_id,
            severity=severity,
            category=category,
            message=message,
            data=data
        )
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent (respecting cooldowns)."""
        cooldown_key = f"{alert.system_id}_{alert.category}"
        cooldown_minutes = self.config['monitoring']['alert_cooldown_minutes']
        
        if cooldown_key in self.alert_cooldowns:
            last_sent = self.alert_cooldowns[cooldown_key]
            if datetime.now() - last_sent < timedelta(minutes=cooldown_minutes):
                return False
        
        return True
    
    def _queue_alert(self, alert: Alert):
        """Queue alert for processing."""
        try:
            # Add cooldown
            cooldown_key = f"{alert.system_id}_{alert.category}"
            self.alert_cooldowns[cooldown_key] = datetime.now()
            
            # Save to database
            self.database.save_alert(alert)
            
            # Send notifications
            if self.config['email']['enabled'] and self.config['email']['recipients']:
                self.notification_manager.send_email_alert(alert, self.config['email']['recipients'])
            
            if self.config['webhooks']['enabled'] and self.config['webhooks']['urls']:
                for url in self.config['webhooks']['urls']:
                    self.notification_manager.send_webhook(alert, url)
            
            logger.info(f"Alert queued and processed: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error queuing alert: {e}")
    
    def _perform_predictive_analysis(self, system_id: str, df: pd.DataFrame):
        """Perform predictive analysis for maintenance scheduling."""
        try:
            if len(df) < 10:
                return
            
            # Prepare historical data
            historical_data = df.copy()
            historical_data['efficiency'] = historical_data.get('Conversion_Efficiency(%)', 0)
            historical_data['power'] = historical_data.get('Pac(W)', 0)
            # Use available temperature columns only
            temp_cols = [col for col in ['INVTemp(℃)', 'AMTemp1(℃)', 'AMTemp2(℃)', 'BTTemp(℃)', 'OUTTemp(℃)'] if col in historical_data.columns]
            if temp_cols:
                historical_data['temperature'] = historical_data[temp_cols].max(axis=1)
            else:
                historical_data['temperature'] = 25.0  # Default temperature
            
            # Analyze degradation trend
            trend_analysis = self.predictive_analytics.analyze_degradation_trend(system_id, historical_data)
            
            # Schedule maintenance based on prediction
            if trend_analysis['urgency'] in ['high', 'medium']:
                self._schedule_predictive_maintenance_task(system_id, trend_analysis)
            
        except Exception as e:
            logger.error(f"Error in predictive analysis for {system_id}: {e}")
    
    def _schedule_predictive_maintenance_task(self, system_id: str, trend_analysis: Dict[str, Any]):
        """Schedule maintenance task based on predictive analysis."""
        try:
            recommendation = trend_analysis.get('maintenance_recommendation', {})
            
            if not recommendation:
                return
            
            # Check if similar task already exists
            existing_tasks = self.database.get_pending_tasks()
            for task in existing_tasks:
                if (task.system_id == system_id and 
                    task.task_type == MaintenanceType.PREDICTIVE and
                    task.status == TaskStatus.PENDING):
                    return  # Task already exists
            
            # Create maintenance task
            task_id = f"pred_{system_id}_{int(datetime.now().timestamp())}"
            scheduled_date = datetime.now() + recommendation.get('schedule_within', timedelta(days=7))
            
            task = MaintenanceTask(
                id=task_id,
                system_id=system_id,
                task_type=MaintenanceType.PREDICTIVE,
                priority=recommendation.get('priority', 3),
                description=f"Predictive maintenance - {recommendation.get('action', 'inspection')}",
                scheduled_date=scheduled_date,
                estimated_duration=timedelta(hours=self.config['maintenance']['default_duration_hours']),
                status=TaskStatus.PENDING
            )
            
            # Add specific tasks from recommendation
            task.notes.extend(recommendation.get('tasks', []))
            task.notes.append(f"Trend analysis: {trend_analysis.get('trend', 'unknown')}")
            task.notes.append(f"Urgency: {trend_analysis.get('urgency', 'unknown')}")
            
            self.database.save_maintenance_task(task)
            logger.info(f"Predictive maintenance task scheduled for {system_id}: {task_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling predictive maintenance: {e}")
    
    def _process_maintenance_queue(self):
        """Process pending maintenance tasks."""
        try:
            pending_tasks = self.database.get_pending_tasks()
            current_time = datetime.now()
            
            for task in pending_tasks:
                # Check if task is due
                if task.scheduled_date <= current_time:
                    logger.info(f"Maintenance task due: {task.id} for system {task.system_id}")
                    # Here you would integrate with your maintenance management system
                    # For now, we'll just log the task
                    
                # Check if task needs advance notification
                advance_notice = timedelta(days=self.config['maintenance']['advance_notice_days'])
                if task.scheduled_date - current_time <= advance_notice:
                    self._send_maintenance_notification(task)
            
        except Exception as e:
            logger.error(f"Error processing maintenance queue: {e}")
    
    def _send_maintenance_notification(self, task: MaintenanceTask):
        """Send notification for upcoming maintenance."""
        try:
            # Create alert for maintenance notification
            alert = Alert(
                id=f"maint_notify_{task.id}",
                timestamp=datetime.now(),
                system_id=task.system_id,
                severity=AlertSeverity.INFO,
                category='maintenance_notification',
                message=f"Scheduled maintenance: {task.description}",
                data={
                    'task_id': task.id,
                    'scheduled_date': task.scheduled_date.isoformat(),
                    'task_type': task.task_type.value,
                    'priority': task.priority,
                    'estimated_duration': str(task.estimated_duration)
                }
            )
            
            # Send notification (respecting cooldowns)
            if self._should_send_alert(alert):
                self._queue_alert(alert)
            
        except Exception as e:
            logger.error(f"Error sending maintenance notification: {e}")
    
    def _schedule_predictive_maintenance(self):
        """Schedule predictive maintenance based on overall fleet analysis."""
        try:
            # Analyze fleet-wide patterns
            if len(self.system_statuses) < 2:
                return
            
            # Identify systems that need attention
            health_scores = [(sid, status.health_score) for sid, status in self.system_statuses.items()]
            health_scores.sort(key=lambda x: x[1])  # Sort by health score
            
            # Schedule maintenance for bottom 20% performers
            bottom_count = max(1, len(health_scores) // 5)
            for system_id, health_score in health_scores[:bottom_count]:
                if health_score < 70:  # Below threshold
                    self._schedule_routine_maintenance(system_id, f"Low health score: {health_score:.1f}")
            
        except Exception as e:
            logger.error(f"Error in predictive maintenance scheduling: {e}")
    
    def _schedule_routine_maintenance(self, system_id: str, reason: str):
        """Schedule routine maintenance task."""
        try:
            # Check if task already exists
            existing_tasks = self.database.get_pending_tasks()
            for task in existing_tasks:
                if (task.system_id == system_id and 
                    task.task_type == MaintenanceType.ROUTINE and
                    task.status == TaskStatus.PENDING):
                    return
            
            task_id = f"routine_{system_id}_{int(datetime.now().timestamp())}"
            scheduled_date = datetime.now() + timedelta(days=7)  # Schedule in 1 week
            
            task = MaintenanceTask(
                id=task_id,
                system_id=system_id,
                task_type=MaintenanceType.ROUTINE,
                priority=4,
                description=f"Routine maintenance - {reason}",
                scheduled_date=scheduled_date,
                estimated_duration=timedelta(hours=2),
                status=TaskStatus.PENDING
            )
            
            self.database.save_maintenance_task(task)
            logger.info(f"Routine maintenance scheduled for {system_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling routine maintenance: {e}")
    
    def _generate_fleet_summary(self):
        """Generate comprehensive fleet summary."""
        try:
            if not self.system_statuses:
                return
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_systems': len(self.system_statuses),
                'fleet_metrics': {
                    'average_health_score': np.mean([s.health_score for s in self.system_statuses.values()]),
                    'average_efficiency': np.mean([s.efficiency for s in self.system_statuses.values()]),
                    'total_power_output': sum([s.power_output for s in self.system_statuses.values()]),
                    'systems_with_faults': len([s for s in self.system_statuses.values() if s.fault_codes]),
                    'systems_needing_attention': len([s for s in self.system_statuses.values() if s.health_score < 70])
                },
                'system_statuses': {sid: {
                    'health_score': status.health_score,
                    'efficiency': status.efficiency,
                    'temperature': status.temperature,
                    'power_output': status.power_output,
                    'fault_codes': status.fault_codes,
                    'performance_trend': status.performance_trend
                } for sid, status in self.system_statuses.items()}
            }
            
            # Save fleet summary
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = f'fleet_summary_{timestamp}.json'
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Fleet summary generated: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating fleet summary: {e}")
    
    def get_system_status(self, system_id: str) -> Optional[SystemStatus]:
        """Get current status of a specific system."""
        return self.system_statuses.get(system_id)
    
    def get_fleet_health(self) -> Dict[str, Any]:
        """Get overall fleet health metrics."""
        if not self.system_statuses:
            return {'status': 'no_data'}
        
        health_scores = [s.health_score for s in self.system_statuses.values()]
        # Filter out NaN and zero efficiency values
        efficiencies = [s.efficiency for s in self.system_statuses.values() 
                       if not pd.isna(s.efficiency) and s.efficiency > 0]
        
        return {
            'total_systems': len(self.system_statuses),
            'average_health_score': np.mean(health_scores) if health_scores else 0,
            'health_score_std': np.std(health_scores) if health_scores else 0,
            'average_efficiency': np.mean(efficiencies) if efficiencies else 0,
            'efficiency_std': np.std(efficiencies) if efficiencies else 0,
            'systems_with_faults': len([s for s in self.system_statuses.values() if s.fault_codes]),
            'systems_critical': len([s for s in self.system_statuses.values() if s.health_score < 50]),
            'systems_warning': len([s for s in self.system_statuses.values() if 50 <= s.health_score < 70]),
            'systems_healthy': len([s for s in self.system_statuses.values() if s.health_score >= 70])
        }

def create_sample_config():
    """Create sample configuration file."""
    config = {
        "monitoring": {
            "interval_seconds": 300,
            "data_retention_days": 365,
            "alert_cooldown_minutes": 60
        },
        "thresholds": {
            "efficiency_min": 85.0,
            "temperature_max": 65.0,
            "health_score_min": 70.0,
            "string_power_variance_max": 20.0
        },
        "email": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "use_tls": True,
            "username": "your_email@gmail.com",
            "password": "your_app_password",
            "smtp_from": "fleet_monitor@yourcompany.com",
            "recipients": ["technician@yourcompany.com", "manager@yourcompany.com"]
        },
        "webhooks": {
            "enabled": False,
            "urls": ["https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"],
            "timeout": 30
        },
        "maintenance": {
            "auto_schedule": True,
            "default_duration_hours": 2,
            "business_hours_only": True,
            "advance_notice_days": 3
        }
    }
    
    with open('scheduler_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: scheduler_config.json")

def main():
    """Main function for testing the automated fleet scheduler."""
    print("🚀 Automated Fleet Scheduler and Real-Time Monitoring System")
    print("=" * 70)
    
    # Create sample config if it doesn't exist
    if not os.path.exists('scheduler_config.json'):
        create_sample_config()
    
    # Initialize scheduler
    scheduler = AutomatedFleetScheduler()
    
    try:
        # Start monitoring
        scheduler.start_monitoring()
        
        # Run for a test period
        print("🔄 Fleet monitoring started. Running for 60 seconds...")
        print("📊 Check fleet_summary_*.json and fleet_scheduler.log for results")
        
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 60:
            # Display real-time fleet health
            fleet_health = scheduler.get_fleet_health()
            if fleet_health.get('total_systems', 0) > 0:
                print(f"\r⚡ Fleet Status: {fleet_health['total_systems']} systems, "
                      f"Avg Health: {fleet_health['average_health_score']:.1f}, "
                      f"Avg Efficiency: {fleet_health['average_efficiency']:.1f}%", end="")
            
            time.sleep(5)
        
        print("\n\n✅ Test monitoring completed")
        
        # Display final fleet health
        final_health = scheduler.get_fleet_health()
        if final_health.get('total_systems', 0) > 0:
            print(f"\n📈 Final Fleet Health Report:")
            print(f"   • Total Systems: {final_health['total_systems']}")
            print(f"   • Average Health Score: {final_health['average_health_score']:.1f} ± {final_health['health_score_std']:.1f}")
            print(f"   • Average Efficiency: {final_health['average_efficiency']:.1f}% ± {final_health['efficiency_std']:.1f}%")
            print(f"   • Systems with Faults: {final_health['systems_with_faults']}")
            print(f"   • Critical Systems: {final_health['systems_critical']}")
            print(f"   • Warning Systems: {final_health['systems_warning']}")
            print(f"   • Healthy Systems: {final_health['systems_healthy']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error in monitoring: {e}")
    finally:
        scheduler.stop_monitoring()
        print("🔚 Automated fleet scheduler stopped")

if __name__ == "__main__":
    main()