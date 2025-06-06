#!/usr/bin/env python3
"""
Advanced Automated Fleet Scheduler and Monitoring System
Real-time fleet monitoring with predictive maintenance and automated alerting
"""

import pandas as pd
import numpy as np
import json
import logging
import smtplib
import requests
import threading
import time
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_fleet_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MaintenanceTask:
    """Maintenance task data structure"""
    system_id: str
    task_type: str
    priority: str  # HIGH, MEDIUM, LOW
    urgency_score: float
    description: str
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    assigned_to: Optional[str] = None
    status: str = "PENDING"  # PENDING, SCHEDULED, IN_PROGRESS, COMPLETED
    estimated_duration: Optional[int] = None  # minutes

@dataclass
class Alert:
    """Alert data structure"""
    system_id: str
    alert_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    acknowledged: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SystemHealth:
    """System health metrics"""
    system_id: str
    efficiency: float
    temperature: float
    power_output: float
    string_health: Dict[str, float]
    fault_codes: List[str]
    health_score: float
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class NotificationManager:
    """Manages email and webhook notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_config = config.get('email', {})
        self.webhook_config = config.get('webhooks', {})
        self.notification_history = []
        
    def send_email_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send email alert"""
        try:
            if not self.email_config.get('enabled', False):
                logger.info("Email notifications disabled")
                return False
                
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            
            if not all([smtp_server, username, password]):
                logger.warning("Email configuration incomplete")
                return False
            
            msg = MimeMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Fleet Alert - {alert.severity}: {alert.system_id}"
            
            body = f"""
Fleet Monitoring Alert

System: {alert.system_id}
Alert Type: {alert.alert_type}
Severity: {alert.severity}
Message: {alert.message}
Timestamp: {alert.timestamp}

Value: {alert.value}
Threshold: {alert.threshold}

Please investigate immediately.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.system_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_webhook_alert(self, alert: Alert, webhook_url: str) -> bool:
        """Send webhook alert"""
        try:
            if not self.webhook_config.get('enabled', False):
                return False
                
            payload = {
                'system_id': alert.system_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp.isoformat()
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for {alert.system_id}")
                return True
            else:
                logger.warning(f"Webhook returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def log_notification(self, alert: Alert, method: str, success: bool):
        """Log notification attempt"""
        self.notification_history.append({
            'alert': asdict(alert),
            'method': method,
            'success': success,
            'sent_at': datetime.now().isoformat()
        })

class PredictiveMaintenanceEngine:
    """Predictive maintenance using ML models"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.historical_data = []
        
    def add_historical_data(self, system_data: Dict[str, Any]):
        """Add data point for model training"""
        self.historical_data.append({
            'timestamp': datetime.now(),
            'system_id': system_data.get('system_id'),
            'efficiency': system_data.get('efficiency', 0),
            'temperature': system_data.get('temperature', 0),
            'power_output': system_data.get('power_output', 0),
            'fault_count': len(system_data.get('fault_codes', [])),
            'string_avg_power': np.mean(list(system_data.get('string_powers', {}).values()))
        })
    
    def train_models(self) -> bool:
        """Train predictive models"""
        try:
            if len(self.historical_data) < 50:
                logger.warning("Insufficient data for model training")
                return False
            
            df = pd.DataFrame(self.historical_data)
            
            # Prepare features for anomaly detection
            features = ['efficiency', 'temperature', 'power_output', 'fault_count', 'string_avg_power']
            X = df[features].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X_scaled)
            
            # Train performance predictor
            y = df['efficiency'].values.astype(float)
            self.performance_predictor.fit(X_scaled, y)
            
            self.is_trained = True
            logger.info("Predictive models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def predict_anomaly(self, system_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Predict if system behavior is anomalous"""
        try:
            if not self.is_trained:
                return False, 0.0
            
            features = [
                system_data.get('efficiency', 0),
                system_data.get('temperature', 0),
                system_data.get('power_output', 0),
                len(system_data.get('fault_codes', [])),
                np.mean(list(system_data.get('string_powers', {}).values())) if system_data.get('string_powers') else 0
            ]
            
            X_scaled = self.scaler.transform([features])
            anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
            is_outlier = self.anomaly_detector.predict(X_scaled)[0] == -1
            
            return is_outlier, abs(anomaly_score)
            
        except Exception as e:
            logger.error(f"Anomaly prediction failed: {e}")
            return False, 0.0
    
    def predict_performance(self, system_data: Dict[str, Any]) -> float:
        """Predict future performance"""
        try:
            if not self.is_trained:
                return system_data.get('efficiency', 0)
            
            features = [
                system_data.get('efficiency', 0),
                system_data.get('temperature', 0),
                system_data.get('power_output', 0),
                len(system_data.get('fault_codes', [])),
                np.mean(list(system_data.get('string_powers', {}).values())) if system_data.get('string_powers') else 0
            ]
            
            X_scaled = self.scaler.transform([features])
            predicted_efficiency = self.performance_predictor.predict(X_scaled)[0]
            
            return max(0, min(100, predicted_efficiency))  # Clamp to reasonable range
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return system_data.get('efficiency', 0)

class AutomatedFleetScheduler:
    """Main automated fleet scheduler and monitoring system"""
    
    def __init__(self, config_path: str = "fleet_scheduler_config.json"):
        self.config = self.load_config(config_path)
        self.notification_manager = NotificationManager(self.config)
        self.predictive_engine = PredictiveMaintenanceEngine()
        
        # Fleet state
        self.fleet_systems = {}
        self.maintenance_queue = []
        self.active_alerts = []
        self.alert_history = []
        self.system_health_history = {}
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Alert cooldown tracking
        self.alert_cooldowns = {}
        
        logger.info("Automated Fleet Scheduler initialized")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load scheduler configuration"""
        default_config = {
            "monitoring": {
                "interval_seconds": 300,  # 5 minutes
                "data_retention_days": 30,
                "enable_predictive_maintenance": True
            },
            "thresholds": {
                "efficiency_min": 85.0,
                "efficiency_critical": 80.0,
                "temperature_max": 65.0,
                "temperature_critical": 70.0,
                "power_deviation_max": 15.0,
                "string_imbalance_max": 20.0
            },
            "maintenance": {
                "auto_schedule": True,
                "max_concurrent_tasks": 5,
                "working_hours_start": 8,
                "working_hours_end": 17
            },
            "alerts": {
                "cooldown_minutes": 60,
                "max_alerts_per_hour": 10,
                "severity_escalation_minutes": 30
            },
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "webhooks": {
                "enabled": False,
                "urls": []
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
        
        return default_config
    
    def discover_fleet_systems(self, data_directory: str = ".") -> List[str]:
        """Discover available fleet systems from cleaned CSV files"""
        try:
            data_path = Path(data_directory)
            csv_files = list(data_path.glob("cleaned_INVERTER_*.csv"))
            
            system_ids = set()
            for file_path in csv_files:
                # Extract system ID from filename
                filename = file_path.name
                if "cleaned_INVERTER_" in filename:
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        system_id = f"INVERTER_{parts[1]}"
                        system_ids.add(system_id)
            
            systems = sorted(list(system_ids))
            logger.info(f"Discovered {len(systems)} fleet systems: {systems}")
            return systems
            
        except Exception as e:
            logger.error(f"Fleet discovery failed: {e}")
            return []
    
    def load_system_data(self, system_id: str, data_directory: str = ".") -> Optional[pd.DataFrame]:
        """Load latest data for a system"""
        try:
            data_path = Path(data_directory)
            # Find the most recent file for this system
            pattern = f"cleaned_{system_id}_*.csv"
            files = list(data_path.glob(pattern))
            
            if not files:
                logger.warning(f"No data files found for {system_id}")
                return None
            
            # Sort by modification time, get the most recent
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            
            df = pd.read_csv(latest_file)
            logger.debug(f"Loaded {len(df)} records for {system_id}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for {system_id}: {e}")
            return None
    
    def analyze_system_health(self, system_id: str, df: pd.DataFrame) -> SystemHealth:
        """Analyze current system health"""
        try:
            # Get latest record
            latest = df.iloc[-1] if len(df) > 0 else pd.Series()
            
            # Calculate efficiency
            efficiency = latest.get('Conversion_Efficiency(%)', 0)
            
            # Get temperature
            temperature = latest.get('INVTemp(℃)', 0)
            
            # Calculate total power output
            total_power = latest.get('Pac(W)', 0)
            
            # Analyze string health
            string_health = {}
            string_powers = {}
            for i in range(1, 33):  # 32 strings
                col_name = f'Pstr{i}(W)'
                if col_name in df.columns:
                    power = latest.get(col_name, 0)
                    string_powers[f'string_{i}'] = power
                    # Health based on power relative to average
                    avg_power = df[col_name].mean() if col_name in df.columns else 1
                    health = min(100, (power / avg_power * 100)) if avg_power > 0 else 0
                    string_health[f'string_{i}'] = health
            
            # Check fault codes
            fault_codes = []
            for fault_col in ['FaultCode', 'WarnCode', 'PidFaultCode']:
                if fault_col in df.columns and latest.get(fault_col, 0) != 0:
                    fault_codes.append(f"{fault_col}:{latest.get(fault_col)}")
            
            # Calculate overall health score
            health_score = self.calculate_health_score(
                efficiency, temperature, len(fault_codes), string_health
            )
            
            health = SystemHealth(
                system_id=system_id,
                efficiency=efficiency,
                temperature=temperature,
                power_output=total_power,
                string_health=string_health,
                fault_codes=fault_codes,
                health_score=health_score
            )
            
            # Add to predictive engine
            system_data = {
                'system_id': system_id,
                'efficiency': efficiency,
                'temperature': temperature,
                'power_output': total_power,
                'fault_codes': fault_codes,
                'string_powers': string_powers
            }
            self.predictive_engine.add_historical_data(system_data)
            
            return health
            
        except Exception as e:
            logger.error(f"Health analysis failed for {system_id}: {e}")
            return SystemHealth(
                system_id=system_id, efficiency=0, temperature=0,
                power_output=0, string_health={}, fault_codes=[],
                health_score=0
            )
    
    def calculate_health_score(self, efficiency: float, temperature: float, 
                             fault_count: int, string_health: Dict[str, float]) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            # Efficiency component (40% weight)
            eff_score = min(100, efficiency * 1.15) if efficiency > 0 else 0
            
            # Temperature component (20% weight)
            temp_threshold = self.config['thresholds']['temperature_max']
            temp_score = max(0, 100 - (temperature - 25) * 2) if temperature > 25 else 100
            
            # Fault component (20% weight)
            fault_score = max(0, 100 - fault_count * 20)
            
            # String balance component (20% weight)
            string_scores = list(string_health.values())
            string_score = np.mean(string_scores) if string_scores else 0
            
            # Weighted average
            health_score = (
                eff_score * 0.4 +
                temp_score * 0.2 +
                fault_score * 0.2 +
                string_score * 0.2
            )
            
            return float(max(0, min(100, health_score)))
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0
    
    def check_alert_conditions(self, health: SystemHealth) -> List[Alert]:
        """Check for alert conditions"""
        alerts = []
        thresholds = self.config['thresholds']
        
        try:
            # Efficiency alerts
            if health.efficiency < thresholds['efficiency_critical']:
                alerts.append(Alert(
                    system_id=health.system_id,
                    alert_type="EFFICIENCY_CRITICAL",
                    severity="CRITICAL",
                    message=f"Efficiency critically low: {health.efficiency:.1f}%",
                    value=health.efficiency,
                    threshold=thresholds['efficiency_critical']
                ))
            elif health.efficiency < thresholds['efficiency_min']:
                alerts.append(Alert(
                    system_id=health.system_id,
                    alert_type="EFFICIENCY_LOW",
                    severity="HIGH",
                    message=f"Efficiency below threshold: {health.efficiency:.1f}%",
                    value=health.efficiency,
                    threshold=thresholds['efficiency_min']
                ))
            
            # Temperature alerts
            if health.temperature > thresholds['temperature_critical']:
                alerts.append(Alert(
                    system_id=health.system_id,
                    alert_type="TEMPERATURE_CRITICAL",
                    severity="CRITICAL",
                    message=f"Temperature critically high: {health.temperature:.1f}°C",
                    value=health.temperature,
                    threshold=thresholds['temperature_critical']
                ))
            elif health.temperature > thresholds['temperature_max']:
                alerts.append(Alert(
                    system_id=health.system_id,
                    alert_type="TEMPERATURE_HIGH",
                    severity="HIGH",
                    message=f"Temperature above threshold: {health.temperature:.1f}°C",
                    value=health.temperature,
                    threshold=thresholds['temperature_max']
                ))
            
            # Fault code alerts
            if health.fault_codes:
                alerts.append(Alert(
                    system_id=health.system_id,
                    alert_type="FAULT_DETECTED",
                    severity="HIGH",
                    message=f"Fault codes detected: {', '.join(health.fault_codes)}",
                    value=len(health.fault_codes)
                ))
            
            # String imbalance alerts
            if health.string_health:
                string_powers = list(health.string_health.values())
                if len(string_powers) > 1:
                    avg_power = np.mean(string_powers)
                    max_deviation = max(abs(p - avg_power) for p in string_powers)
                    deviation_pct = (max_deviation / avg_power * 100) if avg_power > 0 else 0
                    
                    if deviation_pct > thresholds['string_imbalance_max']:
                        alerts.append(Alert(
                            system_id=health.system_id,
                            alert_type="STRING_IMBALANCE",
                            severity="MEDIUM",
                            message=f"String power imbalance: {deviation_pct:.1f}%",
                            value=float(deviation_pct),
                            threshold=thresholds['string_imbalance_max']
                        ))
            
            # Health score alerts
            if health.health_score < 70:
                severity = "CRITICAL" if health.health_score < 50 else "HIGH"
                alerts.append(Alert(
                    system_id=health.system_id,
                    alert_type="HEALTH_SCORE_LOW",
                    severity=severity,
                    message=f"System health score low: {health.health_score:.1f}",
                    value=health.health_score,
                    threshold=70
                ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert condition check failed for {health.system_id}: {e}")
            return []
    
    def should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent (respecting cooldowns)"""
        alert_key = f"{alert.system_id}_{alert.alert_type}"
        cooldown_minutes = self.config['alerts']['cooldown_minutes']
        
        if alert_key in self.alert_cooldowns:
            last_sent = self.alert_cooldowns[alert_key]
            if datetime.now() - last_sent < timedelta(minutes=cooldown_minutes):
                return False
        
        return True