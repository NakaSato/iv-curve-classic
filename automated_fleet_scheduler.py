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