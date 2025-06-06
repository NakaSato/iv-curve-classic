#!/usr/bin/env python3
"""
Real-Time Inverter Fleet Monitoring and Predictive Maintenance System
=======================================================================

Advanced monitoring system with ML-based predictive analytics, real-time alerting,
and automated maintenance scheduling for photovoltaic inverter fleets.

Features:
- Real-time performance monitoring
- ML-based fault prediction
- Automated alert generation
- Maintenance scheduling
- Performance trend analysis
- Anomaly detection
- Fleet optimization recommendations

Author: Advanced PV Analytics System
Date: June 6, 2025
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class RealTimeMonitoringSystem:
    """Advanced real-time monitoring system for PV inverter fleets."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the monitoring system."""
        self.setup_logging()
        
        # Default configuration
        self.config = {
            'alert_thresholds': {
                'efficiency_drop': 5.0,  # % drop from baseline
                'temperature_high': 85.0,  # °C
                'voltage_deviation': 10.0,  # % from nominal
                'current_imbalance': 15.0,  # % between strings
                'power_drop': 10.0,  # % from expected
                'data_quality_min': 95.0  # % minimum quality
            },
            'maintenance_intervals': {
                'routine_inspection': 30,  # days
                'detailed_analysis': 90,  # days
                'deep_maintenance': 365,  # days
            },
            'prediction_window': 7,  # days ahead
            'monitoring_frequency': 300,  # seconds (5 minutes)
            'data_retention': 365,  # days
        }
        
        if config:
            self.config.update(config)
        
        self.fleet_data = {}
        self.historical_data = []
        self.alert_history = []
        self.maintenance_schedule = []
        self.performance_baselines = {}
        
        # ML models
        self.anomaly_detector = None
        self.performance_predictor = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
        self.logger.info("Real-time monitoring system initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('monitoring_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_fleet_data(self, data_directory: str) -> Dict[str, Any]:
        """Load and process fleet data from cleaned CSV files."""
        self.logger.info(f"Loading fleet data from {data_directory}")
        
        fleet_summary = {
            'systems_loaded': 0,
            'total_data_points': 0,
            'date_range': {'start': None, 'end': None},
            'systems': {}
        }
        
        # Find cleaned CSV files
        csv_files = [f for f in os.listdir(data_directory) if f.startswith('cleaned_INVERTER_') and f.endswith('.csv')]
        csv_files.sort()
        
        for csv_file in csv_files:
            try:
                file_path = os.path.join(data_directory, csv_file)
                df = pd.read_csv(file_path)
                
                # Extract inverter ID
                inverter_id = csv_file.split('_')[1]
                
                # Process timestamp
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    df = df.sort_values('Timestamp')
                
                # Store system data
                self.fleet_data[inverter_id] = {
                    'data': df,
                    'last_update': datetime.now(),
                    'data_points': len(df),
                    'date_range': {
                        'start': df['Timestamp'].min() if 'Timestamp' in df.columns else None,
                        'end': df['Timestamp'].max() if 'Timestamp' in df.columns else None
                    }
                }
                
                fleet_summary['systems'][inverter_id] = {
                    'data_points': len(df),
                    'file': csv_file
                }
                
                fleet_summary['systems_loaded'] += 1
                fleet_summary['total_data_points'] += len(df)
                
                # Update fleet date range
                if 'Timestamp' in df.columns:
                    start_date = df['Timestamp'].min()
                    end_date = df['Timestamp'].max()
                    
                    if fleet_summary['date_range']['start'] is None or start_date < fleet_summary['date_range']['start']:
                        fleet_summary['date_range']['start'] = start_date
                    if fleet_summary['date_range']['end'] is None or end_date > fleet_summary['date_range']['end']:
                        fleet_summary['date_range']['end'] = end_date
                
                self.logger.info(f"Loaded {inverter_id}: {len(df)} data points")
                
            except Exception as e:
                self.logger.error(f"Error loading {csv_file}: {str(e)}")
                continue
        
        # Establish performance baselines
        self._establish_baselines()
        
        self.logger.info(f"Fleet data loaded: {fleet_summary['systems_loaded']} systems, {fleet_summary['total_data_points']} total data points")
        return fleet_summary
    
    def _establish_baselines(self):
        """Establish performance baselines for each system."""
        self.logger.info("Establishing performance baselines")
        
        for inverter_id, system_data in self.fleet_data.items():
            df = system_data['data']
            
            # Calculate baselines for key parameters
            baseline = {}
            
            # Efficiency baseline (using median for robustness)
            if 'Efficiency' in df.columns:
                baseline['efficiency'] = {
                    'median': df['Efficiency'].median(),
                    'mean': df['Efficiency'].mean(),
                    'std': df['Efficiency'].std(),
                    'q25': df['Efficiency'].quantile(0.25),
                    'q75': df['Efficiency'].quantile(0.75)
                }
            
            # Temperature baselines
            temp_cols = [col for col in df.columns if 'Temp' in col or 'temp' in col]
            for col in temp_cols:
                baseline[f'temperature_{col}'] = {
                    'median': df[col].median(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'max_normal': df[col].quantile(0.95)
                }
            
            # Power baseline
            if 'DC_Power' in df.columns:
                baseline['power'] = {
                    'median': df['DC_Power'].median(),
                    'mean': df['DC_Power'].mean(),
                    'std': df['DC_Power'].std(),
                    'peak': df['DC_Power'].quantile(0.95)
                }
            
            # String voltage baselines
            voltage_cols = [col for col in df.columns if col.startswith('Vstr')]
            for col in voltage_cols:
                baseline[f'voltage_{col}'] = {
                    'median': df[col].median(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'range': [df[col].quantile(0.05), df[col].quantile(0.95)]
                }
            
            # String current baselines
            current_cols = [col for col in df.columns if col.startswith('Istr')]
            for col in current_cols:
                baseline[f'current_{col}'] = {
                    'median': df[col].median(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'range': [df[col].quantile(0.05), df[col].quantile(0.95)]
                }
            
            self.performance_baselines[inverter_id] = baseline
        
        self.logger.info(f"Baselines established for {len(self.performance_baselines)} systems")
    
    def _get_fleet_date_range(self) -> Optional[Dict[str, Any]]:
        """Get the date range for the entire fleet."""
        if not self.fleet_data:
            return None
        
        start_dates = []
        end_dates = []
        
        for system in self.fleet_data.values():
            if system['date_range']['start'] is not None:
                start_dates.append(system['date_range']['start'])
            if system['date_range']['end'] is not None:
                end_dates.append(system['date_range']['end'])
        
        if not start_dates or not end_dates:
            return None
        
        return {
            'start': min(start_dates),
            'end': max(end_dates)
        }
    
    def train_ml_models(self) -> Dict[str, Any]:
        """Train ML models for anomaly detection and performance prediction."""
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available - skipping model training")
            return {'status': 'skipped', 'reason': 'ML libraries not available'}
        
        self.logger.info("Training ML models for predictive analytics")
        
        training_results = {
            'anomaly_detection': {'status': 'not_trained'},
            'performance_prediction': {'status': 'not_trained'},
            'training_data_points': 0
        }
        
        # Prepare training data
        all_features = []
        all_targets = []
        
        for inverter_id, system_data in self.fleet_data.items():
            df = system_data['data'].copy()
            
            # Create feature matrix
            feature_cols = []
            
            # Add numeric columns as features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Timestamp'] and not df[col].isna().all():
                    feature_cols.append(col)
            
            if len(feature_cols) > 0:
                features = df[feature_cols].fillna(df[feature_cols].median())
                
                # Add time-based features
                if 'Timestamp' in df.columns:
                    features['hour'] = df['Timestamp'].dt.hour
                    features['day_of_week'] = df['Timestamp'].dt.dayofweek
                    features['month'] = df['Timestamp'].dt.month
                
                all_features.append(features)
                
                # Target for performance prediction (efficiency)
                if 'Efficiency' in df.columns:
                    all_targets.extend(df['Efficiency'].fillna(df['Efficiency'].median()).tolist())
        
        if all_features:
            # Combine all features
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_features = combined_features.fillna(combined_features.median())
            
            training_results['training_data_points'] = len(combined_features)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(combined_features)
            
            # Train anomaly detection model
            try:
                self.anomaly_detector = IsolationForest(
                    contamination=0.1,  # Expect 10% anomalies
                    random_state=42,
                    n_estimators=100
                )
                self.anomaly_detector.fit(scaled_features)
                training_results['anomaly_detection']['status'] = 'trained'
                training_results['anomaly_detection']['features'] = list(combined_features.columns)
                self.logger.info("Anomaly detection model trained successfully")
                
            except Exception as e:
                self.logger.error(f"Error training anomaly detection model: {str(e)}")
                training_results['anomaly_detection']['status'] = 'failed'
                training_results['anomaly_detection']['error'] = str(e)
            
            # Train performance prediction model
            if all_targets and len(all_targets) == len(combined_features):
                try:
                    self.performance_predictor = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        max_depth=10
                    )
                    self.performance_predictor.fit(scaled_features, all_targets)
                    
                    # Calculate training accuracy
                    predictions = self.performance_predictor.predict(scaled_features)
                    mse = mean_squared_error(all_targets, predictions)
                    
                    training_results['performance_prediction']['status'] = 'trained'
                    training_results['performance_prediction']['mse'] = mse
                    training_results['performance_prediction']['features'] = list(combined_features.columns)
                    self.logger.info(f"Performance prediction model trained (MSE: {mse:.4f})")
                    
                except Exception as e:
                    self.logger.error(f"Error training performance prediction model: {str(e)}")
                    training_results['performance_prediction']['status'] = 'failed'
                    training_results['performance_prediction']['error'] = str(e)
        
        return training_results
    
    def real_time_analysis(self, inverter_id: str) -> Dict[str, Any]:
        """Perform real-time analysis for a specific inverter."""
        if inverter_id not in self.fleet_data:
            return {'error': f'Inverter {inverter_id} not found in fleet data'}
        
        system_data = self.fleet_data[inverter_id]
        df = system_data['data']
        
        # Get latest data point (simulating real-time)
        latest_data = df.iloc[-1] if len(df) > 0 else None
        if latest_data is None:
            return {'error': 'No data available for analysis'}
        
        analysis_result = {
            'inverter_id': inverter_id,
            'timestamp': datetime.now().isoformat(),
            'latest_reading': latest_data.to_dict(),
            'status': 'normal',
            'alerts': [],
            'recommendations': [],
            'health_score': 100.0,
            'performance_metrics': {},
            'anomaly_score': 0.0,
            'predicted_performance': None
        }
        
        # Performance metrics calculation
        baseline = self.performance_baselines.get(inverter_id, {})
        
        # Efficiency analysis
        if 'Efficiency' in latest_data.index and 'efficiency' in baseline:
            current_efficiency = latest_data['Efficiency']
            baseline_efficiency = baseline['efficiency']['median']
            efficiency_drop = ((baseline_efficiency - current_efficiency) / baseline_efficiency) * 100
            
            analysis_result['performance_metrics']['efficiency'] = {
                'current': current_efficiency,
                'baseline': baseline_efficiency,
                'drop_percentage': efficiency_drop
            }
            
            if efficiency_drop > self.config['alert_thresholds']['efficiency_drop']:
                analysis_result['alerts'].append({
                    'type': 'efficiency_drop',
                    'severity': 'high' if efficiency_drop > 10 else 'medium',
                    'message': f'Efficiency dropped by {efficiency_drop:.1f}% from baseline',
                    'value': current_efficiency,
                    'baseline': baseline_efficiency
                })
                analysis_result['status'] = 'warning'
        
        # Temperature analysis
        temp_cols = [col for col in latest_data.index if 'Temp' in col or 'temp' in col]
        for col in temp_cols:
            if col in latest_data.index:
                temp_value = latest_data[col]
                if temp_value > self.config['alert_thresholds']['temperature_high']:
                    analysis_result['alerts'].append({
                        'type': 'high_temperature',
                        'severity': 'high' if temp_value > 90 else 'medium',
                        'message': f'{col} temperature is {temp_value:.1f}°C (threshold: {self.config["alert_thresholds"]["temperature_high"]}°C)',
                        'value': temp_value,
                        'threshold': self.config['alert_thresholds']['temperature_high']
                    })
                    analysis_result['status'] = 'warning'
        
        # String performance analysis
        voltage_cols = [col for col in latest_data.index if col.startswith('Vstr')]
        current_cols = [col for col in latest_data.index if col.startswith('Istr')]
        
        if voltage_cols and current_cols:
            # Calculate string power and detect imbalances
            string_powers = []
            for i in range(min(len(voltage_cols), len(current_cols))):
                v_col = f'Vstr{i+1:02d}'
                i_col = f'Istr{i+1:02d}'
                if v_col in latest_data.index and i_col in latest_data.index:
                    power = latest_data[v_col] * latest_data[i_col]
                    string_powers.append(power)
            
            if len(string_powers) > 1:
                mean_power = np.mean(string_powers)
                max_deviation = max(abs(p - mean_power) for p in string_powers)
                deviation_percentage = (max_deviation / mean_power) * 100 if mean_power > 0 else 0
                
                analysis_result['performance_metrics']['string_balance'] = {
                    'mean_power': mean_power,
                    'max_deviation_percentage': deviation_percentage,
                    'string_count': len(string_powers)
                }
                
                if deviation_percentage > self.config['alert_thresholds']['current_imbalance']:
                    analysis_result['alerts'].append({
                        'type': 'string_imbalance',
                        'severity': 'medium',
                        'message': f'String power imbalance detected: {deviation_percentage:.1f}% deviation',
                        'value': deviation_percentage,
                        'threshold': self.config['alert_thresholds']['current_imbalance']
                    })
                    analysis_result['status'] = 'warning'
        
        # ML-based anomaly detection
        if self.anomaly_detector is not None and ML_AVAILABLE:
            try:
                # Prepare features for anomaly detection
                feature_cols = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else []
                features = []
                
                for col in feature_cols:
                    if col in latest_data.index:
                        features.append(latest_data[col])
                    else:
                        # Use baseline value if column not present
                        features.append(0.0)  # Or use appropriate default
                
                if features:
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = self.scaler.transform(features_array)
                    anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                    is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                    
                    analysis_result['anomaly_score'] = float(anomaly_score)
                    
                    if is_anomaly:
                        analysis_result['alerts'].append({
                            'type': 'anomaly_detected',
                            'severity': 'high',
                            'message': f'Anomaly detected (score: {anomaly_score:.3f})',
                            'value': anomaly_score
                        })
                        analysis_result['status'] = 'alert'
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {str(e)}")
        
        # Performance prediction
        if self.performance_predictor is not None and ML_AVAILABLE:
            try:
                # Use same features as anomaly detection
                feature_cols = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else []
                features = []
                
                for col in feature_cols:
                    if col in latest_data.index:
                        features.append(latest_data[col])
                    else:
                        features.append(0.0)
                
                if features:
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = self.scaler.transform(features_array)
                    predicted_efficiency = self.performance_predictor.predict(features_scaled)[0]
                    
                    analysis_result['predicted_performance'] = {
                        'efficiency': float(predicted_efficiency),
                        'prediction_window': f"{self.config['prediction_window']} days"
                    }
                
            except Exception as e:
                self.logger.error(f"Error in performance prediction: {str(e)}")
        
        # Calculate overall health score
        health_score = 100.0
        for alert in analysis_result['alerts']:
            if alert['severity'] == 'high':
                health_score -= 20
            elif alert['severity'] == 'medium':
                health_score -= 10
            else:
                health_score -= 5
        
        analysis_result['health_score'] = max(0.0, health_score)
        
        # Generate recommendations
        recommendations = []
        if analysis_result['health_score'] < 80:
            recommendations.append("Schedule immediate inspection")
        if any(alert['type'] == 'high_temperature' for alert in analysis_result['alerts']):
            recommendations.append("Check cooling system and ventilation")
        if any(alert['type'] == 'string_imbalance' for alert in analysis_result['alerts']):
            recommendations.append("Inspect string connections and panel conditions")
        if any(alert['type'] == 'efficiency_drop' for alert in analysis_result['alerts']):
            recommendations.append("Clean panels and check for shading issues")
        
        analysis_result['recommendations'] = recommendations
        
        return analysis_result
    
    def fleet_status_dashboard(self) -> Dict[str, Any]:
        """Generate fleet-wide status dashboard data."""
        self.logger.info("Generating fleet status dashboard")
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'fleet_summary': {
                'total_systems': len(self.fleet_data),
                'systems_online': 0,
                'systems_warning': 0,
                'systems_alert': 0,
                'total_alerts': 0,
                'average_health_score': 0.0,
                'total_power': 0.0
            },
            'system_statuses': {},
            'top_alerts': [],
            'performance_trends': {},
            'maintenance_schedule': self.get_maintenance_schedule()
        }
        
        # Analyze each system
        health_scores = []
        all_alerts = []
        
        for inverter_id in self.fleet_data.keys():
            system_analysis = self.real_time_analysis(inverter_id)
            
            # Update counters
            if system_analysis['status'] == 'normal':
                dashboard_data['fleet_summary']['systems_online'] += 1
            elif system_analysis['status'] == 'warning':
                dashboard_data['fleet_summary']['systems_warning'] += 1
            else:
                dashboard_data['fleet_summary']['systems_alert'] += 1
            
            dashboard_data['fleet_summary']['total_alerts'] += len(system_analysis['alerts'])
            health_scores.append(system_analysis['health_score'])
            
            # Add to all alerts with system ID
            for alert in system_analysis['alerts']:
                alert['inverter_id'] = inverter_id
                all_alerts.append(alert)
            
            # Store system status
            dashboard_data['system_statuses'][inverter_id] = {
                'status': system_analysis['status'],
                'health_score': system_analysis['health_score'],
                'alert_count': len(system_analysis['alerts']),
                'latest_reading_time': system_analysis['timestamp']
            }
            
            # Add power if available
            if 'DC_Power' in system_analysis['latest_reading']:
                dashboard_data['fleet_summary']['total_power'] += system_analysis['latest_reading']['DC_Power']
        
        # Calculate fleet averages
        if health_scores:
            dashboard_data['fleet_summary']['average_health_score'] = np.mean(health_scores)
        
        # Sort and get top alerts
        all_alerts.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}.get(x['severity'], 0), reverse=True)
        dashboard_data['top_alerts'] = all_alerts[:10]  # Top 10 alerts
        
        return dashboard_data
    
    def get_maintenance_schedule(self) -> List[Dict[str, Any]]:
        """Generate maintenance schedule based on system conditions."""
        schedule = []
        
        for inverter_id in self.fleet_data.keys():
            # Get system analysis
            analysis = self.real_time_analysis(inverter_id)
            
            # Determine maintenance priority
            priority = 'routine'
            due_date = datetime.now() + timedelta(days=self.config['maintenance_intervals']['routine_inspection'])
            
            if analysis['health_score'] < 70:
                priority = 'urgent'
                due_date = datetime.now() + timedelta(days=1)
            elif analysis['health_score'] < 85:
                priority = 'high'
                due_date = datetime.now() + timedelta(days=7)
            elif len(analysis['alerts']) > 0:
                priority = 'medium'
                due_date = datetime.now() + timedelta(days=14)
            
            # Create maintenance task
            task = {
                'inverter_id': inverter_id,
                'priority': priority,
                'due_date': due_date.isoformat(),
                'task_type': 'inspection',
                'estimated_duration': '2 hours',
                'health_score': analysis['health_score'],
                'alert_count': len(analysis['alerts']),
                'recommendations': analysis['recommendations']
            }
            
            schedule.append(task)
        
        # Sort by priority and due date
        priority_order = {'urgent': 0, 'high': 1, 'medium': 2, 'routine': 3}
        schedule.sort(key=lambda x: (priority_order.get(x['priority'], 4), x['due_date']))
        
        return schedule
    
    def generate_monitoring_report(self, output_path: str = None) -> str:
        """Generate comprehensive monitoring report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"monitoring_report_{timestamp}.json"
        
        self.logger.info(f"Generating monitoring report: {output_path}")
        
        # Get fleet dashboard data
        dashboard_data = self.fleet_status_dashboard()
        
        # Add additional analytics
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'fleet_monitoring',
                'monitoring_system_version': '1.0.0',
                'data_period': self._get_fleet_date_range()
            },
            'fleet_dashboard': dashboard_data,
            'system_configurations': {
                'alert_thresholds': self.config['alert_thresholds'],
                'maintenance_intervals': self.config['maintenance_intervals'],
                'prediction_window': self.config['prediction_window']
            },
            'ml_model_status': {
                'anomaly_detection': self.anomaly_detector is not None,
                'performance_prediction': self.performance_predictor is not None,
                'ml_libraries_available': ML_AVAILABLE
            },
            'data_summary': {
                'total_systems': len(self.fleet_data),
                'total_data_points': sum(system['data_points'] for system in self.fleet_data.values()),
                'systems_with_baselines': len(self.performance_baselines)
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report saved to {output_path}")
        return output_path
    
    def create_monitoring_dashboard(self, output_path: str = None) -> str:
        """Create interactive monitoring dashboard."""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available - cannot create interactive dashboard")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"monitoring_dashboard_{timestamp}.html"
        
        self.logger.info(f"Creating monitoring dashboard: {output_path}")
        
        # Get dashboard data
        dashboard_data = self.fleet_status_dashboard()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Fleet Health Scores',
                'System Status Distribution',
                'Alert Severity Distribution',
                'Power Output by System',
                'Temperature Status',
                'Maintenance Schedule Priority'
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Fleet health scores
        systems = list(dashboard_data['system_statuses'].keys())
        health_scores = [dashboard_data['system_statuses'][sys]['health_score'] for sys in systems]
        
        fig.add_trace(
            go.Bar(
                x=systems,
                y=health_scores,
                name="Health Score",
                marker_color=['red' if h < 70 else 'orange' if h < 85 else 'green' for h in health_scores]
            ),
            row=1, col=1
        )
        
        # System status distribution
        status_counts = {}
        for sys_data in dashboard_data['system_statuses'].values():
            status = sys_data['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                name="System Status"
            ),
            row=1, col=2
        )
        
        # Alert severity distribution
        alert_severity_counts = {}
        for alert in dashboard_data['top_alerts']:
            severity = alert['severity']
            alert_severity_counts[severity] = alert_severity_counts.get(severity, 0) + 1
        
        if alert_severity_counts:
            fig.add_trace(
                go.Pie(
                    labels=list(alert_severity_counts.keys()),
                    values=list(alert_severity_counts.values()),
                    name="Alert Severity"
                ),
                row=2, col=1
            )
        
        # System power output (if available)
        power_data = []
        for inverter_id in systems:
            if inverter_id in self.fleet_data:
                latest_data = self.fleet_data[inverter_id]['data'].iloc[-1]
                if 'DC_Power' in latest_data.index:
                    power_data.append(latest_data['DC_Power'])
                else:
                    power_data.append(0)
            else:
                power_data.append(0)
        
        fig.add_trace(
            go.Bar(
                x=systems,
                y=power_data,
                name="DC Power (W)",
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        # Temperature status (example with first available temperature)
        temp_data = []
        for inverter_id in systems:
            if inverter_id in self.fleet_data:
                latest_data = self.fleet_data[inverter_id]['data'].iloc[-1]
                temp_cols = [col for col in latest_data.index if 'Temp' in col or 'temp' in col]
                if temp_cols:
                    temp_data.append(latest_data[temp_cols[0]])
                else:
                    temp_data.append(25)  # Default temperature
            else:
                temp_data.append(25)
        
        fig.add_trace(
            go.Scatter(
                x=systems,
                y=temp_data,
                mode='markers+lines',
                name="Temperature (°C)",
                marker=dict(
                    color=['red' if t > 85 else 'orange' if t > 70 else 'green' for t in temp_data],
                    size=10
                )
            ),
            row=3, col=1
        )
        
        # Maintenance priority
        maintenance_schedule = dashboard_data['maintenance_schedule']
        priority_counts = {}
        for task in maintenance_schedule:
            priority = task['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(priority_counts.keys()),
                y=list(priority_counts.values()),
                name="Maintenance Tasks",
                marker_color=['red', 'orange', 'yellow', 'green'][:len(priority_counts)]
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Real-Time Fleet Monitoring Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            height=1200,
            showlegend=True
        )
        
        # Add fleet summary as annotation
        fleet_summary = dashboard_data['fleet_summary']
        summary_text = f"""
        Fleet Summary:
        • Total Systems: {fleet_summary['total_systems']}
        • Online: {fleet_summary['systems_online']} | Warning: {fleet_summary['systems_warning']} | Alert: {fleet_summary['systems_alert']}
        • Avg Health Score: {fleet_summary['average_health_score']:.1f}%
        • Total Alerts: {fleet_summary['total_alerts']}
        • Total Power: {fleet_summary['total_power']:.1f} W
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        # Save dashboard
        fig.write_html(output_path)
        self.logger.info(f"Monitoring dashboard saved to {output_path}")
        
        return output_path

def main():
    """Main function to demonstrate the monitoring system."""
    print("🔄 Real-Time PV Fleet Monitoring System")
    print("=" * 50)
    
    # Initialize monitoring system
    monitor = RealTimeMonitoringSystem()
    
    # Load fleet data
    data_directory = "/Users/chanthawat/Development/py-dev/iv-curve-classic"
    fleet_summary = monitor.load_fleet_data(data_directory)
    
    print(f"✅ Loaded {fleet_summary['systems_loaded']} systems with {fleet_summary['total_data_points']} data points")
    
    # Train ML models
    print("\n🤖 Training ML models...")
    training_results = monitor.train_ml_models()
    print(f"   Anomaly Detection: {training_results['anomaly_detection']['status']}")
    print(f"   Performance Prediction: {training_results['performance_prediction']['status']}")
    
    # Generate fleet dashboard
    print("\n📊 Generating fleet dashboard...")
    dashboard_data = monitor.fleet_status_dashboard()
    
    fleet_summary = dashboard_data['fleet_summary']
    print(f"   Systems Online: {fleet_summary['systems_online']}")
    print(f"   Systems with Warnings: {fleet_summary['systems_warning']}")
    print(f"   Systems with Alerts: {fleet_summary['systems_alert']}")
    print(f"   Average Health Score: {fleet_summary['average_health_score']:.1f}%")
    print(f"   Total Alerts: {fleet_summary['total_alerts']}")
    
    # Show top alerts
    if dashboard_data['top_alerts']:
        print(f"\n⚠️  Top 5 Alerts:")
        for i, alert in enumerate(dashboard_data['top_alerts'][:5], 1):
            print(f"   {i}. [{alert['severity'].upper()}] {alert['inverter_id']}: {alert['message']}")
    
    # Show maintenance schedule
    print(f"\n🔧 Maintenance Schedule (Top 5 priorities):")
    for i, task in enumerate(dashboard_data['maintenance_schedule'][:5], 1):
        due_date = datetime.fromisoformat(task['due_date']).strftime('%Y-%m-%d')
        print(f"   {i}. [{task['priority'].upper()}] {task['inverter_id']} - Due: {due_date} (Health: {task['health_score']:.1f}%)")
    
    # Generate reports
    print("\n📋 Generating reports...")
    report_path = monitor.generate_monitoring_report()
    print(f"   Report saved: {report_path}")
    
    # Create dashboard
    if PLOTLY_AVAILABLE:
        dashboard_path = monitor.create_monitoring_dashboard()
        print(f"   Dashboard saved: {dashboard_path}")
    else:
        print("   Dashboard creation skipped (Plotly not available)")
    
    print("\n✅ Real-time monitoring system demonstration complete!")
    
    return monitor

if __name__ == "__main__":
    main()
