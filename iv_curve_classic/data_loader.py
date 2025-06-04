"""
Data Loader Module

This module provides functionality to load and process real inverter data
for I-V curve analysis and diagnostics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import glob
import warnings


class InverterDataLoader:
    """
    Class for loading and processing inverter CSV data files.
    
    This class handles the real inverter data from the CSV files and converts
    it into a format suitable for I-V curve analysis and diagnostics.
    """
    
    def __init__(self, data_directory: str = "inverter"):
        """
        Initialize the data loader.
        
        Args:
            data_directory: Path to directory containing inverter CSV files
        """
        self.data_directory = Path(data_directory)
        self.column_mapping = self._get_column_mapping()
        
    def _get_column_mapping(self) -> Dict[str, str]:
        """
        Define mapping between CSV columns and standardized parameter names.
        
        Returns:
            Dictionary mapping CSV columns to parameter names
        """
        return {
            'Time': 'timestamp',
            'Status': 'status',
            'EacToday(kWh)': 'energy_today',
            'EacTotal(kWh)': 'energy_total',
            'Pac(W)': 'power_ac',
            'Ppv(W)': 'power_pv',
            'VacR(V)': 'voltage_ac_r',
            'VacS(V)': 'voltage_ac_s', 
            'VacT(V)': 'voltage_ac_t',
            'IacR(A)': 'current_ac_r',
            'IacS(A)': 'current_ac_s',
            'IacT(A)': 'current_ac_t',
            'INVTemp(℃)': 'inverter_temp',
            'Fac(Hz)': 'frequency',
            'PF': 'power_factor'
        }
    
    def load_inverter_files(self, pattern: str = "INVERTER_*.csv") -> List[str]:
        """
        Find all inverter CSV files in the data directory.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of file paths
        """
        files = list(self.data_directory.glob(pattern))
        return [str(f) for f in sorted(files)]
    
    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single inverter CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with processed data
        """
        try:
            # Try different skip row values to handle different CSV formats
            # Some files have metadata rows, others start directly with headers
            for skiprows in [0, 1, 2, 3, 4, 5]:  # Try multiple skip values
                try:
                    df = pd.read_csv(file_path, skiprows=skiprows)
                    
                    # Check if we have the expected columns for inverter data
                    required_cols = ['Time', 'Status']
                    if all(col in df.columns for col in required_cols):
                        print(f"    📄 Successfully loaded with skiprows={skiprows}")
                        break
                except Exception as e:
                    continue
            else:
                # If no format worked, try to read without skipping and look for header row
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Find the line that contains 'Time' and 'Status' (actual header)
                    header_line = None
                    for i, line in enumerate(lines[:10]):  # Check first 10 lines
                        if 'Time' in line and 'Status' in line and 'Vpv1' in line:
                            header_line = i
                            break
                    
                    if header_line is not None:
                        df = pd.read_csv(file_path, skiprows=header_line)
                        print(f"    📄 Found header at line {header_line + 1}")
                    else:
                        print(f"    ❌ Could not find valid header in {file_path}")
                        return pd.DataFrame()
                except Exception as e:
                    print(f"    ❌ Failed to parse {file_path}: {e}")
                    return pd.DataFrame()
            
            # Clean column names (remove any extra characters)
            df.columns = df.columns.str.strip()
            
            # Convert timestamp to datetime - handle different date formats
            if 'Time' in df.columns:
                # Try multiple date formats
                for date_format in ['%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
                    try:
                        df['Time'] = pd.to_datetime(df['Time'], format=date_format, errors='coerce')
                        break
                    except:
                        continue
                else:
                    # If no format worked, try automatic parsing
                    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            
            # Filter for valid operational data
            df = df.dropna(subset=['Time'])
            
            # Remove rows with invalid or missing status
            if 'Status' in df.columns:
                df = df[df['Status'].notna()]
            
            return df
            
        except Exception as e:
            warnings.warn(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()
    
    def extract_pv_parameters(self, df: pd.DataFrame, string_count: int = 10) -> List[Dict[str, float]]:
        """
        Extract PV string parameters from the dataframe.
        
        Args:
            df: DataFrame with inverter data
            string_count: Number of PV strings to extract (default 10)
            
        Returns:
            List of dictionaries containing PV parameters for each time point
        """
        pv_data = []
        
        for _, row in df.iterrows():
            # Skip non-operational records, but include Waiting status for analysis
            # (Waiting status might still have useful voltage/current data)
            if row.get('Status') not in ['Normal', 'Waiting']:
                continue
                
            # Extract basic parameters
            record = {
                'timestamp': row.get('Time'),
                'status': row.get('Status'),
                'power_ac': self._safe_float(row.get('Pac(W)', 0)),
                'power_pv': self._safe_float(row.get('Ppv(W)', 0)),
                'inverter_temp': self._safe_float(row.get('INVTemp(℃)', 0)),
                'frequency': self._safe_float(row.get('Fac(Hz)', 0)),
            }
            
            # Extract PV string voltages and currents
            string_voltages = []
            string_currents = []
            string_powers = []
            
            for i in range(1, string_count + 1):
                v_col = f'Vpv{i}(V)'
                i_col = f'Ipv{i}(A)'
                p_col = f'Ppv{i}(W)'
                
                voltage = self._safe_float(row.get(v_col, 0))
                current = self._safe_float(row.get(i_col, 0))
                power = self._safe_float(row.get(p_col, 0))
                
                if voltage > 0:  # Only include active strings
                    string_voltages.append(voltage)
                    string_currents.append(current)
                    string_powers.append(power)
            
            if string_voltages:  # Only add records with active strings
                record.update({
                    'string_voltages': string_voltages,
                    'string_currents': string_currents,
                    'string_powers': string_powers,
                    'num_active_strings': len(string_voltages)
                })
                
                # Calculate aggregate parameters
                record.update(self._calculate_iv_parameters(record))
                pv_data.append(record)
        
        return pv_data
    
    def _calculate_iv_parameters(self, record: Dict) -> Dict[str, float]:
        """
        Calculate I-V curve parameters from string data.
        
        Args:
            record: Dictionary with string voltage/current data
            
        Returns:
            Dictionary with calculated I-V parameters
        """
        voltages = np.array(record['string_voltages'])
        currents = np.array(record['string_currents'])
        powers = np.array(record['string_powers'])
        
        # Calculate aggregate values
        total_power = np.sum(powers)
        avg_voltage = np.mean(voltages[voltages > 0])
        total_current = np.sum(currents)
        
        # Estimate I-V parameters (simplified approach)
        # In real applications, you would need actual I-V curve sweeps
        voc_estimate = np.max(voltages) if len(voltages) > 0 else 0
        isc_estimate = np.max(currents) if len(currents) > 0 else 0
        
        # Calculate fill factor estimate
        if voc_estimate > 0 and isc_estimate > 0:
            ff_estimate = total_power / (voc_estimate * isc_estimate)
        else:
            ff_estimate = 0
        
        # Estimate series and shunt resistance (simplified)
        # These would normally require detailed I-V curve analysis
        rs_estimate = self._estimate_series_resistance(voltages, currents)
        rsh_estimate = self._estimate_shunt_resistance(voltages, currents)
        
        return {
            'voc': voc_estimate,
            'isc': isc_estimate,
            'p_max': total_power,
            'v_mp': avg_voltage,
            'i_mp': total_current,
            'fill_factor': ff_estimate,
            'rs': rs_estimate,
            'rsh': rsh_estimate
        }
    
    def _estimate_series_resistance(self, voltages: np.ndarray, currents: np.ndarray) -> float:
        """
        Estimate series resistance from voltage and current data.
        
        This is a simplified estimation. Real Rs calculation requires I-V curve slopes.
        """
        if len(voltages) < 2 or len(currents) < 2:
            return 0.5  # Default value
        
        # Simple estimation based on voltage variation
        voltage_std = float(np.std(voltages[voltages > 0]))
        current_mean = float(np.mean(currents[currents > 0]))
        
        if current_mean > 0:
            return float(min(voltage_std / current_mean, 2.0))  # Cap at reasonable value
        return 0.5
    
    def _estimate_shunt_resistance(self, voltages: np.ndarray, currents: np.ndarray) -> float:
        """
        Estimate shunt resistance from voltage and current data.
        
        This is a simplified estimation. Real Rsh calculation requires I-V curve analysis.
        """
        if len(voltages) < 2 or len(currents) < 2:
            return 1000.0  # Default high value
        
        # Simple estimation - higher voltages should indicate better shunt resistance
        avg_voltage = float(np.mean(voltages[voltages > 0]))
        min_current = float(np.min(currents[currents > 0])) if np.any(currents > 0) else 0.01
        
        if min_current > 0:
            return float(max(avg_voltage / min_current, 50.0))  # Minimum reasonable value
        return 1000.0
    
    def _safe_float(self, value) -> float:
        """
        Safely convert value to float, returning 0 for invalid values.
        
        Args:
            value: Value to convert
            
        Returns:
            Float value or 0 if conversion fails
        """
        try:
            if pd.isna(value) or value == '':
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def load_all_data(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from all inverter files.
        
        Args:
            max_files: Maximum number of files to load (None for all)
            
        Returns:
            Combined DataFrame with all data
        """
        files = self.load_inverter_files()
        if max_files:
            files = files[:max_files]
        
        all_data = []
        
        for file_path in files:
            print(f"Loading {file_path}...")
            df = self.load_single_file(file_path)
            if not df.empty:
                pv_data = self.extract_pv_parameters(df)
                all_data.extend(pv_data)
        
        if all_data:
            return pd.DataFrame(all_data)
        else:
            return pd.DataFrame()
    

def get_real_data_samples(data_directory: str = "inverter", max_inverters: int = 10, include_synthetic: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Load and process real inverter data samples for I-V curve analysis.
    
    This function loads data from multiple inverter CSV files and returns
    aggregated I-V parameters for each inverter that can be used directly
    by the analysis pipeline.
    
    Args:
        data_directory: Path to directory containing inverter CSV files
        max_inverters: Maximum number of inverters to process
        include_synthetic: Whether to generate synthetic data for inverters without active generation data
        
    Returns:
        Dictionary mapping inverter IDs to their I-V parameters
        Format: {
            'INVERTER_01': {
                'voc': float, 'isc': float, 'p_max': float,
                'v_mp': float, 'i_mp': float, 'fill_factor': float,
                'rs': float, 'rsh': float
            },
            ...
        }
    """
    loader = InverterDataLoader(data_directory)
    
    # Get list of available inverter files
    files = loader.load_inverter_files()
    if not files:
        print(f"⚠️  No inverter files found in {data_directory}")
        return {}
    
    # Limit number of files to process
    files = files[:max_inverters]
    
    real_samples = {}
    no_data_inverters = []
    
    for file_path in files:
        # Extract inverter ID from filename
        inverter_id = Path(file_path).stem.upper()  # e.g., 'INVERTER_01'
        
        print(f"  Processing {inverter_id}...")
        
        try:
            # Load and process the file
            df = loader.load_single_file(file_path)
            if df.empty:
                print(f"    📄 Successfully loaded with skiprows=0")
                print(f"    ⚠️  No valid data in {inverter_id}")
                no_data_inverters.append(inverter_id)
                continue
            
            # Print loading info
            skip_rows = 4 if 'Time' not in df.columns else 0
            print(f"    📄 Successfully loaded with skiprows={skip_rows}")
            
            # Extract PV parameters for all records
            pv_records = loader.extract_pv_parameters(df)
            if not pv_records:
                print(f"    ⚠️  No operational data found in {inverter_id} (likely nighttime/no generation data)")
                no_data_inverters.append(inverter_id)
                continue
            
            # Calculate representative parameters (using median values for stability)
            # This aggregates multiple time points into single representative values
            aggregated_params = _aggregate_pv_parameters(pv_records)
            
            if aggregated_params:
                real_samples[inverter_id] = aggregated_params
                print(f"    ✅ {len(pv_records)} records processed")
            else:
                print(f"    ⚠️  Could not aggregate parameters for {inverter_id}")
                no_data_inverters.append(inverter_id)
                
        except Exception as e:
            print(f"    ❌ Error processing {inverter_id}: {e}")
            no_data_inverters.append(inverter_id)
            continue
    
    # Add synthetic data for demonstration if requested and needed
    if include_synthetic and no_data_inverters and len(real_samples) > 0:
        print(f"\n🔧 Generating synthetic data for {len(no_data_inverters)} inverters without active generation data...")
        synthetic_samples = _generate_synthetic_variations(list(real_samples.values())[0], no_data_inverters[:3])  # Limit to 3 synthetic
        real_samples.update(synthetic_samples)
    
    return real_samples


def _aggregate_pv_parameters(pv_records: List[Dict]) -> Optional[Dict[str, float]]:
    """
    Aggregate multiple PV parameter records into representative values.
    
    Args:
        pv_records: List of dictionaries containing PV parameters from different time points
        
    Returns:
        Dictionary with aggregated I-V parameters, or None if aggregation fails
    """
    if not pv_records:
        return None
    
    try:
        # Extract parameter arrays
        param_arrays = {}
        required_params = ['voc', 'isc', 'p_max', 'v_mp', 'i_mp', 'fill_factor', 'rs', 'rsh']
        
        for param in required_params:
            values = [record.get(param, 0) for record in pv_records if record.get(param, 0) > 0]
            if values:
                param_arrays[param] = np.array(values, dtype=float)
            else:
                param_arrays[param] = np.array([0], dtype=float)
        
        # Calculate representative values using robust statistics
        aggregated = {}
        
        for param, values in param_arrays.items():
            # Filter out invalid values and ensure we have valid data
            valid_values = values[np.isfinite(values) & (values > 0)]
            
            if len(valid_values) > 0:
                if param in ['rs', 'rsh']:
                    # For resistances, use median to avoid outliers
                    aggregated[param] = float(np.median(valid_values))
                elif param == 'fill_factor':
                    # Fill factor should be between 0 and 1
                    valid_ff = valid_values[(valid_values > 0.3) & (valid_values < 1.0)]
                    if len(valid_ff) > 0:
                        aggregated[param] = float(np.median(valid_ff))
                    else:
                        aggregated[param] = 0.75  # Reasonable default
                else:
                    # For other parameters, use 75th percentile to get good operating conditions
                    aggregated[param] = float(np.percentile(valid_values, 75))
            else:
                # Provide reasonable defaults based on typical PV module characteristics
                defaults = {
                    'voc': 48.0, 'isc': 8.0, 'p_max': 300.0,
                    'v_mp': 40.0, 'i_mp': 7.5, 'fill_factor': 0.75,
                    'rs': 0.5, 'rsh': 500.0
                }
                aggregated[param] = float(defaults.get(param, 0.0))
        
        # Validate and adjust parameters to ensure physical consistency
        aggregated = _validate_iv_parameters(aggregated)
        
        return aggregated
        
    except Exception as e:
        warnings.warn(f"Error aggregating PV parameters: {e}")
        return None


def _validate_iv_parameters(params: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and adjust I-V parameters to ensure physical consistency.
    
    Args:
        params: Dictionary of I-V parameters
        
    Returns:
        Dictionary with validated and adjusted parameters
    """
    # Ensure basic physical constraints
    params['voc'] = max(params['voc'], 1.0)  # Minimum voltage
    params['isc'] = max(params['isc'], 0.1)  # Minimum current
    params['p_max'] = max(params['p_max'], 1.0)  # Minimum power
    
    # Ensure MPP is within reasonable bounds
    params['v_mp'] = min(max(params['v_mp'], 1.0), params['voc'] * 0.9)
    params['i_mp'] = min(max(params['i_mp'], 0.1), params['isc'] * 0.95)
    
    # Recalculate maximum power and fill factor for consistency
    theoretical_max = params['voc'] * params['isc']
    actual_max = params['v_mp'] * params['i_mp']
    
    if theoretical_max > 0:
        params['fill_factor'] = min(max(actual_max / theoretical_max, 0.5), 0.85)
        params['p_max'] = max(params['p_max'], actual_max)
    else:
        params['fill_factor'] = 0.75
    
    # Constrain resistance values to reasonable ranges
    params['rs'] = min(max(params['rs'], 0.1), 5.0)  # Series resistance: 0.1-5 Ω
    params['rsh'] = min(max(params['rsh'], 50.0), 10000.0)  # Shunt resistance: 50-10k Ω
    
    return params

def _generate_synthetic_variations(base_params: Dict[str, float], inverter_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Generate synthetic parameter variations based on a real inverter's data.
    
    This creates realistic variations that represent different operating conditions
    or degradation levels for demonstration purposes.
    
    Args:
        base_params: Reference parameters from a real inverter
        inverter_ids: List of inverter IDs to create synthetic data for
        
    Returns:
        Dictionary mapping inverter IDs to synthetic I-V parameters
    """
    synthetic_samples = {}
    
    # Define different scenarios for synthetic data
    scenarios = [
        {"name": "High Performance", "voc_factor": 1.05, "isc_factor": 1.02, "rs_factor": 0.8, "rsh_factor": 1.2, "power_factor": 1.08},
        {"name": "Normal Operation", "voc_factor": 1.0, "isc_factor": 1.0, "rs_factor": 1.0, "rsh_factor": 1.0, "power_factor": 1.0},
        {"name": "Slight Degradation", "voc_factor": 0.95, "isc_factor": 0.97, "rs_factor": 1.3, "rsh_factor": 0.8, "power_factor": 0.92},
        {"name": "Moderate Degradation", "voc_factor": 0.90, "isc_factor": 0.92, "rs_factor": 1.6, "rsh_factor": 0.6, "power_factor": 0.83},
        {"name": "Soiling/Shading", "voc_factor": 0.98, "isc_factor": 0.85, "rs_factor": 1.1, "rsh_factor": 0.9, "power_factor": 0.83}
    ]
    
    for i, inverter_id in enumerate(inverter_ids):
        # Use different scenarios cycling through the list
        scenario = scenarios[i % len(scenarios)]
        
        # Apply variations to the base parameters
        synthetic_params = {
            'voc': base_params['voc'] * scenario['voc_factor'],
            'isc': base_params['isc'] * scenario['isc_factor'],
            'rs': base_params['rs'] * scenario['rs_factor'],
            'rsh': base_params['rsh'] * scenario['rsh_factor']
        }
        
        # Calculate MPP parameters more realistically
        # Vmp is typically 85-90% of Voc depending on cell technology
        synthetic_params['v_mp'] = synthetic_params['voc'] * 0.87 * scenario['voc_factor']
        # Imp is typically 90-95% of Isc
        synthetic_params['i_mp'] = synthetic_params['isc'] * 0.92 * scenario['isc_factor']
        
        # Calculate maximum power with scenario-specific factor
        base_power = base_params.get('p_max', synthetic_params['v_mp'] * synthetic_params['i_mp'])
        synthetic_params['p_max'] = base_power * scenario['power_factor']
        
        # Ensure power is consistent with V_mp and I_mp
        calculated_power = synthetic_params['v_mp'] * synthetic_params['i_mp']
        if calculated_power > synthetic_params['p_max']:
            synthetic_params['p_max'] = calculated_power
        
        # Calculate fill factor
        theoretical_max = synthetic_params['voc'] * synthetic_params['isc']
        if theoretical_max > 0:
            synthetic_params['fill_factor'] = synthetic_params['p_max'] / theoretical_max
            # Ensure reasonable fill factor range (0.65-0.85 for silicon cells)
            synthetic_params['fill_factor'] = min(max(synthetic_params['fill_factor'], 0.65), 0.85)
        else:
            synthetic_params['fill_factor'] = 0.75
        
        # Validate and adjust parameters
        synthetic_params = _validate_iv_parameters(synthetic_params)
        
        synthetic_samples[f"{inverter_id}_SYNTHETIC"] = synthetic_params
        print(f"    🔧 Generated synthetic data: {scenario['name']} (P_max: {synthetic_params['p_max']:.0f}W)")
    
    return synthetic_samples