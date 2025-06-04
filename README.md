# I-V Curve Analysis for Photovoltaic Systems

A comprehensive Python package for analyzing I-V (Current-Voltage) characteristics of photovoltaic devices, including solar cells and modules. This tool provides detailed parameter extraction, environmental correction, and diagnostic capabilities for PV system performance analysis.

## Features

- **Complete I-V Parameter Extraction**: Calculates all key parameters including Voc, Isc, Vmp, Imp, Pmp, Fill Factor, and parasitic resistances
- **Environmental Corrections**: Normalizes I-V curves to Standard Test Conditions (STC) accounting for irradiance and temperature variations
- **Diagnostic Analysis**: Identifies degradation mechanisms and fault types through parameter analysis
- **Interactive Visualization**: Comprehensive plotting capabilities with parameter annotations
- **Performance Monitoring**: Tools for tracking PV system health over time

## Key Parameters Analyzed

1. **Open-Circuit Voltage (Voc)**: Maximum voltage at zero current
2. **Short-Circuit Current (Isc)**: Maximum current at zero voltage
3. **Maximum Power Point (MPP)**: Optimal operating point for maximum power
4. **Fill Factor (FF)**: Quality indicator of the I-V curve "squareness"
5. **Series Resistance (Rs)**: Internal resistance affecting the curve's knee
6. **Shunt Resistance (Rsh)**: Leakage resistance affecting curve slope

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from iv_analyzer import IVAnalyzer
import numpy as np

# Create sample I-V data
voltage = np.linspace(0, 0.6, 100)
current = # Your I-V measurement data

# Initialize analyzer
analyzer = IVAnalyzer()

# Extract parameters
params = analyzer.extract_parameters(voltage, current)
print(f"Fill Factor: {params['fill_factor']:.3f}")
print(f"Maximum Power: {params['p_max']:.2f} W")

# Plot with annotations
analyzer.plot_iv_curve(voltage, current, show_parameters=True)
```

## Documentation

See the `examples/` directory for detailed usage examples and the `docs/` directory for comprehensive documentation on PV I-V curve analysis theory and implementation.
