"""
I-V Curve Analysis Package

A comprehensive package for analyzing photovoltaic I-V characteristics.
"""

__version__ = "1.0.0"
__author__ = "IV Curve Analysis Team"

# Import modules (avoiding circular imports)
try:
    from .iv_analyzer import IVAnalyzer
    from .parameter_extractor import ParameterExtractor
    from .diagnostics import DiagnosticAnalyzer
    from .environmental_corrections import EnvironmentalCorrections
    from .data_loader import InverterDataLoader
    
    __all__ = [
        'IVAnalyzer',
        'ParameterExtractor', 
        'DiagnosticAnalyzer',
        'EnvironmentalCorrections',
        'InverterDataLoader'
    ]
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []
