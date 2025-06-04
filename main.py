"""
Main script for I-V Curve Analysis with Real Inverter Data

This script demonstrates the use of real inverter data for I-V curve analysis,
parameter extraction, and diagnostic analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from iv_curve_classic.data_loader import InverterDataLoader, get_real_data_samples
from iv_curve_classic.iv_analyzer import IVAnalyzer
from iv_curve_classic.parameter_extractor import ParameterExtractor
from iv_curve_classic.diagnostics import DiagnosticAnalyzer
from iv_curve_classic.environmental_corrections import EnvironmentalCorrections


def main():
    """Main function demonstrating I-V analysis with real data."""
    
    print("🔬 I-V Curve Analysis with Real Inverter Data")
    print("=" * 50)
    
    # Initialize components
    data_loader = InverterDataLoader("inverter")
    analyzer = IVAnalyzer()
    diagnostics = DiagnosticAnalyzer()
    
    try:
        # Load real data samples
        print("\n📊 Loading real inverter data...")
        real_samples = get_real_data_samples("inverter")
        
        if not real_samples:
            print("❌ No real data found. Please check the inverter directory.")
            return
        
        print(f"✅ Loaded data from {len(real_samples)} inverters")
        
        # Analyze each inverter
        for inverter_id, params in real_samples.items():
            print(f"\n🔍 Analyzing {inverter_id}")
            print("-" * 30)
            
            # Display current parameters
            print("Current I-V Parameters:")
            for key, value in params.items():
                if key in ['voc', 'isc', 'p_max', 'v_mp', 'i_mp']:
                    unit = "V" if "v" in key.lower() else "A" if "i" in key.lower() else "W"
                    print(f"  {key.upper()}: {value:.2f} {unit}")
                elif key == 'fill_factor':
                    print(f"  Fill Factor: {value:.3f}")
                elif key in ['rs', 'rsh']:
                    print(f"  {key.upper()}: {value:.2f} Ω")
            
            # Perform diagnostic analysis
            print("\n🏥 Diagnostic Analysis:")
            diagnosis = diagnostics.analyze_degradation(params)
            
            print(f"  Overall Health: {diagnosis['overall_health']}")
            print(f"  Severity Level: {diagnosis['severity']}")
            
            if diagnosis['primary_issues']:
                print("  Primary Issues:")
                for issue in diagnosis['primary_issues']:
                    print(f"    • {issue}")
            
            if diagnosis['recommendations']:
                print("  Recommendations:")
                for rec in diagnosis['recommendations'][:3]:  # Show top 3
                    print(f"    → {rec}")
            
            # Generate synthetic I-V curve for visualization
            print("\n📈 Generating I-V curve...")
            voltage, current = analyzer.generate_iv_curve(
                voc=params['voc'],
                isc=params['isc'],
                rs=params['rs'],
                rsh=params['rsh']
            )
            
            # Create visualization
            create_analysis_plot(inverter_id, voltage, current, params, diagnosis)
        
        print(f"\n✅ Analysis complete! Check generated plots.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Using fallback demonstration...")
        demonstrate_with_synthetic_data()


def create_analysis_plot(inverter_id: str, voltage: np.ndarray, current: np.ndarray, 
                        params: dict, diagnosis: dict):
    """Create a comprehensive analysis plot for an inverter."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'I-V Curve Analysis: {inverter_id}', fontsize=16, fontweight='bold')
    
    # Plot I-V curve
    ax1.plot(voltage, current, 'b-', linewidth=2, label='I-V Curve')
    ax1.axhline(y=params['isc'], color='r', linestyle='--', alpha=0.7, label=f"Isc = {params['isc']:.2f} A")
    ax1.axvline(x=params['voc'], color='g', linestyle='--', alpha=0.7, label=f"Voc = {params['voc']:.2f} V")
    ax1.scatter([params['v_mp']], [params['i_mp']], color='red', s=100, zorder=5, label=f"MPP ({params['v_mp']:.1f}V, {params['i_mp']:.1f}A)")
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title('I-V Characteristic Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot P-V curve
    power = voltage * current
    ax2.plot(voltage, power, 'g-', linewidth=2, label='P-V Curve')
    ax2.scatter([params['v_mp']], [params['p_max']], color='red', s=100, zorder=5, label=f"Pmax = {params['p_max']:.1f} W")
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Power-Voltage Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Parameters summary
    ax3.axis('off')
    ax3.set_title('Key Parameters', fontweight='bold')
    param_text = f"""
    Open Circuit Voltage (Voc): {params['voc']:.2f} V
    Short Circuit Current (Isc): {params['isc']:.2f} A
    Maximum Power (Pmax): {params['p_max']:.1f} W
    Voltage at MPP (Vmp): {params['v_mp']:.2f} V
    Current at MPP (Imp): {params['i_mp']:.2f} A
    Fill Factor (FF): {params['fill_factor']:.3f}
    Series Resistance (Rs): {params['rs']:.2f} Ω
    Shunt Resistance (Rsh): {params['rsh']:.1f} Ω
    """
    ax3.text(0.1, 0.9, param_text, transform=ax3.transAxes, fontsize=11, 
            verticalalignment='top', fontfamily='monospace')
    
    # Diagnostic summary
    ax4.axis('off')
    ax4.set_title('Diagnostic Summary', fontweight='bold')
    
    # Color code health status
    health_color = {'Good': 'green', 'Fair': 'orange', 'Poor': 'red'}.get(diagnosis['overall_health'], 'black')
    severity_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}.get(diagnosis['severity'], 'black')
    
    diagnostic_text = f"Overall Health: {diagnosis['overall_health']}\n"
    diagnostic_text += f"Severity: {diagnosis['severity']}\n\n"
    
    if diagnosis['primary_issues']:
        diagnostic_text += "Primary Issues:\n"
        for issue in diagnosis['primary_issues'][:3]:
            diagnostic_text += f"• {issue}\n"
    
    if diagnosis['recommendations']:
        diagnostic_text += f"\nTop Recommendations:\n"
        for rec in diagnosis['recommendations'][:2]:
            diagnostic_text += f"→ {rec}\n"
    
    ax4.text(0.1, 0.9, diagnostic_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', wrap=True)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"analysis_{inverter_id.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Plot saved as: {filename}")
    plt.close()


def demonstrate_with_synthetic_data():
    """Demonstrate analysis with synthetic data if real data is unavailable."""
    
    print("\n🔬 Demonstrating with synthetic data...")
    
    analyzer = IVAnalyzer()
    diagnostics = DiagnosticAnalyzer()
    
    # Create sample data representing different conditions
    samples = {
        "Healthy Module": {
            'voc': 55.0, 'isc': 8.5, 'p_max': 350.0, 'v_mp': 45.0, 'i_mp': 7.8,
            'fill_factor': 0.78, 'rs': 0.4, 'rsh': 1000.0
        },
        "Degraded Module": {
            'voc': 52.0, 'isc': 7.8, 'p_max': 280.0, 'v_mp': 42.0, 'i_mp': 6.7,
            'fill_factor': 0.69, 'rs': 0.8, 'rsh': 200.0
        }
    }
    
    reference = samples["Healthy Module"]
    
    for name, params in samples.items():
        print(f"\n📊 Analyzing: {name}")
        
        # Perform diagnostic analysis
        if name != "Healthy Module":
            diagnosis = diagnostics.analyze_degradation(params, reference)
        else:
            diagnosis = diagnostics.analyze_degradation(params)
        
        print(f"  Health: {diagnosis['overall_health']} (Severity: {diagnosis['severity']})")
        
        if diagnosis['primary_issues']:
            print("  Issues:", ", ".join(diagnosis['primary_issues'][:2]))


if __name__ == "__main__":
    main()
