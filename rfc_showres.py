from scipy.io import loadmat
import numpy as np
import argparse
import os

def show_single_algorithm(alg_path, alg_name, 
                         eval_extended=True, show_improvements=False,
                         metrics_to_show=['SDR_bss', 'SIR_bss', 'SAR_bss', 'STOI', 'SDR_td'],
                         output_file=None):
    """
    Show results from a single algorithm result file
    
    Args:
        alg_path: Path to .mat result file
        alg_name: Name for the algorithm (for display)
        eval_extended: Whether to show extended metrics
        show_improvements: Whether to show improvements vs absolute values
        metrics_to_show: List of metrics to display
        output_file: Path to save results to a text file (optional)
    """
    # Initialize output capture
    output_lines = []
    
    def print_and_capture(message):
        """Print to console and capture for file output"""
        print(message)
        if output_file:
            output_lines.append(message)
    
    # Load algorithm results
    alg_mat = loadmat(alg_path)
    
    # Add header information if saving to file
    if output_file:
        output_lines.append(f"Algorithm Results")
        output_lines.append(f"Generated on: {np.datetime64('now')}")
        output_lines.append(f"Algorithm: {alg_name} ({alg_path})")
        output_lines.append(f"Extended evaluation: {eval_extended}")
        output_lines.append(f"Show improvements: {show_improvements}")
        output_lines.append(f"Metrics: {metrics_to_show}")
        output_lines.append("="*60)
    
    if not eval_extended:
        alg_mres = np.mean(alg_mat['SNR'])
        print_and_capture(alg_name + f' Mean SNR: {alg_mres:.2f} dB')
    else:
        if not show_improvements:
            alg_mres = np.mean(alg_mat['SNR_mix'])
            print_and_capture('Mixture ' + f' Mean SNR: {alg_mres:.2f} dB')
            
            alg_mres = np.mean(alg_mat['SNR'])
            print_and_capture(alg_name + f' Mean SNR: {alg_mres:.2f} dB')
        else:
            alg_mres = np.mean(alg_mat['SNR'] - alg_mat['SNR_mix'])
            print_and_capture(alg_name + f' Mean SNR_imp: {alg_mres:.2f} dB')
        
        # Process each metric
        for metric in metrics_to_show:
            if metric in alg_mat:
                print_single_metric(alg_mat, alg_name, metric, show_improvements, print_and_capture)
    
    print_and_capture('Done!!')
    
    # Save to file if requested
    if output_file:
        save_results_to_file(output_lines, output_file)

def compare_algorithms(alg1_path, alg1_name, alg2_path, alg2_name, 
                      eval_extended=True, show_improvements=False,
                      metrics_to_show=['SDR_bss', 'SIR_bss', 'SAR_bss', 'STOI', 'SDR_td'],
                      output_file=None):
    """
    Compare results from two algorithm result files
    
    Args:
        alg1_path, alg2_path: Paths to .mat result files
        alg1_name, alg2_name: Names for the algorithms (for display)
        eval_extended: Whether to show extended metrics
        show_improvements: Whether to show improvements vs absolute values
        metrics_to_show: List of metrics to display
        output_file: Path to save results to a text file (optional)
    """
    # Initialize output capture
    output_lines = []
    
    def print_and_capture(message):
        """Print to console and capture for file output"""
        print(message)
        if output_file:
            output_lines.append(message)
    # Load algorithm results
    alg1_mat = loadmat(alg1_path)
    alg2_mat = loadmat(alg2_path)
    
    # Add header information if saving to file
    if output_file:
        output_lines.append(f"Algorithm Comparison Results")
        output_lines.append(f"Generated on: {np.datetime64('now')}")
        output_lines.append(f"Algorithm 1: {alg1_name} ({alg1_path})")
        output_lines.append(f"Algorithm 2: {alg2_name} ({alg2_path})")
        output_lines.append(f"Extended evaluation: {eval_extended}")
        output_lines.append(f"Show improvements: {show_improvements}")
        output_lines.append(f"Metrics: {metrics_to_show}")
        output_lines.append("="*60)
    
    if not eval_extended:
        alg1_mres = np.mean(alg1_mat['SNR'])
        alg2_mres = np.mean(alg2_mat['SNR'])
        print_and_capture(alg1_name + f' Mean SNR: {alg1_mres:.2f} dB')
        print_and_capture(alg2_name + f' Mean SNR: {alg2_mres:.2f} dB')
    else:
        if not show_improvements:
            alg1_mres = np.mean(alg1_mat['SNR_mix'])
            alg2_mres = np.mean(alg2_mat['SNR_mix'])
            print_and_capture('Mixture ' + f' Mean SNR: {alg1_mres:.2f} dB')
            print_and_capture('Mixture ' + f' Mean SNR: {alg2_mres:.2f} dB')
            #
            alg1_mres = np.mean(alg1_mat['SNR'])
            alg2_mres = np.mean(alg2_mat['SNR'])
            print_and_capture(alg1_name + f' Mean SNR: {alg1_mres:.2f} dB')
            print_and_capture(alg2_name + f' Mean SNR: {alg2_mres:.2f} dB') 
        else:
            alg1_mres = np.mean(alg1_mat['SNR'] - alg1_mat['SNR_mix'])
            alg2_mres = np.mean(alg2_mat['SNR'] - alg2_mat['SNR_mix'])
            print_and_capture(alg1_name + f' Mean SNR_imp: {alg1_mres:.2f} dB')
            print_and_capture(alg2_name + f' Mean SNR_imp: {alg2_mres:.2f} dB')
        
        # Process each metric
        for metric in metrics_to_show:
            if metric in alg1_mat and metric in alg2_mat:
                print_metric_comparison(alg1_mat, alg1_name, alg2_mat, alg2_name, 
                                       metric, show_improvements, print_and_capture)
    
    print_and_capture('Done!!')
    
    # Save to file if requested
    if output_file:
        save_results_to_file(output_lines, output_file)

def print_single_metric(alg_mat, alg_name, metric_name, show_improvements=False, print_func=print):
    """Helper function to print a specific metric for a single algorithm"""
    if metric_name == 'SDR_bss':
        if not show_improvements:
            alg_mres = np.mean(alg_mat['SDR_bss'], axis=0)
            print_func(alg_name + f' Mean SDR_bss: {alg_mres[1]:.2f} dB, (mixture: {alg_mres[0]:.2f} dB)')
        else:
            alg_mres = np.mean(alg_mat['SDR_bss'][:,1] - alg_mat['SDR_bss'][:,0], axis=0)
            print_func(alg_name + f' Mean SDR_bss_imp: {alg_mres:.2f} dB')
    
    elif metric_name == 'SIR_bss':
        if not show_improvements:
            alg_mres = np.mean(alg_mat['SIR_bss'], axis=0)
            print_func(alg_name + f' Mean SIR_bss: {alg_mres[1]:.2f} dB, (mixture: {alg_mres[0]:.2f} dB)')
        else:
            alg_mres = np.mean(alg_mat['SIR_bss'][:,1] - alg_mat['SIR_bss'][:,0], axis=0)
            print_func(alg_name + f' Mean SIR_bss_imp: {alg_mres:.2f} dB')
    
    elif metric_name == 'SAR_bss':
        if not show_improvements:
            alg_mres = np.mean(alg_mat['SAR_bss'], axis=0)
            print_func(alg_name + f' Mean SAR_bss: {alg_mres[1]:.2f} dB, (mixture: {alg_mres[0]:.2f} dB)')
        else:
            alg_mres = np.mean(alg_mat['SAR_bss'][:,1] - alg_mat['SAR_bss'][:,0], axis=0)
            print_func(alg_name + f' Mean SAR_bss_imp: {alg_mres:.2f} dB')
    
    elif metric_name == 'STOI':
        if not show_improvements:
            alg_mres = np.mean(alg_mat['STOI'], axis=0)
            print_func(alg_name + f' Mean STOI: {alg_mres[1]:.2f}, (mixture: {alg_mres[0]:.2f})')
        else:
            alg_mres = np.mean(alg_mat['STOI'][:,1] - alg_mat['STOI'][:,0], axis=0)
            print_func(alg_name + f' Mean STOI_imp: {alg_mres:.2f}')
    
    elif metric_name == 'SDR_td':
        if not show_improvements:
            alg_mres = np.mean(alg_mat['SDR_td'], axis=0)
            print_func(alg_name + f' Mean SDR_td: {alg_mres[1]:.2f} dB, (mixture: {alg_mres[0]:.2f} dB)')
        else:
            alg_mres = np.mean(alg_mat['SDR_td'][:,1] - alg_mat['SDR_td'][:,0], axis=0)
            print_func(alg_name + f' Mean SDR_td_imp: {alg_mres:.2f} dB')

def print_metric_comparison(alg1_mat, alg1_name, alg2_mat, alg2_name, 
                           metric_name, show_improvements=False, print_func=print):
    """Helper function to print comparison for a specific metric"""
    if metric_name == 'SDR_bss':
        if not show_improvements:
            alg1_mres = np.mean(alg1_mat['SDR_bss'], axis=0)
            alg2_mres = np.mean(alg2_mat['SDR_bss'], axis=0)
            print_func(alg1_name + f' Mean SDR_bss: {alg1_mres[1]:.2f} dB, (mixture: {alg1_mres[0]:.2f} dB)')
            print_func(alg2_name + f' Mean SDR_bss: {alg2_mres[1]:.2f} dB, (mixture: {alg2_mres[0]:.2f} dB)')             
        else:
            alg1_mres = np.mean(alg1_mat['SDR_bss'][:,1] - alg1_mat['SDR_bss'][:,0], axis=0)
            alg2_mres = np.mean(alg2_mat['SDR_bss'][:,1] - alg2_mat['SDR_bss'][:,0], axis=0)
            print_func(alg1_name + f' Mean SDR_bss_imp: {alg1_mres:.2f} dB')
            print_func(alg2_name + f' Mean SDR_bss_imp: {alg2_mres:.2f} dB')
    
    elif metric_name == 'SIR_bss':
        if not show_improvements:
            alg1_mres = np.mean(alg1_mat['SIR_bss'], axis=0)
            alg2_mres = np.mean(alg2_mat['SIR_bss'], axis=0)
            print_func(alg1_name + f' Mean SIR_bss: {alg1_mres[1]:.2f} dB, (mixture: {alg1_mres[0]:.2f} dB)')
            print_func(alg2_name + f' Mean SIR_bss: {alg2_mres[1]:.2f} dB, (mixture: {alg2_mres[0]:.2f} dB)')             
        else:
            alg1_mres = np.mean(alg1_mat['SIR_bss'][:,1] - alg1_mat['SIR_bss'][:,0], axis=0)
            alg2_mres = np.mean(alg2_mat['SIR_bss'][:,1] - alg2_mat['SIR_bss'][:,0], axis=0)
            print_func(alg1_name + f' Mean SIR_bss_imp: {alg1_mres:.2f} dB')
            print_func(alg2_name + f' Mean SIR_bss_imp: {alg2_mres:.2f} dB')
    
    elif metric_name == 'SAR_bss':
        if not show_improvements:
            alg1_mres = np.mean(alg1_mat['SAR_bss'], axis=0)
            alg2_mres = np.mean(alg2_mat['SAR_bss'], axis=0)
            print_func(alg1_name + f' Mean SAR_bss: {alg1_mres[1]:.2f} dB, (mixture: {alg1_mres[0]:.2f} dB)')
            print_func(alg2_name + f' Mean SAR_bss: {alg2_mres[1]:.2f} dB, (mixture: {alg2_mres[0]:.2f} dB)')             
        else:
            alg1_mres = np.mean(alg1_mat['SAR_bss'][:,1] - alg1_mat['SAR_bss'][:,0], axis=0)
            alg2_mres = np.mean(alg2_mat['SAR_bss'][:,1] - alg2_mat['SAR_bss'][:,0], axis=0)
            print_func(alg1_name + f' Mean SAR_bss_imp: {alg1_mres:.2f} dB')
            print_func(alg2_name + f' Mean SAR_bss_imp: {alg2_mres:.2f} dB')
    
    elif metric_name == 'STOI':
        if not show_improvements:
            alg1_mres = np.mean(alg1_mat['STOI'], axis=0)
            alg2_mres = np.mean(alg2_mat['STOI'], axis=0)
            print_func(alg1_name + f' Mean STOI: {alg1_mres[1]:.2f}, (mixture: {alg1_mres[0]:.2f})')
            print_func(alg2_name + f' Mean STOI: {alg2_mres[1]:.2f}, (mixture: {alg2_mres[0]:.2f})')             
        else:
            alg1_mres = np.mean(alg1_mat['STOI'][:,1] - alg1_mat['STOI'][:,0], axis=0)
            alg2_mres = np.mean(alg2_mat['STOI'][:,1] - alg2_mat['STOI'][:,0], axis=0)
            print_func(alg1_name + f' Mean STOI_imp: {alg1_mres:.2f}')
            print_func(alg2_name + f' Mean STOI_imp: {alg2_mres:.2f}')
    
    elif metric_name == 'SDR_td':
        if not show_improvements:
            alg1_mres = np.mean(alg1_mat['SDR_td'], axis=0)
            alg2_mres = np.mean(alg2_mat['SDR_td'], axis=0)
            print_func(alg1_name + f' Mean SDR_td: {alg1_mres[1]:.2f} dB, (mixture: {alg1_mres[0]:.2f} dB)')
            print_func(alg2_name + f' Mean SDR_td: {alg2_mres[1]:.2f} dB, (mixture: {alg2_mres[0]:.2f} dB)')             
        else:
            alg1_mres = np.mean(alg1_mat['SDR_td'][:,1] - alg1_mat['SDR_td'][:,0], axis=0)
            alg2_mres = np.mean(alg2_mat['SDR_td'][:,1] - alg2_mat['SDR_td'][:,0], axis=0)
            print_func(alg1_name + f' Mean SDR_td_imp: {alg1_mres:.2f} dB')
            print_func(alg2_name + f' Mean SDR_td_imp: {alg2_mres:.2f} dB')

def save_results_to_file(output_lines, output_file):
    """Save the comparison results to a text file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving results to file: {e}")

###
# Usage examples:
# 
# Show single algorithm results:
# python rfc_showres.py --alg1_path results/algorithm1.mat --alg1_name "Algorithm 1"
#
# Compare two algorithms:
# python rfc_showres.py --alg1_path results/algorithm1.mat --alg1_name "Algorithm 1" \
#                       --alg2_path results/algorithm2.mat --alg2_name "Algorithm 2"
#
# Show improvements only:
# python rfc_showres.py --alg1_path results/algorithm1.mat --alg1_name "Algorithm 1" \
#                       --show_improvements
#
# Save results to file:
# python rfc_showres.py --alg1_path results/algorithm1.mat --alg1_name "Algorithm 1" \
#                       --output results_summary.txt
###

def main():
    parser = argparse.ArgumentParser(description='Show results from algorithm result files - single file or comparison between two files')
    parser.add_argument('--alg1_path', type=str, required=True,
                       help='Path to first algorithm result .mat file')
    parser.add_argument('--alg1_name', type=str, required=True,
                       help='Name for the first algorithm (for display)')
    parser.add_argument('--alg2_path', type=str, default=None,
                       help='Path to second algorithm result .mat file (optional - for comparison)')
    parser.add_argument('--alg2_name', type=str, default=None,
                       help='Name for the second algorithm (for display, required if alg2_path is provided)')
    parser.add_argument('--eval_extended', action='store_true', default=True,
                       help='Enable extended evaluation (default: True)')
    parser.add_argument('--show_improvements', action='store_true', default=False,
                       help='Show improvements rather than absolute values (default: False)')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['SDR_bss', 'SIR_bss', 'SAR_bss', 'STOI', 'SDR_td'],
                       help='Metrics to show (default: SDR_bss SIR_bss SAR_bss STOI SDR_td)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results to a text file (optional)')
    
    args = parser.parse_args()
    
    # Check if comparison mode or single mode
    comparison_mode = args.alg2_path is not None
    
    if comparison_mode and args.alg2_name is None:
        print("Error: alg2_name is required when alg2_path is provided for comparison")
        return
    
    # Handle both absolute and relative paths
    if os.path.isabs(args.alg1_path):
        alg1_path = args.alg1_path
    else:
        alg1_path = os.path.join(os.path.dirname(__file__), args.alg1_path)
    
    # Check if first result file exists
    if not os.path.exists(alg1_path):
        print(f"Error: Algorithm result file not found: {alg1_path}")
        return
    
    if comparison_mode:
        if os.path.isabs(args.alg2_path):
            alg2_path = args.alg2_path
        else:
            alg2_path = os.path.join(os.path.dirname(__file__), args.alg2_path)
        
        # Check if second result file exists
        if not os.path.exists(alg2_path):
            print(f"Error: Second algorithm result file not found: {alg2_path}")
            return
    
    # Handle output file path
    output_file = None
    if args.output:
        if os.path.isabs(args.output):
            output_file = args.output
        else:
            output_file = os.path.join(os.path.dirname(__file__), args.output)
    
    if comparison_mode:
        print(f"Comparing algorithms:")
        print(f"  {args.alg1_name}: {alg1_path}")
        print(f"  {args.alg2_name}: {alg2_path}")
        print(f"  Extended evaluation: {args.eval_extended}")
        print(f"  Show improvements: {args.show_improvements}")
        print(f"  Metrics: {args.metrics}")
        if output_file:
            print(f"  Output file: {output_file}")
        print(f"{'='*60}")
        
        # Run comparison
        compare_algorithms(alg1_path, args.alg1_name, alg2_path, args.alg2_name,
                          eval_extended=args.eval_extended,
                          show_improvements=args.show_improvements,
                          metrics_to_show=args.metrics,
                          output_file=output_file)
    else:
        print(f"Showing results for single algorithm:")
        print(f"  {args.alg1_name}: {alg1_path}")
        print(f"  Extended evaluation: {args.eval_extended}")
        print(f"  Show improvements: {args.show_improvements}")
        print(f"  Metrics: {args.metrics}")
        if output_file:
            print(f"  Output file: {output_file}")
        print(f"{'='*60}")
        
        # Show single algorithm results
        show_single_algorithm(alg1_path, args.alg1_name,
                             eval_extended=args.eval_extended,
                             show_improvements=args.show_improvements,
                             metrics_to_show=args.metrics,
                             output_file=output_file)

if __name__ == "__main__":
    main()
