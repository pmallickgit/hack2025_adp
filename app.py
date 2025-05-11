#!/usr/bin/env python3
"""
Flask Application for Azure OpenAI Security Agent

This Flask app provides a web interface for the Azure OpenAI Security Agent,
allowing users to view reports and run analyses.
"""

import os
import asyncio
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from azure_openai_security_agent import AzureOpenAISecurityAgent

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['SECURITY_INSIGHTS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'security_insights')

# Ensure upload and security_insights directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SECURITY_INSIGHTS'], exist_ok=True)

# Set global variable for running status
analysis_running = False
last_analysis_result = None

# Custom filter to format numbers with commas
@app.template_filter('format_number')
def format_number(value):
    """Format a number with commas as thousands separators"""
    try:
        if value is None:
            return "0"
        elif isinstance(value, (int, float)):
            return f"{int(value):,}"
        else:
            return f"{int(float(str(value).replace(',', ''))): ,}"
    except (ValueError, TypeError):
        return str(value)

@app.route('/')
def index():
    """Show the Security Reports Hub as the main dashboard"""
    # Check for insights and status
    report_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'visual_report.html')
    strategy_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'mitigation_strategy.md')
    stats_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'summary_statistics.json')
    
    has_report = os.path.exists(report_path)
    has_strategy = os.path.exists(strategy_path)
    has_stats = os.path.exists(stats_path)
    
    # Get available datasets for analysis
    available_datasets = []
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    if os.path.exists(datasets_dir):
        available_datasets = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    
    # Check for visualizations
    has_visualizations = os.path.exists(os.path.join(app.config['SECURITY_INSIGHTS'], 'visualizations'))
    
    # Get statistics if available
    statistics = None
    if has_stats:
        try:
            import json
            with open(stats_path, 'r') as f:
                statistics = json.load(f)
        except:
            statistics = None
    
    return render_template('report_hub.html', 
                           has_report=has_report,
                           has_strategy=has_strategy,
                           has_stats=has_stats,
                           has_visualizations=has_visualizations,
                           analysis_running=analysis_running,
                           datasets=available_datasets,
                           statistics=statistics,
                           last_analysis_result=last_analysis_result)

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run the security agent analysis"""
    global analysis_running
    global last_analysis_result
    
    if analysis_running:
        return jsonify({'status': 'error', 'message': 'An analysis is already running'})
    
    try:
        analysis_running = True
        # Get dataset from form
        dataset = request.form.get('dataset')
        if not dataset:
            # Default to samples_5000.csv
            dataset = 'dense_security_logs.csv'
        
        # Full path to dataset
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', dataset)
        
        # Run analysis in a background thread
        def run_agent():
            global analysis_running
            global last_analysis_result
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                print("Starting security analysis...")
                agent = AzureOpenAISecurityAgent(dataset_path)
                last_analysis_result = loop.run_until_complete(agent.run_full_analysis())
                print("Analysis completed successfully")
                
            except Exception as e:
                print(f"ERROR in analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                last_analysis_result = {'error': str(e)}
                
            finally:
                print("Setting analysis_running to False")
                analysis_running = False
                loop.close()
        
        import threading
        thread = threading.Thread(target=run_agent)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': f'Analysis started on dataset: {dataset}'})
        
    except Exception as e:
        analysis_running = False
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/run-analysis', methods=['POST'])
def run_analysis_form():
    """Route for form submission from index.html"""
    global analysis_running
    global last_analysis_result
    
    # Don't run if analysis is already in progress
    if analysis_running:
        return redirect(url_for('index'))
    
    # Get dataset from form
    dataset = request.form.get('dataset')
    if not dataset:
        # Default to samples_5000.csv
        dataset = 'samples_5000.csv'
    
    # Full path to dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', dataset)
    
    # Run analysis in a background thread
    analysis_running = True
    
    def run_agent():
        global analysis_running
        global last_analysis_result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            print("Starting security analysis...")
            agent = AzureOpenAISecurityAgent(dataset_path)
            last_analysis_result = loop.run_until_complete(agent.run_full_analysis())
            print("Analysis completed successfully")
            
        except Exception as e:
            print(f"ERROR in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            last_analysis_result = {'error': str(e)}
            
        finally:
            print("Setting analysis_running to False")
            analysis_running = False
            loop.close()
    
    import threading
    thread = threading.Thread(target=run_agent)
    thread.daemon = True
    thread.start()
    
    # Redirect back to index with notification
    return redirect(url_for('index'))

@app.route('/security-report')
def security_report():
    """Redirect to the visual report"""
    return redirect(url_for('view_report'))

@app.route('/mitigation')
def mitigation():
    """Redirect to the mitigation strategy view"""
    return redirect(url_for('view_strategy'))

@app.route('/analysis_status')
def analysis_status():
    """Get the status of the current analysis"""
    return jsonify({
        'running': analysis_running,
        'result': last_analysis_result
    })

@app.route('/reset_analysis_state', methods=['POST'])
def reset_analysis_state():
    """Force reset the analysis running state (useful if the app gets stuck)"""
    global analysis_running
    
    # Reset the analysis state
    analysis_running = False
    
    return jsonify({
        'status': 'success',
        'message': 'Analysis state has been reset'
    })

@app.route('/report/<path:filename>')
def report_file(filename):
    """Serve report files"""
    return send_from_directory(app.config['SECURITY_INSIGHTS'], filename)

@app.route('/visualizations/<path:filename>')
def visualization(filename):
    """Serve visualization files"""
    viz_dir = os.path.join(app.config['SECURITY_INSIGHTS'], 'visualizations')
    return send_from_directory(viz_dir, filename)

@app.route('/view_report')
def view_report():
    """View the full security report"""
    report_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'visual_report.html')
    
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_content = f.read()
        # Use the report.html template which has the Back to Dashboard button
        return render_template('report.html', content=report_content)
    else:
        return "No report available. Please run an analysis first."

@app.route('/view_summary_statistics')
def view_summary_statistics():
    """View the Summary Statistics section"""
    stats_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'summary_statistics.json')
    
    try:
        if os.path.exists(stats_path):
            import json
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                
            # Ensure stats dictionary has all required keys to avoid template errors
            default_keys = [
                'total_records', 'unique_source_ips', 'unique_destination_ips', 
                'unique_domains', 'alert_percentage', 'drop_percentage', 
                'avg_hit_count', 'high_risk_domains_count', 
                'top_source_networks', 'top_destination_networks', 'top_domain_tlds'
            ]
            
            # Initialize missing keys with default values
            for key in default_keys:
                if key not in stats:
                    if key.startswith('top_'):
                        stats[key] = {}
                    else:
                        stats[key] = 0
            
            # Map old variable names to new ones if needed
            # This handles both old files with top_drop_* and new files with top_hit_*
            if 'top_drop_domains' in stats and 'top_hit_domains' not in stats:
                stats['top_hit_domains'] = stats['top_drop_domains']
            
            if 'top_drop_sids' in stats and 'top_hit_sids' not in stats:
                stats['top_hit_sids'] = stats['top_drop_sids']
            
            if 'top_drop_source_ips' in stats and 'top_hit_source_ips' not in stats:
                stats['top_hit_source_ips'] = stats['top_drop_source_ips']
            
            return render_template('summary_statistics.html', statistics=stats)
        else:
            return "No statistics available. Please run an analysis first."
    except Exception as e:
        # Return a user-friendly error page with details
        error_message = f"Error loading statistics: {str(e)}"
        print(f"ERROR: {error_message}")
        return render_template('error.html', 
                              error_title="Statistics Error",
                              error_message=error_message)

@app.route('/view_key_findings')
def view_key_findings():
    """View the Key Findings and Recommendations section"""
    # First check for dedicated key_findings.md file
    key_findings_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'key_findings.md')
    
    if os.path.exists(key_findings_path):
        try:
            with open(key_findings_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Convert markdown to HTML
            try:
                import markdown
                html_content = markdown.markdown(markdown_content, extensions=['tables'])
                
                return render_template('report_section.html', 
                                      title="Key Findings and Recommendations",
                                      additional_content=html_content,
                                      explanation="This section summarizes the key security findings and recommended actions.")
            except ImportError:
                # Fallback if markdown package is not available
                return render_template('report_section.html', 
                                      title="Key Findings and Recommendations",
                                      additional_content=markdown_content,
                                      explanation="This section summarizes the key security findings and recommended actions. Install markdown package for better formatting.")
        except Exception as e:
            return render_template('report_section.html', 
                                  title="Key Findings and Recommendations",
                                  explanation=f"Error reading key findings file: {str(e)}",
                                  additional_content="Please check that the key_findings.md file exists and is readable.")
    else:
        return render_template('report_section.html', 
                              title="Key Findings and Recommendations",
                              explanation="Key findings file not found.",
                              additional_content="Please create a key_findings.md file in the security_insights directory.")

@app.route('/view_detailed_mitigations')
def view_detailed_mitigations():
    """View the Detailed Mitigation Strategies section"""
    # Path to the mitigation strategy file
    strategy_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'mitigation_strategy.md')
    
    if os.path.exists(strategy_path):
        try:
            # Read the mitigation strategy file
            with open(strategy_path, 'r', encoding='utf-8') as f:
                mitigation_content = f.read()
            
            # Convert markdown to HTML using the markdown library
            try:
                import markdown
                html_content = markdown.markdown(mitigation_content, extensions=['tables'])
                
                return render_template('report_section.html', 
                                      title="Detailed Mitigation Strategies",
                                      additional_content=html_content,
                                      explanation="This section details specific mitigation strategies to address the security issues identified.")
            except ImportError:
                # Fallback if markdown library is not available
                return render_template('report_section.html', 
                                      title="Detailed Mitigation Strategies",
                                      additional_content=mitigation_content,
                                      explanation="This section details specific mitigation strategies. Install markdown package for better formatting.")
        except Exception as e:
            return render_template('report_section.html', 
                                  title="Detailed Mitigation Strategies",
                                  explanation=f"Error reading mitigation strategy file: {str(e)}",
                                  additional_content="Please check that the mitigation_strategy.md file exists and is readable.")
    else:
        return render_template('report_section.html', 
                              title="Detailed Mitigation Strategies",
                              explanation="Mitigation strategy file not found.",
                              additional_content="Please create a mitigation_strategy.md file in the security_insights directory.")

@app.route('/view_full_report')
def view_full_report():
    """View the original complete visual report"""
    report_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'visual_report.html')
    
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_content = f.read()
        # Use the report.html template which has the Back to Dashboard button
        return render_template('report.html', content=report_content)
    else:
        return "No report available. Please run an analysis first."

@app.route('/view_strategy')
def view_strategy():
    """View the mitigation strategy"""
    strategy_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'mitigation_strategy.md')
    
    if os.path.exists(strategy_path):
        with open(strategy_path, 'r') as f:
            strategy_content = f.read()
        
        # Convert markdown to HTML using markdown library
        try:
            import markdown
            html_content = markdown.markdown(strategy_content)
        except ImportError:
            # If markdown library is not available, provide a message
            html_content = f"""
            <p>To properly render markdown content, please install the markdown package:</p>
            <pre>pip install markdown</pre>
            <hr>
            <pre>{strategy_content}</pre>
            """
        
        return render_template('markdown.html', content=html_content)
    else:
        return "No mitigation strategy available. Please run an analysis first."

@app.route('/visualizations')
def view_visualizations():
    """View all visualizations"""
    viz_dir = os.path.join(app.config['SECURITY_INSIGHTS'], 'visualizations')
    
    if os.path.exists(viz_dir):
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        viz_with_explanation = []
        
        for viz in viz_files:
            # Check for explanation file
            explanation_file = viz.replace('.png', '_explanation.txt')
            explanation = ""
            if os.path.exists(os.path.join(viz_dir, explanation_file)):
                with open(os.path.join(viz_dir, explanation_file), 'r') as f:
                    explanation = f.read()
            
            viz_with_explanation.append({
                'file': viz,
                'explanation': explanation
            })
            
        return render_template('visualizations.html', 
                              visualizations=viz_with_explanation)
    else:
        return "No visualizations available. Please run an analysis first."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)