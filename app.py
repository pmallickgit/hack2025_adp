#!/usr/bin/env python3
"""
Flask Application for Azure OpenAI Security Agent

This Flask app provides a web interface for the Azure OpenAI Security Agent,
allowing users to view reports and run analyses.
"""

import os
import asyncio
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, current_app
from azure_openai_security_agent import AzureOpenAISecurityAgent, NumpyEncoder
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['SECURITY_INSIGHTS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'security_insights')

# Create a custom jsonify function that uses the NumpyEncoder
def custom_jsonify(*args, **kwargs):
    """Custom jsonify function that handles NumPy types"""
    if args and kwargs:
        raise TypeError('jsonify() behavior undefined when passed both args and kwargs')
    elif len(args) == 1:
        data = args[0]
    else:
        data = args or kwargs

    # Get the JSONIFY_MIMETYPE or use application/json as default
    json_mimetype = current_app.config.get('JSONIFY_MIMETYPE', 'application/json')
    
    return current_app.response_class(
        json.dumps(data, cls=NumpyEncoder) + '\n',
        mimetype=json_mimetype
    )

# Override the standard jsonify
app.jsonify = custom_jsonify

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
    # Convert any NumPy types in the result to standard Python types
    # This is needed because the custom jsonify might not be applied in all cases
    result = last_analysis_result
    if result:
        # If result is a dict, ensure it's using standard Python types
        if isinstance(result, dict):
            # Use json.dumps/loads to convert NumPy types to standard Python types
            result = json.loads(json.dumps(result, cls=NumpyEncoder))
    
    return custom_jsonify({
        'running': analysis_running,
        'result': result
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

@app.route('/view_openai_analysis')
def view_openai_analysis():
    """View the Azure OpenAI Analysis report"""
    # Check for existing analysis results
    stats_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'summary_statistics.json')
    analysis_file_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'openai_analysis.md')
    has_data = os.path.exists(stats_path)
    has_analysis_file = os.path.exists(analysis_file_path)
    
    # Default values
    dataset_name = "Unknown dataset"
    sample_size = "0"
    model_deployment = "Unknown"
    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    analysis_text = ""
    
    print(f"Debug: Loading OpenAI analysis, file exists: {has_analysis_file}")
    
    # First try to load from the saved file
    if has_analysis_file:
        try:
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Debug: Loaded content length: {len(content)}")
                
            # Extract metadata and content from the markdown file
            # The file has format:
            # # Azure OpenAI Security Analysis
            # 
            # Analysis Date: YYYY-MM-DD HH:MM:SS
            # Records Analyzed: N
            # 
            # ---
            # 
            # [actual analysis content]
            # 
            # ---
            # 
            # Azure OpenAI security analysis completed successfully.
            # Estimated model quality: X.XXX
                
            # Extract analysis date
            date_match = re.search(r'Analysis Date: ([^\n]+)', content)
            if date_match:
                analysis_date = date_match.group(1)
                
            # Extract sample size
            size_match = re.search(r'Records Analyzed: ([^\n]+)', content)
            if size_match:
                sample_size = size_match.group(1)
                
            # Extract the main content between the first and second "---"
            content_parts = content.split('---')
            if len(content_parts) >= 3:
                analysis_text = content_parts[1].strip() + '\n\n' + content_parts[2].strip()
            else:
                # If we can't split properly, use everything after the first few lines
                header_lines = content.split('\n')[:5]  # Skip the first 5 lines (approx)
                analysis_text = '\n'.join(content.split('\n')[5:])
                
            print(f"Debug: Extracted analysis text length: {len(analysis_text)}")
            
        except Exception as e:
            print(f"Error reading analysis file: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # If no file exists or we couldn't extract the analysis, try the ml_model approach
    if not analysis_text and (last_analysis_result and 'log_file' in last_analysis_result):
        try:
            print("Debug: Trying to get analysis from agent's ML model")
            # Get the dataset name from the last analysis
            dataset_name = os.path.basename(last_analysis_result['log_file'])
            dataset_path = last_analysis_result['log_file']
            
            # Initialize the security agent
            agent = AzureOpenAISecurityAgent(dataset_path or ".")
            
            # Try to get the ML model data
            if hasattr(agent, 'ml_model') and agent.ml_model:
                if isinstance(agent.ml_model, dict):
                    if 'sample_size' in agent.ml_model:
                        sample_size = str(agent.ml_model['sample_size'])
                    if 'deployment' in agent.ml_model:
                        model_deployment = agent.ml_model['deployment']
                    if 'analysis' in agent.ml_model:
                        analysis_text = agent.ml_model['analysis']
                        print(f"Debug: Got analysis from ML model, length: {len(analysis_text)}")
                        
        except Exception as e:
            print(f"Error getting analysis from agent: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # If we still don't have a dataset name, try to get it from the last analysis
    if dataset_name == "Unknown dataset" and last_analysis_result and 'log_file' in last_analysis_result:
        dataset_name = os.path.basename(last_analysis_result['log_file'])
                    
    # Format the analysis text with HTML
    try:
        import markdown
        
        # Only proceed if we have analysis text
        if analysis_text:
            print("Debug: Converting markdown to HTML")
            analysis_html = markdown.markdown(analysis_text, extensions=['tables'])
            
            # Add some additional formatting for key sections
            analysis_html = analysis_html.replace('<h1>', '<h3 class="mt-4">')
            analysis_html = analysis_html.replace('</h1>', '</h3>')
            analysis_html = analysis_html.replace('<h2>', '<h4 class="mt-4">')
            analysis_html = analysis_html.replace('</h2>', '</h4>')
            analysis_html = analysis_html.replace('<h3>', '<h5 class="mt-3">')
            analysis_html = analysis_html.replace('</h3>', '</h5>')
            
            # Highlight important sections
            patterns = ["Key Patterns", "Important Features", "Anomalous Patterns", 
                        "Top 5", "Recommendations", "Patterns that distinguish", "Findings",
                        "Summary Table", "Structured Analysis"]
            
            for pattern in patterns:
                analysis_html = analysis_html.replace(
                    f'<h3 class="mt-4">{pattern}',
                    f'<h3 class="mt-4 analysis-highlight">{pattern}'
                )
                analysis_html = analysis_html.replace(
                    f'<h4 class="mt-4">{pattern}',
                    f'<h4 class="mt-4 analysis-highlight">{pattern}'
                )
                analysis_html = analysis_html.replace(
                    f'<h5 class="mt-3">{pattern}',
                    f'<h5 class="mt-3 analysis-highlight">{pattern}'
                )
                
            # Ensure tables are properly styled
            analysis_html = analysis_html.replace('<table>', '<table class="table table-striped table-bordered">')
            
            print(f"Debug: Final HTML length: {len(analysis_html)}")
        else:
            analysis_html = "<div class='alert alert-warning'>No OpenAI analysis available. Run a security analysis to generate AI insights.</div>"
            print("Debug: No analysis text found")
            
    except ImportError:
        # Fallback if markdown is not available
        analysis_html = analysis_text.replace('\n', '<br>') if analysis_text else "<div class='alert alert-warning'>No analysis available.</div>"
        print("Debug: Fallback to simple HTML conversion")
        
    return render_template('openai_analysis.html',
                          analysis_date=analysis_date,
                          dataset_name=dataset_name,
                          sample_size=sample_size,
                          model_deployment=model_deployment,
                          analysis_text=analysis_text,
                          analysis_html=analysis_html)

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

@app.route('/query')
def query_interface():
    """Show the natural language query interface for security data"""
    # Check if we have processed security data available
    stats_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'summary_statistics.json')
    has_data = os.path.exists(stats_path)
    
    return render_template('query_interface.html', has_data=has_data)

@app.route('/query/ask', methods=['POST'])
def process_query():
    """Process a natural language query about security data"""
    try:
        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No query provided'
            })
        
        query = data['query']
        
        # Find the log file used for the last analysis
        dataset_path = None
        last_log_file = None
        
        if last_analysis_result and 'log_file' in last_analysis_result:
            last_log_file = last_analysis_result['log_file']
            if os.path.exists(last_log_file):
                dataset_path = last_log_file
        
        # If no specific log file is known, try to use a default dataset
        if not dataset_path:
            datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
            if os.path.exists(os.path.join(datasets_dir, 'dense_security_logs.csv')):
                dataset_path = os.path.join(datasets_dir, 'dense_security_logs.csv')
            elif os.path.exists(os.path.join(datasets_dir, 'samples_5000.csv')):
                dataset_path = os.path.join(datasets_dir, 'samples_5000.csv')
        
        if not dataset_path:
            return jsonify({
                'status': 'error',
                'message': 'No security data available for querying. Please run an analysis first.'
            })
        
        # Initialize the security agent
        agent = AzureOpenAISecurityAgent(dataset_path)
        
        # Load existing data if available
        stats_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'summary_statistics.json')
        if os.path.exists(stats_path):
            try:
                # Load processed data
                agent.load_data()
                agent.processed_df = agent.extract_features()
                
                # Create event loop and run the async query in it
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(agent.ask_natural_language_query(query))
                loop.close()
                
                if response['status'] == 'success':
                    return jsonify({
                        'status': 'success',
                        'query': query,
                        'answer': response['response']  # Changed 'answer' to 'response' to match the key in the agent's response
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': response['message']
                    })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    'status': 'error',
                    'message': f"Error processing query: {str(e)}"
                })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No processed security data available. Please run a security analysis first.'
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/query/ip_time_series', methods=['POST'])
def generate_ip_time_series():
    """Generate a time series visualization for a specific IP address"""
    try:
        # Get IP address from request
        data = request.get_json()
        if not data or 'ip_address' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No IP address provided'
            })
        
        ip_address = data['ip_address']
        print(f"Generating time series for IP: {ip_address}")
        
        # Find the log file used for the last analysis
        dataset_path = None
        if last_analysis_result and 'log_file' in last_analysis_result:
            dataset_path = last_analysis_result['log_file']
        
        if not dataset_path:
            # Try to use a default dataset
            datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
            if os.path.exists(os.path.join(datasets_dir, 'dense_security_logs.csv')):
                dataset_path = os.path.join(datasets_dir, 'dense_security_logs.csv')
            elif os.path.exists(os.path.join(datasets_dir, 'samples_5000.csv')):
                dataset_path = os.path.join(datasets_dir, 'samples_5000.csv')
        
        if not dataset_path:
            return jsonify({
                'status': 'error',
                'message': 'No security data available for analysis. Please run an analysis first.'
            })
        
        # Initialize the security agent
        agent = AzureOpenAISecurityAgent(dataset_path)
        
        # Load the data if needed
        if not hasattr(agent, 'processed_df') or agent.processed_df is None:
            agent.load_data()
            agent.processed_df = agent.extract_features()
        
        # Generate time series visualization
        viz_path = agent.generate_ip_time_series(ip_address)
        
        if viz_path:
            # Get the relative path for the URL
            viz_url = os.path.basename(viz_path)
            explanation_path = viz_path.replace('.png', '_explanation.txt')
            explanation = ""
            
            if os.path.exists(explanation_path):
                with open(explanation_path, 'r') as f:
                    explanation = f.read()
            
            return jsonify({
                'status': 'success',
                'ip_address': ip_address,
                'visualization_url': f"/visualizations/{viz_url}",
                'explanation': explanation
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f"No data available for IP {ip_address} or failed to generate visualization"
            })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Error generating time series: {str(e)}"
        })

@app.route('/view_ip_time_series/<ip_address>')
def view_ip_time_series(ip_address):
    """View a time series visualization for a specific IP address"""
    # Clean the IP address to create the filename
    safe_ip = ip_address.replace('.', '_')
    viz_file = f"ip_time_series_{safe_ip}.png"
    viz_path = os.path.join(app.config['SECURITY_INSIGHTS'], 'visualizations', viz_file)
    explanation_path = viz_path.replace('.png', '_explanation.txt')
    
    # Check if the visualization exists
    if not os.path.exists(viz_path):
        # Try to generate it
        try:
            return redirect(url_for('generate_ip_time_series_view', ip_address=ip_address))
        except:
            return f"No time series visualization found for IP {ip_address}. Please generate it first."
    
    # Get the explanation if available
    explanation = ""
    if os.path.exists(explanation_path):
        with open(explanation_path, 'r') as f:
            explanation = f.read()
    
    # Render the visualization in a template
    return render_template('ip_time_series.html',
                          ip_address=ip_address,
                          visualization_url=f"/visualizations/{viz_file}",
                          explanation=explanation)

@app.route('/generate_ip_time_series_view/<ip_address>')
def generate_ip_time_series_view(ip_address):
    """Generate and view a time series visualization for a specific IP address"""
    # Find the log file used for the last analysis
    dataset_path = None
    if last_analysis_result and 'log_file' in last_analysis_result:
        dataset_path = last_analysis_result['log_file']
    
    if not dataset_path:
        # Try to use a default dataset
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        if os.path.exists(os.path.join(datasets_dir, 'dense_security_logs.csv')):
            dataset_path = os.path.join(datasets_dir, 'dense_security_logs.csv')
        elif os.path.exists(os.path.join(datasets_dir, 'samples_5000.csv')):
            dataset_path = os.path.join(datasets_dir, 'samples_5000.csv')
    
    if not dataset_path:
        return "No security data available for analysis. Please run an analysis first."
    
    # Initialize the security agent
    agent = AzureOpenAISecurityAgent(dataset_path)
    
    # Load the data if needed
    if not hasattr(agent, 'processed_df') or agent.processed_df is None:
        agent.load_data()
        agent.processed_df = agent.extract_features()
    
    # Generate time series visualization
    viz_path = agent.generate_ip_time_series(ip_address)
    
    if viz_path:
        # Redirect to the view page
        return redirect(url_for('view_ip_time_series', ip_address=ip_address))
    else:
        return f"Failed to generate time series visualization for IP {ip_address}. No data available for this IP."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)