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

@app.route('/')
def index():
    """Render the main dashboard page"""
    # Check if security_insights directory has files
    has_insights = False
    insight_files = []
    statistics = None
    
    if os.path.exists(app.config['SECURITY_INSIGHTS']):
        files = os.listdir(app.config['SECURITY_INSIGHTS'])
        has_insights = len(files) > 0
        # List important files
        has_security_report = 'visual_report.html' in files
        has_mitigation = 'mitigation_strategy.md' in files
        has_visualizations = os.path.exists(os.path.join(app.config['SECURITY_INSIGHTS'], 'visualizations'))
        
        # Get statistics if available
        if 'summary_statistics.json' in files:
            import json
            with open(os.path.join(app.config['SECURITY_INSIGHTS'], 'summary_statistics.json'), 'r') as f:
                statistics = json.load(f)
    else:
        has_security_report = False
        has_mitigation = False
        has_visualizations = False
    
    # Get available datasets
    available_datasets = []
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    if os.path.exists(datasets_dir):
        available_datasets = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    
    return render_template('index.html', 
                          has_security_report=has_security_report,
                          has_mitigation=has_mitigation,
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
            dataset = 'samples_5000.csv'
        
        # Full path to dataset
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', dataset)
        
        # Run analysis in a background thread
        def run_agent():
            global analysis_running
            global last_analysis_result
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                agent = AzureOpenAISecurityAgent(dataset_path)
                last_analysis_result = loop.run_until_complete(agent.run_full_analysis())
                
            except Exception as e:
                last_analysis_result = {'error': str(e)}
                
            finally:
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
            agent = AzureOpenAISecurityAgent(dataset_path)
            last_analysis_result = loop.run_until_complete(agent.run_full_analysis())
            
        except Exception as e:
            last_analysis_result = {'error': str(e)}
            
        finally:
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
    """Redirect to the mitigation strategy"""
    return redirect(url_for('view_strategy'))

@app.route('/analysis_status')
def analysis_status():
    """Get the status of the current analysis"""
    return jsonify({
        'running': analysis_running,
        'result': last_analysis_result
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
    """View the main visual report"""
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
    app.run(debug=True, host='0.0.0.0', port=5000)