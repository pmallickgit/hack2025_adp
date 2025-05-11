#!/usr/bin/env python3
"""
Azure OpenAI Security Agent

This agent analyzes security attack logs to draw conclusions, identify attack patterns,
and provide AI-driven mitigation strategies using Azure OpenAI and supervised machine learning.
"""

# Configure Matplotlib backend before importing it
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues on macOS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import asyncio
from datetime import datetime
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import Azure OpenAI wrapper
from azure_wrapper import init_azure_openai

# Set custom color palette using the specified colors
PRIMARY_COLOR = "#1d5532"  # dark green
SECONDARY_COLOR = "#111a20"  # dark
custom_palette = [PRIMARY_COLOR, SECONDARY_COLOR]

# Set visualization style with custom colors
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
sns.set_palette(custom_palette)

# Helper function to apply consistent styling to visualizations
def apply_consistent_styling(ax, title=None, xlabel=None, ylabel=None, alternating_colors=True):
    """
    Apply consistent styling to matplotlib axes.
    
    Args:
        ax: The matplotlib axes object to style
        title: Optional title for the plot
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        alternating_colors: Whether to alternate colors for bars in bar plots
    """
    # Set title and labels if provided
    if title:
        ax.set_title(title, color=PRIMARY_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Remove unnecessary spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Apply alternating colors for bar plots if requested
    if alternating_colors and hasattr(ax, 'patches') and len(ax.patches) > 0:
        for i, bar in enumerate(ax.patches):
            bar.set_color(PRIMARY_COLOR if i % 2 == 0 else SECONDARY_COLOR)
    
    return ax

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def extract_tld(domain):
    """
    Extract top-level domain (TLD) from a domain string.
    
    Args:
        domain (str): A domain name string
        
    Returns:
        str: The top-level domain (e.g., 'com', 'net', 'org', etc.)
    """
    if not domain or not isinstance(domain, str):
        return ''
    
    # Clean up the domain string
    domain = domain.strip().lower()
    
    # Filter out non-domain values
    if domain == '' or domain == 'unknown' or domain == 'nan':
        return ''
    
    try:
        # Split by dots and get the last part
        parts = domain.split('.')
        # Return the last part if there are at least 2 parts
        if len(parts) >= 2:
            tld = parts[-1]
            # Simple validation - TLD should be at least 2 chars
            if len(tld) >= 2:
                return tld
    except Exception as e:
        print(f"Error extracting TLD from domain '{domain}': {e}")
    
    return ''

class AzureOpenAISecurityAgent:
    """
    Security Agent that leverages Azure OpenAI and machine learning to analyze
    attack logs and provide mitigation strategies.
    """
    
    def __init__(self, log_file, output_dir="security_insights"):
        """
        Initialize the security agent.
        
        Args:
            log_file (str): Path to the attack log CSV file
            output_dir (str): Directory to save analysis results
        """
        self.log_file = log_file
        self.output_dir = output_dir
        
        # Create output directory structure
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize data structures
        self.raw_df = None
        self.processed_df = None
        self.ml_model = None
        self.model_accuracy = None
        self.feature_importances = None
        
        # Initialize Azure OpenAI
        try:
            print("Initializing Azure OpenAI...")
            self.openai_client, self.openai_deployment = init_azure_openai()
            self.has_openai = True
            print(f"Azure OpenAI initialized successfully with deployment: {self.openai_deployment}")
        except Exception as e:
            print(f"Warning: Could not initialize Azure OpenAI: {e}")
            self.openai_client = None
            self.openai_deployment = None
            self.has_openai = False
            
        print(f"Azure OpenAI Security Agent initialized")
        print(f"Log file: {log_file}")
        print(f"Output directory: {output_dir}")
    
    def load_data(self):
        """
        Load and parse the attack log CSV file.
        Support both custom log format and standard CSV files.
        """
        print("\nLoading attack log data...")
        
        try:
            # Check if the file exists
            if not os.path.exists(self.log_file):
                print(f"Error: Log file {self.log_file} does not exist")
                return None

            # Read the entire file
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Initialize data structures
            entries = []
            current_timestamp = {
                'epoch': int(time.time()),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Check for header
            if lines and "SRC_IP" in lines[0] and "SID" in lines[0]:
                # Skip header line
                lines = lines[1:]
            
            # Process each line
            for line in lines:
                line = line.strip()
                
                if not line or line.startswith("#"):
                    continue
                    
                if line.startswith("TIME "):
                    # Parse timestamp line
                    parts = line.split(" ", 3)
                    if len(parts) >= 3:
                        epoch_time = int(parts[1])
                        datetime_str = parts[2] + (" " + parts[3] if len(parts) > 3 else "")
                        current_timestamp = {
                            'epoch': epoch_time,
                            'datetime': datetime_str
                        }
                elif " " in line:
                    # Parse data entry line
                    try:
                        parts = line.split()
                        if len(parts) >= 7:
                            sid = int(parts[0]) if parts[0].isdigit() else 0
                            src_ip = parts[1]
                            src_port = int(parts[2]) if parts[2].isdigit() else 0
                            dest_ip = parts[3]
                            dest_port = int(parts[4]) if parts[4].isdigit() else 0
                            action = parts[5]
                            hit_count = int(parts[6]) if parts[6].isdigit() else 0
                            domain = parts[7] if len(parts) > 7 else ""
                            
                            # Add the entry with current timestamp
                            entry = {
                                'timestamp': current_timestamp['epoch'],
                                'datetime': current_timestamp['datetime'],
                                'sid': sid,
                                'src_ip': src_ip,
                                'src_port': src_port,
                                'dest_ip': dest_ip,
                                'dest_port': dest_port,
                                'action': action,
                                'hit_count': hit_count,
                                'domain': domain
                            }
                            entries.append(entry)
                    except Exception as e:
                        print(f"Skipping malformed line: {line[:50]}... ({str(e)})")
                        continue

            # Create DataFrame from entries
            if entries:
                self.raw_df = pd.DataFrame(entries)
                print(f"Successfully loaded {len(self.raw_df)} records using custom parser")
                return self.raw_df
            
            # If all parsing methods failed, log the error
            if self.raw_df is None or len(self.raw_df) == 0:
                print("All parsing methods failed. Creating empty DataFrame.")
                self.raw_df = pd.DataFrame(columns=['timestamp', 'datetime', 'sid', 'src_ip', 
                                                'src_port', 'dest_ip', 'dest_port', 
                                                'action', 'hit_count', 'domain'])
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'datetime', 'src_ip', 'dest_ip', 'action']
            for col in required_columns:
                if col not in self.raw_df.columns:
                    self.raw_df[col] = None
                    print(f"Warning: Column '{col}' was not found and has been added with null values")
            
            # Print summary statistics
            print(f"Data loaded successfully:")
            print(f"  - Total records: {len(self.raw_df)}")
            
            return self.raw_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            
            # If all else fails, create a minimal DataFrame
            print("Creating empty DataFrame after parsing failure")
            self.raw_df = pd.DataFrame(columns=['timestamp', 'datetime', 'sid', 'src_ip', 
                                            'src_port', 'dest_ip', 'dest_port', 
                                            'action', 'hit_count', 'domain'])
            return self.raw_df
    
    def extract_features(self):
        """Extract relevant features from the raw log data for analysis."""
        print("\nExtracting features from log data...")
        
        # Create a copy of the raw DataFrame to avoid modifying the original
        df = self.raw_df.copy() if self.raw_df is not None else pd.DataFrame()
        
        # Check if the DataFrame has records before continuing
        if df.empty:
            print("Warning: The dataset is empty. Cannot extract features from empty dataset.")
            # Create a minimal DataFrame with required columns to allow the pipeline to continue
            df = pd.DataFrame({
                'timestamp': [int(time.time())],
                'datetime': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'src_ip': ['0.0.0.0'],
                'dest_ip': ['0.0.0.0'],
                'src_port': [0],
                'dest_port': [0],
                'action': ['unknown'],
                'sid': [0],
                'domain': [''],
                'hit_count': [0]
            })
            print("Created a placeholder record to allow further processing.")
        else:
            print(f"Processing {len(df)} log entries for feature extraction")
        
        # Ensure all required columns exist (silently add missing columns)
        required_columns = ['timestamp', 'datetime', 'src_ip', 'dest_ip', 'action', 'sid', 'domain', 'hit_count']
        for col in required_columns:
            if col not in df.columns:
                if col in ['timestamp', 'hit_count', 'sid', 'src_port', 'dest_port']:
                    df[col] = 0
                else:
                    df[col] = ''
        
        # Convert timestamp to datetime if needed
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                print("Added datetime column from timestamp")
            except:
                df['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("Failed to convert timestamp to datetime, using current time")
                
        # Convert string columns to strings (in case they're loaded as other types)
        for col in ['src_ip', 'dest_ip', 'action', 'domain']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Convert numeric columns to integers with error handling
        for col in ['hit_count', 'sid', 'src_port', 'dest_port']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Additional feature extraction
        try:
            # Extract TLD from domain
            if 'domain' in df.columns:
                df['domain_tld'] = df['domain'].apply(lambda x: extract_tld(x) if isinstance(x, str) and x else '')
            
            # Parse datetime into components if it exists
            if 'datetime' in df.columns:
                df['parsed_datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                df['date'] = df['parsed_datetime'].dt.date
                df['hour'] = df['parsed_datetime'].dt.hour
                df['day_of_week'] = df['parsed_datetime'].dt.day_name()
                
                # Drop temporary column
                df = df.drop('parsed_datetime', axis=1)
        
            # Calculate source network (first 3 octets of IP)
            if 'src_ip' in df.columns:
                df['src_network'] = df['src_ip'].apply(lambda x: '.'.join(str(x).split('.')[:3]) + '.0/24' 
                                                     if isinstance(x, str) and x != 'nan' else 'unknown')
                
            print(f"Feature extraction completed successfully. Final feature count: {len(df.columns)}")
            
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
        
        self.df = df
        return df
    
    async def ask_natural_language_query(self, query, data_context=None):
        """
        Ask a natural language question about the security data using Azure OpenAI.
        
        Args:
            query (str): The natural language query about the security data
            data_context (str, optional): Additional context to include with the query
        
        Returns:
            dict: A dictionary containing the response and any relevant data
        """
        print(f"Processing natural language query: {query}")
        
        if not self.has_openai or not self.openai_client:
            return {
                "status": "error",
                "error": "Azure OpenAI client not available. Please configure your Azure OpenAI credentials.",
                "response": "Unable to process query: Azure OpenAI not configured."
            }
        
        try:
            # Build context from our data
            context = "Security log analysis data:\n"
            
            # Add statistics if available
            stats_path = os.path.join(self.output_dir, 'summary_statistics.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats_data = json.load(f)
                    context += f"STATISTICS: {json.dumps(stats_data, indent=2)}\n\n"
            
            # Add data sample from our dataframe
            if self.raw_df is not None and len(self.raw_df) > 0:
                # Add dataframe info
                context += f"DATASET INFO:\n"
                context += f"- Total records: {len(self.raw_df)}\n"
                context += f"- Columns: {', '.join(self.raw_df.columns)}\n"
                
                # Add sample data (first few rows)
                context += f"\nDATA SAMPLE (first 5 rows):\n"
                context += self.raw_df.head(5).to_string() + "\n\n"
                
                # Add key aggregate metrics
                context += f"KEY METRICS:\n"
                
                # Count by action type
                if 'action' in self.raw_df.columns:
                    action_counts = self.raw_df['action'].value_counts().to_dict()
                    context += f"- Actions breakdown: {action_counts}\n"
                
                # Top source IPs
                if 'src_ip' in self.raw_df.columns:
                    top_ips = self.raw_df['src_ip'].value_counts().head(5).to_dict()
                    context += f"- Top source IPs: {top_ips}\n"
                
                # Top SIDs if available
                if 'sid' in self.raw_df.columns:
                    top_sids = self.raw_df['sid'].value_counts().head(5).to_dict()
                    context += f"- Top security IDs (SIDs): {top_sids}\n"
            
            # Add user-provided context if available
            if data_context:
                context += f"\nUSER-PROVIDED CONTEXT:\n{data_context}\n"
            
            # Prepare system message for the OpenAI API
            system_message = """You are an expert cybersecurity analyst specializing in threat detection 
and analysis. You're assisting with analyzing security log data. Provide concise, 
accurate answers based on the provided data. If the query cannot be answered with 
the given data, clearly state that and suggest what additional information would 
be needed."""
            
            # Prepare the user message with query and context
            user_message = f"Based on the following security data, please answer this question:\n\nQUESTION: {query}\n\nCONTEXT:\n{context}\n\nPlease provide a clear, concise answer using only the data provided. If the data is insufficient to answer the question completely, state so and suggest what additional data would be helpful."
            
            # Call OpenAI API to process the query
            response = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            # Extract the response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                answer = response.choices[0].message.content
                return {
                    "status": "success",
                    "response": answer,
                    "query": query
                }
            else:
                return {
                    "status": "error",
                    "error": "Received empty response from Azure OpenAI",
                    "query": query,
                    "response": "Unable to process query: No response received."
                }
                
        except Exception as e:
            print(f"Error processing natural language query: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "response": f"Error processing query: {str(e)}"
            }
    
    async def generate_openai_analysis(self):
        """
        Generate a comprehensive security analysis using Azure OpenAI.
        
        This method analyzes the security logs using Azure OpenAI to identify:
        - Attack patterns and trends
        - Potential threat actors
        - Vulnerability exploits
        - Recommended security measures
        
        Returns:
            str: Path to the generated analysis file
        """
        print("\nGenerating AI-powered security analysis...")
        
        if not self.has_openai or not self.openai_client:
            print("Azure OpenAI is not available. Skipping AI security analysis.")
            return None
        
        try:
            # Prepare the output file path
            output_file = os.path.join(self.output_dir, "openai_analysis.md")
            
            # Build context from our data
            context = "## Security Log Analysis Context\n\n"
            
            # Add dataset information
            if self.raw_df is not None and len(self.raw_df) > 0:
                context += f"### Dataset Statistics\n"
                context += f"- Total log entries: {len(self.raw_df)}\n"
                context += f"- Time period: {self.raw_df['datetime'].min()} to {self.raw_df['datetime'].max()}\n"
                
                # Add action breakdown
                if 'action' in self.raw_df.columns:
                    action_counts = self.raw_df['action'].value_counts()
                    context += f"\n### Action Types\n"
                    for action, count in action_counts.items():
                        percentage = (count / len(self.raw_df)) * 100
                        context += f"- {action}: {count} ({percentage:.1f}%)\n"
                
                # Add top source IPs
                if 'src_ip' in self.raw_df.columns:
                    top_ips = self.raw_df['src_ip'].value_counts().head(10)
                    context += f"\n### Top Source IPs\n"
                    for ip, count in top_ips.items():
                        percentage = (count / len(self.raw_df)) * 100
                        context += f"- {ip}: {count} attacks ({percentage:.1f}%)\n"
                
                # Add top security IDs if available
                if 'sid' in self.raw_df.columns:
                    top_sids = self.raw_df['sid'].value_counts().head(10)
                    context += f"\n### Top Security IDs (SIDs)\n"
                    for sid, count in top_sids.items():
                        percentage = (count / len(self.raw_df)) * 100
                        context += f"- SID {sid}: {count} occurrences ({percentage:.1f}%)\n"
                        
                # Add domain information if available
                if 'domain' in self.raw_df.columns and self.raw_df['domain'].notna().sum() > 0:
                    top_domains = self.raw_df['domain'].value_counts().head(10)
                    context += f"\n### Top Domains\n"
                    for domain, count in top_domains.items():
                        if domain and domain != 'nan' and domain != '':
                            percentage = (count / len(self.raw_df)) * 100
                            context += f"- {domain}: {count} occurrences ({percentage:.1f}%)\n"
                
                # Add time-based patterns
                if 'hour' in self.df.columns:
                    hour_counts = self.df['hour'].value_counts().sort_index()
                    peak_hour = hour_counts.idxmax()
                    context += f"\n### Temporal Patterns\n"
                    context += f"- Peak activity hour: {peak_hour}:00 ({hour_counts[peak_hour]} events)\n"
                    
                    if 'day_of_week' in self.df.columns:
                        dow_counts = self.df['day_of_week'].value_counts()
                        peak_day = dow_counts.idxmax()
                        context += f"- Most active day: {peak_day} ({dow_counts[peak_day]} events)\n"
            
            # Prepare system message for the OpenAI API
            system_message = """You are an expert cybersecurity analyst with deep knowledge of network 
security, threat intelligence, and attack patterns. Your task is to analyze security log data 
and provide comprehensive insights and actionable recommendations.

Your analysis should include:
1. Executive summary of key findings
2. Identified attack patterns and techniques
3. Assessment of the threat actors (based on tactics and targeting)
4. Most critical security concerns
5. Detailed technical analysis of significant threats
6. Recommended security measures

Format your response in Markdown with appropriate sections and bullet points.
"""
            
            # Prepare the user message with the security log context
            user_message = f"""Please analyze the following security log data and provide a comprehensive 
security assessment:

{context}

Based on this data, generate a thorough security analysis report identifying attack patterns, 
potential threat actors, vulnerability exploits, and recommended security measures.

Structure your analysis with clear section headings in Markdown format.
"""
            
            # Call OpenAI API to generate the analysis
            print("Calling Azure OpenAI for comprehensive security analysis...")
            response = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            # Extract the analysis from the response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                analysis = response.choices[0].message.content
                
                # Prepend a title and timestamp
                full_analysis = f"# AI-Generated Security Analysis\n\n"
                full_analysis += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                full_analysis += f"**Dataset:** {os.path.basename(self.log_file)}\n\n"
                full_analysis += analysis
                
                # Save the analysis to a markdown file
                with open(output_file, 'w') as f:
                    f.write(full_analysis)
                
                print(f"Security analysis generated and saved to {output_file}")
                return output_file
            else:
                print("Failed to generate security analysis: Empty response from Azure OpenAI")
                return None
                
        except Exception as e:
            print(f"Error generating AI security analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def generate_mitigation_strategy(self):
        """
        Generate a mitigation strategy using Azure OpenAI based on the security analysis.
        
        This method creates specific, actionable recommendations to address the identified
        security threats and vulnerabilities in the analyzed security logs.
        
        Returns:
            str: Path to the generated mitigation strategy file
        """
        print("\nGenerating AI-powered mitigation strategies...")
        
        if not self.has_openai or not self.openai_client:
            print("Azure OpenAI is not available. Skipping mitigation strategy generation.")
            return None
        
        try:
            # Prepare the output file path
            output_file = os.path.join(self.output_dir, "mitigation_strategy.md")
            
            # First, check if we have existing analysis to reference
            openai_analysis_path = os.path.join(self.output_dir, "openai_analysis.md")
            existing_analysis = ""
            if os.path.exists(openai_analysis_path):
                with open(openai_analysis_path, 'r') as f:
                    existing_analysis = f.read()
            
            # Build context from our data
            context = "## Security Log Analysis Context\n\n"
            
            # Add dataset information
            if self.raw_df is not None and len(self.raw_df) > 0:
                context += f"### Dataset Statistics\n"
                context += f"- Total log entries: {len(self.raw_df)}\n"
                
                # Add action breakdown
                if 'action' in self.raw_df.columns:
                    action_counts = self.raw_df['action'].value_counts()
                    context += f"\n### Security Actions\n"
                    for action, count in action_counts.items():
                        percentage = (count / len(self.raw_df)) * 100
                        context += f"- {action}: {count} ({percentage:.1f}%)\n"
                
                # Add top source IPs
                if 'src_ip' in self.raw_df.columns:
                    top_ips = self.raw_df['src_ip'].value_counts().head(5)
                    context += f"\n### Top Threat Sources\n"
                    for ip, count in top_ips.items():
                        percentage = (count / len(self.raw_df)) * 100
                        context += f"- {ip}: {count} events ({percentage:.1f}%)\n"
                
                # Add top security IDs and their meaning if available
                if 'sid' in self.raw_df.columns:
                    top_sids = self.raw_df['sid'].value_counts().head(5)
                    context += f"\n### Top Alert Types (SIDs)\n"
                    for sid, count in top_sids.items():
                        percentage = (count / len(self.raw_df)) * 100
                        context += f"- SID {sid}: {count} occurrences ({percentage:.1f}%)\n"
            
            # Prepare system message for the OpenAI API
            system_message = """You are an expert cybersecurity consultant specializing in security 
architecture, incident response, and threat mitigation. Your task is to develop a 
comprehensive mitigation strategy based on security log analysis.

Your mitigation strategy should include:
1. Executive summary of recommended actions
2. Immediate mitigation steps (prioritized by urgency)
3. Medium-term security enhancements
4. Long-term security architecture improvements
5. Specific technical recommendations (rules, configurations, tools)
6. Security awareness and training recommendations
7. Monitoring and detection strategy improvements

Format your response in Markdown with appropriate sections and bullet points. 
Be specific and actionable in your recommendations."""
            
            # Prepare the user message with the security context and existing analysis
            user_message = """Based on the following security analysis, please provide a 
comprehensive mitigation strategy:

"""

            # Add the context and existing analysis to the user message
            user_message += f"{context}\n\n"
            
            if existing_analysis:
                user_message += "## Previous Security Analysis\n\n" + existing_analysis + "\n\n"
            
            user_message += """Generate a detailed, practical mitigation strategy addressing all identified security 
concerns with specific, actionable recommendations. Include both immediate tactical 
steps and strategic security improvements.

Structure your recommendations with clear section headings in Markdown format.
"""
            
            # Call OpenAI API to generate the mitigation strategy
            print("Calling Azure OpenAI for security mitigation strategy...")
            response = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            # Extract the strategy from the response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                strategy = response.choices[0].message.content
                
                # Prepend a title and timestamp
                full_strategy = f"# Security Mitigation Strategy\n\n"
                full_strategy += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                full_strategy += f"**Dataset:** {os.path.basename(self.log_file)}\n\n"
                full_strategy += strategy
                
                # Save the strategy to a markdown file
                with open(output_file, 'w') as f:
                    f.write(full_strategy)
                
                print(f"Mitigation strategy generated and saved to {output_file}")
                return output_file
            else:
                print("Failed to generate mitigation strategy: Empty response from Azure OpenAI")
                return None
                
        except Exception as e:
            print(f"Error generating mitigation strategy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_full_analysis(self):
        """
        Run the complete security analysis pipeline.
        
        This method coordinates all analysis steps:
        1. Data loading and preprocessing
        2. Feature extraction
        3. Summary statistics generation
        4. Visualization creation
        5. OpenAI-powered insights
        6. Mitigation strategy generation
        
        Returns:
            dict: A summary of analysis results including paths to generated files
        """
        print("\n=== Starting Full Security Log Analysis ===")
        results = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_generated": []
        }
        
        try:
            # Step 1: Load and preprocess data
            print("\nStep 1: Loading and preprocessing data...")
            self.load_data()
            
            # Step 2: Extract features
            print("\nStep 2: Extracting features...")
            self.extract_features()
            
            # Step 3: Generate summary statistics
            # print("\nStep 3: Generating summary statistics...")
            # stats_file = self.generate_summary_statistics()
            # if stats_file:
            #     results["files_generated"].append(stats_file)
                
            # Step 4: Create visualizations
            # print("\nStep 4: Creating visualizations...")
            # viz_files = self.create_all_visualizations()
            # if viz_files:
            #     results["files_generated"].extend(viz_files)
                
            # Step 5: Generate OpenAI analysis if available
            if self.openai_client:
                print("\nStep 5: Generating OpenAI analysis...")
                analysis_file = await self.generate_openai_analysis()
                if analysis_file:
                    results["files_generated"].append(analysis_file)
            else:
                print("\nStep 5: Skipping OpenAI analysis (client not available)")
                
            # Step 6: Generate mitigation strategy
            print("\nStep 6: Generating mitigation strategy...")
            strategy_file = await self.generate_mitigation_strategy()
            if strategy_file:
                results["files_generated"].append(strategy_file)
                
            # Record completion time
            results["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results["status"] = "completed"
            
            # Save results summary
            summary_path = os.path.join(self.output_dir, "analysis_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
                
            print(f"\n=== Analysis completed successfully ===")
            print(f"Generated {len(results['files_generated'])} output files")
            
            return results
            
        except Exception as e:
            print(f"\nError during full analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Record error information
            results["status"] = "error"
            results["error"] = str(e)
            results["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save results even if there was an error
            summary_path = os.path.join(self.output_dir, "analysis_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            return results