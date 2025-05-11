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

            # First try standard CSV format
            # try:
            #     print("Attempting to load as standard CSV...")
            #     self.raw_df = pd.read_csv(self.log_file)
            #     if len(self.raw_df) > 0:
            #         print(f"Successfully loaded {len(self.raw_df)} records as standard CSV")
            #         return self.raw_df
            # except Exception as e:
            #     print(f"Standard CSV parsing failed: {e}")
            
            # If standard CSV fails, try custom format
            # print("Trying custom log format...")
            
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
            
            # If both methods failed, try space-delimited format
            print("Trying space-delimited format...")
            try:
                self.raw_df = pd.read_csv(
                    self.log_file, 
                    sep=r'\s+', 
                    engine='python',
                    comment='#',
                    header=None,
                    names=['sid', 'src_ip', 'src_port', 'dest_ip', 'dest_port', 'action', 'hit_count', 'domain'],
                    skiprows=1, 
                    on_bad_lines='skip'
                )
                # Add timestamp columns
                self.raw_df['timestamp'] = int(time.time())
                self.raw_df['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Filter out TIME rows that might have been parsed as data
                self.raw_df = self.raw_df[~self.raw_df['src_ip'].str.contains('TIME', na=False)]
                
                print(f"Successfully loaded {len(self.raw_df)} records with space-delimited parser")
                return self.raw_df
            except Exception as e:
                print(f"Space-delimited parsing failed: {e}")
            
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
        
    def generate_action_distribution(self):
        """Generate visualization for action distribution."""
        plt.figure(figsize=(10, 6))
        action_counts = self.processed_df['action'].value_counts()
        ax = action_counts.plot(kind='barh', color=PRIMARY_COLOR)
        apply_consistent_styling(ax, title='Distribution of Actions in Security Events', xlabel='Count', ylabel='Action Type')
        
        plt.tight_layout()
        
        # Save both visualization and explanation
        os.makedirs('security_insights/visualizations', exist_ok=True)
        plt.savefig('security_insights/visualizations/01_action_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create explanation file
        explanation = "This chart shows the distribution of different actions observed in the security events. "
        explanation += "The most common actions may indicate patterns of attack or system behavior worth investigating further."
        
        with open('security_insights/visualizations/01_action_distribution_explanation.txt', 'w') as f:
            f.write(explanation)
            
        return "security_insights/visualizations/01_action_distribution.png"
    
    def generate_top_source_ips(self):
        """Generate visualization for top source IP addresses."""
        plt.figure(figsize=(10, 6))
        top_ips = self.processed_df.groupby('src_ip')['hit_count'].sum().sort_values(ascending=False).head(10)
        ax = top_ips.plot(kind='barh', color=PRIMARY_COLOR)
        apply_consistent_styling(ax, title='Top 10 Source IP Addresses by Hit Count', xlabel='Hit Count', ylabel='Source IP')
        
        plt.tight_layout()
        
        # Save both visualization and explanation
        os.makedirs('security_insights/visualizations', exist_ok=True)
        plt.savefig('security_insights/visualizations/02_top_source_ips.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create explanation file
        explanation = "This chart shows the top 10 source IP addresses with the highest hit counts in the security events. "
        explanation += "These IP addresses may represent potential threat actors or compromised systems attempting to "
        explanation += "access your resources repeatedly and should be investigated further."
        
        with open('security_insights/visualizations/02_top_source_ips_explanation.txt', 'w') as f:
            f.write(explanation)
            
        return "security_insights/visualizations/02_top_source_ips.png"
    
    async def run_full_analysis(self):
        """
        Run the complete security analysis pipeline.
        
        This method orchestrates all analysis steps:
        1. Data loading and preprocessing
        2. Feature extraction
        3. Visualization generation
        4. Insights extraction
        5. Report generation
        
        Returns:
            dict: Analysis results summary
        """
        start_time = time.time()
        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "log_file": self.log_file,
            "output_dir": self.output_dir
        }
        
        try:
            # Step 1: Load and process data
            print("\n=== Step 1: Loading and Processing Data ===")
            self.load_data()
            
            # Step 2: Extract features
            print("\n=== Step 2: Extracting Features ===")
            self.processed_df = self.extract_features()
            
            # Step 3: Generate visualizations
            print("\n=== Step 3: Generating Visualizations ===")
            self.generate_action_distribution()
            self.generate_top_source_ips()
            
            # Generate summary statistics
            print("\n=== Step 4: Generating Summary Statistics ===")
            stats = self.generate_summary_statistics()
            results["summary_statistics"] = stats
            
            # Add analysis time
            elapsed_time = time.time() - start_time
            results["analysis_time_seconds"] = elapsed_time
            print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics from processed data.
        Returns a dictionary of statistics.
        """
        if self.processed_df is None:
            print("No processed data available. Run extract_features() first.")
            return {}
        
        df = self.processed_df.copy()
        total_records = len(df)
        if 'action' not in df.columns:
            df['action'] = 'unknown'
        
        # Basic statistics
        alert_count = len(df[df['action'].str.lower() == 'alert'])
        drop_count = len(df[df['action'].str.lower() == 'drop'])
        
        # Calculate percentages
        alert_percentage = (alert_count / total_records * 100) if total_records > 0 else 0
        drop_percentage = (drop_count / total_records * 100) if total_records > 0 else 0
        
        # Get unique counts
        unique_source_ips = df['src_ip'].nunique()
        unique_destination_ips = df['dest_ip'].nunique()
        
        # Extract domains if they exist
        if 'domain' not in df.columns:
            df['domain'] = ''
        
        # Ensure domain_tld column exists and is populated
        if 'domain_tld' not in df.columns:
            # Extract TLD from domain for each row
            df['domain_tld'] = df['domain'].apply(extract_tld)
        
        # Ensure domain_risk_score column exists
        if 'domain_risk_score' not in df.columns:
            df['domain_risk_score'] = 0
        
        unique_domains = df[df['domain'] != '']['domain'].nunique()
        
        stats = {
            "total_records": total_records,
            "alert_count": alert_count,
            "drop_count": drop_count,
            "alert_percentage": alert_percentage,
            "drop_percentage": drop_percentage,
            "unique_source_ips": unique_source_ips,
            "unique_destination_ips": unique_destination_ips,
            "unique_domains": unique_domains,
            "avg_hit_count": df['hit_count'].mean(),
            "high_risk_domains_count": sum(df['domain_risk_score'] > 1)
        }
        
        # Get top hit sources
        top_hit_source_ips = df.groupby('src_ip')['hit_count'].sum().sort_values(ascending=False).head(10).to_dict()
        top_hit_sids = df.groupby('sid')['hit_count'].sum().sort_values(ascending=False).head(10).to_dict()
        top_hit_domains = df[df['domain'] != ''].groupby('domain')['hit_count'].sum().sort_values(ascending=False).head(10).to_dict()
        
        # Network statistics
        # Check if required columns exist for network statistics
        if 'src_network' not in df.columns:
            df['src_network'] = 'unknown'
        if 'dest_network' not in df.columns:
            df['dest_network'] = 'unknown'
        
        top_source_networks = df.groupby('src_network')['hit_count'].sum().sort_values(ascending=False).head(5).to_dict()
        top_destination_networks = df.groupby('dest_network')['hit_count'].sum().sort_values(ascending=False).head(5).to_dict()
        
        # Create top domain TLDs dictionary - only include non-empty TLDs
        top_domain_tlds = df[df['domain_tld'] != ''].groupby('domain_tld')['hit_count'].sum().sort_values(ascending=False).head(10).to_dict()
        
        # Add to stats dictionary
        stats.update({
            "top_hit_source_ips": top_hit_source_ips,
            "top_hit_sids": top_hit_sids,
            "top_hit_domains": top_hit_domains,
            "top_source_networks": top_source_networks,
            "top_destination_networks": top_destination_networks,
            "top_domain_tlds": top_domain_tlds
        })
        
        # Save to JSON file
        stats_file = os.path.join(self.output_dir, "summary_statistics.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, cls=NumpyEncoder)
        
        print(f"Summary statistics saved to {stats_file}")
        return stats