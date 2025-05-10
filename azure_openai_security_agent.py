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
SECONDARY_COLOR = "#111a20"  # very dark blue/black
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
        """
        print("\nLoading attack log data...")
        
        try:
            # Read raw lines from the file
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Initialize data structures
            timestamps = []
            current_timestamp = None
            entries = []
            
            # Skip the header line
            header_line = lines[0].strip()
            
            # Process each line
            for line in lines[1:]:
                line = line.strip()
                
                if line.startswith("TIME "):
                    # Parse timestamp line
                    parts = line.split(" ", 2)
                    if len(parts) >= 3:
                        epoch_time = int(parts[1])
                        datetime_str = parts[2].split(" ")[0]
                        current_timestamp = {
                            'epoch': epoch_time,
                            'datetime': datetime_str
                        }
                else:
                    # Parse data entry line
                    try:
                        parts = line.split("  ")
                        if len(parts) >= 7:
                            sid = int(parts[0])
                            src_ip = parts[1]
                            src_port = int(parts[2])
                            dest_ip = parts[3]
                            dest_port = int(parts[4])
                            action = parts[5]
                            hit_count = int(parts[6])
                            domain = parts[7] if len(parts) > 7 else ""
                            
                            # Add the entry with current timestamp
                            if current_timestamp:
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
                        # Skip malformed lines
                        continue
            
            # Create DataFrame
            self.raw_df = pd.DataFrame(entries)
            
            # Print summary statistics
            print(f"Data loaded successfully:")
            print(f"  - Total records: {len(self.raw_df)}")
            print(f"  - Time range: {self.raw_df['datetime'].min()} to {self.raw_df['datetime'].max()}")
            print(f"  - Unique SIDs: {self.raw_df['sid'].nunique()}")
            print(f"  - Actions: {dict(self.raw_df['action'].value_counts())}")
            
            return self.raw_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def extract_features(self):
        """
        Extract and engineer features from the raw data.
        """
        if self.raw_df is None:
            self.load_data()
            
        print("\nExtracting and engineering features...")
        df = self.raw_df.copy()
        
        # 1. IP-based features
        print("  - Creating network-based features...")
        
        # Extract IP address components
        def get_ip_class(ip):
            """Determine IP address class (A, B, C, D, or E)"""
            first_octet = int(ip.split('.')[0])
            if 1 <= first_octet <= 126:
                return 'A'
            elif 128 <= first_octet <= 191:
                return 'B'
            elif 192 <= first_octet <= 223:
                return 'C'
            elif 224 <= first_octet <= 239:
                return 'D'
            else:
                return 'E'
                
        # Extract network information
        df['src_network'] = df['src_ip'].apply(lambda ip: '.'.join(ip.split('.')[:2]))
        df['dest_network'] = df['dest_ip'].apply(lambda ip: '.'.join(ip.split('.')[:2]))
        df['src_ip_class'] = df['src_ip'].apply(get_ip_class)
        df['dest_ip_class'] = df['dest_ip'].apply(get_ip_class)
        
        # 2. Domain-based features
        print("  - Creating domain-based features...")
        
        # Extract domain TLD and check for suspicious patterns
        df['domain_tld'] = df['domain'].apply(lambda d: d.split('.')[-1] if d and '.' in d else '')
        
        # Function to calculate domain risk score based on suspicious keywords
        suspicious_patterns = [
            'malware', 'trojan', 'hack', 'exploit', 'attack', 'phish', 'virus',
            'botnet', 'ransom', 'spyware', 'threat', 'c2', 'command', 'control',
            'worm', 'dark', 'shadow', 'covert', 'hidden', 'leak', 'steal'
        ]
        
        def calculate_domain_risk(domain):
            if not domain:
                return 0
                
            domain = domain.lower()
            score = 0
            
            # Check for suspicious keywords
            for pattern in suspicious_patterns:
                if pattern in domain:
                    score += 1
            
            # Check for numeric patterns (often used in algorithmically generated domains)
            digit_count = sum(c.isdigit() for c in domain)
            if digit_count > 3:
                score += 1
            
            # Check for long domains (potential obfuscation)
            if len(domain) > 20:
                score += 1
                
            return score
            
        df['domain_risk_score'] = df['domain'].apply(calculate_domain_risk)
        
        # 3. Temporal features
        print("  - Creating temporal features...")
        
        # Convert timestamp to datetime and extract components
        df['hour_of_day'] = pd.to_datetime(df['datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda day: 1 if day >= 5 else 0)  # 5,6 = Saturday,Sunday
        
        # 4. SID-based features
        print("  - Creating SID-based features...")
        
        # Group SIDs by prefix
        df['sid_category'] = df['sid'].astype(str).str[:3]
        
        # 5. Port-based features
        print("  - Creating port-based features...")
        
        # Categorize ports
        def categorize_port(port):
            """Categorize ports into well-known, registered, or dynamic"""
            if port <= 1023:
                return 'well-known'
            elif 1024 <= port <= 49151:
                return 'registered'
            else:
                return 'dynamic'
                
        df['src_port_category'] = df['src_port'].apply(categorize_port)
        df['dest_port_category'] = df['dest_port'].apply(categorize_port)
        
        # Store processed data
        self.processed_df = df
        
        # Print feature summary
        new_features = list(set(self.processed_df.columns) - set(self.raw_df.columns))
        print(f"Feature engineering complete. New features created: {len(new_features)}")
        print(f"  - {', '.join(sorted(new_features))}")
        
        return self.processed_df
        
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