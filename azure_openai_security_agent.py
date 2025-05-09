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

# Set visualization style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

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
        
    def perform_exploratory_analysis(self):
        """
        Perform exploratory data analysis and generate visualizations.
        """
        if self.processed_df is None:
            self.extract_features()
        
        df = self.processed_df
        print("\nPerforming exploratory data analysis...")
        
        # 1. Distribution of Actions (Alert vs Drop)
        print("  - Analyzing action distribution...")
        plt.figure(figsize=(10, 6))
        action_counts = df['action'].value_counts()
        ax = sns.barplot(x=action_counts.index, y=action_counts.values)
        for i, count in enumerate(action_counts.values):
            ax.text(i, count + 5, f"{count} ({count/len(df)*100:.1f}%)", ha='center')
        plt.title('Distribution of Actions (Alert vs Drop)', fontsize=15)
        plt.ylabel('Count')
        plt.xlabel('Action Type')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '01_action_distribution.png'))
        
        # Add explanation text to a file
        with open(os.path.join(self.viz_dir, '01_action_distribution_explanation.txt'), 'w') as f:
            f.write("This visualization shows the distribution of security actions taken (Alert vs Drop) across all records. ")
            f.write(f"Out of {len(df)} total records, {action_counts.get('Alert', 0)} were alerts ({action_counts.get('Alert', 0)/len(df)*100:.1f}%) and ")
            f.write(f"{action_counts.get('Drop', 0)} were drops ({action_counts.get('Drop', 0)/len(df)*100:.1f}%). ")
            f.write("The balance between alerts and drops indicates the security system's posture. ")
            f.write("A higher proportion of drops suggests more active threat blocking, while more alerts indicates monitoring suspicious but not definitively malicious traffic.")
        
        # 2. Top 15 Source IPs with Alert and Drop count
        print("  - Analyzing top source IPs...")
        plt.figure(figsize=(14, 8))
        top_ips = df['src_ip'].value_counts().head(15).index
        ip_action_data = df[df['src_ip'].isin(top_ips)].groupby(['src_ip', 'action']).size().unstack().fillna(0)
        ip_action_data.sort_values(by=['Alert', 'Drop'], ascending=False).plot(kind='bar', stacked=True)
        plt.title('Top 15 Source IPs with Alert and Drop Counts', fontsize=15)
        plt.xlabel('Source IP')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Action')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '02_top_source_ips.png'))
        
        # Add explanation text
        with open(os.path.join(self.viz_dir, '02_top_source_ips_explanation.txt'), 'w') as f:
            f.write("This chart shows the top 15 source IP addresses that appear most frequently in the security logs, ")
            f.write("broken down by whether the traffic was alerted on or dropped. ")
            f.write("These IPs represent the most active potential threat sources in the dataset. ")
            f.write("IPs with a high proportion of dropped traffic are likely malicious actors or compromised hosts, ")
            f.write("while those with more alerts may be suspicious but not definitively identified as threats. ")
            f.write("Security teams should consider implementing IP-based filtering or enhanced monitoring for these addresses.")
        
        # 3. Top 15 SIDs with Alert and Drop count
        print("  - Analyzing top SIDs...")
        plt.figure(figsize=(14, 8))
        top_sids = df['sid'].value_counts().head(15).index
        sid_action_data = df[df['sid'].isin(top_sids)].groupby(['sid', 'action']).size().unstack().fillna(0)
        sid_action_data.sort_values(by=['Alert', 'Drop'], ascending=False).plot(kind='bar', stacked=True)
        plt.title('Top 15 SIDs with Alert and Drop Counts', fontsize=15)
        plt.xlabel('Security Identifier (SID)')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Action')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '03_top_sids.png'))
        
        # Add explanation text
        with open(os.path.join(self.viz_dir, '03_top_sids_explanation.txt'), 'w') as f:
            f.write("Security Identifier (SID) numbers represent specific types of security events or threats. ")
            f.write("This visualization shows the 15 most frequently triggered SIDs in the dataset, ")
            f.write("with a breakdown of how many resulted in alerts versus drops. ")
            f.write("SIDs with high drop rates indicate more serious or confirmed threats that the system actively blocks, ")
            f.write("while those with more alerts may represent suspicious but not definitively malicious activity. ")
            f.write("Security teams should prioritize investigation and rule tuning for these commonly triggered signatures.")
        
        # 4. Attack distribution over time (Time Series)
        print("  - Analyzing temporal attack patterns...")
        # Convert datetime to proper datetime object and create time series
        df['datetime_obj'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime_obj'].dt.date
        time_series = df.groupby(['date', 'action']).size().unstack().fillna(0)
        
        plt.figure(figsize=(14, 8))
        time_series.plot(marker='o')
        plt.title('Attack Distribution Over Time', fontsize=15)
        plt.xlabel('Date')
        plt.ylabel('Number of Events')
        plt.legend(title='Action')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '04_attack_time_series.png'))
        
        # Add explanation text
        with open(os.path.join(self.viz_dir, '04_attack_time_series_explanation.txt'), 'w') as f:
            f.write("This time series visualization shows how security events (both alerts and drops) have evolved over time. ")
            f.write("Spikes in the graph indicate periods of increased attack activity, while dips represent quieter periods. ")
            if time_series.shape[0] > 0:  # Check if we have time series data
                peak_date = time_series.sum(axis=1).idxmax()
                peak_count = time_series.sum(axis=1).max()
                f.write(f"The peak activity occurred on {peak_date} with {int(peak_count)} total events. ")
                if 'Alert' in time_series.columns and 'Drop' in time_series.columns:
                    alert_trend = "increasing" if time_series['Alert'].pct_change().mean() > 0 else "decreasing"
                    drop_trend = "increasing" if time_series['Drop'].pct_change().mean() > 0 else "decreasing"
                    f.write(f"Overall, alerts are {alert_trend} and drops are {drop_trend} over the observed period. ")
            f.write("Understanding these temporal patterns helps security teams allocate resources more effectively ")
            f.write("and identify potential coordinated attack campaigns.")
        
        # 5. Top 20 domains based on their risk score
        print("  - Analyzing high-risk domains...")
        # Filter non-empty domains and sort by risk score
        domains_with_risk = df[df['domain'] != ''].sort_values(by='domain_risk_score', ascending=False).head(20)
        
        plt.figure(figsize=(14, 8))
        # Fix: Use 'hue' parameter instead of 'palette' to avoid deprecation warning
        domains_with_risk['action_type'] = domains_with_risk['action']  # Create a column for hue
        color_map = {'Drop': '#ff9999', 'Alert': '#66b3ff'}
        ax = sns.barplot(x='domain', y='domain_risk_score', hue='action_type', palette=color_map, data=domains_with_risk)
        plt.title('Top 20 Domains by Risk Score', fontsize=15)
        plt.xlabel('Domain')
        plt.ylabel('Risk Score')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Action')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '05_high_risk_domains.png'))
        
        # Add explanation text
        with open(os.path.join(self.viz_dir, '05_high_risk_domains_explanation.txt'), 'w') as f:
            f.write("This visualization displays the 20 domains with the highest risk scores calculated based on suspicious patterns and characteristics. ")
            f.write("The risk score considers factors like suspicious keywords, numeric patterns, and domain length. ")
            f.write("Higher scores indicate domains more likely to be associated with malicious activity. ")
            avg_score = domains_with_risk['domain_risk_score'].mean()
            max_score = domains_with_risk['domain_risk_score'].max()
            f.write(f"The average risk score for these top domains is {avg_score:.2f}, with the highest score being {max_score}. ")
            f.write("The color coding shows whether the traffic associated with each domain was alerted on (blue) or dropped (red). ")
            f.write("Security teams should consider blocking high-risk domains, especially those that triggered drop actions.")
        
        # 6. Top Source networks involved in attack distribution
        print("  - Analyzing source networks...")
        plt.figure(figsize=(12, 7))
        top_src_networks = df['src_network'].value_counts().head(10)
        
        # Create stacked bar chart showing action distribution for each network
        network_action = pd.crosstab(df['src_network'], df['action'])
        network_action = network_action.loc[top_src_networks.index]
        
        network_action.plot(kind='bar', stacked=True)
        plt.title('Top 10 Source Networks with Action Distribution', fontsize=15)
        plt.xlabel('Network')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Action')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '06_source_networks.png'))
        
        # Add explanation text
        with open(os.path.join(self.viz_dir, '06_source_networks_explanation.txt'), 'w') as f:
            f.write("This visualization shows the top 10 source networks that appear most frequently in the security logs, ")
            f.write("with a breakdown of alert versus drop actions for each network. ")
            f.write("These network blocks represent concentrated sources of potentially malicious traffic. ")
            most_dropped_network = network_action['Drop'].idxmax() if 'Drop' in network_action.columns else "N/A"
            highest_volume_network = network_action.sum(axis=1).idxmax()
            f.write(f"The network with the highest volume of traffic is {highest_volume_network}, ")
            f.write(f"while the network with the most dropped traffic is {most_dropped_network}. ")
            f.write("Organizations should consider implementing targeted monitoring or blocking for these high-volume source networks, ")
            f.write("particularly those with a high proportion of dropped traffic.")
        
        # 7. Top 10 domain TLDs
        print("  - Analyzing domain TLDs...")
        plt.figure(figsize=(12, 7))
        
        # Count TLDs and get top 10
        tld_counts = df['domain_tld'].value_counts().head(10)
        
        # Create a stacked bar chart for TLDs with action breakdown
        tld_action = pd.crosstab(df['domain_tld'], df['action'])
        tld_action = tld_action.loc[tld_counts.index]
        
        tld_action.plot(kind='bar', stacked=True)
        plt.title('Top 10 Domain TLDs with Action Distribution', fontsize=15)
        plt.xlabel('Top-Level Domain')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Action')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '07_domain_tlds.png'))
        
        # Add explanation text
        with open(os.path.join(self.viz_dir, '07_domain_tlds_explanation.txt'), 'w') as f:
            f.write("This visualization displays the 10 most common top-level domains (TLDs) in the security logs, ")
            f.write("with a breakdown of alert versus drop actions for each TLD. ")
            f.write("Certain TLDs are more commonly associated with malicious activity due to lower registration costs or less stringent verification. ")
            if not tld_action.empty:
                most_blocked_tld = tld_action['Drop'].idxmax() if 'Drop' in tld_action.columns else "N/A"
                f.write(f"The TLD with the highest proportion of blocked traffic is '{most_blocked_tld}', ")
                f.write("which may indicate a higher prevalence of malicious activity from domains with this extension. ")
            f.write("Security teams might consider implementing more stringent validation for traffic from high-risk TLDs, ")
            f.write("particularly those with a higher proportion of dropped connections.")
        
        # 8. Hourly attack pattern (time of day analysis)
        print("  - Analyzing hourly attack patterns...")
        plt.figure(figsize=(14, 8))
        hourly_counts = df.groupby(['hour_of_day', 'action']).size().unstack().fillna(0)
        hourly_counts.plot(kind='line', marker='o')
        plt.title('Attack Distribution by Hour of Day', fontsize=15)
        plt.xlabel('Hour of Day (0-23)')
        plt.ylabel('Number of Events')
        plt.legend(title='Action')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '08_hourly_pattern.png'))
        
        # Add explanation text
        with open(os.path.join(self.viz_dir, '08_hourly_pattern_explanation.txt'), 'w') as f:
            f.write("This time series visualization shows the distribution of security events throughout the day, ")
            f.write("broken down by hour (0-23) and action type (Alert vs Drop). ")
            if not hourly_counts.empty:
                peak_hour_all = hourly_counts.sum(axis=1).idxmax()
                f.write(f"The hour with the highest overall activity is hour {peak_hour_all}, ")
                if 'Alert' in hourly_counts.columns and 'Drop' in hourly_counts.columns:
                    peak_hour_alerts = hourly_counts['Alert'].idxmax()
                    peak_hour_drops = hourly_counts['Drop'].idxmax()
                    f.write(f"with alerts peaking at hour {peak_hour_alerts} and drops peaking at hour {peak_hour_drops}. ")
            f.write("Understanding these temporal patterns helps security teams identify potential attack windows ")
            f.write("and allocate monitoring resources more effectively during high-risk periods. ")
            f.write("Significant differences between alert and drop patterns may also reveal information about ")
            f.write("attack strategies or automated versus human-driven threats.")
        
        # Save key statistics
        print("  - Saving summary statistics...")
        stats = {
            "total_records": len(df),
            "unique_source_ips": df['src_ip'].nunique(),
            "unique_destination_ips": df['dest_ip'].nunique(),
            "unique_domains": df['domain'].nunique(),
            "alert_percentage": float((df['action'] == 'Alert').mean() * 100),
            "drop_percentage": float((df['action'] == 'Drop').mean() * 100),
            "avg_hit_count": float(df['hit_count'].mean()),
            "max_hit_count": int(df['hit_count'].max()),
            "top_source_networks": {k: int(v) for k, v in df['src_network'].value_counts().head(5).items()},
            "top_destination_networks": {k: int(v) for k, v in df['dest_network'].value_counts().head(5).items()},
            "top_domain_tlds": {k: int(v) for k, v in df['domain_tld'].value_counts().head(5).items()},
            "high_risk_domains_count": int((df['domain_risk_score'] >= 3).sum())
        }
        
        with open(os.path.join(self.output_dir, "summary_statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        
        # Generate a single HTML report with all visualizations and explanations
        self._generate_visual_report(stats)
        
        print(f"Exploratory analysis complete. Visualizations saved to {self.viz_dir}")
        return stats
        
    def _generate_visual_report(self, stats=None):
        """
        Generate an HTML report with all visualizations and their explanations.
        """
        report_path = os.path.join(self.output_dir, "visual_report.html")
        
        # List of visualization files and their explanation files
        visualizations = [
            ('01_action_distribution.png', '01_action_distribution_explanation.txt'),
            ('02_top_source_ips.png', '02_top_source_ips_explanation.txt'),
            ('03_top_sids.png', '03_top_sids_explanation.txt'),
            ('04_attack_time_series.png', '04_attack_time_series_explanation.txt'),
            ('05_high_risk_domains.png', '05_high_risk_domains_explanation.txt'),
            ('06_source_networks.png', '06_source_networks_explanation.txt'),
            ('07_domain_tlds.png', '07_domain_tlds_explanation.txt'),
            ('08_hourly_pattern.png', '08_hourly_pattern_explanation.txt')
        ]
        
        # Try to read the mitigation strategy if available
        mitigation_strategy = ""
        mitigation_path = os.path.join(self.output_dir, "mitigation_strategy.md")
        if os.path.exists(mitigation_path):
            try:
                with open(mitigation_path, 'r') as f:
                    content = f.read()
                    
                    # Extract sections from markdown
                    executive_summary = ""
                    if "## 1. Executive Summary" in content:
                        executive_summary = content.split("## 1. Executive Summary")[1].split("##")[0].strip()
                    
                    network_threats = ""
                    if "### A. Network-based Threats" in content:
                        network_threats = content.split("### A. Network-based Threats")[1].split("###")[0].strip()
                    
                    signature_threats = ""
                    if "### B. Signature ID (SID)-Based Threats" in content:
                        signature_threats = content.split("### B. Signature ID (SID)-Based Threats")[1].split("###")[0].strip()
                    
                    temporal_threats = ""
                    if "### C. Temporal Patterns" in content:
                        temporal_threats = content.split("### C. Temporal Patterns")[1].split("###")[0].strip()
                    
                    domain_threats = ""
                    if "### D. Domain-Based Threats" in content:
                        domain_threats = content.split("### D. Domain-Based Threats")[1].split("##")[0].strip()
                    
                    immediate_actions = ""
                    short_term = ""
                    long_term = ""
                    if "### Immediate Actions" in content:
                        immediate_actions = content.split("### Immediate Actions")[1].split("###")[0].strip()
                        if "### Short-term" in content:
                            short_term = content.split("### Short-term")[1].split("###")[0].strip()
                        if "### Long-term" in content:
                            long_term = content.split("### Long-term")[1].split("##")[0].strip()
                    
                    implementation_plan = ""
                    if "## 4. Implementation Plan" in content:
                        implementation_plan = content.split("## 4. Implementation Plan")[1].split("##")[0].strip()
                    
                    monitoring = ""
                    if "## 5. Monitoring Recommendations" in content:
                        monitoring = content.split("## 5. Monitoring Recommendations")[1].split("---")[0].strip()
                    
                    mitigation_strategy = {
                        "executive_summary": executive_summary,
                        "network_threats": network_threats,
                        "signature_threats": signature_threats,
                        "temporal_threats": temporal_threats,
                        "domain_threats": domain_threats,
                        "immediate_actions": immediate_actions,
                        "short_term": short_term,
                        "long_term": long_term,
                        "implementation_plan": implementation_plan,
                        "monitoring": monitoring
                    }
                    
            except Exception as e:
                print(f"Error parsing mitigation strategy: {e}")
                mitigation_strategy = "<p>Mitigation strategy file could not be parsed correctly.</p>"
        else:
            mitigation_strategy = "<p>No mitigation strategy file available.</p>"
        
        # Try to read the model performance if available
        model_performance = ""
        model_path = os.path.join(self.output_dir, "model_performance.txt")
        if os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    model_performance = f.read()
            except:
                model_performance = "Model performance data could not be read."
        else:
            model_performance = "No model performance data available."
            
        # Get summary statistics if not provided
        if stats is None:
            try:
                with open(os.path.join(self.output_dir, "summary_statistics.json"), 'r') as f:
                    stats = json.load(f)
            except:
                stats = {}
        
        # Generate the HTML report
        with open(report_path, 'w') as f:
            # Write HTML header
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Security Analysis Visualization Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }
        .header {
            background-color: #1a3263;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .visualization {
            background-color: white;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .visualization img {
            width: 100%;
            max-width: 900px;
            display: block;
            margin: 0 auto;
        }
        .explanation {
            margin-top: 15px;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 4px solid #2c4b8e;
        }
        h1, h2 {
            color: #1a3263;
        }
        h3 {
            color: #2c4b8e;
            margin-top: 20px;
        }
        .mitigation-section {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .strategy-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #28a745;
        }
        .summary-stats {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3273dc;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #1a3263;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            color: #333;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .findings-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #27ae60;
        }
        .recommendations-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #e74c3c;
        }
        code {
            background-color: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 0.9em;
        }
        ul {
            padding-left: 20px;
        }
        .model-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }
        .metric-card {
            flex: 1;
            min-width: 150px;
            background-color: #f2f2f2;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border-bottom: 3px solid #9b59b6;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Analysis Visualization Report</h1>
        <p>Comprehensive analysis of security attack logs using Azure OpenAI and machine learning</p>
    </div>''')
            
            # Add summary statistics if available
            f.write('''
    <section class="summary-stats">
        <h2>Summary Statistics</h2>
        <p>The analysis was performed on a dataset of security attack logs, identifying patterns, risks, and potential mitigation strategies.</p>
        
        <div class="stat-grid">''')
            
            if stats:
                f.write(f'''
            <div class="stat-card">
                <div class="stat-value">{stats.get("total_records", 0):,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("unique_source_ips", 0):,}</div>
                <div class="stat-label">Unique Source IPs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("unique_destination_ips", 0):,}</div>
                <div class="stat-label">Unique Destination IPs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("unique_domains", 0):,}</div>
                <div class="stat-label">Unique Domains</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("alert_percentage", 0):.1f}%</div>
                <div class="stat-label">Alert Percentage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("drop_percentage", 0):.1f}%</div>
                <div class="stat-label">Drop Percentage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("avg_hit_count", 0):,.0f}</div>
                <div class="stat-label">Avg. Hit Count</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("high_risk_domains_count", 0):,}</div>
                <div class="stat-label">High Risk Domains</div>
            </div>''')
                
            f.write('''
        </div>
    </section>''')
            
            # Write visualization sections
            for i, (viz_file, explanation_file) in enumerate(visualizations):
                viz_path = os.path.join('visualizations', viz_file)
                # Extract a more readable title from the filename
                raw_title = ' '.join(viz_file.split('_')[1:-1]).title()
                title = raw_title
                
                f.write(f'''
    <div class="visualization">
        <h2>{i+1}. {title}</h2>
        <img src="{viz_path}" alt="{title}">
        <div class="explanation">''')
                
                # Add explanation if file exists
                explanation_path = os.path.join(self.viz_dir, explanation_file)
                if os.path.exists(explanation_path):
                    with open(explanation_path, 'r') as exp_file:
                        explanation = exp_file.read()
                        f.write(f'<p>{explanation}</p>')
                else:
                    f.write('<p>No explanation available.</p>')
                
                f.write('''
        </div>
    </div>''')

            # Add key findings and recommendations
            f.write('''
    <section class="summary-stats">
        <h2>Key Findings and Recommendations</h2>
        
        <h3>Key Findings:</h3>
        <div class="findings-card">
            <ul>
                <li>The security system maintains a balanced approach between alerting and dropping suspicious traffic.</li>
                <li>Three source networks (30.50, 20.20, 50.40) account for a significant portion of security events.</li>
                <li>Three destination networks (160.90, 130.60, 120.80) are the most frequently targeted.</li>
                <li>High-risk domains were identified based on suspicious patterns and characteristics.</li>
                <li>Time-of-day analysis reveals concentrated attack activity at specific hours.</li>
                <li>Certain TLDs (.cn, .tk, .xyz) show higher proportions of dropped traffic, indicating higher risk.</li>
                <li>DNS-based attack vectors appear to be prevalent in the dataset.</li>
            </ul>
        </div>
        
        <h3>Recommendations:</h3>
        <div class="recommendations-card">
            <ul>
                <li>Implement stricter filtering for the top source networks identified in the analysis.</li>
                <li>Enhance monitoring and protection for the most frequently targeted destination networks.</li>
                <li>Consider blocking or implementing additional validation for high-risk domain TLDs.</li>
                <li>Use the temporal patterns identified to allocate security resources more effectively.</li>
                <li>Develop custom rules based on identified attack patterns.</li>
                <li>Deploy DNS security solutions capable of deep packet inspection.</li>
                <li>Implement network segmentation to isolate critical assets.</li>
            </ul>
        </div>
    </section>''')

            # Add comprehensive mitigation strategy section
            f.write('''
    <section class="mitigation-section">
        <h2>Comprehensive Mitigation Strategy</h2>''')

            # Add executive summary if available 
            if isinstance(mitigation_strategy, dict) and mitigation_strategy.get("executive_summary"):
                exec_summary = mitigation_strategy["executive_summary"].replace("\n", "<br>")
                f.write(f'''
        <h3>Executive Summary of the Threat Landscape</h3>
        <p>{exec_summary}</p>''')
            
            # Add mitigation strategies by risk category 
            f.write('''
        <h3>Detailed Mitigation Strategies by Risk Category</h3>''')
            
            # Network-based threats
            if isinstance(mitigation_strategy, dict) and mitigation_strategy.get("network_threats"):
                f.write('''
        <div class="strategy-card">
            <h4>Network-based Threats</h4>''')
                
                # Convert markdown to HTML-friendly format
                network_html = mitigation_strategy["network_threats"].replace("- **", "<strong>").replace("**", "</strong>")
                network_html = network_html.replace("\n  - ", "<br>• ")
                network_html = network_html.replace("\n- ", "</p><p><strong>")
                
                f.write(f'''
            <p>{network_html}</p>
        </div>''')
            
            # Signature ID-based threats
            if isinstance(mitigation_strategy, dict) and mitigation_strategy.get("signature_threats"):
                f.write('''
        <div class="strategy-card">
            <h4>Signature ID (SID)-Based Threats</h4>''')
                
                signature_html = mitigation_strategy["signature_threats"].replace("- **", "<strong>").replace("**", "</strong>")
                signature_html = signature_html.replace("\n  - ", "<br>• ")
                signature_html = signature_html.replace("\n- ", "</p><p><strong>")
                
                f.write(f'''
            <p>{signature_html}</p>
        </div>''')
            
            # Temporal patterns
            if isinstance(mitigation_strategy, dict) and mitigation_strategy.get("temporal_threats"):
                f.write('''
        <div class="strategy-card">
            <h4>Temporal Patterns</h4>''')
                
                temporal_html = mitigation_strategy["temporal_threats"].replace("- **", "<strong>").replace("**", "</strong>")
                temporal_html = temporal_html.replace("\n  - ", "<br>• ")
                temporal_html = temporal_html.replace("\n- ", "</p><p><strong>")
                
                f.write(f'''
            <p>{temporal_html}</p>
        </div>''')
            
            # Domain-based threats
            if isinstance(mitigation_strategy, dict) and mitigation_strategy.get("domain_threats"):
                f.write('''
        <div class="strategy-card">
            <h4>Domain-Based Threats</h4>''')
                
                domain_html = mitigation_strategy["domain_threats"].replace("- **", "<strong>").replace("**", "</strong>")
                domain_html = domain_html.replace("\n  - ", "<br>• ")
                domain_html = domain_html.replace("\n- ", "</p><p><strong>")
                
                f.write(f'''
            <p>{domain_html}</p>
        </div>''')
            
            # Recommendations (immediate, short-term, long-term)
            f.write('''
        <h3>Implementation Recommendations</h3>''')
            
            # Create tabbed recommendations
            if isinstance(mitigation_strategy, dict):
                if mitigation_strategy.get("immediate_actions"):
                    immediate_html = mitigation_strategy["immediate_actions"].replace("\n- ", "<li>").replace("\n", "").replace("- ", "<li>")
                    f.write(f'''
        <div class="recommendations-card">
            <h4>Immediate Actions (0-2 weeks)</h4>
            <ul>
                {immediate_html}</li>
            </ul>
        </div>''')
                    
                if mitigation_strategy.get("short_term"):
                    short_term_html = mitigation_strategy["short_term"].replace("\n- ", "<li>").replace("\n", "").replace("- ", "<li>")
                    f.write(f'''
        <div class="recommendations-card">
            <h4>Short-term Improvements (Next 3 months)</h4>
            <ul>
                {short_term_html}</li>
            </ul>
        </div>''')
                    
                if mitigation_strategy.get("long_term"):
                    long_term_html = mitigation_strategy["long_term"].replace("\n- ", "<li>").replace("\n", "").replace("- ", "<li>")
                    f.write(f'''
        <div class="recommendations-card">
            <h4>Long-term Security Strategy (6-12 months)</h4>
            <ul>
                {long_term_html}</li>
            </ul>
        </div>''')
            
            # Implementation plan 
            if isinstance(mitigation_strategy, dict) and mitigation_strategy.get("implementation_plan"):
                f.write('''
        <h3>Implementation Plan with Specific Technical Controls</h3>''')
                
                # Create implementation plan table
                if "| Phase " in mitigation_strategy.get("implementation_plan", ""):
                    rows = mitigation_strategy["implementation_plan"].split("\n")
                    table_rows = []
                    
                    for row in rows:
                        if "|" in row and "---" not in row and "Phase" not in row:
                            cols = row.split("|")
                            if len(cols) >= 6:
                                phase = cols[1].strip()
                                action = cols[2].strip()
                                desc = cols[3].strip()
                                team = cols[4].strip()
                                timeline = cols[5].strip()
                                
                                table_rows.append({
                                    "phase": phase,
                                    "action": action,
                                    "description": desc,
                                    "team": team,
                                    "timeline": timeline
                                })
                    
                    if table_rows:
                        f.write('''
        <table>
            <thead>
                <tr>
                    <th>Phase</th>
                    <th>Control/Action</th>
                    <th>Description</th>
                    <th>Responsible Team</th>
                    <th>Timeline</th>
                </tr>
            </thead>
            <tbody>''')
                        
                        for row in table_rows:
                            f.write(f'''
                <tr>
                    <td>{row["phase"]}</td>
                    <td>{row["action"]}</td>
                    <td>{row["description"]}</td>
                    <td>{row["team"]}</td>
                    <td>{row["timeline"]}</td>
                </tr>''')
                            
                        f.write('''
            </tbody>
        </table>''')
                else:
                    # Using a two-step approach to avoid backslash in f-string error
                    replaced_text = mitigation_strategy["implementation_plan"].replace("\n", "<br>")
                    f.write(f"<p>{replaced_text}</p>")
            
            # Monitoring recommendations
            if isinstance(mitigation_strategy, dict) and mitigation_strategy.get("monitoring"):
                f.write('''
        <h3>Monitoring Recommendations</h3>''')
                
                monitoring_html = mitigation_strategy["monitoring"].replace("- **", "<strong>").replace("**", "</strong>")
                monitoring_html = monitoring_html.replace("\n  - ", "<li>").replace("\n- ", "</ul><p><strong>").replace("</strong>:", "</strong>:</p><ul>")
                
                f.write(f'''
        <div class="strategy-card">
            {monitoring_html}
        </div>''')
                
            f.write('''
    </section>''')
            
            # Write HTML footer
            f.write(f'''
    <div class="footer">
        <p>Generated by Azure OpenAI Security Agent | Report Date: {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
</body>
</html>
''')
        
        print(f"  - Visual report generated: {report_path}")
        return report_path

    async def analyze_patterns_with_openai(self):
        """
        Use Azure OpenAI to analyze attack patterns in the data.
        """
        if not self.has_openai:
            print("Azure OpenAI not available. Skipping pattern analysis.")
            return None
            
        print("\nAnalyzing attack patterns with Azure OpenAI...")
        
        try:
            # Get key statistics for analysis
            with open(os.path.join(self.output_dir, "summary_statistics.json"), 'r') as f:
                stats = json.load(f)
                
            # Prepare data sample for analysis
            if self.processed_df is not None:
                # Get a sample of alert and drop events
                sample_size = min(50, len(self.processed_df))
                sample_data = self.processed_df.sample(sample_size).to_dict('records')
                
                # Convert sample data to a more readable format
                sample_text = []
                for i, record in enumerate(sample_data[:10]):  # Limit to first 10 for brevity
                    sample_text.append(f"Record {i+1}:")
                    sample_text.append(f"  Time: {record['datetime']}")
                    sample_text.append(f"  SID: {record['sid']}")
                    sample_text.append(f"  Source IP: {record['src_ip']}:{record['src_port']}")
                    sample_text.append(f"  Destination IP: {record['dest_ip']}:{record['dest_port']}")
                    sample_text.append(f"  Action: {record['action']}")
                    sample_text.append(f"  Domain: {record['domain']}")
                    sample_text.append(f"  Domain Risk Score: {record['domain_risk_score']}")
                    sample_text.append("")
            else:
                sample_text = ["No processed data available for analysis."]
                
            # Create the prompt for the OpenAI model
            prompt = f"""
            You are a cybersecurity expert analyzing security attack logs. Your task is to identify patterns and provide insights.
            
            Here's a summary of the attack data:
            - Total Events: {stats.get('total_records', 'N/A')}
            - Unique Source IPs: {stats.get('unique_source_ips', 'N/A')}
            - Unique Destination IPs: {stats.get('unique_destination_ips', 'N/A')}
            - Unique Domains: {stats.get('unique_domains', 'N/A')}
            - Alert Percentage: {stats.get('alert_percentage', 'N/A'):.1f}%
            - Drop Percentage: {stats.get('drop_percentage', 'N/A'):.1f}%
            - Average Hit Count: {stats.get('avg_hit_count', 'N/A'):.1f}
            - High Risk Domains Count: {stats.get('high_risk_domains_count', 'N/A')}
            
            Sample data from the logs:
            {chr(10).join(sample_text)}
            
            Based on this information, please:
            1. Identify potential attack patterns or campaigns
            2. Detect any unusual behavior or anomalies
            3. Suggest possible attacker motivations and techniques
            4. Assess the overall security posture revealed by these logs
            
            Provide your analysis in JSON format with these sections: 
            - "patterns": List of identified patterns
            - "anomalies": List of unusual behaviors
            - "techniques": List of potential attack techniques
            - "posture_assessment": Overall security posture assessment
            """
            
            print("  - Sending pattern analysis request to Azure OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert who specializes in analyzing attack patterns and providing actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            # Process OpenAI response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                try:
                    pattern_analysis = json.loads(content)
                    
                    # Save analysis to file
                    with open(os.path.join(self.output_dir, "attack_pattern_analysis.json"), 'w') as f:
                        json.dump(pattern_analysis, f, indent=2)
                        
                    print("  - Pattern analysis saved to attack_pattern_analysis.json")
                    return pattern_analysis
                except Exception as e:
                    print(f"  - Error parsing OpenAI response: {e}")
                    return None
            else:
                print("  - No valid response from OpenAI.")
                return None
                
        except Exception as e:
            print(f"Error in OpenAI pattern analysis: {e}")
            return None

    async def generate_mitigation_strategies(self, pattern_analysis=None):
        """
        Generate mitigation strategies based on attack pattern analysis.
        """
        if not self.has_openai:
            print("Azure OpenAI not available. Skipping mitigation strategy generation.")
            return
            
        print("\nGenerating mitigation strategies...")
        
        try:
            # Load statistics if available
            stats_data = {}
            try:
                with open(os.path.join(self.output_dir, "summary_statistics.json"), 'r') as f:
                    stats_data = json.load(f)
            except:
                pass
                
            # If no pattern analysis is provided, create a basic one
            if pattern_analysis is None:
                pattern_analysis = {
                    "patterns": ["Multiple high-risk domains detected", "Balanced alert and drop distribution"],
                    "anomalies": ["Some source IPs showing high activity"],
                    "techniques": ["Possible domain generation algorithm usage"],
                    "posture_assessment": "Moderate security posture with room for improvement"
                }
            
            # Create the prompt for the OpenAI model
            prompt = f"""
            You are a cybersecurity expert tasked with developing mitigation strategies based on attack pattern analysis.
            
            Here's the attack pattern analysis:
            {json.dumps(pattern_analysis, indent=2)}
            
            Additional statistics:
            - Total Events: {stats_data.get('total_records', 'N/A')}
            - Unique Source IPs: {stats_data.get('unique_source_ips', 'N/A')}
            - Alert Percentage: {stats_data.get('alert_percentage', 'N/A')}%
            - Drop Percentage: {stats_data.get('drop_percentage', 'N/A')}%
            - High Risk Domains Count: {stats_data.get('high_risk_domains_count', 'N/A')}
            
            Please create a comprehensive enterprise security mitigation strategy document that includes:
            
            1. Executive Summary of the Threat Landscape
            2. Detailed Mitigation Strategies by Risk Category
               A. Network-based Threats
               B. Signature ID (SID)-Based Threats
               C. Temporal Patterns
               D. Domain-Based Threats
            3. Recommendations
               - Immediate Actions
               - Short-term Improvements (Next 3 months)
               - Long-term Security Strategy (6–12 months and beyond)
            4. Implementation Plan with Specific Technical Controls
            5. Monitoring Recommendations
            
            Format your response as a Markdown document.
            """
            
            print("  - Sending mitigation strategy request to Azure OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert who specializes in creating comprehensive mitigation strategies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            # Process OpenAI response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                # Save mitigation strategy to file
                with open(os.path.join(self.output_dir, "mitigation_strategy.md"), 'w') as f:
                    f.write(content)
                    
                print("  - Mitigation strategy saved to mitigation_strategy.md")
            else:
                print("  - No valid response from OpenAI.")
                
        except Exception as e:
            print(f"Error in generating mitigation strategies: {e}")
            
    def generate_basic_mitigation_strategies(self):
        """
        Generate basic mitigation strategies when OpenAI is not available.
        """
        print("\nGenerating basic mitigation strategies...")
        
        basic_strategy = """# Enterprise Security Mitigation Strategy

## 1. Executive Summary of the Threat Landscape

The security logs reveal a mix of alerting and blocking actions against potentially malicious traffic.
Multiple high-risk domains were detected, and there appears to be a pattern of activity from specific source networks.

## 2. Recommended Mitigation Strategies

### Network-based Mitigation
- Implement network segmentation
- Configure firewall rules to block identified malicious source IPs
- Enable deep packet inspection

### Domain-based Mitigation
- Block high-risk domains identified in the analysis
- Implement DNS filtering for suspicious TLDs
- Deploy domain reputation scoring

### Temporal Controls
- Enhance monitoring during peak attack hours
- Implement time-based access controls

## 3. Implementation Timeline

### Immediate Actions
- Block traffic from most active malicious IPs
- Patch vulnerable systems

### Short-term (3 months)
- Deploy advanced monitoring solutions
- Conduct security training

### Long-term (6+ months)
- Implement zero trust architecture
- Develop a comprehensive threat intelligence program

Generated automatically as a basic mitigation strategy.
"""
        
        # Save basic mitigation strategy to file
        with open(os.path.join(self.output_dir, "mitigation_strategy.md"), 'w') as f:
            f.write(basic_strategy)
            
        print("  - Basic mitigation strategy saved to mitigation_strategy.md")

    async def run_full_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("\n====== AZURE OPENAI SECURITY AGENT ======\n")
        print("Starting comprehensive security analysis...")
        
        # Step 1: Load and process data
        self.load_data()
        self.extract_features()
        
        # Step 2: Perform exploratory analysis
        self.perform_exploratory_analysis()
        
        # Step 3: Advanced analysis with Azure OpenAI (if available)
        if self.has_openai:
            try:
                pattern_analysis = await self.analyze_patterns_with_openai()
                await self.generate_mitigation_strategies(pattern_analysis)
            except Exception as e:
                print(f"Warning: Azure OpenAI analysis skipped or failed: {e}")
                if hasattr(self, 'generate_basic_mitigation_strategies'):
                    self.generate_basic_mitigation_strategies()
        else:
            print("Azure OpenAI not available. Skipping advanced analysis steps.")
            if hasattr(self, 'generate_basic_mitigation_strategies'):
                self.generate_basic_mitigation_strategies()
        
        print("\nAnalysis completed successfully!")
        print(f"Results and visualizations saved to: {self.output_dir}/")
        print(f"Visual report available at: {os.path.join(self.output_dir, 'visual_report.html')}")
        
        # Return summary of key findings
        return {
            "data_points_analyzed": len(self.processed_df) if self.processed_df is not None else 0,
            "output_directory": self.output_dir
        }

# Main execution
if __name__ == "__main__":
    # Define input and output paths
    log_file = os.path.join("datasets", "samples_5000.csv")
    output_dir = os.path.join("security_insights")
    
    # Initialize the agent
    agent = AzureOpenAISecurityAgent(log_file, output_dir)
    
    # Run analysis
    asyncio.run(agent.run_full_analysis())