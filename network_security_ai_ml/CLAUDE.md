# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository is focused on network security applications using AI/ML techniques. This is a defensive security codebase - only assist with defensive security tasks, vulnerability analysis, intrusion detection, security monitoring, and protective measures.

## Development Guidelines

### Security Focus
- This codebase is for **defensive security only**
- Assist with: intrusion detection systems, network anomaly detection, security analytics, vulnerability scanners, threat hunting tools, security monitoring dashboards
- Do not create or modify code for offensive security, penetration testing tools, or exploit development

### Common Patterns for Network Security AI/ML Projects
- **Data preprocessing**: Network traffic data often requires normalization, feature extraction from packet headers, flow aggregation
- **Model types**: Anomaly detection (isolation forests, autoencoders), classification (random forest, neural networks), time series analysis
- **Evaluation metrics**: Focus on precision/recall for security applications, false positive rates are critical
- **Real-time processing**: Many security applications require streaming data processing

### Recommended Project Structure
```
├── data/                 # Network datasets, PCAP files, logs
├── models/              # Trained ML models
├── preprocessing/       # Data cleaning and feature engineering
├── detection/          # Anomaly and threat detection algorithms  
├── evaluation/         # Model performance and security metrics
├── visualization/      # Security dashboards and analysis plots
└── config/            # Configuration files for different environments
```

### Development Commands
Since this is a new repository, common commands will be added as the project structure is established. Typical commands for ML security projects include:
- Data preprocessing pipelines
- Model training scripts
- Real-time detection services
- Evaluation and testing frameworks

### Security Considerations
- Never include actual network credentials, API keys, or sensitive network configurations
- Use synthetic or anonymized datasets for development
- Implement proper input validation for network data processing
- Consider privacy implications when working with network traffic data