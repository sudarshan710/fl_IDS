This project implements a privacy-preserving Intrusion Detection System (IDS) using Federated Learning across distributed edge nodes (cloud servers or VMs). The system enables collaborative training of IDS models without sharing raw data between nodes..

```bash
# Install necessary libraries
pip install -r requirements.txt

# To run the application
python3 app.py
```

## 🔧 Architecture Overview
- Edge Nodes train local IDS models (e.g., SVM, Random Forest) using private data like network traffic logs and system logs.
- Central Aggregator Server receives encrypted model updates (not raw data), aggregates them using Federated Averaging (FedAvg), and sends the updated global model back to edge nodes for the next training round.
- Secure Communication is established using HTTPS/TLS protocols to encrypt data in transit.

## 🔐 Privacy & Security Enhancements
- Differential Privacy: Adds noise to model updates to protect individual data points.
- Homomorphic Encryption: Allows encrypted updates to be aggregated without decryption.
- No Raw Data Transmission: Only model parameters (weights or gradients) are shared between nodes and the aggregator.

## 🧠 Intrusion Detection Model
Each node:
- Preprocesses its local logs (e.g., extracting IP addresses, traffic patterns, protocol types).
- Performs consistent feature engineering to ensure compatibility across nodes.
- Trains a local machine learning model to classify normal vs. malicious traffic.

## 🔄 Model Update and Aggregation Flow
- Local Training: Each node trains on its own data.
- Model Update: Sends learned parameters to the central server.
- Federated Averaging: Central server aggregates updates.
- Global Model Distribution: Sends back the updated model for the next training round.

## 🧰 Tech Stack
- Languages: Python, Bash
- ML Libraries: Scikit-learn, PyTorch
- Security: OpenSSL, HTTPS/TLS
- Web & API: Flask
- Environment: Linux, Docker
