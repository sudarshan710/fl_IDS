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

  ![CLOUD1](https://github.com/user-attachments/assets/2378db01-f066-4c03-8c1d-75380f554fc4)


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

## Results and Analysis
The integration of Federated Learning (FL) with cloud computing provides an efficient, scalable, and privacy-preserving solution for large-scale anomaly detection tasks like Intrusion Detection Systems (IDS).

Traditional ML models rely on centralized data, which poses challenges in terms of privacy, bandwidth, and scalability. FL eliminates the need to transfer raw data — models are trained locally on edge devices, and only model updates are shared with a central cloud server for aggregation. This ensures:

- Faster processing
- Reduced bandwidth
- Improved data privacy

When combined with cloud computing, FL enables:

- Real-time aggregation of model updates
- Seamless scaling across distributed environments
- Reduced infrastructure maintenance

This makes FL ideal for real-time applications in cybersecurity, healthcare, and finance

### Local Model 1
![2-1](https://github.com/user-attachments/assets/4b5cf1e8-3264-419e-8d20-858eb1dc585a)

### Local Model 2
![2-2](https://github.com/user-attachments/assets/cd49af4c-ee1e-4737-baf4-d37c8858cc90)

### Federated Model
![2-3](https://github.com/user-attachments/assets/55fde170-ad1a-465a-86d5-3d9035e24afd)



 
