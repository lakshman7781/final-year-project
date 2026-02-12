# AWS Deployment Guide

## 1. Connect to AWS
Use the provided `.pem` key to SSH into your instance:
```bash
ssh -i "fnal-year.pem" ec2-user@52.66.234.59
```
*(If using Ubuntu, replace `ec2-user` with `ubuntu`)*

## 2. Setup Environment
Once connected, navigate to the project folder and run:
```bash
cd fake_news_app
chmod +x setup_aws.sh
./setup_aws.sh
```
This script will:
- Install Python and dependencies.
- Automatically extract the pre-loaded Kaggle datasets (`True.csv.zip`, `Fake.csv.zip`).

## 3. Train Model (Real Data)
Run the training script to build the model using the new 40k+ articles:
```bash
python3 ml_engine/train_model.py
```

## 4. Run Application
Start the server:
```bash
python3 backend/app.py
```
Access the app at `http://52.66.234.59:5001`.

## 5. Demo
The code is structured for a "Step-by-Step" demo. Open `ml_engine/train_model.py` to walk through the data pipeline.
