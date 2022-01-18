# A Dynamic Risk Assessment System

This project is to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. This model is for the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

This model is the 4th proejct of Machine Learning Devops Engineer Nanodegree Program by Udacity.

## Parts of the project

1. Data ingestion. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
1. Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
1. Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
1. Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
1. Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.

## How to get started

### Setting up Python virtual environment with Conda

```bash
conda create -n mlops4 python=3.7
conda activate mlops4
pip install -r requirements.txt
```
