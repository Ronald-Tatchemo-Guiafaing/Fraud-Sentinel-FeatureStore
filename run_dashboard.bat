@echo off
cd /d "%~dp0"
echo Fraud Sentinel Dashboard
pip install streamlit -q
streamlit run dashboard/app.py
