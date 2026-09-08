@echo off
title TVS Certified - Tamil Nadu Vehicle Price Forecasting Tool
echo =======================================================
echo  TVS Certified - Tamil Nadu Vehicle Price Forecasting Tool
echo =======================================================
echo Launching Streamlit Web App...
python -m streamlit run app.py --server.headless=false
pause
