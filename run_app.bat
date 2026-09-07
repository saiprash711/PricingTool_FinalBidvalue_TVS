@echo off
title TVS Mobility - AP/TS Vehicle Price Forecasting Tool
echo =======================================================
echo  TVS Mobility - AP/TS Vehicle Price Forecasting Tool
echo =======================================================
echo Launching Streamlit Web App...
python -m streamlit run app.py --server.headless=false
pause
