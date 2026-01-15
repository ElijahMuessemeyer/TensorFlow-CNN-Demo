#!/bin/bash
# Double-click this file in Finder to start Jupyter Notebook

cd ~/tensorflow-demo
source venv/bin/activate
echo "Starting Jupyter Notebook..."
echo "Your browser should open automatically."
echo "If not, go to: http://localhost:8888"
echo ""
echo "Press Ctrl+C in this window to stop Jupyter when done."
echo ""
jupyter notebook
