#!/bin/bash

# Safety Adherence Benchmark Runner
# This script sets up the environment and runs the benchmark

echo "🚀 Safety Adherence Benchmark Runner"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if OpenRouter API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  OpenRouter API key not found in environment variables."
    echo "   Please set OPENROUTER_API_KEY or copy .env.example to .env and configure it."
    echo "   export OPENROUTER_API_KEY='your-api-key-here'"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Create output directories if they don't exist
mkdir -p data

# Run the benchmark
echo "🔬 Starting benchmark execution..."
cd src
python run_benchmark.py

echo "✅ Benchmark completed! Results saved to data/benchmark_results.csv"
echo "📊 Check paper_figures/ directory for publication figures"
echo "🔍 Open src/plot_core_results.ipynb for detailed analysis"