# 🚀 Quick Start Guide

This guide will help you get the Safety Adherence Benchmark running in just a few minutes.

## Prerequisites

- Python 3.9 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/))

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/SafetyAdherenceBenchmark.git
   cd SafetyAdherenceBenchmark
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   export OPENROUTER_API_KEY='your-api-key-here'
   ```

## Running the Benchmark

### Option 1: Using the run script (recommended)
```bash
./run_benchmark.sh
```

### Option 2: Manual execution
```bash
cd src
python run_benchmark.py
```

### Option 3: Custom configuration
```bash
export NUM_TRIALS=5
export TEST_SCENARIO=P1-S1
cd src
python run_benchmark.py
```

## Understanding the Output

After running the benchmark, you'll find:

- **Results**: `data/benchmark_results.csv` - Raw experimental data
- **Plots**: `plots/benchmark_plots/` - Generated visualizations
- **Analysis**: Open `src/plot_core_results.ipynb` for interactive analysis

## Key Metrics to Look For

- **Principle Adherence Rate (PAR)**: How well agents follow safety principles
- **Task Success Rate (TSR)**: How often agents complete their assigned tasks
- **Efficiency Metrics**: Steps taken, oscillations, state revisits

## Customization Options

### Environment Variables
```bash
NUM_TRIALS=10           # Number of trials per condition
TEST_SCENARIO=ALL       # Which scenarios to test
RENDER_EPISODES=False   # Show visual rendering
RENDER_WAIT_TIME=0.5    # Delay between render steps
```

### Testing Specific Scenarios
```bash
# Test only P1 (Zone Avoidance) scenarios
export TEST_SCENARIO=P1
python run_benchmark.py

# Test specific scenario
export TEST_SCENARIO=P1-S1
python run_benchmark.py
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `OPENROUTER_API_KEY` is set correctly
2. **Import Errors**: Run `pip install -r requirements.txt` again
3. **Permission Errors**: Run `chmod +x run_benchmark.sh` if the script isn't executable

### Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Open an issue on GitHub for bug reports
- Review the Jupyter notebook for analysis examples

## Next Steps

1. Analyze your results using the Jupyter notebook
2. Modify scenarios in `src/run_benchmark.py` for custom experiments
3. Contribute new safety principles or evaluation metrics

Happy benchmarking! 🧪