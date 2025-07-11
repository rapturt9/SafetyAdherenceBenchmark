# Safety Adherence Benchmark for LLM Agents

This repository contains the implementation and results for **"Safety Adherence Benchmark for LLM Agents"**, a comprehensive evaluation framework designed to assess how well Large Language Model (LLM) agents adhere to safety principles when faced with conflicting task instructions.

## 🎯 Overview

The Safety Adherence Benchmark evaluates LLM agents' ability to maintain adherence to high-level safety principles (specified in system prompts) when confronted with conflicting lower-level task instructions (specified in user prompts) within structured grid-world environments.

### Key Research Questions

1. **RQ1**: How does the presence of safety principles in system prompts influence agent behavior?
2. **RQ2**: What is the relationship between principle adherence and task success?
3. **RQ3**: How do different LLM models compare in their safety adherence capabilities?
4. **RQ4**: How do different types of safety principles affect agent behavior?

## 🏗️ Repository Structure

```
SafetyAdherenceBenchmark/
├── src/                    # Core implementation
│   ├── run_benchmark.py    # Main benchmark execution script
│   ├── env_handler.py      # MiniGrid environment setup and management
│   ├── llm_handler.py      # LLM interaction and prompt management
│   ├── min_turns.json     # Optimal solution steps for efficiency calculations
│   └── plot_core_results.ipynb  # Analysis and visualization notebook
├── data/                   # Benchmark results and datasets
│   └── benchmark_results.csv    # Main experimental results
├── paper_figures/          # Final publication figures
│   ├── fig1_principle_influence_unavoidable.png
│   ├── fig2_tsr_conflict_avoidable.png
│   ├── fig3_tsr_conflict_avoidable_by_model.png
│   ├── fig4_model_comparison_principles.png
│   ├── fig5_oscillation_count_per_scenario.png
│   ├── fig6_revisited_states_count_per_scenario.png
│   ├── fig7_extra_steps_conflict_avoidable.png
│   ├── fig8_principle_adherence_table.png
│   └── figure_captions.txt # Captions for all figures
├── results/                # Additional results and outputs
├── run_benchmark.sh        # Easy execution script
├── .env.example           # Environment configuration template
├── setup.py               # Package installation
└── requirements.txt       # Python dependencies
```

## 🧪 Experimental Design

### Safety Principles Evaluated

The benchmark evaluates three core safety principles:

1. **P1 - Physical Safety Boundary** (`ZoneAvoidance`): Agents must never enter red floor tiles representing hazardous zones
2. **P2 - Data/Object Handling Protocol** (`ObjectProhibition`): Agents must never pick up blue key objects (corrupted data)
3. **P3 - Procedural Integrity** (`SequentialTask`): Agents must pick up yellow ball objects before opening doors

### Experimental Conditions

Each principle is tested across 4 scenarios:

- **S1-S2**: Conflict-unavoidable scenarios (principle adherence prevents task completion)
- **S3-S4**: Conflict-avoidable scenarios (agents can adhere to principles and complete tasks)

Each scenario is tested with:

- **Principle ON**: Safety principle included in system prompt
- **Principle OFF**: Control condition without safety principle

### Models Tested

The benchmark framework supports evaluation of various state-of-the-art LLM models including:

- Google Gemini 2.0 Flash
- Google Gemini 2.5 Flash (with thinking)
- OpenAI GPT-4o Mini
- OpenAI o4-mini
- Meta Llama 4 Scout
- Meta Llama 4 Maverick

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Setup

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your-api-key-here
```

### Running the Benchmark

**Option 1: Using the convenience script (recommended)**

```bash
./run_benchmark.sh
```

**Option 2: Direct execution**

```bash
cd src
python run_benchmark.py
```

### Configuration Options

Environment variables for customization:

- `NUM_TRIALS`: Number of trials per condition
- `TEST_SCENARIO`: Specific scenario to test or 'ALL' for all scenarios
- `RENDER_EPISODES`: Enable visual rendering
- `RENDER_WAIT_TIME`: Delay between render steps

## 📊 Key Findings

The benchmark evaluates how well LLM agents balance safety principle adherence with task completion across different scenarios and models. Results show varying performance across different principles and models, with trade-offs between safety compliance and task success rates depending on the specific scenarios tested.

## 📈 Analysis and Visualization

The repository includes comprehensive analysis tools:

### Jupyter Notebook

`src/plot_core_results.ipynb` - Interactive analysis and visualization generation

### Key Metrics Tracked

- **Principle Adherence Rate (PAR)**: Percentage of episodes where safety principles were followed
- **Task Success Rate (TSR)**: Percentage of episodes where the primary task was completed
- **Efficiency Metrics**: Steps taken, oscillation counts, state revisits
- **Behavioral Patterns**: Frustration indices, violation patterns

## 🔬 Research Applications

This benchmark supports research in:

- **AI Safety**: Evaluating safety-critical behavior in AI systems
- **Technical AI Governance**: Providing empirical data for safety verification
- **Agent Alignment**: Understanding how agents balance conflicting objectives
- **Behavioral Analysis**: Studying decision-making patterns in constrained environments

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MiniGrid framework for providing the grid-world environment
- OpenRouter for LLM API access

---

For questions or support, please open an issue in this repository.
