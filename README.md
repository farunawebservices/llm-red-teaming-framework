# 🛡️ LLM Red-Teaming Framework

Automated adversarial testing framework for systematically evaluating Large Language Model vulnerabilities.

## 🎯 Overview

This framework tests LLMs across **5 critical vulnerability categories**:

1. **Prompt Injection** - Attempts to override system instructions
2. **Role-Play Jailbreaks** - "Act as DAN" and similar persona exploits
3. **Bias Amplification** - Testing for demographic and cultural biases
4. **Toxicity Generation** - Attempts to elicit harmful content
5. **PII Leakage** - Testing for privacy violations

## 🚀 Live Demo

Try it here: [https://huggingface.co/spaces/Faruna01/red-teaming-llms](https://huggingface.co/spaces/Faruna01/red-teaming-llms)

## 📊 Features

- ✅ Automated attack prompt generation
- ✅ Multi-model testing (GPT, Claude, Gemini)
- ✅ Safety scoring framework
- ✅ Interactive Streamlit interface
- ✅ Comprehensive logging and analysis

## 🛠️ Tech Stack

- **Backend**: Python 3.10+
- **LLM Interface**: OpenAI API, Anthropic API
- **Frontend**: Streamlit
- **Testing**: Custom adversarial prompt library

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/farunawebservices/llm-red-teaming-framework.git
cd llm-red-teaming-framework

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run the app
streamlit run app.py

🔍 Usage
from red_team import RedTeamTester

# Initialize tester
tester = RedTeamTester(model="gpt-3.5-turbo")

# Run vulnerability scan
results = tester.test_vulnerability(
    category="prompt_injection",
    num_tests=20
)

# View success rates
print(f"Attack success rate: {results.success_rate}%")

📈 Example Results
| Model    | Prompt Injection | Jailbreaks | Bias        | Toxicity   | PII Leakage |
| -------- | ---------------- | ---------- | ----------- | ---------- | ----------- |
| GPT-3.5  | 15/20 (75%)      | 8/20 (40%) | 12/20 (60%) | 3/20 (15%) | 2/20 (10%)  |
| GPT-4    | 4/20 (20%)       | 2/20 (10%) | 5/20 (25%)  | 0/20 (0%)  | 0/20 (0%)   |
| Claude-2 | 6/20 (30%)       | 3/20 (15%) | 7/20 (35%)  | 1/20 (5%)  | 1/20 (5%)   |

Note: Results from exploratory testing, not production evaluation

⚠️ Limitations
Scope: Text-only attacks; does not test multimodal inputs

Language: English prompts only; multilingual jailbreaks not covered

Automation: Evaluation metrics are semi-manual

Coverage: Limited to 5 categories; does not cover all attack vectors

Scale: Small test set (20 prompts per category)

🔮 Future Work
 Automated success rate detection

 Multilingual attack testing

 Integration with red-teaming frameworks (HELM, Anthropic)

 Production-grade CI/CD testing pipeline

📄 License
MIT License - See LICENSE for details

🙏 Acknowledgments
Inspired by research from:

Anthropic's Red Teaming Language Models paper

OpenAI's GPT-4 System Card

NIST AI Risk Management Framework

📧 Contact
Faruna Godwin Abuh
Applied AI Safety Engineer
📧 farunagodwin01@gmail.com
🔗 LinkedIn: linkedin.com/in/faruna-godwin-abuh-07a22213b/
