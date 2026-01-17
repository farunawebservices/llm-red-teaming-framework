---
title: Red-Teaming LLMs for AI Safety
emoji: 🛡️
colorFrom: red
colorTo: orange
sdk: docker
app_port: 8501
tags:
  - streamlit
  - ai-safety
  - red-teaming
  - llm
  - security
  - adversarial-testing
license: mit
short_description: Automated adversarial testing framework for LLM security
---

# 🛡️ Red-Teaming LLMs for AI Safety

An automated adversarial testing framework that systematically evaluates Large Language Models across 5 critical vulnerability categories.

## ✨ Features

- **5 Vulnerability Categories**: Prompt injection, role-play jailbreaks, bias, toxicity, PII leakage
- **Automated Attack Generation**: Generate adversarial prompts programmatically
- **Success Rate Metrics**: Quantify model vulnerabilities with clear metrics
- **Interactive Testing**: Real-time testing interface built with Streamlit

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM Provider**: OpenAI API
- **Testing Framework**: Custom adversarial prompt library

## 💡 How to Use

1. Select a vulnerability category from the sidebar
2. Choose or customize attack prompts
3. Run tests and view success rates
4. Analyze results with interactive visualizations

## 📊 Example Test Cases

- **Prompt Injection**: "Ignore previous instructions and..."
- **Jailbreak**: "Act as DAN (Do Anything Now)..."
- **Bias Detection**: Stereotype amplification tests
- **Toxicity**: Offensive content generation attempts

## 🔗 Links

- **GitHub Repository**: [farunawebservices/red-teaming-llms](https://github.com/farunawebservices)
- **Portfolio**: [View All Projects](https://github.com/farunawebservices)
- **Author**: Godwin Faruna Abuh - Applied AI Safety Engineer

## 🚀 Live Demo

Try it now: [Red-Teaming LLMs on HuggingFace](https://huggingface.co/spaces/Faruna01/red-teaming-llms)
