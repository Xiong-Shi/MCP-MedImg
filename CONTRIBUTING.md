# Contributing to MCP-MedImg

Thank you for your interest in improving MCP-MedImg! This project welcomes contributions in the following areas.

## How to Contribute

### Reporting Issues
- Use [GitHub Issues](https://github.com/Xiong-Shi/MCP-MedImg/issues) to report bugs or request features.
- Include a minimal reproduction for bugs, and reference the relevant paper section for protocol questions.

### Code Contributions
1. Fork the repository and create a feature branch.
2. Ensure your changes include tests (see `tests/` for patterns).
3. Run the test suite locally:
   ```bash
   pytest tests/ -v --tb=short
   ```
4. Submit a Pull Request with a clear description of the change and its motivation.

### Areas of Interest
- Additional DICOM modality adapters (L1 Transport)
- Security hardening and audit-trail enhancements (L2)
- New AI model backends and routing logic (L3)
- Clinical workflow integrations (L4)

## Development Setup

```bash
git clone https://github.com/Xiong-Shi/MCP-MedImg.git
cd MCP-MedImg
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if available
```

## Code of Conduct

Be respectful and constructive. All contributions are reviewed against the project's goal of safe, reproducible medical-imaging AI integration.
