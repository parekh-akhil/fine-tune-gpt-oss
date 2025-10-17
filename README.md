# Fine-Tune GPT OSS 20B

A repository for fine-tuning the GPT OSS 20B model using the Unsloth platform to generate human-like responses. This setup uses efficient 4-bit quantization and LoRA (Low-Rank Adaptation) for memory-efficient training on consumer hardware.

## 🚀 Features

- **Efficient Training**: Uses Unsloth's optimized training with 4-bit quantization
- **LoRA Adaptation**: Memory-efficient fine-tuning without modifying the base model
- **Flexible Data Format**: Supports JSON, JSONL, and CSV input formats
- **Interactive Inference**: Test your fine-tuned model with interactive prompts
- **Configurable**: YAML-based configuration for easy experimentation
- **Production Ready**: Includes data preparation, training, and inference scripts

## 📋 Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended: 24GB+ VRAM for full training)
- 50GB+ free disk space

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/parekh-akhil/fine-tune-gpt-oss.git
cd fine-tune-gpt-oss
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
fine-tune-gpt-oss/
├── config/
│   └── training_config.yaml    # Training configuration
├── data/
│   ├── raw/                    # Raw data files
│   │   └── example_data.jsonl # Example dataset
│   └── processed/              # Processed data ready for training
├── models/                     # Fine-tuned models and checkpoints
├── scripts/
│   ├── prepare_data.py        # Data preparation script
│   ├── finetune.py           # Main training script
│   └── inference.py          # Inference script
├── requirements.txt           # Python dependencies
└── README.md
```

## 🎯 Quick Start

### Step 1: Prepare Your Data

The training data should be in instruction-response format. You can use the provided example or create your own:

**Example format (JSONL):**
```json
{"instruction": "What is machine learning?", "response": "Machine learning is..."}
{"instruction": "Explain quantum computing", "response": "Quantum computing is..."}
```

Prepare your data:
```bash
python scripts/prepare_data.py \
    --input data/raw/your_data.jsonl \
    --output-dir data/processed \
    --eval-ratio 0.1
```

### Step 2: Configure Training

Edit `config/training_config.yaml` to adjust training parameters:
- Model settings (quantization, sequence length)
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters (learning rate, batch size, epochs)
- Dataset paths and prompt template

### Step 3: Fine-tune the Model

Start training:
```bash
python scripts/finetune.py \
    --config config/training_config.yaml \
    --output models/fine_tuned_model
```

**Optional: Save 16-bit model for faster inference:**
```bash
python scripts/finetune.py \
    --config config/training_config.yaml \
    --output models/fine_tuned_model \
    --save-16bit
```

### Step 4: Run Inference

**Interactive mode:**
```bash
python scripts/inference.py \
    --model-path models/fine_tuned_model \
    --config config/training_config.yaml \
    --interactive
```

**Single prompt:**
```bash
python scripts/inference.py \
    --model-path models/fine_tuned_model \
    --config config/training_config.yaml \
    --instruction "Explain artificial intelligence in simple terms"
```

## 🎛️ Configuration Details

### Model Configuration

- `name`: Base model to fine-tune (default: EleutherAI/gpt-neox-20b)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Enable 4-bit quantization for memory efficiency

### LoRA Parameters

- `r`: LoRA rank, higher = more parameters (default: 16)
- `lora_alpha`: LoRA scaling factor (default: 16)
- `lora_dropout`: Dropout probability (default: 0.05)
- `target_modules`: Which model layers to adapt

### Training Parameters

- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Steps before updating weights
- `learning_rate`: Learning rate (default: 2e-4)
- `optim`: Optimizer (default: adamw_8bit for efficiency)

## 📊 Data Preparation Tips

1. **Quality over Quantity**: Focus on high-quality, human-like responses
2. **Diversity**: Include varied instructions and response styles
3. **Consistency**: Maintain consistent formatting and tone
4. **Length**: Aim for responses between 50-500 tokens for best results
5. **Balance**: Ensure your dataset covers different topics and complexities

## 💡 Training Tips

1. **Start Small**: Test with a small dataset first
2. **Monitor Loss**: Watch training loss to avoid overfitting
3. **Adjust LoRA Rank**: Higher rank = more capacity but slower training
4. **Gradient Accumulation**: Increase if you face memory issues
5. **Learning Rate**: Start with 2e-4, adjust if training is unstable

## 🔍 Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `load_in_4bit` if not already enabled
- Reduce `max_seq_length`

### Slow Training

- Increase `per_device_train_batch_size` if memory allows
- Use gradient checkpointing (enabled by default)
- Ensure you're using the correct GPU drivers

### Poor Quality Responses

- Increase training data quality and quantity
- Train for more epochs
- Adjust prompt template to better match your use case
- Increase LoRA rank for more capacity

## 📚 Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [GPT-NeoX Model](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 🙏 Acknowledgments

- [Unsloth AI](https://github.com/unslothai/unsloth) for the optimized training framework
- [EleutherAI](https://www.eleuther.ai/) for the GPT-NeoX model
- [Hugging Face](https://huggingface.co/) for the transformers library

## 📧 Contact

For questions or issues, please open an issue on GitHub.