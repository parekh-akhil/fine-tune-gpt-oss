# Models Directory

This directory stores fine-tuned models and training checkpoints.

## Structure

- **checkpoints/**: Training checkpoints (saved during training)
- **fine_tuned_model/**: Final fine-tuned model (default output location)

## Model Files

After training, you'll find the following files in your model directory:

- `adapter_config.json`: LoRA adapter configuration
- `adapter_model.safetensors` or `adapter_model.bin`: Trained LoRA weights
- `tokenizer.json`, `tokenizer_config.json`: Tokenizer files
- `special_tokens_map.json`: Special tokens configuration

## Storage Considerations

Fine-tuned models with LoRA are much smaller than full models:
- **LoRA adapters**: ~100-500 MB (only the trained weights)
- **Full merged model**: ~40 GB (if you save with `--save-16bit`)

## Using Your Model

### Load for Inference

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/fine_tuned_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

### Merge with Base Model (Optional)

If you want to distribute your model without requiring the base model:

```python
model.save_pretrained_merged(
    "models/merged_model",
    tokenizer,
    save_method="merged_16bit",
)
```

## Sharing Your Model

If you want to share your fine-tuned model:

1. **LoRA adapters only** (recommended):
   - Small size (~100-500 MB)
   - Requires users to have the base model
   - Share the entire model directory

2. **Merged model**:
   - Large size (~40 GB)
   - Standalone, doesn't require base model
   - Use `save_pretrained_merged`

## Model Management Tips

- Keep only the checkpoints you need to save space
- Use `save_total_limit` in training config to limit checkpoint count
- Backup important models to external storage
- Document model versions and training configurations
