# LLM Modules: Knowledge Transfer from a Large to a Small Model using Enhanced Cross-Attention

This repository contains the source code for a paired LLM model that transfers knowledge from a large pre-trained model (Qwen2-1.5B) to a smaller model (GPT-Neo-125M) using an Enhanced Cross-Attention mechanism.

## Repository Contents

- **model.py**: Source code for the paired model, including the implementation of training and inference routines.
- **model_checkpoint.pth**: Model weights checkpoint saved after training.
- **compare-responses-from-models.md**: Contains answers to test questions for different models (necessary for my research)
- **paper_llm_modules.pdf**: LLM Module Research Paper

## Pre-trained Weights

The original pre-trained weights for the model are available on Hugging Face.

You can download them from: [https://huggingface.co/kkolomeitsev/llm-modules](https://huggingface.co/kkolomeitsev/llm-modules)

## Requirements

The project requires the following libraries:

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) (version 1.7 or above)
- [Transformers](https://huggingface.co/transformers/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [tqdm](https://github.com/tqdm/tqdm)

You can install the required packages via pip:

```bash
pip install torch transformers datasets tqdm
```

Alternatively, you can create and activate a virtual environment:

```bash
python -m venv venv
# For Linux/MacOS:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
pip install torch transformers datasets tqdm
```

## Running Training

By default, the model.py file is configured to run the training process. To start training, simply execute:

```bash
python model.py
```

The model will be trained according to the specified parameters, and the checkpoint will be saved as `model_checkpoint.pth`.

## Running Inference (Interactive Chat)

To run inference, you need to disable the training code and enable the interactive chat mode. In the model.py file, comment out the training function call and uncomment the interactive_chat() call. For example, modify the main section as follows:

```bash
if __name__ == "__main__":
    # main()  # Comment this line to disable training
    interactive_chat()  # Uncomment this line to run inference
```

Then run:

```bash
python model.py
```

An interactive session will start in the console, allowing you to enter queries and view the model's generated responses.
Additional Notes

- Ensure you have sufficient computational resources for training the model.
- For reproducibility, consider setting a fixed seed for random operations.
- You can adjust model parameters and training settings directly in the model.py file.

## Citation

```
@misc{Kolomeitsev2023LLMModules,
      title = {LLM Modules: Knowledge Transfer from a Large to a Small Model using Enhanced Cross-Attention},
      author = {Konstantin Kolomeitsev},
      year = {2025}}
}

```

## Contact

If you have any questions, please raise an issue or contact with me [uol92kot@gmail.com](uol92kot@gmail.com).
