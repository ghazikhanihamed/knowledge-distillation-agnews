# Knowledge Distillation for AG News Classification

This project implements a knowledge distillation pipeline for text classification on the AG News dataset.

## Requirements

- Python 3.8+
- torch
- transformers
- datasets
- evaluate
- numpy
- pandas
- matplotlib

Install requirements with:

```bash
pip install torch transformers datasets evaluate numpy pandas matplotlib
```

## File & Folder Structure

File/Folder      Description
main.py          Main Python file to run the training and knowledge distillation pipelines.
main_results/    Contains the main results and plots from the final/optimized experiments.
results/         Additional results generated during different experimentations and runs.
other_py/        Additional Python files/scripts for ablation, testing, or auxiliary experiments.
5%/              Results and outputs when only 5% of the dataset was used for training.
20%/             Results and outputs when 20% of the dataset was used for training.


## How to Run

1.	Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/ghazikhanihamed/knowledge-distillation-agnews.git
cd knowledge-distillation-agnews
```
2.	Install dependencies as shown above.
3. 	Run the main experiment:
```bash
python main.py
```

## Contact
Hamed Ghazikhani
hamed.ghazikhani@gmail.com
