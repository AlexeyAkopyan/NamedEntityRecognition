<div align="center">

  # Named Entity Recognition
</div>
<img width="987" alt="image" src="https://github.com/AlexeyAkopyan/NamedEntityRecognition/assets/45924298/6a30c355-2ca7-4a2a-8f4a-c99d5851c358">

## Overview
This repository presents a solution for the fine-tuning [DeBERTa-v3](microsoft/deberta-v3-base) model for the Named Entity Recognition task on the English part of the [MutiNERD](https://huggingface.co/datasets/Babelscape/multinerd) dataset. There are two system conditions: 

- **System A** – fine-tuning to predict all entity types  presented in the dataset;  
- **System B** – fine-tuning to predict only 5 entity types (PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM)) and convert all other types to O tag.

The repository contains:  
- ner_deberta_base_A.ipynb - jupyter notebook with fine-tuning, test and inference of the system A model  
- ner_deberta_base_B.ipynb - jupyter notebook with fine-tuning, test and inference of the system B model
- ner_deberta_large_lora_A.ipynb - jupyter notebook with fine-tuning with LoRA, test and inference of the system A model
- requirements.txt - file with neccessary dependencies
  
## Installation
```bash
pip install -r requirements.txt
```
## Usage
### DeBERTa-v3-base
You can easily try the fine-tuned DeBERTa-v3 models for both systems right on the Hugging Face website via the Inference API  
- system A: [alexeyak/deberta-v3-base-ner-A](https://huggingface.co/alexeyak/deberta-v3-base-ner-A#:~:text=F32-,Inference%20API,-Token%20Classification)  
- system B: [alexeyak/deberta-v3-base-ner-B](https://huggingface.co/alexeyak/deberta-v3-base-ner-B#:~:text=F32-,Inference%20API,-Token%20Classification)

or use a pipeline
```python
from transformers import pipeline

ner_pipeline = pipeline('ner', model='alexeyak/deberta-v3-base-ner-A')

text = "He scored his first Premier League goal of the season, assisting Fernandodias Torres."
result = ner_pipeline(text)
```
or load model directly
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("alexeyak/deberta-v3-base-ner-A")
model = AutoModelForTokenClassification.from_pretrained("alexeyak/deberta-v3-base-ner-A")
```
An example of usage is also provided in the _ner_deberta_base_A.ipynb_ notebook under the Inference section.

### DeBERTa-v3-large-lora
The solution with this approach implemented only for the system A. The adapter also can be found on Hugging Face, but the adapter do not have Inference API.
- system A: [alexeyak/deberta-v3-large-ner-lora](https://huggingface.co/alexeyak/deberta-v3-large-ner-lora)
  
Usage of DeBERTa-v3-large-lora is less straightforward. Firsly it is required to set a label2id dict, then initialize tokenizer and model from base pre-trained model and eventually apply trained LoRA to base model. 
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

label2id = {
    "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6,
    "B-ANIM": 7, "I-ANIM": 8, "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12,
    "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16, "B-FOOD": 17, "I-FOOD": 18,
    "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23, "I-MYTH": 24,
    "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27,"I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}

id2label = {v: k for k, v in label2id.items()}
label_list = [id2label[i] for i in range(len(id2label))]

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
model = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-v3-large")
peft_model = PeftModel.from_pretrained(model, "alexeyak/deberta-v3-large-ner-lora")
```

An example of usage is also provided in the _ner_deberta_large_lora_A.ipynb_ notebook under the Inference section.

## Results
<div align="center">
  
| Model                   | Accuracy | Precision | Recall | F1    |
|-------------------------|----------|-----------|--------|-------|
| DeBERTa-v3-base-A       | 0.988    | 0.943     | 0.963  | 0.953 |
| DeBERTa-v3-base-B       | 0.993    | 0.967     | 0.976  | 0.971 |
| DeBERTa-v3-large-lora-A | 0.984    | 0.916     | 0.941  | 0.929 |

</div>


Fine-tuning and evaluation charts (Weights & Biases)
- [System A](https://api.wandb.ai/links/leshaakopyan2019/7jg7r0sl)  
- [System B](https://api.wandb.ai/links/leshaakopyan2019/uvzlssov)
- [System A LoRA](https://api.wandb.ai/links/leshaakopyan2019/uqknstby)


## Report
In this solution, I fine-tuned the [DeBERTa-v3](https://arxiv.org/abs/2111.09543) model, as it is commonly used in Kaggle NLP competitions by top leaders. For both systems, I employed token label alignment, which assigns labels to each token in a word but only preserves the B-[tag] for the first token in the first word of a named entity. I anticipated that this approach would increase the count of each entity type and enable the model to learn from entire words rather than just the initial token.

I quickly achieved high-quality results with the DeBERTa-base model using nearly default hyperparameters and then focused on attaining similar results with the DeBERTa-large model. Due to restrictions in Google Colab GPU memory, instead of fine-tuning the entire model, I applied the [LoRA](https://arxiv.org/abs/2106.09685) method specifically to query and key matrices in attention heads and lowered the LoRA rank to reduce the number of trainable parameters. My goal was to maximize the training batch size, [believing](https://arxiv.org/abs/1804.00247) that larger batch sizes lead to better results with large models. To achieve this, I employed several techniques (FP16, gradient checkpoint, and gradient accumulation) to reduce memory usage for trainable parameters and increase the batch size. I also experimented with loading the model in quantized int8 mode but found it significantly slowed down training speed, so I abandoned this method. Additionally, I observed that incorporating warmup steps and setting a higher learning rate contributed to achieving better results in fewer steps.

Although I obtained decent results with the large model, I did not manage to match the performance of the base model. It's possible that I underestimated the impact of class imbalance (I did).  I also came across an [article](https://arxiv.org/abs/2310.01208) where authors fine-tuned the Llama-2 model for the NER task, but I didn't have enough time to explore this approach. Nonetheless, I achieved great results with the base model and satisfactory results with the large one.

