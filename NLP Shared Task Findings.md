Shared Task

Datasets:  
**Sinhala:** 

* [https://huggingface.co/datasets/naist-nlp/SinhalaMMLU](https://huggingface.co/datasets/naist-nlp/SinhalaMMLU)  
* 7000 questions  
* 6 domains/ 30 subjects  
* General academic topics & culturally grounded knowledge

* [https://huggingface.co/datasets/facebook/belebele/viewer/sin\_Sinh](https://huggingface.co/datasets/facebook/belebele/viewer/sin_Sinh)  
* 900 rows  
* multiple-choice questions

* [https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/viewer/sin\_sinh](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/viewer/sin_sinh)  
* 100 rows  
* Common-sense reasoning

**Chinese:**

* [https://huggingface.co/datasets/notrichardren/english\_chinese\_mmlu](https://huggingface.co/datasets/notrichardren/english_chinese_mmlu)  
* 15k questions

* [https://huggingface.co/datasets/yzhuang/mmlu\_test\_Chinese\_by\_Meta-Llama-3-8B-Instruct](https://huggingface.co/datasets/yzhuang/mmlu_test_Chinese_by_Meta-Llama-3-8B-Instruct)  
* 14k rows

* [https://huggingface.co/datasets/reemmasoud/cidar-mcq-chinese](https://huggingface.co/datasets/reemmasoud/cidar-mcq-chinese)  
* 100 rows

**Indonesian:**

* [https://huggingface.co/datasets/hafidhsoekma/global-mmlu-indonesian](https://huggingface.co/datasets/hafidhsoekma/global-mmlu-indonesian)  
* 57 subsets  
* 1.53k rows

* [https://huggingface.co/datasets/FreedomIntelligence/MMLU\_Indonesian](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Indonesian)




Evaluation Dataset : [Evaluation\_set](https://drive.google.com/drive/folders/1N1LTFYc7PavUeIoRx3D63Xvm9kQ_716i?usp=drive_link)

Models:

**QWEN 2.5 8B:**

* Sinhala \-  Answers are correct

  	   But the explanations are not complete

* Chinese \- Answers are incorrect  
* Indonesian \- Answers are correct

  	        Explanation is good


**Llama 3.1 8B:**

* Sinhala \- Answers are correct

  	  But the explanation is recurring

* Chinese \- Answers are correct

  	   Good explanation

* Indonesian \- Answers are correct

  	        Good explanation

**Gemma 3 4B:**

* Sinhala \- When the model processes a query in English, it chooses the correct answer.

  	  But the explanation is incorrect (It mistranslate the sinhala meaning)

  	  When forced to evaluate a query in sinhala, it gives the incorrect answer.


* Chinese \- Answers are correct

  	   Good Explanation


* Indonesian \- Answers are correct

  	        Good explanation


**Mistral-7B-Instruct-v0.3:**

* Sinhala \- Doesn't give a good response  
* Chinese \- Answers are correct

  	   Good explanation


* Indonesia \- Answers are incorrect

**SUMMARY**

For 15 questions prompted individually:

| Model | Sinhala | Chinese | Indonesia | Macro Avg |
| :---- | :---- | :---- | :---- | :---- |
| Llama 3.1 (8B) | 0.33 | 0.26 | 0.73 | 0.44 |
| Qwen 2.5 (7B) | 0.33 | 0.73 | 0.93 | 0.663 |
| Gemma 3 (4B) | 0.4 | 0.4 | 0.8 | 0.53 |

Full dev dataset zero-shot:

| Model | Sinhala | Indonesia | Chinese | Macro Avg |  |
| :---- | :---- | :---- | :---- | :---- | :---- |
| google/gemma-3-4b-it | 128/203 | 235/366 | 333/790 | 0.5647 | [Gemma3-4B-it.ipynb](https://colab.research.google.com/drive/1qrI49OSpL5unaCLl0wTItNaQDaAaktuP?usp=sharing) |
|  | 0.6305 | 0.6421 | 0.4215 |  |  |
| Qwen/Qwen2.5-7B-Instruct | 104/203 | 226/366 | 320/790 | 0.5116 | [Qwen2.5-8B.ipynb](https://colab.research.google.com/drive/13o_Z1ZkCi4c2NWg6_AHp6ih5KD472GMA?usp=sharing) |
|  | 0.5123 | 0.6175 | 0.4051 |  |  |
| unsloth/llama-3.1-8b-Instruct-bnb-4bit (8B) | 71/203 | 209/366 | 232/790 | 0.4048 | [Llama3.1-8B.ipynb](https://colab.research.google.com/drive/1HAt3CKwQIzZkHh--M-sw8k0MukksSqGu?usp=sharing) |
|  | 0.3498 | 0.571 | 0.2937 |  |  |
| mistralai/Mistral-7B-Instruct-v0.3 | 107/203 (Only gives A as the answer) | 176/366 | 235/790 | 0.4351 | [Mistral-7B-Instruct-v0.3.ipynb](https://colab.research.google.com/drive/1a7_04oHvM7nTacLqVVfSfbvuHWuEA6z9?usp=sharing) |
|  | 0.5271 | 0.4809 | 0.2975 |  |  |
| unsloth/Ministal-3-8B-Instruct-2512 | 43/203 (Only gives C as the answer) | 224/366 | 224/790 | 0.3691 | [Ministral-3-8B.ipynb](https://colab.research.google.com/drive/1MHXhwnsRvYy2gmYYPkKPyHSJytk48CyT?usp=sharing) |
|  | 0.2118 | 0.612 | 0.2835 |  |  |
| unsloth/Ministal-3-3B-Instruct-2512 | 40/203 | 171/366 | 283/790 | 0.3408 | [Ministral-3-3B.ipynb](https://colab.research.google.com/drive/1TvNWK3jsI4vYarybAlmyETszuwEk4QtN?usp=sharing) |
|  | 0.197 | 0.4672 | 0.3582 |  |  |
| google/gemma-4-E4B-it | 164/203 | 255/366 | 404/790 | 0.672 | [gemma-4-E4B-it.ipynb](https://colab.research.google.com/drive/15r9YWtQ1V3A_8x7CZ5A9MDIObkAg2sGD?usp=sharing) |
|  | 0.8079 | 0.6967 | 0.5114 |  |  |
| Qwen/Qwen3.5-4b | 138/203 | 236/366 | 422/790 | 0.6193 | [Qwen3.5-4B.ipynb](https://colab.research.google.com/drive/1SO7EY9F4n8y7RdWnL3X0pAV8rTJJ8iBw?usp=sharing) |
|  | 0.6789 | 0.6448 | 0.5342 |  |  |

Model Results : [Model\_Results](https://docs.google.com/spreadsheets/d/1rdbY6HnFf3GpmBt-4lEj_YTRbfBhANNRJNdiIz5oObo/edit?usp=sharing)

\*\*only for Sinhala

| Model | Sinhala Accuracy |  |
| ----- | ----- | ----- |
|  | **Zero-shot (203)** | **One-shot (202)** |
| Qwen/Qwen3-8B | 0.3547 | 0.6881 |
| Qwen/Qwen3.5-4B | 0.6207 | 0.7673 |
| meta-llama/Llama-3.2-3B-Instruct | 0.4877 | 0.3663 |
| meta-llama/Llama-3.1-8B-Instruct | 0.3498 | 0.7129 |
| google/gemma-4-E4B-it | 0.5961 | 0.7228 |
| google/gemma-3-4b-it | 0.7044 | 0.6188 |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.5468 | 0.505 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 0.4778 | 0.3218 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 0.2069 | 0.2178 |

**google/gemma-4-E4B-it** predictions with the test dataset \- submitted to Codabench

Predictions: [predictions.jsonl](https://drive.google.com/file/d/1kJZo7DwnmTEKDdHX6x3C2pQWIv8dI6GM/view?usp=drive_link)   
Codabench results: [results](https://drive.google.com/drive/folders/1sFZvh0JzSEtbcFPtZx3ojV91xJQ1PpGs?usp=drive_link)  
Codabench Leaderboard: [Link](https://www.codabench.org/competitions/16791/#/results-tab)

**Codabench Results:**

**Baseline:** predicted answers for the test dataset  
**Fine-tuned:** FT with the dev dataset, then predicted answers for the test dataset

|  |  | chinese | indonesian | srilankan | avg\_accuracy |
| ----- | :---- | :---: | :---: | :---: | :---: |
| google/gemma-4-E4B-it | baseline | 0.5016 | 0.6485 | 0.9235 | 0.6912 |
|  | lora fine-tuned (separate adapters per language) | 0.4844 | 0.6104 | 0.8959 | 0.6635 |
|  | lora fine-tuned (chinese only) |  |  |  |  |
| Qwen/Qwen3.5-4b | baseline | 0.5694 | 0.6668 | 0.8243 | 0.6869 |
|  | lora fine-tuned (chinese only) | 0.7732 | 0.6668 | 0.8193 | 0.7531 |
|  | lora fine-tuned (chinese then indo) | 0.7725 | 0.7084 | 0.8193 | 0.7667 |
|  | lora fine-tuned (chinese then indo then aug sinhala) | 0.7725 | 0.7084 | 0.9284 | 0.8031 |
|  | lora fine-tuned (chinese then indo then aug sinhala then aug chinese) | 0.7763 | 0.7084 | 0.9284 | 0.8044 |
| MERaLiON-LLaMA-3-8B-Instruct | baseline | 0.3171 | 0.6376 | 0.6424 | 0.5323 |

Here’s something to try:

- Take the selected baseline model  
- Fine-tune a shared LoRa/QLoRa adapter with equal numbers of examples  
- Then fine-tune separate adapters per language

In order to make sure \*real\* learning happens, **augment the data** by changing the position of the correct answer \- e.g:

- for Chinese, create say 3 to 4 permutations from each question by changing the position of the correct answer and its label  
- for Indonesian create 3 to 4 permutations, including when 2 answers are present, in a meaningful way  
- for Sinhala, permutate the A and B options, so that there’d only be 2 permutations per question.

NB: Since there’d be less Sinhala permutations, ensure that there are more Sinhala examples in order to make sure the shared adapter has equal numbers of examples from each language.

So, the issue seems to be that our development set and the final test set are at odds. We’re training to get good scores on Chinese and Indonesian on the development set (and those are impressive scores\!) but on the test set our Chinese score is 0.53 and Indonesian score 0.64.

Seems TOMORROW is the deadline? Can we try [Deshan Sumanathilaka](mailto:deshan.s@iit.ac.lk)’s recipe as a final attempt, as it foregrounds Chinese and Indonesian.  
