# LoRA Finetuning: RoBERTa-Base on Few-NERD (Supervised)

This section documents the full finetuning pipeline used to train a parameter-efficient NER model using **LoRA adapters** on top of **RoBERTa-base**, optimized for deployment on low-memory environments (AWS EC2, Lambda, containers).

## What is Named Entity Recognition (NER)?

Named Entity Recognition (NER) is a core Natural Language Processing (NLP) task where a model identifies and classifies meaningful pieces of text (called entities) into predefined categories.

In simple terms, NER turns unstructured text into structured information by answering two questions:

What is the entity? (e.g., “Barack Obama”)

What type is it? (e.g., person-politician)

---

## 📊 Dataset Configuration

We use the **Few-NERD (supervised)** setting with fine-grained entity types. A larger dataset is sampled for LoRA efficiency:

| Split      | Samples Used | % of Original        |
| ---------- | ------------ | -------------------- |
| Train      | **100,000**  | 75.9%                |
| Validation | **15,000**   | 79.7%                |
| Test       | **37,648**   | 100% (full test set) |

**Total training samples:** 115,000
**Expected training time:** ~20–30 minutes (A100 / T4 with LoRA)

---

## ⚙️ LoRA Configuration

| Parameter      | Value                |
| -------------- | -------------------- |
| Task           | Token Classification |
| Rank (r)       | **16**               |
| Alpha          | **32**               |
| Dropout        | **0.1**              |
| Target Modules | `query`, `value`     |
| Bias           | none                 |

---

## 🧮 Parameter Efficiency

| Metric                  | Value                |
| ----------------------- | -------------------- |
| Total model parameters  | **124,747,910**      |
| Trainable params (LoRA) | **641,347**          |
| Trainable %             | **0.51%**            |
| Frozen params           | 124,106,563 (99.49%) |

**Parameter efficiency:** 194.5× fewer trainable parameters
**Memory savings:** ~99.5% reduction in trainable memory footprint

---

## 🏋️ Training Configuration

| Setting               | Value    |
| --------------------- | -------- |
| Epochs                | **5**    |
| Batch size            | 16       |
| Gradient accumulation | 2        |
| Effective batch size  | **32**   |
| Learning rate         | **3e-4** |
| Warmup ratio          | 0.1      |
| Mixed precision       | FP16     |
| Total training steps  | ~15,625  |

---

## 📈 Validation Metrics (per Epoch)

| Epoch | Train Loss | Val Loss | Precision | Recall | F1         |
| ----- | ---------- | -------- | --------- | ------ | ---------- |
| 1     | 0.2823     | 0.2627   | 0.6284    | 0.6739 | 0.6503     |
| 2     | 0.2627     | 0.2478   | 0.6488    | 0.6843 | 0.6661     |
| 3     | 0.2464     | 0.2450   | 0.6449    | 0.6936 | 0.6683     |
| 4     | 0.2350     | 0.2390   | 0.6617    | 0.6940 | 0.6774     |
| 5     | 0.2303     | 0.2380   | 0.6607    | 0.7003 | **0.6800** |

**Best validation F1:** 0.6800

---

## 🧪 Test Set Results

| Metric         | Score      |
| -------------- | ---------- |
| **Test F1**    | **0.6744** |
| Test Precision | 0.6539     |
| Test Recall    | 0.6961     |
| Test Loss      | 0.2431     |

---

## 📦 Model Artifacts

### **LoRA Adapters MERGED TO BASE MODEL**

* Path: `./roberta-lora-fewnerd-merged`
* Loads like a standard Transformers model:

  ```python
  AutoModelForTokenClassification.from_pretrained("roberta-lora-fewnerd-merged")
  ```

---

