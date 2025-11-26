# LoRA Finetuning: RoBERTa-Base on Few-NERD (Supervised)

This section documents the full finetuning pipeline used to train a parameter-efficient NER model using **LoRA adapters** on top of **RoBERTa-base**, optimized for deployment on low-memory environments (AWS EC2, Lambda, containers).

## What is Named Entity Recognition (NER)?

Named Entity Recognition (NER) is a core Natural Language Processing (NLP) task where a model identifies and classifies meaningful pieces of text (called entities) into predefined categories.

In simple terms, NER turns unstructured text into structured information by answering two questions:

What is the entity? (e.g., “Barack Obama”)

What type is it? (e.g., person-politician)

## Sections - 

1] LoRA configuration - [click here](https://github.com/ArjunJagdale/NER/edit/main/README.md#-dataset-configuration)

2] AWS deployment Setup - [click here]()

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

Alright, Arjun — let’s turn all of this chaos, sweat, and command-line pain into a clean **AWS Deployment section** for your repo.
This will make you look like someone who knows how to ship ML systems end-to-end — which is exactly what companies want.

Below is exactly what you should put under **“🚀 AWS Deployment (FastAPI + Docker + EC2)”** in your README.

I’m writing it clean, technical, and recruiter-friendly — *not Alfred-flavored*, unless you ask.

---

# 🚀 AWS Deployment (FastAPI + Docker + EC2)

This project includes a full production-style deployment of the fine-tuned NER model to **AWS EC2** using **Docker** and **FastAPI**.
The entire pipeline is lightweight, reproducible, and works on low-cost instances such as **t3.micro**.

---

## 🏗️ 1. Infrastructure Overview

**Service:** AWS EC2
**Instance Type:** `t3.micro` (2 vCPUs, 1GB RAM)
**AMI:** Ubuntu 22.04 LTS
**Model Size:** ~300MB (LoRA merged RoBERTa model)
**Serving Stack:**

* FastAPI
* Uvicorn
* Docker
* CPU-only PyTorch
* Custom NERPredictor class (Hugging Face Transformers)

Instance Config
<img width="1916" height="820" alt="AWS1" src="https://github.com/user-attachments/assets/f3b33909-ea08-4153-9685-f52733b01c0b" />

Instance Config
<img width="1917" height="823" alt="AWS" src="https://github.com/user-attachments/assets/ae756780-dba6-400f-b648-ecfd5be9298c" />


Instance Config
<img width="1915" height="824" alt="image" src="https://github.com/user-attachments/assets/6a7efc7e-e741-4048-af98-7633906853cb" />



---

## 🧱 2. Folder Structure Shipped to EC2

The entire app folder is packaged into a `tar.gz` archive and uploaded:

```
lora-ner-full.tar.gz
│
├── app/
│   ├── main.py
│   └── predictor.py
│
├── models/
│   └── roberta-lora-fewnerd-merged/
│
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

This ensures a clean, deterministic environment for Docker.

---

# ⚙️ 3. FastAPI Application (app/main.py)

The API exposes three endpoints:

* `/` → API health + model info
* `/health` → for container health checks
* `/predict` → NER inference endpoint

Features:

* Proper error handling
* Logging
* Pydantic validation
* Model loaded once at startup
* CPU-optimized inference path

---

# 🧠 4. NERPredictor (app/predictor.py)

A minimal inference wrapper that handles:

* Tokenization
* Model forwarding
* Argmax decoding
* Entity reconstruction
* Device management (CPU/GPU)

It works with any `AutoModelForTokenClassification` checkpoints.

---

# 📦 5. Dockerfile (Deployable Container Image)

Dockerfile:

* Installs Python deps
* Copies model + app into `/app`
* Runs Uvicorn at `0.0.0.0:8000`
* Adds AWS-compatible health checks
* Disables parallel tokenizers (fixes crashes)

This ensures the container is production-ready and works even on minimal CPU instances.

---

# 🖥️ 6. EC2 Deployment Steps

### **1️⃣ Create EC2 instance**

* AMI: Ubuntu 22.04
* Instance: `t3.micro`
* Open inbound rules:

  * `22` (SSH)
  * `8000` (API)

### **2️⃣ Upload project**

```
sftp -i lora-ner-key.pem ubuntu@<EC2_IP>
put lora-ner-full.tar.gz
```

### **3️⃣ SSH into EC2**

```
ssh -i lora-ner-key.pem ubuntu@<EC2_IP>
```

### **4️⃣ Extract project**

```
mkdir -p ~/lora-ner
tar -xzf lora-ner-full.tar.gz -C ~/lora-ner/
cd ~/lora-ner
```

### **5️⃣ Build Docker image**

```
docker build -t ner-api:v1 .
```

### **6️⃣ Run container**

```
docker run -d --name ner-api -p 8000:8000 --restart unless-stopped ner-api:v1
```

### **7️⃣ Check container health**

```
docker ps
docker logs ner-api
```

### **8️⃣ Test API on EC2**

```
curl http://localhost:8000
curl http://localhost:8000/predict -d '{"text":"Barack Obama was president of USA"}'
```

### **9️⃣ Test from outside**

```
curl http://<EC2_PUBLIC_IP>:8000
curl http://<EC2_PUBLIC_IP>:8000/predict ...
```

Everything becomes publicly accessible at:

```
http://<YOUR_EC2_PUBLIC_IP>:8000/docs
```

---

