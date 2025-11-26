# predictor.py
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class NERPredictor:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        current = None

        for idx, (tok, pred_id) in enumerate(zip(tokens, preds)):
            if tok in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue

            tok_clean = tok.replace("Ġ", " ")

            label = self.model.config.id2label[pred_id]
            if label != 'O':
                if current is None or current["label"] != label:
                    if current:
                        entities.append(current)
                    current = {
                        "entity": tok_clean.strip(),
                        "label": label,
                        "start": idx,
                        "end": idx
                    }
                else:
                    current["entity"] += tok_clean
                    current["end"] = idx
            else:
                if current:
                    entities.append(current)
                    current = None

        if current:
            entities.append(current)

        return entities
