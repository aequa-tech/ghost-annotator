import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig,pipeline,set_seed,AutoModel,AutoModelForMaskedLM
from typing import List,Dict
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()  # carica le variabili dal file .env

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print(token)
class ConformalGeneration:



    def __init__(self,model_name,device='mps',target_labels=['0','1','2','3','4']):

        if torch.cuda.is_available():
            device = torch.device("cuda")  # GPU NVIDIA
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # GPU Apple (M1/M2/M3)
            print("Using Apple MPS GPU")
        else:
            device = torch.device("cpu")  # CPU fallback
            print("Using CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto",token=token)
        self.device = device
        self.model.to(self.device)
        self.target_labels = target_labels

    def generate_probs(self,prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        logits = torch.stack(outputs.scores)  
        probs = torch.softmax(logits, dim=-1)  


        generated_ids = outputs.sequences[0][inputs.input_ids.shape[-1]:]
        generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]
        
        i_num = next((i for i, t in enumerate(generated_tokens) if t.strip() in ["0", "1", "2", "3", "4"]), None)
        if i_num is None:
            raise ValueError("No numeric token found in generated output.")

        vocab_probs = probs[i_num, 0]  # shape [vocab_size]
        token_probs = {}
        tot = 0
        for tok in self.target_labels:
            tok_id = self.tokenizer(tok, add_special_tokens=False).input_ids[0]
            token_probs[tok] = float(vocab_probs[tok_id])
            tot += vocab_probs[tok_id].item()

        for tok, p in token_probs.items():
            token_probs[tok] = p / tot
        
        return token_probs
    
    def brier(self,probs,label):
        conf_scores = dict()
        for pred,prob in probs.items():
            if int(prob) == label:
                conf_score = (1-prob)**2
                conf_scores[pred] = conf_score
            else:
                conf_score = (0-prob)**2
                conf_scores[pred] = conf_score
            
        non_conformity = np.mean([x for x in conf_scores.values()])

        return conf_scores,non_conformity


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "meta-llama/Llama-3.3-70B-Instruct-evals"

df = pd.read_csv('data/measuring_hatespeech/corpus.csv')
df = df[df.annotator_id==4047]

cg = ConformalGeneration(model_name, target_labels=['0', '1', '2', '3', '4'])

scores = list()
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):

    item = row.text

    prompt = f"""Task: you are a participant to an annotation task for the recognition of offensiveness

    Instruction: read the following social media post and annotate it with one value from the following options

    Options: 0,1,2,3,4. 0 == non violent, 4 == extremely violent

    Question: How much the following social media post is violent? {item}

    Output format: the answer should follow this template {{'answer': option}}

    Answer only in JSON. No extra text.
    """
    try:
        res = cg.generate_probs(prompt)
        conformities,score = cg.brier(res,row.label)
        scores.append(score)
    except Exception as e:
        print(e)
        continue


print(scores,np.mean(scores))