import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import shutil

def clear_model_cache(model_name):
    # Trova la cartella della cache per il modello
    model_cache_dir = os.path.join(os.getenv('HOME'), '.cache', 'huggingface', 'transformers', model_name)

    if os.path.exists(model_cache_dir):
        # Rimuove la cartella della cache del modello
        shutil.rmtree(model_cache_dir)
        print(f"Cache del modello {model_name} eliminata.")
    else:
        print(f"Cache per il modello {model_name} non trovata.")


torch.set_grad_enabled(False)

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
            if torch.version.hip:
                device = torch.device("cuda")  # ROCm backend
                print("Using AMD ROCm GPU")
            else:
                device = torch.device("cpu")  # CPU fallback
                print("Using CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto",token=token)
        self.device = device
        #self.model.to(self.device)
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
        #print(generated_tokens)
        i_num = next((i for i, t in enumerate(generated_tokens) if t.strip() in ["0", "1", "2", "3", "4"]), None)

        if i_num is None:
            raise ValueError(f"No numeric token found in generated output: {generated_tokens}")

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
            if int(pred) == label:
                conf_score = (1-prob)**2
                conf_scores[pred] = conf_score
            else:
                conf_score = (0-prob)**2
                conf_scores[pred] = conf_score
            
        non_conformity = np.mean([x for x in conf_scores.values()])

        return conf_scores,non_conformity



for model_name in [
                    #piccoli
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct",
                    #medi
                    "Qwen/Qwen2.5-7B-Instruct",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    #grandi
                    "Qwen2.5-Coder-14B-Instruct",
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                    #grandissimi? Non per ora
                    #...



    ]:

    for dataset in [
        'data/measuring_hatespeech/cade - acceptability.csv',
        'data/measuring_hatespeech/corpus - violence.csv',
        'data/measuring_hatespeech/attitudes - hate speech.csv',
        'data/measuring_hatespeech/davani - offensiveness.csv',
    ]:

        df = pd.read_csv(dataset)


        model_tag = model_name.split("/")[-1]
        dataset_name=dataset.split("/")[-1].split(".")[0].split("-")[0].strip()
        task_name=dataset.split("/")[-1].split(".")[0].split("-")[1].strip()
        result_file = f"results_{model_tag}_{dataset_name}.csv"

        df = df.dropna(subset=['text', 'label', 'annotator_id', 'social_group'])
        df = df.dropna(subset=['label'])
        print(dataset_name,task_name)
        # Se esiste già, caricalo e salta gli ID già processati
        if os.path.exists(result_file):
            done_df = pd.read_csv(result_file)
            done_ids = set(done_df["comment_id"])
            print(f"Found {len(done_ids)} processed items for {model_tag}")
        else:
            done_ids = set()
            print(f"Starting fresh for {model_tag}")


        new_rows = []
        target_labels = sorted(df['label'].unique().tolist())
        print(sorted(df['social_group'].unique().tolist()))
        #print(target_labels)
        target_labels =  list(map(int, target_labels))
        #print(target_labels)

        scores = list()

        df_grouped = df.groupby('comment_id').agg(
            text=('text', 'first'),
            labels=('label', list),
            annotators=('annotator_id', list),
            social_groups=('social_group', list)
        ).reset_index()
        for row in tqdm(df_grouped.itertuples(), total=len(df_grouped)):
            if row.comment_id in done_ids:
                continue
            text = row.text
            labels = row.labels


            prompt = f"""Task: you are a participant to an annotation task for the recognition of offensiveness
        
            Instruction: read the following social media post and annotate it with one value from the following options
        
            Options: {",".join(map(str, target_labels))}. {min(target_labels)} == no {task_name}, {max(target_labels)} == extremely {task_name}
            
            Output format: the answer should follow this template {{'answer': option}}
        
            Answer only in JSON. No extra text.
        
            Question: How much the following social media post contains {task_name}?  "{text}" 
          
    
            """

            cg = ConformalGeneration(model_name, target_labels=list(map(str,target_labels)))

            try:
                res = cg.generate_probs(prompt)
                print(res)
                for label, annotator_id, social_group in zip(row.labels, row.annotators, row.social_groups):

                    conformities,score = cg.brier(res,label)
                    new_rows.append({
                        "comment_id": row.comment_id,
                        "text": row.text,
                        "label": label,
                        "annotator_id": annotator_id,
                        "social_group": social_group,
                        "probs": json.dumps(res),
                        "brier_score": score,
                    })
            except Exception as e:
                print(e)
                print(text)
                continue

            # Salva parziale ogni 100 commenti
            if len(new_rows) >= 100:
                partial_df = pd.DataFrame(new_rows)
                if os.path.exists(result_file):
                    partial_df.to_csv(result_file, mode='a', index=False, header=False)
                else:
                    partial_df.to_csv(result_file, index=False)
                new_rows = []


        # Salva gli ultimi risultati
        if new_rows:
            partial_df = pd.DataFrame(new_rows)
            if os.path.exists(result_file):
                partial_df.to_csv(result_file, mode='a', index=False, header=False)
            else:
                partial_df.to_csv(result_file, index=False)

        print(f"✅ Finished model {model_tag}, results saved to {result_file}")
    clear_model_cache(model_name)