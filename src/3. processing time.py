import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import shutil
import time

def clear_model_cache(model_name):
    # Trova la cartella della cache per il modello
    model_cache_dir = os.path.join(os.getenv('HOME'), '.cache', 'huggingface', 'hub')

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

    def __init__(self, model_name, device='mps', target_labels=['0', '1', '2', '3', '4']):

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto",
                                                          token=token)
        self.device = device
        # self.model.to(self.device)
        self.target_labels = target_labels

    def generate_probs(self, prompt):
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
        # print(generated_tokens)
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

    def brier(self, probs, label):
        conf_scores = dict()
        for pred, prob in probs.items():
            if int(float(pred)) == label:
                conf_score = (1 - prob) ** 2
                conf_scores[pred] = conf_score
            else:
                conf_score = (0 - prob) ** 2
                conf_scores[pred] = conf_score

        non_conformity = np.mean([x for x in conf_scores.values()])

        return conf_scores, non_conformity

# Lista per memorizzare i tempi di esecuzione
processing_times = []

for model_name in [
    # piccoli
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    # medi
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    # grandi
    # "Qwen2.5-Coder-14B-Instruct",
    # "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # grandissimi? Non per ora
    # ...

]:

    for (dataset, corpus_prompt) in [
        (
                'data/measuring_hatespeech/corpus - violence.csv',
                """Task: you are a participant to an annotation task for the recognition of violence

               Instruction: read the following social media post and annotate it with one value from the following options

               Options: 0, 1, 2, 3, 4. 0 == no violence, 4 == extremely violence

               Output format: the answer should follow this template {'answer': option}

               Answer only in JSON.No extra text.

               Question: How much does the following social media post contain violence?    """
        ),
        (
                'data/measuring_hatespeech/attitudes - hate speech.csv',
                """
                Task: you are a participant to an annotation task for the recognition of hate speech

                Instruction: read the following social media post and annotate it with one value from the following options

                Options: 1,2,3,4,5. 1 == no hate speech, 5 == extremely hate speech

                Output format: the answer should follow this template {'answer': option}

                Answer only in JSON. No extra text.

                Question: How much does the following social media post contain hate speech?    """
        ),
        (
                'data/measuring_hatespeech/cade - acceptability.csv',
                """Task: you are a participant to an annotation task for the recognition of unacceptability
                Instruction: read the following social media post and annotate it with one value from the following options

                Options: 1,2,3,4. 1 == acceptable, 4 == extremely unacceptable

                Output format: the answer should follow this template {'answer': option}

                Answer only in JSON. No extra text.

                Question: How much does the following social media post contain unacceptable content?    """
        ),

        ('data/measuring_hatespeech/davani - offensiveness.csv',
         """Task: you are a participant to an annotation task for the recognition of offensiveness

                Instruction: read the following social media post and annotate it with one value from the following options

                Options: 0,1,2,3,4. 0 == no offensiveness, 4 == extremely offensiveness

                Output format: the answer should follow this template {'answer': option}

                Answer only in JSON. No extra text.

                Question: How much does the following social media post contain offensiveness?    """

         )
    ]:

        df = pd.read_csv(dataset)

        model_tag = model_name.split("/")[-1]
        dataset_name = dataset.split("/")[-1].split(".")[0].split("-")[0].strip()
        task_name = dataset.split("/")[-1].split(".")[0].split("-")[1].strip()

        # Solo per ottenere i primi 100 commenti
        df_grouped = df.groupby('comment_id').agg(
            text=('text', 'first'),
            labels=('label', list),
            annotators=('annotator_id', list),
            social_groups=('social_group', list)
        ).reset_index()

        # Iniziamo a misurare il tempo
        start_time = time.time()

        for idx, row in tqdm(enumerate(df_grouped.itertuples(), 1), total=100):
            if idx > 100:
                break

            text = row.text
            labels = row.labels
            prompt = corpus_prompt + f'  "{text}"'

            cg = ConformalGeneration(model_name, target_labels=list(map(str, labels)))

            try:
                res = cg.generate_probs(prompt)
                for label, annotator_id, social_group in zip(row.labels, row.annotators, row.social_groups):
                    cg.brier(res, label)
            except Exception as e:
                print(f"Error processing comment: {e}")
                continue

        # Misuriamo il tempo trascorso per i primi 100 commenti
        end_time = time.time()
        time_for_100 = end_time - start_time
        time_per_entry = time_for_100 / 100
        estimated_total_time = time_per_entry * len(df_grouped)

        # Aggiungi i tempi al risultato
        processing_times.append({
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_row": len(df_grouped),
            "processing_time_100_entry": time_for_100,
            "estimated_total_processing_time": estimated_total_time
        })

        print(f"✅ Model {model_name} processed {dataset_name} for time estimation.")
    clear_model_cache(model_name)
# Salvataggio dei tempi in un CSV
processing_df = pd.DataFrame(processing_times)
processing_df.to_csv('output/processing_time.csv', index=False)

print(f"✅ All time estimations have been saved to processing_time.csv.")
