from logging import warning
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import json

class CTCScorer():
    def __init__(self, model_type) -> None:
        self.model_type = model_type
        import nltk
        nltk.download('stopwords')

        from ctc_score import StyleTransferScorer, SummarizationScorer, DialogScorer
        if model_type == 'D-cnndm':
            self.scorer = SummarizationScorer(align='D-cnndm')
        elif model_type =='E-roberta':
            self.scorer = SummarizationScorer(align='E-roberta')
        elif model_type == 'R-cnndm':
            self.scorer = SummarizationScorer(align='R-cnndm')
    def score(self, premise: list, hypo: list):
        assert len(premise) == len(hypo), "Premise and hypothesis should have the same length"
        
        output_scores = []
        for one_pre, one_hypo in tqdm(zip(premise, hypo), total=len(premise), desc="Evaluating by ctc"):
            score_for_this_example = self.scorer.score(doc=one_pre, refs=[], hypo=one_hypo, aspect='consistency')
            if score_for_this_example is not None:
                output_scores.append(score_for_this_example)
            else:
                output_scores.append(1e-8)
        output = None, torch.tensor(output_scores), None

        return output

class SimCSEScorer():
    def __init__(self, model_type, device) -> None:
        self.model_type = model_type
        self.device = device
        from transformers import AutoModel, AutoTokenizer

        # refer to the model list on https://github.com/princeton-nlp/SimCSE for the list of models
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model = AutoModel.from_pretrained(model_type).to(self.device)
        self.spacy = spacy.load('en_core_web_sm')

        self.batch_size = 64

    def score(self, premise: list, hypo: list):
        assert len(premise) == len(hypo)

        output_scores = []
        premise_sents = []
        premise_index = [0]
        hypo_sents = []
        hypo_index = [0]

        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc="Sentenizing", total=len(premise)):
            premise_sent = sent_tokenize(one_pre) #[each.text for each in self.spacy(one_pre).sents]
            hypo_sent = sent_tokenize(one_hypo) #[each.text for each in self.spacy(one_hypo).sents]
            premise_sents.extend(premise_sent)
            premise_index.append(len(premise_sents))

            hypo_sents.extend(hypo_sent)
            hypo_index.append(len(hypo_sents))

        all_sents = premise_sents + hypo_sents
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(self.chunks(all_sents, self.batch_size), total=int(len(all_sents)/self.batch_size), desc="Evaluating by SimCSE"):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                embeddings.append(self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output)
            embeddings = torch.cat(embeddings)

            assert len(premise_index) == len(hypo_index)
            for i in range(len(premise_index)-1):
                premise_embeddings = embeddings[premise_index[i]: premise_index[i+1]]
                hypo_embeddings = embeddings[len(premise_sents)+hypo_index[i]:len(premise_sents)+hypo_index[i+1]]
                cos_sim = cosine_similarity(premise_embeddings.cpu(), hypo_embeddings.cpu())
                score_p = cos_sim.max(axis=0).mean()
                score_r = cos_sim.max(axis=1).mean()
                score_f = 2 * score_p * score_r / (score_p + score_r)
                output_scores.append(score_f)

        return torch.Tensor(output_scores), torch.Tensor(output_scores), None
    
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

class BleurtScorer():
    def __init__(self, checkpoint) -> None:
        self.checkpoint = checkpoint

        from bleurt import score
        # BLEURT-20 can also be switched to other checkpoints to improve time
        # No avaliable api to specify cuda number
        self.model = score.BleurtScorer(self.checkpoint)

    def scorer(self, premise:list, hypo: list):
        assert len(premise) == len(hypo)

        output_scores = self.model.score(references=premise, candidates=hypo, batch_size=8)
        output_scores = [s for s in output_scores]
        return torch.Tensor(output_scores), torch.Tensor(output_scores), torch.Tensor(output_scores)

class BertScoreScorer():
    def __init__(self, model_type, metric, device, batch_size) -> None:
        self.model_type = model_type
        self.device = device
        self.metric = metric
        self.batch_size = batch_size

        from bert_score import score
        self.model = score
    
    def scorer(self, premise: list, hypo: list):
        assert len(premise) == len(hypo)

        precision, recall, f1 = self.model(premise, hypo, model_type=self.model_type, lang='en', rescale_with_baseline=True, verbose=True, device=self.device, batch_size=self.batch_size)

        f1 = [f for f in f1]
        precision = [p for p in precision]
        recall = [r for r in recall]

        if self.metric == 'f1':
            return torch.Tensor(f1), torch.Tensor(f1), None
        elif self.metric == 'precision':
            return torch.Tensor(precision), torch.Tensor(precision), None
        elif self.metric == 'recall':
            return torch.Tensor(recall), torch.Tensor(recall), None
        else:
            ValueError("metric type not in f1, precision or recall.")

class BartScoreScorer():
    def __init__(self, checkpoint, device) -> None:
        self.checkpoint = checkpoint
        self.device = device
        import os, sys
        sys.path.append('baselines/BARTScore')
        from bart_score import BARTScorer
        self.model = BARTScorer(device=self.device, checkpoint=self.checkpoint)
    
    def scorer(self, premise: list, hypo: list):
        assert len(premise) == len(hypo)

        output_scores = self.model.score(premise, hypo, batch_size=4)
        normed_score = torch.exp(torch.Tensor(output_scores))
        
        return normed_score, normed_score, normed_score

### Below are baselines in SummaC
### MNLI, NER, FactCC, DAE, FEQA, QuestEval, SummaC-ZS, SummaC-Conv
class MNLIScorer():
    def __init__(self, model="roberta-large-mnli", device='cuda:0', batch_size=32) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size

    def scorer(self, premise: list, hypo: list):
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]
        
        batch = self.batch_tokenize(premise, hypo)
        output_score_tri = []

        for mini_batch in tqdm(batch, desc="Evaluating MNLI"):
        # for mini_batch in batch:
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(**mini_batch)
                model_output_tri = model_output.logits
                model_output_tri = self.softmax(model_output_tri).cpu()

            output_score_tri.append(model_output_tri[:,2])

        output_score_tri = torch.cat(output_score_tri)
        
        return output_score_tri, output_score_tri, output_score_tri

    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(hypo), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
            try:
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation='only_first', padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            except:
                warning('text_b too long...')
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            batch.append(mini_batch)

        return batch

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

class NERScorer():
    def __init__(self) -> None:
        import os, sys
        sys.path.append('baselines/summac/summac')
        from model_guardrails import NERInaccuracyPenalty
        self.ner = NERInaccuracyPenalty()
    
    def scorer(self, premise, hypo):
        score_return = self.ner.score(premise, hypo)['scores']
        oppo_score = [float(not each) for each in score_return]
        
        tensor_score = torch.tensor(oppo_score)

        return tensor_score, tensor_score, tensor_score
class UniEvalScorer():
    def __init__(self, task='fact', device='cuda:0') -> None:
        import os, sys
        sys.path.append('baselines/UniEval')
        from metric.evaluator import get_evaluator

        self.evaluator = get_evaluator(task, device=device)
    
    def scorer(self, premise, hypo):
        from utils import convert_to_json
        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=hypo, src_list=premise)
        # Initialize evaluator for a specific task
        
        # Get factual consistency scores
        eval_scores = self.evaluator.evaluate(data, print_result=True)
        score_list = [each['consistency'] for each in eval_scores]

        return torch.tensor(score_list), torch.tensor(score_list), torch.tensor(score_list)

class FEQAScorer():
    def __init__(self) -> None:
        import os, sys
        sys.path.append('baselines/feqa')
        import benepar
        import nltk

        benepar.download('benepar_en3')
        nltk.download('stopwords')

        from feqa import FEQA
        self.feqa_model = FEQA(squad_dir=os.path.abspath('baselines/feqa/qa_models/squad1.0'), bart_qa_dir=os.path.abspath('baselines/feqa/bart_qg/checkpoints/'), use_gpu=True)
    
    def scorer(self, premise, hypo):
        eval_score = self.feqa_model.compute_score(premise, hypo, aggregate=False)

        return torch.tensor(eval_score), torch.tensor(eval_score), torch.tensor(eval_score)


class QuestEvalScorer():
    def __init__(self) -> None:
        import os, sys
        sys.path.append('baselines/QuestEval')
        from questeval.questeval_metric import QuestEval
        self.questeval = QuestEval(no_cuda=False)

    def scorer(self, premise, hypo):
        score = self.questeval.corpus_questeval(
                hypothesis=hypo, 
                sources=premise
            )
        final_score = score['ex_level_scores']

        return torch.tensor(final_score), torch.tensor(final_score), torch.tensor(final_score)

class QAFactEvalScorer():
    def __init__(self, model_folder, device='cuda:0') -> None:
        import os, sys
        sys.path.append('baselines/QAFactEval')
        sys.path.append(os.path.abspath('baselines/qaeval/'))
        from qafacteval import QAFactEval
        kwargs = {"cuda_device": int(device.split(':')[-1]), "use_lerc_quip": True, \
                "verbose": True, "generation_batch_size": 32, \
                "answering_batch_size": 32, "lerc_batch_size": 8}

        self.metric = QAFactEval(
            lerc_quip_path=f"{model_folder}/quip-512-mocha",
            generation_model_path=f"{model_folder}/generation/model.tar.gz",
            answering_model_dir=f"{model_folder}/answering",
            lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
            lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
            **kwargs
                            )
    def scorer(self, premise, hypo):
        results = self.metric.score_batch_qafacteval(premise, [[each] for each in hypo], return_qa_pairs=True)
        score = [result[0]['qa-eval']['lerc_quip'] for result in results]
        return torch.tensor(score), torch.tensor(score), torch.tensor(score)

class MoverScorer():
    def __init__(self) -> None:
        pass

class BERTScoreFFCIScorer():
    def __init__(self) -> None:
        pass

class DAEScorer():
    def __init__(self, model_dir, device=0) -> None:
        import os, sys
        sys.path.insert(0, "baselines/factuality-datasets/")
        from evaluate_generated_outputs import daefact
        self.dae = daefact(model_dir, model_type='electra_dae', gpu_device=device)
    
    def scorer(self, premise, hypo):
        return_score = torch.tensor(self.dae.score_multi_doc(premise, hypo))

        return return_score, return_score, return_score

class SummaCScorer():
    def __init__(self, summac_type='conv', device='cuda:0') -> None:
        self.summac_type = summac_type
        import os, sys
        sys.path.append("baselines/summac")
        from summac.model_summac import SummaCZS, SummaCConv

        if summac_type == 'conv':
            self.model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=device, start_file="default", agg="mean")
        elif summac_type == 'zs':
            self.model = SummaCZS(granularity="sentence", model_name="vitc", device=device) # If you have a GPU: switch to: device="cuda"
    
    def scorer(self, premise, hypo):
        assert len(premise) == len(hypo)
        scores = self.model.score(premise, hypo)['scores']
        return_score = torch.tensor(scores)

        return return_score, return_score, return_score

class FactCCScorer():
    def __init__(self, script_path, test_data_path,result_path) -> None:
        self.script_path = script_path
        self.result_path = result_path
        self.test_data_path = test_data_path
    def scorer(self, premise, hypo):
        import subprocess
        import pickle

        self.generate_json_file(premise, hypo)
        subprocess.call(f"sh {self.script_path}", shell=True)
        print("Finishing FactCC")
        results = pickle.load(open(self.result_path, 'rb'))
        results = [-each+1 for each in results]

        return torch.tensor(results), torch.tensor(results), torch.tensor(results)
        
    def generate_json_file(self, premise, hypo):
        output = []
        assert len(premise) == len(hypo)
        i = 0
        for one_premise, one_hypo in zip(premise, hypo):
            example = dict()
            example['id'] = i
            example['text'] = one_premise
            example['claim'] = one_hypo
            example['label'] = 'CORRECT'

            i += 1
            output.append(example)
        with open(self.test_data_path, 'w', encoding='utf8') as f:
            for each in output:
                json.dump(each, f, ensure_ascii=False)
                f.write('\n')

class BLANCScorer():
    def __init__(self, device='cuda', batch_size=64) -> None:
        from blanc import BlancHelp, BlancTune
        self.blanc_help = BlancHelp(device=device, inference_batch_size=batch_size)
        

    def scorer(self, premise, hypo):
        score = self.blanc_help.eval_pairs(premise, hypo)

        return_score = torch.tensor(score)

        return return_score, return_score, return_score
        

class BLEUScorer():
    def __init__(self, n_grams=1) -> None:
        self.n_grams = n_grams
        self.n_gram_map = {
            1: (1,0,0,0),
            2: (0.5,0.5,0,0),
            3: (1./3,1./3,1./3,0),
            4: (0.25,0.25,0.25,0.25)
        }

    def scorer(self, premise, hypo):
        from nltk.translate.bleu_score import sentence_bleu
        assert len(premise) == len(hypo), "premise and hypothesis should be the same length!"

        output_score = []

        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc=f"Evaluating BLEU-{self.n_grams}", total=len(premise)):
            scores = []
            pre_sents = sent_tokenize(one_pre)
            references = [[each for each in sent.split()] for sent in pre_sents]
            for hypo_sent in sent_tokenize(one_hypo):
                hypothesis = [each for each in hypo_sent.split()]
                scores.append(sentence_bleu(references=references, hypothesis=hypothesis, weights=self.n_gram_map[self.n_grams]))
            output_score.append(sum(scores)/len(scores) if len(scores)>0 else 0.)
            # from IPython import embed; embed()

        return torch.tensor(output_score), torch.tensor(output_score), torch.tensor(output_score)

class ROUGEScorer():
    def __init__(self, rouge_type='1') -> None:
        from rouge import Rouge 
        self.rouge = Rouge()
        self.rouge_type = rouge_type

    def scorer(self, premise, hypo):
        
        assert len(premise) == len(hypo), "premise and hypothesis should be the same length!"

        output_score = []

        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc=f"Evaluating ROUGE-{self.rouge_type}", total=len(premise)):
            scores = []
            for pre_sent in sent_tokenize(one_pre):
                for hypo_sent in sent_tokenize(one_hypo):
                    try:
                        scores.append(self.rouge.get_scores(pre_sent, hypo_sent)[0][f"rouge-{self.rouge_type}"]['f'])
                    except:
                        if len(pre_sent.strip()) == 0:
                            print('premise sent is empty')
                        elif len(hypo_sent.strip()) == 0:
                            print('hypo sent is empty')
                        scores.append(0.0)
            scores = np.array(scores)
            scores = scores.reshape((len(sent_tokenize(one_pre)), len(sent_tokenize(one_hypo))))
            scores = scores.max(axis=0).mean()
            output_score.append(scores.item())

        return torch.tensor(output_score), torch.tensor(output_score), torch.tensor(output_score)


class GPTScoreScorer():
    def __init__(self, api_key, gpt_model='davinci003') -> None:
        import os, sys
        sys.path.append('../BaselineForNLGEval/GPTScore')
        from gpt3_score import gpt3score

        self.gpt3score = gpt3score
        self.api_key = api_key
        self.gpt_model = gpt_model

        self.consistency_prefix = "Generate factually consistent summary for the following text: " 
        self.consistency_suffix = " \n\nTl;dr "


    def scorer(self, premise: list, hypothesis: list):
        assert len(premise) == len(hypothesis)
        output_score = []
        for p, h in tqdm(zip(premise, hypothesis), total=len(premise), desc="Evaluating GPTScore"):
            score = self.gpt3score(input=self.consistency_prefix + p + self.consistency_suffix, output=h, gpt3model=self.gpt_model, api_key=self.api_key)
            output_score.append(score)

        output_score = torch.tensor(output_score)
        
        return None, output_score, None
    
class ChatGPTLuo2023Scorer():
    def __init__(self, task, api_key, chat_model='gpt-3.5-turbo') -> None:
        openai.api_key = api_key
        assert isinstance(task, list) and len(task) == 1

        self.task = task[0]
        self.chat_model = chat_model
        self.instruct = """Score the following summary given the corresponding article with respect to consistency from 1 to 10. Note that consistency measures how much information included in the summary is present in the source article. 10 points indicate the summary contains only statements that are entailed by the source document."""
    
    def scorer(self, premise: list, hypothesis: list):
        import time
        assert len(premise) == len(hypothesis)
        output_score = []
        i = -1

        for p, h in tqdm(zip(premise, hypothesis), total=len(premise), desc="Evaluating ChatGPTLuo2023"):
            i += 1
            if i <= -1: continue

            attempt = 0
            max_attempt = 5
            while attempt < max_attempt:
                try:
                    response = openai.ChatCompletion.create(
                                model=self.chat_model,
                                messages=[
                            #         {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": f"""Score the following summary given the corresponding article with respect to consistency from 1 to 10. Note that consistency measures how much information included in the summary is present in the source article. 10 points indicate the summary contains only statements that are entailed by the source document.

                                        Summary: {h}

                                        Article: {p} """},
                                ],
                                temperature=0,
                                max_tokens=10
                            )
                    res_content = response['choices'][0]['message']['content']
                    break
                except:
                    attempt += 1
                    print("openai api failed")
                    if max_attempt == attempt:
                        print("maximum failed attempts reached. exiting...")
                        exit()
            json.dump({i: res_content}, open(f'exp_results/nlg_eval_fact/baselines/ChatGPTLuo2023-output/{self.task}.json', 'a'))
            with open(f'exp_results/nlg_eval_fact/baselines/ChatGPTLuo2023-output/{self.task}.json', 'a') as f:
                f.write('\n')
            
            try:
                score = int(res_content)
            except:
                print("unknown score")
                score = 0.0
            output_score.append(score)
            # time.sleep(1)

        output_score = torch.tensor(output_score)
        
        return None, output_score, None

class ChatGPTGao2023Scorer():
    def __init__(self, task, api_key, chat_model='gpt-3.5-turbo') -> None:
        openai.api_key = api_key
        assert isinstance(task, list) and len(task) == 1

        self.task = task[0]
        self.chat_model = chat_model
    
    def scorer(self, premise: list, hypothesis: list):
        import time
        assert len(premise) == len(hypothesis)
        output_score = []
        i = -1

        for p, h in tqdm(zip(premise, hypothesis), total=len(premise), desc="Evaluating ChatGPTGao2023"):
            i += 1
            if i <= -1: continue

            attempt = 0
            max_attempt = 5
            while attempt < max_attempt:
                try:
                    response = openai.ChatCompletion.create(
                                model=self.chat_model,
                                messages=[
                                    # {"role": "system", "content": "You are a human annotator that rates the quality of summaries"},
                                    # {"role": "user", "content": f"""Imagine you are a human annotator now. You will evaluate the quality of summaries written for a news article. Please follow these steps:\n\n 1. Carefully read the news article, and be aware of the information it contains.\n 2. Read the proposed summary.\n 3. Rate the summary on four dimensions: relevance, consistency, fluency, and coherence. You should rate on a scale from 1 (worst) to 5 (best).\n\n  Definitions are as follows:\n Relevance: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.\n Consistency: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.\n Fluency: This rating measures the quality of individual sentences, whether they are well-written and grammatically correct. Consider the quality of individual sentences.\n Coherence: The rating measures the quality of all sentences collectively, to fit together and sound natural. Consider the quality of the summary as a whole.\n\n The article and the summary are given below:\n Article: {p}\n Summary: {h}"""},
                                    {"role": "user", "content": f"""Evaluate the quality of summaries written for a news article. Rate each summary on four dimensions: relevance, faithfulness, fluency, and coherence. You should rate on a scale from 1 (worst) to 5 (best).\n\n Article: {p}\n Summary: {h}"""},
                                ],
                                temperature=0,
                                # max_tokens=10
                            )
                    res_content = response['choices'][0]['message']['content']
                    break
                except:
                    attempt += 1
                    print("openai api failed")
                    if max_attempt == attempt:
                        print("maximum failed attempts reached. exiting...")
                        exit()
            json.dump({i: res_content}, open(f'exp_results/nlg_eval_fact/baselines/ChatGPTGao2023-output/{self.task}.json', 'a'))
            with open(f'exp_results/nlg_eval_fact/baselines/ChatGPTGao2023-output/{self.task}.json', 'a') as f:
                f.write('\n')
            
            try:
                score = int(res_content)
            except:
                print("unknown score")
                score = 0.0
            output_score.append(score)
            # time.sleep(1)

        output_score = torch.tensor(output_score)
        
        return None, output_score, None
    
class ChatGPTYiChen2023Scorer():
    def __init__(self, task, api_key, chat_model='gpt-3.5-turbo') -> None:
        ### Explicit score by ChatGPT
        openai.api_key = api_key
        assert isinstance(task, list) and len(task) == 1

        self.task = task[0]
        self.chat_model = chat_model
    
    def scorer(self, premise: list, hypothesis: list):
        import time
        assert len(premise) == len(hypothesis)
        output_score = []
        i = -1

        for p, h in tqdm(zip(premise, hypothesis), total=len(premise), desc="Evaluating ChatGPTYiChen2023"):
            i += 1
            if i <= -1: continue

            attempt = 0
            max_attempt = 5
            while attempt < max_attempt:
                try:
                    response = openai.ChatCompletion.create(
                                model=self.chat_model,
                                messages=[
                                    # {"role": "system", "content": "You are a human annotator that rates the quality of summaries"},
                                    # {"role": "user", "content": f"""Imagine you are a human annotator now. You will evaluate the quality of summaries written for a news article. Please follow these steps:\n\n 1. Carefully read the news article, and be aware of the information it contains.\n 2. Read the proposed summary.\n 3. Rate the summary on four dimensions: relevance, consistency, fluency, and coherence. You should rate on a scale from 1 (worst) to 5 (best).\n\n  Definitions are as follows:\n Relevance: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.\n Consistency: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.\n Fluency: This rating measures the quality of individual sentences, whether they are well-written and grammatically correct. Consider the quality of individual sentences.\n Coherence: The rating measures the quality of all sentences collectively, to fit together and sound natural. Consider the quality of the summary as a whole.\n\n The article and the summary are given below:\n Article: {p}\n Summary: {h}"""},
                                    {"role": "user", "content": f"""Score the following storyline given the beginning of the story on a continual scale from 0 (worst) to 100 (best), where score of 0 means "The storyline makes no sense and is totally not understandable" and score of 100 means "The storyline is perfect-written and highly consistent with the given beginning of the story". \n\n The beginning of the story: {p} \n\n Storyline: {h} \n\n Score: """},
                                ],
                                temperature=0,
                                # max_tokens=10
                            )
                    res_content = response['choices'][0]['message']['content']
                    break
                except:
                    attempt += 1
                    print("openai api failed")
                    if max_attempt == attempt:
                        print("maximum failed attempts reached. exiting...")
                        exit()
            json.dump({i: res_content}, open(f'exp_results/nlg_eval_fact/baselines/ChatGPTYiChen2023-output/{self.task}.json', 'a'))
            with open(f'exp_results/nlg_eval_fact/baselines/ChatGPTYiChen2023-output/{self.task}.json', 'a') as f:
                f.write('\n')
            
            try:
                score = int(res_content)
            except:
                print("unknown score")
                score = 0.0
            output_score.append(score)
            # time.sleep(1)

        output_score = torch.tensor(output_score)
        
        return None, output_score, None
    
class ChatGPTShiqiChen2023Scorer():
    def __init__(self, task, api_key, chat_model='gpt-3.5-turbo') -> None:
        ### Explicit score by ChatGPT
        openai.api_key = api_key
        assert isinstance(task, list) and len(task) == 1

        self.task = task[0]
        self.chat_model = chat_model
    
    def scorer(self, premise: list, hypothesis: list):
        import time
        assert len(premise) == len(hypothesis)
        output_score = []
        i = -1

        for p, h in tqdm(zip(premise, hypothesis), total=len(premise), desc="Evaluating ChatGPTShiqiChen2023"):
            i += 1
            if i <= -1: continue
            hypo_sents = sent_tokenize(h)
            hypo_sents = ' \n '.join([f"{i+1}. "+each for i, each in enumerate(hypo_sents)])
            attempt = 0
            max_attempt = 5
            while attempt < max_attempt:
                try:
                    response = openai.ChatCompletion.create(
                                model=self.chat_model,
                                messages=[
                                    # {"role": "system", "content": "You are a human annotator that rates the quality of summaries"},
                                    # {"role": "user", "content": f"""Imagine you are a human annotator now. You will evaluate the quality of summaries written for a news article. Please follow these steps:\n\n 1. Carefully read the news article, and be aware of the information it contains.\n 2. Read the proposed summary.\n 3. Rate the summary on four dimensions: relevance, consistency, fluency, and coherence. You should rate on a scale from 1 (worst) to 5 (best).\n\n  Definitions are as follows:\n Relevance: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.\n Consistency: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.\n Fluency: This rating measures the quality of individual sentences, whether they are well-written and grammatically correct. Consider the quality of individual sentences.\n Coherence: The rating measures the quality of all sentences collectively, to fit together and sound natural. Consider the quality of the summary as a whole.\n\n The article and the summary are given below:\n Article: {p}\n Summary: {h}"""},
                                    {"role": "user", "content": f"""Source Document: \n {p} \n\n Q: Can the following statement be inferred from the above document? Yes or No?\n {hypo_sents} \n A: 1. """},
                                ],
                                temperature=0,
                                # max_tokens=10
                            )
                    res_content = response['choices'][0]['message']['content']
                    break
                except:
                    attempt += 1
                    print("openai api failed")
                    if max_attempt == attempt:
                        print("maximum failed attempts reached. exiting...")
                        exit()
            json.dump({i: res_content}, open(f'exp_results/nlg_eval_fact/baselines/ChatGPTShiqiChen2023-output/{self.task}.json', 'a'))
            with open(f'exp_results/nlg_eval_fact/baselines/ChatGPTShiqiChen2023-output/{self.task}.json', 'a') as f:
                f.write('\n')
            
            try:
                score = int(res_content)
            except:
                print("unknown score")
                score = 0.0
            output_score.append(score)
            # time.sleep(1)

        output_score = torch.tensor(output_score)
        
        return None, output_score, None


if __name__ == '__main__':
    prem = ["The cat is on the mat."]
    hypo = ["There is a cat on the mat."]
    blue_scorer = BLEUScorer(n_grams=1)
    rouge_scorer = ROUGEScorer()
    # bleurt_scorer = BleurtScorer()
    # bert_scorer = BertScoreScorer()


    blue_score = blue_scorer.scorer(prem, hypo)
    rouge_score = rouge_scorer.scorer(prem, hypo)
    # bleurt_score = bleurt_scorer.scorer(prem, hypo)
    # bert_score = bert_scorer.scorer(prem, hypo)


    print(blue_score)
    print(rouge_score)
    # print(bleurt_score)
    # print(bert_score)

    passage = ["""
How to Calculate When You Should Go to Sleep
Are you struggling to keep track of your sleep schedule? This video explains what to keep in mind to calculate how much sleep you need.
0 seconds of 1 minute, 4 seconds Volume 0%
Your sleep needs vary during your lifetime and by how many cycles of sleep you need to feel rested.
How much sleep did you get last night? What about the night before? How much sleep do you actually need?
Keeping track of your sleep schedule might not always be your top priority, but getting enough sleep is critical to your health in many ways.
You may not realize it, but the amount of sleep you get can affect everything from weight and metabolism to brain function and mood.
For many people, wake-up time remains fairly constant from day to day. The time you go to sleep, however, might vary, depending on any number of things:
- your social life
- your work schedule
- family obligations
- the newest show streaming on Netflix
- the time you start to feel tired
But since you know when you need to get up, knowing the specific amount of sleep you need to function at your best can help you determine what time to go to bed.
Below, you’ll find out how to calculate the best time to go to bed based on your wake time and natural sleep cycles. We’ll also offer more insight on how sleep cycles work and why sleep, or lack thereof, can affect your health.
How much sleep you need changes throughout your lifetime. An infant may need up to 17 hours of sleep each day, while an older adult may get by on just 7 hours of sleep a night.
Sleep guidelines can offer a place to start determining your sleep needs by providing research-backed recommendations for the ideal amount of sleep for optimal health.
- Birth to 3 months: 14 to 17 hours
- 4 to 11 months: 12 to 16 hours
- 1 to 2 years: 11 to 14 hours
- 3 to 5 years: 10 to 13 hours
- 6 to 12 years: 9 to 12 hours
- 13 to 18 years: 8 to 10 hours
- 18 to 64 years: 7 to 9 hours
- 65 years and older: 7 to 8 hours
Was this helpful?
Keep in mind, though, that sleep needs can still vary, even within the same age group.
You might need at least 9 hours of sleep a night to feel well rested, while your partner may wake up naturally after 7 hours, feeling perfectly refreshed and ready for the day.
The thing to keep in mind is how you feel when you get various amounts of sleep.
Here are a few questions to consider when evaluating your sleep needs:
- Do I feel rested after 7 hours of sleep, or do I need at least 8 or 9?
- Do I experience any daytime drowsiness?
- Do I rely on caffeine to keep me going throughout the day?
- Has my sleeping partner noticed me tossing and turning, or having any sleep issues during the night?
HEALTHLINE PARTNER SOLUTIONS
Check your vitamin levels with a micronutrient test
This micronutrient test checks for vitamin B12, D, E, Magnesium, Copper, Selenium & Zinc. Get your results in 2-5 days from an accredited laboratory with free shipping.
- your wake-up time
- completing five or six 90-minute sleep cycles
- allowing 15 minutes to fall asleep
|Wake-up time|| Bedtime: |
7.5 hours of sleep
(5 cycles)
| Bedtime: |
9 hours of sleep
(6 cycles)
|4 a.m.||8:15 p.m.||6:45 p.m.|
|4:15 a.m.||8:30 p.m.||7 p.m.|
|4:30 a.m.||8:45 p.m.||7:15 p.m.|
|4:45 a.m.||9 p.m.||7:30 p.m.|
|5 a.m.||9:15 p.m.||7:45 p.m.|
|5:15 a.m.||9:30 p.m.||8 p.m.|
|5:30 a.m.||9:45 p.m.||8:15 p.m.|
|5:45 a.m.||10 p.m.||8:30 p.m.|
|6 a.m.||10:15 p.m.||8:45 p.m.|
|6:15 a.m.||10:30 p.m.||9 p.m.|
|6:30 a.m.||10:45 p.m.||9:15 p.m.|
|6:45 a.m.||11 p.m.||9:30 p.m.|
|7 a.m.||11:15 p.m.||9:45 p.m.|
|7:15 a.m.||11:30 p.m.||10 p.m.|
|7:30 a.m.||11:45 p.m.||10:15 p.m.|
|7:45 a.m.||12 p.m.||10:30 p.m.|
|8 a.m.||12:15 a.m.||10:45 p.m.|
|8:15 a.m.||12:30 a.m.||11 p.m.|
|8:30 a.m.||12:45 a.m.||11:15 p.m.|
|8:45 a.m.||1 a.m.||11:30 p.m.|
|9 a.m.||1:15 a.m.||11:45 p.m.|
Sleep deprivation is a real concern for many people, especially those faced with consistent work and life challenges that can further disrupt sleep.
Health and mental health concerns — depression , anxiety , obstructive sleep apnea , and chronic pain , just to name a few — can contribute to sleep deprivation. But a lack of quality sleep can also worsen symptoms of these conditions and fuel a distressing cycle of sleeplessness.
The occasional night of poor sleep generally won’t have a serious impact on your health. All the same, experts have linked ongoing sleep deprivation to serious health consequences, including a higher risk of chronic diseases and early death.
Sleep deprivation can have short-term and long-term physical, emotional, and cognitive health impacts.
For most people, a night of poor sleep can bring on noticeable physical effects, including:
Long-term sleep deprivation can take a more severe toll on your physical health, leading to:
- reduced immunity , which can make it harder for your body to fight off infections
- high cortisol , which can contribute to high blood pressure and other health concerns
- increased appetite and cravings for sugar and carbs
Without a doubt, a night of bad sleep can affect your mood the next day.
When you don’t get enough sleep, you’re more likely to:
- feel cranky and irritable
- notice abrupt mood changes and difficulty managing emotions
When you don’t get enough sleep, your brain can’t work as efficiently. As a result, you’ll likely have trouble concentrating and remembering things after a night of poor sleep.
Research has found evidence to suggest that sleep deprivation negatively affects functions associated with the brain’s frontal lobe , including:
- attention
- alertness
- decision making
- judgment
- memory
- response
These effects can play a part in:
- declining performance at work or school
- changes in judgment and impulse control
- accidents
When you fall asleep, your brain and body go through several cycles of sleep. Each cycle includes four distinct stages .
- The first three stages are part of non-rapid eye movement (NREM) sleep.
- The last stage is rapid eye movement (REM) sleep.
The stages used to be classified as stages 1, 2, 3, 4, and REM. Now,
as:
- N1 (formerly stage 1). This first stage of sleep marks the period between being awake and falling asleep.
- N2 (formerly stage 2). The onset of sleep begins at this stage, as you become unaware of your surroundings. Your body temperature drops slightly, and your breathing and heart rate become regular.
- N3 (formerly stages 3 and 4). During this deepest and most restorative sleep stage, breathing slows, blood pressure drops, muscles relax, hormones are released, healing occurs, and your body becomes re-energized.
- REM. This final stage takes up about 25 percent of your sleep cycle. During REM sleep, your brain is most active, dreams occur, and your eyes move back and forth rapidly under your eyelids. REM sleep helps boost your mental and physical performance when you wake up.
It takes, on average, about 90 minutes to go through each cycle. Completing five cycles a night means you’d get 7.5 hours of sleep, while six full cycles translates to about 9 hours of sleep.
Ideally, you want to wake up at the end of a sleep cycle instead of in the middle of it — that’s because you’ll typically feel more refreshed and energized if you wake up at the end of a cycle.
- helps regulate the release of hormones that control appetite, metabolism, growth, and healing
- boosts brain function , concentration, focus, and productivity
- reduces your risk for heart disease and stroke
- lowers your risk for chronic health conditions, such as diabetes and high blood pressure
- improves athletic performance , reaction time, and speed
- may lower your risk for depression
- improves libido and sexual function
Supplements 101: Vitamin D
Watch this video to learn the benefits of vitamin D, plus information about downsides, how much you need, and foods that are rich in vitamin D.
0 seconds of 3 minutes, 19 seconds Volume 0%
3:19
You’ll find answers to some common questions about sleep below.
Yes, your need for sleep does change with age, though it typically stabilizes around the age of 20 .
As you get older, you need less sleep, as a general rule.
Various environmental, behavioral, and medical factors can influence how much sleep you need, though, and those may change throughout your life.
For instance:
- A young adult may want to do more — and stay up later — than they could as a teenager.
- An adult in their 40s has a higher chance of chronic health conditions that might affect their sleep needs.
- Changing lifestyles in older age, including an irregular schedule, may lead to more time spent in bed.
There are a few possible reasons you might wake up tired , even after sleeping for 8 hours. A good place to start exploring these reasons? Consider your sleep habits and sleep hygiene practices .
When it comes to sleep, quality matters just as much as quantity. Things that could detract from the quality of your sleep include:
- your sleep environment (Is it noisy? Too hot or cold? Too bright?)
- who you share your bed with (A partner who snores or fidgets? A restless pet?)
- sleep disorders like insomnia or sleep apnea
- chronic pain
- an underlying medical or mental health condition
Pulling all-nighters, or working the graveyard shift and then sleeping in the day, may contribute to some negative health effects, including increased risk for cardiovascular disease and type 2 diabetes.
Research suggests that being a night owl could also affect your eating habits and lead to erratic eating patterns, including:
- skipping breakfast and overeating later in the day
- consuming more sugar, caffeine, alcohol, and fast food
What’s more, getting quality sleep during the day can be a challenge, with all the distractions and noise of life happening around you.
When you don’t have any option beyond working at night and sleeping during the day, these tips can help you get better rest.
To improve your sleep health, consider the following tips.
- Exercise regularly, but try to schedule your workouts at least a few hours before you go to sleep. Exercising too close to bedtime may lead to interrupted sleep.
- Increase your exposure to sunlight or bright lights during the day. This can help maintain your body’s circadian rhythms , which affect your sleep-wake cycle.
- Try not to take long naps , especially late in the afternoon.
- Limit alcohol, caffeine, and nicotine in the evening. These substances have the potential to interrupt your sleep or make it difficult to fall asleep.
- Switch off electronics at least 30 minutes before bedtime. The light from these devices can stimulate your brain and make it harder to fall asleep.
- Get into the habit of a relaxing routine before bedtime , like taking a hot bath or listening to soothing music.
- Turn down the lights shortly before bedtime to help your brain understand that it’s time to sleep.
- Turn down the thermostat in your bedroom. 65°F (18.3°C) is an ideal sleeping temperature .
- Avoid screen time in bed to reduce blue light exposure , which can disrupt sleep.
- Read a book or listen to white noise to help you relax.
- Close your eyes, relax your muscles, and focus on steady breathing .
- If you’re unable to fall asleep, get out of bed and move to another room. Read a book or listen to music until you start feeling tired, then go back to bed.
If you’re aiming for 7 to 9 hours of sleep each night, a sleep calculator (like the one above) can help you figure out what time to go to bed based on your wake-up time.
Ideally, you’ll want to wake up at the end of your sleep cycle, which is when you’re most likely to feel the most rested.
A good night’s sleep is essential to good health, so if you’re having trouble falling asleep or staying asleep, consider reaching out to a healthcare professional. They can help you explore underlying causes of sleep difficulties and offer guidance.
"""]


    claim = ["Another way is to use a sleep calculator."]
    from alignscore import AlignScore

    scorer = AlignScore(model='roberta-base', batch_size=32, device='cpu', ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt', evaluation_mode='nli_sp')
    score = scorer.score(contexts=passage, claims=claim)
    print(score)
