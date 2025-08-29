'''
conda activate mistral
export CUDA_VISIBLE_DEVICES=0
nohup /maindata/data/shared/public/yangchao.zhou/anaconda3/envs/mistral/bin/python /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/evaluation/linky_score.py > linky_score.log 2>&1 &
ps -ef | grep linky_score.py

'''
from transformers import pipeline
import logging
import json
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from openai import OpenAI
import jinja2


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

openai_api_key = "EMPTY"
openai_api_base = "http://172.20.207.171:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


class ModelEvaluator:
    def __init__(self, model_path, data_path, prompt_path):
        self.model_path = model_path
        self.data_path = data_path
        self.client = client
        self.data = self.load_data(data_path)
        self.prompt_path = prompt_path

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_prompt_template(self):
        with open(self.prompt_path, 'r') as f:
            template = f.read()
        template = jinja2.Template(template)
        return template
    
    def evaluate_output(self, data_entry):
        instruction = self.process_prompt_template()
        conversations = data_entry.get('conversations', [])
        is_correct_score_list = []
        messages = []
        old_messages = []

        round = 0
        for example in conversations:
            
            tmp_instruction = instruction.render({"round":round})
            if example["from"] == 'user':
                round+=1
                input = example["value"]
                old_messages += [{"role": "user", "content":input}]

            elif example["from"] == 'assistant':
                npc_replay = example["value"]
                expect_score = example["score"]
                old_messages += [{"role": "assistant", "content": npc_replay}]
                res = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=old_messages+ [{"role": "user", "content":tmp_instruction}],
                    max_tokens=32768
                    
                )
                res_txt = res.choices[0].message.content
                logging.info(f"round:{round}") 
                logging.info(f"input:\n{input}") 
                logging.info(f"npc_replay:\n{npc_replay}")
                logging.info(f"res_txt:\n {res_txt}")
                logging.info(f'expect_score:\n{expect_score}')
                logging.info('===============')
                # is_correct_score = self.if_correct_score(res_txt, expect_score)
                # is_correct_score_list.append(is_correct_score)


        # messages +=  [{"role": "user", "content": input}]
        # res = self.client.chat.completions.create(
        #     model=self.model_path,
        #     messages=messages
        # )
        # messages = res['choices'][0]['message']['content']
        # is_correct_score = self.if_correct_score(messages, last_expect_score)
        # is_correct_score_list.append(is_correct_score)

                
        return is_correct_score_list

    def if_correct_score(self, res_txt, expect_score):

        generated_text = res_txt
        # print(f"generated_text:\n\n {generated_text}\n\n")
        # print(f"expect_score:\n\n {expect_score}\n\n")
        
        # score_in_generated = self.extract_score(generated_text)
        # score_in_expected = self.extract_score(expect_score)
        reason_advantage_pre, reason_disadvantage_pre, score_pre, comment_pre = self.extract_fields(generated_text)
        reason_advantage_truth, reason_disadvantage_truth, score_truth, comment_truth = self.extract_fields(expect_score)
        
        comment_pair = (comment_pre, comment_truth)
        advantage_pair = (reason_advantage_pre, reason_advantage_truth)
        disadvantage_pair = (reason_disadvantage_pre, reason_disadvantage_truth)
        score_pair = (score_pre, score_truth)
        is_correct_score = (score_pre == score_truth)
        distance_score = score_truth - score_pre
        abs_distance_score = abs(distance_score)
        return [advantage_pair, disadvantage_pair, score_pair, is_correct_score, distance_score, abs_distance_score, comment_pair]


    @staticmethod
    def extract_score(text):
        import re
        match = re.search(r'Score:\s*(\d)', text)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_fields(text):
        import re
        
        reason_advantage_match = re.search(r'\*\*Reason-advantage\*\*:\s*(.*?)\n', text)
        reason_disadvantage_match = re.search(r'\*\*Reason-disadvantage\*\*:\s*(.*?)\n', text)
        comment_match = re.search(r'Comment:\s*(.*?)\n', text)
        
        score_match = re.search(r'\*\*Score:\*\*\s*(\d)', text)
        if not score_match:
            score_match = re.search(r'\*\*Score\*\*:\s*(\d)', text)
            comment_match = re.search(r'\*\*Comment\*\*:\s*(.*?)\n', text)
        
        reason_advantage = reason_advantage_match.group(1) if reason_advantage_match else None
        reason_disadvantage = reason_disadvantage_match.group(1) if reason_disadvantage_match else None
        score = int(score_match.group(1)) if score_match else None
        
        return reason_advantage, reason_disadvantage, score, comment_match

    def run_evaluation(self):
        all_is_correct_score_list = []
        all_predicted_scores = []
        all_true_scores = []

        for entry in tqdm(self.data, desc="Evaluating"):
            is_correct_score_list = self.evaluate_output(entry)
            all_is_correct_score_list.append(is_correct_score_list)
            all_predicted_scores.extend([score[2][0] for score in is_correct_score_list])
            all_true_scores.extend([score[2][1] for score in is_correct_score_list])
        
        # Flatten the list
        flat_is_correct_score_list = [score[3] for sublist in all_is_correct_score_list for score in sublist]
        
        # Calculate accuracy
        accuracy = accuracy_score([True] * len(flat_is_correct_score_list), flat_is_correct_score_list)
        
        # Calculate the ratio of True values
        true_ratio = sum(flat_is_correct_score_list) / len(flat_is_correct_score_list) if flat_is_correct_score_list else 0
        
        print(f"Accuracy: {accuracy}")
        print(f"True ratio: {true_ratio:.2%}")
        print(f"all_is_correct_score_list: {all_is_correct_score_list}")

        # Calculate per-score accuracy
        per_score_accuracy = accuracy_score(all_true_scores, all_predicted_scores)
        print(f"Per-score accuracy: {per_score_accuracy}")

        # # Calculate macro average accuracy
        # macro_avg_accuracy = accuracy_score(all_true_scores, all_predicted_scores, average='macro')
        # print(f"Macro average accuracy: {macro_avg_accuracy}")

        # # Calculate weighted average accuracy
        # weighted_avg_accuracy = accuracy_score(all_true_scores, all_predicted_scores, average='weighted')
        # print(f"Weighted average accuracy: {weighted_avg_accuracy}")

        # Calculate F1 score
        f1 = f1_score(all_true_scores, all_predicted_scores, average='weighted')
        print(f"F1 score: {f1}")

        # Print confusion matrix
        conf_matrix = confusion_matrix(all_true_scores, all_predicted_scores)
        print(f"Confusion matrix:\n{conf_matrix}")

        # Print classification report
        class_report = classification_report(all_true_scores, all_predicted_scores)
        print(f"Classification report:\n{class_report}")

        # Save to CSV
        df = pd.DataFrame(all_is_correct_score_list, columns=['advantage_pair', 'disadvantage_pair', 'score_pair', 'is_correct_score', 'distance_score', 'abs_distance_score'])
        df.to_csv('evaluation_results.csv', index=False)
        df.to_excel('evaluation_results.xlsx', index=False)

    
    # def run_evaluation(self):
    #     all_is_correct_score_list = []
    #     for entry in tqdm(self.data, desc="Evaluating"):
    #         is_correct_score_list = self.evaluate_output(entry)
    #         all_is_correct_score_list.append(is_correct_score_list)
        
    #     # Flatten the list
    #     flat_is_correct_score_list = [score[3] for sublist in all_is_correct_score_list for score in sublist]
    #     distance_score_list = [score[4] for sublist in all_is_correct_score_list for score in sublist]
    #     abs_distance_score_list = [score[5] for sublist in all_is_correct_score_list for score in sublist]
        
        
    #     # Calculate accuracy
    #     accuracy = accuracy_score([True] * len(flat_is_correct_score_list), flat_is_correct_score_list)
    #     avg_score_distance = sum(distance_score_list) / len(distance_score_list) if distance_score_list else 0
    #     abs_score_distance = sum(abs_distance_score_list) / len(abs_distance_score_list) if abs_distance_score_list else 0
        
    #     # Calculate the ratio of True values
    #     true_ratio = sum(flat_is_correct_score_list) / len(flat_is_correct_score_list) if flat_is_correct_score_list else 0
        
    #     print(f"Accuracy: {accuracy}")
    #     print(f"True ratio: {true_ratio:.2%}")
    #     # print(f"all_is_correct_score_list: {all_is_correct_score_list}")
    #     print(f"avg_score_distance: {avg_score_distance}")
    #     print(f"abs_distance_score: {abs_score_distance}")
    

        # df = pd.DataFrame(all_is_correct_score_list, columns=['advantage_pair', 'disadvantage_pair', 'score_pair', 'is_correct_score', 'distance_score', 'abs_distance_score'])
        # df.to_csv('evaluation/evaluation_results.csv', index=False)
        # df.to_excel('evaluation/evaluation_results.xlsx', index=False)

if __name__ == "__main__":
    model_path = "/maindata/data/shared/ai_story_workspace-dsw/nlp_models/mistralai/Mistral-Small-24B-Instruct-2501"
    # model_path = "/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-score-20250220/checkpoint-107/"
    # model_path = '/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-score-20250221-lima-deepseek_10/checkpoint-400'
    data_path = '/maindata/data/shared/public/yangchao.zhou/projects/mistral_pro/data/instruction/20250226-sharegpt/train/combined_data_20250221_3.json'
    prompt_path = "/maindata/data/shared/public/yangchao.zhou/projects/mistral_pro/prompt/按照轮次打分.tmpl"
    evaluator = ModelEvaluator(model_path, data_path, prompt_path)
    evaluator.run_evaluation()
