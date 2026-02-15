import argparse
import json

from jsonlines import jsonlines

from model.Qwen_model import QwenTool
from tool.Retriever import Retriever
from tool.compute_score import eval_ex_match, LLM_eval


def augments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Qwen')
    parser.add_argument("--dataset", type=str, default='hitab')
    parser.add_argument("--qa_path", type=str, default='/home/zys/MyTQA/dataset/hitab/test.jsonl')
    parser.add_argument("--table_folder", type=str,default='/home/zys/MyTQA/dataset/hitab/mini_raw/' )
    parser.add_argument("--max_iteration_depth", type=int, default=4)
    parser.add_argument("--start",  type=int,default=1385) #比test小1  162
    parser.add_argument("--end",  type=int,default=1585)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    return args
def load_model(args):
    model=QwenTool(args)
    return  model


def load_data(args):
    querys,answers,table_captions,tables,table_paths,top_heads,left_heads= [],[],[],[],[],[],[]
    if args.dataset in ('hitab','Hitab'):
        qas = []
        with open(args.qa_path, "r+", encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                qas.append(item)
        qas = qas[args.start:args.end]
        for qa in qas:
            # print(qa['table_id'])
            table_path = args.table_folder + qa['table_id'] + '.json'
            with open(table_path, "r+", encoding='utf-8') as f:
                table = json.load(f)
            table_captions.append(table['title'])
            answers.append('|'.join([str(i) for i in qa['answer']]))
            querys.append( qa['question'])
    return querys



def parse_txt(path):
    a_list = []
    b_list = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 查找 model: 和 answer:
            try:
                # 找到 "model:" 的位置
                model_pos = line.index("model:") + len("model:")
                # 找到 "answer:" 的位置
                answer_pos = line.index("answer:")

                # model 值在 model: 后到 answer: 前
                model_value = line[model_pos:answer_pos].strip()

                # answer 值在 answer: 后
                answer_value = line[answer_pos + len("answer:"):].strip()

                a_list.append(model_value)
                b_list.append(answer_value)

            except ValueError:
                # 如果格式不对，则跳过该行
                continue

    return a_list, b_list


def main():
    args = augments()
    model= load_model(args)
    querys= load_data(args)
    a,b=parse_txt("/home/zys/MyTQA/result/v11_result.txt")
    EM=0
    LLM_EVAL=0
    total_num=0
    for query,ans, label in zip(querys, a,b):

        p=eval_ex_match(ans,label)
        EM+=p
        k= LLM_eval(model,query,ans,label)
        LLM_EVAL+=k
        total_num +=1
        print(ans+"  "+label+"  "+str(p)+"  "+str(k))
    print('EM:',EM/total_num)
    print('LLM EVAL:',LLM_EVAL/total_num)

if __name__ == '__main__':
    main()