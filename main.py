import argparse
import json
import random
import jsonlines
from model.Qwen_API import QwenAPI
from model.Qwen_model import QwenTool
from model.LLama import  LLaMa
from tool.Retriever import Retriever
from tool.V_Retriever import V_Retriever
from tool.compute_score import eval_ex_match, LLM_eval
from tool.extract_head import extract_paths, process_table, remove_commas_in_2d_str_list, replace_comma_with_space


def augments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Qwen')
    parser.add_argument("--dataset", type=str, default='hitab')
    parser.add_argument("--qa_path", type=str, default='/home/zys/MyTQA/dataset/one/one.jsonl')
    parser.add_argument("--table_folder", type=str,default='/home/zys/MyTQA/dataset/one/raw/' )
    parser.add_argument("--max_iteration_depth", type=int, default=4)
    parser.add_argument("--start",  type=int,default=0) #比test小1  162
    parser.add_argument("--end",  type=int,default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    return args

def load_model(args):
    model=LLaMa()
    return  model


def load_data(args):
    querys,answers,table_captions,tables,table_paths,top_heads,left_heads= [],[],[],[],[],[],[]
    table_id=[]
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
            answers.append(', '.join([str(i) for i in qa['answer']]))
            querys.append( qa['question'])
            table_id.append(qa['table_id'])
            table_paths.append(table_path)
            text=table['texts']
            top_head_rows_num = table['top_header_rows_num']
            left_head_columns_num = table['left_header_columns_num']
            new_text=process_table(text,top_head_rows_num,left_head_columns_num)
            new_text=remove_commas_in_2d_str_list(new_text)
            tables.append(new_text)
            top_head = extract_paths(table['top_root'], table['texts'], axis='top')
            left_head = extract_paths(table['left_root'], table['texts'], axis='left')
            top_head=remove_commas_in_2d_str_list(top_head)
            left_head=remove_commas_in_2d_str_list(left_head)
            top_head=replace_comma_with_space(top_head)
            left_head = replace_comma_with_space(left_head)
            top_heads.append(top_head)
            left_heads.append(left_head)

    elif args.dataset in ('AIT-QA','ait-qa'):
        with open(args.qa_path, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        qas = qas[args.start:args.end]

        for qa in qas:
            tables.append(qa['table'])
            answers.append('|'.join([str(i) for i in qa['answers']]))
            querys.append( qa['question'])
            table_captions.append('')
            table_paths.append(qa)
    return querys,answers,table_captions,tables,top_heads,left_heads,table_id

def main():
    args = augments()
    model= load_model(args)
    querys, answers, table_captions, tables,top_heads,left_heads,table_id= load_data(args)
    total_num,EM,LLM_EVAL = args.start+1,0,0
    file = "/home/zys/MyTQA/result/v_one.txt"
    with open(file, "a", encoding="utf-8") as f:
        for query, label, caption, table,top_head,left_head,id in zip(querys, answers, table_captions, tables,top_heads,left_heads,table_id):
            # if total_num in a:
            # retriever = Retriever(query, caption, table, model, top_head, left_head, id, total_num)
            # text, table, first_reason, second_reason, ans, i = retriever.run()
            retriever = V_Retriever(query, caption, table, model, top_head, left_head, id, total_num)
            table, text, first_reason, second_reason, org_ans, ans, i = retriever.run()
            print("模型回答为:", ans, ' 答案为:', label + "\n")

            f.write(str(total_num) + ': ' + query + "\n" + table + "\n")
            f.write("text:\n" + text + '\nReason:\n' + first_reason + "\nCorrect:\n" + second_reason + "\n" + "i:" + str(i) + "\n")
            f.write("纠错前:" + str(org_ans)+"\n")
            f.write("模型回答为:" + str(ans) + ' 答案为:' + label + "\n\n")

            total_num += 1
            # EM += eval_ex_match(result,answer)
            # LLM_EVAL += LLM_eval(model,query,output,answer)
            # print('EM:',EM/total_num)
            # print('LLM EVAL:',LLM_EVAL/total_num)

if __name__ == '__main__':
    main()