import re

from model.Qwen_model import QwenTool
from tool.compute_score import eval_ex_match

def extract_model_answers(txt_path):
    p=r"纠错前:\s*(.*)"
    pattern = r"模型回答为:(.+?)答案为:(.+)"  # 匹配两种可能的冒号格式
    EM=0
    total=0
    last=""
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            a=re.search(p,line.strip())
            if a:
                last=a.group(1).strip()
                last=last.replace("%","").replace("-","").lower().replace(".0","").replace("\"","")
            match = re.search(pattern, line.strip())
            if match:
                total = total + 1
                result = match.group(1).strip()
                result=result.replace("\"","").replace(".0","").replace("%","").replace("-","").lower()
                answer = match.group(2).strip()
                if last!="none" and result!=last:
                    print("last:"+last+" result:"+result+" label:"+answer)
                # print(str(total) + ":model:" + result + " answer:" + answer)
                # k=eval_ex_match(result,answer)
                # EM += k
                # if k==0:
                #     print(str(total)+":model:"+result+" answer:"+answer)
                #     # print(str(total),end=",")
                # if total==1600:
                #     break



    print(EM)
    print(total)
    print(EM/total)

extract_model_answers("/home/zys/MyTQA/result/v_llama.txt")