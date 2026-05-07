import ast
import json
import operator
import os
import re
import sys
from typing import List, Tuple

from tool.Retriever import build_full_table, list_to_markdown, extract_headers
from tool.v_prompt import PROMPT_TEMPLATE

class V_Retriever:
    def __init__(self,query, caption, table, model,top_head,left_head,id,index):
        self.table=table
        self.query=query
        self.caption=caption
        self.model=model
        self.top_head=top_head
        self.left_head=left_head
        self.id=id
        self.index=index
        self.json_list=[f for f in os.listdir("/home/zys/MyTQA/dataset/hitab/understand") if f.endswith(".json")]

    def flatten_heads(self, top, left):
        system_instruction = PROMPT_TEMPLATE['Flatten_system_instruction']
        examples = PROMPT_TEMPLATE['Flatten_examples']
        if all(len(row) == 1 for row in top):
            new_top = [row[0] for row in top]
        else:
            top_prompt = PROMPT_TEMPLATE['Flatten_prompt'].replace('{example}', examples).replace('{Headers}', str(top))
            top_result = self.model.generate(top_prompt, system_instruction)
            new_top = extract_headers(top_result)
        if all(len(row) == 1 for row in left):
            new_left = [row[0] for row in left]
        else:
            left_prompt = PROMPT_TEMPLATE['Flatten_prompt'].replace('{example}', examples).replace('{Headers}',
                                                                                                   str(left))
            left_result = self.model.generate(left_prompt, system_instruction)
            new_left = extract_headers(left_result)
        return new_top, new_left
    def understand(self,caption,top,left):
        system_instruction = PROMPT_TEMPLATE['understand_system_instruction']
        examples = PROMPT_TEMPLATE['understand_examples']
        prompt = PROMPT_TEMPLATE['understand_prompt'].replace('{question}',self.query).replace('{caption}', caption).replace('{top_headers}', str(top)).replace('{left_headers}', str(left)).replace('{examples}',examples)
        result = self.model.generate(prompt, system_instruction)
        return result

    def answer(self,table,caption):
        system_instruction = PROMPT_TEMPLATE['answer_system_instruction']
        # examples = PROMPT_TEMPLATE['answer_example']
        prompt = PROMPT_TEMPLATE['answer_prompt'].replace('{question}',self.query).replace('{table}', table).replace('{caption}', caption)
        result = self.model.generate(prompt, system_instruction)
        return result

    def check(self,table,reason,amend,caption):
        system_instruction = PROMPT_TEMPLATE['checker_system_instruction']
        prompt = PROMPT_TEMPLATE['checker_prompt'].replace('{question}', self.query).replace('{reason}', reason).replace('{table}', str(table)).replace('{amend}',amend).replace('{caption}',caption)
        result = self.model.generate(prompt, system_instruction)
        return result

    def run(self):
        top_global_headers, top_head = extract_global_headers(self.top_head)
        if top_global_headers != []:
            str1 = "The top header means " + "、".join(top_global_headers)
            self.caption = self.caption + ". " + str1 + "."
        left_global_headers, left_head = extract_global_headers(self.left_head)
        if left_global_headers != []:
            str2 = "The left header means " + "、".join(left_global_headers)
            self.caption = self.caption + ". " + str2 + "."
        try:
            new_top_heads, new_left_heads = self.flatten_heads(top_head, left_head)
            table = build_full_table(new_top_heads, new_left_heads, self.table)
            print(table)
            table = list_to_markdown(table)
            print(table)
            text = self.understand(self.caption, new_top_heads, new_left_heads)
            print(text)
            first_reason = self.answer(table, text)
            print(first_reason)
            org_ans = extract_answer_output(first_reason)
            second_reason = ""
            ans = ""
            judge = verify_compute_and_sort(first_reason)
            print("judge:" + judge)
            tmp_reason = first_reason
            i = 0
            while judge != "correct" and i < 3:
                second_reason = self.check(table, tmp_reason, judge, self.caption)
                print("sed reason:" + second_reason)
                judge = verify_compute_and_sort(second_reason)
                print("sed judge:" + judge)
                tmp_reason = second_reason
                i += 1
            if second_reason == "":
                ans = extract_answer_output(first_reason)
            else:
                ans = extract_answer_output(second_reason)
            return table, text, first_reason, second_reason, org_ans, ans, i

        except Exception as e:
            # 返回错误字符串
            return "","","","","","错误",0



def extract_global_headers(header_paths: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    提取表头路径中的全局表头元素。
    """
    if not header_paths:
        return [], []
    if len(header_paths)==1:
        return [], header_paths
    n_levels = max(len(path) for path in header_paths)
    global_headers = []
    remove_indices = []
    for level in range(n_levels):
        # 收集当前层所有元素
        elems = [path[level] for path in header_paths if len(path) > level]
        if len(elems) == len(header_paths) and len(set(elems)) == 1:
            # 该层元素在所有路径中完全相同 → 全局表头
            global_headers.append(elems[0])
            remove_indices.append(level)
    # 构造新路径，去掉全局表头所在层级
    new_paths = []
    for path in header_paths:
        new_path = [elem for idx, elem in enumerate(path) if idx not in remove_indices]
        new_paths.append(new_path)
    return global_headers, new_paths

def extract_answer_output(text: str):
    """
    从 reasoning 内容中提取 ANSWER 步骤的 output 值（只抓取 output: 后同一行内容）
    """
    pattern = r"STEP\s*\d+:\s*ANSWER\(.*?\)[\s\S]*?output:\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    # 只取本行内容，不吃后续内容
    line = match.group(1).strip()
    line = line.splitlines()[0].strip()
    return line



def load_three_fields(folder_path, filename):
    """
    从指定文件夹下的某个 json 文件中读取三个字段的内容

    参数:
        folder_path: 文件夹路径
        filename: json 文件名（例如 'data.json'）
        field1, field2, field3: 要读取的字段名字符串

    返回:
        (value1, value2, value3)
    """
    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("top"), data.get("left"), data.get("table")


import re
import ast
import operator
from typing import Union, Dict, Any

NUM_RE = r"-?\d+(?:\.\d+)?"

# ---------------------------------------------------------
# 安全 evaluator（支持 + - * / ()，不允许变量，不允许函数）
# ---------------------------------------------------------
allowed_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

def safe_eval_expr(expr: str) -> float:
    """
    安全解析表达式，支持括号和 + - * /，拒绝其他语法。
    """
    node = ast.parse(expr, mode='eval')

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        elif isinstance(n, ast.Num):  # 旧版本 Python
            return float(n.n)

        elif hasattr(ast, "Constant") and isinstance(n, ast.Constant):  # 新版本
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("Invalid constant")

        elif isinstance(n, ast.BinOp):
            if type(n.op) not in allowed_ops:
                raise ValueError("Invalid operator")
            return allowed_ops[type(n.op)](_eval(n.left), _eval(n.right))

        elif isinstance(n, ast.UnaryOp):
            if type(n.op) not in allowed_ops:
                raise ValueError("Invalid unary operator")
            return allowed_ops[type(n.op)](_eval(n.operand))

        else:
            raise ValueError("Invalid syntax element")

    return _eval(node)


# ---------------------------------------------------------
# 表达式格式检查（允许括号）
# ---------------------------------------------------------
def _is_strict_multi_compute_formula_mixed(expr: str) -> bool:
    """
    严格检查格式： left = right
    left 可包含数字 + - * / ()，但不能出现非法字符
    """
    if expr.count("=") != 1:
        return False

    expr_ns = re.sub(r"\s+", "", expr)

    left, right = expr_ns.split("=", 1)

    # right 必须是纯数字
    if not re.fullmatch(r"-?\d+(\.\d+)?", right):
        return False

    # left 允许数字、括号、+ - * /
    if not re.fullmatch(r"[0-9+\-*/().]+", left):
        return False

    # 括号必须成对
    if left.count("(") != left.count(")"):
        return False

    return True


# ---------------------------------------------------------
# 解析表达式
# ---------------------------------------------------------
def _parse_multi_compute_formula_mixed(expr: str):
    if expr.count("=") != 1:
        return None

    expr_ns = re.sub(r"\s+", "", expr)
    left, right = expr_ns.split("=", 1)

    try:
        expected = float(right)
    except ValueError:
        return None

    # 使用安全 evaluator
    try:
        computed = safe_eval_expr(left)
    except Exception:
        return None

    return computed, expected


# ---------------------------------------------------------
# 排序检查（不变）
# ---------------------------------------------------------
def _is_strict_number_list(s: str) -> bool:
    return re.fullmatch(rf"\s*\[\s*{NUM_RE}(?:\s*,\s*{NUM_RE})*\s*\]\s*", s) is not None

def _parse_number_list(s: str):
    if not _is_strict_number_list(s):
        return None
    nums = re.findall(NUM_RE, s)
    return [float(x) for x in nums]

def is_sorted(nums, reverse=False):
    op = operator.ge if reverse else operator.le
    return all(op(a, b) for a, b in zip(nums, nums[1:]))


# ---------------------------------------------------------
# 主函数：验证 COMPUTE 与 SORT
# ---------------------------------------------------------
def verify_compute_and_sort(text: str) -> Union[str, Dict[str, Any]]:
    step_iter = list(re.finditer(r"(STEP\s*(\d+)\s*:)", text, flags=re.IGNORECASE))
    blocks = []
    for i, m in enumerate(step_iter):
        start = m.start()
        step_no = int(m.group(2))
        end = step_iter[i + 1].start() if i + 1 < len(step_iter) else len(text)
        blocks.append((step_no, text[start:end]))

    for step_no, block in blocks:
        header = block.splitlines()[0] if block.splitlines() else ""

        # ------------------ COMPUTE ------------------
        if re.search(r"\bCOMPUTE\b", block, flags=re.IGNORECASE):
            m_formula = re.search(r"formula\s*:\s*(.+)", block, flags=re.IGNORECASE)
            if not m_formula:
                return "COMPUTE step is missing formula field or unable to extract formula."
            raw_formula = m_formula.group(1).strip()

            if not _is_strict_multi_compute_formula_mixed(raw_formula):
                return f"Format error: Invalid COMPUTE formula in {raw_formula}"

            parsed = _parse_multi_compute_formula_mixed(raw_formula)
            if parsed is None:
                return f"Format error: Invalid COMPUTE formula in {raw_formula}"
            computed, expected = parsed

            if abs(computed - expected) > 1e-2:
                return (
                    f"Incorrect calculation: {raw_formula} "
                    f"(correct result is {round(computed, 2)})"
                )

        # ------------------ SORT ------------------
        if re.search(r"\bSORT\b", block, flags=re.IGNORECASE):
            m_output = re.search(r"output\s*:\s*(.+)", block, flags=re.IGNORECASE)
            if not m_output:
                return "SORT step is missing output field or unable to extract output"

            raw_output = m_output.group(1).strip()

            if not _is_strict_number_list(raw_output):
                return "Format error: SORT output must be a strict number list like [1,2,3]"

            nums = _parse_number_list(raw_output)
            if nums is None:
                return "Format error: SORT output must be a strict number list like [1,2,3]"

            if not (is_sorted(nums) or is_sorted(nums, reverse=True)):
                return f"{nums} sorting error, correct ascending: {sorted(nums)}"

    return "correct"
