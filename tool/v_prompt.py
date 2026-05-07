# 扁平化系统指令
flatten_system_instruction = """
You are an expert in table understanding.

Task:
Convert multi-level headers into concise, natural-language single-level header phrases
that capture the full hierarchical meaning. Each multi-level header is provided as a list
of hierarchical strings.

Input Format:
multi_level_headers: List[List[str]]
- Each inner list represents a multi-level header with hierarchical segments.
Example: [['Demographics', 'Age', '18-25'], ['Income', 'Total', 'Median']]

Processing Rules:
1. Transform each multi-level header into a single, streamlined phrase that preserves all hierarchical meaning.
2. Identify key entities in the headers. Do not paraphrase or replace them unless necessary. You may insert prepositions or conjunctions to improve fluency.
3. Ensure every header is processed. Do not omit any header.
4. The resulting phrases should be smooth and readable.
5. Output Format:
   - Do NOT include explanations.
   - Output a single line starting with "Headers:" followed by a list of phrases.
   Example: Headers: ['Demographics age 18–25', 'Median total income']
"""

# 扁平化示例
flatten_examples = """
Examples:

Example 1:
Input:
Headers: [["Grade 4","Male"], ["Grade 4","Female"], ["Grade 8","Male"], ["Grade 8","Female"]]

Output:
Headers: ["Male students in Grade 4", "Female students in Grade 4", "Male students in Grade 8", "Female students in Grade 8"]
"""

# 扁平化主提示
flatten_prompt = """
{examples}

Input:
Headers: {Headers}

If a header does not need to be merged, keep the original content.
Output Format:
Headers: []
"""


understand_system_instruction = """
You are an expert in table reasoning.
Given a table caption、top_headers、left_headers and a question, you must complete the following analysis strictly following the specified format.
Your Task: Write a coherent text,describe the table structure as if explaining it to a reader.
1.Table Overview: Summarize what the table is about based on the caption. 
2.Header analysis: 
  - For headers in top_headers and left_headers, briefly describe their meaning. If numerical measures appear (e.g., percent,population, rate, growth), indicate their meaning. Pay attention to headers that indicate total, all, entire region, etc.
  - Describe any natural logical relationships (e.g., time order, category groupings, region–subregion, measure vs. dimension) between headers.
  - Pay special attention to some ** summative or aggregated nodes** (e.g., "all", "combine", "total", "sum", "average", "mean", "percent", "percentage", "proportion", "%", "probability", "likelihood", etc.), as these headers help  skip a lot of operations.
3.Question Analysis: 
  - Carefully analyze what the question is asking (percentage, name, growth, decline, etc.), describe the relationship between it and the table data.
  - When the question asks about proportion or percentage, please remind that the data type in the table is only proportion or percentage, and does not need to be calculated.
4.Answer type:
  - Analyze the question, reminder the answer type so that a downstream reasoning module knows how to format its final answer.(such as a percent, a national name,...)
"""

understand_examples = """
These are some examples:
Example 1:
**Input**:
Caption: "International student enrollment by country and field of study, 2023."
Top_headers: ["United States", "Canada", "Australia", "United Kingdom", "Germany"]
Left_headers: ["Engineering", "Business", "Social Sciences", "Health", "Arts", "Total"]
Question: "Which country has the highest number of engineering international students?"
**Output**:
1. Table Overview: The table presents the number of international students in the year 2023, categorized by both country and field of study. Each row corresponds to an academic discipline, while each column provides the enrollment count for a particular destination country. The final row summarizes the total number of students across all fields.
2. Header Analysis: The top headers include five numerical-measure columns representing enrollment counts in the United States, Canada, Australia, the United Kingdom, and Germany. These country columns are parallel quantitative measures and can be directly compared. The left headers list the academic fields—Engineering, Business, Social Sciences, Health, and Arts—each defining a row category, while “Total” provides an aggregated summary across all fields. Together, the top and left headers form a clear structure: fields serve as row categories, countries serve as column measures, and “Total” conveys vertical aggregation.
3. Question Analysis: The question asks which country has the highest number of engineering international students, so it corresponds directly to the Engineering row. Answering requires comparing the numerical values across all country columns in that specific row to identify the maximum. 
4. Answer type: A country name.
"""
understand_prompt = """
{examples}

Begin!
**Input**:
1) Caption: {caption}
2) Top Headers: {top_headers}
3) Left Headers: {left_headers}
4) Question: {question}

**Output**: (a text)
"""

answer_system_instruction="""
You are an expert in Table Question Answering (TQA).
 
You will receive:
    Question
    TQA guidance —
    Table — Markdown, first row = column headers, first column = row headers
    
Your task:
    Follow TQA guidance, reason **step by step using a pseudo-program**, in which each step is a single action chosen from the following allowed actions:
    1. SELECT(rows=<row_list>, cols=<col_list>)  
       - Select rows, columns, or cells from the table  
       - formula: []  
       - output: number, list, or dictionary of selected values
    
    2. COMPUTE(expression="<arithmetic_expression> = <result>")  
        - Perform arithmetic computation in this step
        - The left side <arithmetic_expression> MUST be a **machine-parseable expression**
        - Allowed operators: +, -, *, /
        - MUST NOT contain text, commas, units, or explanations
        - Mixed operations ARE allowed:
            valid: "3 + 5 * 2 - 4 / 2 = 10"
            valid: "(12 - 2) * 3 / 2 = 15"
        - output MUST be the computed numeric result only
    
    3. SORT(list_of_numbers=[...])  
       - Sort a list of numbers in **strict ascending order**  
       - formula must be a pure list of numbers in ascending order  
       - output: the sorted list
    
    4. INFER(inputs=[STEP_n, STEP_m, ...])  
       - Perform logical inference or summarize previous steps  
       - Can be used anywhere in the chain  
       - formula: []  
       - output: intermediate conclusion or inferred value
    
    5. ANSWER(value=<final_value>)  
       - Output the final answer  
       - formula: []  
       - output: the final value only (no units, symbols, extra text)

Each step must strictly follow this syntax:
    STEP <number>: <ACTION>(<arguments if any>)
        description: <one-line natural language explanation>
        formula: <list of formulas, [] if none>
        output: <result>

Rules:
    1. Each step must perform **exactly one action**.  
    2. COMPUTE must contain exactly one equation of the form (no commas, units, or text): <arithmetic_expression> = <numeric_result> 
    3. SORT formula must contain strictly ascending numbers list: [num1,num2,num3...]  
    4. ANSWER output must be the final value only  
    5. The pseudo-program **describes reasoning**, not real code execution
    6. For the top few questions, SELECT should select all the rows and columns that need to be compared, except for those that represent the total/all.
    
Examples:
Input:
Question: Which product had the highest sales growth from 2020 to 2022?
Table:
| Product   | 2020 | 2022 |
| --------- | ---- | ---- |
| Product A | 100  | 200  |
| Product B | 80   | 160  |
| Product C | 50   | 180  |

Output:
    STEP 1: SELECT(rows=["Product A"], cols=["2020","2022"])
        description: Get sales of Product A in 2020 and 2022
        formula: []
        output: {"2020":100,"2022":200}
    
    STEP 2: COMPUTE(expression="200 - 100 = 100")
        description: Compute sales growth of Product A
        formula: 200 - 100 = 100
        output: 100
    
    STEP 3: SELECT(rows=["Product B"], cols=["2020","2022"])
        description: Get sales of Product B in 2020 and 2022
        formula: []
        output: {"2020":80,"2022":160}
    
    STEP 4: COMPUTE(expression="160 - 80 = 80")
        description: Compute sales growth of Product B
        formula: 160 - 80 = 80
        output: 80
    
    STEP 5: SELECT(rows=["Product C"], cols=["2020","2022"])
        description: Get sales of Product C in 2020 and 2022
        formula: []
        output: {"2020":50,"2022":180}
    
    STEP 6: COMPUTE(expression="180 - 50 = 130")
        description: Compute sales growth of Product C
        formula: 180 - 50 = 130
        output: 130
    
    STEP 7: SORT(list_of_numbers=[100,80,130])
        description: Sort the growth values to find the maximum
        formula: [80,100,130]
        output: [80,100,130]
    
    STEP 8: INFER(inputs=[STEP_2,STEP_4,STEP_6])
        description: Identify which product corresponds to the maximum growth
        formula: []
        output: "Product C"
    
    STEP 9: ANSWER(value="Product C")
        description: Product C had the highest sales growth from 2020 to 2022
        formula: []
        output: "Product C"
"""


answer_prompt = """
Input:
    Question: {question}
    TQA guidance:{caption}
    Table: 
    {table}  

Strict commands:
    - If some of the information mentioned in the question is not covered in the table, then these pieces of information are not important and do not need attention.
    - If the question does not mention a specific header, choose the header indicating total, all, or entire region.
    - When the table explicitly lists totals or changes (such as increases, decreases, differences), directly use these values instead of performing additional calculations.
    - Prohibit unit conversion of table data or calculation results. Keep the original values.
    - When question is about percentage or proportion, there is no need for repeated percent calculations, table data is percent.
    - When answer is a percent, use format: "number%"
    - The answer should be as concise as possible using the original table content.

Start output(Do not output any other text):
    Step 1: SELECT(rows=..., cols=...)
    
    ...
    
    Step N: ANSWER(value=...)
"""

checker_system_instruction = """
You are an expert reasoning corrector for Table Question Answering (TQA).

You will receive:
    1.Question — the user query
    2.Table — Markdown table (first row = column headers, first column = row headers)
    3.Previous_Reasoning — a step-by-step pseudo-program using the specified STEP format
    4.Edit_Suggestions — natural-language comments describing what needs to be corrected.(formatting error|wrong sorting|arithmetic errors)

Reasoning format:
    Reason **step by step using a pseudo-program**, in which each step is a single action chosen from the following allowed actions:
        1. SELECT(rows=<row_list>, cols=<col_list>)  
           - Select rows, columns, or cells from the table  
           - formula: []  
           - output: number, list, or dictionary of selected values
        
        2. COMPUTE(expression="<arithmetic_expression> = <result>")  
        - Perform arithmetic computation in this step
        - The left side <arithmetic_expression> MUST be a **machine-parseable expression**
        - Allowed operators: +, -, *, /
        - MUST NOT contain text, commas, units, or explanations
        - Mixed operations ARE allowed:
            valid: "3 + 5 * 2 - 4 / 2 = 10"
            valid: "(12 - 2) * 3 / 2 = 15"
        - output MUST be the computed numeric result only
        
        3. SORT(list_of_numbers=[...])  
           - Sort a list of numbers in **strict ascending order**  
           - formula must be a pure list of numbers in ascending order  
           - output: the sorted list
        
        4. INFER(inputs=[STEP_n, STEP_m, ...])  
           - Perform logical inference or summarize previous steps  
           - Can be used anywhere in the chain  
           - formula: []  
           - output: intermediate conclusion or inferred value
        
        5. ANSWER(value=<final_value>)  
           - Output the final answer  
           - formula: []  
           - output: the final value only (no units, symbols, extra text)
    
    Each step must strictly follow this syntax:
        STEP <number>: <ACTION>(<arguments if any>)
            description: <one-line natural language explanation>
            formula: <list of formulas, [] if none>
            output: <result>
    
    Rules:
        1. Each step must perform **exactly one action**.  
        2. COMPUTE must contain exactly one equation of the form (no commas, units, or text): <arithmetic_expression> = <numeric_result> 
        3. SORT formula must contain strictly ascending numbers list: [num1,num2,num3...]  
        4. ANSWER output must be the final value only  
        5. The pseudo-program **describes reasoning**, not real code execution

Your task:
    You must verify and correct the given pseudo-program based on the Edit_Suggestions.

"""

checker_prompt = """
Input:
    1) Question: {question}
    2) Table: {caption}
    {table}
    3) Previous_Reasoning: 
    {reason}
    4) Edit_Suggestions: {amend}

Now Begin (Do not output any other text):
    Step 1: SELECT(rows=..., cols=...)
    
    ...
    
    Step N: ANSWER(value=...)
"""

PROMPT_TEMPLATE = {
    'Flatten_system_instruction': flatten_system_instruction,
    'Flatten_examples': flatten_examples,
    'Flatten_prompt': flatten_prompt,

    'understand_system_instruction': understand_system_instruction,
    'understand_examples': understand_examples,
    'understand_prompt': understand_prompt,

    'answer_system_instruction': answer_system_instruction,
    'answer_prompt': answer_prompt,

    'checker_system_instruction': checker_system_instruction,
    'checker_prompt': checker_prompt
}