import os
import re
import numpy as np
import pandas as pd
import sys
import textstat
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
import random
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
sbert = SentenceTransformer("all-mpnet-base-v2")

import warnings
from transformers import logging
from transformers import pipeline

logging.set_verbosity_error()

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from datetime import datetime
from openai import OpenAI
import time
try:
    from google.colab import userdata
except ImportError:
    userdata = None

stop_words = set(stopwords.words("english"))


class PromptGenerator:

  def __init__(self):
    pass


  def extract_ex(self, ex_dat, dl, text_l, n_ex, rationale_dat):

    # extract randomly selected examples
    # for the target difficulty level

    # input:
    # ex_dat = dataset with examples in JSON format
    # dl = target math difficulty level
    # text_l = text complexity level (e.g., low, medium, high)
    # n_ex = number of examples for the prompt template

    ex_dat_dl = [ex for ex in ex_dat if ex['DL'] == dl]
    ex_dat_dl_text_l = [ex for ex in ex_dat_dl if ex['FKGL_cat'] == text_l]
    tot_n_ex = len(ex_dat_dl_text_l)

    if (tot_n_ex > n_ex):
      ex_ind = random.sample(range(0, tot_n_ex), n_ex)
    else:
      ex_ind = range(0, tot_n_ex)

    ex_dat_samp = [ex_dat_dl_text_l[i] for i in ex_ind]

    if rationale_dat:
      examples = "Here are some examples that include problem (Q), answer (A), formula (F), and rationale (R):\n"
      j = 0
      for i, ex in enumerate(ex_dat_samp, start = 1):
        examples += f"\nExample {j}:\nQ: {ex['Q']}\nA: {ex['A']}\nF: {ex['F']}\nR: {ex['R']}\n"
    else:
      examples = "Here are some examples that include problem (Q), answer (A), and formula (F):\n"
      j = 0
      for i, ex in enumerate(ex_dat_samp, start = 1):
        examples += f"\nExample {j}:\nQ: {ex['Question2']}\nA: {ex['Answer2']}\nF: {ex['Formula2']}\n"

    return examples


  def extract_standard_statement(self, standard_dat, dl):

    # extract standard statement
    # for the target difficulty level

    # input:
    # standard_dat = includes information about standards in JSON format
    # dl = target difficulty level

    return standard_dat[str(dl)]['Extended Standards']


  def join_with_or(self, items):

    # join the list elements
    # separated by , and
    # last item by ", and"

    if not items:
        out = ""
    elif len(items) == 1:
        out = str(items[0])
    elif len(items) == 2:
        out = f"{items[0]} or {items[1]}"
    else:
        # join all but the last element with the , separator
        all_but_last = ", ".join(map(str, items[:-1]))
        # combine this string with the last element using " and "
        out = f"{all_but_last}, or {items[-1]}"

    return out


  def extract_operators(self, standard_dat, dl):

    # extract operators required to generate questions
    # for the target difficulty level

    # input:
    # standard_dat = includes information about standards in JSON format
    # dl = target difficulty level

    dl_standards_info = standard_dat[str(dl)]

    operators = []
    omit_operators = []

    if dl_standards_info['Addition'] == 'Y':
      operators.append("addition")
    else:
      omit_operators.append("addition")

    if dl_standards_info['Subtraction'] == 'Y':
      operators.append("subtraction")
    else:
      omit_operators.append("subtraction")

    if dl_standards_info['Multiplication'] == 'Y':
      operators.append("multiplication")
    else:
      omit_operators.append("multiplication")

    if dl_standards_info['Division'] == 'Y':
      operators.append("division")
    else:
      omit_operators.append("division")

    return operators, omit_operators


  def extract_max_num_digit(self, standard_dat, dl):

    # extract maximum number digit required
    # for the target difficulty level

    # input:
    # standard_dat = includes information about standards in JSON format
    # dl = target difficulty level

    dl_standards_info = standard_dat[str(dl)]

    max_num_digit = ""

    if dl_standards_info['Three Digit'] == 'Y':
      max_num_digit = "three digit"
    elif dl_standards_info['Two Digit'] == 'Y':
      max_num_digit = "two digit"
    elif dl_standards_info['One Digit'] == 'Y':
      max_num_digit = "single digit"

    return max_num_digit


  def extract_n_step(self, standard_dat, dl):

    # extract number of steps required
    # for the target difficulty level

    # input:
    # standard_dat = includes information about standards in JSON format
    # dl = target difficulty level

    dl_standards_info = standard_dat[str(dl)]

    n_step = ""
    if dl_standards_info['# of Steps'] == 1:
      n_step = "single"
    elif dl_standards_info['# of Steps'] == '>=2':
      n_step = "at least two"

    return n_step


  def extract_must_operator(self, dl_operator):

    if len(dl_operator) == 1:
      must_operator = dl_operator[0]
    elif "multiplication" in dl_operator:
      must_operator = "multiplication or division"
    else:
      must_operator = "addition or subtraction"

    return must_operator


  def generate_prompt_template(self, ex_dat, standard_dat, dl, text_l, n_ex, n_items, rationale_dat):

    dl_examples = self.extract_ex(ex_dat, dl, text_l, n_ex, rationale_dat)

    dl_statement = self.extract_standard_statement(standard_dat, dl)
    dl_operators, dl_omit_operators = self.extract_operators(standard_dat, dl)
    dl_must_operators = self.extract_must_operator(dl_operators)
    dl_max_num_digit = self.extract_max_num_digit(standard_dat, dl)
    dl_n_step = self.extract_n_step(standard_dat, dl)

    # Difficulty-level guideline (math structure guidance)
    if dl == 2:
        dl_guideline = (
            "The formula and answer must involve one one-digit integer between 1 and 9 inclusive and one two-digit integer between 10 and 99 inclusive."
        )
    elif dl == 3:
        dl_guideline = (
            "The formula and answer must involve one one-digit integer between 1 and 9 inclusive, and the formula can involve more than two steps to solve the problem."
        )
    elif dl == 4:
        dl_guideline = (
            "The formula and answer must involve one integer between 1 and 99 inclusive."
        )
    elif dl == 5:
        dl_guideline = (
            "A three-digit integer between 100 and 999 inclusive must apprear only once in the problem. The formula can involve more than two steps to solve the problem."
        )
    elif dl == 6:
        dl_guideline = (
            "The correct answer must be a two-digit integer between 10 and 99 inclusive."
        )
    elif dl == 8:
        dl_guideline = (
            "The problem must involve one one-digit (between 1 and 9 inclusive) and one two-digit (between 10 and 99 inclusive) integers. And, the correct answer must be a three-digit integer between 100 and 999 inclusive."
        )
    elif dl == 9:
        dl_guideline = (
            "The problem must involve only one three-digit integer between 100 and 999 inclusive. One of its quotient and divisor must be a one-digit integer between 1 and 9 inclusive."
        )
    elif dl == 10:
        dl_guideline = (
            "The correct answer must be less than 100. The formula must include one + or - and one * or /. Note that multiplication must not be between two two-digit integers."
        )
    elif dl == 11:
        dl_guideline = (
            "The problem and its answer must involve only one three-digit integer between 100 and 999 inclusive. The formula must include one + or - and one * or /. The formula must include a three-digit number between 100 and 999 inclusive. Note that multiplication must not be between two two-digit integers. Also, when division occurs, one of its quotient and divisor must be a one-digit integer between 1 and 9 inclusive."
        )
    else:
        dl_guideline = ""

    # text complexity guideline
    if text_l == "low":
        text_complexity_guideline = (
            f"In the generated problems, the number of words per sentence should be less than 8."
        )
    elif text_l == "medium":
        text_complexity_guideline = (
            f"In the generated problems, the number of words per sentence should be between 8 and 14."
        )
    else:  # high
        text_complexity_guideline = (
            f"In the generated problems, the number of words per sentence should be greater than 14."
        )

    prompt_template = f"""
You are a professional math tutor who can create Grade 3 math word problems aligned with the standard of difficulty level {dl}, which states that students can '{dl_statement}'

Each generated item must follow these rules:
Math/Logic
1. The problem can involve the following operators: {self.join_with_or(dl_operators)}.
2. The problem should involve at least one of the following operators: {dl_must_operators}.
3. The problem should not involve any of the following operations: {self.join_with_or(dl_omit_operators)}.
4. The problem should involve at least one number in {dl_max_num_digit} in its formula or in its answer.
5. The problem should not involve any number exceeding {dl_max_num_digit} in its formula and answer.
6. Use exactly {dl_n_step} operators in the solution formula.
7. Numbers are integers only (no decimals).
8. The question must contain exactly one question mark (one question only).
9. The answer should not involve any number exceeding {dl_max_num_digit}.
10. The answer must not be leaked inside the question story text.
11. Provide the correct answer and formula. {dl_guideline}

Readability/Text-Complexity
12. Keep language clear and the wording age-appropriate for Grade 3 students.
13. Avoid overused tropes (e.g., apples, stickers, chairs) and avoid contexts already seen in the examples.
14. {text_complexity_guideline}

Structure/Output
15. The generated problems are associated with unique key values ranging from 0.
16. For a separate problem, generate valid JSON (single object) with exactly the following keys and nothing else: "Q" (Word Problem), "A" (Answer), "F" (Formula), "R" (Rationale).
17. JSON rules: double quotes for all keys/strings, no trailing commas.
18. "F" must be a single line arithmetic expression that gives the answer "A" when solved.
19. "R" is a brief step-by-step explanation that matches "Q" and derives "A" from "F".
20. For "R", you need to read the problem carefully, restate key facts and numbers from the problem, explain your reasoning step by step, show calculations where needed, and write the final answer clearly and concisely. Remember your rationale should align with the "F" formula and gives the "A" value as the final answer. The rationale should include "Key facts" and "Calculation" only. The rationale must be enclosed in double quotes.

{dl_examples}

Now, generate {str(n_items)} math problems.
The generated problems should have different numbers, person names, objects, settings, and scenarios. Also, verbs, adjectives, and descriptive language should be varied in the problems.
And, within each problem, encourage imagination (e.g., everyday life, nature, hobbies, school events, community places, etc.). Do not repeat the same words, names, numbers, and scenarios across the problems.
Remember that the problems should not use stereotypical scenarios. Also, note that {text_complexity_guideline}
After you create the problems, check it against the rules specified above to make sure that the output meets all the requirement.
If any check fails, repeat the process silently, then emit only the final JSON format output without extra commentary.
All keys should be enclosed in double quotes.
Values for the "Q", "F", and "R" keys should be string, and values for the "A" should be an integer.
JSON key-value pairs are separated by comma.
You should stick with the following output format: {{
"0": {{ "Q": "", "R": "", "F": "", "A": "" }},
"1": {{ "Q": "", "R": "", "F": "", "A": "" }},
"2": {{ "Q": "", "R": "", "F": "", "A": "" }},
"3": {{ "Q": "", "R": "", "F": "", "A": "" }},
"4": {{ "Q": "", "R": "", "F": "", "A": "" }},
"5": {{ "Q": "", "R": "", "F": "", "A": "" }},
"6": {{ "Q": "", "R": "", "F": "", "A": "" }},
"7": {{ "Q": "", "R": "", "F": "", "A": "" }},
"8": {{ "Q": "", "R": "", "F": "", "A": "" }},
"9": {{ "Q": "", "R": "", "F": "", "A": "" }}
}}
Also, make sure that your problem is clear to understand, that your rationale and formula is correct, and that your formula gives the final answer.
You are good at this.
"""

    return prompt_template


class ProblemEvaluator:

  def __init__(self, api_key=None, target_dl=None, target_text_l=None, problem_db=None):
    
    if api_key is None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        self.api_key = api_key
    self.client = OpenAI()

    self.target_dl = target_dl
    self.target_text_l = target_text_l
    self.problem_db = problem_db
    self.passfail_res = []


  def evaluate_one_item(self, ex_dat, standards_dat, n_examples, problem_cand, key, model_option):

    gen_question = problem_cand[key]
    gen_question = self.llm_as_judge(gen_question, model_option)

    if (gen_question != "incorrect"):
      item_data = {
          'difficulty_level': self.target_dl,
          'text_complexity': self.target_text_l,
          'problem': gen_question}

      self.problem_db.append(item_data)

    return gen_question


  def llm_as_judge(self, gen_question, model_option):

    match_in_f = False
    while (match_in_f == False):
      eval_math_logic = self.evaluate_math_logic(gen_question, model_option)
      eval_math_logic = json.loads(eval_math_logic)

      if (eval_math_logic['math_correctness'] == "incorrect"):
        break
        return "incorrect"

      else:
        try:
          match_in_f = (gen_question['A'] == eval_math_logic['correct_answer']) and (eval(gen_question['F']) == gen_question['A'])
        except Exception:
          match_in_f = False
          continue

        gen_question['R'] = eval_math_logic['explanation']

    return gen_question


  def evaluate_math_logic(self, gen_question, model_option):

    messages = [
        {"role": "system", "content": "You are an expert in evaluating math word problems about their accuracy in mathematical logic."},
        {"role": "user",
         "content": self.gen_eval_prompt(gen_question)
        }
    ]

    response = self.client.chat.completions.create(
        model=model_option,
        messages=messages,
        temperature=0.9,
        top_p=0.95,
        max_tokens=1500
    )

    eval_output = response.choices[0].message.content

    return eval_output


  def gen_eval_prompt(self, gen_question):

    eval_prompt = f"""
      You are an expert in evaluating math word problems regarding their accuracy in mathematical logic.

      Evaluate the math word problem regarding its mathematical logic.
      Carefully read through the "R" rationale and evaluate whether it is a right way to solve the problem.
      Evaluate whether the "F" formula aligns with the "R" key. Also, solving the "F" formula should give the "A" answer.

      Here is the math word problem: {gen_question}

      Please Generate valid JSON (single object) with exactly the following keys and nothing else: "math_correctness", "explanation", "correct_answer", and "correct_formula".
      For the "math_correctness" key, there are only two options for the value, "correct" or "incorrect".
      For the "explanation" key, restate key facts and numbers from the problem, explain your reasoning step by step, show calculations where needed, and write the final answer clearly and concisely. The explanation must be enclosed in double quotes.
      For the "correct_formula" key, its value should provide a symbol-free single line "arithmetic" expression based on your explanation. It must be an arithmetic expression only without any words, enclosed in double quotes.
      For the "correct_answer" key, its valude provide only an answer using the correct_formula. It has to be an integer.

      Here is the output format: {{"math_correctness": <correct or incorrect>, "explanation": <step-by-step reasoning>, "correct_formula": <math formula>, "correct_answer": <final integer answer>}}

      The JSON key-value pairs should be separated by comma.
      Remember you do not say the problem is incorrect in its mathematical logic when it is indeed correct.
    """

    return eval_prompt


  def evaluate_alignment(self, gen_question, item_bank):

    gen_text_dl = self.text_complexity_map(gen_question)
    gen_math_dl = self.difficulty_map(self.extract_features(gen_question, self.target_dl))

    if (len(item_bank) > 0):
      gen_novelty = [self.evaluate_novelty(gen_question['Q'], item['problem']['Q']) for item in item_bank]
    else:
      gen_novelty = [True]

    alignment_passed = (gen_text_dl == self.target_text_dl) and (gen_math_dl == self.target_dl) and all(gen_novelty)

    return alignment_passed


  def difficulty_map(self, gen_question):

    dl = 0

    if (gen_question['multiplication'] == "Y") or (gen_question['division'] == "Y"):
      if (gen_question['three_digit'] == "Y"):
        if (gen_question['n_steps'] == 1):
          if (gen_question['multiplication'] == "Y") and (gen_question['division'] == "N"):
            dl = 8
          elif (gen_question['multiplication'] == "N") and (gen_question['division'] == "Y"):
            dl = 9
        elif (gen_question['n_steps'] >= 2):
          dl = 11
      elif (gen_question['three_digit'] == "N"):
        if (gen_question['n_steps'] == 1):
          if (gen_question['multiplication'] == "Y") and (gen_question['division'] == "N"):
            dl = 6
          elif (gen_question['multiplication'] == "N") and (gen_question['division'] == "Y"):
            dl = 7
        elif (gen_question['n_steps'] >= 2):
          dl = 10
    else:
      if (gen_question['three_digit'] == "Y"):
        if (gen_question['n_steps'] == 1):
          dl = 4
        elif (gen_question['n_steps'] >= 2):
          dl = 5
      elif (gen_question['three_digit'] == "N"):
        if (gen_question['n_steps'] == 1):
          if (gen_question['two_digit'] == "Y"):
            dl = 2
          else:
            dl = 1
        elif (gen_question['n_steps'] >= 2):
          dl = 3

    return dl


  def extract_features(self, gen_question):

    question = gen_question['Q']
    answer = gen_question['A']
    formula = gen_question['F']
    rationale = gen_question['R']

    # check operators
    addition = "Y" if "+" in formula else "N"
    subtraction = "Y" if "-" in formula else "N"
    multiplication = "Y" if "*" in formula else "N"
    division = "Y" if "/" in formula else "N"

    # check number of steps
    addition_n = formula.count("+")
    subtraction_n = formula.count("-")
    multiplication_n = formula.count("*")
    division_n = formula.count("/")
    n_steps = addition_n + subtraction_n + multiplication_n + division_n

    # check number size
    numbers = re.findall(r'\d+', formula)
    numbers.append(str(answer))

    single_digit = "Y" if any(len(n) == 1 for n in numbers) else "N"
    two_digit = "Y" if any(len(n) == 2 for n in numbers) else "N"
    three_digit = "Y" if any(len(n) == 3 for n in numbers) else "N"

    new_data = {
        'Q': question,
        'A': answer,
        'F': formula,
        'R': rationale,
        'addition': addition,
        'subtraction': subtraction,
        'multiplication': multiplication,
        'division': division,
        'addition_n': addition_n,
        'subtraction_n': subtraction_n,
        'multiplication_n': multiplication_n,
        'division_n': division_n,
        'n_steps': n_steps,
        'single_digit': single_digit,
        'two_digit': two_digit,
        'three_digit': three_digit,
        'target_DL': self.target_dl
    }

    return new_data


  def text_complexity_map(self, gen_question):

    # input variables
    # gen_question: dictionary with Q, F, A, and R keys

    FKGL = textstat.flesch_kincaid_grade(gen_question['Q'])

    if (FKGL < 2.5):
      text_comp_l = "low"
    elif (FKGL < 6.5):
      text_comp_l = "medium"
    else:
      text_comp_l = "high"

    return text_comp_l


  def jaccard_similarity(self, set1, set2):
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0


  def extract_context_terms(self, text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [w.lower() for w, pos in tagged if pos.startswith("NN")]
    nouns = [w for w in nouns if w.isalpha() and w not in stop_words]
    return set(nouns)


  def cosine_similarity(self, text1, text2):
    v1 = sbert.encode([text1], normalize_embeddings=True)
    v2 = sbert.encode([text2], normalize_embeddings=True)
    return float(np.dot(v1[0], v2[0]))


  def evaluate_novelty(self, ref, hyp):

    jaccard = self.jaccard_similarity(self.extract_context_terms(ref), self.extract_context_terms(hyp))
    cosine = self.cosine_similarity(ref, hyp)

    return (jaccard < 0.5 and cosine < 0.7)


  def check_text_complexity(self, gen_question):

    # input variables
    # gen_question: dictionary with Q, F, A, and R keys

    FKGL = textstat.flesch_kincaid_grade(gen_question['Q'])

    if (FKGL < 2.5):
      text_comp_l = "low"
    elif (FKGL < 5.5):
      text_comp_l = "medium"
    else:
      text_comp_l = "high"

    return {
        'check': 'text_complexity',
        'passed': (self.target_text_l == text_comp_l),
        'target_level': self.target_text_l,
        'actual_level': text_comp_l
        }


  def check_math_difficulty(self, gen_question):

    features = self.extract_features(gen_question)
    gen_math_dl = self.difficulty_map(features)

    return {
        'check': 'math_difficulty',
        'passed': (self.target_dl == gen_math_dl),
        'target_level': self.target_dl,
        'actual_level': gen_math_dl
        }


  def check_math_logic(self, gen_question, model_option):

    eval_math_logic = self.evaluate_math_logic(gen_question, model_option)
    eval_math_logic = json.loads(eval_math_logic)

    if (eval_math_logic['math_correctness'] == "incorrect"):
      match_in_f = False

    else:
      try:
        match_in_f = (gen_question['A'] == eval_math_logic['correct_answer']) and (eval(gen_question['F']) == gen_question['A'])
      except Exception:
        match_in_f = False
        gen_question['R'] = eval_math_logic['explanation']

    return {
        'check': 'math_logic',
        'passed': match_in_f
        }


  def check_novelty(self, gen_question):

    if (len(self.problem_db) > 0):
      gen_novelty = [self.evaluate_novelty(gen_question['Q'], item['Q']) for item in self.problem_db]
    else:
      gen_novelty = [True]

    return {
        'check': 'novelty',
        'passed': all(gen_novelty)
        }


  def check_fairness(self, gen_question):

    fairness_llm_result = self.fairness_llm(gen_question)
    fairness_roberta_result = self.fairness_roberta(gen_question)

    fairness_passed = ((fairness_llm_result == "No") and (fairness_roberta_result == False))

    return {
        'check': 'fairness',
        'passed': fairness_passed
        }


  def fairness_llm(self, gen_quesion):

    client = OpenAI()

    problem = "Your math word problem here"

    messages = [{
            "role": "user",
            "content": f"""Analyze this math word problem for potential fairness issues:

    {gen_quesion}

    Check for:
    1. Stereotypes (gender, racial, cultural, occupational)
    2. Cultural assumptions or bias
    3. Socioeconomic assumptions
    4. Representation and inclusivity
    5. Accessibility of language

    Aggregate your response by providing "Yes" if your response is "Yes" for any of the five categories. Otherwise, provide "No"."""
    }]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=500,
        messages=messages
    )

    fairness_issue = response.choices[0].message.content

    return fairness_issue


  def fairness_roberta(self, gen_question):

    classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", top_k = None)

    results = classifier(gen_question)

    flag = False

    for result in results[0]:
      if result['score'] > 0.1:
        flag = True
        break

    return flag


  def parallel_evaluation(self, gen_questions):

    #gen_questions = json.loads(gen_question)
    keys = list(gen_questions.keys())
    new_qualified_problems = []

    i = 0
    while (len(self.problem_db) < 5) and (i < len(keys)):

      key = keys[i]
      gen_q = gen_questions[key]
      result_q = {
          'problem': gen_q,
          'checks': {},
          'all_passed': True,
      }

      # use ThreadPoolExecutor to run checks in parallel
      with ThreadPoolExecutor(max_workers=5) as executor:
          # submit all tasks
          future_to_check = {
              executor.submit(self.check_math_difficulty, gen_q): 'math_difficulty',
              executor.submit(self.check_text_complexity, gen_q): 'text_complexity',
              executor.submit(self.check_math_logic, gen_q, "gpt-4.1-mini"): 'math_logic',
              executor.submit(self.check_novelty, gen_q): 'novelty',
              executor.submit(self.check_fairness, gen_q['Q']): 'fairness'
          }

          # collect results as they complete
          for future in as_completed(future_to_check):
              check_name = future_to_check[future]
              result = future.result()
              result_q['checks'][check_name] = result

              if not result.get('passed', False):
                  result_q['all_passed'] = False

      self.passfail_res.append(result_q)
      if result_q['all_passed']:
        self.problem_db.append(gen_q)
        new_qualified_problems.append(gen_q)

      i += 1

    return new_qualified_problems


class MWPGenerator:

  def __init__(self, api_key=None, target_dl=None, target_text_l=None, curr_session_data=None):
    
    if api_key is None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        self.api_key = api_key
    self.client = OpenAI()

    self.target_dl = target_dl
    self.target_text_l = target_text_l
    self.curr_session_data = curr_session_data
    self.problem_db = []
    self.n_corrects = 0
    self.evaluator = ProblemEvaluator(api_key=self.api_key, target_dl=self.target_dl, target_text_l=self.target_text_l, problem_db=self.problem_db)



  def generate_item(self, ex_dat, standards_dat, n_examples, n_items, rationale_dat, model_option):

    # uses different prompt generating function for different text_level
    promptgen_obj = PromptGenerator()
    prompt = promptgen_obj.generate_prompt_template(ex_dat, standards_dat, self.target_dl, self.target_text_l, n_examples, n_items, rationale_dat)
    problem_cand = self.generate_openai(prompt, model_option)
    problem_cand = json.loads(problem_cand)

    #gen_question = {key: problem_cand[key] for key in problem_cand if self.evaluator.evaluate_alignment(problem_cand[key], dl, text_l, self.evaluator.problem_db)}
    #gen_question = dict(islice(gen_question.items(), 5))

    return problem_cand


  def generate_openai(self, prompt, model_option):

    messages = [
        {"role": "system", "content": "You are an excellent, professional, and creative math problem writer generating Grade 3 math word problems with solutions."},
        {"role": "user",
         "content": prompt
        }
    ]

    response = self.client.chat.completions.create(
        model=model_option,
        messages=messages,
        temperature=0.9,
        top_p=0.95,
        max_tokens=2500
    )

    problem_cand = response.choices[0].message.content

    return problem_cand


  


class MWPSession:

  def __init__(self, api_key=None, user_id=None, output_dir=None):
    self.user_id = user_id
    self.output_dir = Path(output_dir)
    self.user_dir = self.output_dir / user_id
    self.user_dir.mkdir(parents=True, exist_ok=True) # make user directory
    if api_key is None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        if isinstance(api_key, tuple):
            self.api_key = api_key[0]  
        else:
            self.api_key = str(api_key)
                
    self.api_key = api_key,
    self.session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    self.session_data = self.load_data()
    self.prev_session_id = "" if len(self.session_data) == 0 else list(self.session_data.keys())[-1]
    self.curr_dl = 6 if self.prev_session_id=="" else self.session_data[self.prev_session_id]['final_dl']
    self.curr_text_l = 'medium' if self.prev_session_id=="" else self.session_data[self.prev_session_id]['final_text_l']
    self.curr_session_data = []
    self.generator = MWPGenerator(api_key=self.api_key, target_dl=self.curr_dl, target_text_l=self.curr_text_l)



  def save_data(self, filename=None):

    if filename is None:
      filename = f"{self.user_id}_mwp_bank.json"

    file_path = self.user_dir / filename

    self.session_data[self.session_id] = {
        'user_id': self.user_id,
        'session_id': self.session_id,
        'total_problems': len(self.curr_session_data),
        'final_dl':self.curr_dl,
        'final_text_l':self.curr_text_l,
        'problems': self.curr_session_data}

    with open(file_path, 'w') as f:
        json.dump(self.session_data, f, indent=2)


  def load_data(self):

    filename = f"{self.user_id}_mwp_bank.json"

    file_path = self.user_dir / filename

    if file_path.exists():
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    else:
        return {}


