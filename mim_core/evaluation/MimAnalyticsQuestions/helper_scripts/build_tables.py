'''
This file is part of Mim.
Mim is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
Mim is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Mim.
If not, see <https://www.gnu.org/licenses/>.
'''
"""
Build Tables for Paper

Author(s): Marko Sterbentz
September 21, 2022

Example Usage:

    python build_tables.py --gold ../data/analytic_data.json --preds ../results/answers_full_2.json --plans ../results/plans_full_2.json  --method Mim
    python build_tables.py --gold ../data/analytic_data.json --preds ../results/answers_irrr_1.json  --method IRRR

"""
import json
import argparse
import dateparser
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from datetime import timedelta
from typing import List, Tuple
from mim_core.evaluation.MimAnalyticsQuestions.score_answers import evaluate, evaluate_numeric, compute_full_em_score, f1_score_normalized, exact_match_score, exact_match_score_numeric, threshold_match_score

def map_question_id_to_question(original_questions: List[dict]) -> dict[dict]:
    return {q["id"]:q for q in original_questions}

def annotate_complex_classification(questions: dict[dict],
                                    plans: dict[str, dict]) -> None:
    """
    Adds a "classification" field to the questions dictionary that denotes whether the complex-simple classifier
    marked this question as simple or complex.
    :param questions:
    :param plans:
    :return:
    """

    for p in plans.values():
        question_id = p["plan"]["q_id"]
        if p["plan"]["steps"][0]["step_type"] == "simple":
            questions[question_id]["classification"] = "simple"
        else:
            questions[question_id]["classification"] = "complex"

    return None

def annotate_decomposition_quality(questions: dict[dict],
                                   plans: dict[str, dict]) -> None:
    """
    Adds a "decomposition" field to the questions dictionary that denotes the operators of the decomposition were for that question was.
    Adds a "decomposition_contains_expected_operator" field to the questions dictionary.
    :param questions:
    :param plans:
    :return:
    """

    for p in plans.values():

        question_id = p["plan"]["q_id"]

        questions[question_id]["decomposition"] = tuple([s["operator_type"] for s in p["plan"]["steps"]])

        question_operator = questions[question_id]["operator"]

        # Check that the complex question was actually decomposed
        if len(p["plan"]["steps"]) <= 1:
            questions[question_id]["no_decomposition"] = True
        else:
            questions[question_id]["no_decomposition"] = False

        # Check that the plan has an expected decomposition structure
        # Default this to false
        questions[question_id]["decomposition_contains_expected_operator"] = False
        questions[question_id]["decomposition_has_expected_structure"] = False

        # Get the plan graph of simple questions for the current plan
        p_simple_steps = tuple([s["operator_type"] for s in p["plan"]["steps"][1:]])

        if question_operator == "addition":
            addition_templates = [('select', 'select', 'aggregate', 'aggregate', 'arithmetic'),
                                  ('select', 'select', 'arithmetic')]
            if p_simple_steps in addition_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "arithmetic" and step["operator_args"][0] == "sum":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue
        elif question_operator == "boolean_equality":
            boolean_equality_templates = [('select', 'select', 'boolean'),
                                          ('boolean', 'boolean', 'boolean'),
                                          ('select', 'select', 'aggregate', 'aggregate', 'boolean')]
            if p_simple_steps in boolean_equality_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "boolean" and step["operator_subtype"] == "boolean-inequality":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue
        elif question_operator == "boolean_existence":
            boolean_existence_templates = [('select', 'boolean'),
                                           ('boolean', 'boolean', 'boolean'),
                                           ('select', 'select', 'boolean')]

            if p_simple_steps in boolean_existence_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "boolean":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue
        elif question_operator == "comparison":
            comparison_templates = [('select', 'select', 'aggregate', 'aggregate', 'comparison'),
                                    ('boolean', 'boolean', 'comparison'),
                                    ('select', 'select', 'comparison')]

            if p_simple_steps in comparison_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "comparison":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue
        elif question_operator == "discard":
            discard_templates = [('select', 'discard'),
                                 ('select', 'filter')]

            if p_simple_steps in discard_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "discard":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue
        elif question_operator == "intersection":
            intersection_templates = [('select', 'select', 'intersection'),
                                      ('select', 'filter')]

            if p_simple_steps in intersection_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "intersection":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue
        elif question_operator == "subtraction":
            subtraction_templates = [('select', 'select', 'aggregate', 'aggregate', 'arithmetic'),
                                     ('select', 'select', 'project', 'project', 'arithmetic'),
                                     ('select', 'select', 'arithmetic')]

            if p_simple_steps in subtraction_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "arithmetic" and step["operator_args"][0] == "difference":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue
        elif question_operator == "bridge":
            two_hop_bridge_templates = [('select', 'project')]
            three_hop_bridge_templates = [('select', 'project', 'project')]
            four_hop_bridge_templates = [('select', 'project', 'project', 'project')]

            num_hops = questions[question_id]["num_hops"]

            if num_hops == 2 and p_simple_steps in two_hop_bridge_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue
            if num_hops == 3 and p_simple_steps in three_hop_bridge_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue
            if num_hops == 4 and p_simple_steps in four_hop_bridge_templates:
                questions[question_id]["decomposition_has_expected_structure"] = True
                continue

            # for step in p["plan"]["steps"]:
            #     if step["operator_type"] == "project":
            #         questions[question_id]["decomposition_contains_expected_operator"] = True
            #         continue

    return None

def threshold_match_score_datetime(prediction, gold_answer, threshold):
    if prediction and gold_answer:
        a = (prediction - gold_answer)
        return -threshold <= a <= threshold
    else:
        return False

def get_best_metrics_for_datetime_question(prediction: str,
                                           gold: List[str],
                                           threshold: timedelta=timedelta(days=1)) -> Tuple[float, float]:
    best_em = 0.0
    best_threshold_match = 0.0

    # Parse the predicted datetime
    predicted_date = dateparser.parse(prediction)

    for gold_answer in gold:
        # Parse the gold datetime
        gold_date = dateparser.parse(gold_answer)

        # Check if the gold and prediction are exact matches
        em = (predicted_date == gold_date)

        # Compare the gold and prediction with the given timedelta for threshold match
        threshold_match = threshold_match_score_datetime(predicted_date, gold_date, threshold)
        if best_threshold_match < threshold_match:
            best_threshold_match = threshold_match
            best_em = em

    return best_em, best_threshold_match

def get_best_metrics_for_numeric_question(prediction: str,
                                          gold: List[str],
                                          threshold: float) -> Tuple[float, float]:
    best_threshold_match = 0.0
    best_em = 0.0
    for gold_answer in gold:
        em = exact_match_score_numeric(prediction, gold_answer)
        threshold_match = threshold_match_score(prediction, gold_answer, threshold)
        if best_threshold_match < threshold_match:
            best_threshold_match = threshold_match
            best_em = em

    return best_em, best_threshold_match

def get_best_metrics_for_single_question(prediction: str,
                                         gold: List[str]) -> Tuple[float, bool, float, float]:
    best_f1 = 0.0
    best_em = 0.0
    best_prec = 0.0
    best_recall = 0.0
    for gold_answer in gold:
        em = exact_match_score(prediction, gold_answer)
        f1, prec, recall = f1_score_normalized(prediction, gold_answer)
        if best_f1 < f1:
            best_f1 = f1
            best_em = em
            best_prec = prec
            best_recall = recall
    return best_f1, best_em, best_prec, best_recall

def annotate_correctness(questions: dict[dict],
                         # plans: dict[str, dict],
                         answers: dict[str, str],
                         threshold: float=0.01) -> None:
    """
    Adds a "is_correct" field to the questions dictionary that denotes whether the question was answered correctly.
    :param questions:
    :param plans:
    :return:
    """

    for q in questions.values():
        # mim_answer = plans[q["id"]]["answer"]
        mim_answer = answers[q["id"]]

        if q["operator"] in ["addition", "subtraction"]:
            em, threshold_match = get_best_metrics_for_numeric_question(mim_answer, q["answer"], threshold)
            if em or threshold_match:
                # Mark this question as correct
                q["is_correct"] = True
            else:
                # Mark this question as incorrect
                q["is_correct"] = False
        else:
            f1, em, prec, recall = get_best_metrics_for_single_question(mim_answer, q["answer"])
            if f1 > 0.8 or em:
                # Mark this question as correct
                q["is_correct"] = True
            else:
                # Mark this question as incorrect
                q["is_correct"] = False

    return None

def get_child_steps(step, plan):
    child_steps = []
    if step["child_steps"]:
        for child_ref_id in step["child_steps"]:
            for step in plan["steps"]:
                if child_ref_id == step["reference_id"]:
                    child_steps.append(step)
        return child_steps
    else:
        return []

def get_matching_step(plan: dict[str, str],
                      ent_support: str) -> dict[str, str]:

    def process_str(ent_support):
        import string
        out = ent_support.lower()
        out = out.translate(str.maketrans('', '', string.punctuation))
        return out

    simple_steps = [s for s in plan["steps"] if s["step_type"] == "simple"]
    for step in simple_steps:
        if process_str(ent_support) in process_str(step["qdmr"]):
            # If the child of this step is an aggregate step, return the child instead
            child_steps = get_child_steps(step, plan)
            for child in child_steps:
                if child["operator_type"] == "aggregate":
                    return child

            # Return the step with qdmr that contains the support entity string:
            return step

    return None

def annotate_retrieval_correctness(questions: dict[dict],
                                   plans: dict[str, dict],
                                   threshold: float = 0.01,
                                   date_threshold: timedelta=timedelta(days=1)) -> None:
    # TODO: Handle <threshold> values in "ent_support_n" column differently.
    ops_to_check = ['bridge', 'boolean_equality', 'comparison', 'addition', 'subtraction']
    questions_to_check = [q for q in questions.values() if q["operator"] in ops_to_check]
    step_not_found_count = 0
    for q in questions_to_check:
        q_id = q["id"]
        plan = plans[q_id]["plan"]

        # Set the default value to True
        q["is_retrieval_correct"] = True

        support_nums = ['1', '2', '3', '4']
        step_found = True
        for n in support_nums:
            # Make sure there is actually supporting evidence to check
            if not q["support_" + n]:
                break

            # Find the matching step, if it exists
            matching_step = get_matching_step(plan, q["ent_support_" + n])

            if matching_step:
                # Build the pandas dataframe for this step
                result_dict = json.loads(matching_step["result"])
                df = pd.DataFrame.from_dict(result_dict)

                # Ensure there is an answer for this step
                if len(df) < 1:
                    q["is_retrieval_correct"] = False
                    break

                # Check if the top predicted answer matches the expected answer for this step
                top_answer_pred = str(df["answer"].iloc[0])
                gold_answer = q["support_" + n]
                if q["type_support_" + n] == 'numeric':
                    em, threshold_match = get_best_metrics_for_numeric_question(top_answer_pred, [gold_answer], threshold)
                    if not (em or threshold_match):
                        q["is_retrieval_correct"] = False
                elif q["type_support_" + n] == 'date':
                    em, threshold_match = get_best_metrics_for_datetime_question(top_answer_pred, [gold_answer], date_threshold)
                    if not (em or threshold_match):
                        q["is_retrieval_correct"] = False
                else:  # (i.e. boolean, entity)
                    f1, em, prec, recall = get_best_metrics_for_single_question(top_answer_pred, [gold_answer])
                    if not (f1 > 0.8 or em):
                        q["is_retrieval_correct"] = False

                if not q["is_retrieval_correct"]:
                    # print(f"{matching_step['qdmr'].lower()} || PRED: {top_answer_pred} || GOLD: {gold_answer}")
                    break # Breaks out of the support_nums loop

            else:
                step_found = False
                # print("="*40)
                # print(f"No step found for support entity: {q['ent_support_' + n]}")
                # print(f"QUESTION: {q['question']}")
                # print(f"STEPS: {[step['qdmr'] for step in plan['steps'] if step['step_type'] == 'simple']}")


        # Keep track of the number of questions where no corresponding step was found
        if not step_found:
            step_not_found_count += 1

    print(f"Number of questions for which no corresponding step was found: {step_not_found_count}")

    return None

def annotate_plans(questions, plans, threshold=0.01) -> None:
    annotate_complex_classification(questions, plans)
    annotate_decomposition_quality(questions, plans)
    annotate_retrieval_correctness(questions, plans, threshold=threshold, date_threshold=timedelta(days=1))

def get_incorrect_complex_classification_percentage(questions) -> float:
    count = 0
    total = 0
    for q in questions.values():
        if not q['is_correct']:
            total += 1
            if q['classification'] == "simple":
                count += 1
    return 100.0 * count / total if total != 0 else 0

def get_incorrect_decomposition_percentage(questions) -> float:
    count = 0
    total = 0
    for q in questions.values():
        if not q['is_correct']:
            total += 1
            if not q['decomposition_has_expected_structure']:
                count += 1
    return 100.0 * count / total if total != 0 else 0

def get_incorrect_decomposition_count(questions, operator_type) -> Tuple[int, int]:
    count = 0
    total = 0
    for q in questions.values():
        if not q['is_correct'] and operator_type == q['operator']:
            total += 1
            if not q['decomposition_has_expected_structure']:
                count += 1
    return count, total

def get_no_decomposition_percentage(questions) -> float:
    count = 0
    total = 0
    for q in questions.values():
        if not q['is_correct']:
            total += 1
            if q['no_decomposition']:
                count += 1
    return 100.0 * count / total if total != 0 else 0

def get_retrieval_failure_count(questions, operator_type) -> Tuple[int, int]:
    count = 0
    total = 0
    for q in questions.values():
        if 'is_retrieval_correct' in q and operator_type == q['operator'] and not q['is_correct'] and q['decomposition_has_expected_structure'] and not q['no_decomposition']:
            total += 1
            if not q['is_retrieval_correct']:
                count += 1
    return count, total

def get_multihop_correct_percentage(questions, num_hops: int) -> float:
    count = 0
    total = 0
    for q in questions.values():
        if q['operator'] == 'bridge' and q['num_hops'] == num_hops:
            total += 1
            if q['is_correct']:
                count += 1

    return 100.0 * count / total if total != 0 else 0

def get_incorrect_count(questions) -> int:
    count = 0
    for q in questions.values():
        if not q['is_correct']:
            count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Produce the scores and metrics needed for the tables in the paper.')
    parser.add_argument('--gold')
    parser.add_argument('--preds')
    parser.add_argument('--plans', nargs='?', type=str, const=None)
    parser.add_argument('--method', nargs='?', type=str, const=None)
    args = parser.parse_args()

    #######################################################
    # Read in the predictions and gold answers
    #######################################################
    with open(args.preds) as f:
        answers = json.load(f)

    if 'answers' in answers:
        answers = answers['answers']
    res = {'answers': answers}

    with open(args.gold) as f:
        gold_data = json.load(f)
        questions = map_question_id_to_question(gold_data)

    # Build the separate list of gold data for the numeric vs. non-numeric questions
    numeric_gold_data = [dp for dp in gold_data if dp['operator'] in ['addition', 'subtraction']]
    other_gold_data = [dp for dp in gold_data if dp['operator'] not in ['addition', 'subtraction']]

    # Build the separate answer dictionaries for the numeric vs. non-numeric questions
    numeric_predictions = {"answers": {ex["id"]: res["answers"][ex["id"]] for ex in numeric_gold_data}}
    other_predictions = {"answers": {ex["id"]: res["answers"][ex["id"]] for ex in other_gold_data}}

    #######################################################
    # Produce the F1, EM scores for the non-numeric questions
    #######################################################
    non_numeric_metrics = evaluate(other_gold_data, other_predictions, False)

    #######################################################
    # Produce the 3 TM scores
    #######################################################
    numeric_metrics_one_percent = evaluate_numeric(numeric_gold_data, numeric_predictions, 0.01, False)
    numeric_metrics_five_percent = evaluate_numeric(numeric_gold_data, numeric_predictions, 0.05, False)
    numeric_metrics_ten_percent = evaluate_numeric(numeric_gold_data, numeric_predictions, 0.1, False)

    #######################################################
    # Produce the EM score for the full dataset
    #######################################################
    full_exact_match_metric = compute_full_em_score(non_numeric_metrics, numeric_metrics_one_percent, other_predictions, numeric_predictions)

    # Annotate the dataset questions with whether or not they were answered correctly
    annotate_correctness(questions, res["answers"], threshold=0.01)

    #######################################################
    # Build the general result tables
    #######################################################
    table_one = [['Method', 'EM (all operators)', 'F1 (non-numeric operators)', 'TM (t=0.01)', 'TM (t=0.05)', 'TM (t=0.1)'],
                 [args.method, full_exact_match_metric, non_numeric_metrics['f1'], numeric_metrics_one_percent['threshold_match'], numeric_metrics_five_percent['threshold_match'], numeric_metrics_ten_percent['threshold_match']]]
    print(tabulate(table_one, headers='firstrow', tablefmt='fancy_grid',floatfmt='.4f'))


    numeric_operators = ['addition', 'subtraction']
    non_numeric_operators = ['boolean_equality', 'boolean_existence', 'comparison', 'discard', 'intersection',  'bridge']
    table_two = [['Operation', 'Method', 'EM (all operators)', 'F1 (non-numeric operators)', 'TM (t=0.01)', 'TM (t=0.05)', 'TM (t=0.1)']]
    for op in numeric_operators:
        table_two.append([
            op, args.method, numeric_metrics_one_percent[op+'_em'], None, numeric_metrics_one_percent[op+'_threshold_match'], numeric_metrics_five_percent[op+'_threshold_match'], numeric_metrics_ten_percent[op+'_threshold_match']
        ])

    for op in non_numeric_operators:
        table_two.append([
            op, args.method, non_numeric_metrics[op+'_em'], non_numeric_metrics[op+'_f1'], None, None, None
        ])
    print(tabulate(table_two, headers='firstrow', tablefmt='fancy_grid', floatfmt='.4f'))

    # Table for information retrieval capabilities
    table_four = [['Method', '2-hop', '3-hop', '4-hop']]
    table_four.append([args.method, get_multihop_correct_percentage(questions, 2), get_multihop_correct_percentage(questions, 3), get_multihop_correct_percentage(questions, 4)])
    print(tabulate(table_four, headers='firstrow', tablefmt='fancy_grid', floatfmt='.2f'))

    #######################################################
    # Build the error analysis tables
    #######################################################
    if args.plans:
        # Read in the plans
        with open(args.plans) as f:
            plans = json.load(f)

        # Annotate the plans with data necessary for Error Analysis table
        annotate_plans(questions, plans)

        # Table for errors in question analysis
        table_three = [['Erroneous Behavior', '% of Incorrectly Answered Questions']]
        table_three.append(['Incorrect complex classification', get_incorrect_complex_classification_percentage(questions)])
        table_three.append(['No decomposition produced', get_no_decomposition_percentage(questions)])
        table_three.append(['Unexpected decomposition form', get_incorrect_decomposition_percentage(questions)])
        print(tabulate(table_three, headers='firstrow', tablefmt='fancy_grid', floatfmt='.2f'))

        # Table for breakdown how many incorrectly answered questions have the incorrect decomposition form
        table_six = [['Operator', 'Method', 'Number of incorrectly answered questions with unexpected decomposition form', 'Total number', 'Percentage']]
        addition_count, addition_total = get_incorrect_decomposition_count(questions, 'addition')
        table_six.append(['addition', args.method, addition_count, addition_total, 100.0 * addition_count / addition_total])
        subtraction_count, subtraction_total = get_incorrect_decomposition_count(questions, 'subtraction')
        table_six.append(['subtraction', args.method, subtraction_count, subtraction_total, 100.0 * subtraction_count / subtraction_total])
        boolean_equality_count, boolean_equality_total = get_incorrect_decomposition_count(questions, 'boolean_equality')
        table_six.append(['boolean_equality', args.method, boolean_equality_count, boolean_equality_total, 100.0 * boolean_equality_count / boolean_equality_total])
        boolean_existence_count, boolean_existence_total = get_incorrect_decomposition_count(questions, 'boolean_existence')
        table_six.append(['boolean_existence', args.method, boolean_existence_count, boolean_existence_total, 100.0 * boolean_existence_count / boolean_existence_total])
        comparison_count, comparison_total = get_incorrect_decomposition_count(questions, 'comparison')
        table_six.append(['comparison', args.method, comparison_count, comparison_total, 100.0 * comparison_count / comparison_total])
        discard_count, discard_total = get_incorrect_decomposition_count(questions, 'discard')
        table_six.append(['discard', args.method, discard_count, discard_total, 100.0 * discard_count / discard_total])
        intersection_count, intersection_total = get_incorrect_decomposition_count(questions, 'intersection')
        table_six.append(['intersection', args.method, intersection_count, intersection_total, 100.0 * intersection_count / intersection_total])
        bridge_count, bridge_total = get_incorrect_decomposition_count(questions, 'bridge')
        table_six.append(['bridge', args.method, bridge_count, bridge_total, 100.0 * bridge_count / bridge_total])
        print(tabulate(table_six, headers='firstrow', tablefmt='fancy_grid', floatfmt='.2f'))

        # Table for evidence retrieval failure
        table_five = [['Operator', 'Method',  'Number of incorrectly answered questions with expected decompositions with retrieval failure', 'Total number', 'Percentage']]
        addition_count, addition_total = get_retrieval_failure_count(questions, 'addition')
        table_five.append(['addition', args.method, addition_count, addition_total, 100.0 * addition_count / addition_total])
        subtraction_count, subtraction_total = get_retrieval_failure_count(questions, 'subtraction')
        table_five.append(['subtraction', args.method, subtraction_count, subtraction_total, 100.0 * subtraction_count / subtraction_total])
        boolean_equality_count, boolean_equality_total = get_retrieval_failure_count(questions, 'boolean_equality')
        table_five.append(['boolean_equality', args.method, boolean_equality_count, boolean_equality_total, 100.0 * boolean_equality_count / boolean_equality_total])
        comparison_count, comparison_total = get_retrieval_failure_count(questions, 'comparison')
        table_five.append(['comparison', args.method, comparison_count, comparison_total, 100.0 * comparison_count / comparison_total])
        bridge_count, bridge_total = get_retrieval_failure_count(questions, 'bridge')
        table_five.append(['bridge', args.method, bridge_count, bridge_total, 100.0 * bridge_count / bridge_total])
        # table_five.append(['Total', args.method, 100.0 * (addition_count + subtraction_count + boolean_equality_count + comparison_count + bridge_count) / (addition_total + subtraction_total + boolean_equality_total + comparison_total + bridge_total)])
        print(tabulate(table_five, headers='firstrow', tablefmt='fancy_grid', floatfmt='.2f'))

        print(f"Number of question incorrectly answered: {get_incorrect_count(questions)}")