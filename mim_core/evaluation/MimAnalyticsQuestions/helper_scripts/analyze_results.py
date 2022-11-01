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
Analyze Results

Usage: python analyze_results.py ../data/analytic_data.json ../results/plans.json

Author(s): Marko Sterbentz
September 3, 2022
"""

import json
import plac
from typing import List, Tuple
from mim_core.evaluation.MimAnalyticsQuestions.score_answers import f1_score_normalized, exact_match_score, exact_match_score_numeric, threshold_match_score

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

        # Default this to false
        questions[question_id]["decomposition_contains_expected_operator"] = False

        if question_operator == "addition":
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "arithmetic" and step["operator_args"][0] == "sum":
                    questions[question_id]["decomposition_contains_expected_operator"] = True
                    continue
        elif question_operator == "boolean_equality":
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "boolean" and step["operator_subtype"] == "boolean-inequality":
                    questions[question_id]["decomposition_contains_expected_operator"] = True
                    continue
        elif question_operator == "boolean_existence":
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "boolean":
                    questions[question_id]["decomposition_contains_expected_operator"] = True
                    continue
        elif question_operator == "comparison":
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "comparison":
                    questions[question_id]["decomposition_contains_expected_operator"] = True
                    continue
        elif question_operator == "discard":
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "discard":
                    questions[question_id]["decomposition_contains_expected_operator"] = True
                    continue
        elif question_operator == "intersection":
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "intersection":
                    questions[question_id]["decomposition_contains_expected_operator"] = True
                    continue
        elif question_operator == "subtraction":
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "arithmetic" and step["operator_args"][0] == "difference":
                    questions[question_id]["decomposition_contains_expected_operator"] = True
                    continue

    return None

def annotate_correctness(questions: dict[dict],
                         plans: dict[str, dict],
                         threshold: float=0.05) -> None:
    """
    Adds a "is_correct" field to the questions dictionary that denotes whether the question was answered correctly.
    :param questions:
    :param plans:
    :return:
    """

    for q in questions.values():
        mim_answer = plans[q["id"]]["answer"]

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

def output_complex_classification_results(questions: dict[dict],
                                          plans: dict[str, dict]) -> None:
    simple_dict = {
        "total_count": 0,
        "addition_count": 0,
        "boolean_equality_count": 0,
        "boolean_existence_count": 0,
        "comparison_count": 0,
        "discard_count": 0,
        "intersection_count": 0,
        "subtraction_count": 0,
        "addition_questions": [],
        "boolean_equality_questions": [],
        "boolean_existence_questions": [],
        "comparison_questions": [],
        "discard_questions": [],
        "intersection_questions": [],
        "subtraction_questions": []
    }

    complex_dict = {
        "total_count": 0,
        "addition_count": 0,
        "boolean_equality_count": 0,
        "boolean_existence_count": 0,
        "comparison_count": 0,
        "discard_count": 0,
        "intersection_count": 0,
        "subtraction_count": 0,
        "addition_questions": [],
        "boolean_equality_questions": [],
        "boolean_existence_questions": [],
        "comparison_questions": [],
        "discard_questions": [],
        "intersection_questions": [],
        "subtraction_questions": []
    }
    for p in plans.values():
        question_id = p["plan"]["q_id"]
        # if len(p["plan"]["steps"]) <= 1:
        if p["plan"]["steps"][0]["step_type"] == "simple":
            simple_dict["total_count"] += 1
            if questions[question_id]["operator"] == "addition":
                simple_dict["addition_questions"].append(questions[question_id]["question"])
                simple_dict["addition_count"] += 1
            elif questions[question_id]["operator"] == "boolean_equality":
                simple_dict["boolean_equality_questions"].append(questions[question_id]["question"])
                simple_dict["boolean_equality_count"] += 1
            elif questions[question_id]["operator"] == "boolean_existence":
                simple_dict["boolean_existence_questions"].append(questions[question_id]["question"])
                simple_dict["boolean_existence_count"] += 1
            elif questions[question_id]["operator"] == "comparison":
                simple_dict["comparison_questions"].append(questions[question_id]["question"])
                simple_dict["comparison_count"] += 1
            elif questions[question_id]["operator"] == "discard":
                simple_dict["discard_questions"].append(questions[question_id]["question"])
                simple_dict["discard_count"] += 1
            elif questions[question_id]["operator"] == "intersection":
                simple_dict["intersection_questions"].append(questions[question_id]["question"])
                simple_dict["intersection_count"] += 1
            elif questions[question_id]["operator"] == "subtraction":
                simple_dict["subtraction_questions"].append(questions[question_id]["question"])
                simple_dict["subtraction_count"] += 1

        else:
            complex_dict["total_count"] += 1
            if questions[question_id]["operator"] == "addition":
                complex_dict["addition_questions"].append(questions[question_id]["question"])
                complex_dict["addition_count"] += 1
            elif questions[question_id]["operator"] == "boolean_equality":
                complex_dict["boolean_equality_questions"].append(questions[question_id]["question"])
                complex_dict["boolean_equality_count"] += 1
            elif questions[question_id]["operator"] == "boolean_existence":
                complex_dict["boolean_existence_questions"].append(questions[question_id]["question"])
                complex_dict["boolean_existence_count"] += 1
            elif questions[question_id]["operator"] == "comparison":
                complex_dict["comparison_questions"].append(questions[question_id]["question"])
                complex_dict["comparison_count"] += 1
            elif questions[question_id]["operator"] == "discard":
                complex_dict["discard_questions"].append(questions[question_id]["question"])
                complex_dict["discard_count"] += 1
            elif questions[question_id]["operator"] == "intersection":
                complex_dict["intersection_questions"].append(questions[question_id]["question"])
                complex_dict["intersection_count"] += 1
            elif questions[question_id]["operator"] == "subtraction":
                complex_dict["subtraction_questions"].append(questions[question_id]["question"])
                complex_dict["subtraction_count"] += 1

    # Print the results
    print("=" * 40)
    print("Simple Questions")
    print(json.dumps(simple_dict, indent=4))
    print("=" * 40)
    print("Complex Questions")
    print(json.dumps(complex_dict, indent=4))
    print("=" * 40)

    return None


def output_decomposition_quality_results(questions: dict[dict],
                                         plans: dict[str, dict]) -> None:
    """
    Check if the generated plans have the expected analytic step.
    :param questions:
    :param plans:
    :return:
    """

    results = {
        "addition_correct": 0,
        "addition_total": 0,
        "addition_decomps": {},
        "boolean_equality_correct": 0,
        "boolean_equality_total": 0,
        "boolean_equality_decomps": {},
        "boolean_existence_correct": 0,
        "boolean_existence_total": 0,
        "boolean_existence_decomps": {},
        "bridge_correct": 0,
        "bridge_total": 0,
        "bridge_decomps": {},
        "comparison_correct": 0,
        "comparison_total": 0,
        "comparison_decomps": {},
        "discard_correct": 0,
        "discard_total": 0,
        "discard_decomps": {},
        "intersection_correct": 0,
        "intersection_total": 0,
        "intersection_decomps": {},
        "subtraction_correct": 0,
        "subtraction_total": 0,
        "subtraction_decomps": {}
    }
    for p in plans.values():
        question_id = p["plan"]["q_id"]
        question_operator = questions[question_id]["operator"]

        if question_operator == "addition":
            # Keep track of total number of questions for this operator type
            results["addition_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["addition_decomps"]:
                results["addition_decomps"][decomp] = 1
            else:
                results["addition_decomps"][decomp] += 1

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "arithmetic" and step["operator_args"][0] == "sum":
                    results["addition_correct"] += 1
                    break
        elif question_operator == "boolean_equality":
            # Keep track of total number of questions for this operator type
            results["boolean_equality_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["boolean_equality_decomps"]:
                results["boolean_equality_decomps"][decomp] = 1
            else:
                results["boolean_equality_decomps"][decomp] += 1

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "boolean" and step["operator_subtype"] == "boolean-inequality":
                    results["boolean_equality_correct"] += 1
                    break
        elif question_operator == "boolean_existence":
            # Keep track of total number of questions for this operator type
            results["boolean_existence_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["boolean_existence_decomps"]:
                results["boolean_existence_decomps"][decomp] = 1
            else:
                results["boolean_existence_decomps"][decomp] += 1

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "boolean":
                    results["boolean_existence_correct"] += 1
                    break
        elif question_operator == "comparison":
            # Keep track of total number of questions for this operator type
            results["comparison_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["comparison_decomps"]:
                results["comparison_decomps"][decomp] = 1
            else:
                results["comparison_decomps"][decomp] += 1

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "comparison":
                    results["comparison_correct"] += 1
                    break
        elif question_operator == "discard":
            # Keep track of total number of questions for this operator type
            results["discard_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["discard_decomps"]:
                results["discard_decomps"][decomp] = 1
            else:
                results["discard_decomps"][decomp] += 1

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "discard":
                    results["discard_correct"] += 1
                    break
        elif question_operator == "intersection":
            # Keep track of total number of questions for this operator type
            results["intersection_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["intersection_decomps"]:
                results["intersection_decomps"][decomp] = 1
            else:
                results["intersection_decomps"][decomp] += 1

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "intersection":
                    results["intersection_correct"] += 1
                    break
        elif question_operator == "subtraction":
            # Keep track of total number of questions for this operator type
            results["subtraction_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["subtraction_decomps"]:
                results["subtraction_decomps"][decomp] = 1
            else:
                results["subtraction_decomps"][decomp] += 1

            # if decomp == "select, select, select, arithmetic":
            #     print([s["qdmr"] for s in p["plan"]["steps"][1:]])

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "arithmetic" and step["operator_args"][0] == "difference":
                    results["subtraction_correct"] += 1
                    break

        elif question_operator == "bridge":
            # Keep track of total number of questions for this operator type
            results["bridge_total"] += 1

            # Keep track of the number of decomposition graphs for this operator type
            decomp = ', '.join([s["operator_type"] for s in p["plan"]["steps"]])
            if decomp not in results["bridge_decomps"]:
                results["bridge_decomps"][decomp] = 1
            else:
                results["bridge_decomps"][decomp] += 1

            # Keep track of how many decompositions contain the expected analytic operator
            for step in p["plan"]["steps"]:
                if step["operator_type"] == "project":
                    results["bridge_correct"] += 1
                    break

    # Print the results
    print("=" * 40)
    print("Decomposition Quality Results")
    operator_types = ["addition", "boolean_equality", "boolean_existence", "comparison", "discard", "intersection", "subtraction", "bridge"]
    for op in operator_types:
        print(op.upper())
        print(f"    {op} questions with expected operator step present:")
        print(f"         {100.0 * results[op + '_correct'] / results[op + '_total']:.2f}% ({results[op + '_correct']} / {results[op + '_total']})")
        print("    Decompositions:")
        print("    " + json.dumps(results[op + "_decomps"], indent=8))
    print("=" * 40)

    return None

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

def output_correctly_answered_questions(questions: dict[dict],
                                        plans: dict[str, dict]) -> None:
    num_correctly_answered = 0
    num_incorrectly_answered = 0
    correctly_answered_questions = []
    incorrectly_answered_questions = []
    for q in questions.values():
        mim_answer = plans[q["id"]]["answer"]
        f1, em, prec, recall = get_best_metrics_for_single_question(mim_answer, q["answer"])
        if f1 > 0.8 or em:
            num_correctly_answered += 1
            correctly_answered_questions.append(q["question"])

            # Mark this question as correct
            q["is_correct"] = True
        else:
            num_incorrectly_answered += 1
            incorrectly_answered_questions.append(q["question"])

            # Mark this question as incorrect
            q["is_correct"] = False

    # Print some metrics
    total_questions = num_correctly_answered + num_incorrectly_answered
    print(f"Questions correctly answered: {num_correctly_answered} / {total_questions} ({100 * num_correctly_answered / total_questions}%)")
    print(json.dumps(correctly_answered_questions, indent=4))
    print(f"Questions incorrectly answered: {num_incorrectly_answered} / {total_questions} ({100 * num_incorrectly_answered / total_questions}%)")
    print(json.dumps(incorrectly_answered_questions, indent=4))

    return None

def output_metrics_in_aggregate(questions: dict[dict]) -> None:
    #   How many questions were classified as simple?
    #   How many questions were classified as complex?
    num_simple_questions = 0
    num_complex_questions = 0
    for q in questions.values():
        if q["classification"] == "simple":
            num_simple_questions += 1
        elif q["classification"] == "complex":
            num_complex_questions += 1
        else:
            raise ValueError("Unexpected classification type.")
    print(f"Questions classified as simple: {num_simple_questions} / {len(questions.values())} ({100 * num_simple_questions / len(questions.values()):.2f}%)")
    print(f"Questions classified as complex: {num_complex_questions} / {len(questions.values())} ({100 * num_complex_questions / len(questions.values()):.2f}%)")

    #   How many questions contained the expected analytic operator?
    num_containing_expected_operator = 0
    for q in questions.values():
        if q["decomposition_contains_expected_operator"]:
            num_containing_expected_operator += 1

    print(f"Questions containing expected analytic operator: {num_containing_expected_operator} / {len(questions.values())} ({100 * num_containing_expected_operator / len(questions.values()):.2f}%)")

def output_metrics_per_correctness(questions: dict[dict]) -> None:
    #   How many questions were correctly answered?
    #   How many questions were incorrectly answered?
    correct_questions = [q for q in questions.values() if q["is_correct"]]
    incorrect_questions = [q for q in questions.values() if not q["is_correct"]]
    print(f"Questions correctly answered: {len(correct_questions)} / {len(questions.values())} ({100 * len(correct_questions) / len(questions.values()):.2f}%)")
    print(f"Questions incorrectly answered: {len(incorrect_questions)} / {len(questions.values())} ({100 * len(incorrect_questions) / len(questions.values()):.2f}%)")

    #   How many correct questions were classified as simple?
    #   How many correct questions were classified as complex?
    num_correct_classified_simple = 0
    num_correct_classified_complex = 0
    for q in correct_questions:
        if q["classification"] == "simple":
            num_correct_classified_simple += 1
        elif q["classification"] == "complex":
            num_correct_classified_complex += 1
        else:
            raise ValueError("Unexpected classification type.")
    print(f"Correctly answered questions classified as simple: {num_correct_classified_simple} / {len(correct_questions)} ({100 * num_correct_classified_simple / len(correct_questions):.2f}%)")
    print(f"Correctly answered questions classified as complex: {num_correct_classified_complex} / {len(correct_questions)} ({100 * num_correct_classified_complex / len(correct_questions):.2f}%)")

    #   How many incorrect questions were classified as complex
    #   How many incorrect questions were classified as complex
    num_incorrect_classified_simple = 0
    num_incorrect_classified_complex = 0
    for q in incorrect_questions:
        if q["classification"] == "simple":
            num_incorrect_classified_simple += 1
        elif q["classification"] == "complex":
            num_incorrect_classified_complex += 1
        else:
            raise ValueError("Unexpected classification type.")
    print(f"Incorrectly answered questions classified as simple: {num_incorrect_classified_simple} / {len(incorrect_questions)} ({100 * num_incorrect_classified_simple / len(incorrect_questions):.2f}%)")
    print(f"Incorrectly answered questions classified as complex: {num_incorrect_classified_complex} / {len(incorrect_questions)} ({100 * num_incorrect_classified_complex / len(incorrect_questions):.2f}%)")

    #   How many correct questions contained the expected analytic operator?
    #   How many incorrect questions contained the expected analytic operator?
    num_correct_decomposed_containing_expected_operator = 0
    num_incorrect_decomposed_containing_expected_operator = 0
    for q in correct_questions:
        if q["decomposition_contains_expected_operator"]:
            num_correct_decomposed_containing_expected_operator += 1
    for q in incorrect_questions:
        if q["decomposition_contains_expected_operator"]:
            num_incorrect_decomposed_containing_expected_operator += 1

    print(f"Correctly answered questions containing expected analytic operator: {num_correct_decomposed_containing_expected_operator} / {len(correct_questions)} ({100 * num_correct_decomposed_containing_expected_operator / len(correct_questions):.2f}%)")
    print(f"Incorrectly answered questions containing expected analytic operator: {num_incorrect_decomposed_containing_expected_operator} / {len(incorrect_questions)} ({100 * num_incorrect_decomposed_containing_expected_operator / len(incorrect_questions):.2f}%)")

    return None

def output_metrics_per_operator_type(questions: dict[dict]) -> None:

    return None

@plac.annotations(
    plan_file=('Path to the Mim Analytics Dataset question JSON file.', 'positional', None, str)
)
def main(question_file: str,
         plan_file: str):

    # Read in the questions from the file
    with open(question_file) as f:
        original_questions = json.load(f)
        questions = map_question_id_to_question(original_questions)

    # Read in the plans from the file
    with open(plan_file) as f:
        plans = json.load(f)

    annotate_complex_classification(questions, plans)
    annotate_decomposition_quality(questions, plans)
    annotate_correctness(questions, plans)

    # Produce lists of questions that were correctly or incorrectly answered
    # output_correctly_answered_questions(questions, plans)

    # Check that the complex-simple classifier is working
    output_complex_classification_results(questions, plans)

    # Check if the decompositions are as expected for each of the operator types
    output_decomposition_quality_results(questions, plans)

    # TODO: Check if the final answers are of an expected type (numeric for addition/subtraction, yes/no for booleans, non yes/no strings for others)

    # TODO: Check if the evidence retrieved at each step is correct

    # output_metrics_in_aggregate(questions)
    # output_metrics_per_correctness(questions)

    # TODO: Statistics to Report
    #   For each question type, what were the decomposition graphs and how many of each were there?
    #   For each of the questions type, which is the distribution of decomposition graphs for correctly answered questions?
    #   For each of the questions type, which is the distribution of decomposition graphs for incorrectly answered questions?

if __name__ == "__main__":
    plac.call(main)
