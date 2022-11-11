import os

from flask import (Flask, render_template, request)
from mim_core.components.Mim import Mim

test_plan = {
            "q_id": "810cc54d-41e2-4b20-ba5f-0d6dcd1a1aa6",
            "steps": [
                {
                    "qdmr": "What is the sum of the oxygen and hydrogen atoms in H2O?",
                    "q_id": "810cc54d-41e2-4b20-ba5f-0d6dcd1a1aa6",
                    "step_type": "complex",
                    "reference_id": 1,
                    "operator_type": "select",
                    "operator_subtype": "select-single",
                    "operator_args": [
                        "What is the sum of the oxygen and hydrogen atoms in H2O?"
                    ],
                    "entities": {
                        "the sum": {
                            "id": 19100278,
                            "label": "The Sum",
                            "description": None
                        },
                        "atoms": {
                            "id": 1324392,
                            "label": "Tokyo Yakult Swallows",
                            "description": "Nippon Professional Baseball team in the Central League"
                        },
                        "H2O": {
                            "id": 283,
                            "label": "water",
                            "description": "chemical compound; main constituent of the fluids of most living organisms"
                        }
                    },
                    "entity_class": None,
                    "question_text": "What is the sum of the oxygen and hydrogen atoms in h2o?",
                    "relationship": {
                        "id": None,
                        "label": "part of",
                        "inverted": False
                    },
                    "expected_answer_type": [
                        "NUMERIC"
                    ],
                    "parent_steps": [],
                    "child_steps": [],
                    "result": "{\"id\":{\"0\":0},\"answer\":{\"0\":null},\"source\":{\"0\":null},\"confidence\":{\"0\":0.0},\"answer_confidence\":{\"0\":null},\"doc_retrieval_confidence\":{\"0\":null},\"question_string\":{\"0\":null}}",
                    "timing": {
                        "total": 0.0009696483612060547
                    },
                    "errors": [],
                    "misc": {},
                    "is_built": True
                },
                {
                    "qdmr": "oxygen atoms in H2O",
                    "q_id": "810cc54d-41e2-4b20-ba5f-0d6dcd1a1aa6",
                    "step_type": "simple",
                    "reference_id": 1,
                    "operator_type": "select",
                    "operator_subtype": "select-class",
                    "operator_args": [
                        "oxygen atoms in H2O"
                    ],
                    "entities": {
                        "atoms": {
                            "id": 1324392,
                            "label": "Tokyo Yakult Swallows",
                            "description": "Nippon Professional Baseball team in the Central League"
                        },
                        "H2O": {
                            "id": 283,
                            "label": "water",
                            "description": "chemical compound; main constituent of the fluids of most living organisms"
                        }
                    },
                    "entity_class": {
                        "id": 629,
                        "label": "oxygen"
                    },
                    "question_text": "What are the oxygen atoms in H2O?",
                    "relationship": {
                        "id": None,
                        "label": "part of",
                        "inverted": False
                    },
                    "expected_answer_type": [
                        "ENTITY"
                    ],
                    "parent_steps": [],
                    "child_steps": [
                        3
                    ],
                    "result": "{\"id\":{\"0\":0},\"answer\":{\"0\":\"two lone pairs\"},\"confidence\":{\"0\":1.0},\"answer_confidence\":{\"0\":1.0},\"question_string\":{\"0\":\"What are the oxygen atoms in H2O?\"},\"source\":{\"0\":\"IRRRAnsweringComponent\"},\"ConfidenceScorer\":{\"0\":1.0},\"SourceScorer\":{\"0\":1.0},\"final_score\":{\"0\":2.0}}",
                    "timing": {
                        "total": 150.7961061000824
                    },
                    "errors": [],
                    "misc": {
                        "question_strings": [
                            "What are the oxygen atoms in H2O?"
                        ]
                    },
                    "is_built": True
                },
                {
                    "qdmr": "hydrogen atoms in H2O",
                    "q_id": "810cc54d-41e2-4b20-ba5f-0d6dcd1a1aa6",
                    "step_type": "simple",
                    "reference_id": 2,
                    "operator_type": "select",
                    "operator_subtype": "select-class",
                    "operator_args": [
                        "hydrogen atoms in H2O"
                    ],
                    "entities": {
                        "atoms": {
                            "id": 1324392,
                            "label": "Tokyo Yakult Swallows",
                            "description": "Nippon Professional Baseball team in the Central League"
                        },
                        "H2O": {
                            "id": 283,
                            "label": "water",
                            "description": "chemical compound; main constituent of the fluids of most living organisms"
                        }
                    },
                    "entity_class": {
                        "id": 6643508,
                        "label": "hydrogen atom"
                    },
                    "question_text": "What are the hydrogen atoms in H2O?",
                    "relationship": {
                        "id": None,
                        "label": "part of",
                        "inverted": False
                    },
                    "expected_answer_type": [
                        "ENTITY"
                    ],
                    "parent_steps": [],
                    "child_steps": [
                        3
                    ],
                    "result": "{\"id\":{\"0\":0},\"answer\":{\"0\":\"covalently bonded to oxygen in a water molecule\"},\"confidence\":{\"0\":1.0},\"answer_confidence\":{\"0\":1.0},\"question_string\":{\"0\":\"What are the hydrogen atoms in H2O?\"},\"source\":{\"0\":\"IRRRAnsweringComponent\"},\"ConfidenceScorer\":{\"0\":1.0},\"SourceScorer\":{\"0\":1.0},\"final_score\":{\"0\":2.0}}",
                    "timing": {
                        "total": 150.89874148368835
                    },
                    "errors": [],
                    "misc": {
                        "question_strings": [
                            "What are the hydrogen atoms in H2O?"
                        ]
                    },
                    "is_built": True
                },
                {
                    "qdmr": "sum of @@1@@ and @@2@@",
                    "q_id": "810cc54d-41e2-4b20-ba5f-0d6dcd1a1aa6",
                    "step_type": "simple",
                    "reference_id": 3,
                    "operator_type": "arithmetic",
                    "operator_subtype": "arithmetic",
                    "operator_args": [
                        "sum",
                        "@@1@@",
                        "@@2@@"
                    ],
                    "entities": {
                        "sum": {
                            "id": 218005,
                            "label": "sum",
                            "description": "addition of a sequence of numbers"
                        }
                    },
                    "entity_class": None,
                    "question_text": "",
                    "relationship": None,
                    "expected_answer_type": [
                        "NUMERIC"
                    ],
                    "parent_steps": [
                        2,
                        1
                    ],
                    "child_steps": [],
                    "result": "{\"id\":{},\"answer\":{},\"source\":{},\"confidence\":{},\"answer_confidence\":{},\"doc_retrieval_confidence\":{},\"question_string\":{},\"final_score\":{},\"@@1@@\":{},\"@@2@@\":{}}",
                    "timing": {
                        "total": 0.01355743408203125
                    },
                    "errors": [
                        "NoRelationshipFoundWarning: No relationship was found.",
                        "ValueError: No valid number words found! Please enter a valid number word (eg. two million twenty three thousand and forty nine)"
                    ],
                    "misc": {},
                    "is_built": True
                }
            ],
            "timing": {
                "question_analyzer": {
                    "complex_question_classification": 1.3589859008789062e-05,
                    "falcon_extraction": 8.106231689453125e-06,
                    "qdmr_operator_parsing": 0.001703023910522461,
                    "qdmr_graph_linking": 0.00011396408081054688,
                    "canonical_entity_extraction": 8.207500457763672,
                    "entity_class_extraction": 0.3158090114593506,
                    "suboperator_parsing": 4.267692565917969e-05,
                    "question_generation": 0.14919233322143555,
                    "canonical_relation_extraction": 0.7653367519378662,
                    "answer_type_classification": 0.05196213722229004,
                    "question_classification": 1.049041748046875e-05,
                    "question_decomposition": 0.6290404796600342,
                    "qdmr_validation": 0.0007622241973876953,
                    "plan_compiling": 0.0003139972686767578,
                    "total": 10.123914241790771
                },
                "answer_engine": {
                    "answer_evaluator": {
                        "total": 0.010516881942749023,
                        "ConfidenceScorer": 0.0019769668579101562,
                        "SourceScorer": 0.0038781166076660156,
                        "score_merging": 0.0028083324432373047,
                        "answer_thresholding_and_sorting": 0.001491546630859375
                    },
                    "HybridAnsweringComponent": {
                        "total": 301.6844472885132,
                        "document_retrieval": 0,
                        "answer_retrieval": 0,
                        "answer_processing": 0
                    },
                    "answer_merging": 5.4836273193359375e-06,
                    "IRRRAnsweringComponent": {
                        "total": 0,
                        "document_retrieval": 0,
                        "answer_retrieval": 301.6601459980011,
                        "answer_processing": 0
                    },
                    "total": 301.7095534801483
                }
            },
            "errors": []
        }

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)#, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # from . import mim_run
    # app.register_blueprint(mim_run.bp)

    mim_system = Mim("mim_core/evaluation/MimAnalyticsQuestions/mim_config.json")

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    # a simple homepage with navigation bar
    @app.route('/', methods=['GET', 'POST'])
    def home():
        if 'question_box' in request.form:
            answer, mqr = mim_system.answer_question(request.form['question_box'])
            # answer = 'springfield'
            answer_output = "Answer: " + answer
            steps_output = format_plan_output(mqr.to_json())
        else:
            answer_output = ""
            steps_output = []

        return render_template('home.html', answer_field=answer_output, steps=steps_output)

    return app

def format_plan_output(plan) -> list[dict]:
    out = []
    for s in plan['steps'][1:]:
        out.append({
            'step_type': s['step_type'],
            'qdmr': s['qdmr'],
            'operator_type': s['operator_type'],
            'operator_subtype': s['operator_subtype'],
            'generated_question': s['question_text'] if s['question_text'] else s['qdmr'],
            'entities': list(s['entities'].keys()),
            'answer_type': s['expected_answer_type'][0]
        })
    # for s in plan.steps[1:]:
    #     out.append({
    #         'step_type': s.step_type,
    #         'qdmr': s.qdmr,
    #         'operator_type': s.operator_type,
    #         'operator_subtype': s.operator_subtype,
    #         'generated_question': s.question_text if s.question_text else s.qdmr,
    #         'entities': list(s.entities.keys()),
    #         'answer_type': s.expected_answer_type[0]
    #     })
    return out



