import json
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "reports" / "answer_evaluation.json"
OUTPUT_PATH = BASE_DIR / "reports" / "answer_evaluation_scored.json"


# 四项评分权重
WEIGHTS = {
    "factual_accuracy": 0.40,
    "policy_faithfulness": 0.30,
    "answer_completeness": 0.20,
    "business_boundary": 0.10,
}


# 人工初步评分
MANUAL_SCORES = {
    "Q1": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 4,
            "answer_completeness": 4,
            "business_boundary": 5,
        },
        "analysis": "正确说明7天退货期限，但没有主动强调这是项目模拟规则。",
        "suggestion": "回答中适当说明7天属于本项目模拟平台规则。",
    },
    "Q2": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 5,
            "answer_completeness": 5,
            "business_boundary": 5,
        },
        "analysis": "正确区分普通退货期限与质量问题等特殊情况。",
        "suggestion": "暂时无需优化。",
    },
    "Q3": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 5,
            "answer_completeness": 5,
            "business_boundary": 5,
        },
        "analysis": "完整说明订单编号、支付状态、签收状态和申请期限。",
        "suggestion": "暂时无需优化。",
    },
    "Q4": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 5,
            "answer_completeness": 5,
            "business_boundary": 5,
        },
        "analysis": "正确说明退款到账时间取决于支付渠道和平台处理结果，没有编造具体天数。",
        "suggestion": "暂时无需优化。",
    },
    "Q5": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 5,
            "answer_completeness": 5,
            "business_boundary": 5,
        },
        "analysis": "完整说明质量问题处理流程、所需材料和人工核实要求。",
        "suggestion": "暂时无需优化。",
    },
    "Q6": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 5,
            "answer_completeness": 5,
            "business_boundary": 5,
        },
        "analysis": "正确说明换货申请、订单核实、人工审核和库存安排。",
        "suggestion": "暂时无需优化。",
    },
    "Q7": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 4,
            "answer_completeness": 5,
            "business_boundary": 5,
        },
        "analysis": "整体回答正确，但将提交成功表述为申请已受理，语义不够严谨。",
        "suggestion": "改为提交成功仅表示退款申请已提交，不代表审核通过或退款完成。",
    },
    "Q8": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 5,
            "answer_completeness": 5,
            "business_boundary": 5,
        },
        "analysis": "正确拒绝无法保证的到账时间，并说明退款处理流程。",
        "suggestion": "暂时无需优化。",
    },
    "Q9": {
        "scores": {
            "factual_accuracy": 4,
            "policy_faithfulness": 3,
            "answer_completeness": 4,
            "business_boundary": 5,
        },
        "analysis": "将政策明确列出的两种换货情形描述为全部换货情形，存在扩大解释。",
        "suggestion": "说明现有政策明确规定的换货情形，同时指出无理由换货没有明确规定。",
    },
    "Q10": {
        "scores": {
            "factual_accuracy": 5,
            "policy_faithfulness": 5,
            "answer_completeness": 4,
            "business_boundary": 5,
        },
        "analysis": "能够理解第二轮追问的历史上下文，但没有充分区分用户需要提供的材料与系统需要核验的条件。",
        "suggestion": "分别说明需要提供的订单编号、特殊情况证明材料，以及系统核验的订单状态和申请期限。",
    },
}


def calculate_score(scores: dict) -> float:
    """根据权重计算百分制总分。"""

    total = 0

    for dimension, weight in WEIGHTS.items():
        value = scores[dimension]

        if not isinstance(value, (int, float)) or not 0 <= value <= 5:
            raise ValueError(
                f"{dimension} 评分必须在0到5之间"
            )

        total += value / 5 * weight * 100

    return round(total, 2)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到原始评估报告：{INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    results = report.get("results", [])

    # 建立问题ID与原始回答的对应关系
    original_results = {
        item["id"]: item
        for item in results
    }

    # 检查10个评估场景是否齐全
    expected_ids = set(MANUAL_SCORES.keys())
    actual_ids = set(original_results.keys())

    if expected_ids != actual_ids:
        raise ValueError(
            f"评估问题不匹配。缺少：{expected_ids - actual_ids}，"
            f"多出：{actual_ids - expected_ids}"
        )

    scored_results = []

    for question_id, evaluation in MANUAL_SCORES.items():
        scores = evaluation["scores"]
        total_score = calculate_score(scores)

        original = original_results[question_id]

        scored_item = {
            "id": question_id,
            "question": (
                original.get("question")
                if question_id != "Q10"
                else "多轮追问：商品签收后几天可以申请退货？那需要提供什么材料？"
            ),
            "answer": (
                original.get("answer")
                if question_id != "Q10"
                else original.get("turns")
            ),
            "scores": scores,
            "weights": WEIGHTS,
            "total_score": total_score,
            "analysis": evaluation["analysis"],
            "suggestion": evaluation["suggestion"],
            "review_status": "初步人工评分，待确认",
        }

        scored_results.append(scored_item)

    average_score = round(
        sum(item["total_score"] for item in scored_results)
        / len(scored_results),
        2,
    )

    output = {
        "evaluation_time": datetime.now().astimezone().isoformat(),
        "source_report": "answer_evaluation.json",
        "evaluation_method": "人工初步评分",
        "scoring_dimensions": WEIGHTS,
        "total_questions": len(scored_results),
        "average_score": average_score,
        "results": scored_results,
        "conclusion": (
            "本报告基于当前10个评估场景进行初步人工评分，"
            "评分结果需要人工确认，不代表系统在所有场景下的准确率。"
        ),
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("=" * 60)
    print("RAG 回答质量人工评分完成")
    print("=" * 60)

    for item in scored_results:
        print(
            f"{item['id']}：{item['total_score']}分"
        )

    print("-" * 60)
    print(f"评估题目：{len(scored_results)}")
    print(f"平均得分：{average_score}")
    print(f"报告位置：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()