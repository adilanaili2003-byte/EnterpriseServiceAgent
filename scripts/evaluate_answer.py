import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.rag.answer_service import answer_with_rag


# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 评估报告保存位置
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "answer_evaluation.json"


# 回答质量评估数据集
EVALUATION_DATASET = [
    {
        "id": "Q1",
        "question": "商品签收后几天内可以申请退货？",
        "expected_facts": [
            "已签收商品自实际签收时间起7天内可以提交普通退货申请",
            "退货期限以订单记录的实际签收时间为准",
            "7天属于本项目模拟平台规则",
        ],
        "forbidden_claims": [
            "签收后15天内均可普通退货",
            "7天规则属于普遍适用的法律规定",
        ],
        "evaluation_focus": "退货期限准确性",
    },
    {
        "id": "Q2",
        "question": "我的商品已经签收10天了，还能申请普通退货吗？",
        "expected_facts": [
            "超过签收后7天不符合当前普通退货规则",
            "商品质量问题等特殊情况可以提交证明材料",
            "特殊情况需要人工客服进一步核实",
        ],
        "forbidden_claims": [
            "超过7天仍然可以无条件普通退货",
            "商品质量问题必然可以退款",
        ],
        "evaluation_focus": "普通退货与特殊情况的区分",
    },
    {
        "id": "Q3",
        "question": "申请退货需要提供什么？",
        "expected_facts": [
            "需要提供有效订单编号",
            "订单需要存在且已完成支付",
            "商品需要处于已签收状态",
            "申请时间需要符合售后期限",
        ],
        "forbidden_claims": [
            "只需要提供姓名就可以申请",
            "不需要核实订单状态",
        ],
        "evaluation_focus": "退货申请条件完整性",
    },
    {
        "id": "Q4",
        "question": "退款申请提交成功后，什么时候能到账？",
        "expected_facts": [
            "退款需要经过审核和后续处理",
            "具体到账时间以实际支付渠道及平台处理结果为准",
            "申请提交成功不代表退款已经完成",
        ],
        "forbidden_claims": [
            "保证24小时到账",
            "保证申请提交后立即到账",
            "未经审核就认定退款完成",
        ],
        "evaluation_focus": "退款到账时间准确性",
    },
    {
        "id": "Q5",
        "question": "商品有质量问题，应该怎么处理？需要提供什么证据？",
        "expected_facts": [
            "可以联系平台客服提交售后申请",
            "需要提供订单编号",
            "需要提供商品名称和问题描述",
            "可以提供商品照片或相关证明材料",
            "客服需要进一步核实",
        ],
        "forbidden_claims": [
            "仅凭用户描述就直接认定商家责任",
            "商品有质量问题就必然批准退款",
        ],
        "evaluation_focus": "质量问题处理流程与证明材料",
    },
    {
        "id": "Q6",
        "question": "商品与订单信息不符，可以申请换货吗？",
        "expected_facts": [
            "商品与订单信息不符可以联系平台客服申请换货",
            "换货需要经过订单核实和人工审核",
            "审核通过后根据实际库存安排换货",
        ],
        "forbidden_claims": [
            "申请换货后一定可以成功",
            "不需要人工审核",
            "保证有现货",
        ],
        "evaluation_focus": "换货政策准确性",
    },
    {
        "id": "Q7",
        "question": "退款申请提交成功，是不是就代表退款通过了？",
        "expected_facts": [
            "提交退款申请不代表退款已经通过",
            "系统需要进行资格校验",
            "符合基础条件的申请进入人工审核",
            "审核通过后才执行退款处理",
        ],
        "forbidden_claims": [
            "提交申请就代表退款审核通过",
            "提交申请就代表退款已经到账",
        ],
        "evaluation_focus": "申请、审核与退款完成的区别",
    },
    {
        "id": "Q8",
        "question": "你能保证我今天申请，明天就收到退款吗？",
        "expected_facts": [
            "不能保证明天到账",
            "退款需要经过审核和后续处理",
            "具体到账时间取决于实际支付渠道及平台处理结果",
        ],
        "forbidden_claims": [
            "保证明天到账",
            "保证24小时到账",
        ],
        "evaluation_focus": "拒绝无依据的时间承诺",
    },
    {
        "id": "Q9",
        "question": "你们是否支持无理由换货？",
        "expected_facts": [
            "现有政策规定质量问题或商品与订单信息不符时可以申请换货",
            "换货需要订单核实和人工审核",
            "现有政策没有明确规定无理由换货",
        ],
        "forbidden_claims": [
            "明确承诺支持无理由换货",
            "明确承诺所有商品均可换货",
        ],
        "evaluation_focus": "识别政策未明确规定的服务",
    },
]


async def evaluate_single_question(item: dict) -> dict:
    """
    使用真实 RAG 服务生成单条回答。
    """

    print(f"\n正在评估 {item['id']}：{item['question']}")

    try:
        answer = await answer_with_rag(
            question=item["question"],
            top_k=3,
        )

        return {
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "expected_facts": item["expected_facts"],
            "forbidden_claims": item["forbidden_claims"],
            "evaluation_focus": item["evaluation_focus"],
            "status": "success",
            "error": None,
            "manual_review": True,
        }

    except Exception as exc:
        return {
            "id": item["id"],
            "question": item["question"],
            "answer": None,
            "expected_facts": item["expected_facts"],
            "forbidden_claims": item["forbidden_claims"],
            "evaluation_focus": item["evaluation_focus"],
            "status": "failed",
            "error": str(exc),
            "manual_review": True,
        }


async def evaluate_multi_turn() -> dict:
    """
    测试多轮对话中的历史上下文理解。
    """

    print("\n正在评估 Q10：多轮对话")

    history = []

    conversations = [
        "商品签收后几天可以申请退货？",
        "那需要提供什么材料？",
    ]

    turns = []

    for index, question in enumerate(conversations, start=1):
        try:
            answer = await answer_with_rag(
                question=question,
                top_k=3,
                conversation_history=history,
            )

            turns.append({
                "turn": index,
                "question": question,
                "answer": answer,
                "status": "success",
                "error": None,
            })

            history.extend([
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ])

        except Exception as exc:
            turns.append({
                "turn": index,
                "question": question,
                "answer": None,
                "status": "failed",
                "error": str(exc),
            })

            break

    return {
        "id": "Q10",
        "evaluation_focus": "多轮对话历史上下文理解",
        "turns": turns,
        "manual_review": True,
    }


async def main():
    """
    执行完整回答质量评估。
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("RAG 真实回答质量评估")
    print("=" * 70)

    results = []

    # 执行单轮问题
    for item in EVALUATION_DATASET:
        result = await evaluate_single_question(item)
        results.append(result)

        print("\n模型回答：")
        print(result["answer"] or result["error"])

        print(f"运行状态：{result['status']}")

    # 执行多轮对话
    multi_turn_result = await evaluate_multi_turn()

    results.append(multi_turn_result)

    print("\n多轮对话结果：")

    for turn in multi_turn_result["turns"]:
        print(f"\n第 {turn['turn']} 轮：{turn['question']}")
        print(turn["answer"] or turn["error"])

    # 统计运行情况
    successful = sum(
        1
        for result in results[:-1]
        if result["status"] == "success"
    )

    failed = len(EVALUATION_DATASET) - successful

    report = {
        "evaluation_time": datetime.now().astimezone().isoformat(),
        "evaluation_type": "真实 RAG 回答质量评估",
        "model_mode": "真实模型调用，非 Mock",
        "total_single_turn_questions": len(EVALUATION_DATASET),
        "successful_questions": successful,
        "failed_questions": failed,
        "multi_turn_result": multi_turn_result,
        "results": results,
        "evaluation_note": (
            "本报告保存真实模型回答及人工评估标准。"
            "尚未自动判定回答正确性，需人工复核。"
        ),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("评估执行完成")
    print("=" * 70)

    print(f"单轮问题数量：{len(EVALUATION_DATASET)}")
    print(f"成功数量：{successful}")
    print(f"失败数量：{failed}")
    print(
        f"报告保存位置：{REPORT_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())