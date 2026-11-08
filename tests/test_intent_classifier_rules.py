from app.agents.intent_classifier import classify_intent_by_rule


def test_policy_questions_are_general():
    messages = [
        "商品签收后几天内可以申请退货？",
        "商品已经超过七天，还能无理由退货吗？",
        "支持七天无理由退货吗？",
        "退款审核后多久到账？",
        "商品质量问题怎么办？",
        "我想申请退款，但是不知道流程",
    ]

    for message in messages:
        assert classify_intent_by_rule(message) == "general"


def test_specific_business_requests_are_after_sales():
    messages = [
        "帮我申请退款",
        "我要退掉 ORD202611001",
        "查询 ORD202611001 的订单状态",
        "帮我查询 ORD202611002 的物流",
        "申请换货",
    ]

    for message in messages:
        assert classify_intent_by_rule(message) == "after_sales"


def test_uncertain_messages_return_none():
    messages = [
        "你好",
        "你们有哪些商品？",
        "谢谢",
    ]

    for message in messages:
        assert classify_intent_by_rule(message) is None