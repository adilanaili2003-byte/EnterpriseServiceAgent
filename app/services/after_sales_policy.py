from datetime import datetime, timedelta

from app.schemas.order import OrderInfo


# 项目演示规则：签收后 7 天内可以进入售后资格审核。
AFTER_SALES_DAYS = 7


def check_after_sales_eligibility(
    order: OrderInfo | None,
    request_type: str,
    current_time: datetime | None = None,
) -> dict:
    """
    检查订单是否符合基础售后申请条件。

    注意：
    这里只判断基础资格，不代表退款已经批准。
    """

    # 1. 检查售后类型
    if request_type not in ("refund", "return"):
        return {
            "eligible": False,
            "reason": "暂不支持该售后申请类型",
            "code": "INVALID_REQUEST_TYPE",
        }

    # 2. 检查订单是否存在
    if order is None:
        return {
            "eligible": False,
            "reason": "未找到对应订单，请检查订单编号",
            "code": "ORDER_NOT_FOUND",
        }

    # 3. 检查支付状态
    if order.payment_status != "已支付":
        return {
            "eligible": False,
            "reason": "订单尚未支付，不符合退款申请条件",
            "code": "ORDER_NOT_PAID",
        }

    # 4. 检查订单是否已经签收
    if order.status != "已签收" or order.signed_at is None:
        return {
            "eligible": False,
            "reason": "订单尚未签收，暂不符合当前售后申请规则",
            "code": "ORDER_NOT_DELIVERED",
        }

    # 5. 检查签收时间
    now = current_time or datetime.now()

    if order.signed_at > now:
        return {
            "eligible": False,
            "reason": "订单签收时间异常，暂时无法处理",
            "code": "INVALID_SIGNED_TIME",
        }

    deadline = order.signed_at + timedelta(
        days=AFTER_SALES_DAYS
    )

    if now > deadline:
        return {
            "eligible": False,
            "reason": "订单已超过演示规则规定的 7 天售后期限",
            "code": "AFTER_SALES_EXPIRED",
        }

    # 6. 满足基础申请条件
    return {
        "eligible": True,
        "reason": "订单符合基础售后申请条件，可以进入后续审核",
        "code": "ELIGIBLE",
    }