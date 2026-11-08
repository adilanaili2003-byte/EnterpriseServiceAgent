from datetime import datetime

from app.services.order_service import get_order_by_id
from app.services.after_sales_policy import (
    check_after_sales_eligibility,
)


def test_eligible_refund():
    """测试签收期限内的正常售后申请。"""

    order = get_order_by_id("ORD202611001")

    result = check_after_sales_eligibility(
        order=order,
        request_type="refund",
        current_time=datetime(2026, 11, 5, 12, 0),
    )

    assert result["eligible"] is True
    assert result["code"] == "ELIGIBLE"


def test_order_not_found():
    """测试不存在的订单。"""

    order = get_order_by_id("ORD999999999")

    result = check_after_sales_eligibility(
        order=order,
        request_type="refund",
    )

    assert result["eligible"] is False
    assert result["code"] == "ORDER_NOT_FOUND"


def test_order_not_delivered():
    """测试尚未签收的订单。"""

    order = get_order_by_id("ORD202611002")

    result = check_after_sales_eligibility(
        order=order,
        request_type="refund",
    )

    assert result["eligible"] is False
    assert result["code"] == "ORDER_NOT_DELIVERED"


def test_after_sales_expired():
    """测试超过售后期限的订单。"""

    order = get_order_by_id("ORD202611001")

    result = check_after_sales_eligibility(
        order=order,
        request_type="return",
        current_time=datetime(2026, 11, 15, 12, 0),
    )

    assert result["eligible"] is False
    assert result["code"] == "AFTER_SALES_EXPIRED"


def test_invalid_request_type():
    """测试不支持的售后申请类型。"""

    order = get_order_by_id("ORD202611001")

    result = check_after_sales_eligibility(
        order=order,
        request_type="exchange",
    )

    assert result["eligible"] is False
    assert result["code"] == "INVALID_REQUEST_TYPE"