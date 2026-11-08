from app.services.order_service import get_order_by_id


def test_get_existing_order():
    """测试查询存在的订单"""

    order = get_order_by_id("ORD202611001")

    assert order is not None
    assert order.order_id == "ORD202611001"
    assert order.product_name == "智能蓝牙耳机"
    assert order.amount == 299.00
    assert order.status == "已签收"


def test_get_order_not_found():
    """测试查询不存在的订单"""

    order = get_order_by_id("ORD999999999")

    assert order is None


def test_get_order_payment_status():
    """测试订单支付状态"""

    order = get_order_by_id("ORD202611002")

    assert order is not None
    assert order.payment_status == "已支付"