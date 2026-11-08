from datetime import datetime

from app.schemas.order import OrderInfo


MOCK_ORDERS = {
    "ORD202611001": {
        "order_id": "ORD202611001",
        "product_name": "智能蓝牙耳机",
        "amount": 299.00,
        "status": "已签收",
        "payment_status": "已支付",
        "created_at": datetime(2026, 11, 1, 10, 30),
        "signed_at": datetime(2026, 11, 2, 15, 30),
    },
    "ORD202611002": {
        "order_id": "ORD202611002",
        "product_name": "智能手表",
        "amount": 599.00,
        "status": "运输中",
        "payment_status": "已支付",
        "created_at": datetime(2026, 11, 2, 14, 20),
        "signed_at": None,
    },
    "ORD202611003": {
        "order_id": "ORD202611003",
        "product_name": "无线键盘",
        "amount": 199.00,
        "status": "待发货",
        "payment_status": "已支付",
        "created_at": datetime(2026, 11, 3, 9, 15),
        "signed_at": None,
    },
}


def get_order_by_id(order_id: str) -> OrderInfo | None:
    """根据订单编号查询订单。"""

    order_data = MOCK_ORDERS.get(order_id)

    if order_data is None:
        return None

    return OrderInfo(**order_data)