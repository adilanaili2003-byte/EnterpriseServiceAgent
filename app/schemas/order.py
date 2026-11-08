from datetime import datetime
from pydantic import BaseModel, Field


class OrderInfo(BaseModel):
    """订单信息模型"""

    order_id: str = Field(..., description="订单编号")

    product_name: str = Field(..., description="商品名称")

    amount: float = Field(..., ge=0, description="订单金额")

    status: str = Field(..., description="订单状态")

    payment_status: str = Field(..., description="支付状态")

    created_at: datetime = Field(..., description="订单创建时间")

    signed_at: datetime | None = Field(
        default=None,
        description="商品签收时间"
    )