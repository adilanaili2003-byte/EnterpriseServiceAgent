import asyncio
from unittest.mock import AsyncMock, patch

from app.agents.nodes import after_sales_response, general_response


def run_node(message: str):
    with patch(
        "app.agents.nodes.answer_with_rag",
        new_callable=AsyncMock,
        return_value="绗﹀悎鍩虹鐢宠鏉′欢锛屽彲浠ヨ繘鍏ヤ汉宸ュ鏍告祦绋嬶紝浣嗗皻鏈幏寰楁渶缁堟壒鍑嗐€?,
    ):
        return asyncio.run(
            after_sales_response({
                "user_message": message
            })
        )


def test_after_sales_with_valid_order():
    """娴嬭瘯鏈夋晥璁㈠崟鐨勯€€娆剧敵璇枫€?""

    result = run_node(
        "鎴戠殑璁㈠崟 ORD202611001 鏈夎川閲忛棶棰橈紝鎴戞兂鐢宠閫€娆俱€?
    )

    assert result["status"] == "success"
    assert result["order_id"] == "ORD202611001"
    assert result["request_type"] == "refund"
    assert result["eligibility_result"]["eligible"] is True
    assert "浜哄伐瀹℃牳" in result["answer"]


def test_after_sales_without_order_id():
    """娴嬭瘯鐢ㄦ埛娌℃湁鎻愪緵璁㈠崟缂栧彿銆?""

    result = run_node(
        "鎴戠殑鍟嗗搧鏈夎川閲忛棶棰橈紝鎴戞兂鐢宠閫€娆俱€?
    )

    assert result["status"] == "success"
    assert "璁㈠崟缂栧彿" in result["answer"]


def test_after_sales_order_not_found():
    """娴嬭瘯涓嶅瓨鍦ㄧ殑璁㈠崟銆?""

    result = run_node(
        "鎴戠殑璁㈠崟 ORD999999999 鏈夐棶棰橈紝鎴戞兂鐢宠閫€娆俱€?
    )

    assert result["status"] == "success"
    assert result["eligibility_result"]["eligible"] is False
    assert result["eligibility_result"]["code"] == "ORDER_NOT_FOUND"


def test_after_sales_missing_request_type():
    """娴嬭瘯鐢ㄦ埛鎻愪緵璁㈠崟缂栧彿浣嗘病鏈夋槑纭敵璇风被鍨嬨€?""

    result = run_node(
        "鎴戠殑璁㈠崟 ORD202611001 鏈夎川閲忛棶棰樸€?
    )

    assert result["status"] == "success"
    assert "閫€娆捐繕鏄€€璐? in result["answer"]

def test_after_sales_rag_failure():
    """娴嬭瘯 RAG 澶辫触鏃朵粛鐒惰繑鍥炶鍒欏紩鎿庣粨鏋溿€?""

    with patch(
        "app.agents.nodes.answer_with_rag",
        new_callable=AsyncMock,
        side_effect=RuntimeError("妯℃嫙 RAG 鏈嶅姟寮傚父"),
    ):
        result = asyncio.run(
            after_sales_response({
                "user_message": "鎴戠殑璁㈠崟 ORD202611001 鏈夎川閲忛棶棰橈紝鎴戞兂鐢宠閫€娆俱€?
            })
        )

    assert result["status"] == "success"
    assert result["eligibility_result"]["eligible"] is True
    assert "浜哄伐瀹℃牳" in result["answer"]

def test_general_response_uses_top_5():
    """娴嬭瘯鏅€氬挩璇㈣妭鐐逛娇鐢?Top 5 妫€绱㈢煡璇嗗簱銆?""

    with patch(
        "app.agents.nodes.answer_with_rag",
        new_callable=AsyncMock,
        return_value="鏅€氶€€璐ф湡闄愪负绛炬敹鍚?澶┿€?,
    ) as mock_rag:

        result = asyncio.run(
            general_response({
                "user_message": "鍟嗗搧绛炬敹瓒呰繃7澶╄繕鑳界敵璇烽€€璐у悧锛?
            })
        )

    assert result["status"] == "success"
    assert result["answer"] == "鏅€氶€€璐ф湡闄愪负绛炬敹鍚?澶┿€?

    mock_rag.assert_awaited_once_with(
        question="鍟嗗搧绛炬敹瓒呰繃7澶╄繕鑳界敵璇烽€€璐у悧锛?,
        top_k=5,
        conversation_history=[],
    )
