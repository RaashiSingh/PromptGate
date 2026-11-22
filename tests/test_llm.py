
import pytest
from unittest.mock import Mock, call

from promptgate import llm


# ==========================================
# TEST CONSTANTS
# ==========================================

SYSTEM_PROMPT = "You are a customer support classifier."
USER_INPUT = "I was charged twice for my subscription."
MODEL = "openai/gpt-oss-20b"

VALID_RESPONSE = (
    '{"priority":"high","category":"billing",'
    '"summary":"Duplicate subscription charge"}'
)


# ==========================================
# FIXTURES AND HELPERS
# ==========================================

def make_response(
    content=VALID_RESPONSE,
    input_tokens=100,
    output_tokens=50,
    include_usage=True,
):
    response = Mock()
    response.choices = [
        Mock(message=Mock(content=content))
    ]

    if include_usage:
        response.usage = Mock(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    else:
        response.usage = None

    return response


@pytest.fixture(autouse=True)
def isolate_client(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)

    monkeypatch.setenv(
        "LLM_API_KEY", "mock-test-key"
    )

    monkeypatch.setenv(
        "LLM_BASE_URL",
        "https://example.invalid/v1",
    )


@pytest.fixture
def mock_sdk(monkeypatch):
    constructor = Mock()

    monkeypatch.setattr(
        llm, "OpenAI", constructor
    )

    return constructor


@pytest.fixture
def mock_api(mock_sdk):
    client = mock_sdk.return_value

    create = client.chat.completions.create

    create.return_value = make_response()

    return create


@pytest.fixture
def mock_sleep(monkeypatch):
    sleep = Mock()

    monkeypatch.setattr(
        llm.time, "sleep", sleep
    )

    return sleep


def invoke(**kwargs):
    options = {
        "system_prompt": SYSTEM_PROMPT,
        "user_input": USER_INPUT,
        "model": MODEL,
    }

    options.update(kwargs)

    return llm.complete(**options)


# ==========================================
# SECTION 1: CLIENT INITIALIZATION
# ==========================================

def test_client_initially_uninitialized():
    assert llm._client is None


def test_client_created(mock_sdk):
    client = llm._get_client()

    assert client is mock_sdk.return_value


def test_client_correct_api_key(mock_sdk):
    llm._get_client()

    assert (
        mock_sdk.call_args.kwargs["api_key"]
        == "mock-test-key"
    )


def test_client_correct_base_url(mock_sdk):
    llm._get_client()

    assert (
        mock_sdk.call_args.kwargs["base_url"]
        == "https://example.invalid/v1"
    )


def test_client_sdk_retries(mock_sdk):
    llm._get_client()

    assert (
        mock_sdk.call_args.kwargs["max_retries"]
        == 5
    )


def test_client_timeout(mock_sdk):
    llm._get_client()

    assert (
        mock_sdk.call_args.kwargs["timeout"]
        == 60.0
    )


def test_client_reused(mock_sdk):
    first = llm._get_client()
    second = llm._get_client()

    assert first is second
    assert mock_sdk.call_count == 1


def test_existing_client_reused(
    monkeypatch, mock_sdk
):
    existing_client = Mock()

    monkeypatch.setattr(
        llm, "_client", existing_client
    )

    assert llm._get_client() is existing_client

    mock_sdk.assert_not_called()


def test_missing_api_key(
    monkeypatch, mock_sdk
):
    monkeypatch.delenv(
        "LLM_API_KEY", raising=False
    )

    with pytest.raises(KeyError):
        llm._get_client()

    mock_sdk.assert_not_called()


def test_missing_base_url(
    monkeypatch, mock_sdk
):
    monkeypatch.delenv(
        "LLM_BASE_URL", raising=False
    )

    with pytest.raises(KeyError):
        llm._get_client()

    mock_sdk.assert_not_called()


def test_failed_initialization_not_cached(
    mock_sdk,
):
    mock_sdk.side_effect = RuntimeError(
        "Initialization failed"
    )

    with pytest.raises(RuntimeError):
        llm._get_client()

    assert llm._client is None


# ==========================================
# SECTION 2: API REQUEST CONSTRUCTION
# ==========================================

def test_correct_model_sent(mock_api):
    invoke()

    assert (
        mock_api.call_args.kwargs["model"]
        == MODEL
    )


def test_default_temperature(mock_api):
    invoke()

    assert (
        mock_api.call_args.kwargs["temperature"]
        == 0.0
    )


@pytest.mark.parametrize(
    "temperature",
    [0.0, 0.2, 0.5, 0.7, 1.0],
)
def test_custom_temperature(
    mock_api, temperature
):
    invoke(temperature=temperature)

    assert (
        mock_api.call_args.kwargs["temperature"]
        == temperature
    )


def test_max_tokens(mock_api):
    invoke()

    assert (
        mock_api.call_args.kwargs["max_tokens"]
        == 500
    )


def test_system_prompt_sent(mock_api):
    invoke()

    messages = (
        mock_api.call_args.kwargs["messages"]
    )

    assert messages[0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }


def test_user_input_sent(mock_api):
    invoke()

    messages = (
        mock_api.call_args.kwargs["messages"]
    )

    assert messages[1] == {
        "role": "user",
        "content": USER_INPUT,
    }


def test_message_order(mock_api):
    invoke()

    messages = (
        mock_api.call_args.kwargs["messages"]
    )

    assert [
        message["role"]
        for message in messages
    ] == ["system", "user"]


def test_exact_request_arguments(mock_api):
    invoke()

    mock_api.assert_called_once_with(
        model=MODEL,
        temperature=0.0,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_INPUT,
            },
        ],
    )


def test_custom_model_forwarded(mock_api):
    invoke(model="custom-model")

    assert (
        mock_api.call_args.kwargs["model"]
        == "custom-model"
    )


def test_unicode_input_preserved(mock_api):
    message = "Payment failed: ₹500 — refund needed."

    invoke(user_input=message)

    assert (
        mock_api.call_args.kwargs["messages"][1]
        ["content"] == message
    )


def test_multiline_prompt_preserved(mock_api):
    prompt = "Classify tickets.\nReturn JSON only."

    invoke(system_prompt=prompt)

    assert (
        mock_api.call_args.kwargs["messages"][0]
        ["content"] == prompt
    )


# ==========================================
# SECTION 3: RESPONSE EXTRACTION
# ==========================================

def test_successful_response_text(mock_api):
    result = invoke()

    assert result["text"] == VALID_RESPONSE


def test_successful_response_no_error(mock_api):
    result = invoke()

    assert result["error"] is None


def test_response_schema(mock_api):
    result = invoke()

    assert set(result.keys()) == {
        "text",
        "latency",
        "cost",
        "input_tokens",
        "output_tokens",
        "error",
    }


def test_none_content_becomes_empty(mock_api):
    mock_api.return_value = make_response(
        content=None
    )

    result = invoke()

    assert result["text"] == ""
    assert result["error"] is None


def test_empty_content(mock_api):
    mock_api.return_value = make_response(
        content=""
    )

    result = invoke()

    assert result["text"] == ""
    assert result["error"] is None


def test_whitespace_preserved(mock_api):
    mock_api.return_value = make_response(
        content="  response  "
    )

    assert invoke()["text"] == "  response  "


def test_plain_text_response(mock_api):
    mock_api.return_value = make_response(
        content="A normal text response."
    )

    assert (
        invoke()["text"]
        == "A normal text response."
    )


def test_unicode_response(mock_api):
    mock_api.return_value = make_response(
        content="Refund of ₹500 requested."
    )

    assert (
        invoke()["text"]
        == "Refund of ₹500 requested."
    )


def test_first_choice_selected(mock_api):
    response = make_response(
        content="First response"
    )

    response.choices.append(
        Mock(
            message=Mock(
                content="Second response"
            )
        )
    )

    mock_api.return_value = response

    assert invoke()["text"] == "First response"


# ==========================================
# SECTION 4: TOKEN USAGE
# ==========================================

def test_input_tokens(mock_api):
    result = invoke()

    assert result["input_tokens"] == 100


def test_output_tokens(mock_api):
    result = invoke()

    assert result["output_tokens"] == 50


def test_missing_usage(mock_api):
    mock_api.return_value = make_response(
        include_usage=False
    )

    result = invoke()

    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


@pytest.mark.parametrize(
    "input_tokens,output_tokens",
    [
        (0, 0),
        (1, 1),
        (100, 50),
        (1000, 500),
        (10000, 2000),
    ],
)
def test_variable_token_usage(
    mock_api,
    input_tokens,
    output_tokens,
):
    mock_api.return_value = make_response(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    result = invoke()

    assert (
        result["input_tokens"] == input_tokens
    )

    assert (
        result["output_tokens"] == output_tokens
    )


# ==========================================
# SECTION 5: COST CALCULATION
# ==========================================

def test_groq_model_recorded_zero_cost(
    mock_api,
):
    assert invoke()["cost"] == 0.0


def test_unknown_model_zero_cost(mock_api):
    result = invoke(
        model="unknown-model"
    )

    assert result["cost"] == 0.0


def test_paid_model_cost(mock_api):
    result = invoke(model="gpt-4o-mini")

    expected = (
        100 * 0.15 + 50 * 0.60
    ) / 1_000_000

    assert result["cost"] == pytest.approx(
        expected
    )


def test_zero_tokens_zero_cost(mock_api):
    mock_api.return_value = make_response(
        input_tokens=0,
        output_tokens=0,
    )

    result = invoke(model="gpt-4o-mini")

    assert result["cost"] == 0.0


def test_missing_usage_zero_cost(mock_api):
    mock_api.return_value = make_response(
        include_usage=False
    )

    result = invoke(model="gpt-4o-mini")

    assert result["cost"] == 0.0


@pytest.mark.parametrize(
    "input_tokens,output_tokens",
    [
        (100, 200),
        (1000, 500),
        (0, 1000),
        (1000, 0),
        (1000000, 1000000),
    ],
)
def test_variable_cost_calculation(
    mock_api,
    input_tokens,
    output_tokens,
):
    mock_api.return_value = make_response(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    result = invoke(model="gpt-4o-mini")

    expected = (
        input_tokens * 0.15
        + output_tokens * 0.60
    ) / 1_000_000

    assert result["cost"] == pytest.approx(
        expected
    )


def test_custom_model_pricing(
    mock_api, monkeypatch
):
    monkeypatch.setitem(
        llm.PRICES,
        "custom-paid-model",
        (2.0, 4.0),
    )

    result = invoke(
        model="custom-paid-model"
    )

    expected = (
        100 * 2.0 + 50 * 4.0
    ) / 1_000_000

    assert result["cost"] == pytest.approx(
        expected
    )


# ==========================================
# SECTION 6: LATENCY MEASUREMENT
# ==========================================

def test_latency_calculation(
    mock_api, monkeypatch
):
    clock = Mock(
        side_effect=[10.0, 10.75]
    )

    monkeypatch.setattr(
        llm.time, "perf_counter", clock
    )

    result = invoke()

    assert result["latency"] == pytest.approx(
        0.75
    )


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (10.0, 10.1, 0.1),
        (20.0, 21.5, 1.5),
        (100.0, 105.0, 5.0),
        (10.0, 10.0, 0.0),
    ],
)
def test_variable_latency(
    mock_api,
    monkeypatch,
    start,
    end,
    expected,
):
    monkeypatch.setattr(
        llm.time,
        "perf_counter",
        Mock(side_effect=[start, end]),
    )

    assert invoke()["latency"] == pytest.approx(
        expected
    )


# ==========================================
# SECTION 7: RETRY MECHANISM
# ==========================================

def test_success_no_retry(
    mock_api, mock_sleep
):
    result = invoke()

    assert result["error"] is None
    assert mock_api.call_count == 1

    mock_sleep.assert_not_called()


def test_success_after_one_retry(
    mock_api, mock_sleep
):
    mock_api.side_effect = [
        RuntimeError("Temporary failure"),
        make_response(),
    ]

    result = invoke()

    assert result["error"] is None
    assert result["text"] == VALID_RESPONSE
    assert mock_api.call_count == 2

    mock_sleep.assert_called_once_with(1)


def test_success_after_two_retries(
    mock_api, mock_sleep
):
    mock_api.side_effect = [
        RuntimeError("First failure"),
        RuntimeError("Second failure"),
        make_response(),
    ]

    result = invoke()

    assert result["error"] is None
    assert mock_api.call_count == 3

    assert mock_sleep.call_args_list == [
        call(1),
        call(2),
    ]


@pytest.mark.parametrize(
    "retries", [1, 2, 3, 4, 5]
)
def test_retry_attempt_count(
    mock_api, mock_sleep, retries
):
    mock_api.side_effect = RuntimeError(
        "API unavailable"
    )

    result = invoke(retries=retries)

    assert mock_api.call_count == retries

    assert (
        mock_sleep.call_count == retries - 1
    )

    assert result["error"] == "API unavailable"


def test_exponential_backoff(
    mock_api, mock_sleep
):
    mock_api.side_effect = RuntimeError(
        "Temporary API failure"
    )

    invoke(retries=5)

    assert mock_sleep.call_args_list == [
        call(1),
        call(2),
        call(4),
        call(8),
    ]


def test_no_sleep_after_final_failure(
    mock_api, mock_sleep
):
    mock_api.side_effect = RuntimeError(
        "Permanent failure"
    )

    invoke(retries=1)

    mock_sleep.assert_not_called()


def test_request_preserved_across_retries(
    mock_api, mock_sleep
):
    mock_api.side_effect = [
        RuntimeError("Temporary failure"),
        make_response(),
    ]

    invoke()

    first, second = mock_api.call_args_list

    assert first == second


def test_retry_returns_successful_metrics(
    mock_api, mock_sleep
):
    mock_api.side_effect = [
        RuntimeError("Temporary failure"),
        make_response(
            input_tokens=250,
            output_tokens=75,
        ),
    ]

    result = invoke()

    assert result["input_tokens"] == 250
    assert result["output_tokens"] == 75
    assert result["error"] is None


def test_retry_latency_successful_attempt_only(
    mock_api,
    mock_sleep,
    monkeypatch,
):
    mock_api.side_effect = [
        RuntimeError("Temporary failure"),
        make_response(),
    ]

    monkeypatch.setattr(
        llm.time,
        "perf_counter",
        Mock(
            side_effect=[10.0, 20.0, 20.5]
        ),
    )

    result = invoke()

    assert result["latency"] == pytest.approx(
        0.5
    )


# ==========================================
# SECTION 8: API FAILURE HANDLING
# ==========================================

@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("429 Rate limit exceeded"),
        TimeoutError("Request timed out"),
        ConnectionError("Network unavailable"),
        RuntimeError("500 Internal server error"),
        RuntimeError("503 Service unavailable"),
        RuntimeError("401 Unauthorized"),
        RuntimeError("Invalid model"),
    ],
    ids=[
        "rate_limit",
        "timeout",
        "connection",
        "server_error",
        "service_unavailable",
        "unauthorized",
        "invalid_model",
    ],
)
def test_api_error_handling(
    mock_api, mock_sleep, exception
):
    mock_api.side_effect = exception

    result = invoke()

    assert result["error"] == str(exception)
    assert result["text"] == ""
    assert mock_api.call_count == 3


def test_rate_limit_recovery(
    mock_api, mock_sleep
):
    mock_api.side_effect = [
        RuntimeError("429 Rate limit exceeded"),
        make_response(),
    ]

    result = invoke()

    assert result["error"] is None
    assert result["text"] == VALID_RESPONSE


def test_timeout_recovery(
    mock_api, mock_sleep
):
    mock_api.side_effect = [
        TimeoutError("Request timed out"),
        make_response(),
    ]

    result = invoke()

    assert result["error"] is None
    assert mock_api.call_count == 2


def test_last_error_is_returned(
    mock_api, mock_sleep
):
    mock_api.side_effect = [
        RuntimeError("First error"),
        RuntimeError("Second error"),
        RuntimeError("Final error"),
    ]

    result = invoke()

    assert result["error"] == "Final error"


def test_failure_result_schema(
    mock_api, mock_sleep
):
    mock_api.side_effect = RuntimeError(
        "API failure"
    )

    result = invoke()

    assert result == {
        "text": "",
        "latency": 0.0,
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": "API failure",
    }


def test_initialization_failure_handled(
    mock_sdk, mock_sleep
):
    mock_sdk.side_effect = RuntimeError(
        "Client unavailable"
    )

    result = invoke()

    assert result["error"] == "Client unavailable"
    assert mock_sdk.call_count == 3
    assert llm._client is None


def test_initialization_retry_recovery(
    mock_sdk, mock_sleep
):
    client = Mock()

    client.chat.completions.create.return_value = (
        make_response()
    )

    mock_sdk.side_effect = [
        RuntimeError("Initialization failed"),
        client,
    ]

    result = invoke()

    assert result["error"] is None
    assert mock_sdk.call_count == 2
    assert llm._client is client


# ==========================================
# SECTION 9: MALFORMED API RESPONSES
# ==========================================

def test_empty_choices(
    mock_api, mock_sleep
):
    response = make_response()
    response.choices = []

    mock_api.return_value = response

    result = invoke()

    assert result["error"] is not None
    assert result["text"] == ""
    assert mock_api.call_count == 3


def test_none_response(
    mock_api, mock_sleep
):
    mock_api.return_value = None

    result = invoke()

    assert result["error"] is not None
    assert result["text"] == ""


def test_missing_message(
    mock_api, mock_sleep
):
    response = make_response()

    response.choices = [
        Mock(message=None)
    ]

    mock_api.return_value = response

    result = invoke()

    assert result["error"] is not None


def test_malformed_usage(
    mock_api, mock_sleep
):
    response = make_response()

    response.usage.prompt_tokens = None

    mock_api.return_value = response

    result = invoke()

    assert result["error"] is not None
    assert result["input_tokens"] == 0


def test_recovery_from_malformed_response(
    mock_api, mock_sleep
):
    malformed = make_response()
    malformed.choices = []

    mock_api.side_effect = [
        malformed,
        make_response(),
    ]

    result = invoke()

    assert result["error"] is None
    assert result["text"] == VALID_RESPONSE
    assert mock_api.call_count == 2


# ==========================================
# SECTION 10: CLIENT REUSE ACROSS REQUESTS
# ==========================================

def test_multiple_successful_requests(
    mock_sdk, mock_api
):
    first = invoke()
    second = invoke()

    assert first["error"] is None
    assert second["error"] is None

    assert mock_sdk.call_count == 1
    assert mock_api.call_count == 2


def test_client_reused_after_api_failure(
    mock_sdk, mock_api, mock_sleep
):
    mock_api.side_effect = [
        RuntimeError("Temporary failure"),
        make_response(),
    ]

    result = invoke()

    assert result["error"] is None
    assert mock_sdk.call_count == 1


def test_independent_response_metrics(mock_api):
    mock_api.side_effect = [
        make_response(
            content="Response one",
            input_tokens=100,
            output_tokens=20,
        ),
        make_response(
            content="Response two",
            input_tokens=200,
            output_tokens=40,
        ),
    ]

    first = invoke()
    second = invoke()

    assert first["text"] == "Response one"
    assert second["text"] == "Response two"

    assert first["input_tokens"] == 100
    assert second["input_tokens"] == 200

    assert first["output_tokens"] == 20
    assert second["output_tokens"] == 40