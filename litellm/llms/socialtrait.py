import httpx
import requests
import json
import uuid
import time
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field

import litellm

from typing import Optional, Callable
from .base import BaseLLM
from litellm.utils import (
    ModelResponse,
    Choices,
    Message,
    Usage,
    EmbeddingResponse,
    ImageResponse,
    convert_to_streaming_response,
)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


# TODO: where to pass the use_bearer token? in optional parameters?
def format_request(
    model: str,
    messages: str,
    api_base: str,
    api_key: Optional[str] = None,
    headers: Optional[dict] = None,
) -> dict:
    """
    Format the request to be sent to the API.
    """
    if not headers:
        headers = {"accept": "application/json", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{api_base}/chat/completions"

    request = {
        "url": url,
        "headers": headers,
        "data": json.dumps(payload),
    }
    return request


def convert_to_model_response_object(
    response_object: Optional[dict] = None,
    model_response_object: Optional[
        Union[ModelResponse, EmbeddingResponse, ImageResponse]
    ] = None,
    response_type: Literal[
        "completion", "embedding", "image_generation"
    ] = "completion",
    stream=False,
    start_time=None,
    end_time=None,
):
    try:
        if response_type == "completion" and (
            model_response_object is None
            or isinstance(model_response_object, ModelResponse)
        ):
            if response_object is None or model_response_object is None:
                raise Exception("Error in response object format")
            if stream == True:
                # for returning cached responses, we need to yield a generator
                return convert_to_streaming_response(response_object=response_object)
            choice_list = []
            for idx, choice in enumerate(response_object["choices"]):
                message = Message(
                    content=choice["message"].get("content", None),
                    role=choice["message"]["role"],
                    function_call=choice["message"].get("function_call", None),
                    tool_calls=choice["message"].get("tool_calls", None),
                )
                finish_reason = choice.get("finish_reason", None)
                if finish_reason == None:
                    # gpt-4 vision can return 'finish_reason' or 'finish_details'
                    finish_reason = choice.get("finish_details")
                logprobs = choice.get("logprobs", None)
                enhancements = choice.get("enhancements", None)
                choice = Choices(
                    finish_reason=finish_reason,
                    index=idx,
                    message=message,
                    logprobs=logprobs,
                    enhancements=enhancements,
                )
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
                model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
                model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

            if "created" in response_object:
                model_response_object.created = response_object["created"]

            if "id" in response_object:
                model_response_object.id = response_object["id"]

            if "system_fingerprint" in response_object:
                model_response_object.system_fingerprint = response_object[
                    "system_fingerprint"
                ]

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            if start_time is not None and end_time is not None:
                model_response_object._response_ms = (  # type: ignore
                    end_time - start_time
                ).total_seconds() * 1000

            return model_response_object
        elif response_type == "embedding" and (
            model_response_object is None
            or isinstance(model_response_object, EmbeddingResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = EmbeddingResponse()

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            if "object" in response_object:
                model_response_object.object = response_object["object"]

            model_response_object.data = response_object["data"]

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
                model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
                model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

            if start_time is not None and end_time is not None:
                model_response_object._response_ms = (  # type: ignore
                    end_time - start_time
                ).total_seconds() * 1000  # return response latency in ms like openai

            return model_response_object
        elif response_type == "image_generation" and (
            model_response_object is None
            or isinstance(model_response_object, ImageResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = ImageResponse()

            if "created" in response_object:
                model_response_object.created = response_object["created"]

            if "data" in response_object:
                model_response_object.data = response_object["data"]

            return model_response_object
    except Exception as e:
        raise Exception(f"Invalid response object {e}")


class SocialtraitError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class SocialtraitChatCompletion(BaseLLM):
    def completion(
        self,
        model_response: ModelResponse,
        timeout: float,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        print_verbose: Optional[Callable] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        acompletion: bool = False,
        logging_obj=None,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[dict] = None,
        custom_prompt_dict: dict = {},
        client=None,
    ):
        # Call the base class's completion method (if needed)
        super().completion()

        request = format_request(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
        )

        if acompletion == True:
            if optional_params.get("stream", False):
                raise NotImplementedError
            else:
                return self.acompletion(
                    api_base=api_base,
                    data=request,
                    headers=headers,
                    #  model_response=model_response,
                    prompt=str(messages),
                    api_key=api_key,
                    logging_obj=logging_obj,
                    model=model,
                    timeout=timeout,
                )

        if client is None:
            client = self.create_client_session()
        # make the HTTP request using the client session
        response = client.post(**request)

        return convert_to_model_response_object(
            response_object=response.json(), model_response_object=model_response
        )

    async def acompletion(
        self,
        logging_obj,
        api_base: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        prompt: str,
        api_key: str,
        model: str,
        timeout: float,
    ):
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    api_base,
                    json=data,
                    headers=headers,
                    timeout=litellm.request_timeout,
                )
                response_json = response.json()
                if response.status_code != 200:
                    raise SocialtraitError(
                        status_code=response.status_code, message=response.text
                    )

                ## LOGGING
                logging_obj.post_call(
                    input=prompt,
                    api_key=api_key,
                    original_response=response,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                    },
                )

                ## RESPONSE OBJECT
                return self.convert_to_model_response_object(
                    response_object=response_json, model_response_object=model_response
                )
            except Exception as e:
                raise e


# class SocialtraitChatCompletion(BaseLLM):


#      def completion(
#         self,
#         model: Optional[str] = None,
#         messages: Optional[list] = None,
#         api_key: Optional[str] = None,
#         api_base: Optional[str] = None,
#         acompletion: bool = False,
#         logging_obj=None,
#         optional_params=None,
#         headers: Optional[dict] = None,
#         client=None,
#     ):
#         super().completion()


#         try:
#             if acompletion is True:
#                 if optional_params.get("stream", False):
#                     return self.async_streaming(
#                         logging_obj=logging_obj,
#                         headers=headers,
#                         data=data,
#                         model=model,
#                         api_base=api_base,
#                         api_key=api_key,
#                         timeout=timeout,
#                         client=client,
#                         max_retries=max_retries,
#                     )
#                 else:
#                     return self.acompletion(
#                         data=data,
#                         headers=headers,
#                         logging_obj=logging_obj,
#                         model_response=model_response,
#                         api_base=api_base,
#                         api_key=api_key,
#                         timeout=timeout,
#                         client=client,
#                         max_retries=max_retries,
#                     )
#             elif optional_params.get("stream", False):
#                 return self.streaming(
#                     logging_obj=logging_obj,
#                     headers=headers,
#                     data=data,
#                     model=model,
#                     api_base=api_base,
#                     api_key=api_key,
#                     timeout=timeout,
#                     client=client,
#                     max_retries=max_retries,
#                 )
#             else:
#                 if not isinstance(max_retries, int):
#                     raise SocialtraitError(
#                         status_code=422, message="max retries must be an int"
#                     )
#                 if client is None:
#                     openai_client = OpenAI(
#                         api_key=api_key,
#                         base_url=api_base,
#                         http_client=litellm.client_session,
#                         timeout=timeout,
#                         max_retries=max_retries,
#                     )
#                 else:
#                     openai_client = client

#                 ## LOGGING
#                 logging_obj.pre_call(
#                     input=messages,
#                     api_key=openai_client.api_key,
#                     additional_args={
#                         "headers": headers,
#                         "api_base": openai_client._base_url._uri_reference,
#                         "acompletion": acompletion,
#                         "complete_input_dict": data,
#                     },
#                 )

#                 response = openai_client.chat.completions.create(**data, timeout=timeout)  # type: ignore
#                 stringified_response = response.model_dump()
#                 logging_obj.post_call(
#                     input=messages,
#                     api_key=api_key,
#                     original_response=stringified_response,
#                     additional_args={"complete_input_dict": data},
#                 )
#                 return convert_to_model_response_object(
#                     response_object=stringified_response,
#                     model_response_object=model_response,
#                 )

#     async def acompletion(
#         self,
#         model: Optional[str] = None,
#         messages: Optional[list] = None,
#         api_key: Optional[str] = None,
#         api_base: Optional[str] = None,
#         acompletion: bool = False,
#         logging_obj=None,
#         optional_params=None,
#         headers: Optional[dict] = None,
#         client=None,
#     ):

#         logging_obj.pre_call(
#             input=data["messages"],
#             api_key=openai_aclient.api_key,
#             additional_args={
#                 "headers": {"Authorization": f"Bearer {openai_aclient.api_key}"},
#                 "api_base": openai_aclient._base_url._uri_reference,
#                 "acompletion": True,
#                 "complete_input_dict": data,
#             },
#         )

#         response = await openai_aclient.chat.completions.create(
#             **data, timeout=timeout
#         )
#         stringified_response = response.model_dump()
#         logging_obj.post_call(
#             input=data["messages"],
#             api_key=api_key,
#             original_response=stringified_response,
#             additional_args={"complete_input_dict": data},
#         )
#         return convert_to_model_response_object(
#             response_object=stringified_response,
#             model_response_object=model_response,
#         )
