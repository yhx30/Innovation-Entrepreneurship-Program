# coding=utf-8
# Implements API for fine-tuned models in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python api_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint
# Visit http://localhost:8000/docs for document.


import time
import torch
import uvicorn
from threading import Thread
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from transformers import TextIteratorStreamer
from sse_starlette import EventSourceResponse
from typing import Any, Dict, List, Literal, Optional, Union

import transformers
old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
def adaptive_ntk_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    self.dim = dim
    self.base = base
    old_init(self, dim, max_position_embeddings, base, device)

def adaptive_ntk_forward(self, x, seq_len=None):
    if seq_len > self.max_seq_len_cached:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        inv_freq = self.inv_freq
        dim = self.dim
        alpha = seq_len / 1024 - 1
        base = self.base * alpha ** (dim / (dim-2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(x.device) / dim ))

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        return (
            cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )
    return (
        self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
    )
transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = adaptive_ntk_forward
transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = adaptive_ntk_init


from utils import (
    Template,
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    question: str
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatRequest(BaseModel):

    messages: List[ChatMessage]
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))



@app.post("/judgment", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatRequest):
    global model, tokenizer, source_prefix, generating_args
    # print(request)
    query = request.messages[-1].content
    prefix = ""

    history = []
    # print(query)
    inputs = tokenizer([prompt_template.get_prompt(query, history, prefix)], return_tensors="pt")
    inputs = inputs.to(model.device)

    gen_kwargs = generating_args.to_dict()
    
    gen_kwargs.update({
        "input_ids": inputs["input_ids"],
        "temperature": 0.5, ##0.1
        "top_p": 0.7, ##0.4
        "logits_processor": get_logits_processor()
    })
    # if request.max_length:
    gen_kwargs.pop("max_new_tokens", None)
    gen_kwargs["max_length"] = 4096

    gen_kwargs.pop("max_length", None)
    gen_kwargs["max_new_tokens"] = 16000
    # print(gen_kwargs)
    # if request.stream:
    #     generate = predict(gen_kwargs, request.model)
    #     return EventSourceResponse(generate, media_type="text/event-stream")

    generation_output = model.generate(**gen_kwargs)
    outputs = generation_output.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model="llm_for_judge", choices=[choice_data], object="chat.completion")



async def predict(gen_kwargs: Dict[str, Any], model_id: str):
    global model, tokenizer

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield chunk.json(exclude_unset=True, ensure_ascii=False)

    for new_text in streamer:
        if len(new_text) == 0:
            continue

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield chunk.json(exclude_unset=True, ensure_ascii=False)

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield chunk.json(exclude_unset=True, ensure_ascii=False)
    yield "[DONE]"


if __name__ == "__main__":
    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

    uvicorn.run(app, host="0.0.0.0", port=3333, workers=1)
