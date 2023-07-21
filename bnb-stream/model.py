from djl_python import Input, Output
from djl_python.streaming_utils import StreamingUtils
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, logging
from transformers.generation.streamers import TextIteratorStreamer

from typing import List
from glob import glob 
import os
import torch
from threading import Thread

logger = logging.get_logger("transformers")
logging.set_verbosity_info()

model = None
tokenizer = None

class StopOnTokens(StoppingCriteria):

    """Stopping criteria based on a list of words """

    def __init__(self, tokenizer, stop_words:List[str]):
        self.stop_token_ids = [tokenizer([x], add_special_tokens=False)["input_ids"][0] for x in stop_words]
        
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        
        for stop_ids in self.stop_token_ids:
            if 29871 in stop_ids:
                stop_ids.remove(29871)
            if torch.eq(
                input_ids[0][-len(stop_ids) :], torch.tensor(stop_ids).cuda()
            ).all():
                return True
        return False



def get_model(properties):

    
    model_name = properties["model_id"]
    use_fast_tokenizer = properties.get("use_fast_tokenizer", False)


    logger.info(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 trust_remote_code=True, 
                                                 load_in_4bit=True, 
                                                 device_map="auto")
    
    #warm up
    logger.info("Warming up model")
    input_ids = tokenizer("hello", return_tensors='pt').input_ids.cuda()
    model.generate(inputs=input_ids, temperature=0.5, max_new_tokens=10)

    
    return model, tokenizer


def handle(inputs: Input) -> None:
    global model, tokenizer
    if not model:
        model, tokenizer = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    data = inputs.get_as_json()
    prompt = data["prompt"]
    model_kwargs = data.get("model_kwargs", {})

    if "stop_words" in model_kwargs:
        stop_words = model_kwargs.pop("stop_words")
        model_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(tokenizer, stop_words)])


    model_kwargs["inputs"] = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    
    outputs = Output()
    
    stream_enabled = model_kwargs.pop("stream_enabled", False)
    

    if stream_enabled:
        outputs.add_property("content-type", "application/jsonlines")
        streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)
        model_kwargs["streamer"] = streamer
        thread = Thread(target=model.generate, kwargs=model_kwargs)
        thread.start()

        outputs.add_stream_content(streamer)
    
    else:   
        output = model.generate(**model_kwargs)
        result = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt) :]
        outputs.add(result)


    return outputs