import re

import torch
import transformers

from .base import ModelInference

def contains(text, words):
    return any(word in text for word in words)

def get_pred_score(scores, tokens):
    probabilities = torch.nn.functional.softmax(scores, dim=-1).detach().cpu()
    probability   = probabilities.take(torch.tensor([tokens])).squeeze()
    p_true, p_false_avg = probability[0].item(), probability[1:].mean().item()
    return float(p_true / (p_true + p_false_avg))

DEFAULT_INSTRUCTIONS = [
    "Faithfully answer the question given as input to the best of your knowledge.",
    "Answer only what is asked, do not add extra information unless requested.",
    "Generate your response after the response indicator."
]

DEFAULT_PROMPT_TEMPLATES = {
    'instruct-ft': (
        "{instructions}\n\n{examples}\ninput: {input}\nresponse: "
    ),
    'instruct-ft+chat': (
        "You will be provided with a set of instructions, followed by an input which provides further context. "
        "Generate a response that appropriately completes the input as per the instructions.\n\n"
        "### Instructions\n{instructions}\n\n"
        "{examples}\n\n### Input\n{input}\n\n### Response\n"
    ),
    'custom': {
        'nlpcloud/instruct-gpt-j-fp16': (
            "{instructions}\n\n{examples}\n\ninput:\n{input}\nresponse:\n"
        )
    }
}

DEFAULT_EXAMPLE_TEMPLATES = {
    'instruct-ft': "input: {0}\nresponse: {1}",
    'instruct-ft+chat': "### Input\n{0}\n\n### Response\n{1}",
    'custom': {
        'nlpcloud/instruct-gpt-j-fp16': "input:\n{0}\nresponse:\n{1}",
    }
}

class HuggingFaceModelInference(ModelInference):
    def __init__(self, name, lazy_load=True, **model_kwargs) -> None:
        super().__init__(name)

        tokenizer_args = {}
        if "bloom" in name.lower() or "gpt" in name.lower() or "llama-3" in name.lower():
            tokenizer_args['padding_side'] = "left"
        if "llama" in name.lower() or "mistral" in name.lower():
            tokenizer_args['add_prefix_space'] = True
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.name, **tokenizer_args)

        if getattr(self.tokenizer, 'pad_token', None) is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
            # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = None
        self.decoding_mode = "simple"
        if not model_kwargs:
            model_kwargs = dict(device_map="auto")
        self.model_kwargs = model_kwargs

        self.config = transformers.AutoConfig.from_pretrained(name)
        if any("causal" in arch for arch in getattr(self.config, 'architectures', [])):
            self.config.is_decoder = True

        if not lazy_load: self.load()

    def load(self):
        if "t5" in self.name.lower():
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.name, **self.model_kwargs)
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.name, **self.model_kwargs)
        self.model.eval()

        if getattr(self.model.generation_config, 'pad_token_id') is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            if isinstance(self.model.generation_config.eos_token_id, list):
                self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id[0]

    def __generate(self, inputs, is_chat=False, strip_output=True, **gen_kwargs):
        if is_chat:
            inputs = self.tokenizer.apply_chat_template(inputs, tokenize=False)
            inputs = self.tokenizer(inputs, return_tensors="pt", padding="longest")
        else:
            inputs = self.tokenizer(inputs, return_tensors="pt", padding="longest")
        input_len = inputs['input_ids'].shape[-1]
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return [
            output.strip() for output in \
                self.tokenizer.batch_decode([
                    output[input_len:] if strip_output and "t5" not in self.name else output for output in outputs
                ], skip_special_tokens=True)
        ]

    def make_prompt(self, query, instructions=None, examples=None, prompt_template=None,
                    example_template=None, join_token=None, chat=False, *args, **kwargs):
        if instructions is None:
            instructions = DEFAULT_INSTRUCTIONS

        if join_token is None:
            join_token = '\n'
            if contains(self.name.lower(), ('instruct', 'llama')):
                join_token = '\n\n'

        if prompt_template is None:
            prompt_template = DEFAULT_PROMPT_TEMPLATES[
                "instruct-ft+chat" if contains(self.name.lower(), ('llama', 'mistral')) else "instruct-ft"
            ]
            if self.name in DEFAULT_PROMPT_TEMPLATES["custom"]:
                prompt_template = DEFAULT_PROMPT_TEMPLATES["custom"][self.name]

        if examples is not None and example_template is None:
            example_template = DEFAULT_EXAMPLE_TEMPLATES[
                "instruct-ft+chat" if contains(self.name.lower(), ('llama', 'mistral')) else "instruct-ft"
            ]
            if self.name in DEFAULT_EXAMPLE_TEMPLATES["custom"]:
                example_template = DEFAULT_EXAMPLE_TEMPLATES["custom"][self.name]

        queries = query
        if not isinstance(query, (list, tuple)):
            queries = [query]

        if chat and ('chat' in self.name.lower() or self.tokenizer.chat_template is not None):
            system_instruction = [
                { "role": "system", "content": "You are a helpful assistant." }
            ] if 'system' in (self.tokenizer.chat_template or self.tokenizer.default_chat_template) else []

            if examples:
                main_eg  = examples[0]
                examples = [
                    { "role": "assistant" if i==1 else "user", "content": entry }
                    for example in list(examples)[1:]
                    for i, entry in enumerate(example)
                ]

            return [
                [
                    *system_instruction,
                    { "role": "user", "content": '\n'.join(instructions) + '\n\n' + main_eg[0] },
                    { "role": "assistant", "content": main_eg[1] }, *examples,
                    { "role": "user", "content": query }
                ]
                if examples is not None else
                [
                    *system_instruction,
                    { "role": "user", "content": '\n'.join(instructions) + '\n\n' + query }
                ]
                for query in queries
            ]

        else:
            if examples:
                return [
                    prompt_template.format(
                        instructions='\n'.join(instructions), input=query,
                        examples=(join_token.join(example_template.format(*ex) for ex in examples))
                    )
                    for query in queries
                ]
            else:
                prompt_template = re.sub(r"(?ui)\{examples\}\s*", "", prompt_template)
                return [
                    prompt_template.format(
                        instructions='\n'.join(instructions), input=query
                    )
                    for query in queries
                ]

    def tokenize(self, prompt, **tokenizer_kwargs):
        return self.tokenizer(prompt, **tokenizer_kwargs)['input_ids']

    def next_prediction_logits(self, prompt, chat=False, all_logits=False):
        if chat and ('chat' in self.name.lower() or self.tokenizer.chat_template is not None):
            inputs = self.tokenizer.apply_chat_template(prompt, tokenize=False)
            inputs = self.tokenizer(inputs, return_tensors="pt", padding="longest")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding="longest")

        if self.model is None: self.load()
        inputs = inputs.to(self.model.device)

        if self.config.is_encoder_decoder:
            outputs = self.model(**inputs, decoder_input_ids=self.tokenizer("", return_tensors="pt")['input_ids'])
        else:
            outputs = self.model(**inputs)

        logits = outputs.logits.detach().cpu()
        if all_logits:
            return logits if isinstance(prompt, list) and len(prompt) > 1 else logits[0]
        else:
            return logits [:, -1] if isinstance(prompt, list) and len(prompt) > 1 else logits[0, -1]

    def postprocess_output(self, output):
        if self.decoding_mode == "aggressive":
            if match := re.search(r"(?ui)[\r\n#]+(.|\n)*$", output):
                output = output[:match.span()[0]]
        return output.strip()

    def generate(self, inputs, counterfactuals=None, return_score=False, chat=False, strip_output=True, decoding='simple', **gen_kwargs):
        if self.model is None: self.load()
        self.decoding_mode = decoding or 'simple'

        if chat and ('chat' in self.name.lower() or self.tokenizer.chat_template is not None):
            inputs = self.tokenizer.apply_chat_template(inputs, tokenize=False)
            inputs = self.tokenizer(inputs, return_tensors="pt", padding="longest")
        else:
            inputs = self.tokenizer(inputs, return_tensors="pt", padding="longest")
        inputs = inputs.to(self.model.device)

        input_len = inputs['input_ids'].shape[-1]
        if gen_kwargs.get('num_return_sequences', 1) > 1 or (return_score and counterfactuals is not None):
            gen_kwargs['return_dict_in_generate'] = True
            gen_kwargs['output_scores'] = True

        outputs = self.model.generate(**inputs, **gen_kwargs)
        if gen_kwargs.get('return_dict_in_generate'):
            if return_score and counterfactuals is not None:
                pred_tokens = [
                    self.tokenizer(
                        [ corr, *incorr ], padding='longest',
                        return_tensors='np', add_special_tokens=False
                    )['input_ids'][:,0]
                    for corr, incorr in counterfactuals
                ]
                sequences = [
                    (self.postprocess_output(output), get_pred_score(scores.detach().cpu(), tokens)) for output, tokens, scores in \
                        zip(
                            self.tokenizer.batch_decode([
                                output[input_len:] if strip_output and "t5" not in self.name else output
                                for output in outputs.sequences
                            ], skip_special_tokens=True),
                            pred_tokens,
                            outputs.scores[0]
                        )
                ]
            else:
                sequences = [
                    self.postprocess_output(output) for output in \
                        self.tokenizer.batch_decode([
                            output[input_len:] if strip_output and "t5" not in self.name else output
                            for output in outputs.sequences
                        ], skip_special_tokens=True)
                ]
            seq_per_batch = gen_kwargs.get('num_return_sequences', 1)
            if seq_per_batch > 1:
                return [
                    sequences[i:i+seq_per_batch] for i in range(0, len(sequences), seq_per_batch)
                ]
            else:
                return sequences
        else:
            return [
                self.postprocess_output(output) for output in \
                    self.tokenizer.batch_decode([
                        output[input_len:] if strip_output and "t5" not in self.name else output for output in outputs
                    ], skip_special_tokens=True)
            ]
