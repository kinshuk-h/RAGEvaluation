import time
import itertools

import regex
import openai
import tiktoken

from .base import ModelInference

DEFAULT_INSTRUCTIONS = [
    "Faithfully answer the question given as input to the best of your knowledge.",
    "Answer only what is asked, do not add extra information unless requested.",
    "Generate your response after the response indicator."
]

class OpenAIModelInference(ModelInference):
    BATCH_INDICATOR = 'Answer the following together as instructed, with the associated numbering to match responses.\n\n'

    def __init__(self, name, api_key=None, endpoint=None, org_id=None, deployment=None, api_version=None) -> None:
        super().__init__(name)
        self.azure_model = "azure" in name.lower()
        if self.azure_model:
            self.name = self.name[6:]
            self.client = openai.AzureOpenAI(
                azure_endpoint=endpoint, azure_deployment=deployment,
                api_key=api_key, api_version=api_version or "2024-02-01"
            )
        else:
            self.client = openai.OpenAI(
                organization=org_id, api_key=api_key
            )
        self.tokenizer = tiktoken.encoding_for_model(self.name)

    def make_prompt(self, query, instructions=None, examples=None, add_system=True, group_batch=True, *args, **kwargs):
        if instructions is None:
            instructions = DEFAULT_INSTRUCTIONS

        queries = query
        if not isinstance(query, (list, tuple)):
            queries = [query]

        if examples:
            main_eg  = examples[0]
            examples = [
                { "role": "assistant" if i==1 else "user", "content": entry }
                for example in list(examples)[1:]
                for i, entry in enumerate(example)
            ]

        system_message = [ { "role": "system", "content": "You are a helpful assistant." } ]
        if not add_system: system_message = []

        if group_batch and len(queries) > 1:
            grouped_query = self.BATCH_INDICATOR + '\n'.join(f"{i}. {query}" for i, query in enumerate(queries, 1))

            return [
                [
                   *system_message,
                    { "role": "user", "content": '\n'.join(instructions) + '\n\n' + main_eg[0] },
                    { "role": "assistant", "content": main_eg[1] }, *examples,
                    { "role": "user", "content": grouped_query }
                ]
                if examples else
                [
                    *system_message,
                    { "role": "user", "content": ('\n'.join(instructions) + '\n\n' + grouped_query).strip() }
                ]
            ]

        else:
            return [
                [
                    *system_message,
                    { "role": "user", "content": '\n'.join(instructions) + '\n\n' + main_eg[0] },
                    { "role": "assistant", "content": main_eg[1] }, *examples,
                    { "role": "user", "content": query }
                ]
                if examples else
                [
                    *system_message,
                    { "role": "user", "content": ('\n'.join(instructions) + '\n\n' + query).strip() }
                ]
                for query in queries
            ]

    def tokenize(self, prompt):
        if isinstance(prompt, list):
            return self.tokenizer.encode(prompt[-1]['content'])
        else:
            return self.tokenizer.encode(prompt)

    def next_prediction_logits(self, prompt):
        return super().next_prediction_logits(prompt)

    def __select_best_response(self, chat_completion, counterfactuals=None, return_score=False):
        if return_score and counterfactuals:
            answer_token = self.tokenize(counterfactuals[0])[0]
            token_freq = {
                self.tokenize(false_pred)[0]: 0
                for false_pred in counterfactuals[1]
            }
            token_freq[answer_token] = 0
            for choice in chat_completion.choices:
                token = self.tokenize(choice.message.content)[0]
                if token in token_freq: token_freq[token] += 1
            p_true = token_freq[answer_token]
            p_false_avg = sum(freq for tok, freq in token_freq.items() if tok != answer_token) / (len(token_freq) - 1)
            score = float(p_true / (p_true + p_false_avg)) if p_true != 0 and p_false_avg != 0 else 0
            return (chat_completion.choices[0].message.content, score)

        else:
            for choice in chat_completion.choices:
                if choice.finish_reason == "stop":
                    return choice.message.content
            return chat_completion.choices[0].message.content

    def get_response(self, max_calls=10, **gen_kwargs):
        call_count = 0
        while call_count < max_calls:
            try:
                response = self.client.chat.completions.create(model=self.name, **gen_kwargs)
                return response
            except openai.OpenAIError as err:
                # print(gen_kwargs['messages'][0]['content'])
                # print('-' * 100)
                print(err)
                print('=' * 100)
                if not isinstance(err, openai.RateLimitError): raise err
                call_count += 1
                time.sleep(2 ** call_count)
                if call_count == max_calls: raise err

    def __has_grouped_batch(self, messages):
        return any(self.BATCH_INDICATOR in message['content'] for message in messages if message['role']=='user')

    def __unpack_grouped_batch(self, batched_output):
        return [
            regex.sub(r"(?ui)^\d+[.)]?", "", output).lstrip()
            for output in regex.split(r"(?ui)[\p{Zl}\n\r]", batched_output)
            if regex.search(r"(?ui)^\d+[.)]?", output) is not None
        ]

    def generate(self, inputs, return_score=False, counterfactuals=None, transform=None, decoding='simple', *gen_args, **gen_kwargs):
        if isinstance(inputs, str):
            inputs = [ inputs ]
        if transform is not None:
            inputs = transform(inputs)

        if 'chat' in gen_kwargs: del gen_kwargs['chat']
        if 'max_new_tokens' in gen_kwargs:
            gen_kwargs['max_tokens'] = gen_kwargs['max_new_tokens']
            del gen_kwargs['max_new_tokens']
        if 'num_return_sequences' in gen_kwargs:
            gen_kwargs['n'] = gen_kwargs['num_return_sequences']
            del gen_kwargs['num_return_sequences']
        if counterfactuals is not None:
            if gen_kwargs.get('n', 1) == 1:
                gen_kwargs['n'] = 5

        replies = [ self.get_response(messages=messages, **gen_kwargs) for messages in inputs ]
        outputs = [
            self.__select_best_response(reply, counterfactuals=counterfactual, return_score=return_score)
            for reply, counterfactual in zip(replies, counterfactuals or itertools.repeat(None))
        ]
        if len(outputs) == 1 and self.__has_grouped_batch(inputs[0]):
            outputs = self.__unpack_grouped_batch(outputs[0])
        return outputs