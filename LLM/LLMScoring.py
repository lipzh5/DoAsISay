# coding=utf-8
'''
@Author: Peizhen Li
@Desc: None
'''

import openai
import math
import numpy as np
# from Affordance.AffordanceScoring import affordance_scoring


LLM_CACHE = {}

MAX_PROMPT = 20

# assume the argument 'prompt' is with legal length, so we should preprocess it before calling this function 
def gpt3_call(engine='text-ada-001', prompt='', max_tokens=128, temperature=0, logprobs=1, echo=False):
    full_query = ''
    for p in prompt: # prompt should be a list of str
        full_query += p
    _id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
    response = LLM_CACHE.get(_id)
    
    if response is None:
        response = openai.Completion.create(engine=engine, 
                                                prompt=prompt, 
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                logprobs=logprobs,
                                                ) # echo=echo
        LLM_CACHE[_id] = response
    return response
        

def scoring_for_options(options, response_choice, option_start='\n'):
    scores = {} 
    tokens = response_choice['logprobs']['tokens']
    token_logprobs = response_choice['logprobs']['token_logprobs']
    print('tokens: \n', tokens)
    for option in options:
        total_prob = 0
        for token, token_logprob in zip(tokens, token_logprobs):
            if option_start in token:
                # print('token == option start', token)
                break
            if token in option:
                if option == 'robot.pick_and_place(yellow block, blue bowl)':
                    print('token in option: ', token, token_logprob)
                total_prob += np.exp(token_logprob)
        scores[option] = total_prob
    return scores


def batched_gpt_call(query, options, engine):
    """Call gpt several times, because we are limited by the #options"""
    # 1. batch options regarding MAX_POMPT==20, assume batches<=20, i.e., total options <= 400 for the moment
    batches = math.ceil(len(options) / MAX_PROMPT)
    assert batches <= 20, "Too many options! check make_options function please."
    select_n_from_each = MAX_PROMPT // batches
    candidate_options = []
    # 2. scoring per batch and select candidates from each batch
    for i in range(batches):
        sub_options = options[i*MAX_PROMPT: (i+1)*MAX_PROMPT]
        # gpt3_prompt_options = [query + option for option in sub_options]
        response = gpt3_call(engine=engine,
                             prompt=query,
                            # prompt=gpt3_prompt_options,
                            logprobs=1,
                            temperature=0)
        scores = scoring_for_options(sub_options, response['choices'][0])
        # you may want to add affordance score
        # afford_scores = affordance_scoring(sub_options)
        # scores = {option: score * afford_scores[option] for option, score in scores.items()}

        sorted_scores = sorted(scores.items(), key=lambda x:-x[1])[:select_n_from_each]
        candidate_options.extend([item[0] for item in sorted_scores])
    # 3. scoring over candidates
    # gpt3_prompt_options = [query + option for option in candidate_options]
    response = gpt3_call(engine=engine,
                            prompt=query,
                            # prompt=gpt3_prompt_options,
                            logprobs=1,
                            temperature=0)
    scores = scoring_for_options(candidate_options, response['choices'][0])
    # afford_scores = affordance_scoring(candidate_options)
    # scores = {option: score * afford_scores[option] for option, score in scores.items()}
    
    return scores
    


def gpt3_scoring(query, options, engine='text-ada-001', limit_num_options: int=None, option_start='\n', verbose=False, print_tokens=False):
    if limit_num_options:
        options = options[:limit_num_options]
    verbose and print(f'Scoring {len(options)} options')
    scores = batched_gpt_call(query, options, engine)

    for i, option, in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)):
        verbose and print(f'{option[1]}\t{option[0]}')
        if i > 10:
            break
    return scores


if __name__ == "__main__":
    d = {'a': -1, 'b': -2, 'c': 4}
    sd = sorted(d.items(), key=lambda x: -x[1]) # list, ascending order
    print(type(sd), sd)



    