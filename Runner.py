# coding=utf-8
'''
@Author: Peizhen Li
@Desc: None
'''

import torch
import numpy as np
from PickPlaceEnv import env
import clip
import jax.numpy as jnp
from Cliport.Cliport import eval_step, get_pretrained_optim
import matplotlib.pyplot as plt
import utils
from utils import make_options, output_cached_video, get_pretrained_clip, get_coords, step_to_nlp
from const import PICK_TARGETS, PLACE_TARGETS
from LLM.PromptEngineering import TERMINATION_STRING, get_processed_context #GPT3_CONTEXT
from LLM.LLMScoring import gpt3_scoring
from Configs import LLM_ENGINE, ENV_CONF, RAW_INPUT, MAX_TASKS, RPM
from PickPlaceEnv import env
import time

import openai
openai.api_key = "PUT YOUR API KEY HERE"

# make sure the resource is ready
utils.try_load_all_assets()



def run():
    obs = env.reset(ENV_CONF)  # reset environment 
    _pick_targets = {k: None for k in ENV_CONF['pick'] if k not in PICK_TARGETS}
    _pick_targets.update(PICK_TARGETS)
    _place_targets = {k: None for k in ENV_CONF['place'] if k not in PLACE_TARGETS}
    _place_targets.update(PLACE_TARGETS)
    options = make_options(_pick_targets, _place_targets, termitation_string=TERMINATION_STRING)
    gpt3_prompt = get_processed_context() + '\n# ' + RAW_INPUT + '\n'
    selected_task = ''
    num_tasks = 0
    steps_text = []
    all_llm_scores = []
    while selected_task != TERMINATION_STRING:
        num_tasks += 1
        if num_tasks > MAX_TASKS:
            break
        time.sleep(60//RPM)  # because of the RPM
        scores = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=LLM_ENGINE)
        selected_task = max(scores, key=scores.get)
        steps_text.append(selected_task)
        print(f'{num_tasks} .Selecting: {selected_task}')
        gpt3_prompt += selected_task + '\n'

        all_llm_scores.append(scores)
    # # execute 
    for i, step in enumerate(steps_text):
        if step in {'', TERMINATION_STRING}:
            break
        print(f'Step {i}: {step}')
        nlp_step = step_to_nlp(step)
        obs = run_cliport(obs, nlp_step)


def run_cliport(obs, text):
    before = env.get_camera_image() 
    pre_obs = obs['image'].copy()

    # tokenize text and get CLIP features
    text_tokens = clip.tokenize(text) # .cuda()
    clip_model = get_pretrained_clip()
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens).float()
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    text_feats = np.float32(text_feats.cpu())

    # normalize image and add batch dimension
    coords = get_coords()
    img = obs['image'][None, ...] / 255
    img = np.concatenate((img, coords[None, ...]), axis=3)

    # run TransporterNets to get pick and place heatmaps
    batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
    optim = get_pretrained_optim()
    pick_map, place_map = eval_step(optim.target, batch)
    pick_map, place_map = np.float32(pick_map), np.float32(place_map)

    # get pick position
    pick_max = np.argmax(np.float32(pick_map)).squeeze()
    pick_yx = (pick_max // 224, pick_max % 224) # (row, col)
    pick_yx = np.clip(pick_yx, 20, 204)
    pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

    # get place position
    place_max = np.argmax(np.float32(place_map)).squeeze()
    place_yx = (place_max // 224, place_max % 224)
    place_yx = np.clip(place_yx, 20, 204)
    place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

    # step environment
    act = {'pick': pick_xyz, 'place': place_xyz}
    obs, _, _, _ = env.step(act)

    # show pick and place action
    plt.title(text)
    plt.imshow(pre_obs)
    plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], 
              color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.title('Pick Heatmap')
    plt.imshow(pick_map.reshape(224, 224))
    plt.subplot(1, 2, 2)
    plt.title('Place Heatmap')
    plt.imshow(place_map.reshape(224, 224))
    plt.show()

    # save environment rollout as video
    output_cached_video(env)

    # show camera image before and after
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    plt.title('After')
    plt.imshow(env.get_camera_image())
    plt.show()

    return obs


if __name__ == "__main__":
    run()
    # np.random.seed(42)
    # obs = env.reset(ENV_CONF)
    # steps_text = ['robot.pick_and_place(blue block,blue bowl)']
    # nlp_step = step_to_nlp(steps_text[0])
    # print('nlp step: ', nlp_step)
    # for step in steps_text:
    #     nlp_step = step_to_nlp(step)
    #     obs = run_cliport(obs, nlp_step)


    




