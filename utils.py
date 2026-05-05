# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import zipfile

import gdown
import const
from moviepy.editor import ImageSequenceClip
import clip
import numpy as np

def get_pick_place_from_step(step):
	step = step.replace('robot.pick_and_place(', '')
	step = step.replace(')', '')
	pick, place = step.split(',')
	return pick, place

def step_to_nlp(step):
	# step = step.replace('robot.pick_and_place(', '')
	# step = step.replace(')', '')
	# pick, place = step.split(',')
	pick, place = get_pick_place_from_step(step)
	return f'Pick the {pick} and place it on the {place}.'


def make_options(pick_targets, place_targets, options_in_api_form=True, termitation_string='done()'):
	"""generate all possible options given pick and place targets in the environments"""
	options = []
	form_str = 'robot.pick_and_place({}, {})' if options_in_api_form else 'Pick the {} and place it on the {}'
	for pick in pick_targets:
		for place in place_targets:
			options.append(form_str.format(pick, place))
	options.append(termitation_string)

	print(f'Considering {len(options)} options')
	return options


def get_pretrained_clip():
    clip_model, _ = clip.load('ViT-B/32')
    clip_model.eval() # or clip_model.cuda().eval() if cuda is available
    return clip_model


def get_coords():
    # Coordinate map (i.e. position encoding).
    coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
    coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)
    return coords


def output_cached_video(env, out_name='my_video'):
	not os.path.exists('outputs') and os.makedirs('outputs')
	if not env.cache_video:
		print('No cached video, abort...')
	debug_clip = ImageSequenceClip(env.cache_video, fps=25)
	debug_clip.write_videofile(f'outputs/{out_name}.mp4')
	# clear cache
	env.cache_video = []


def try_load_all_assets():
	not os.path.exists('assets') and os.makedirs('assets')
	for ast in const.ASSET_NAMES:
		if os.path.exists(const.ASSETS_PATH_DICT[ast]):
			continue
		zip_file_name = 'assets/{}.zip'.format(ast)
		gdown.download(const.GDOWN_URL_DICT[ast], output=zip_file_name)
		with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
			zip_ref.extractall('assets')


if __name__ == "__main__":
	step = 'robot.pick_and_place(blue block,middle)'
	nlp_step = step_to_nlp(step)
	
	# try_load_all_assets()
	
