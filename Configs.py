# coding=utf-8
'''
@Author: Peizhen Li
@Desc: task specific configurations, e.g., environment config, and user input or task (command to the system)
Feel free to add/modify these configs to accommodate more tasks.
'''

MAX_TASKS = 5


RPM = 3   # request per min, you can change according to you own api limit

LLM_ENGINE =  "gpt-3.5-turbo-instruct" # "text-davinci-002"  # "text-ada-001"

# environment configuration

# ENV_CONF = {'pick':  ['yellow block', 'green block', 'blue block'],  # 'red block'
#           'place': ['yellow bowl', 'green bowl', 'blue bowl',]}  # 'red bowl'

ENV_CONF = {'pick':  ['yellow block', 'red block', 'blue block', 'blue block'],  # 'red block'
          'place': ['green bowl', 'red bowl']}  # 

# mimic user input
RAW_INPUT =  'move the block to the bowl' 
# 'stack the blocks'

# 'put the blue one to the red thing'
# 'put the yellow one to the green thing'
# 'move the light colored block to the bowl'
