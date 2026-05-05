# coding=utf-8
'''
@Author: Peizhen Li
@Desc: None
'''


TERMINATION_STRING = 'done()'

GPT3_CONTEXT = """
objects = [red block, yellow block, blue block, green bowl]
# put the yellow one to the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

# put the yellow block to the green bowl.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [yellow block, blue block, red block, blue bowl]
# move the light colored block to the bowl.
robot.pick_and_place(yellow block, blue bowl)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()
"""

def get_processed_context():
    gpt3_context_lines = GPT3_CONTEXT.split('\n')
    gpt3_context_lines_keep = []
    for line in gpt3_context_lines:
        if 'objects =' in line:
            continue
        gpt3_context_lines_keep.append(line)
    return '\n'.join(gpt3_context_lines_keep)


if __name__ == "__main__":
    # print(get_processed_context())
    RAW_INPUT = 'put the yellow one to the green thing'
    context = get_processed_context() + '\n# ' + RAW_INPUT + '\n'
    print(context)
    

