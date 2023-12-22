from parse import *
#Test file to check intersting Python behavior
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

dict_array = ({"role" : "user", "content" : "q1"}, {"role" : "assistant", "content" : "a1"},
                {"role" : "user", "content" : "q2"}, {"role" : "assistant", "content" : "a2"},
                {"role" : "user", "content" : "q3"}, {"role" : "assistant", "content" : "a3"},
                {"role" : "user", "content" : "q4"}, {"role" : "assistant", "content" : "a4"},
                {"role" : "user", "content" : "q5"}, {"role" : "assistant", "content" : "a5"},
                {"role" : "user", "content" : "q6"}, {"role" : "assistant", "content" : "a6"},
                {"role" : "user", "content" : "q7"})

#Some very pythonic code to look at each pair of two contiguous list entries, and create a tuple
#with the 2nd dictionary entry fom each in each of the two items
# (a bit dangerous, assumes 'content' is the 2nd dictionary entry)
#Then formats a string to put the first piece of content, the user question inside of the [INST]
#block and puts the second piece of content, presumably the answer from the assistant, outside
#of the block and wrap the whole message in <s> blocks.
#Took me much longer to write this comment than it would have to write a more readable for loop
#to do the same thing, but this is reused code and maybe it is common in Python.
messages_list = [
    f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
    for prompt, answer in zip(dict_array[::2], dict_array[1::2])
]

#This pythonic code also assumes an odd number of list elements with the last being the most 
#recent question to the language model.
messages_list.append(
    f"{BOS}{B_INST} {(dict_array[-1]['content']).strip()} {E_INST}")

for message in messages_list:
    print(message)


#Loop through the entire response until we get the last section after the last instruction
#Keep going until we hit the final close inst tag.
def retrive_last_llama2_response(response_str: str):
    remaining_string = response_str
    return_val = None
    while return_val is None:
        items = parse("{instructions}[/INST]{remainder}", remaining_string)
        if items is not None:
            remaining_string = items['remainder']
        else:
            return_val = remaining_string

    #Double check to see if we ended with an instruction and no response
    final_parse = parse("{instructions}[/INST]", return_val)
    if not final_parse is None:
        return ""
    return return_val

response_str = "[INST] <<SYS>>\nYou are a helpful AI assistant. Reply in mardkown format.\n<</SYS>>\n\ntell me a joke [/INST] Unterscheidung! *wink* Here's a joke for you:\n\nWhy couldn't the bicycle stand up by itself?\n\nBecause it was two-tired! *groan* I hope that brought a smile to your face! Do you want to hear another one?"
response_str2 = """[INST] <<SYS>>\nYou are a helpful AI assistant. Reply in mardkown format.\n<</SYS>>\n\ntell me a joke [/INST]
Unterscheidung! *wink* Here's a joke for you:\n\nWhy couldn't the bicycle stand up by itself?\n\n
Because it was two-tired! *groan* I hope that brought a smile to your face! Do you want to hear another one?"
[INST]yes[/INST] joke2
[INST]tell another[/INST] joke3"""

items = parse("[INST]{instruction}[/INST]{response}", response_str)

last_response = retrive_last_llama2_response(response_str2)
print(last_response)
