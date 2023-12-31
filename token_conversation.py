
from transformers import (PreTrainedTokenizer, Conversation)
import torch
import logging
from typing import Tuple


logger = logging.getLogger('llama2_streamlit.token_conversation')

class TokenConversation():
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    [`ConversationalPipeline`]. The conversation contains several utility functions to manage the addition of new user
    inputs and generated model responses. A conversation needs to contain an unprocessed user input before being passed
    to the [`ConversationalPipeline`]. This user input is either created when the class is instantiated, or by calling
    `conversational_pipeline.append_response("input")` after a conversation turn.

    Arguments:
        tokenizer (`PretrainedTokenizer`):
            Tokenizer to use for generating tokens
        system_prompt (`str`):
            System prompt to use when generating new prompt
        max_tokens (`int=1000`, *optional*):
            Maximum number of tokens in a prompt, conversation history is pruned if this number is exceeded
        conversation_id (`uuid.UUID`, *optional*):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.
        past_user_inputs (`List[str]`, *optional*):
            Eventual past history of the conversation of the user. You don't need to pass it manually if you use the
            pipeline interactively but if you want to recreate history you need to set both `past_user_inputs` and
            `generated_responses` with equal length lists of strings
        generated_responses (`List[str]`, *optional*):
            Eventual past history of the conversation of the model. You don't need to pass it manually if you use the
            pipeline interactively but if you want to recreate history you need to set both `past_user_inputs` and
            `generated_responses` with equal length lists of strings

    Usage:

    ```python
    conversation = Conversation("Going to the movies tonight - any suggestions?")

    # Steps usually performed by the model when generating a response:
    # 1. Mark the user input as processed (moved to the history)
    conversation.mark_processed()
    # 2. Append a mode response
    conversation.append_response("The Big lebowski.")

    conversation.add_user_input("Is it good?")
    ```"""


    #Update the number of system prompt tokens when the system prompt is
    #set or updated
    def _update_system_prompt(self, prompt_str: str) -> None:
        system_prompt_format = "<<SYS>>\n {prompt_str}\n<</SYS>>\n\n"
        self.full_system_prompt = system_prompt_format.format(prompt_str=prompt_str)
        system_prompt_tokens = self.tokenizer.encode(self.full_system_prompt, return_tensors='pt')
        #Determine the length of the system prompt
        self.system_prompt_tokens = len(system_prompt_tokens[0])


    def __init__(self, tokenizer:  PreTrainedTokenizer,
                 system_prompt: str,
                 max_tokens: int=1000):
        if tokenizer is None:
            logger.error("TokenConversation created with None as tokenizer")
            raise ValueError()
        if system_prompt is None:
            logger.error("TokenConversation created with None as system prompt")
            raise ValueError()
        self.tokenizer = tokenizer
        self._update_system_prompt(system_prompt)
        self.max_tokens = max_tokens
        self.total_tokens = 0
        self.response_sizes = []
        self.user_sizes = []
        self.base_conversation = Conversation()
        

    def reset_config(self, tokenizer:  PreTrainedTokenizer, system_prompt: str,
                     max_tokens: int=1000) -> None:
        if tokenizer is None:
            logger.error("reset_config called with None as tokenizer")
            raise ValueError()
        if system_prompt is None:
            logger.error("reset_config called with None as system prompt")
            raise ValueError()
        self.tokenizer = tokenizer
        self._update_system_prompt(system_prompt)
        self.max_tokens = max_tokens

   
    #Given the full set of tokens given after LLM generate,
    #update the response and token counts
    def append_response_from_tokens(self, tokens: torch.LongTensor) -> str:
        #This class requires one response to be paired with one request
        #while the partner conversation class allows multiple answers to follow
        #a single request
        if(len(self.base_conversation.generated_responses) > len(self.base_conversation.past_user_inputs)):
            logger.warn("Attempted to add second answer to conversation without new user input")
            raise ValueError()
        
        if len(tokens.size()) != 2:
            logger.error("Expected 2D tensor with 1 array of tensors as returned by transformer.generate")
            raise ValueError()
        
        #Presume a 2D tensor as returned from generate
        new_total_tokens = len(tokens[0])
        answer_len = new_total_tokens - self.total_tokens

        #Get the list of tokens starting after our last request
        chatbot_answer_tensor = tokens[:,self.total_tokens:]
        chatbot_answer_list = self.tokenizer.batch_decode(chatbot_answer_tensor, skip_special_tokens=True)
        #Full string is in the first element
        chatbot_answer_str = chatbot_answer_list[0]

        self.base_conversation.append_response(chatbot_answer_str)
        self.response_sizes.append(answer_len)
        self.total_tokens = new_total_tokens
        self.base_conversation.mark_processed()

        return chatbot_answer_str


    #Append a partial string response if the response was stopped by the user
    def append_partial_response(self, response_str: str) -> None:
         #This class requires one response to be paired with one request
        #while the partner conversation class allows multiple answers to follow
        #a single request
        if(len(self.base_conversation.generated_responses) > len(self.base_conversation.past_user_inputs)):
            logger.warn("Attempted to add second answer to conversation without new user input")
            raise ValueError()
        
        #Determine the number of tokens in the partial response
        response_tokens = self.tokenizer.encode(response_str, return_tensors='pt')
        #Determine the length of the system prompt
        response_len = len(response_tokens[0])

        self.base_conversation.append_response(response_str)
        self.response_sizes.append(response_len)
        self.total_tokens += response_len
        self.base_conversation.mark_processed()
  



    #Update queues tracking the tokens used for each input and prune off early conversation elements if
    #conversation has become too long
    def rightsize_conversation(self, input_tensor)  -> Tuple[torch.Tensor, int]:
        have_previous_data = (not self.base_conversation.past_user_inputs is None) and (
                                len(self.base_conversation.past_user_inputs) > 0)

     
        new_total_tokens = len(input_tensor[0])
        new_input_tokens = new_total_tokens - self.total_tokens

        #Don't count the system tokens against the first user prompt
        if not have_previous_data:
            new_input_tokens = new_input_tokens - self.system_prompt_tokens

        #Append the number of tokens for this query
        self.user_sizes.append(new_input_tokens)

        #If we are under the allowed tokens or if we only have this one request just return the input tensor
        if (new_total_tokens <= self.max_tokens) or not have_previous_data:
            logger.debug(f"Sending {new_total_tokens} tokens")
            #Total will be updated again after answer is returned
            self.total_tokens = new_total_tokens
            return input_tensor, 0
        
        logger.info(f"&&&& Request to send {new_total_tokens} tokens, must be pruned &&&&")
        #otherwise, we need to start pruning old elements from the conversation
        #until we get under the limit
        #count the number of request/response pairs that we need to prune
        rounds_pruned=0
        while (new_total_tokens > self.max_tokens) and (
            rounds_pruned < len(self.response_sizes)):
            round_tokens = (self.user_sizes[rounds_pruned] + self.response_sizes[rounds_pruned])
            new_total_tokens = new_total_tokens - round_tokens
            rounds_pruned = rounds_pruned+1
        logger.debug(f"pruned {rounds_pruned} conversation rounds")

        #Adjust our stored length vectors
        self.user_sizes = self.user_sizes[rounds_pruned:]
        self.response_sizes = self.response_sizes[rounds_pruned:]

        new_conversation = Conversation()
        trimmed_responses = self.base_conversation.generated_responses[rounds_pruned:]
        trimmed_inputs = self.base_conversation.past_user_inputs[rounds_pruned:]
        
        #If we have any previous conversation rounds left, put those in
        #the new conversation
        if len(trimmed_inputs) > 0:
            #Add the system prompt to the first new input
            trimmed_inputs[0] = self.full_system_prompt + trimmed_inputs[0]
            logger.debug(f"First request now: \n{trimmed_inputs[0]}\n")
            #Add the old trimmed inputs and responses to the new conversation
            for input, response in zip(trimmed_inputs, trimmed_responses):
                new_conversation.add_user_input(input)
                new_conversation.append_response(response)
                new_conversation.mark_processed()
            #Add the current request to the conversation
            new_conversation.add_user_input(self.base_conversation.new_user_input)
        #Otherwise if we deleted all of our previous rounds, add the system prompt to the current
        #input
        else:
            new_conversation.add_user_input(self.full_system_prompt + self.base_conversation.new_user_input)

        #update the conversation
        self.base_conversation = new_conversation

        #finally build new input ids and return a new tensor
        input_ids = self.tokenizer._build_conversation_input_ids(new_conversation)
        input_tensor = self.tokenizer.encode(input_ids, return_tensors='pt')
        new_total_tokens = len(input_tensor[0])
        #Will be updated again once response is received
        self.total_tokens = new_total_tokens
        logger.debug("reduced to {} tokens".format(new_total_tokens))
        return input_tensor, rounds_pruned



    #Given the proposed input, generate the next set of prompt tokens
    #pruning the conversation if necessary
    def create_next_prompt_tokens(self, input: str) -> Tuple[torch.Tensor, int]:
        #If there are no prevous user requests in the conversation
        #prepend the system prompt
        if (self.base_conversation.past_user_inputs is None) or (
            len(self.base_conversation.past_user_inputs) == 0):
            self.base_conversation.add_user_input(self.full_system_prompt + input)
            logger.debug("First request now: \n{}\n".format(self.full_system_prompt + input))
        else:
            self.base_conversation.add_user_input(input)

        #allow the tokenizer class to actually format the prompt
        input_ids = self.tokenizer._build_conversation_input_ids(self.base_conversation)
        input_tensor = self.tokenizer.encode(input_ids, return_tensors='pt')

        #Update the number of tokens with this current addition and
        #prune the conversation if necessary
        input_tensor, rounds_pruned = self.rightsize_conversation(input_tensor)
        return input_tensor, rounds_pruned
    

    #Iterate over all of the messages received so far for printing
    def iter_texts(self):
        return self.base_conversation.iter_texts()
    
    #return a string representation of the conversation so far
    #adding the total tokens
    def __repr__(self):
        conversation = self.base_conversation.__repr__()
        return conversation + "\n{} total tokens in conversation\n".format(self.total_tokens)