from time import sleep
from typing import List, Optional
from core import AbstractLLM, AbstractLLMConfig
from mocks.mocks import nmpcMockOptions

import tiktoken
from streamlit import empty, session_state
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from openai import OpenAI


TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4")

class Plan(BaseModel):
  tasks: List[str] = Field(description="list of all tasks to be carried out")
  
  def pretty_print(cls):
    pretty_msg = "Tasks:\n"
    for i, task in enumerate(cls.tasks):
      pretty_msg += f"{i+1}. {task}\n"
    return pretty_msg+'\n'

class Objective(BaseModel):
  objective: str = Field(description="objective function to be applied to MPC")

  def pretty_print():
    pass

class Optimization(BaseModel):
  objective: str = Field(description="objective function to be applied to MPC")
  equality_constraints: List[str] = Field(description="equality constraints to be applied to MPC")
  inequality_constraints: List[str] = Field(description="inequality constraints to be applied to MPC")

  def pretty_print(cls):
    pretty_msg = "Applying the following MPC fomulation:\n```\n"
    pretty_msg += f"min {cls.objective}\n"
    pretty_msg += f"s.t.\n"
    for c in cls.equality_constraints:
      pretty_msg += f"\t {c} = 0\n"
    for c in cls.inequality_constraints:
      pretty_msg += f"\t {c} <= 0\n"
    return pretty_msg+"\n```\n"

class StreamHandler(BaseCallbackHandler):

  def __init__(self, avatar:str, parser: PydanticOutputParser) -> None:
    super().__init__()
    self.avatar = avatar
    self.parser = parser

  def on_llm_start(self, serialized, prompts, **kwargs) -> None:
    """Run when LLM starts running."""
    self.text = ""
    self.container = empty()

  def on_llm_new_token(self, token: str, *, chunk, run_id, parent_run_id=None, **kwargs):
    super().on_llm_new_token(token, chunk=chunk, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    self.text += token
    self.container.write(self.text + "▌")

  def on_llm_end(self, response, **kwargs):
    pretty_text = self.parser.parse(self.text).pretty_print()
    self.container.markdown(pretty_text, unsafe_allow_html=False)
    session_state.messages.append({"type": self.avatar, "content": pretty_text})

import re
def preprocessing(content):
  str_content=content
  task_list=str_content.split('\n')
  for item in task_list:
      if len(item)<=2:
          task_list.remove(item)
  tasks=task_list[1:-1]
  tasks_de_repeat=[]
  for i,task in enumerate(tasks):
    task=task.lower()
    if 'repeat' in task:
        range_pattern = r'\d+-\d+'

        # Search for the pattern in the sentence
        range_match = re.search(range_pattern, task)

        # Extract the range if found
        extracted_range = range_match.group() if range_match else None
        if extracted_range==None:
            start_index=1
            end_index=i
        else:
            start_index=int(extracted_range.split('-')[0])
            end_index=int(extracted_range.split('-')[1])

        # Split the sentence by periods to isolate relevant part
        parts = task.split('.')

        # Find the part of the sentence that contains the word 'Repeat' or 'repeat'
        relevant_part = next((part for part in parts if 'repeat' in part.lower()), None)

        # If a relevant part is found, extract block names that follow 'Repeat'
        if relevant_part:
        # Regular expression to find the block names after 'Repeat'
            block_pattern = r'block_\d+'
            blocks_near_repeat = re.findall(block_pattern, relevant_part)
        else:
            blocks_near_repeat = []

        repeat_tasks=[]
        for task_ in tasks[start_index-1:end_index]:
            block_pattern = r'block_\d+'
            blocks_to_replace = re.findall(block_pattern, task_)
            
            try:
                task_=task_.replace(blocks_to_replace[0],blocks_near_repeat[0])
               
            except:
                pass
            try:
                task_=task_.replace(blocks_to_replace[1],blocks_near_repeat[1])
              
            except:
                pass
            repeat_tasks.append(task_)
        
        tasks_de_repeat=tasks_de_repeat+repeat_tasks
        #repeate_tasks=[]
    else:
        tasks_de_repeat.append(task)
  return tasks_de_repeat

def simulate_stream(avatar:str, text:str, pretty_text:Optional[str]=None):
  """ Function used to simulate stream in case of harcoded GPT responses """
  placeholder = empty()
  # Simulate stream of response with milliseconds delay
  partial_text = ""
  for chunk in TOKEN_ENCODER.decode_batch([[x] for x in TOKEN_ENCODER.encode(text)]):
      partial_text += chunk
      sleep(0.05)
      # Add a blinking cursor to simulate typing
      placeholder.markdown(partial_text + "▌")
  # store message in streamlit
  if pretty_text is None:
    placeholder.markdown(text)
    session_state.messages.append({"type": avatar, "content":text})
  else:
    placeholder.markdown(pretty_text)
    session_state.messages.append({"type": avatar, "content":pretty_text})  

ParsingModel = {
  "plan": Plan,
  "objective": Objective,
  "optimization": Optimization
}


class BaseLLM(AbstractLLM):

  def __init__(self, cfg: AbstractLLMConfig) -> None:
    super().__init__(cfg)
    # init parser
    self.parser = PydanticOutputParser(pydantic_object=ParsingModel[self.cfg.parsing])
    # init model
    self.model = ChatOpenAI(
      model_name=self.cfg.model_name, 
      temperature=self.cfg.temperature,
      streaming=self.cfg.streaming,
      callbacks=None if not self.cfg.streaming else [StreamHandler(self.cfg.avatar, self.parser)]
    )
    # init prompt
    system_prompt = SystemMessagePromptTemplate.from_template(self.cfg.prompt)
    self.messages = [system_prompt.format(format_instructions=self.parser.get_format_instructions())]    
    

  def run(self, user_message:str) -> str:
    self.messages.append(HumanMessage(content=user_message))
    if self.cfg.mock_task is None:
      model_message = self.model(self.messages)
    else:
      model_message = AIMessage(content=nmpcMockOptions[self.cfg.mock_task])
      text = model_message.content
      try:
        pretty_text = self.parser.parse(text).pretty_print()
      except:
        pretty_text = ""
      simulate_stream(self.cfg.avatar, text, pretty_text)
    self.messages.append(model_message)
    return self.parser.parse(model_message.content)
