from time import sleep
from typing import List
from core import AbstractLLM, AbstractLLMConfig
from mocks.mocks import nmpcMockOptions

import langchain
langchain.verbose = False

import tiktoken
from streamlit import empty, session_state
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.base import BaseCallbackHandler

TOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4")

class Plan(BaseModel):
  tasks: List[str] = Field(description="list of all tasks that the robot has to carry out")
  
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
  constraints: List[str] = Field(description="constraints to be applied to MPC")

  def pretty_print(cls):
    pretty_msg = "Applying the following MPC fomulation:\n```\n"
    pretty_msg += f"min {cls.objective}\n"
    pretty_msg += f"s.t.\n"
    for c in cls.constraints:
      pretty_msg += f"\t {c}\n"
    return pretty_msg+"\n```\n"

class StreamHandler(BaseCallbackHandler):

  def __init__(self, parser: PydanticOutputParser) -> None:
    super().__init__()
    self.parser = parser

  def on_llm_start(self, serialized, prompts, **kwargs) -> None:
    """Run when LLM starts running."""
    self.text = ""
    self.container = empty()

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text + "▌")

  def on_llm_end(self, response, **kwargs):
    pretty_text = self.parser.parse(self.text).pretty_print()
    self.container.markdown(pretty_text)
    session_state.messages.append(AIMessage(content=pretty_text))


def simulate_stream(text:str):
  """ Function used to simulate stream in case of harcoded GPT responses """
  placeholder = empty()
  # Simulate stream of response with milliseconds delay
  partial_text = ""
  for chunk in TOKEN_ENCODER.decode_batch([[x] for x in TOKEN_ENCODER.encode(text)]):
      partial_text += chunk
      sleep(0.05)
      # Add a blinking cursor to simulate typing
      placeholder.markdown(partial_text + "▌")
  placeholder.markdown(text)
  # return AI message
  session_state.messages.append(AIMessage(content=text))


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
      callbacks=None if not self.cfg.streaming else [StreamHandler(self.parser)]
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
      simulate_stream(self.parser.parse(model_message.content).pretty_print())
    self.messages.append(model_message)
    #print(f"\33[92m {model_message.content} \033[0m \n")
    return self.parser.parse(model_message.content)
