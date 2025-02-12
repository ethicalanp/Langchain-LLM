from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

#Functions
def generate_celebrity_info(name: str) -> str:
    return f"Details about {name}"

def generate_birth_date(person_info: str) -> str:
    return f"{person_info}\nBirth date:..."

#LLM
llm = OpenAI(temperature=0.8)

#prompt templates
first_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity {name}"
)

second_prompt = PromptTemplate(
    input_variables=["person_info"],
    template="When was {person_info} born? Provide exact birth date."
)


chain1 = (
    RunnableLambda(generate_celebrity_info)
    | first_prompt
    | llm
)

chain2 = (
    RunnableLambda(generate_birth_date)
    | second_prompt
    | llm
)


parent_chain = chain1 | chain2


input_text = "Tom Cruise"
output = parent_chain.invoke(input_text)
print(output)