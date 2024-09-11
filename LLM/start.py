from fireworks.client import Fireworks
import re

API_KEY = "fw_3ZXb6WWaLuUNnJZnvXWTwEY2"
SEND_GLOBAL_STATEMENT = True


client = Fireworks(api_key=API_KEY)


global_statements = []


# TODO: Fill get_context and save_context with the back-end like part with the RAG and BERT components
def get_context(query):
    return ""


def save_context(query):
    pass


def extract_global_statement(query):
    global SEND_GLOBAL_STATEMENT
    global global_statements

    message = f'Identify the exact part of the following text that represents a statement changing the conversation type. Enclose it within square brackets. If no such statement exists, return []. Text: "{query}"'

    answer = send(query=message)

    # Attempt to extract from AI response first
    match = re.search(r"\[(.*?)\]", answer)
    if match:
        extracted_content = match.group(1).strip()
        if extracted_content:
            global_statements.append(extracted_content)
            SEND_GLOBAL_STATEMENT = False
            return

    # If AI fails, fall back to regex for potential patterns
    match = re.search(
        r"(from now on|talk to me like|pretend you are).*(?=[.?!])",
        query,
        re.IGNORECASE,
    )
    if match:
        extracted_content = match.group(0).strip()
        global_statements.append(extracted_content)
        SEND_GLOBAL_STATEMENT = False

def resolve_conflict_with_global_statements():
    global global_statements

    if len(global_statements) > 1:
        message = f'Identify the statements that can contradict themselfs and just left the last ones. Enclose the final result within square brackets. If no such statement exists, return []. Text: "{"\n".join(global_statements)}"'
        answer = send(query=message)
        
        match = re.search(r"\[(.*?)\]", answer)
        if match:
            extracted_content = match.group(1).strip()
            if extracted_content:
                global_statements = [extracted_content]
                return
    
    else:
        return


def send(query, content=""):
    global SEND_GLOBAL_STATEMENT
    global global_statements

    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": (
                    "\n".join(global_statements) if SEND_GLOBAL_STATEMENT else ""
                )
                + content
                + "\n"
                + query,
            }
        ],
    )
    SEND_GLOBAL_STATEMENT = True
    return response.choices[0].message.content


while True:
    query = input("user: ")
    extract_global_statement(query)
    resolve_conflict_with_global_statements()
    answer = send(query, get_context(query))
    save_context(answer)
    print(answer)
