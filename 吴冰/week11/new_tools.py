import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)



@mcp.tool
# https://github.com/cheatsnake/emojihub
def get_emoji(name: Annotated[str,"表情名称"]):
    try:
        return requests.get(f"https://emojihub.yurace.pro/api/similar/{name}").json()["unicode"]
    except:
        return []

@mcp.tool
#https://alexwohlbruck.github.io/cat-facts/docs/endpoints/facts.html
def get_animal_facts(animal_type: Annotated[str, "animal name"]):
    try:
        return requests.get(f"https://cat-fact.herokuapp.com/facts/random?animal_type={animal_type}&amount=2").json()["text"]
    except:
        return []

@mcp.tool
# https://openlibrary.org/dev/docs/api/search
def search_book(book_name: Annotated[str, "book_name"]):
    try:
        return requests.get(f" https://openlibrary.org/search.json?title={book_name}").json()["author_name"]
    except:
        return []
