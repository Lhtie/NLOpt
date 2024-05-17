import openai
import google.generativeai as genai
import time

from typing import Dict, Optional, Union, List

def __openai_call(
    client, 
    prompt: Union[str, List], 
    model: Optional[str] = "gpt-3.5-turbo"
) -> str:
    if isinstance(prompt, str):
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    else:
        for p in prompt:
            assert isinstance(p, dict)
            assert "role" in p and "content" in p
    
    cnt = 3
    while cnt > 0:
        try:
            time.sleep(5)
            completion = client.chat.completions.create(
                model=model,
                messages=prompt,
                seed=cnt,
            )
            content = completion.choices[0].message.content
            return content

        except Exception as e:
            print(e)
            cnt -= 1
            if cnt == 0:
                raise RuntimeError("Model call failure.")

            print(f"Error, Try another {cnt} times.")
            
def __google_call(
    client,
    prompt: str
) -> str:
    if isinstance(prompt, list):
        content = {}
        for p in prompt:
            content[p["role"]] = p["content"]
        assert "system" in content and "user" in content
        prompt = content["system"] + content["user"]
        
    cnt = 5
    while cnt > 0:
        try:
            time.sleep(0.6 * (2 ** (3 - cnt)))
            completion = client.generate_content(prompt)
            content = completion.text
            print("*" * 20)
            print(prompt)
            print(content)
            print("*" * 20)
            return content

        except Exception as e:
            print(e)
            cnt -= 1
            if cnt == 0:
                raise RuntimeError("Model call failure.")
            
            print(f"Error, Try another {cnt} times.")

def model_call(client, prompt: Union[str, List], model: Optional[str] = None):
    if isinstance(client, openai.Client):
        if model:
            return __openai_call(client, prompt, model)
        else:
            return __openai_call(client, prompt)
    elif isinstance(client, genai.GenerativeModel):
        return __google_call(client, prompt)
    else:
        raise NotImplementedError("Unknown client")