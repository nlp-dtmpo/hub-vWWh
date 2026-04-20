response = client.chat.completions.create(
 model='qwenvlmax',
messages=[
 {
 "role": "user",
 "content": [
 {"type": "text", "text": "将图⽚内容提取为 Markdown 格式，输出关键信息。"},
 {
 "type": "image_url",
 "imageurl": {
 "url": f"data:image/jpeg;base64,{base64image}"
 },
 },
 ],
 }
 ],
)