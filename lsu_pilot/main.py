import requests
from .functions import functions, run_function
import json
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from .questions import answer_question
from whisper_cpp_python import Whisper

whisper = Whisper(model_path="./models/ggml-tiny.bin")

CODE_PROMPT = """
Here are two input:output examples for code generation. Please use these and follow the styling for future requests that you think are pertinent to the request.
Make sure All HTML is generated with the JSX flavoring.
// SAMPLE 1
// A Blue Box with 3 yellow cirles inside of it that have a red outline
<div style={{   backgroundColor: 'blue',
  padding: '20px',
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'center',
  width: '300px',
  height: '100px', }}>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
</div>
"""

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the CSV file
csv_path = os.path.join(current_dir, "processed", "embeddings.csv")
df = pd.read_csv(csv_path, index_col=0)
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

load_dotenv()  # take environment variables from .env.

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tg_bot_token = os.getenv("TG_BOT_TOKEN")


messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}, {
   "role": "system",
   "content": CODE_PROMPT
}]

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    messages.append({"role": "user", "content": update.message.text})
    initial_response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=functions)
    initial_response_message = initial_response.choices[0].message
    messages.append(initial_response_message)
    tool_calls = initial_response_message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            response = run_function(name, args)
            print(tool_calls)
            if name == 'svg_to_png_bytes':
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id, photo=response
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": name,
                        "content": str(response) + "Image was sent to the user, do not send the base64 string to them. Only send back 'here is the svg rendered as requested'"
                    }
                )
            else:
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": name,
                        "content": str(response)
                    }
                )

            final_response = openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages
            )
            final_answer = final_response.choices[0].message
            if final_answer:
                messages.append(final_answer)
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=final_answer.content
                )
            else:
                # Send an error message if something went wrong
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="something wrong happened, please try again"
                )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=initial_response_message.content
        )

async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
      answer = answer_question(df, question=update.message.text, debug=True)
      await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
      response = openai.images.generate(
          prompt=update.message.text,
          model="dall-e-3",
          n=1,
          size="1024x1024"
      )
      image_url = response.data[0].url
      image_response = requests.get(image_url)
      await context.bot.send_photo(
          chat_id=update.effective_chat.id,
          photo=image_response.content
      )

async def transcription(update: Update, context: ContextTypes.DEFAULT_TYPE):
      await update.message.reply_text("We're preparing your transcription, stay tuned...")
      voice_id = update.message.voice.file_id
      file = await context.bot.get_file(voice_id)
      await file.download_to_drive(f"voice_note_{voice_id}.ogg")
      file_path = open(f"voice_note_{voice_id}.ogg", "rb")
      transcript = whisper.transcribe(file_path)
      await update.message.reply_text(f"Transcription finished: {transcript['text']}")

if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()

  start_handler = CommandHandler('start', start)
  chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)
  transcription_handler = MessageHandler(filters.VOICE, transcription)
  mozilla_handler = CommandHandler('mozilla', mozilla)
  image_handler = CommandHandler('image', image)

  application.add_handler(start_handler)
  application.add_handler(chat_handler)
  application.add_handler(mozilla_handler)
  application.add_handler(image_handler)
  application.add_handler(transcription_handler)

  application.run_polling()
