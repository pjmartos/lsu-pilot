from .functions import functions, run_function
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
from ollama import AsyncClient
import pandas as pd
import numpy as np
from .questions import answer_question

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
Again, make sure All HTML is generated with the JSX flavoring.
If you instead need to call a tool, please make sure you're providing all required arguments with their proper names.
Any misbehaviour will be severely punished.
"""

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the CSV file
csv_path = os.path.join(current_dir, "processed", "embeddings.csv")
df = pd.read_csv(csv_path, index_col=0)
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

load_dotenv()  # take environment variables from .env.

ollama = AsyncClient()
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
    initial_response = await ollama.chat(
        model="llama3.1:70b", messages=messages, tools=functions)
    initial_response_message = initial_response["message"]
    messages.append(initial_response_message)
    tool_calls = initial_response_message.get('tool_calls')
    if tool_calls:
        print(tool_calls)
        for tool_call in tool_calls:
            name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            response = run_function(name, args)

            if name == 'svg_to_png_bytes':
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id, photo=response
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": str(response) + "Image was sent to the user, do not send the base64 string to them. ONLY send back 'here is the svg rendered as requested'"
                    }
                )
            else:
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": str(response)
                    }
                )

        final_response = await ollama.chat(
            model='llama3.1:70b', messages=messages
        )
        final_answer = final_response["message"]
        if final_answer:
            messages.append(final_answer)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=final_answer["content"]
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="something wrong happened, please try again"
            )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=initial_response_message["content"]
        )

async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
      answer = answer_question(df, question=update.message.text, debug=True)
      await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()

  start_handler = CommandHandler('start', start)
  chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)
  mozilla_handler = CommandHandler('mozilla', mozilla)

  application.add_handler(start_handler)
  application.add_handler(chat_handler)
  application.add_handler(mozilla_handler)

  application.run_polling()
