import joblib
from aiogram import Bot, Dispatcher, types, executor
from tools import preprocess, vectorise
from config import token

# Load your sklearn model
model = joblib.load("./data/classifier.pkl")
# Initialize bot and dispatcher
bot = Bot(token=token)
dp = Dispatcher(bot)


# message handler
@dp.message_handler()
async def check_suicidal_message(message: types.Message):
    # Predict if the message is suicidal
    prediction = model.predict(vectorise([preprocess(message.text)]))

    if prediction[0]:
        response = "This message seems to be suicidal. Please seek help."
        await message.reply(response)



# start the bot
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
