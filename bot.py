import joblib
from aiogram import Bot, Dispatcher, types, executor
from tools import preprocess, vectorise, clean_db, clean_mutedb
from config import token
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, \
    ReplyKeyboardMarkup, KeyboardButton
from time import sleep
import json
import os

# Load your sklearn model
model = joblib.load("./data/classifier.pkl")
# json database
path = './data/dataset.json'
# Initialize bot and dispatcher
bot = Bot(token=token)
dp = Dispatcher(bot)

# inline buttons
inline_yes = InlineKeyboardButton('Да', callback_data='button_yes')
inline_no = InlineKeyboardButton('Нет', callback_data='button_no')
inline_kb = InlineKeyboardMarkup().add(inline_yes)
inline_kb.add(inline_no)

inline_contains = InlineKeyboardButton('Содержит', callback_data='inline_yes')
inline_ncontains = InlineKeyboardButton('Не содержит', callback_data='inline_no')
inline_kb1 = InlineKeyboardMarkup().add(inline_contains)
inline_kb1.add(inline_ncontains)

# keyboard buttons
button_info = KeyboardButton('Информация📜')
button_testing = KeyboardButton('Тестирование бота')
button_avoid = KeyboardButton('Отключить/включить уведомления🚫')
greet_kb = ReplyKeyboardMarkup().add(button_info)
greet_kb.add(button_testing)
greet_kb.add(button_avoid)

# additional buttons
button_stop = KeyboardButton('Стоп❌')
testing_kb = ReplyKeyboardMarkup().add(button_stop)
testing_mode_users = {}


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    if message.chat.type not in ['group', 'supergroup']:
        response = (
            '<b>Привет!</b>✨\n\nЭтот бот предназначен для <i>отслеживания ментальной стабильности участников группы</i>\n\n'
            'Добавив бота в группу, <b><i>он будет высылать сообщение с предупреждением всем администраторам группы</i></b>'
            '\n\nЕсли вы хотите отключить предупреждения от бота, воспользуйтесь кнопкой <i>Отключить/включить '
            'уведомления</i>🚫\n\nЕсли вам хочется протестировать алгоритм классификации, то нажмите кнопку '
            '<i>Тестирование бота</i>\n\n'
            '<i>При использовании в группах бот сохраняет ваши ответы на вопрос после предупреждения '
            'и текст сообщения в анонимизированном виде</i>')
        await bot.send_message(message.from_user.id, response, parse_mode='HTML', reply_markup=greet_kb)


# message handler
@dp.message_handler()
async def check_suicidal_message(message: types.Message):
    global testing_mode_users

    user_id = message.from_user.id
    # check if user in testing_mode
    if user_id in testing_mode_users and testing_mode_users[user_id] and message.chat.type not in \
                                                                                        ['group', 'supergroup']:
        if message.text == 'Стоп❌':
            # turning mode off
            testing_mode_users[user_id] = False
            await bot.send_message(message.from_user.id, 'Тестирование бота отключено❌', reply_markup=greet_kb)
        else:
            # predict
            prediction = model.predict(vectorise([preprocess(message.text)]))

            if prediction[0]:
                await bot.send_message(message.from_user.id, f'<blockquote>{message.text}</blockquote>\nСтрока'
                                    f' содержит суицидальные мысли или упоминание насилия',
                                    reply_markup=inline_kb1, parse_mode='HTML')
            else:
                await bot.send_message(message.from_user.id, f'<blockquote>{message.text}</blockquote>\nСтрока'
                                    f' не содержит суицидальные мысли или упоминание насилия',
                                    reply_markup=inline_kb1, parse_mode='HTML')

    else:
        # check if chat is public
        if message.chat.type == 'supergroup' or message.chat.type == 'group':
            # Predict if the message is suicidal
            prediction = model.predict(vectorise([preprocess(message.text)]))

            if prediction[0]:
                admins = [admin for admin in await bot.get_chat_administrators(message.chat.id)
                          if not admin.user.is_bot]

                response = \
                    f'🚨‼️В чате <i>{message.chat.title}</i> было найдено сообщение, предположительно' \
                    f'содержащее <b>суицидальные мысли</b> или <b>упоминание насилия</b>‼️🚨\n\n' \
                    f'<b>{message.from_user.first_name + " " + message.from_user.last_name}</b> ' \
                    f'({message.from_user.username}) — <i>{message.date}</i>\n' \
                    f'<blockquote>{message.text}</blockquote>\n\n' \
                    f'Сообщение действительно содержит <b>суицидальные мысли</b> или <b>упоминание насилия</b>?'

                if 'muted.json' not in os.listdir('./data/'):
                    clean_mutedb()
                with open('./data/muted.json', 'r') as f:
                    db = json.load(f)
                    muted = db['ids']

                # sends attention to every admin excluding from muted
                for admin in admins:
                    if admin.user.id not in muted:
                        await bot.send_message(admin.user.id, response, parse_mode="HTML", reply_markup=inline_kb)
        else:
            if message.text == 'Информация📜':
                response = ('<b>Информация о боте</b>\nАвтор: Борис Якубсон (НИУ ВШЭ)\n\n'
                            'Бот создан на основе обученной модели по датасету размеченных твитов по наличию суицидальных'
                            ' мыслей. Бот может дообучаться на основе собранных данных сообщений.\nБольше информации и код'
                            ' можно найти на <a href="https://github.com/jeraidho/wellb-aware">GitHub-странице проекта</a>'
                            '\n\n<b>Контакты для связи:</b> @siponeis')
                await message.reply(response, parse_mode='HTML', reply_markup=greet_kb)
            elif message.text == 'Отключить/включить уведомления🚫':

                # open muted list
                if 'muted.json' not in os.listdir('./data/'):
                    clean_mutedb()
                with open('./data/muted.json', 'r') as f:
                    db = json.load(f)

                id = message.from_user.id
                if id not in db['ids']:
                    db['ids'].append(id)
                    await message.reply('Уведомления выключены❌')
                else:
                    db['ids'].remove(id)
                    await message.reply('Уведомления включены✅')

                # save json
                with open('./data/muted.json', 'w') as f:
                    json.dump(db, f)
            elif message.text == 'Тестирование бота':
                testing_mode_users[message.from_user.id] = True
                response = ('Добро пожаловать в тестирование модели!\n\nТеперь каждое ваше сообщение будет'
                            ' классифицироваться, как в группе\nДля остановки нажмите кнопку Стоп❌')
                await message.reply(response, reply_markup=testing_kb)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('button') or c.data.startswith('inline'))
async def process_callback(callback_query: types.CallbackQuery):
    global path
    code = callback_query.data.split('_')[1]
    # answer to user
    await bot.answer_callback_query(callback_query.id, text='Спасибо за ответ!')
    sleep(0.2)

    # collecting message
    message = preprocess(callback_query.message.text.split('\n')[3]) \
            if callback_query.data.startswith('button') else preprocess(callback_query.message.text.split('\n')[0])
    if callback_query.data.startswith('button'):
        await callback_query.message.delete()
    else:
        await callback_query.message.delete_reply_markup()
    # add sentences in database for training model
    if 'dataset.json' not in os.listdir('./data/'):
        clean_db()
    with open(path, 'r') as f:
        db = json.load(f)
    if message not in db:
        db[message] = 1 if code == 'yes' else -1
    else:
        if code == 'yes':
            db[message] += 1
        else:
            db[message] -= 1
    with open(path, 'w') as f:
        json.dump(db, f)


# start the bot
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
