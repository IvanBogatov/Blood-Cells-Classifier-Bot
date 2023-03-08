import os
import logging

from aiogram import Bot, Dispatcher, executor, types

from config import LOGFILE_PATH
from back import img_handler


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=LOGFILE_PATH,
                    filemode='a')

TOKEN = os.getenv('TOKEN')

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    await bot.send_message(chat_id=user_id, text='Please, send a picture.')

@dp.message_handler(content_types=['photo'])
async def photo_handler(callback: types.CallbackQuery):
    user_name = callback.from_user.username
    user_id = callback.from_user.id
    file_id = callback.photo[0]['file_id']
    file = await bot.get_file(file_id)
    file_path = file.file_path
    logging.info(f"{user_name=} {user_id=} have sent photo: {file_id=}.")
    img_io = await bot.download_file(file_path)
    # Check if the 2nd highest probability is big enough to be shown.
    label_1, prob_1, label_2, prob_2 = img_handler(img_io)
    if prob_2 < 0.1:
        await bot.send_message(chat_id=user_id, text=f'It\'s {label_1} with prob. = {prob_1:.0%}')
    else:
        await bot.send_message(chat_id=user_id, text=f'It\'s {label_1} with prob. = {prob_1:.0%} \n({label_2} with prob. = {prob_2:.0%})')


if __name__=='__main__':
    executor.start_polling(dp, skip_updates=True)